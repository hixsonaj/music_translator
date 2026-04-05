"""
Spanish → English translation test.

For each line:
  1. Sends to the translation pipeline via a local test agent
  2. Counts syllables on the result and checks it matches the target exactly
  3. Asks Claude to score semantic similarity 0–10 and give a brief note

Run with:
    python test_spanish.py

The orchestrator + all agents must already be running locally.
"""

import os
import re
import asyncio
from collections import OrderedDict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / "keys.env")

import anthropic
from uagents import Agent, Context
from models import LyricLine, AcceptedLine

try:
    import pyphen
    PYPHEN_AVAILABLE = True
except ImportError:
    PYPHEN_AVAILABLE = False

ORCHESTRATOR_ADDRESS = "agent1qga4t2sj75u7kxrw8gpj3jpx0wg5dch0jc06elsfmc54mrjsdntyukzvkgf"

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# (spanish lyric, syllable count, english gloss for reference)
SPANISH_LINES = [
    # Short / simple
    ("Te quiero tanto",                     5,  "I love you so much"),
    ("Eres mi vida",                        5,  "You are my life"),
    ("No me dejes solo",                    6,  "Don't leave me alone"),
    ("Bailamos juntos esta noche",          9,  "We dance together tonight"),
    # Medium
    ("Siempre estaré a tu lado",            8,  "I will always be by your side"),
    ("Eres la luz de mi vida",              7,  "You are the light of my life"),
    ("Te quiero con todo mi corazón",      10,  "I love you with all my heart"),
    ("Nunca voy a olvidarte",               8,  "I'm never going to forget you"),
    ("El cielo es testigo de mi amor",     10,  "The sky witnesses my love"),
    # Longer / more complex
    ("Cuando la noche cae y estoy sin ti", 11,  "When night falls and I'm without you"),
    ("Volar contigo hasta el fin del mundo",12, "Fly with you to the end of the world"),
    ("Mi corazón late solo por tu amor",   11,  "My heart beats only for your love"),
    # Idiomatic / hard to translate literally
    ("Échame la culpa",                     6,  "Blame it on me"),
    ("Lo que pasó, pasó",                   6,  "What happened, happened"),
    ("Duele el corazón",                    6,  "The heart aches"),
    ("Dame más de tu querer",               7,  "Give me more of your love"),
    ("Sin ti no soy nada",                  6,  "Without you I am nothing"),
    ("Cada vez que te veo sonrío",          9,  "Every time I see you I smile"),
]

# Map line_id → (spanish, target_syllables, gloss) for lookup on receipt
PENDING: dict[int, tuple[str, int, str]] = OrderedDict()
RESULTS: dict[int, AcceptedLine] = {}


# ── Syllable counter (mirrors syllable_critic.py) ────────────────────────────

def count_syllables(text: str, lang: str = "en") -> int:
    if PYPHEN_AVAILABLE:
        try:
            dic = pyphen.Pyphen(lang=lang)
            total = 0
            for word in text.split():
                hyphenated = dic.inserted(word)
                total += hyphenated.count("-") + 1
            return total
        except Exception:
            pass
    return sum(max(1, len(re.findall(r'[aeiouAEIOU]+', word))) for word in text.split())


# ── Semantic similarity judge ─────────────────────────────────────────────────

def score_semantics(original_es: str, gloss: str, translation_en: str) -> tuple[int, str]:
    prompt = (
        f"You are evaluating a song lyric translation from Spanish to English.\n\n"
        f"Spanish original: \"{original_es}\"\n"
        f"Literal meaning:  \"{gloss}\"\n"
        f"Pipeline output:  \"{translation_en}\"\n\n"
        f"Score how well the pipeline output preserves the meaning and feeling of the original "
        f"on a scale of 0 to 10, where 10 is identical meaning and 0 is completely unrelated.\n"
        f"Reply with exactly two lines:\n"
        f"Score: <number>\n"
        f"Note: <one sentence>\n"
        f"No other text."
    )
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=80,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    score = 0
    note = "N/A"
    for line in text.splitlines():
        if line.lower().startswith("score:"):
            try:
                score = int(re.search(r"\d+", line).group())
            except Exception:
                pass
        elif line.lower().startswith("note:"):
            note = line.split(":", 1)[1].strip()
    return score, note


# ── Test agent ────────────────────────────────────────────────────────────────

test_agent = Agent(
    name="test_runner",
    seed="test_runner_seed_phrase",
    port=8010,
    endpoint=["http://localhost:8010/submit"],
    mailbox=False,
)


@test_agent.on_event("startup")
async def send_all(ctx: Context):
    for i, (spanish, target_syl, gloss) in enumerate(SPANISH_LINES):
        line_id = 200 + i
        PENDING[line_id] = (spanish, target_syl, gloss)
        await ctx.send(ORCHESTRATOR_ADDRESS, LyricLine(
            line_id=line_id,
            original=spanish,
            translated="",
            target_lang="en",
            target_syllables=target_syl,
        ))
    ctx.logger.info(f"Sent {len(SPANISH_LINES)} lines to orchestrator.")


@test_agent.on_message(model=AcceptedLine)
async def receive_result(ctx: Context, sender: str, msg: AcceptedLine):
    RESULTS[msg.line_id] = msg

    if len(RESULTS) == len(SPANISH_LINES):
        print_results()
        # All done — stop the agent
        raise SystemExit(0)


def print_results():
    print("\n" + "=" * 70)
    print("Spanish → English  |  syllable check + semantic score")
    print("=" * 70)

    passed_syllables = 0
    semantic_scores = []
    total = len(SPANISH_LINES)

    for i, (spanish, target_syl, gloss) in enumerate(SPANISH_LINES):
        line_id = 200 + i
        print(f"\n[{i+1}/{total}]")
        print(f"  Spanish:     {spanish}")
        print(f"  Gloss:       {gloss}")
        print(f"  Target syl:  {target_syl}")

        result = RESULTS.get(line_id)
        if result is None:
            print("  RESULT:      (not received)")
            continue

        translation = result.final_translation
        attempts = result.attempts
        actual_syl = count_syllables(translation, "en")
        syl_ok = actual_syl == target_syl
        if syl_ok:
            passed_syllables += 1

        sem_score, sem_note = score_semantics(spanish, gloss, translation)
        semantic_scores.append(sem_score)

        syl_mark = "PASS" if syl_ok else f"FAIL (got {actual_syl})"
        print(f"  Translation: {translation}")
        print(f"  Attempts:    {attempts}")
        print(f"  Syllables:   {syl_mark}")
        print(f"  Semantics:   {sem_score}/10 — {sem_note}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Syllable exact match: {passed_syllables}/{total} ({passed_syllables/total:.0%})")
    if semantic_scores:
        avg = sum(semantic_scores) / len(semantic_scores)
        print(f"  Avg semantic score:   {avg:.1f}/10")
        buckets = {"high (8-10)": 0, "mid (5-7)": 0, "low (0-4)": 0}
        for s in semantic_scores:
            if s >= 8:
                buckets["high (8-10)"] += 1
            elif s >= 5:
                buckets["mid (5-7)"] += 1
            else:
                buckets["low (0-4)"] += 1
        for label, count in buckets.items():
            print(f"    {label}: {count}")
    print()


if __name__ == "__main__":
    test_agent.run()
