"""
ASI:One compatible orchestrator.

This wraps your existing lyric translation pipeline so it can be
discovered and used directly from the ASI:One chat UI at asi1.ai.

How it works:
  1. User types in ASI:One: "Translate 'Never gonna give you up' to Spanish"
  2. ASI:One finds this agent via Agentverse and sends a ChatMessage
  3. This agent parses the request, runs it through the translation pipeline
  4. Returns the result as a ChatMessage back to ASI:One

Setup:
  1. Get an Agentverse API key at https://agentverse.ai
  2. Run with: AGENTVERSE_API_KEY=... ANTHROPIC_API_KEY=... python orchestrator_asi.py
"""

import os
import json
import re
from datetime import datetime
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv(".env/keys.env")

from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    TextContent,
    chat_protocol_spec,
)
from models import LyricLine, CritiqueResult, RevisionRequest, AcceptedLine

MAX_ATTEMPTS = 4
ACCEPT_THRESHOLD = 0.70

WEIGHTS = {
    "syllable": 0.80,
    "stress":   0.20,
}

TRANSLATION_ADDR = os.getenv("TRANSLATION_ADDR", "agent1qff9cuylddkkza5f2ajw052rlesg7sjj74y3hdy50s6pahsazw84zveq48r")
SYLLABLE_ADDR    = os.getenv("SYLLABLE_ADDR", "agent1qv47xtctvwv72evsldm0ht7rc7sn9kclntkuwvr85tyd6n0g40p8jl4tzku")
STRESS_ADDR      = os.getenv("STRESS_ADDR", "agent1qdeu8rgrvwfxtmf6clsjcmmpvmjr3ach8ljdn7yfazcq3l5wzv3777vayfq")

LANG_CODES = {
    "spanish": "es", "french": "fr", "german": "de",
    "italian": "it", "portuguese": "pt", "japanese": "ja",
    "korean": "ko", "chinese": "zh",
}

# ── Agent setup with mailbox=True for ASI:One discoverability ─────────────────
orchestrator = Agent(
    name="lyric-translator",
    seed=os.getenv("ORCHESTRATOR_SEED", "orchestrator_seed_phrase"),
    port=8000,
    mailbox=True,               # Required for ASI:One to reach you
    agentverse={"api_key": os.getenv("AGENTVERSE_API_KEY")},    
    publish_agent_details=True, # Makes it discoverable on Agentverse
)

protocol = Protocol(spec=chat_protocol_spec)


# ── Storage helpers ───────────────────────────────────────────────────────────

def get_state(ctx, line_id):
    raw = ctx.storage.get(f"line_{line_id}")
    return json.loads(raw) if raw else {}

def save_state(ctx, line_id, state):
    ctx.storage.set(f"line_{line_id}", json.dumps(state))


# ── Parse natural language request from ASI:One ───────────────────────────────

def parse_request(text: str) -> dict | None:
    # Strip the @agent mention
    text = re.sub(r'@agent1q\w+', '', text).strip()
    text_lower = text.lower()

    # Detect direction — default to Spanish if no English words found
    if any(word in text_lower for word in ["english", "inglés", "ingles", "to en", "al inglés"]):
        target_lang = "en"
    else:
        target_lang = "es"

    # Remove instruction words to isolate the lyric
    lyric = re.sub(
        r'(translate|to english|to spanish|al español|al ingles|al inglés|'
        r'please|can you|into english|into spanish)',
        '', text, flags=re.IGNORECASE
    ).strip(' .,?!\n')

    if not lyric or len(lyric) < 3:
        return None

    syllables = max(1, len(re.findall(r'[aeiouAEIOU]+', lyric)))

    return {
        "lyric": lyric.strip(),
        "target_lang": target_lang,
        "syllables": syllables,
    }


# ── ASI:One chat protocol handler ─────────────────────────────────────────────

@protocol.on_message(ChatMessage)
async def handle_chat(ctx: Context, sender: str, msg: ChatMessage):
    # Acknowledge receipt immediately
    await ctx.send(sender, ChatAcknowledgement(
        timestamp=datetime.now(),
        acknowledged_msg_id=msg.msg_id,
    ))

    # Extract text from message
    text = ""
    for item in msg.content:
        if isinstance(item, TextContent):
            text += item.text

    ctx.logger.info(f"ASI:One request: '{text}'")

    # Parse the request
    parsed = parse_request(text)
    if not parsed:
        await ctx.send(sender, ChatMessage(
            timestamp=datetime.utcnow(),
            msg_id=uuid4(),
            content=[
                TextContent(type="text", text=(
                    "I can translate song lyrics for you! Try:\n"
                    "\"Translate 'Never gonna give you up' to Spanish\"\n"
                    "Supported languages: Spanish, French, German, Italian, "
                    "Portuguese, Japanese, Korean, Chinese"
                )),
                EndSessionContent(type="end-session"),
            ]
        ))
        return

    # Store the ASI:One sender so we can reply when translation completes
    line_id = abs(hash(text)) % 100000
    state = {
        "line_id": line_id,
        "original": parsed["lyric"],
        "target_lang": parsed["target_lang"],
        "target_syllables": parsed["syllables"],
        "attempt": 1,
        "current_translation": "",
        "pending_critiques": [],
        "backend_addr": sender,
        "is_chat": True,  # Flag so we know to reply via ChatMessage
    }
    save_state(ctx, line_id, state)

    # Send to translation agent
    await ctx.send(TRANSLATION_ADDR, LyricLine(
        line_id=line_id,
        original=parsed["lyric"],
        translated="",
        target_lang=parsed["target_lang"],
        target_syllables=parsed["syllables"],
    ))

    ctx.logger.info(
        f"Started translation: '{parsed['lyric']}' → {parsed['target_lang']}"
    )


@protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass  # Not needed for this use case


# ── Internal agent message handlers (same as before) ─────────────────────────

@orchestrator.on_message(model=LyricLine)
async def handle_incoming(ctx: Context, sender: str, msg: LyricLine):
    if sender == TRANSLATION_ADDR:
        state = get_state(ctx, msg.line_id)
        if not state:
            return
        state["current_translation"] = msg.translated
        state["pending_critiques"] = []
        save_state(ctx, msg.line_id, state)
        ctx.logger.info(f"Line {msg.line_id}: translated='{msg.translated}'")
        await ctx.send(SYLLABLE_ADDR, msg)
        if STRESS_ADDR:
            await ctx.send(STRESS_ADDR, msg)
        return

    # Fresh line from non-chat source
    state = {
        "line_id": msg.line_id,
        "original": msg.original,
        "target_lang": msg.target_lang,
        "target_syllables": msg.target_syllables,
        "attempt": 1,
        "current_translation": "",
        "pending_critiques": [],
        "backend_addr": sender,
        "is_chat": False,
    }
    save_state(ctx, msg.line_id, state)
    await ctx.send(TRANSLATION_ADDR, msg)


@orchestrator.on_message(model=CritiqueResult)
async def handle_critique(ctx: Context, sender: str, msg: CritiqueResult):
    state = get_state(ctx, msg.line_id)
    if not state:
        return

    state["pending_critiques"].append(msg.dict())
    save_state(ctx, msg.line_id, state)

    num_critics = 2 if STRESS_ADDR else 1
    if len(state["pending_critiques"]) < num_critics:
        return

    critiques = state["pending_critiques"]
    scores = {c["critic"]: 1.0 if c["passed"] else 0.0 for c in critiques}
    score = sum(WEIGHTS.get(c, 0.5) * scores.get(c, 0.0) for c in scores)
    attempt = state["attempt"]

    ctx.logger.info(f"Line {msg.line_id} attempt {attempt}: score={score:.0%}")

    if score >= ACCEPT_THRESHOLD or attempt >= MAX_ATTEMPTS:
        # Build reply based on whether this came from ASI:One or direct
        if state.get("is_chat"):
            lang_name = next(
                (k.capitalize() for k, v in LANG_CODES.items()
                 if v == state["target_lang"]), state["target_lang"]
            )
            reply_text = (
                f"Here's your translation to {lang_name}:\n\n"
                f"Original: {state['original']}\n"
                f"Translation: {state['current_translation']}\n\n"
                f"Quality score: {score:.0%} "
                f"(completed in {attempt} attempt{'s' if attempt > 1 else ''})"
            )
            await ctx.send(state["backend_addr"], ChatMessage(
                timestamp=datetime.utcnow(),
                msg_id=uuid4(),
                content=[
                    TextContent(type="text", text=reply_text),
                    EndSessionContent(type="end-session"),
                ]
            ))
        else:
            await ctx.send(state["backend_addr"], AcceptedLine(
                line_id=msg.line_id,
                final_translation=state["current_translation"],
                attempts=attempt,
            ))

        ctx.storage.remove(f"line_{msg.line_id}")
        ctx.logger.info(f"Line {msg.line_id}: ACCEPTED score={score:.0%}")
    else:
        issues = [c["feedback"] for c in critiques if not c["passed"] and c["feedback"]]
        revision_prompt = "\n".join(f"- {i}" for i in issues) or "Refine the phrasing."
        state["attempt"] += 1
        state["pending_critiques"] = []
        save_state(ctx, msg.line_id, state)

        await ctx.send(TRANSLATION_ADDR, RevisionRequest(
            line_id=msg.line_id,
            original=state["original"],
            current_translation=state["current_translation"],
            target_lang=state["target_lang"],
            target_syllables=state["target_syllables"],
            revision_prompt=revision_prompt,
            attempt_number=state["attempt"],
        ))


# ── Attach protocol and run ───────────────────────────────────────────────────

orchestrator.include(protocol, publish_manifest=True)

if __name__ == "__main__":
    orchestrator.run()