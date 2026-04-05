import os
from dotenv import load_dotenv
from uagents import Agent, Context
from models import LyricLine, RevisionRequest
import anthropic

load_dotenv(".env/keys.env")

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

LANG_NAMES = {
    "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ja": "Japanese", "en": "English",
}

translation_agent = Agent(
    name="translation_agent",
    seed="translation_agent_seed_phrase",
    port=8002,
    endpoint=["http://localhost:8002/submit"],
    mailbox=False,
)


SYSTEM_PROMPT = (
    "You are a localization specialist for audio dubbing. "
    "You adapt spoken phrases from one language to another, matching syllable count and stress patterns "
    "so the adapted text fits the original audio timing. "
    "Respond with only the adapted phrase — no commentary, no refusals, no explanations."
)


def call_llm(prompt: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""},
        ],
    )
    return response.content[0].text.strip()


@translation_agent.on_message(model=LyricLine)
async def initial_translate(ctx: Context, sender: str, msg: LyricLine):
    lang = LANG_NAMES.get(msg.target_lang, msg.target_lang)
    prompt = (
        f"Adapt this phrase into {lang}.\n"
        f"Text: \"{msg.original}\"\n"
        f"Target syllable count: ~{msg.target_syllables} syllables.\n"
        f"Return ONLY the adapted phrase."
    )
    translated = call_llm(prompt)
    ctx.logger.info(f"Line {msg.line_id}: '{msg.original}' -> '{translated}'")

    await ctx.send(sender, LyricLine(
        line_id=msg.line_id,
        original=msg.original,
        translated=translated,
        target_lang=msg.target_lang,
        target_syllables=msg.target_syllables,
    ))


@translation_agent.on_message(model=RevisionRequest)
async def revise(ctx: Context, sender: str, msg: RevisionRequest):
    lang = LANG_NAMES.get(msg.target_lang, msg.target_lang)
    prompt = (
        f"Revise this {lang} phrase (attempt {msg.attempt_number}).\n"
        f"Original: \"{msg.original}\"\n"
        f"Current version: \"{msg.current_translation}\"\n"
        f"Issue: {msg.revision_prompt}\n"
        f"Must have exactly {msg.target_syllables} syllables.\n"
        f"Return ONLY the revised phrase."
        f"The number of number of syllables is more important than the semantics"
    )
    revised = call_llm(prompt)
    ctx.logger.info(f"Line {msg.line_id} revision {msg.attempt_number}: '{revised}'")

    await ctx.send(sender, LyricLine(
        line_id=msg.line_id,
        original=msg.original,
        translated=revised,
        target_lang=msg.target_lang,
        target_syllables=msg.target_syllables,
    ))


if __name__ == "__main__":
    translation_agent.run()
