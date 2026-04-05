import os
from uagents import Agent, Context
from models import LyricLine, RevisionRequest
import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

LANG_NAMES = {
    "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ja": "Japanese",
}

translation_agent = Agent(
    name="translation_agent",
    seed="translation_agent_seed_phrase",
    port=8002,
    endpoint=["http://localhost:8002/submit"],
    mailbox=False,
)


def call_llm(prompt: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


@translation_agent.on_message(model=LyricLine)
async def initial_translate(ctx: Context, sender: str, msg: LyricLine):
    lang = LANG_NAMES.get(msg.target_lang, msg.target_lang)
    prompt = (
        f"Translate this English song lyric into {lang}.\n"
        f"Original: \"{msg.original}\"\n"
        f"Match the syllable count as closely as possible (~{msg.target_syllables} syllables).\n"
        f"Return ONLY the translated line, no explanation."
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
        f"Rewrite this {lang} song lyric translation.\n"
        f"Original English: \"{msg.original}\"\n"
        f"Current translation (attempt {msg.attempt_number}): \"{msg.current_translation}\"\n"
        f"Problem: {msg.revision_prompt}\n"
        f"Must have exactly {msg.target_syllables} syllables.\n"
        f"Return ONLY the revised line, no explanation."
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
