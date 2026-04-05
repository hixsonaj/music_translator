import os
import re
from uagents import Agent, Context
from models import LyricLine, CritiqueResult

try:
    import pyphen
    PYPHEN_AVAILABLE = True
except ImportError:
    PYPHEN_AVAILABLE = False

TOLERANCE = 1

syllable_agent = Agent(
    name="syllable_critic",
    seed="syllable_critic_seed_phrase",
    port=8001,
    endpoint=["http://localhost:8001/submit"],
    mailbox=False,
)


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
    return max(1, len(re.findall(r'[aeiouAEIOU]+', text)))


@syllable_agent.on_message(model=LyricLine)
async def critique(ctx: Context, sender: str, msg: LyricLine):
    original_count = count_syllables(msg.original, "en")
    translated_count = count_syllables(msg.translated, msg.target_lang)
    delta = translated_count - original_count
    passed = abs(delta) <= TOLERANCE

    feedback = ""
    if not passed:
        direction = "fewer" if delta > 0 else "more"
        feedback = (
            f"Translation has {abs(delta)} {direction} syllable(s) than needed. "
            f"Target: {original_count}, current: {translated_count}."
        )

    ctx.logger.info(f"Line {msg.line_id} | delta={delta:+d} passed={passed}")

    await ctx.send(sender, CritiqueResult(
        line_id=msg.line_id,
        critic="syllable",
        passed=passed,
        syllable_delta=delta,
        feedback=feedback,
    ))


if __name__ == "__main__":
    syllable_agent.run()