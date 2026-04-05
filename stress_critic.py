import os
import re
from uagents import Agent, Context
from models import LyricLine, CritiqueResult

try:
    import pronouncing
    PRONOUNCING_AVAILABLE = True
except ImportError:
    PRONOUNCING_AVAILABLE = False

stress_agent = Agent(
    name="stress_critic",
    seed="stress_critic_seed_phrase",
    port=8003,
    endpoint=["http://localhost:8003/submit"],
    mailbox=False,
)


def get_stress_pattern(text: str) -> list[int]:
    """
    Returns a list of stress values for each syllable in the text.
    1 = stressed, 0 = unstressed

    Example:
      "Never gonna" -> [1, 0, 1, 0]
    """
    pattern = []
    words = re.sub(r'[^\w\s]', '', text.lower()).split()

    for word in words:
        if PRONOUNCING_AVAILABLE:
            phones_list = pronouncing.phones_for_word(word)
            if phones_list:
                # Extract stress digits from phonemes (0, 1, or 2)
                stresses = [
                    int(p[-1]) for p in phones_list[0].split()
                    if p[-1].isdigit()
                ]
                # Convert to binary: 1 = stressed (1 or 2), 0 = unstressed
                pattern.extend([1 if s >= 1 else 0 for s in stresses])
                continue

        # Fallback: assume stress on first syllable of each word
        vowel_groups = re.findall(r'[aeiouAEIOU]+', word)
        syllable_count = max(1, len(vowel_groups))
        pattern.extend([1] + [0] * (syllable_count - 1))

    return pattern


def compare_stress_patterns(original: list[int], translated: list[int]) -> float:
    """
    Compare two stress patterns and return a similarity score 0.0 - 1.0.

    Strategy: align the patterns by resampling the shorter one to match
    the length of the longer one, then compute % of positions that match.

    This handles cases where syllable count is slightly different.
    """
    if not original or not translated:
        return 1.0  # Can't compare, give benefit of the doubt

    # Resample translated pattern to match original length
    orig_len = len(original)
    trans_len = len(translated)

    if orig_len == trans_len:
        resampled = translated
    else:
        # Linear interpolation resample
        resampled = []
        for i in range(orig_len):
            idx = int(i * trans_len / orig_len)
            idx = min(idx, trans_len - 1)
            resampled.append(translated[idx])

    # Count matching positions
    matches = sum(1 for a, b in zip(original, resampled) if a == b)
    return matches / orig_len


STRESS_THRESHOLD = 0.65  # 65% of stress positions must match


@stress_agent.on_message(model=LyricLine)
async def critique_stress(ctx: Context, sender: str, msg: LyricLine):
    orig_pattern = get_stress_pattern(msg.original)
    trans_pattern = get_stress_pattern(msg.translated)
    score = compare_stress_patterns(orig_pattern, trans_pattern)
    passed = score >= STRESS_THRESHOLD

    feedback = ""
    if not passed:
        orig_str = " ".join(["S" if s else "u" for s in orig_pattern])
        trans_str = " ".join(["S" if s else "u" for s in trans_pattern])
        feedback = (
            f"Stress pattern mismatch (score: {score:.0%}). "
            f"Original pattern: [{orig_str}]. "
            f"Translation pattern: [{trans_str}]. "
            f"Try to match which syllables are emphasized — "
            f"stressed syllables should land on the beat."
        )

    ctx.logger.info(
        f"Line {msg.line_id} | stress score={score:.2f} passed={passed} "
        f"orig={orig_pattern} trans={trans_pattern}"
    )

    await ctx.send(sender, CritiqueResult(
        line_id=msg.line_id,
        critic="stress",
        passed=passed,
        syllable_delta=0,
        feedback=feedback,
    ))


if __name__ == "__main__":
    stress_agent.run()
