"""
Step 5: Syllable count per line using a simple heuristic.

- Count contiguous groups of vowels (one group = one syllable contribution).
- Each word that contains at least one letter contributes at least 1 syllable.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

# Letters treated as vowels for grouping (covers common English + Spanish lyric text).
_VOWELS: Final[frozenset[str]] = frozenset("aeiouyáéíóúü")


def _letters_only(token: str) -> str:
    return "".join(
        c.lower()
        for c in token
        if unicodedata.category(c).startswith("L")
    )


def _vowel_groups_in_letters(letters: str) -> int:
    if not letters:
        return 0
    n = 0
    in_group = False
    for ch in letters:
        if ch in _VOWELS:
            if not in_group:
                n += 1
                in_group = True
        else:
            in_group = False
    return n


def _syllables_in_word(token: str) -> int:
    letters = _letters_only(token)
    if not letters:
        return 0
    groups = _vowel_groups_in_letters(letters)
    return max(1, groups)


def count_syllables_in_line(line: str) -> int:
    """
    Total syllable count for one line (sum over whitespace-separated tokens).

    Heuristic:
    - Vowel groups: each maximal run of vowel letters counts as one.
    - Per word (token with ≥1 letter): at least 1 syllable.
    """
    if not line or not line.strip():
        return 0
    # Split on whitespace; keep lyric tokens like "don't" as one word.
    tokens = re.split(r"\s+", line.strip())
    return sum(_syllables_in_word(t) for t in tokens if t)


if __name__ == "__main__":
    assert count_syllables_in_line("hello world") == 3  # hello: e,o; world: o
    assert count_syllables_in_line("the") == 1
    assert count_syllables_in_line("strength") == 1
    assert count_syllables_in_line("a") == 1
    assert count_syllables_in_line("rhythm") == 1  # y counts as vowel → one group
    assert count_syllables_in_line("") == 0
    assert count_syllables_in_line("   ") == 0
    assert count_syllables_in_line("¿Hola, mundo?") == 4  # Hola: o,a; mundo: u,o
    print("syllable_counter: ok", count_syllables_in_line("hello world"))
