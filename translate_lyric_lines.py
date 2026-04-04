"""
Step 4–6: Translate Spanish lyric lines to English with an LLM, then (optional)
match English syllable count to the Spanish line (Step 6).

Expects Step 3 output: [{"line": str, "start": float, "end": float}, ...].
Requires OPENAI_API_KEY (see https://platform.openai.com/docs/overview).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from syllable_counter import count_syllables_in_line

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_CHUNK_SIZE = 28
DEFAULT_SYLLABLE_REWRITE_ATTEMPTS = 3

_SYSTEM = """You are a translator for song lyrics.

You translate Spanish into English. Rules:
- Keep meaning, tone, and imagery accurate (slang, idioms, emotional nuance).
- The English should read like natural song lyrics: concise, singable, and idiomatic—not stiff or like a legal document.
- Do not add explanations, notes, or alternate options.
- Do not merge or split lines; each input line maps to exactly one English line."""

_USER_INSTRUCTIONS = """Return a JSON object with exactly one key, "translations", whose value is an array of strings.
There must be exactly {n} strings, in the same order as the Spanish lines below (line 1 → first string, etc.).

Spanish lines:
{numbered}"""


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", t, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def _parse_single_translation_json(raw: str) -> str:
    text = _strip_json_fence(raw)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Model JSON must be an object with key 'translation'")
    t = data.get("translation")
    if not isinstance(t, str) or not t.strip():
        raise ValueError("Model JSON must contain non-empty string 'translation'")
    return t.strip()


def _parse_translations_json(raw: str, expected_n: int) -> list[str]:
    text = _strip_json_fence(raw)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Model JSON must be an object with key 'translations'")
    arr = data.get("translations")
    if not isinstance(arr, list):
        raise ValueError("Model JSON must contain 'translations' array")
    out = [str(x).strip() for x in arr]
    if len(out) != expected_n:
        raise ValueError(f"Expected {expected_n} translations, got {len(out)}")
    return out


_REWRITE_SYSTEM = """You rewrite English song lyric lines to match a syllable target.

Syllable counting (you must satisfy this exact rule when aiming for the target):
- Split the line on whitespace into words.
- Each word that contains letters contributes at least one syllable.
- Within each word, count one syllable per maximal consecutive run of these vowels: a, e, i, o, u, y, á, é, í, ó, ú, ü.

The English line's total syllable count by that rule must equal the target number.

Keep the same meaning and emotional tone as the Spanish original. The English should sound like natural, singable lyrics—not stiff or explanatory.

Return JSON only with shape: {"translation": "<single English line>"}"""


def _rewrite_line_to_syllable_count(
    spanish_line: str,
    english_line: str,
    target_syllables: int,
    *,
    client: OpenAI,
    model: str,
) -> str:
    user = f"""Spanish original:
{spanish_line}

Current English line:
{english_line}

Rewrite this English line to have exactly {target_syllables} syllables (counted with the rule above). Preserve meaning and lyric quality."""

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _REWRITE_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content
    if not raw:
        raise RuntimeError("Empty response from model (syllable rewrite)")
    return _parse_single_translation_json(raw)


def _match_translation_syllables(
    spanish_line: str,
    english_line: str,
    *,
    client: OpenAI,
    model: str,
    max_iterations: int = DEFAULT_SYLLABLE_REWRITE_ATTEMPTS,
) -> str:
    """
    If English syllable count differs from Spanish, ask the LLM to rewrite (up to
    ``max_iterations`` times). Returns the last candidate if still unmatched.
    """
    es = str(spanish_line).strip()
    current = str(english_line).strip()
    target = count_syllables_in_line(es)
    if target == 0:
        return current
    if count_syllables_in_line(current) == target:
        return current
    attempts = max(0, int(max_iterations))
    for _ in range(attempts):
        current = _rewrite_line_to_syllable_count(
            es, current, target, client=client, model=model
        )
        if count_syllables_in_line(current) == target:
            return current
    return current


def _translate_chunk(
    lines: list[dict[str, Any]],
    *,
    client: OpenAI,
    model: str,
    match_syllables: bool,
    syllable_rewrite_attempts: int,
) -> list[dict[str, Any]]:
    n = len(lines)
    numbered = "\n".join(
        f"{i + 1}. {str(row.get('line', '')).strip()}" for i, row in enumerate(lines)
    )
    user = _USER_INSTRUCTIONS.format(n=n, numbered=numbered)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content
    if not raw:
        raise RuntimeError("Empty response from model")
    translations = _parse_translations_json(raw, n)

    result: list[dict[str, Any]] = []
    for row, en in zip(lines, translations, strict=True):
        es_line = str(row.get("line", "")).strip()
        translation = en
        if match_syllables:
            translation = _match_translation_syllables(
                es_line,
                translation,
                client=client,
                model=model,
                max_iterations=syllable_rewrite_attempts,
            )
        result.append(
            {
                "line": es_line,
                "translation": translation,
                "start": float(row["start"]),
                "end": float(row["end"]),
            }
        )
    return result


def translate_lyric_lines(
    lines: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    client: OpenAI | None = None,
    match_syllables: bool = True,
    syllable_rewrite_attempts: int = DEFAULT_SYLLABLE_REWRITE_ATTEMPTS,
) -> list[dict[str, Any]]:
    """
    Translate each Spanish lyric line to English.

    Args:
        lines: Step 3 items with "line", "start", "end".
        model: OpenAI chat model id.
        chunk_size: Lines per API call (long songs are chunked).
        client: Optional ``OpenAI()`` instance (for tests or custom base_url).
        match_syllables: If True (default), rewrite English so its syllable count
            matches the Spanish line (same heuristic as ``syllable_counter``),
            trying up to ``syllable_rewrite_attempts`` LLM rewrites when needed.
        syllable_rewrite_attempts: Max rewrite calls per line (default 3).

    Returns:
        [
            {
                "line": str,           # original Spanish
                "translation": str,  # English (syllable-matched when enabled)
                "start": float,
                "end": float,
            },
            ...
        ]
    """
    if not lines:
        return []

    # API key: set OPENAI_API_KEY; optional OPENAI_BASE_URL for proxies/Azure-compatible endpoints.
    c = client or OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)

    out: list[dict[str, Any]] = []
    size = max(1, int(chunk_size))
    for i in range(0, len(lines), size):
        chunk = lines[i : i + size]
        out.extend(
            _translate_chunk(
                chunk,
                client=c,
                model=model,
                match_syllables=match_syllables,
                syllable_rewrite_attempts=syllable_rewrite_attempts,
            )
        )
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path

    demo_lines = [
        {"line": "Voy caminando solo bajo la luna.", "start": 0.0, "end": 3.2},
        {"line": "El viento cuenta secretos que nadie escucha.", "start": 3.5, "end": 7.1},
    ]

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "Set OPENAI_API_KEY to run translation. Example structure (no API call):",
            file=sys.stderr,
        )
        sys.stderr.flush()
        print(
            json.dumps(
                [
                    {
                        "line": row["line"],
                        "translation": "<English here>",
                        "start": row["start"],
                        "end": row["end"],
                    }
                    for row in demo_lines
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        sys.exit(0)

    if len(sys.argv) >= 2:
        path = Path(sys.argv[1])
        lines = json.loads(path.read_text(encoding="utf-8"))
    else:
        lines = demo_lines

    merged = translate_lyric_lines(lines)
    print(json.dumps(merged, indent=2, ensure_ascii=False))
