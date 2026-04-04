"""
Step 3: Group transcribed words into lyric lines by inter-word silence.

If the gap between the end of one word and the start of the next exceeds
``gap_threshold_sec`` (default 0.8), the next word begins a new line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_GAP_NEW_LINE_SEC = 0.8


def _flush_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    parts = [str(g.get("word", "")).strip() for g in group]
    line = " ".join(p for p in parts if p)
    return {
        "line": line,
        "start": float(group[0]["start"]),
        "end": float(group[-1]["end"]),
    }


def group_words_into_lines(
    words: list[dict[str, Any]],
    *,
    gap_threshold_sec: float = DEFAULT_GAP_NEW_LINE_SEC,
) -> list[dict[str, Any]]:
    """
    Build lyric lines from Whisper-style word entries.

    Args:
        words: [{"word": str, "start": float, "end": float}, ...] in time order.
        gap_threshold_sec: Start a new line when (next.start - prev.end) > this.

    Returns:
        [{"line": str, "start": float, "end": float}, ...]
    """
    if not words:
        return []

    lines: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = [words[0]]

    for i in range(1, len(words)):
        w = words[i]
        prev = words[i - 1]
        gap = float(w["start"]) - float(prev["end"])
        if gap > gap_threshold_sec:
            lines.append(_flush_group(current))
            current = [w]
        else:
            current.append(w)

    lines.append(_flush_group(current))
    return lines


if __name__ == "__main__":
    import json
    import sys

    # Synthetic check: gap 1.0s between "world" and "after" → two lines
    demo = [
        {"word": "hello", "start": 0.0, "end": 0.4},
        {"word": "world", "start": 0.5, "end": 0.9},
        {"word": "after", "start": 2.0, "end": 2.4},
    ]
    grouped = group_words_into_lines(demo)
    assert len(grouped) == 2
    assert grouped[0]["line"] == "hello world"
    assert grouped[0]["start"] == 0.0 and grouped[0]["end"] == 0.9
    assert grouped[1]["line"] == "after"
    assert grouped[1]["start"] == 2.0 and grouped[1]["end"] == 2.4
    print("synthetic demo:", json.dumps(grouped, indent=2))

    if len(sys.argv) >= 2:
        from whisper_transcribe import transcribe_words

        wav = Path(sys.argv[1])
        model = sys.argv[2] if len(sys.argv) > 2 else "base"
        lang_arg = sys.argv[3] if len(sys.argv) > 3 else "auto"
        language = None if lang_arg.lower() == "auto" else lang_arg
        words = transcribe_words(wav, model_size=model, language=language)
        lines = group_words_into_lines(words)
        print(json.dumps(lines, indent=2, ensure_ascii=False))
