"""
Step 7: Generate English audio per translated line via OpenAI TTS.

Expects Step 4–6 items with a ``translation`` field (English text).
Writes one MP3 per non-empty line under ``output/audio/`` by default.

Requires OPENAI_API_KEY (same as translation). For duration matching, see
``match_audio_timing.match_line_audio_timing`` (Step 8).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

DEFAULT_OUTPUT_DIR = "output/audio"
DEFAULT_VOICE = "alloy"
DEFAULT_TTS_MODEL = "tts-1"


def generate_english_audio(
    lines: list[dict[str, Any]],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    voice: str = DEFAULT_VOICE,
    model: str = DEFAULT_TTS_MODEL,
    file_stem: str | None = None,
    client: OpenAI | None = None,
) -> list[str | None]:
    """
    Call TTS once per line and save MP3 files.

    Args:
        lines: Dicts with ``translation`` (English). Other keys ignored here.
        output_dir: Directory for MP3s (created if missing).
        voice: OpenAI TTS voice (e.g. alloy, echo, fable, onyx, nova, shimmer).
        model: ``tts-1`` or ``tts-1-hd``.
        file_stem: Filename prefix; default ``en_line`` if omitted.
        client: Optional ``OpenAI()`` instance.

    Returns:
        Absolute paths per input index; ``None`` where ``translation`` was empty
        (no file written).
    """
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = (file_stem or "en_line").strip() or "en_line"

    c = client or OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)
    paths: list[str | None] = []

    for i, row in enumerate(lines):
        text = str(row.get("translation", "")).strip()
        if not text:
            paths.append(None)
            continue

        path = out / f"{stem}_{i:04d}.mp3"
        response = c.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
        )
        path.write_bytes(response.content)
        paths.append(str(path))

    return paths


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to generate TTS audio.", file=sys.stderr)
        sys.stderr.flush()
        example = [
            {
                "line": "Hola.",
                "translation": "Hello there.",
                "start": 0.0,
                "end": 1.0,
            }
        ]
        print(json.dumps({"paths_would_be": [str(Path(DEFAULT_OUTPUT_DIR).resolve() / "en_line_0000.mp3")], "example_input": example}, indent=2))
        sys.exit(0)

    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    file_stem = sys.argv[2] if len(sys.argv) > 2 else None

    if json_path is None:
        demo = [
            {"line": "Una línea.", "translation": "One lyric line.", "start": 0.0, "end": 2.0},
        ]
        lines = demo
    else:
        lines = json.loads(json_path.read_text(encoding="utf-8"))

    paths = generate_english_audio(lines, file_stem=file_stem)
    print(json.dumps(paths, indent=2))
