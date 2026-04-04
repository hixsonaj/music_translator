"""
Step 8: Stretch or compress each TTS clip to match lyric line duration (end - start).

Uses bundled ffmpeg (via imageio_ffmpeg) with the ``atempo`` filter (pitch-preserving).
Chains multiple ``atempo`` stages when the required ratio is outside [0.5, 2.0].
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import imageio_ffmpeg

DEFAULT_OUTPUT_DIR = "output/audio"
DEFAULT_OUTPUT_SUFFIX = "_timed"
MIN_TARGET_SEC = 0.05
_RATIO_NEAR_ONE = 1e-3

_DURATION_RE = re.compile(
    r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)",
    re.IGNORECASE,
)


def probe_audio_duration_seconds(path: str | Path) -> float:
    """Decode container duration via ``ffmpeg -i`` stderr (works for MP3)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Audio not found: {p}")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    r = subprocess.run(
        [ffmpeg, "-nostdin", "-i", str(p.resolve())],
        capture_output=True,
        text=True,
        timeout=120,
    )
    m = _DURATION_RE.search(r.stderr or "")
    if not m:
        raise RuntimeError(f"Could not parse duration from ffmpeg for: {p}")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def _atempo_factors(ratio: float) -> list[float]:
    """
    Build ``atempo`` factors whose product equals ``ratio`` (source / target duration).

    Each factor must be in [0.5, 2.0] per ffmpeg limits.
    """
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    factors: list[float] = []
    x = ratio
    while x < 0.5 - 1e-9:
        factors.append(0.5)
        x /= 0.5
    while x > 2.0 + 1e-9:
        factors.append(2.0)
        x /= 2.0
    if abs(x - 1.0) > 1e-5:
        factors.append(max(0.5, min(2.0, x)))
    if not factors:
        factors.append(1.0)
    return factors


def _atempo_filter_chain(ratio: float) -> str:
    parts = [f"atempo={f:.6g}" for f in _atempo_factors(ratio)]
    return ",".join(parts)


def _run_ffmpeg_atempo(src: Path, dst: Path, ratio: float) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    filt = _atempo_filter_chain(ratio)
    cmd = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-i",
        str(src.resolve()),
        "-vn",
        "-filter:a",
        filt,
        "-acodec",
        "libmp3lame",
        "-q:a",
        "4",
        str(dst.resolve()),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg atempo failed ({proc.returncode}):\n{proc.stderr or proc.stdout}"
        )


def match_line_audio_timing(
    entries: list[dict[str, Any]],
    *,
    path_key: str = "audio_path",
    start_key: str = "start",
    end_key: str = "end",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    file_stem: str | None = None,
    output_suffix: str = DEFAULT_OUTPUT_SUFFIX,
    min_target_sec: float = MIN_TARGET_SEC,
) -> list[str | None]:
    """
    Produce time-stretched copies of each clip to match ``end - start``.

    Args:
        entries: Dicts with optional ``path_key`` (clip path), ``start_key``, ``end_key``.
        path_key: Key for input MP3 path; missing/null skips that index (returns None).
        output_dir: Where to write ``{stem}_{i:04d}{suffix}.mp3``.
        file_stem: Output prefix; if None, uses ``en_line``.
        output_suffix: Inserted before ``.mp3`` (default ``_timed``).
        min_target_sec: Floor for target duration to avoid extreme stretch.

    Returns:
        Absolute paths to adjusted files, ``None`` where there was no input path.
    """
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = (file_stem or "en_line").strip() or "en_line"
    suffix = output_suffix if output_suffix.startswith("_") else f"_{output_suffix}"

    results: list[str | None] = []
    for i, row in enumerate(entries):
        raw_path = row.get(path_key)
        if raw_path is None or (isinstance(raw_path, str) and not raw_path.strip()):
            results.append(None)
            continue

        src = Path(str(raw_path).strip()).resolve()
        start = float(row[start_key])
        end = float(row[end_key])
        target = max(float(end) - float(start), float(min_target_sec))

        dst = out / f"{stem}_{i:04d}{suffix}.mp3"
        source_dur = probe_audio_duration_seconds(src)
        if source_dur <= 0:
            raise RuntimeError(f"Non-positive duration for {src}")

        ratio = source_dur / target
        if abs(ratio - 1.0) <= _RATIO_NEAR_ONE:
            shutil.copy2(src, dst)
        else:
            _run_ffmpeg_atempo(src, dst, ratio)

        # Verify duration roughly matches (encode padding may add tens of ms).
        new_dur = probe_audio_duration_seconds(dst)
        if new_dur <= 0:
            raise RuntimeError(f"Adjusted file has invalid duration: {dst}")

        results.append(str(dst.resolve()))

    return results


def match_timing_for_generated_clips(
    lines: list[dict[str, Any]],
    tts_paths: list[str | None],
    **kwargs: Any,
) -> list[str | None]:
    """
    Zip bilingual/TTS rows with paths from :func:`generate_english_audio.generate_english_audio`.

    ``lines`` and ``tts_paths`` must have the same length. Each row should include
    ``start`` and ``end``; ``audio_path`` is injected from ``tts_paths``.
    """
    if len(lines) != len(tts_paths):
        raise ValueError("lines and tts_paths must have the same length")
    merged: list[dict[str, Any]] = []
    for row, ap in zip(lines, tts_paths, strict=True):
        d = dict(row)
        d["audio_path"] = ap
        merged.append(d)
    return match_line_audio_timing(merged, **kwargs)


if __name__ == "__main__":
    # Sanity-check atempo decomposition
    for r in (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
        fs = _atempo_factors(r)
        prod = 1.0
        for f in fs:
            prod *= f
        assert abs(prod - r) < 1e-5, (r, fs, prod)
    print("atempo factors: ok")

    if len(sys.argv) < 2:
        print(
            "(optional) python match_audio_timing.py <entries.json> [file_stem]\n"
            "Each entry: {\"start\": float, \"end\": float, \"audio_path\": str|null}",
            file=sys.stderr,
        )
        sys.exit(0)

    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    stem = sys.argv[2] if len(sys.argv) > 2 else None
    out_paths = match_line_audio_timing(data, file_stem=stem)
    print(json.dumps(out_paths, indent=2))
