"""
Step 9: Concatenate timed TTS clips in order into one PCM WAV.

Inserts short silence between segments. ``None`` entries become a silent spacer
so order stays aligned with lyric indices.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import imageio_ffmpeg

DEFAULT_OUTPUT_PATH = "output/audio/final_lyrics.wav"
DEFAULT_SILENCE_BETWEEN_SEC = 0.15
DEFAULT_SAMPLE_RATE = 44100


def combine_adjusted_clips_to_wav(
    clip_paths: list[str | None],
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    silence_between_sec: float = DEFAULT_SILENCE_BETWEEN_SEC,
    silence_for_missing_sec: float | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> str:
    """
    Merge clips left-to-right into a single 16-bit PCM stereo WAV.

    Args:
        clip_paths: Ordered paths (e.g. from Step 8); ``None`` → silent spacer.
        output_path: Destination ``.wav`` (parent dirs created if needed).
        silence_between_sec: Gap inserted after each segment except the last.
        silence_for_missing_sec: Duration for ``None`` slots; defaults to
            ``silence_between_sec`` when omitted.
        sample_rate: Output sample rate (inputs are resampled to match).

    Returns:
        Absolute path to the written WAV file.

    Raises:
        ValueError: Empty list or no non-null clip paths.
    """
    if not clip_paths:
        raise ValueError("clip_paths must be non-empty")

    missing_dur = (
        float(silence_for_missing_sec)
        if silence_for_missing_sec is not None
        else float(silence_between_sec)
    )
    gap = float(silence_between_sec)

    parts: list[tuple[str, str | float]] = []
    for p in clip_paths:
        if p is not None and str(p).strip():
            fp = Path(str(p).strip()).resolve()
            if not fp.is_file():
                raise FileNotFoundError(f"Clip not found: {fp}")
            parts.append(("file", str(fp)))
        else:
            parts.append(("silence", missing_dur))

    if not any(k == "file" for k, _ in parts):
        raise ValueError("At least one non-null audio path is required")

    expanded: list[tuple[str, str | float]] = []
    for i, part in enumerate(parts):
        expanded.append(part)
        if i < len(parts) - 1:
            expanded.append(("silence", gap))

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd: list[str] = [ffmpeg, "-nostdin", "-y"]
    fmt_chain: list[str] = []
    labels: list[str] = []
    in_idx = 0

    for seg_i, (kind, val) in enumerate(expanded):
        lab = f"s{seg_i}"
        if kind == "file":
            cmd.extend(["-i", str(val)])
            fmt_chain.append(
                f"[{in_idx}:a]aformat=sample_fmts=s16:sample_rates={sample_rate}"
                f":channel_layouts=stereo[{lab}]"
            )
            labels.append(f"[{lab}]")
            in_idx += 1
        else:
            dur = float(val)
            lavfi = (
                f"anullsrc=channel_layout=stereo:sample_rate={sample_rate},"
                f"atrim=duration={dur},asetpts=PTS-STARTPTS"
            )
            cmd.extend(["-f", "lavfi", "-i", lavfi])
            fmt_chain.append(
                f"[{in_idx}:a]aformat=sample_fmts=s16:sample_rates={sample_rate}"
                f":channel_layouts=stereo[{lab}]"
            )
            labels.append(f"[{lab}]")
            in_idx += 1

    n_seg = len(labels)
    concat_expr = "".join(labels) + f"concat=n={n_seg}:v=0:a=1[outa]"
    filter_complex = ";".join(fmt_chain) + ";" + concat_expr

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[outa]",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(int(sample_rate)),
            "-ac",
            "2",
            str(out),
        ]
    )

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg combine failed ({proc.returncode}):\n{proc.stderr or proc.stdout}"
        )

    if not out.is_file():
        raise FileNotFoundError(f"Expected WAV not written: {out}")

    return str(out)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python combine_adjusted_audio.py <paths.json> <out.wav> [silence_between]\n"
            "paths.json: JSON array of string paths or null, in play order.",
            file=sys.stderr,
        )
        sys.exit(1)

    paths_data: list[Any] = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_wav = Path(sys.argv[2])
    gap = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_SILENCE_BETWEEN_SEC

    result = combine_adjusted_clips_to_wav(paths_data, out_wav, silence_between_sec=gap)
    print(result)
