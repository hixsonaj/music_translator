"""
Pre-Translation Pipeline
-------------------------
Takes a song URL (YouTube or direct audio link) and produces:
  - separated/vocals.wav
  - separated/no_vocals.wav
  - segments.json  (Whisper transcription with timestamps + syllable counts)

These outputs feed directly into the translation agents and then audio_pipeline.py.

Pipeline:
  1. Download audio via yt-dlp
  2. Separate vocals/instrumental via demucs
  3. Transcribe vocals via Whisper (word-level timestamps)
  4. Count syllables per segment
  5. Write segments.json

Requirements:
  pip install yt-dlp demucs openai-whisper pyphen

Usage:
  python pre_pipeline.py --url "https://www.youtube.com/watch?v=..." --lang en
  python pre_pipeline.py --file my_song.mp3 --lang en
"""

import os
import re
import json
import argparse
import subprocess
import tempfile
from pathlib import Path

try:
    import pyphen
    PYPHEN_AVAILABLE = True
except ImportError:
    PYPHEN_AVAILABLE = False


# ── Syllable counting (same logic as syllable_critic.py) ─────────────────────

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
    return sum(max(1, len(re.findall(r'[aeiouAEIOU]+', word))) for word in text.split())


# ── Step 1: Download ──────────────────────────────────────────────────────────

def download_audio(url: str, output_dir: str) -> str:
    """
    Download audio from a URL using yt-dlp.
    Returns path to downloaded mp3.
    """
    print(f"Downloading: {url}")
    output_template = os.path.join(output_dir, "source.%(ext)s")

    subprocess.run(
        [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "--output", output_template,
            "--no-playlist",
            url,
        ],
        check=True,
    )

    output_path = os.path.join(output_dir, "source.mp3")
    if not os.path.exists(output_path):
        raise FileNotFoundError("yt-dlp finished but no source.mp3 found")

    print(f"  Downloaded: {output_path}")
    return output_path


# ── Step 2: Vocal separation ──────────────────────────────────────────────────

def separate_vocals(audio_path: str, output_dir: str) -> tuple[str, str]:
    """
    Run demucs htdemucs model to split vocals from instrumental.
    Returns (vocals_path, no_vocals_path).
    """
    print("Separating vocals (demucs)...")

    subprocess.run(
        [
            "python", "-m", "demucs",
            "--two-stems", "vocals",
            "--out", output_dir,
            audio_path,
        ],
        check=True,
    )

    # Demucs outputs to: output_dir/htdemucs/<track_stem>/vocals.wav
    stem = Path(audio_path).stem
    htdemucs_dir = Path(output_dir) / "htdemucs" / stem

    vocals_path = htdemucs_dir / "vocals.wav"
    no_vocals_path = htdemucs_dir / "no_vocals.wav"

    if not vocals_path.exists():
        raise FileNotFoundError(f"Expected vocals at {vocals_path}")
    if not no_vocals_path.exists():
        raise FileNotFoundError(f"Expected instrumental at {no_vocals_path}")

    print(f"  Vocals:       {vocals_path}")
    print(f"  Instrumental: {no_vocals_path}")
    return str(vocals_path), str(no_vocals_path)


# ── Step 3: Transcribe ────────────────────────────────────────────────────────

def transcribe(vocals_path: str, source_lang: str = "en", whisper_model: str = "medium") -> list[dict]:
    """
    Run Whisper on the vocal track to get timestamped segments.
    Returns list of {"start", "end", "text"} dicts.
    """
    import hashlib
    cache_path = vocals_path + ".whisper.json"
    if os.path.exists(cache_path):
        print(f"Loading cached Whisper transcript: {cache_path}")
        with open(cache_path, "r") as f:
            segments = json.load(f)
        print(f"  {len(segments)} segments loaded from cache")
        return segments

    print(f"Transcribing with Whisper ({whisper_model})...")
    import whisper
    model = whisper.load_model(whisper_model)
    result = model.transcribe(
        vocals_path,
        language=source_lang,
        word_timestamps=True,
        verbose=False,
    )
    segments = []
    for seg in result["segments"]:
        text = seg["text"].strip()
        if not text:
            continue
        segments.append({
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": text,
        })
        print(f"  [{seg['start']:.2f}s – {seg['end']:.2f}s] {text}")
    print(f"  {len(segments)} segments found")
    # Save to cache
    with open(cache_path, "w") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    return segments


# ── Step 4: Add syllable counts ───────────────────────────────────────────────

def add_syllable_counts(segments: list[dict], lang: str = "en") -> list[dict]:
    for seg in segments:
        seg["syllables"] = count_syllables(seg["text"], lang)
    return segments


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(
    url: str = None,
    file: str = None,
    output_dir: str = "separated",
    source_lang: str = "en",
    whisper_model: str = "medium",
    segments_out: str = "segments.json",
) -> dict:
    """
    Run the full pre-translation pipeline.

    Pass either `url` (YouTube/direct link) or `file` (local audio path).

    Returns dict with keys:
      vocals_path, instrumental_path, segments
    """
    if not url and not file:
        raise ValueError("Provide either url or file")

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Download (skip if local file provided)
    if url:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = download_audio(url, tmpdir)
            vocals_path, instrumental_path = separate_vocals(audio_path, output_dir)
    else:
        audio_path = file
        vocals_path, instrumental_path = separate_vocals(audio_path, output_dir)

    # Step 3: Transcribe
    segments = transcribe(vocals_path, source_lang, whisper_model)

    # Step 4: Syllable counts
    segments = add_syllable_counts(segments, source_lang)

    # Step 5: Write segments.json
    with open(segments_out, "w") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    print(f"\nSegments written to {segments_out}")

    result = {
        "vocals_path": vocals_path,
        "instrumental_path": instrumental_path,
        "segments": segments,
    }

    print("\n── Pre-pipeline complete ──────────────────────────")
    print(f"  Vocals:       {vocals_path}")
    print(f"  Instrumental: {instrumental_path}")
    print(f"  Segments:     {segments_out}  ({len(segments)} lines)")
    print("  Next: run translation agents, then audio_pipeline.py")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-translation audio pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url",  help="YouTube or direct audio URL")
    group.add_argument("--file", help="Path to local audio file")
    parser.add_argument("--lang",          default="en",        help="Source language code (default: en)")
    parser.add_argument("--output-dir",    default="separated", help="Where to write demucs output")
    parser.add_argument("--whisper-model", default="medium",    help="Whisper model size (tiny/base/small/medium/large)")
    parser.add_argument("--segments-out",  default="segments.json", help="Output path for segments JSON")
    args = parser.parse_args()

    run(
        url=args.url,
        file=args.file,
        output_dir=args.output_dir,
        source_lang=args.lang,
        whisper_model=args.whisper_model,
        segments_out=args.segments_out,
    )
