"""
Step 2: Transcribe WAV with OpenAI Whisper (word-level timestamps).

Loads PCM WAV with the stdlib (no ffmpeg on PATH required). Whisper's
built-in loader shells out to ``ffmpeg``, which is often missing on Windows.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np
import whisper

_WHISPER_SR = 16000


def _load_wav_16k_mono(path: Path) -> np.ndarray:
    """Load a 16-bit WAV and return float32 mono audio at 16 kHz."""
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit PCM WAV, got sample width {wf.getsampwidth()}")
        nch = wf.getnchannels()
        sr = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        audio = audio.reshape(-1, nch).mean(axis=1)

    if sr == _WHISPER_SR:
        return audio.astype(np.float32)

    x_old = np.linspace(0.0, 1.0, num=len(audio), dtype=np.float64)
    new_len = max(1, int(round(len(audio) * _WHISPER_SR / sr)))
    x_new = np.linspace(0.0, 1.0, num=new_len, dtype=np.float64)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def transcribe_words(
    wav_path: str | Path,
    *,
    model_size: str = "base",
    language: str | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """
    Transcribe a WAV file and return one entry per word.

    Returns:
        [{"word": str, "start": float, "end": float}, ...]
    """
    path = Path(wav_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio not found: {path}")
    if path.suffix.lower() != ".wav":
        raise ValueError(f"Expected a .wav file, got: {path}")

    model = whisper.load_model(model_size, device=device)
    audio = _load_wav_16k_mono(path)
    result = model.transcribe(
        audio,
        language=language,
        word_timestamps=True,
        verbose=False,
    )

    words: list[dict[str, Any]] = []
    for segment in result.get("segments") or []:
        for w in segment.get("words") or []:
            text = str(w.get("word", "")).strip()
            if not text:
                continue
            words.append(
                {
                    "word": text,
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                }
            )

    return words


if __name__ == "__main__":
    import sys

    wav = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/audio/kJQP7kiw5Fk.wav")
    model = sys.argv[2] if len(sys.argv) > 2 else "base"
    lang_arg = sys.argv[3] if len(sys.argv) > 3 else "es"
    language = None if lang_arg.lower() == "auto" else lang_arg

    out = transcribe_words(wav, model_size=model, language=language)
    print(f"total words: {len(out)}")
    for i, row in enumerate(out[:20]):
        print(f"  {i}: {row}")

    if not out:
        raise SystemExit("no words returned")

    for i, row in enumerate(out):
        if "start" not in row or "end" not in row:
            raise SystemExit(f"missing timestamps at index {i}: {row}")
        if row["end"] < row["start"]:
            raise SystemExit(f"end < start at index {i}: {row}")

    print("timestamps: ok (every word has start/end, end >= start)")
