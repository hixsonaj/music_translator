"""
Step 10: End-to-end pipeline — YouTube URL → final English WAV + translated lyrics.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from combine_adjusted_audio import combine_adjusted_clips_to_wav
from generate_english_audio import DEFAULT_TTS_MODEL, DEFAULT_VOICE, generate_english_audio
from group_words_into_lines import DEFAULT_GAP_NEW_LINE_SEC, group_words_into_lines
from match_audio_timing import DEFAULT_OUTPUT_SUFFIX, match_timing_for_generated_clips
from translate_lyric_lines import DEFAULT_CHUNK_SIZE, DEFAULT_MODEL, translate_lyric_lines
from whisper_transcribe import transcribe_words
from youtube_audio import download_youtube_audio


def run_pipeline(
    youtube_url: str,
    *,
    audio_dir: str | Path = "output/audio",
    whisper_model: str = "base",
    whisper_language: str | None = "es",
    whisper_device: str | None = None,
    line_gap_sec: float = DEFAULT_GAP_NEW_LINE_SEC,
    translation_model: str = DEFAULT_MODEL,
    translation_chunk_size: int = DEFAULT_CHUNK_SIZE,
    openai_client: OpenAI | None = None,
    match_syllables: bool = True,
    syllable_rewrite_attempts: int = 3,
    tts_model: str = DEFAULT_TTS_MODEL,
    tts_voice: str = DEFAULT_VOICE,
    timed_suffix: str = DEFAULT_OUTPUT_SUFFIX,
    final_wav: str | Path | None = None,
    silence_between_clips: float = 0.15,
    silence_for_missing: float | None = None,
    lyrics_json_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run steps 1–9 in order.

    Args:
        youtube_url: Video URL for yt-dlp.
        audio_dir: Where WAV/MP3/mix files are stored.
        whisper_model / whisper_language / whisper_device: Passed to Whisper.
        line_gap_sec: New lyric line when inter-word gap exceeds this (seconds).
        translation_model / translation_chunk_size: OpenAI chat translation.
        openai_client: Shared ``OpenAI()`` for translate + TTS; if ``None``, a client
            is created when ``OPENAI_API_KEY`` is set.
        match_syllables / syllable_rewrite_attempts: Step 6 behavior.
        tts_model / tts_voice: OpenAI speech API.
        timed_suffix: Suffix for time-stretched clips (Step 8).
        final_wav: Output mix path; default ``{audio_dir}/{video_id}_final.wav``.
        silence_between_clips / silence_for_missing: Step 9 gaps.
        lyrics_json_path: If set, writes translated lyrics JSON to this path.

    Returns:
        {
            "final_wav": str,
            "lyrics": list[dict],  # line, translation, start, end per row
            "source_wav": str,
            "video_id": str,
            "lyrics_json": str | None,  # path if written
        }
    """
    audio_dir = Path(audio_dir).resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)

    source_wav = download_youtube_audio(youtube_url, output_dir=audio_dir)
    video_id = Path(source_wav).stem

    words = transcribe_words(
        source_wav,
        model_size=whisper_model,
        language=whisper_language,
        device=whisper_device,
    )
    lines = group_words_into_lines(words, gap_threshold_sec=line_gap_sec)
    if not lines:
        raise RuntimeError("No lyric lines produced (empty transcription or grouping).")

    client = openai_client
    if client is None and os.environ.get("OPENAI_API_KEY"):
        client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)

    bilingual = translate_lyric_lines(
        lines,
        model=translation_model,
        chunk_size=translation_chunk_size,
        client=client,
        match_syllables=match_syllables,
        syllable_rewrite_attempts=syllable_rewrite_attempts,
    )

    tts_paths = generate_english_audio(
        bilingual,
        output_dir=audio_dir,
        file_stem=video_id,
        voice=tts_voice,
        model=tts_model,
        client=client,
    )

    timed_paths = match_timing_for_generated_clips(
        bilingual,
        tts_paths,
        output_dir=audio_dir,
        file_stem=video_id,
        output_suffix=timed_suffix,
    )

    out_wav = Path(final_wav).resolve() if final_wav else audio_dir / f"{video_id}_final.wav"
    final_path = combine_adjusted_clips_to_wav(
        timed_paths,
        out_wav,
        silence_between_sec=silence_between_clips,
        silence_for_missing_sec=silence_for_missing,
    )

    lyrics_json_written: str | None = None
    if lyrics_json_path is not None:
        lp = Path(lyrics_json_path).resolve()
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(
            json.dumps(bilingual, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        lyrics_json_written = str(lp)

    return {
        "final_wav": final_path,
        "lyrics": bilingual,
        "source_wav": str(Path(source_wav).resolve()),
        "video_id": video_id,
        "lyrics_json": lyrics_json_written,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="YouTube → English cover audio pipeline")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--audio-dir",
        default="output/audio",
        help="Directory for downloads, clips, and final WAV",
    )
    parser.add_argument("--whisper-model", default="base", help="Whisper model size")
    parser.add_argument(
        "--whisper-lang",
        default="es",
        help="Whisper language (use 'auto' for auto-detect)",
    )
    parser.add_argument("--translation-model", default=DEFAULT_MODEL)
    parser.add_argument("--tts-voice", default=DEFAULT_VOICE)
    parser.add_argument(
        "--lyrics-out",
        default=None,
        help="Optional path to write translated lyrics JSON",
    )
    parser.add_argument(
        "--no-syllables",
        action="store_true",
        help="Disable syllable matching on translations",
    )
    args = parser.parse_args()

    wl = None if args.whisper_lang.lower() == "auto" else args.whisper_lang

    result = run_pipeline(
        args.url,
        audio_dir=args.audio_dir,
        whisper_model=args.whisper_model,
        whisper_language=wl,
        lyrics_json_path=args.lyrics_out,
        match_syllables=not args.no_syllables,
    )

    print(
        json.dumps(
            {
                "final_wav": result["final_wav"],
                "video_id": result["video_id"],
                "line_count": len(result["lyrics"]),
                "lyrics_json": result["lyrics_json"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
