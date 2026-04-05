"""
Translation Glue + Audio Pipeline Runner
-----------------------------------------
Reads segments.json, sends each line through the translation agents,
collects the results, then runs audio_pipeline.py to produce the final song.

Usage:
  # Make sure these are running first:
  #   python orchestrator_asi.py
  #   python translation_agent.py
  #   python syllable_critic.py
  #   python stress_critic.py  (optional)

  python translate_and_run.py \
      --segments segments.json \
      --vocals separated/htdemucs/source/vocals.wav \
      --instrumental separated/htdemucs/source/no_vocals.wav \
      --target-lang es \
      --output translated_song.mp3
"""

import os
import json
import argparse
from dotenv import load_dotenv
from uagents import Agent, Context

load_dotenv("keys.env")

from models import LyricLine, AcceptedLine
from audio_pipeline import AudioPipeline

ORCHESTRATOR_ADDR = os.getenv(
    "ORCHESTRATOR_ADDR",
    "agent1qga4t2sj75u7kxrw8gpj3jpx0wg5dch0jc06elsfmc54mrjsdntyukzvkgf",
)

runner = Agent(
    name="translate_runner",
    seed="translate_runner_seed_phrase",
    port=8010,
    endpoint=["http://localhost:8010/submit"],
    mailbox=False,
)

# Shared state between handlers
_state = {
    "segments": [],         # original segments from JSON
    "target_lang": "es",
    "pending": set(),       # line_ids still waiting
    "results": {},          # line_id -> translation string
    "vocals_path": "",
    "instrumental_path": "",
    "output_path": "",
    "elevenlabs_api_key": "",
    "singer_name": "singer",
}


@runner.on_event("startup")
async def send_all(ctx: Context):
    segments = _state["segments"]
    target_lang = _state["target_lang"]

    ctx.logger.info(f"Sending {len(segments)} lines to orchestrator ({target_lang})")

    for i, seg in enumerate(segments):
        _state["pending"].add(i)
        await ctx.send(ORCHESTRATOR_ADDR, LyricLine(
            line_id=i,
            original=seg["text"],
            translated="",
            target_lang=target_lang,
            target_syllables=seg["syllables"],
        ))

    ctx.logger.info("All lines sent — waiting for translations...")


@runner.on_message(model=AcceptedLine)
async def handle_result(ctx: Context, sender: str, msg: AcceptedLine):
    _state["results"][msg.line_id] = msg.final_translation
    _state["pending"].discard(msg.line_id)

    seg = _state["segments"][msg.line_id]
    ctx.logger.info(
        f"[{msg.line_id + 1}/{len(_state['segments'])}] "
        f"'{seg['text']}' → '{msg.final_translation}' "
        f"({msg.attempts} attempt{'s' if msg.attempts > 1 else ''})"
    )

    if not _state["pending"]:
        ctx.logger.info("All translations received — running audio pipeline...")
        _run_audio_pipeline(ctx)


def _run_audio_pipeline(ctx):
    segments = _state["segments"]
    results = _state["results"]

    # Build final segments list for audio_pipeline.py
    translated_segments = []
    for i, seg in enumerate(segments):
        translated_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "translation": results.get(i, seg["text"]),  # fallback to original if missing
        })

    # Save translated segments for reference
    with open("segments_translated.json", "w") as f:
        json.dump(translated_segments, f, indent=2, ensure_ascii=False)
    ctx.logger.info("Saved segments_translated.json")

    # Run audio pipeline
    pipeline = AudioPipeline(elevenlabs_api_key=_state["elevenlabs_api_key"])
    pipeline.run(
        vocal_path=_state["vocals_path"],
        instrumental_path=_state["instrumental_path"],
        segments=translated_segments,
        output_path=_state["output_path"],
        singer_name=_state["singer_name"],
    )

    ctx.logger.info(f"Done! Output: {_state['output_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate segments and produce final audio")
    parser.add_argument("--segments",       default="segments.json",   help="Path to segments.json from pre_pipeline.py")
    parser.add_argument("--vocals",         required=True,             help="Path to vocals.wav from demucs")
    parser.add_argument("--instrumental",   required=True,             help="Path to no_vocals.wav from demucs")
    parser.add_argument("--target-lang",    default="es",              help="Target language code (default: es)")
    parser.add_argument("--output",         default="translated_song.mp3", help="Output file path")
    parser.add_argument("--elevenlabs-key", default=os.getenv("ELEVENLABS_API_KEY", ""), help="ElevenLabs API key")
    parser.add_argument("--singer-name",    default="singer",              help="Singer name for voice clone (default: singer)")
    args = parser.parse_args()

    with open(args.segments) as f:
        segments = json.load(f)

    _state["segments"] = segments
    _state["target_lang"] = args.target_lang
    _state["vocals_path"] = args.vocals
    _state["instrumental_path"] = args.instrumental
    _state["output_path"] = args.output
    _state["elevenlabs_api_key"] = args.elevenlabs_key
    _state["singer_name"] = args.singer_name

    print(f"Loaded {len(segments)} segments from {args.segments}")
    print(f"Target language: {args.target_lang}")
    print(f"Output: {args.output}")
    print()

    runner.run()
