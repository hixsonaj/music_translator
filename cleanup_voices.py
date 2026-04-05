"""
Delete all cloned voices from ElevenLabs.

Run with:
    python cleanup_voices.py
"""

import os
from dotenv import load_dotenv
from pathlib import Path
from elevenlabs.client import ElevenLabs

load_dotenv(Path(__file__).parent / "keys.env")

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))


def delete_all_cloned_voices():
    voices = client.voices.get_all().voices

    cloned = [v for v in voices if v.category == "cloned"]

    if not cloned:
        print("No cloned voices found.")
        return

    print(f"Found {len(cloned)} cloned voice(s):")
    for v in cloned:
        print(f"  - {v.name} ({v.voice_id})")

    confirm = input("\nDelete all? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    for v in cloned:
        client.voices.delete(v.voice_id)
        print(f"  Deleted: {v.name} ({v.voice_id})")

    print("Done.")


if __name__ == "__main__":
    delete_all_cloned_voices()
