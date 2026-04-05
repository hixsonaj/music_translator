"""
Post-Translation Audio Pipeline
---------------------------------
Takes translated lyrics + Whisper segments and produces a final
remixed audio file with the translated vocals over the instrumental.

Pipeline:
  1. Clone the singer's voice from the separated vocal track
  2. Generate TTS for each translated line in the cloned voice
  3. (Optional) Global per-segment pitch correction
  4. Time-stretch each segment to match original duration
  5. Reassemble all segments into a full vocal track
  6. Mix translated vocals over the instrumental

Requirements:
  pip install elevenlabs pydub librosa soundfile numpy pyrubberband

Usage:
  from audio_pipeline import AudioPipeline

  pipeline = AudioPipeline(elevenlabs_api_key="sk_...")
  pipeline.run(
      vocal_path="vocals.mp3",
      instrumental_path="instrumental.mp3",
      segments=[
          {"start": 0.0, "end": 2.1, "translation": "Nunca te voy a soltar"},
          {"start": 2.1, "end": 4.2, "translation": "Nunca te voy a decepcionar"},
      ],
      output_path="final_output.mp3",
  )
"""

import os
import shutil
import tempfile
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv(".env/keys.env")
from elevenlabs import VoiceSettings
from pydub import AudioSegment
import librosa
import soundfile as sf
import pyrubberband as pyrb
import whisper


class AudioPipeline:

    def __init__(self, elevenlabs_api_key: str):
        self.client = ElevenLabs(api_key=elevenlabs_api_key)
        self.voice_id = None
        self._tts_cache = {}  # text -> file path — avoids duplicate API calls

    # ── Voice cloning ────────────────────────────────────────────────────────

    def clone_voice(self, vocal_path: str, singer_name: str = "singer", tmpdir: str = None) -> str:
        """Clone a single voice from the vocal track."""
        print(f"Cloning voice from {vocal_path}...")

        # Convert to mp3 for ElevenLabs if needed
        if vocal_path.endswith(".wav"):
            mp3_path = os.path.join(tmpdir, "vocal_sample.mp3")
            AudioSegment.from_wav(vocal_path).export(mp3_path, format="mp3")
        else:
            mp3_path = vocal_path

        with open(mp3_path, "rb") as f:
            voice = self.client.voices.ivc.create(
                name=singer_name,
                description=f"Singer voice ({singer_name})",
                files=[("vocals.mp3", f, "audio/mpeg")],
            )
        self.voice_id = voice.voice_id
        print(f"  Cloned '{singer_name}': {self.voice_id}")
        return self.voice_id

    # ── Transcription ────────────────────────────────────────────────────────

    def transcribe_vocals(self, vocal_path: str, model_size: str = "medium") -> list[dict]:
        """Transcribe the vocal track using OpenAI Whisper."""
        print(f"Transcribing vocals with Whisper ({model_size})...")

        model = whisper.load_model(model_size)
        result = model.transcribe(
            vocal_path,
            word_timestamps=True,
            verbose=False,
        )

        segments = []
        for seg in result["segments"]:
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            })
            print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text'].strip()}")

        print(f"Transcribed {len(segments)} segments")
        return segments

    # ── TTS generation ───────────────────────────────────────────────────────

    def generate_tts_segment(self, text: str, output_path: str) -> str:
        """
        Generate speech for a translated line using the cloned voice.
        Results are cached by text so duplicate lyrics reuse the same audio.
        """
        if not self.voice_id:
            raise ValueError("Must clone voice first — call clone_voice()")

        # Return cached version if available
        if text in self._tts_cache:
            cached_path = self._tts_cache[text]
            print(f"  TTS cache hit: '{text}'")
            shutil.copy(cached_path, output_path)
            return output_path

        print(f"  Generating TTS: '{text}'")

        audio = self.client.text_to_speech.convert(
            text=text,
            voice_id=self.voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            voice_settings=VoiceSettings(
                stability=0.7,
                similarity_boost=0.8,
                style=0.0,
                use_speaker_boost=True,
            )
        )

        with open(output_path, "wb") as f:
            for chunk in audio:
                if isinstance(chunk, bytes):
                    f.write(chunk)

        self._tts_cache[text] = output_path
        return output_path

    # ── Global pitch correction (optional) ───────────────────────────────────

    def pitch_correct_segment(
        self,
        tts_path: str,
        vocal_path: str,
        seg_start: float,
        seg_end: float,
        output_path: str,
        max_shift_semitones: float = 2.0,
    ) -> str:
        """
        Apply a single global pitch shift to align the TTS median F0
        with the original vocal's median F0 for this segment.

        Shift is clamped to ±max_shift_semitones to avoid artifacts.
        """
        y_tts, sr = librosa.load(tts_path, sr=None)
        y_orig, _ = librosa.load(
            vocal_path, sr=sr, offset=seg_start, duration=seg_end - seg_start,
        )

        if len(y_orig) == 0 or len(y_tts) == 0:
            shutil.copy(tts_path, output_path)
            return output_path

        # Extract median F0 from both
        orig_f0, _, _ = librosa.pyin(y_orig, fmin=80, fmax=600, sr=sr)
        tts_f0, _, _ = librosa.pyin(y_tts, fmin=80, fmax=600, sr=sr)

        orig_voiced = orig_f0[orig_f0 > 0]
        tts_voiced = tts_f0[tts_f0 > 0]

        if len(orig_voiced) == 0 or len(tts_voiced) == 0:
            shutil.copy(tts_path, output_path)
            return output_path

        shift = 12 * np.log2(np.median(orig_voiced) / np.median(tts_voiced))
        shift = float(np.clip(shift, -max_shift_semitones, max_shift_semitones))

        if abs(shift) < 0.1:
            print(f"  Pitch: no correction needed ({shift:+.1f} st)")
            shutil.copy(tts_path, output_path)
            return output_path

        print(f"  Pitch: {shift:+.1f} semitones (orig {np.median(orig_voiced):.0f}Hz, tts {np.median(tts_voiced):.0f}Hz)")
        y_shifted = pyrb.pitch_shift(y_tts, sr, shift)
        sf.write(output_path, y_shifted, sr)
        return output_path

    # ── Time stretching ──────────────────────────────────────────────────────

    def time_stretch_segment(
        self,
        input_path: str,
        output_path: str,
        target_duration_seconds: float,
    ) -> str:
        """
        Time-stretch or compress the audio to match the target duration.
        Uses pyrubberband for higher quality than librosa's phase vocoder.
        """
        y, sr = librosa.load(input_path, sr=None)
        current_duration = len(y) / sr

        if abs(current_duration - target_duration_seconds) < 0.05:
            shutil.copy(input_path, output_path)
            return output_path

        rate = current_duration / target_duration_seconds
        rate = max(0.5, min(2.0, rate))

        print(f"  Stretching: {current_duration:.2f}s → {target_duration_seconds:.2f}s (rate={rate:.2f})")

        y_stretched = pyrb.time_stretch(y, sr, rate)

        # Trim or pad to exact target length
        target_samples = int(target_duration_seconds * sr)
        if len(y_stretched) > target_samples:
            y_stretched = y_stretched[:target_samples]
        elif len(y_stretched) < target_samples:
            y_stretched = np.pad(y_stretched, (0, target_samples - len(y_stretched)))

        sf.write(output_path, y_stretched, sr)
        return output_path

    # ── Assembly ─────────────────────────────────────────────────────────────

    def assemble_vocal_track(
        self,
        segments: list[dict],
        segment_audio_paths: list[str],
        total_duration_seconds: float,
        output_path: str,
    ) -> str:
        """
        Place each generated segment at the correct timestamp to build
        the full translated vocal track.
        """
        print("Assembling vocal track...")

        total_ms = int(total_duration_seconds * 1000)
        vocal_track = AudioSegment.silent(duration=total_ms)

        for i, (seg, audio_path) in enumerate(zip(segments, segment_audio_paths)):
            if not os.path.exists(audio_path):
                print(f"  Warning: segment {i} audio not found, skipping")
                continue

            segment_audio = AudioSegment.from_file(audio_path)
            start_ms = int(seg["start"] * 1000)

            vocal_track = vocal_track.overlay(segment_audio, position=start_ms)
            original = seg.get("text", "")
            print(f"  Placed segment {i} at {seg['start']:.2f}s: '{original}' -> '{seg['translation']}'")

        vocal_track.export(output_path, format="mp3")
        print(f"Vocal track assembled: {output_path}")
        return output_path

    # ── Mixing ───────────────────────────────────────────────────────────────

    def mix_with_instrumental(
        self,
        vocal_path: str,
        instrumental_path: str,
        output_path: str,
        vocal_volume_db: float = 3.0,
        instrumental_volume_db: float = -3.0,
    ) -> str:
        """Layer the translated vocal track over the instrumental."""
        print("Mixing vocals with instrumental...")

        vocals = AudioSegment.from_file(vocal_path)
        instrumental = AudioSegment.from_file(instrumental_path)

        vocals = vocals + vocal_volume_db
        instrumental = instrumental + instrumental_volume_db

        if len(vocals) < len(instrumental):
            vocals = vocals + AudioSegment.silent(duration=len(instrumental) - len(vocals))
        elif len(instrumental) < len(vocals):
            instrumental = instrumental + AudioSegment.silent(duration=len(vocals) - len(instrumental))

        final = instrumental.overlay(vocals)
        final.export(output_path, format="mp3")
        print(f"Final mix exported: {output_path}")
        return output_path

    # ── Full pipeline ────────────────────────────────────────────────────────

    def run(
        self,
        vocal_path: str,
        instrumental_path: str,
        segments: list[dict] = None,
        output_path: str = "translated_song.mp3",
        singer_name: str = "singer",
        whisper_model: str = "medium",
        voice_id: str = None,
        pitch_correction: bool = True,
        max_pitch_shift: float = 2.0,
    ) -> str:
        """
        Run the full post-translation pipeline.

        Pipeline: TTS → (optional pitch shift) → time stretch → assemble → mix

        Args:
            vocal_path: path to separated vocals (from demucs)
            instrumental_path: path to separated instrumental
            segments: list of {start, end, translation} dicts
            output_path: where to write the final mp3
            singer_name: name for the cloned voice
            whisper_model: whisper model size for transcription
            voice_id: pre-cloned ElevenLabs voice ID (skips cloning)
            pitch_correction: whether to apply global pitch correction
            max_pitch_shift: max pitch shift in semitones (default ±2)
        """
        with tempfile.TemporaryDirectory() as tmpdir:

            # Step 0: Transcribe if no segments provided
            if segments is None:
                transcription = self.transcribe_vocals(vocal_path, model_size=whisper_model)
                print("\nTranscription complete. Segments need translation before proceeding.")
                print("Pass translated segments with 'translation' key to run().")
                return transcription

            # Step 1: Use pre-cloned voice or clone a new one
            if voice_id:
                self.voice_id = voice_id
                print(f"Using pre-cloned voice: {self.voice_id}")
            else:
                self.clone_voice(vocal_path, singer_name, tmpdir)

            # Step 2: Generate TTS, pitch correct, time-stretch each segment
            segment_paths = []
            for i, seg in enumerate(segments):
                tts_path = os.path.join(tmpdir, f"seg_{i}_tts.mp3")
                pitch_path = os.path.join(tmpdir, f"seg_{i}_pitch.wav")
                stretched_path = os.path.join(tmpdir, f"seg_{i}_stretched.wav")
                target_duration = seg["end"] - seg["start"]

                # Generate TTS
                self.generate_tts_segment(seg["translation"], tts_path)

                # Optional global pitch correction
                if pitch_correction:
                    self.pitch_correct_segment(
                        tts_path, vocal_path,
                        seg["start"], seg["end"],
                        pitch_path,
                        max_shift_semitones=max_pitch_shift,
                    )
                else:
                    shutil.copy(tts_path, pitch_path)

                # Time-stretch to match original duration
                self.time_stretch_segment(pitch_path, stretched_path, target_duration)
                segment_paths.append(stretched_path)

            # Step 3: Assemble vocal track
            total_duration = max(seg["end"] for seg in segments) + 1.0
            vocal_track_path = os.path.join(tmpdir, "translated_vocals.mp3")
            self.assemble_vocal_track(
                segments, segment_paths, total_duration, vocal_track_path
            )

            # Step 4: Mix with instrumental
            self.mix_with_instrumental(
                vocal_track_path, instrumental_path, output_path,
            )

        return output_path


# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import re
    import anthropic

    claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    pipeline = AudioPipeline(
        elevenlabs_api_key=os.environ.get("ELEVENLABS_API_KEY", "sk_...")
    )

    segments = [
        {"start": 0.0, "end": 5.1, "text": "You wouldn't get this from any other guy."},
        {"start": 5.6, "end": 9.8, "text": "I just want to tell you how I'm feeling."},
        {"start": 10.4, "end": 12.8, "text": "Gotta make you understand."},
        {"start": 13.2, "end": 15.1, "text": "Never gonna give you up."},
        {"start": 15.4, "end": 17.1, "text": "Never gonna let you down."},
        {"start": 17.4, "end": 21.2, "text": "Never gonna run around and desert you."},
        {"start": 21.5, "end": 23.4, "text": "Never gonna make you cry."},
        {"start": 23.7, "end": 25.4, "text": "Never gonna say goodbye."},
        {"start": 25.8, "end": 29.0, "text": "Never gonna tell a lie and hurt you."},
        {"start": 30.3, "end": 34.7, "text": "We've known each other for so long."},
        {"start": 35.2, "end": 39.4, "text": "Your heart's been aching but you're too shy to say it."},
        {"start": 39.4, "end": 42.4, "text": "Inside we both know what's been going on."},
        {"start": 43.6, "end": 47.4, "text": "We know the game and we're gonna play it."},
        {"start": 48.4, "end": 51.9, "text": "And if you ask me how I'm feeling,"},
        {"start": 52.3, "end": 55.1, "text": "Don't tell me you're too blind to see."},
        {"start": 55.4, "end": 57.3, "text": "Never gonna give you up."},
        {"start": 57.3, "end": 59.4, "text": "Never gonna let you down."},
        {"start": 59.7, "end": 63.4, "text": "Never gonna run around and desert you."},
        {"start": 63.8, "end": 65.7, "text": "Never gonna make you cry."},
        {"start": 66.0, "end": 67.7, "text": "Never gonna say goodbye."},
        {"start": 68.1, "end": 72.0, "text": "Never gonna tell a lie and hurt you."},
        {"start": 72.3, "end": 78.4, "text": "Never gonna make you cry."},
        {"start": 82.9, "end": 84.6, "text": "Never gonna say goodbye."},
    ]

    # ── Translate via Claude directly ─────────────────────────────────────
    target_lang = "es"
    text_cache = {}  # text -> translation (reuse for duplicate lyrics)

    # Deduplicate
    unique_texts = list({seg["text"] for seg in segments})
    total_segments = len(segments)
    print(f"{total_segments} segments, {len(unique_texts)} unique lyrics to translate "
          f"({total_segments - len(unique_texts)} duplicates skipped)")

    def translate_line(text: str, lang: str = "Spanish") -> str:
        syllable_count = max(1, len(re.findall(r'[aeiouAEIOU]+', text)))
        response = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            messages=[{"role": "user", "content": (
                f"Translate this English song lyric into {lang}.\n"
                f"Original: \"{text}\"\n"
                f"Match the syllable count as closely as possible (~{syllable_count} syllables).\n"
                f"Return ONLY the translated line, no explanation."
            )}],
        )
        return response.content[0].text.strip().strip('"')

    print(f"Translating {len(unique_texts)} unique lines to {target_lang}...")
    for text in unique_texts:
        translation = translate_line(text)
        text_cache[text] = translation
        print(f"  '{text}' -> '{translation}'")

    # Apply translations to all segments
    for seg in segments:
        seg["translation"] = text_cache[seg["text"]]

    # ── Run pipeline (will clone a fresh voice) ────────────────────────
    result = pipeline.run(
        vocal_path="separated/htdemucs/test_song/vocals_short.wav",
        instrumental_path="separated/htdemucs/test_song/no_vocals_short.wav",
        segments=segments,
        output_path="translated_song.mp3",
        pitch_correction=True,
        max_pitch_shift=2.0,
    )

    print(f"\nDone! Output: {result}")