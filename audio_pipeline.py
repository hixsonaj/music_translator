"""
Post-Translation Audio Pipeline
---------------------------------
Takes translated lyrics + Whisper segments and produces a final
remixed audio file with the translated vocals over the instrumental.

Pipeline:
  1. Clone the singer's voice from the separated vocal track
  2. Generate TTS for each translated line in the cloned voice
  3. Time-stretch each generated segment to match original duration
  4. Transfer the original segment's pitch contour onto the generated audio
  5. Reassemble all segments into a full vocal track
  6. Mix translated vocals over the instrumental

Requirements:
  pip install elevenlabs pydub librosa soundfile numpy pyworld

Usage:
  from audio_pipeline import AudioPipeline

  pipeline = AudioPipeline(elevenlabs_api_key="sk_...")
  pipeline.run(
      vocal_path="vocals.mp3",          # from demucs
      instrumental_path="instrumental.mp3",  # from demucs
      segments=[                         # from whisper + translation agent
          {"start": 0.0, "end": 2.1, "translation": "Nunca te voy a soltar"},
          {"start": 2.1, "end": 4.2, "translation": "Nunca te voy a decepcionar"},
      ],
      output_path="final_output.mp3",
  )
"""

import os
import tempfile
import numpy as np
from pathlib import Path
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment
import librosa
import soundfile as sf
import pyworld as pw


class AudioPipeline:

    def __init__(self, elevenlabs_api_key: str):
        self.client = ElevenLabs(api_key=elevenlabs_api_key)
        self.voice_id = None  # Set after cloning

    def delete_voice(self):
        """Delete the cloned voice from ElevenLabs to free up voice slots."""
        if not self.voice_id:
            print("No voice to delete.")
            return
        
        self.client.voices.delete(self.voice_id)
        print(f"Deleted voice: {self.voice_id}")
        self.voice_id = None

    # ── Step 1: Clone voice from vocal track ─────────────────────────────────

    def clone_voice(self, vocal_path: str, singer_name: str = "singer") -> str:
        print(f"Cloning voice from {vocal_path}...")

        # Convert to mp3 if needed — ElevenLabs prefers mp3
        if vocal_path.endswith(".wav"):
            mp3_path = vocal_path.replace(".wav", ".mp3")
            audio = AudioSegment.from_wav(vocal_path)
            # Trim to 60 seconds max — ElevenLabs only needs a short sample
            audio = audio[:60000]
            audio.export(mp3_path, format="mp3")
            vocal_path = mp3_path
            print(f"  Converted to mp3: {mp3_path}")

        with open(vocal_path, "rb") as f:
            voice = self.client.voices.ivc.create(
              name=singer_name,
              description="Singer voice for lyric translation",
              files=[("vocals.mp3", f, "audio/mpeg")],
        )

        self.voice_id = voice.voice_id
        print(f"Voice cloned successfully: {self.voice_id}")
        return self.voice_id

    # ── Step 2: Generate TTS for a single translated line ────────────────────

    def generate_tts_segment(self, text: str, output_path: str) -> str:
        """
        Generate speech for a single translated line using the cloned voice.
        Uses eleven_multilingual_v2 which handles both English and Spanish.
        """
        if not self.voice_id:
            raise ValueError("Must clone voice first — call clone_voice()")

        print(f"  Generating TTS: '{text}'")

        audio = self.client.text_to_speech.convert(
            text=text,
            voice_id=self.voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )

        # Write bytes to file
        with open(output_path, "wb") as f:
            for chunk in audio:
                if isinstance(chunk, bytes):
                    f.write(chunk)

        return output_path

    # ── Step 3: Time-stretch to match original segment duration ──────────────

    def time_stretch_segment(
        self,
        input_path: str,
        output_path: str,
        target_duration_seconds: float,
    ) -> str:
        """
        Stretch or compress the generated audio to exactly fit
        the original segment's duration from Whisper timestamps.

        Uses librosa's phase vocoder for time stretching without
        changing pitch (so the voice still sounds natural).
        """
        # Load the generated TTS audio
        y, sr = librosa.load(input_path, sr=None)
        current_duration = len(y) / sr

        if abs(current_duration - target_duration_seconds) < 0.05:
            # Already close enough, no need to stretch
            import shutil
            shutil.copy(input_path, output_path)
            return output_path

        # Calculate stretch rate
        # rate > 1 = faster (compress), rate < 1 = slower (stretch)
        rate = current_duration / target_duration_seconds
        rate = max(0.5, min(2.0, rate))  # Clamp to reasonable range

        print(f"  Stretching: {current_duration:.2f}s → {target_duration_seconds:.2f}s (rate={rate:.2f})")

        # Time stretch using phase vocoder
        y_stretched = librosa.effects.time_stretch(y, rate=rate)

        # Trim or pad to exact target length
        target_samples = int(target_duration_seconds * sr)
        if len(y_stretched) > target_samples:
            y_stretched = y_stretched[:target_samples]
        else:
            y_stretched = np.pad(y_stretched, (0, target_samples - len(y_stretched)))

        sf.write(output_path, y_stretched, sr)
        return output_path

    # ── Step 4: Transfer pitch contour from original segment ─────────────────

    def pitch_transfer(
        self,
        original_segment_path: str,
        generated_path: str,
        output_path: str,
    ) -> str:
        """
        Apply the pitch contour of the original vocal segment onto the
        generated (time-stretched) TTS audio.

        The generated audio keeps its own timbre and spectral character
        (i.e. the cloned voice), but now follows the melody the original
        singer was hitting note-for-note.

        Uses WORLD vocoder (pyworld):
          - Decompose generated audio into F0, spectral envelope, aperiodicity
          - Extract F0 from original segment
          - Resample original F0 to match the generated audio's frame count
          - Resynthesize with the transferred F0
        """
        # Load both audio files at their native sample rates
        y_orig, sr_orig = librosa.load(original_segment_path, sr=None, mono=True)
        y_gen,  sr_gen  = librosa.load(generated_path,        sr=None, mono=True)

        # WORLD requires float64
        y_orig_64 = y_orig.astype(np.float64)
        y_gen_64  = y_gen.astype(np.float64)

        # Decompose the generated audio with WORLD
        f0_gen, sp_gen, ap_gen = pw.wav2world(y_gen_64, sr_gen)

        # Extract F0 from the original segment using pyin (more accurate on singing)
        f0_orig_raw, voiced_flag, _ = librosa.pyin(
            y_orig_64,
            fmin=librosa.note_to_hz("C2"),   # ~65 Hz — covers bass singers
            fmax=librosa.note_to_hz("C6"),   # ~1047 Hz — covers soprano
            sr=sr_orig,
            frame_length=2048,
        )

        # Replace NaNs (unvoiced frames) with 0 — WORLD treats 0 as unvoiced
        f0_orig_raw = np.nan_to_num(f0_orig_raw, nan=0.0)

        # Resample original F0 to match the number of WORLD frames in the generated audio
        n_frames_gen = len(f0_gen)
        n_frames_orig = len(f0_orig_raw)

        if n_frames_orig != n_frames_gen:
            indices = np.linspace(0, n_frames_orig - 1, n_frames_gen)
            f0_transferred = np.interp(indices, np.arange(n_frames_orig), f0_orig_raw)
        else:
            f0_transferred = f0_orig_raw.copy()

        # Where original is unvoiced (0), keep generated F0 so the voice
        # doesn't go silent on spoken/breathy parts of the translation
        unvoiced = f0_transferred == 0.0
        f0_transferred[unvoiced] = f0_gen[unvoiced]

        # Resynthesize with transferred pitch
        y_out = pw.synthesize(f0_transferred, sp_gen, ap_gen, sr_gen)
        y_out = y_out.astype(np.float32)

        sf.write(output_path, y_out, sr_gen)
        print(f"  Pitch transferred: {Path(generated_path).name} → {Path(output_path).name}")
        return output_path

    # ── Step 5: Assemble all segments into a full vocal track ────────────────

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

        Gaps between segments (silence between lyrics) are preserved
        so the track stays in sync with the instrumental.
        """
        print("Assembling vocal track...")

        # Create a silent base track of the full song duration
        total_ms = int(total_duration_seconds * 1000)
        vocal_track = AudioSegment.silent(duration=total_ms)

        for i, (seg, audio_path) in enumerate(zip(segments, segment_audio_paths)):
            if not os.path.exists(audio_path):
                print(f"  Warning: segment {i} audio not found, skipping")
                continue

            segment_audio = AudioSegment.from_file(audio_path)
            start_ms = int(seg["start"] * 1000)

            # Overlay the segment at its correct timestamp
            vocal_track = vocal_track.overlay(segment_audio, position=start_ms)
            print(f"  Placed segment {i} at {seg['start']:.2f}s: '{seg['translation']}'")

        vocal_track.export(output_path, format="mp3")
        print(f"Vocal track assembled: {output_path}")
        return output_path

    # ── Step 5: Mix vocals over instrumental ─────────────────────────────────

    def mix_with_instrumental(
        self,
        vocal_path: str,
        instrumental_path: str,
        output_path: str,
        vocal_volume_db: float = 0.0,
        instrumental_volume_db: float = -3.0,
    ) -> str:
        """
        Layer the translated vocal track over the instrumental.

        vocal_volume_db: adjust vocal volume (0 = no change, -3 = quieter)
        instrumental_volume_db: adjust instrumental volume (-3 recommended
                                 to give vocals some space in the mix)
        """
        print("Mixing vocals with instrumental...")

        vocals = AudioSegment.from_file(vocal_path)
        instrumental = AudioSegment.from_file(instrumental_path)

        self.delete_voice()

        # Adjust volumes
        vocals = vocals + vocal_volume_db
        instrumental = instrumental + instrumental_volume_db

        # Match lengths (pad shorter one with silence)
        if len(vocals) < len(instrumental):
            vocals = vocals + AudioSegment.silent(duration=len(instrumental) - len(vocals))
        elif len(instrumental) < len(vocals):
            instrumental = instrumental + AudioSegment.silent(duration=len(vocals) - len(instrumental))

        # Mix
        final = instrumental.overlay(vocals)
        final.export(output_path, format="mp3")
        print(f"Final mix exported: {output_path}")
        return output_path

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def run(
        self,
        vocal_path: str,
        instrumental_path: str,
        segments: list[dict],
        output_path: str,
        singer_name: str = "singer",
    ) -> str:
        """
        Run the full post-translation pipeline.

        segments format:
          [
            {"start": 0.0, "end": 2.1, "translation": "Nunca te voy a soltar"},
            {"start": 2.1, "end": 4.2, "translation": "Nunca te voy a decepcionar"},
            ...
          ]
        """
        with tempfile.TemporaryDirectory() as tmpdir:

            # Step 1: Clone voice
            self.clone_voice(vocal_path, singer_name)

            # Pre-load the full vocal track once for slicing original segments
            full_vocals = AudioSegment.from_file(vocal_path)

            # Step 2, 3 & 4: Generate, time-stretch, and pitch-transfer each segment
            segment_paths = []
            for i, seg in enumerate(segments):
                raw_path          = os.path.join(tmpdir, f"seg_{i}_raw.mp3")
                stretched_path    = os.path.join(tmpdir, f"seg_{i}_stretched.wav")
                orig_seg_path     = os.path.join(tmpdir, f"seg_{i}_original.wav")
                pitched_path      = os.path.join(tmpdir, f"seg_{i}_pitched.wav")
                target_duration   = seg["end"] - seg["start"]

                # Slice the original vocal segment for pitch reference
                start_ms = int(seg["start"] * 1000)
                end_ms   = int(seg["end"]   * 1000)
                full_vocals[start_ms:end_ms].export(orig_seg_path, format="wav")

                self.generate_tts_segment(seg["translation"], raw_path)
                self.time_stretch_segment(raw_path, stretched_path, target_duration)
                self.pitch_transfer(orig_seg_path, stretched_path, pitched_path)
                segment_paths.append(pitched_path)

            # Step 5: Assemble vocal track
            # Get total duration from the last segment end time
            total_duration = max(seg["end"] for seg in segments) + 1.0
            vocal_track_path = os.path.join(tmpdir, "translated_vocals.mp3")
            self.assemble_vocal_track(
                segments, segment_paths, total_duration, vocal_track_path
            )

            # Step 6: Mix with instrumental
            self.mix_with_instrumental(vocal_track_path, instrumental_path, output_path, vocal_volume_db=10.0, instrumental_volume_db=-18.0)

        return output_path


# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example: run the full pipeline on a test song
    pipeline = AudioPipeline(
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY")
    )

    # These come from demucs (vocal separation) + whisper + your translation agents
    result = pipeline.run(
        vocal_path="separated/htdemucs/test_song/vocals.wav",
        instrumental_path="separated/htdemucs/test_song/no_vocals.wav",
        segments=[
            {"start": 0.5, "end": 2.6, "translation": "Nunca te voy a soltar"},
            {"start": 2.6, "end": 4.7, "translation": "Nunca te voy a decepcionar"},
            {"start": 4.7, "end": 7.8, "translation": "Nunca voy a correr y abandonarte"},
            {"start": 7.8, "end": 9.9, "translation": "Nunca te voy a hacer llorar"},
        ],
        output_path="translated_song.mp3",
        singer_name="rick_astley",
    )

    print(f"\nDone! Output: {result}")
