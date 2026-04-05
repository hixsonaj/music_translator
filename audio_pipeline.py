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

    # ── Voice cleanup ────────────────────────────────────────────────────────

    def delete_voice(self) -> None:
        """Delete the cloned voice from ElevenLabs to avoid accumulating voices."""
        if not self.voice_id:
            return
        try:
            self.client.voices.delete(voice_id=self.voice_id)
            print(f"  Deleted cloned voice: {self.voice_id}")
        except Exception as e:
            print(f"  Warning: failed to delete voice {self.voice_id}: {e}")
        finally:
            self.voice_id = None

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
                similarity_boost=1.0,
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

    # ── Pause detection ──────────────────────────────────────────────────────

    def _find_all_gaps(
        self, vocal_path: str, segments: list[dict], top_db: int = 30,
    ) -> tuple[list[float], float]:
        """
        Scan only the voiced portions of the track (within segment
        boundaries) and collect the duration of every silent gap.
        This excludes the long silences *between* lines so they don't
        skew the threshold.

        Returns (all_gap_durations, threshold_sec).
        The threshold is set at  mean + 1.5 * std  of the intra-line
        gap durations, floored at  median * 1.5  — whichever is larger.
        """
        frame_length = 2048
        hop_length = 512
        gaps = []

        for seg in segments:
            duration = seg["end"] - seg["start"]
            if duration < 0.1:
                continue
            y_seg, sr = librosa.load(
                vocal_path, sr=None, offset=seg["start"], duration=duration,
            )
            if len(y_seg) == 0:
                continue

            rms = librosa.feature.rms(y=y_seg, frame_length=frame_length, hop_length=hop_length)[0]
            threshold_amp = np.max(rms) * (10 ** (-top_db / 20.0))
            is_silent = rms < threshold_amp

            in_silence = False
            silence_start = 0
            total_frames = len(rms)
            for i, silent in enumerate(is_silent):
                if silent and not in_silence:
                    in_silence = True
                    silence_start = i
                elif not silent and in_silence:
                    in_silence = False
                    # Skip leading/trailing silence within this segment
                    start_frac = silence_start / total_frames
                    end_frac = i / total_frames
                    if start_frac < 0.05 or end_frac > 0.95:
                        continue
                    dur = (i - silence_start) * hop_length / sr
                    if dur > 0.02:  # ignore sub-20ms noise
                        gaps.append(dur)

        if len(gaps) < 3:
            # Not enough internal gaps — conservative fallback
            return gaps, 0.2

        gaps_arr = np.array(gaps)
        mean_gap = np.mean(gaps_arr)
        std_gap = np.std(gaps_arr)
        median_gap = np.median(gaps_arr)

        # Threshold: gaps significantly longer than normal
        stat_threshold = mean_gap + 1.5 * std_gap
        median_threshold = median_gap * 1.5
        pause_threshold = max(stat_threshold, median_threshold, 0.08)

        return gaps, float(pause_threshold)

    def _detect_pauses(
        self, y: np.ndarray, sr: int, min_pause_sec: float = 0.15, top_db: int = 30,
    ) -> list[dict]:
        """
        Find significant internal pauses in an audio signal.

        Args:
            min_pause_sec: minimum gap duration (in seconds) to be considered
                a pause.  Use _find_all_gaps() on the full track to compute
                this adaptively.

        Returns list of {start_frac, end_frac, duration_sec} — fractional
        positions (0.0–1.0) and absolute duration of each pause.
        """
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        threshold = np.max(rms) * (10 ** (-top_db / 20.0))
        is_silent = rms < threshold
        total_duration = len(y) / sr

        pauses = []
        in_silence = False
        silence_start = 0
        for i, silent in enumerate(is_silent):
            if silent and not in_silence:
                in_silence = True
                silence_start = i
            elif not silent and in_silence:
                in_silence = False
                start_sec = silence_start * hop_length / sr
                end_sec = i * hop_length / sr
                duration = end_sec - start_sec
                if duration >= min_pause_sec:
                    pauses.append({
                        "start_frac": start_sec / total_duration,
                        "end_frac": end_sec / total_duration,
                        "duration_sec": duration,
                    })

        # Exclude leading/trailing silence — only keep internal pauses
        if pauses and pauses[0]["start_frac"] < 0.05:
            pauses.pop(0)
        if pauses and pauses[-1]["end_frac"] > 0.95:
            pauses.pop()

        return pauses

    # ── Time stretching ──────────────────────────────────────────────────────

    def time_stretch_segment(
        self,
        input_path: str,
        output_path: str,
        target_duration_seconds: float,
        pauses: list[dict] = None,
    ) -> str:
        """
        Time-stretch audio to match target duration.  If pauses are provided,
        pause durations are preserved and only speech portions are stretched.
        """
        y, sr = librosa.load(input_path, sr=None)
        current_duration = len(y) / sr

        if abs(current_duration - target_duration_seconds) < 0.05 and not pauses:
            shutil.copy(input_path, output_path)
            return output_path

        if not pauses:
            # Simple uniform stretch (original behavior)
            rate = current_duration / target_duration_seconds
            rate = max(0.5, min(2.0, rate))
            print(f"  Stretching: {current_duration:.2f}s → {target_duration_seconds:.2f}s (rate={rate:.2f})")
            y_stretched = pyrb.time_stretch(y, sr, rate)
        else:
            # Pause-aware: preserve pause durations, stretch only speech
            total_pause_dur = sum(p["duration_sec"] for p in pauses)
            speech_target = target_duration_seconds - total_pause_dur
            total_samples = len(y)

            if speech_target <= 0.1:
                rate = current_duration / target_duration_seconds
                rate = max(0.5, min(2.0, rate))
                y_stretched = pyrb.time_stretch(y, sr, rate)
            else:
                # Split TTS at proportional pause positions
                regions = []
                prev_end = 0
                for p in pauses:
                    pause_pos = int(p["start_frac"] * total_samples)
                    pause_pos = max(prev_end, min(pause_pos, total_samples))
                    if pause_pos > prev_end:
                        regions.append(("speech", y[prev_end:pause_pos]))
                    regions.append(("pause", p["duration_sec"]))
                    prev_end = pause_pos
                if prev_end < total_samples:
                    regions.append(("speech", y[prev_end:]))

                total_speech_dur = sum(len(r[1]) / sr for r in regions if r[0] == "speech")

                if total_speech_dur < 0.05:
                    rate = current_duration / target_duration_seconds
                    rate = max(0.5, min(2.0, rate))
                    y_stretched = pyrb.time_stretch(y, sr, rate)
                else:
                    speech_rate = total_speech_dur / speech_target
                    speech_rate = max(0.5, min(2.0, speech_rate))

                    pause_desc = [f"{p['duration_sec']:.2f}s@{int(p['start_frac']*100)}%" for p in pauses]
                    print(f"  Pause-aware stretch: {len(pauses)} pause(s) [{', '.join(pause_desc)}], "
                          f"speech rate={speech_rate:.2f}")

                    parts = []
                    for rtype, rdata in regions:
                        if rtype == "speech" and len(rdata) > 0:
                            parts.append(pyrb.time_stretch(rdata, sr, speech_rate))
                        elif rtype == "pause":
                            parts.append(np.zeros(int(rdata * sr)))
                    y_stretched = np.concatenate(parts)

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

            # Track whether we cloned a new voice (so we know to clean it up)
            cloned_here = False

            try:
                # Step 1: Use pre-cloned voice or clone a new one
                if voice_id:
                    self.voice_id = voice_id
                    print(f"Using pre-cloned voice: {self.voice_id}")
                else:
                    self.clone_voice(vocal_path, singer_name, tmpdir)
                    cloned_here = True

                # Step 2: Analyze intra-line gaps for adaptive pause threshold
                all_gaps, pause_threshold = self._find_all_gaps(vocal_path, segments)
                if all_gaps:
                    print(f"  Pause analysis: {len(all_gaps)} intra-line gaps, "
                          f"threshold={pause_threshold*1000:.0f}ms "
                          f"(mean={np.mean(all_gaps)*1000:.0f}ms, "
                          f"median={np.median(all_gaps)*1000:.0f}ms, "
                          f"std={np.std(all_gaps)*1000:.0f}ms)")
                else:
                    print(f"  Pause analysis: no intra-line gaps found, using {pause_threshold*1000:.0f}ms fallback")

                # Step 3: Generate TTS, pitch correct, time-stretch each segment
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

                    # Detect significant pauses in the original vocal
                    y_orig_seg, sr_orig = librosa.load(
                        vocal_path, sr=None,
                        offset=seg["start"], duration=target_duration,
                    )
                    pauses = self._detect_pauses(y_orig_seg, sr_orig, min_pause_sec=pause_threshold)
                    if pauses:
                        print(f"  Detected {len(pauses)} internal pause(s) in original")

                    # Time-stretch to match original duration (pause-aware)
                    self.time_stretch_segment(pitch_path, stretched_path, target_duration, pauses=pauses)
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
            finally:
                # Clean up the cloned voice from ElevenLabs
                if cloned_here:
                    self.delete_voice()
                self._tts_cache.clear()

        return output_path


# ── Full end-to-end usage ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import re
    import json
    import argparse
    import anthropic
    from pre_pipeline import run as run_pre_pipeline

    parser = argparse.ArgumentParser(description="Translate a song end-to-end")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url",  help="YouTube or direct audio URL")
    group.add_argument("--file", help="Path to local audio file")
    group.add_argument("--segments-json", help="Skip pre-pipeline, use existing segments.json + separated audio")
    parser.add_argument("--lang", default="es", help="Target language code (default: es)")
    parser.add_argument("--lang-name", default="Spanish", help="Full language name for translation prompt")
    parser.add_argument("--source-lang", default="en", help="Source language code (default: en)")
    parser.add_argument("--whisper-model", default="medium", help="Whisper model size")
    parser.add_argument("--output", default="translated_song.mp3", help="Output path")
    parser.add_argument("--vocals", help="Path to vocals wav (required with --segments-json)")
    parser.add_argument("--instrumental", help="Path to instrumental wav (required with --segments-json)")
    parser.add_argument("--output-dir", default="separated", help="Demucs output directory")
    args = parser.parse_args()

    # ── Step 1: Get vocals, instrumental, and transcribed segments ─────
    if args.segments_json:
        # Use pre-existing segments and audio
        if not args.vocals or not args.instrumental:
            parser.error("--vocals and --instrumental are required with --segments-json")
        with open(args.segments_json) as f:
            segments = json.load(f)
        vocal_path = args.vocals
        instrumental_path = args.instrumental
        print(f"Loaded {len(segments)} segments from {args.segments_json}")
    else:
        # Run full pre-pipeline: download (if URL) -> demucs -> whisper
        pre_result = run_pre_pipeline(
            url=args.url,
            file=args.file,
            output_dir=args.output_dir,
            source_lang=args.source_lang,
            whisper_model=args.whisper_model,
        )
        segments = pre_result["segments"]
        vocal_path = pre_result["vocals_path"]
        instrumental_path = pre_result["instrumental_path"]

    # ── Step 2: Translate via Claude ──────────────────────────────────
    claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    target_lang = args.lang_name
    text_cache = {}  # text -> translation (reuse for duplicate lyrics)

    unique_texts = list({seg["text"] for seg in segments})
    total_segments = len(segments)
    print(f"\n{total_segments} segments, {len(unique_texts)} unique lyrics to translate "
          f"({total_segments - len(unique_texts)} duplicates skipped)")

    def translate_line(text: str, lang: str = target_lang) -> str:
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

    # ── Step 3: Run audio pipeline ────────────────────────────────────
    pipeline = AudioPipeline(
        elevenlabs_api_key=os.environ.get("ELEVENLABS_API_KEY", "sk_...")
    )

    result = pipeline.run(
        vocal_path=vocal_path,
        instrumental_path=instrumental_path,
        segments=segments,
        output_path=args.output,
        pitch_correction=True,
        max_pitch_shift=2.0,
    )

    print(f"\nDone! Output: {result}")