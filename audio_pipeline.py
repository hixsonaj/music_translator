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

        # 🔥 Stable cache directory (prevents temp overwrite bugs)
        self.cache_dir = os.path.join(tempfile.gettempdir(), "tts_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self._tts_cache = {}  # text -> cache file path

    # ── Voice cloning ─────────────────────────────────────────────

    def clone_voice(self, vocal_path: str, singer_name: str = "singer", tmpdir: str = None) -> str:
        print(f"Cloning voice from {vocal_path}...")

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

    def delete_voice(self):
        if not self.voice_id:
            return
        try:
            self.client.voices.delete(voice_id=self.voice_id)
            print(f"  Deleted cloned voice: {self.voice_id}")
        except Exception as e:
            print(f"  Warning: failed to delete voice: {e}")
        finally:
            self.voice_id = None

    # ── TTS ───────────────────────────────────────────────────────

    def generate_tts_segment(self, text: str, output_path: str) -> str:
        if not self.voice_id:
            raise ValueError("Must clone voice first")

        cache_path = os.path.join(self.cache_dir, f"{hash(text)}.wav")

        # Cache hit
        if text in self._tts_cache and os.path.exists(self._tts_cache[text]):
            cached = self._tts_cache[text]

            if os.path.abspath(cached) != os.path.abspath(output_path):
                shutil.copy(cached, output_path)

            print(f"  TTS cache hit: '{text}'")
            return output_path

        # Generate
        print(f"  Generating TTS: '{text}'")

        audio = self.client.text_to_speech.convert(
            text=text,
            voice_id=self.voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            voice_settings=VoiceSettings(
                stability=0.7,
                similarity_boost=1.0,
                style=0.1,
                use_speaker_boost=True,
            )
        )

        with open(cache_path, "wb") as f:
            for chunk in audio:
                if isinstance(chunk, bytes):
                    f.write(chunk)

        shutil.copy(cache_path, output_path)
        self._tts_cache[text] = cache_path

        return output_path

    # ── Phrase detection ──────────────────────────────────────────

    def _detect_phrase_boundaries(self, audio, sr):
        hop_length = 512

        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=sr, hop_length=hop_length, backtrack=True
        )
        onset_samples = librosa.frames_to_samples(onset_frames, hop_length=hop_length)

        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        rms = rms / (np.max(rms) + 1e-8)

        silence_frames = np.where(rms < 0.1)[0]
        silence_samples = librosa.frames_to_samples(silence_frames, hop_length=hop_length)

        boundaries = np.concatenate(([0], onset_samples, silence_samples, [len(audio)]))
        boundaries = np.unique(boundaries)

        min_len = int(0.15 * sr)
        merged = [boundaries[0]]

        for b in boundaries[1:]:
            if b - merged[-1] < min_len:
                continue
            merged.append(b)

        return np.array(merged)

    def _should_split_phrase(self, audio, sr):
        duration = len(audio) / sr
        if duration < 2.5:
            return False

        bounds = self._detect_phrase_boundaries(audio, sr)
        return (len(bounds) - 1) >= 3

    # ── Phrase-based TTS ──────────────────────────────────────────

    def generate_phrase_spliced_segment(
        self,
        text,
        vocal_path,
        seg_start,
        seg_end,
        output_path,
        tmpdir,
    ):
        y_orig, sr = librosa.load(
            vocal_path, sr=None,
            offset=seg_start,
            duration=seg_end - seg_start
        )

        if len(y_orig) == 0 or not self._should_split_phrase(y_orig, sr):
            return self.generate_tts_segment(text, output_path)

        bounds = self._detect_phrase_boundaries(y_orig, sr)

        min_phrase_duration = 0.35

        phrase_durations = [
            max((bounds[i+1] - bounds[i]) / sr, min_phrase_duration)
            for i in range(len(bounds) - 1)
        ]

        words = text.split()
        n = len(phrase_durations)

        if n <= 1 or len(words) < 4:
            return self.generate_tts_segment(text, output_path)

        splits = np.linspace(0, len(words), n + 1, dtype=int)

        chunks = [
            " ".join(words[splits[i]:splits[i+1]])
            for i in range(n)
            if splits[i] < splits[i+1]
        ]

        print(f"  Phrase splitting into {len(chunks)} chunks")

        audio_chunks = []

        for i, chunk_text in enumerate(chunks):
            chunk_path = os.path.join(tmpdir, f"phrase_{i}_{hash(chunk_text)}.wav")
            stretched_path = os.path.join(tmpdir, f"phrase_{i}_stretch.wav")

            self.generate_tts_segment(chunk_text, chunk_path)

            self.time_stretch_segment(
                chunk_path,
                stretched_path,
                phrase_durations[i]
            )

            y_chunk, _ = librosa.load(stretched_path, sr=sr)
            audio_chunks.append(y_chunk)

        result = audio_chunks[0]
        crossfade = int(0.01 * sr)

        for chunk in audio_chunks[1:]:
            cf = min(crossfade, len(result), len(chunk))
            if cf > 1:
                fade_out = np.linspace(1, 0, cf)
                fade_in = np.linspace(0, 1, cf)
                result[-cf:] = result[-cf:] * fade_out + chunk[:cf] * fade_in
                result = np.concatenate([result, chunk[cf:]])
            else:
                result = np.concatenate([result, chunk])

        sf.write(output_path, result, sr)
        return output_path

    # ── Pitch ─────────────────────────────────────────────────────

    def pitch_correct_segment(
        self,
        tts_path,
        vocal_path,
        seg_start,
        seg_end,
        output_path,
        max_shift_semitones=2.0,
    ):
        y_tts, sr = librosa.load(tts_path, sr=None)
        y_orig, _ = librosa.load(
            vocal_path, sr=sr,
            offset=seg_start,
            duration=seg_end - seg_start,
        )

        if len(y_orig) == 0 or len(y_tts) == 0:
            shutil.copy(tts_path, output_path)
            return output_path

        orig_f0, _, _ = librosa.pyin(y_orig, fmin=80, fmax=600, sr=sr)
        tts_f0, _, _ = librosa.pyin(y_tts, fmin=80, fmax=600, sr=sr)

        orig = orig_f0[orig_f0 > 0]
        tts = tts_f0[tts_f0 > 0]

        if len(orig) == 0 or len(tts) == 0:
            shutil.copy(tts_path, output_path)
            return output_path

        shift = 12 * np.log2(np.median(orig) / np.median(tts))
        shift = float(np.clip(shift, -max_shift_semitones, max_shift_semitones))

        if abs(shift) < 0.1:
            shutil.copy(tts_path, output_path)
            return output_path

        y_shifted = pyrb.pitch_shift(y_tts, sr, shift)
        sf.write(output_path, y_shifted, sr)
        return output_path

    # ── Time stretch ──────────────────────────────────────────────

    def time_stretch_segment(self, input_path, output_path, target_duration):
        y, sr = librosa.load(input_path, sr=None)
        current = len(y) / sr

        if abs(current - target_duration) < 0.05:
            shutil.copy(input_path, output_path)
            return output_path

        rate = max(0.5, min(2.0, current / target_duration))

        y_stretched = pyrb.time_stretch(y, sr, rate)

        target_samples = int(target_duration * sr)
        y_stretched = np.pad(y_stretched, (0, max(0, target_samples - len(y_stretched))))
        y_stretched = y_stretched[:target_samples]

        sf.write(output_path, y_stretched, sr)
        return output_path

    # ── Assembly + mix ────────────────────────────────────────────

    def assemble_vocal_track(self, segments, paths, total_duration, output_path):
        track = AudioSegment.silent(duration=int(total_duration * 1000))

        for seg, path in zip(segments, paths):
            audio = AudioSegment.from_file(path)
            track = track.overlay(audio, position=int(seg["start"] * 1000))

        track.export(output_path, format="mp3")
        return output_path

    def mix_with_instrumental(self, vocal_path, instrumental_path, output_path):
        vocals = AudioSegment.from_file(vocal_path)
        inst = AudioSegment.from_file(instrumental_path)

        if len(vocals) < len(inst):
            vocals += AudioSegment.silent(len(inst) - len(vocals))
        else:
            inst += AudioSegment.silent(len(vocals) - len(inst))

        final = inst.overlay(vocals)
        final.export(output_path, format="mp3")
        return output_path

    # ── Pipeline ──────────────────────────────────────────────────

    def run(
        self,
        vocal_path,
        instrumental_path,
        segments,
        output_path="translated_song.mp3",
        singer_name="singer",
        voice_id=None,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:

            if voice_id:
                self.voice_id = voice_id
            else:
                self.clone_voice(vocal_path, singer_name, tmpdir)

            paths = []

            for i, seg in enumerate(segments):
                tts_path = os.path.join(tmpdir, f"seg_{i}.wav")
                stretch_path = os.path.join(tmpdir, f"seg_{i}_stretch.wav")

                self.generate_phrase_spliced_segment(
                    seg["translation"],
                    vocal_path,
                    seg["start"],
                    seg["end"],
                    tts_path,
                    tmpdir,
                )

                self.time_stretch_segment(
                    tts_path,
                    stretch_path,
                    seg["end"] - seg["start"]
                )

                paths.append(stretch_path)

            vocal_track = os.path.join(tmpdir, "vocals.mp3")

            self.assemble_vocal_track(
                segments,
                paths,
                max(s["end"] for s in segments) + 1,
                vocal_track
            )

            self.mix_with_instrumental(
                vocal_track,
                instrumental_path,
                output_path
            )

            self.delete_voice()

        return output_path

if __name__ == "__main__":
    import re
    import anthropic
    import os

    # ── Init Claude ─────────────────────────────────────────────
    claude = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    # ── Init Pipeline ───────────────────────────────────────────
    pipeline = AudioPipeline(
        elevenlabs_api_key=os.environ.get("ELEVENLABS_API_KEY", "sk_...")
    )

    # ── Input Segments (original English) ───────────────────────
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

    # ── Translation Setup ───────────────────────────────────────
    target_lang = "Spanish"
    text_cache = {}

    unique_texts = list({seg["text"] for seg in segments})
    total_segments = len(segments)

    print(
        f"{total_segments} segments, {len(unique_texts)} unique lyrics "
        f"({total_segments - len(unique_texts)} duplicates skipped)"
    )

    def translate_line(text: str, lang: str = "Spanish") -> str:
        syllable_count = max(1, len(re.findall(r"[aeiouAEIOU]+", text)))

        response = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            messages=[{
                "role": "user",
                "content": (
                    f"Translate this English song lyric into {lang}.\n"
                    f"Original: \"{text}\"\n"
                    f"Match the syllable count as closely as possible (~{syllable_count}).\n"
                    f"Return ONLY the translated line."
                )
            }],
        )

        return response.content[0].text.strip().strip('"')

    # ── Translate (deduplicated) ────────────────────────────────
    print(f"Translating {len(unique_texts)} lines...")

    for text in unique_texts:
        translation = translate_line(text, target_lang)
        text_cache[text] = translation
        print(f"  '{text}' -> '{translation}'")

    # Apply translations
    for seg in segments:
        seg["translation"] = text_cache[seg["text"]]

    # ── Run Pipeline ────────────────────────────────────────────
    result = pipeline.run(
        vocal_path="separated/htdemucs/test_song/vocals_short.wav",
        instrumental_path="separated/htdemucs/test_song/no_vocals_short.wav",
        segments=segments,
        output_path="translated_song.mp3",
    )

    print(f"\n✅ Done! Output: {result}")