"""
Step 1: Download YouTube audio as WAV via yt-dlp.

Uses yt-dlp for the download and the ffmpeg binary from imageio-ffmpeg
so WAV conversion works on Windows without a separate ffmpeg/ffprobe install.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg
import yt_dlp


def download_youtube_audio(url: str, output_dir: str | Path = "output/audio") -> str:
    """
    Download audio from a YouTube URL and save as 16-bit PCM WAV (44.1 kHz stereo).

    Returns the absolute path to the .wav file.
    """
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    outtmpl = str(out / "%(id)s.%(ext)s")

    ydl_opts: dict = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": False,
        "no_warnings": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info["id"]
        fp = info.get("filepath")
        downloaded = Path(fp) if fp else Path(ydl.prepare_filename(info))

    wav_path = out / f"{video_id}.wav"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-i",
        str(downloaded),
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({result.returncode}):\n{result.stderr or result.stdout}"
        )

    if downloaded.resolve() != wav_path.resolve() and downloaded.is_file():
        downloaded.unlink()

    if not wav_path.is_file():
        raise FileNotFoundError(f"Expected WAV not found: {wav_path}")

    return str(wav_path.resolve())


if __name__ == "__main__":
    import sys

    # Luis Fonsi - Despacito ft. Daddy Yankee (Spanish)
    default_url = "https://www.youtube.com/watch?v=kJQP7kiw5Fk"
    url = sys.argv[1] if len(sys.argv) > 1 else default_url
    path = download_youtube_audio(url)
    print(path)
