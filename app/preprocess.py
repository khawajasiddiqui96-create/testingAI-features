import os
import subprocess
import tempfile
from .config import TARGET_SR


def normalize_audio(input_path: str) -> str:
    """Convert any audio/video to 16 kHz mono 16-bit PCM WAV.

    Whisper is trained on 16 kHz mono audio; normalizing here makes the rest
    of the pipeline format-agnostic (WAV, MP3, M4A, FLAC, MP4, etc.).
    """
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1",
        "-ar", str(TARGET_SR),
        "-sample_fmt", "s16",
        "-vn",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise RuntimeError(
            f"FFmpeg failed to normalize audio: {e.stderr.decode(errors='ignore')[:300]}"
        )
    return output_path