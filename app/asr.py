from faster_whisper import WhisperModel
from .config import MODEL_SIZE, DEVICE, COMPUTE_TYPE, LANGUAGE

_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    """Lazy-load the ASR model once per process."""
    global _model
    if _model is None:
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model


def transcribe(audio_path: str) -> dict:
    """Transcribe a normalized WAV and return segment-level timestamps.

    - vad_filter=True  -> skips silence and reduces hallucination.
    - beam_size=5      -> small quality boost at low cost.
    - Timestamps are absolute to the original file (faster-whisper handles
      internal 30s windowing + VAD offsets for us).
    """
    model = get_model()
    segments, info = model.transcribe(
        audio_path,
        language=LANGUAGE,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
    )
    return {
        "language": info.language,
        "duration_sec": round(info.duration, 2),
        "segments": [
            {
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "text": s.text.strip(),
            }
            for s in segments
        ],
    }