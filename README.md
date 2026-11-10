# testingAI-features# Transcription Pipeline

A service that accepts an audio file (WAV, MP3, M4A, FLAC, OGG, WEBM, MP4), transcribes spoken language into text, and returns **segment-level timestamps**.

## Architecture

```
upload → FFmpeg normalize → faster-whisper (with VAD) → JSON segments
```

- **API:** FastAPI (`POST /transcribe`, `GET /health`)
- **ASR:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — open-source Whisper reimplementation
- **Preprocessing:** FFmpeg → 16 kHz mono 16-bit PCM WAV
- **VAD:** built into faster-whisper (`vad_filter=True`)

## Key design decisions

### 1. ASR engine: faster-whisper
- Open-source, runs locally, no API keys.
- Returns segment-level timestamps natively.
- Faster and lower-memory than the reference `openai/whisper` implementation.
- Model is lazily loaded once per process (`app/asr.py: get_model()`).

### 2. Format handling: FFmpeg normalization
Every input is normalized to **16 kHz mono 16-bit PCM WAV** before transcription (`app/preprocess.py`).
- Whisper is trained on 16 kHz audio; higher sample rates don't help and cost compute.
- Downsampling to 16 kHz preserves full speech intelligibility (2–4 kHz range).
- One normalization step makes the ASR layer **format-agnostic** — no per-format branching.

```bash
ffmpeg -y -i input.mp3 -ac 1 -ar 16000 -sample_fmt s16 -vn output.wav
```

### 3. Long audio: VAD-based segmentation
- Whisper processes audio in 30-second windows and can hallucinate text on long silences.
- faster-whisper's built-in `vad_filter=True` uses voice activity detection to skip silence and reduces hallucination.
- For **very long files (>1 hour)** or when fine control is needed, pre-segment with **Silero VAD** and call `model.transcribe(clip_timestamps=[start, end])` per chunk. This yields absolute timestamps and constant memory.

### 4. Timestamps
Each response contains a `segments` array with `start`, `end`, and `text` per segment, in seconds, absolute to the original audio. Consumable for subtitles (SRT/VTT), navigation, or downstream NLP.

```json
{
  "language": "en",
  "duration_sec": 7.4,
  "segments": [
    {"start": 0.0, "end": 3.1, "text": "Hello and welcome."},
    {"start": 3.1, "end": 7.4, "text": "This is a transcription test."}
  ]
}
```

### 5. API surface
- `POST /transcribe` — multipart file upload → returns JSON with segments
- `GET /health` — liveness check
- Rejects unsupported extensions with HTTP 400 (`ALLOWED_EXT` in `app/main.py`)

## System design (production)

The current implementation is synchronous and single-process for clarity. In production:

| Concern | Approach |
|---------|----------|
| **Concurrent uploads** | Decouple with a task queue (Celery + Redis). API enqueues a job and returns `job_id`; N GPU workers process jobs concurrently. Idempotency key on upload. |
| **Storage** | Audio in S3/MinIO, content-addressed (`{sha256}.wav`). Transcripts in PostgreSQL (`segments JSONB`), with status, model version, duration, language. Lifecycle policy expires raw uploads after N days. |
| **Retries / recovery** | Job statuses: `queued → processing → done / failed`. Exponential backoff (3 attempts), dead-letter queue for persistent failures. Jobs are idempotent — retries overwrite partial results safely. `GET /jobs/{id}` exposes status; manual re-enqueue allowed. |
| **API** | `POST /transcribe` → `job_id`; `GET /jobs/{id}` → status + segments. API key / OAuth2, rate limiting, OpenAPI docs, presigned upload URLs for large files, optional webhook callbacks. |

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# install FFmpeg (macOS: brew install ffmpeg, Ubuntu: apt install ffmpeg)
uvicorn app.main:app --reload
```

```bash
curl http://localhost:8000/health
curl -F "file=@sample.mp3" http://localhost:8000/transcribe
```

Or with Docker:

```bash
docker compose up --build
```

## Environment variables

See `.env.example`:

| Var | Default | Notes |
|-----|---------|-------|
| `WHISPER_MODEL` | `base` | Use `large-v3` for production quality |
| `WHISPER_DEVICE` | `cpu` | `cuda` for GPU |
| `WHISPER_COMPUTE_TYPE` | `int8` | `float16` on GPU |
| `WHISPER_LANGUAGE` | `en` | ISO code |

## Project layout

```
app/
  main.py         FastAPI routes
  preprocess.py   FFmpeg normalization
  asr.py          faster-whisper wrapper
  config.py       env-driven settings
tests/
  test_health.py
Dockerfile
docker-compose.yml
requirements.txt
```

## Limitations / next steps
- Synchronous endpoint — long uploads block. Move to task queue in production.
- No speaker diarization (single-speaker output).
- No word-level timestamps (add `word_timestamps=True` if needed).