import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from .preprocess import normalize_audio
from .asr import transcribe

app = FastAPI(title="Transcription Pipeline", version="0.1.0")

ALLOWED_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported format: {ext}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        raw_path = tmp.name

    wav_path = None
    try:
        wav_path = normalize_audio(raw_path)
        result = transcribe(wav_path)
        result["filename"] = file.filename
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")
    finally:
        for p in (raw_path, wav_path):
            if p and os.path.exists(p):
                os.unlink(p)