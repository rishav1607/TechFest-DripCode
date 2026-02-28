"""
REST API for AI Voice Detector
Uses the fine-tuned wav2vec2 model for binary classification.
"""

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import load_model, predict

# Globals set during startup
model = None
feature_extractor = None
device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_extractor, device

    print("Loading wav2vec2 model...")
    model, feature_extractor, device = load_model()
    print(f"wav2vec2 loaded on {device}")

    yield


app = FastAPI(
    title="AI Voice Detector API",
    description="AI voice detection using fine-tuned wav2vec2.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma"}


# ── Response models ──────────────────────────────────────────

class Probabilities(BaseModel):
    human: float
    ai: float


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Probabilities


class URLRequest(BaseModel):
    url: str


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


# ── Core prediction logic ────────────────────────────────────

def run_prediction(audio_path: str) -> PredictionResponse:
    """Run wav2vec2 prediction on an audio file."""
    label, confidence, probs = predict(audio_path, model, feature_extractor, device)

    return PredictionResponse(
        prediction=label,
        confidence=round(confidence, 4),
        probabilities=Probabilities(
            human=round(float(probs[0]), 4),
            ai=round(float(probs[1]), 4),
        ),
    )


# ── Endpoints ────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model="wav2vec2-base-finetuned",
        device=str(device),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_upload(file: UploadFile = File(...)):
    """Classify an uploaded audio file."""
    ext = Path(file.filename or "audio.wav").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format '{ext}'. Use: {', '.join(ALLOWED_EXTENSIONS)}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        return run_prediction(tmp_path)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/predict-url", response_model=PredictionResponse)
async def predict_from_url(body: URLRequest):
    """Download audio from a URL and classify."""
    tmp_path = None
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            resp = await client.get(body.url)
            resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(resp.content)

        return run_prediction(tmp_path)
    except httpx.HTTPError as e:
        raise HTTPException(400, f"Failed to download audio: {e}")
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
