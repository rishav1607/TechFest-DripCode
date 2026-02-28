"""Voice classifier client — calls the AI Voice Detector API (Dataset/api.py).

Sends audio to http://localhost:8000/predict and returns AI/human classification.
Fails open (defaults to "human") if the classifier API is unavailable.
"""

import logging
from urllib.request import Request, urlopen
from urllib.error import URLError
import json as json_mod

logger = logging.getLogger(__name__)

CLASSIFIER_URL = "http://localhost:8000"


def classify_audio(wav_bytes: bytes, timeout: float = 10.0) -> dict:
    """Send WAV audio to the AI Voice Detector API and return classification.

    Returns:
        {
            "prediction": "ai" | "human",
            "confidence": float,
            "probabilities": {"human": float, "ai": float},
        }
        On error, returns a fail-open default (human).
    """
    import io

    # Build multipart/form-data body manually (avoids `requests` dependency)
    boundary = "----KarmaAIBoundary"
    body = io.BytesIO()

    # File field
    body.write(f"--{boundary}\r\n".encode())
    body.write(
        b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
    )
    body.write(b"Content-Type: audio/wav\r\n\r\n")
    body.write(wav_bytes)
    body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())

    data = body.getvalue()

    req = Request(
        f"{CLASSIFIER_URL}/predict",
        data=data,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=timeout) as resp:
            result = json_mod.loads(resp.read().decode())
            logger.info(
                "Voice classification: %s (confidence %.2f%%)",
                result.get("prediction", "?"),
                result.get("confidence", 0) * 100,
            )
            return result
    except Exception as e:
        logger.warning("Voice classification failed: %s — defaulting to human", e)
        return _default_human()


def is_classifier_healthy() -> bool:
    """Check if the classifier API is running and ready."""
    try:
        req = Request(f"{CLASSIFIER_URL}/health")
        with urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _default_human() -> dict:
    """Fail-open default: assume human."""
    return {
        "prediction": "human",
        "confidence": 0.0,
        "probabilities": {"human": 0.5, "ai": 0.5},
    }
