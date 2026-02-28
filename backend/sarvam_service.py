"""Sarvam AI API service - STT and TTS only. LLM is handled by llm_service.py."""

import base64
import io
import logging
import os
import time
import requests

logger = logging.getLogger(__name__)

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_BASE_URL = "https://api.sarvam.ai"

_session = requests.Session()
_session.headers.update({"api-subscription-key": SARVAM_API_KEY})

MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds


def _retry_request(method, url, retries=MAX_RETRIES, **kwargs):
    """Make an HTTP request with automatic retries on 5xx errors."""
    for attempt in range(1, retries + 1):
        resp = method(url, **kwargs)
        if resp.ok or resp.status_code < 500:
            return resp
        logger.warning("Sarvam %d error (attempt %d/%d): %s",
                       resp.status_code, attempt, retries, resp.text[:200])
        if attempt < retries:
            time.sleep(RETRY_DELAY * attempt)
    return resp  # return last response so caller can raise


def speech_to_text(audio_bytes: bytes, language_code: str = "hi-IN") -> str:
    """Convert speech audio to text using Sarvam STT.

    Args:
        audio_bytes: Raw audio bytes (WAV format preferred).
        language_code: BCP-47 language code (default hi-IN for Hindi).

    Returns:
        Transcribed text string.
    """
    url = f"{SARVAM_BASE_URL}/speech-to-text"

    files = {
        "file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav"),
    }
    data = {
        "model": "saaras:v3",
        "language_code": language_code,
    }

    resp = _retry_request(_session.post, url, files=files, data=data, timeout=30)
    if not resp.ok:
        logger.error("Sarvam STT error %d: %s", resp.status_code, resp.text[:500])
    resp.raise_for_status()
    result = resp.json()
    return result.get("transcript", "")


def text_to_speech(
    text: str,
    language_code: str = "hi-IN",
    speaker: str = "kavya",
    sample_rate: str = "8000",
) -> bytes:
    """Convert text to speech using Sarvam TTS.

    Args:
        text: Text to speak (supports Hindi-English code-mixed).
        language_code: Target language code.
        speaker: Voice name (default 'kavya').
        sample_rate: Audio sample rate. Use 8000 for telephony, 22050 for web.

    Returns:
        WAV audio bytes.
    """
    url = f"{SARVAM_BASE_URL}/text-to-speech"

    payload = {
        "text": text,
        "target_language_code": language_code,
        "model": "bulbul:v3",
        "speaker": speaker,
        "speech_sample_rate": sample_rate,
        "pace": 1.0,
    }

    resp = _retry_request(
        _session.post, url,
        json=payload,
        headers={
            "api-subscription-key": SARVAM_API_KEY,
            "Content-Type": "application/json",
        },
        timeout=30,
    )

    if not resp.ok:
        logger.error("Sarvam TTS error %d: %s", resp.status_code, resp.text[:500])
    resp.raise_for_status()
    result = resp.json()

    audio_b64 = result["audios"][0]
    return base64.b64decode(audio_b64)
