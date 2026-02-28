"""Unified speech service — routes TTS/STT to Sarvam or Cartesia based on .env config.

Environment variables:
    TTS_PROVIDER: "sarvam" (default) or "cartesia"
    STT_PROVIDER: "sarvam" (default) or "cartesia"
"""

import logging
import os

logger = logging.getLogger(__name__)

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "sarvam").lower().strip()
STT_PROVIDER = os.getenv("STT_PROVIDER", "sarvam").lower().strip()

# Lazy-import providers to avoid loading unused API keys / sessions
_sarvam = None
_cartesia = None


def _get_sarvam():
    global _sarvam
    if _sarvam is None:
        import sarvam_service as mod
        _sarvam = mod
    return _sarvam


def _get_cartesia():
    global _cartesia
    if _cartesia is None:
        import cartesia_service as mod
        _cartesia = mod
    return _cartesia


def speech_to_text(audio_bytes: bytes, language_code: str = "hi-IN") -> str:
    """Convert speech audio to text using the configured STT provider.

    Args:
        audio_bytes: Raw audio bytes (WAV format).
        language_code: BCP-47 language code (default hi-IN for Hindi).

    Returns:
        Transcribed text string.
    """
    if STT_PROVIDER == "cartesia":
        logger.debug("STT via Cartesia ink-whisper")
        return _get_cartesia().speech_to_text(audio_bytes, language_code)
    else:
        logger.debug("STT via Sarvam saaras:v3")
        return _get_sarvam().speech_to_text(audio_bytes, language_code)


def text_to_speech(
    text: str,
    language_code: str = "hi-IN",
    speaker: str = "kavya",
    sample_rate: str = "8000",
) -> bytes:
    """Convert text to speech using the configured TTS provider.

    Args:
        text: Text to speak.
        language_code: Target language code.
        speaker: Voice name (used by Sarvam; ignored by Cartesia).
        sample_rate: Audio sample rate. Use "8000" for telephony, "22050" for web.

    Returns:
        WAV audio bytes.
    """
    if TTS_PROVIDER == "cartesia":
        logger.debug("TTS via Cartesia Sonic 3")
        return _get_cartesia().text_to_speech(text, language_code, speaker, sample_rate)
    else:
        logger.debug("TTS via Sarvam bulbul:v3")
        return _get_sarvam().text_to_speech(text, language_code, speaker, sample_rate)


def get_provider_info() -> dict:
    """Return current provider configuration for logging/health checks."""
    return {
        "tts_provider": TTS_PROVIDER,
        "stt_provider": STT_PROVIDER,
        "tts_model": "Cartesia Sonic 3" if TTS_PROVIDER == "cartesia" else "Sarvam bulbul:v3",
        "stt_model": "Cartesia ink-whisper" if STT_PROVIDER == "cartesia" else "Sarvam saaras:v3",
    }
