"""Cartesia AI API service - TTS (Sonic 3) and STT (ink-whisper)."""

import base64
import io
import json
import logging
import os
import struct
import time
import uuid

import requests
import websocket  # websocket-client library

logger = logging.getLogger(__name__)

CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
CARTESIA_VERSION = "2025-04-16"
CARTESIA_TTS_URL = "https://api.cartesia.ai/tts/bytes"
CARTESIA_STT_WS_URL = "wss://api.cartesia.ai/stt/websocket"

# Default Hindi female voice — "Indian Lady"
# Override via CARTESIA_VOICE_ID in .env
CARTESIA_VOICE_ID = os.getenv(
    "CARTESIA_VOICE_ID", "3b554273-4299-48b9-9aaf-eefd438e3941"
)

_session = requests.Session()
_session.headers.update({
    "X-API-Key": CARTESIA_API_KEY,
    "Cartesia-Version": CARTESIA_VERSION,
})

MAX_RETRIES = 3
RETRY_DELAY = 1


def _retry_request(method, url, retries=MAX_RETRIES, **kwargs):
    """Make an HTTP request with automatic retries on 5xx errors."""
    for attempt in range(1, retries + 1):
        try:
            resp = method(url, **kwargs)
            if resp.ok or resp.status_code < 500:
                return resp
            logger.warning(
                "Cartesia %d error (attempt %d/%d): %s",
                resp.status_code, attempt, retries, resp.text[:200],
            )
        except requests.exceptions.RequestException as e:
            logger.warning(
                "Cartesia request error (attempt %d/%d): %s",
                attempt, retries, e,
            )
            if attempt == retries:
                raise
        if attempt < retries:
            time.sleep(RETRY_DELAY * attempt)
    return resp


def text_to_speech(
    text: str,
    language_code: str = "hi-IN",
    speaker: str = "",
    sample_rate: str = "8000",
) -> bytes:
    """Convert text to speech using Cartesia Sonic 3.

    Args:
        text: Text to speak.
        language_code: BCP-47 language code (mapped to Cartesia's 2-letter code).
        speaker: Ignored for Cartesia (voice is set via CARTESIA_VOICE_ID).
        sample_rate: Audio sample rate. Use "8000" for telephony, "22050" for web.

    Returns:
        WAV audio bytes.
    """
    # Map BCP-47 to Cartesia 2-letter language code
    lang_map = {
        "hi-IN": "hi",
        "en-US": "en",
        "en-IN": "en",
    }
    cartesia_lang = lang_map.get(language_code, language_code.split("-")[0])

    sr = int(sample_rate)

    # For telephony (8kHz), use mu-law encoding for better Twilio compatibility
    if sr == 8000:
        encoding = "pcm_s16le"
    else:
        encoding = "pcm_s16le"

    payload = {
        "model_id": "sonic-3",
        "transcript": text,
        "voice": {
            "mode": "id",
            "id": CARTESIA_VOICE_ID,
        },
        "output_format": {
            "container": "wav",
            "encoding": encoding,
            "sample_rate": sr,
        },
        "language": cartesia_lang,
    }

    resp = _retry_request(
        _session.post,
        CARTESIA_TTS_URL,
        json=payload,
        headers={
            "X-API-Key": CARTESIA_API_KEY,
            "Cartesia-Version": CARTESIA_VERSION,
            "Content-Type": "application/json",
        },
        timeout=30,
    )

    if not resp.ok:
        logger.error("Cartesia TTS error %d: %s", resp.status_code, resp.text[:500])
    resp.raise_for_status()

    return resp.content


def _extract_raw_pcm_from_wav(wav_bytes: bytes) -> bytes:
    """Strip WAV header and return raw PCM data."""
    buf = io.BytesIO(wav_bytes)
    # Skip RIFF header (44 bytes for standard WAV)
    buf.read(4)  # "RIFF"
    buf.read(4)  # file size
    buf.read(4)  # "WAVE"

    # Find "data" chunk
    while True:
        chunk_id = buf.read(4)
        if len(chunk_id) < 4:
            # Fallback: just skip first 44 bytes
            return wav_bytes[44:]
        chunk_size = struct.unpack("<I", buf.read(4))[0]
        if chunk_id == b"data":
            return buf.read(chunk_size)
        buf.read(chunk_size)


def speech_to_text(audio_bytes: bytes, language_code: str = "hi-IN") -> str:
    """Convert speech audio to text using Cartesia ink-whisper (WebSocket STT).

    Args:
        audio_bytes: Raw audio bytes (WAV format).
        language_code: BCP-47 language code.

    Returns:
        Transcribed text string.
    """
    lang_map = {
        "hi-IN": "hi",
        "en-US": "en",
        "en-IN": "en",
    }
    cartesia_lang = lang_map.get(language_code, language_code.split("-")[0])

    # Extract raw PCM from WAV for WebSocket streaming
    raw_pcm = _extract_raw_pcm_from_wav(audio_bytes)

    ws_url = (
        f"{CARTESIA_STT_WS_URL}"
        f"?api_key={CARTESIA_API_KEY}"
        f"&cartesia_version={CARTESIA_VERSION}"
        f"&model=ink-whisper"
        f"&language={cartesia_lang}"
        f"&encoding=pcm_s16le"
        f"&sample_rate=16000"
    )

    transcript_parts = []
    error_msg = None
    done_received = False

    def on_message(ws, message):
        nonlocal done_received
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            if msg_type == "transcript":
                # Collect final transcripts
                for word in data.get("words", []):
                    transcript_parts.append(word.get("word", ""))
            elif msg_type == "done":
                done_received = True
                ws.close()
            elif msg_type == "error":
                nonlocal error_msg
                error_msg = data.get("message", "Unknown STT error")
                ws.close()
        except json.JSONDecodeError:
            pass

    def on_error(ws, error):
        nonlocal error_msg
        error_msg = str(error)
        logger.error("Cartesia STT WebSocket error: %s", error)

    def on_open(ws):
        # Send audio data in chunks
        chunk_size = 8192  # 8KB chunks
        for i in range(0, len(raw_pcm), chunk_size):
            chunk = raw_pcm[i : i + chunk_size]
            ws.send(chunk, opcode=websocket.ABNF.OPCODE_BINARY)

        # Signal end of audio
        ws.send("done")

    ws = websocket.WebSocketApp(
        ws_url,
        on_message=on_message,
        on_error=on_error,
        on_open=on_open,
    )

    # Run with timeout
    ws.run_forever(ping_interval=10, ping_timeout=5)

    if error_msg:
        logger.error("Cartesia STT error: %s", error_msg)
        raise RuntimeError(f"Cartesia STT error: {error_msg}")

    transcript = " ".join(transcript_parts).strip()
    logger.info("Cartesia STT transcript: %s", transcript)
    return transcript


# ---------------------------------------------------------------------------
#  Streaming TTS via WebSocket — outputs pcm_mulaw 8kHz for Twilio
# ---------------------------------------------------------------------------

_LANG_MAP = {"hi-IN": "hi", "en-US": "en", "en-IN": "en"}


class CartesiaTTSStreamer:
    """Persistent Cartesia WebSocket connection for streaming TTS.

    Outputs raw pcm_mulaw at 8kHz — the base64 data from each chunk can be
    forwarded directly to Twilio Media Streams with zero conversion.
    """

    def __init__(self, language_code: str = "hi-IN"):
        self.language = _LANG_MAP.get(language_code, language_code.split("-")[0])
        self.ws = None
        self._connect()

    def _connect(self):
        ws_url = (
            f"wss://api.cartesia.ai/tts/websocket"
            f"?api_key={CARTESIA_API_KEY}"
            f"&cartesia_version={CARTESIA_VERSION}"
        )
        self.ws = websocket.create_connection(ws_url, timeout=30)
        logger.debug("Cartesia TTS WebSocket connected")

    def speak(self, text: str, on_chunk):
        """Generate audio for *text* and call ``on_chunk(b64_mulaw)`` for each chunk.

        Each chunk is base64-encoded raw mulaw 8 kHz mono audio — ready to be
        placed directly into a Twilio ``media.payload`` field.
        """
        context_id = uuid.uuid4().hex

        msg = {
            "model_id": "sonic-3",
            "transcript": text,
            "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
            "language": self.language,
            "context_id": context_id,
            "output_format": {
                "container": "raw",
                "encoding": "pcm_mulaw",
                "sample_rate": 8000,
            },
            "continue": False,
        }

        self.ws.send(json.dumps(msg))

        while True:
            raw = self.ws.recv()
            if not raw:
                break
            data = json.loads(raw)

            if data.get("type") == "chunk" and data.get("data"):
                on_chunk(data["data"])

            if data.get("done"):
                break

            if data.get("type") == "error":
                logger.error("Cartesia TTS stream error: %s", data.get("error"))
                break

    def close(self):
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
