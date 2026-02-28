"""Twilio Media Streams handler — bidirectional WebSocket streaming pipeline.

Replaces the old Gather/Play TwiML approach with a real-time streaming
architecture:

    Twilio mulaw 8kHz ──WebSocket──▶ Server VAD ──▶ STT (Sarvam)
                                                       │
                                                       ▼
    Twilio ◀──WebSocket── Cartesia TTS (streaming) ◀── LLM (streaming)

Audio from Cartesia is pcm_mulaw 8kHz — byte-for-byte compatible with
Twilio's expected format, so it flows through with zero transcoding.
"""

import audioop
import base64
import json
import logging
import struct
import time
import wave
import io

from simple_websocket.errors import ConnectionClosed

from cartesia_service import CartesiaTTSStreamer
from conversation import GREETING_TEXT
from database import save_intel, save_message
from intel_extractor import extract_intel
from llm_service import chat_completion_streaming
from speech_service import speech_to_text
from voice_classifier import classify_audio

logger = logging.getLogger(__name__)

# Try to import webrtcvad; fall back to energy-based VAD if unavailable
try:
    import webrtcvad

    _webrtcvad_available = True
except ImportError:
    _webrtcvad_available = False
    logger.warning(
        "webrtcvad not installed — using energy-based VAD fallback. "
        "Install with: pip install webrtcvad"
    )


def _energy_vad(pcm16_frame: bytes, _sample_rate: int, threshold: int = 500) -> bool:
    """Simple RMS energy-based voice activity detection (fallback)."""
    n_samples = len(pcm16_frame) // 2
    if n_samples == 0:
        return False
    total = 0
    for i in range(n_samples):
        sample = struct.unpack_from("<h", pcm16_frame, i * 2)[0]
        total += sample * sample
    rms = (total / n_samples) ** 0.5
    return rms > threshold


class TwilioStreamHandler:
    """Handles a single Twilio Media Stream WebSocket connection.

    Receives caller audio, detects speech via VAD, runs the full
    STT → LLM (streaming) → TTS (streaming) pipeline, and streams
    the response audio back to Twilio in real time.
    """

    # Audio framing constants (Twilio = mulaw 8kHz mono)
    SAMPLE_RATE = 8000
    FRAME_DURATION_MS = 20
    FRAME_SIZE_MULAW = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 160 bytes
    FRAME_SIZE_PCM16 = FRAME_SIZE_MULAW * 2  # 320 bytes

    # VAD thresholds
    SILENCE_THRESHOLD_FRAMES = 25  # 500ms of silence to end utterance
    MIN_SPEECH_FRAMES = 20  # 400ms minimum speech to avoid noise triggers

    # Post-response cooldown — discard inbound audio briefly after AI speaks
    # to avoid picking up echo of our own TTS playback
    COOLDOWN_FRAMES = 50  # 1 second (50 × 20ms)

    # Classification phase duration (seconds)
    CLASSIFICATION_DURATION = 3.5

    def __init__(self, ws, *, socketio, conversation_mgr, mute_state,
                 greeting_mulaw=None, ai_detected_mulaw=None):
        self.ws = ws
        self.socketio = socketio
        self.conversation_mgr = conversation_mgr
        self.mute_state = mute_state
        self.greeting_mulaw = greeting_mulaw
        self.ai_detected_mulaw = ai_detected_mulaw

        # Stream identifiers (set on 'start' event)
        self.stream_sid = None
        self.call_sid = None
        self.caller = None

        # --- Voice classification state ---
        self.classification_phase = True
        self.classification_audio = bytearray()  # raw PCM16 8kHz
        self.classification_start_time = None
        self.classification_speech_frames = 0  # track if caller spoke
        self.caller_is_ai = False

        # VAD state
        if _webrtcvad_available:
            self.vad = webrtcvad.Vad(3)  # most aggressive
        else:
            self.vad = None
        self.mulaw_buffer = bytearray()
        self.audio_buffer = bytearray()  # accumulated PCM16 speech
        self.speech_started = False
        self.silence_count = 0
        self.speech_frame_count = 0

        # Processing flag (for clarity; single-threaded so not strictly needed)
        self.is_processing = False
        self.cooldown_remaining = 0  # frames to skip after AI response

    # ------------------------------------------------------------------
    #  Main loop
    # ------------------------------------------------------------------
    def run(self):
        """Receive and process Twilio media stream messages until the stream ends."""
        try:
            while True:
                message = self.ws.receive()
                if message is None:
                    break
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue
                self._handle_message(data)
        except ConnectionClosed:
            pass  # normal — Twilio closes WebSocket when call ends
        except Exception as e:
            logger.error("TwilioStreamHandler error: %s", e, exc_info=True)
        finally:
            logger.info("Media stream ended (call %s)", self.call_sid)

    # ------------------------------------------------------------------
    #  Message dispatch
    # ------------------------------------------------------------------
    def _handle_message(self, data: dict):
        event = data.get("event")
        if event == "connected":
            logger.info("Twilio media stream connected")
        elif event == "start":
            self._on_start(data)
        elif event == "media":
            self._on_media(data)
        elif event == "mark":
            self._on_mark(data)
        elif event == "stop":
            logger.info("Twilio media stream stopped (call %s)", self.call_sid)

    # ------------------------------------------------------------------
    #  Event handlers
    # ------------------------------------------------------------------
    def _on_start(self, data: dict):
        start = data.get("start", {})
        self.stream_sid = start.get("streamSid")
        self.call_sid = start.get("callSid")
        params = start.get("customParameters", {})
        self.caller = params.get("caller", "unknown")

        logger.info(
            "Stream started: call=%s stream=%s caller=%s",
            self.call_sid, self.stream_sid, self.caller,
        )

        # Send greeting first, then enter classification phase.
        # The caller hears "Haaaan? Hello? Kaun bol raha hai?" and responds —
        # their response audio is what gets classified as AI or human.
        self._send_greeting()
        self.cooldown_remaining = self.COOLDOWN_FRAMES
        self.classification_phase = True
        # classification_start_time is set once cooldown expires (in _on_media)
        self._broadcast_status("CLASSIFYING CALLER...")
        logger.info("Greeting sent (call %s) — classification will begin after cooldown",
                     self.call_sid)

    def _on_media(self, data: dict):
        if self.caller_is_ai:
            return  # call classified as AI — ignore all further audio

        if self.is_processing:
            return  # ignore audio while pipeline is running

        payload = data.get("media", {}).get("payload", "")
        if not payload:
            return

        mulaw_bytes = base64.b64decode(payload)

        # ── Classification phase: skip greeting echo, then accumulate caller audio ──
        if self.classification_phase:
            # First, burn through cooldown frames (echo of our greeting)
            n_frames = len(mulaw_bytes) // self.FRAME_SIZE_MULAW
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= max(n_frames, 1)
                if self.cooldown_remaining <= 0:
                    # Cooldown done — start the classification timer now
                    self.cooldown_remaining = 0
                    self.classification_start_time = time.time()
                    logger.info("Cooldown done — collecting caller audio for classification (call %s)",
                                self.call_sid)
                return

            # Convert mulaw → PCM16 and accumulate caller's response
            pcm16 = audioop.ulaw2lin(mulaw_bytes, 2)
            self.classification_audio.extend(pcm16)

            # Track if there's any speech (for logging)
            for offset in range(0, len(pcm16), self.FRAME_SIZE_PCM16):
                frame = pcm16[offset: offset + self.FRAME_SIZE_PCM16]
                if len(frame) == self.FRAME_SIZE_PCM16 and self._check_vad(frame):
                    self.classification_speech_frames += 1

            # Check if classification window has elapsed
            elapsed = time.time() - self.classification_start_time
            if elapsed >= self.CLASSIFICATION_DURATION:
                self._run_classification()
            return

        # ── Normal audio processing ──
        self.mulaw_buffer.extend(mulaw_bytes)

        # Process in fixed-size frames for VAD
        while len(self.mulaw_buffer) >= self.FRAME_SIZE_MULAW:
            frame_mulaw = bytes(self.mulaw_buffer[: self.FRAME_SIZE_MULAW])
            del self.mulaw_buffer[: self.FRAME_SIZE_MULAW]

            # Post-response cooldown — skip frames to avoid echo
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
                continue

            # Convert mulaw → PCM16
            frame_pcm16 = audioop.ulaw2lin(frame_mulaw, 2)

            # Run VAD
            is_speech = self._check_vad(frame_pcm16)

            if is_speech:
                self.speech_started = True
                self.silence_count = 0
                self.speech_frame_count += 1
                self.audio_buffer.extend(frame_pcm16)

            elif self.speech_started:
                self.silence_count += 1
                self.audio_buffer.extend(frame_pcm16)  # keep trailing silence

                if self.silence_count >= self.SILENCE_THRESHOLD_FRAMES:
                    if self.speech_frame_count >= self.MIN_SPEECH_FRAMES:
                        # Utterance complete — process it
                        pcm_data = bytes(self.audio_buffer)
                        self._reset_vad_state()
                        self.is_processing = True
                        try:
                            self._process_speech(pcm_data)
                        except Exception as e:
                            logger.error(
                                "Error in speech pipeline (call %s): %s",
                                self.call_sid, e, exc_info=True,
                            )
                        finally:
                            self.is_processing = False
                            self.cooldown_remaining = self.COOLDOWN_FRAMES
                    else:
                        # Too short — probably noise, discard
                        self._reset_vad_state()

    def _on_mark(self, data: dict):
        name = data.get("mark", {}).get("name", "")
        logger.debug("Mark received: %s (call %s)", name, self.call_sid)

    # ------------------------------------------------------------------
    #  Voice classification
    # ------------------------------------------------------------------
    def _run_classification(self):
        """Classify the caller as AI or human using collected audio."""
        self.classification_phase = False
        pcm_data = bytes(self.classification_audio)
        self.classification_audio.clear()

        logger.info(
            "Classification: collected %.1fs of audio (%d speech frames) for call %s",
            len(pcm_data) / (self.SAMPLE_RATE * 2),
            self.classification_speech_frames,
            self.call_sid,
        )

        # If virtually no speech was detected, default to human and proceed
        if self.classification_speech_frames < 5:
            logger.info("Too little speech for classification — defaulting to human (call %s)",
                        self.call_sid)
            self._finish_classification_as_human()
            return

        # Wrap PCM16 8kHz in WAV for the classifier
        wav_bytes = self._pcm_to_wav(pcm_data, sample_rate=self.SAMPLE_RATE)

        try:
            result = classify_audio(wav_bytes, timeout=10.0)
        except Exception as e:
            logger.error("Classification error (call %s): %s — defaulting to human",
                         self.call_sid, e)
            self._finish_classification_as_human()
            return

        prediction = result.get("prediction", "human").lower()
        confidence = result.get("confidence", 0)

        logger.info(
            "Classification result (call %s): %s (confidence %.1f%%)",
            self.call_sid, prediction.upper(), confidence * 100,
        )

        # Broadcast classification result to dashboard
        self.socketio.emit("transcript_message", {
            "call_sid": self.call_sid,
            "speaker": "system",
            "text": f"Caller classified as {prediction.upper()} (confidence: {confidence:.0%})",
        }, room="dashboard")

        if prediction == "ai":
            self.caller_is_ai = True
            self._broadcast_status("AI CALLER DETECTED")

            # Play "Caller is classified as AI" announcement
            self._play_ai_detected_message()

            logger.info("AI caller detected — skipping conversation (call %s)", self.call_sid)
        else:
            self._finish_classification_as_human()

    def _finish_classification_as_human(self):
        """Classification done — caller is human.  Send greeting and begin."""
        self._broadcast_status("DEFENDING")
        self._send_greeting()
        self.cooldown_remaining = self.COOLDOWN_FRAMES

    def _play_ai_detected_message(self):
        """Play the pre-cached 'Caller is classified as AI' audio."""
        if self.ai_detected_mulaw:
            chunk_size = 640  # 80 ms at 8 kHz mulaw
            for i in range(0, len(self.ai_detected_mulaw), chunk_size):
                chunk = self.ai_detected_mulaw[i: i + chunk_size]
                b64 = base64.b64encode(chunk).decode("ascii")
                self._send_audio_to_twilio(b64)
            self._send_mark("ai_detected_end")
        else:
            # Generate on the fly via Cartesia
            tts = CartesiaTTSStreamer(language_code="en")
            try:
                tts.speak("Caller is classified as AI.", self._send_audio_to_twilio)
            finally:
                tts.close()
            self._send_mark("ai_detected_end")

    # ------------------------------------------------------------------
    #  VAD helpers
    # ------------------------------------------------------------------
    def _check_vad(self, pcm16_frame: bytes) -> bool:
        try:
            if self.vad is not None:
                return self.vad.is_speech(pcm16_frame, self.SAMPLE_RATE)
            return _energy_vad(pcm16_frame, self.SAMPLE_RATE)
        except Exception:
            return False

    def _reset_vad_state(self):
        self.audio_buffer.clear()
        self.mulaw_buffer.clear()
        self.speech_started = False
        self.silence_count = 0
        self.speech_frame_count = 0

    # ------------------------------------------------------------------
    #  Core pipeline: STT → LLM (streaming) → TTS (streaming) → Twilio
    # ------------------------------------------------------------------
    def _process_speech(self, pcm16_8k: bytes):
        """Run the full pipeline for a single utterance."""

        # 1. Upsample 8 kHz → 16 kHz (most STT models expect 16 kHz)
        pcm16_16k, _ = audioop.ratecv(pcm16_8k, 2, 1, 8000, 16000, None)

        # 2. Wrap in WAV for STT
        wav_bytes = self._pcm_to_wav(pcm16_16k, sample_rate=16000)

        # 3. STT
        self._broadcast_status("ANALYZING...")
        transcript = speech_to_text(wav_bytes, language_code="hi-IN")

        if not transcript or not transcript.strip():
            logger.debug("Empty transcript — skipping (call %s)", self.call_sid)
            return

        logger.info("Scammer (call %s): %s", self.call_sid, transcript)

        # 4. Save scammer message & broadcast
        save_message(self.call_sid, "user", transcript)
        self._broadcast_transcript("scammer", transcript)

        # 5. Intel extraction
        intel_items = extract_intel(transcript)
        for item in intel_items:
            save_intel(
                self.call_sid,
                item["field_name"],
                item["field_value"],
                item["confidence"],
            )
        self._broadcast_intel(intel_items)

        # 6. Check mute
        if self.mute_state.get(self.call_sid, False):
            self._broadcast_status("MUTED")
            return

        # 7. LLM (streaming) → TTS (streaming) → Twilio
        self._broadcast_status("DEFENDING")

        messages = self.conversation_mgr.add_user_message(
            self.call_sid, transcript
        )

        tts = CartesiaTTSStreamer(language_code="hi-IN")
        full_response_parts: list[str] = []
        sentence_buf = ""

        try:
            for token in chat_completion_streaming(messages, temperature=0.8):
                sentence_buf += token
                full_response_parts.append(token)

                # Try to extract a complete sentence
                sentence = self._extract_sentence(sentence_buf)
                if sentence:
                    sentence_buf = sentence_buf[len(sentence):].lstrip()
                    tts.speak(sentence, self._send_audio_to_twilio)

            # Flush remaining text
            if sentence_buf.strip():
                tts.speak(sentence_buf.strip(), self._send_audio_to_twilio)
        finally:
            tts.close()

        # 8. Save AI response
        ai_response = "".join(full_response_parts)
        self.conversation_mgr.add_assistant_message(self.call_sid, ai_response)
        save_message(self.call_sid, "assistant", ai_response)
        self._broadcast_transcript("ai", ai_response)

        logger.info("AI (call %s): %s", self.call_sid, ai_response)

        # Send mark so we know when playback finishes
        self._send_mark("response_end")

    # ------------------------------------------------------------------
    #  Sentence splitting
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_sentence(text: str) -> str | None:
        """Return a complete sentence from the start of *text*, or None."""
        enders = ".!?।"
        min_len = 15  # don't split on very short fragments

        for i, ch in enumerate(text):
            if ch in enders and len(text[: i + 1].strip()) >= min_len:
                # Avoid splitting on "..." mid-sentence
                if ch == "." and i + 1 < len(text) and text[i + 1] == ".":
                    continue
                return text[: i + 1].strip()
        return None

    # ------------------------------------------------------------------
    #  Greeting
    # ------------------------------------------------------------------
    def _send_greeting(self):
        """Stream the pre-cached greeting or generate it on the fly."""
        if self.greeting_mulaw:
            # Send cached raw mulaw in ~80 ms chunks
            chunk_size = 640  # 80 ms at 8 kHz mulaw
            for i in range(0, len(self.greeting_mulaw), chunk_size):
                chunk = self.greeting_mulaw[i: i + chunk_size]
                b64 = base64.b64encode(chunk).decode("ascii")
                self._send_audio_to_twilio(b64)
            self._send_mark("greeting_end")
        else:
            # Generate via Cartesia streaming TTS
            tts = CartesiaTTSStreamer(language_code="hi-IN")
            try:
                tts.speak(GREETING_TEXT, self._send_audio_to_twilio)
            finally:
                tts.close()
            self._send_mark("greeting_end")

    # ------------------------------------------------------------------
    #  Twilio WebSocket helpers
    # ------------------------------------------------------------------
    def _send_audio_to_twilio(self, b64_mulaw: str):
        """Send a media message containing audio back to Twilio."""
        self.ws.send(json.dumps({
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"payload": b64_mulaw},
        }))

    def _send_mark(self, name: str):
        self.ws.send(json.dumps({
            "event": "mark",
            "streamSid": self.stream_sid,
            "mark": {"name": name},
        }))

    def _clear_twilio_audio(self):
        """Clear any queued outbound audio in Twilio (for barge-in)."""
        self.ws.send(json.dumps({
            "event": "clear",
            "streamSid": self.stream_sid,
        }))

    # ------------------------------------------------------------------
    #  Audio format helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
        """Wrap raw PCM16 mono data in a WAV container."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return buf.getvalue()

    # ------------------------------------------------------------------
    #  Dashboard broadcasting
    # ------------------------------------------------------------------
    def _broadcast_transcript(self, speaker: str, text: str):
        self.socketio.emit(
            "transcript_message",
            {"call_sid": self.call_sid, "speaker": speaker, "text": text},
            room="dashboard",
        )

    def _broadcast_status(self, status: str):
        self.socketio.emit(
            "ai_status", {"status": status}, room="dashboard",
        )

    def _broadcast_intel(self, intel_items: list[dict]):
        if not intel_items:
            return
        update = {}
        for item in intel_items:
            field = item["field_name"]
            value = item["field_value"]
            if field == "scammer_name":
                update["scammer_name"] = value
            elif field == "scam_type":
                update["scam_type"] = value
            elif field == "organization_claimed":
                update["organization_claimed"] = value
            elif field == "bank_mentioned":
                update["organization_claimed"] = value
            elif field == "upi_id":
                update["upi_id"] = value
            elif field == "phone_number":
                update["phone_number"] = value
        if update:
            update["call_sid"] = self.call_sid
            self.socketio.emit("intel_update", update, room="dashboard")
