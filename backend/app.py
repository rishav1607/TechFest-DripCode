"""Karma AI - Reverse Scam Call Agent.

Twilio + STT/TTS (Sarvam or Cartesia, configurable) + OpenRouter GPT-4o (LLM).

Modes (configurable via MODE in .env):
  - "web"    : Browser-based voice call via WebSocket
  - "twilio" : Phone call via Twilio webhooks
  - "both"   : Both interfaces active simultaneously

Live Dashboard:
  - Socket.IO broadcasts to all connected dashboard clients
  - REST APIs for analytics and call archive
  - Frontend served from /dashboard/
"""

import audioop
import base64
import io
import logging
import os
import wave

from dotenv import load_dotenv

# Load .env BEFORE importing modules that read env vars at module level
load_dotenv()

from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from flask_sock import Sock
from flask_socketio import SocketIO, emit, join_room

from conversation import ConversationManager, GREETING_TEXT
from database import (
    create_call, delete_call, end_call as db_end_call, get_active_calls,
    get_call, get_call_history, get_call_intel, get_call_transcript,
    get_stats, get_total_calls, init_db, save_intel, save_message,
)
from intel_extractor import extract_intel
from llm_service import chat_completion
from speech_service import speech_to_text, text_to_speech, get_provider_info
from twilio_stream import TwilioStreamHandler
from voice_classifier import classify_audio, is_classifier_healthy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "karma-ai-secret-key")

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    max_http_buffer_size=10 * 1024 * 1024,  # 10 MB for audio blobs
    async_mode="threading",
)

sock = Sock(app)

MODE = os.getenv("MODE", "both").lower().strip()

conversation_mgr = ConversationManager()

# Mute state per call
mute_state: dict[str, bool] = {}

RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

BASE_URL = os.getenv("BASE_URL", "http://localhost:5000")

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

# Initialize database
init_db()

# AI/Human classification: text announced when caller is detected as AI
AI_DETECTED_TEXT = "Caller is classified as AI."

# Track which web sessions have been classified already
classified_sessions: dict[str, str] = {}  # session_id → "ai" | "human"

# Pre-cache greeting TTS audio at startup (saves ~1s on first call)
_cached_greeting_twilio: bytes | None = None   # 8kHz WAV for legacy/audio endpoint
_cached_greeting_web: bytes | None = None      # 22050Hz WAV for browser
_cached_greeting_mulaw: bytes | None = None    # raw mulaw 8kHz for Twilio Media Streams
_cached_ai_detected_web: bytes | None = None   # 22050Hz WAV "Caller is classified as AI"
_cached_ai_detected_mulaw: bytes | None = None # raw mulaw 8kHz for Twilio


def _wav_to_mulaw(wav_bytes: bytes) -> bytes:
    """Extract PCM from a WAV file and convert to raw mulaw bytes."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        pcm_data = wf.readframes(wf.getnframes())
    return audioop.lin2ulaw(pcm_data, 2)


def _precache_greetings():
    global _cached_greeting_twilio, _cached_greeting_web, _cached_greeting_mulaw
    global _cached_ai_detected_web, _cached_ai_detected_mulaw
    try:
        logger.info("Pre-caching greeting TTS audio...")
        _cached_greeting_twilio = text_to_speech(
            text=GREETING_TEXT, language_code="hi-IN", speaker="kavya", sample_rate="8000",
        )
        _cached_greeting_web = text_to_speech(
            text=GREETING_TEXT, language_code="hi-IN", speaker="kavya", sample_rate="22050",
        )

        # Convert 8kHz WAV greeting to raw mulaw for Media Streams
        if _cached_greeting_twilio:
            _cached_greeting_mulaw = _wav_to_mulaw(_cached_greeting_twilio)
            logger.info(
                "Greeting audio cached (twilio=%d, web=%d, mulaw=%d bytes)",
                len(_cached_greeting_twilio),
                len(_cached_greeting_web),
                len(_cached_greeting_mulaw),
            )
        else:
            logger.info(
                "Greeting audio cached (twilio=%d, web=%d bytes)",
                len(_cached_greeting_twilio or b""),
                len(_cached_greeting_web or b""),
            )
    except Exception as e:
        logger.warning("Failed to pre-cache greeting: %s (will generate on first call)", e)

    # Pre-cache "Caller is classified as AI" announcement
    try:
        logger.info("Pre-caching AI-detected announcement audio...")
        ai_det_twilio = text_to_speech(
            text=AI_DETECTED_TEXT, language_code="en-IN", speaker="kavya", sample_rate="8000",
        )
        _cached_ai_detected_web = text_to_speech(
            text=AI_DETECTED_TEXT, language_code="en-IN", speaker="kavya", sample_rate="22050",
        )
        if ai_det_twilio:
            _cached_ai_detected_mulaw = _wav_to_mulaw(ai_det_twilio)
            logger.info("AI-detected announcement cached (mulaw=%d, web=%d bytes)",
                        len(_cached_ai_detected_mulaw), len(_cached_ai_detected_web or b""))
    except Exception as e:
        logger.warning("Failed to pre-cache AI-detected audio: %s", e)


_precache_greetings()


# ---------------------------------------------------------------------------
# Dashboard helpers — broadcast to all dashboard viewers
# ---------------------------------------------------------------------------
def broadcast_call_started(call_sid: str, caller: str = "Unknown"):
    """Notify dashboard that a new call started."""
    from datetime import datetime, timezone
    socketio.emit("call_started", {
        "call_sid": call_sid,
        "caller": caller,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, room="dashboard")


def broadcast_call_ended(call_sid: str, duration: int = 0):
    """Notify dashboard that a call ended."""
    socketio.emit("call_ended", {
        "call_sid": call_sid,
        "duration": duration,
    }, room="dashboard")


def broadcast_transcript(call_sid: str, speaker: str, text: str):
    """Send a transcript message to the dashboard."""
    from datetime import datetime, timezone
    socketio.emit("transcript_message", {
        "call_sid": call_sid,
        "speaker": speaker,  # "scammer" or "ai"
        "text": text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, room="dashboard")


def broadcast_ai_status(status: str):
    """Update AI status on dashboard."""
    socketio.emit("ai_status", {"status": status}, room="dashboard")


def broadcast_typing(call_sid: str, typing: bool):
    """Show/hide typing indicator on dashboard."""
    socketio.emit("typing_indicator", {
        "call_sid": call_sid,
        "typing": typing,
    }, room="dashboard")


def broadcast_call_list():
    """Send updated call list to all dashboard clients."""
    active = get_active_calls()
    recent = get_call_history(limit=10)
    socketio.emit("call_list_update", {
        "active_calls": active,
        "recent_calls": recent,
    }, room="dashboard")


def broadcast_intel(call_sid: str, intel_items: list[dict]):
    """Send extracted intel to dashboard."""
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
        elif field in ("bank_mentioned",):
            update["organization_claimed"] = value
        elif field == "upi_id":
            update["upi_id"] = value
        elif field == "phone_number":
            update["phone_number"] = value

    if update:
        update["call_sid"] = call_sid
        socketio.emit("intel_update", update, room="dashboard")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def process_scammer_speech(call_sid: str, speech_text: str) -> str | None:
    """Process scammer speech through LLM and intel extraction.

    Returns AI response text, or None if muted.
    """
    # Save scammer message to DB
    save_message(call_sid, "user", speech_text)

    # Broadcast to dashboard
    broadcast_ai_status("ANALYZING...")
    broadcast_transcript(call_sid, "scammer", speech_text)

    # Extract intel from scammer message
    intel_items = extract_intel(speech_text)
    for item in intel_items:
        save_intel(call_sid, item["field_name"], item["field_value"], item["confidence"])
    broadcast_intel(call_sid, intel_items)

    # Check if AI is muted for this call
    if mute_state.get(call_sid, False):
        broadcast_ai_status("MUTED")
        return None

    # Show typing indicator while AI processes
    broadcast_typing(call_sid, True)

    # Get AI response via OpenRouter GPT-4o
    messages = conversation_mgr.add_user_message(call_sid, speech_text)
    ai_response = chat_completion(messages, temperature=0.8)
    logger.info("AI response (CallSid: %s): %s", call_sid, ai_response)
    conversation_mgr.add_assistant_message(call_sid, ai_response)

    # Hide typing indicator
    broadcast_typing(call_sid, False)

    # Save AI message to DB
    save_message(call_sid, "assistant", ai_response)

    # Broadcast to dashboard
    broadcast_ai_status("DEFENDING")
    broadcast_transcript(call_sid, "ai", ai_response)

    return ai_response


# ===================================================================
#  SOCKET.IO — Connection handling (dashboard + web clients)
# ===================================================================

@socketio.on("connect")
def ws_connect():
    """Client connected — determine if dashboard or web client."""
    role = request.args.get("role", "")

    if role == "dashboard":
        join_room("dashboard")
        logger.info("Dashboard client connected: %s", request.sid)

        # Send currently active calls
        active = get_active_calls()
        for call in active:
            emit("call_started", {
                "call_sid": call["id"],
                "caller": call.get("caller_number", "Unknown"),
                "timestamp": call["start_time"],
            })
        return

    # Web mode client
    if MODE not in ("web", "both"):
        return

    session_id = request.sid
    logger.info("Web client connected: %s", session_id)
    conversation_mgr.get_or_create(session_id)

    # Register as active call in DB
    create_call(session_id, caller="web-client", mode="web")
    broadcast_call_started(session_id, "Web Client")
    broadcast_call_list()

    # Send greeting immediately so the caller hears it first
    try:
        greeting_audio = _cached_greeting_web or text_to_speech(
            text=GREETING_TEXT,
            language_code="hi-IN",
            speaker="kavya",
            sample_rate="22050",
        )
        emit("audio_response", {
            "audio": base64.b64encode(greeting_audio).decode("utf-8"),
            "text": GREETING_TEXT,
            "type": "greeting",
        })
        save_message(session_id, "assistant", GREETING_TEXT)
        broadcast_transcript(session_id, "ai", GREETING_TEXT)
    except Exception as e:
        logger.error("Error sending greeting: %s", e)
        emit("error", {"message": "Failed to generate greeting audio"})

    # Mark session as pending classification — the caller's first audio
    # response will be classified as AI or human.
    classified_sessions[session_id] = "pending"


@socketio.on("audio_data")
def ws_audio_data(data):
    """Process audio from browser microphone (web mode)."""
    if MODE not in ("web", "both"):
        return

    session_id = request.sid
    logger.info("Received audio from web client: %s", session_id)

    try:
        # Decode audio
        audio_b64 = data.get("audio", "")
        audio_bytes = base64.b64decode(audio_b64)
        audio_format = data.get("format", "wav")

        if audio_format == "webm":
            wav_bytes = _convert_webm_to_wav(audio_bytes)
        else:
            wav_bytes = audio_bytes

        # ── First audio: run AI/Human classification ──
        # Greeting was already played on connect; this is the caller's response.
        if classified_sessions.get(session_id) == "pending":
            emit("processing", {"stage": "classifying"})
            logger.info("Running voice classification for session %s", session_id)

            result = classify_audio(wav_bytes, timeout=10.0)
            prediction = result.get("prediction", "human").lower()
            confidence = result.get("confidence", 0)

            logger.info("Classification (session %s): %s (%.1f%%)",
                        session_id, prediction.upper(), confidence * 100)

            # Broadcast to dashboard
            broadcast_transcript(
                session_id, "system",
                f"Caller classified as {prediction.upper()} (confidence: {confidence:.0%})",
            )

            classified_sessions[session_id] = prediction

            if prediction == "ai":
                # Send AI-detected announcement and block further processing
                broadcast_ai_status("AI CALLER DETECTED")
                ai_audio = _cached_ai_detected_web or text_to_speech(
                    text=AI_DETECTED_TEXT,
                    language_code="en-IN",
                    speaker="kavya",
                    sample_rate="22050",
                )
                emit("transcript", {"text": AI_DETECTED_TEXT, "role": "system"})
                emit("audio_response", {
                    "audio": base64.b64encode(ai_audio).decode("utf-8"),
                    "text": AI_DETECTED_TEXT,
                    "type": "ai_detected",
                })
                return

            # Human — fall through to process this first audio normally

        # ── If caller was classified as AI, reject further audio ──
        if classified_sessions.get(session_id) == "ai":
            emit("error", {"message": "Caller classified as AI — conversation blocked."})
            return

        # ── Normal processing pipeline ──

        # Step 1: STT
        emit("processing", {"stage": "stt"})
        transcript = speech_to_text(wav_bytes, language_code="hi-IN")
        logger.info("STT result (session %s): %s", session_id, transcript)

        if not transcript.strip():
            emit("error", {"message": "Sunai nahi diya... please phir se bolo!"})
            return

        emit("transcript", {"text": transcript, "role": "user"})

        # Step 2: LLM + intel extraction
        emit("processing", {"stage": "thinking"})
        ai_response = process_scammer_speech(session_id, transcript)

        if ai_response is None:
            emit("error", {"message": "AI is muted"})
            return

        emit("transcript", {"text": ai_response, "role": "assistant"})

        # Step 3: TTS
        emit("processing", {"stage": "tts"})
        response_audio = text_to_speech(
            text=ai_response,
            language_code="hi-IN",
            speaker="kavya",
            sample_rate="22050",
        )

        emit("audio_response", {
            "audio": base64.b64encode(response_audio).decode("utf-8"),
            "text": ai_response,
            "type": "response",
        })

    except Exception as e:
        logger.error("Error processing web audio (session %s): %s", session_id, e)
        emit("error", {"message": f"Error: {str(e)}"})


@socketio.on("end_call")
def ws_end_call():
    """Client ended the call."""
    session_id = request.sid
    logger.info("Web client ending call: %s", session_id)
    conversation_mgr.end_conversation(session_id)
    mute_state.pop(session_id, None)
    classified_sessions.pop(session_id, None)
    db_end_call(session_id, "completed")
    broadcast_call_ended(session_id)
    broadcast_call_list()
    emit("call_ended", {"message": "Call ended. Phir milenge!"})


@socketio.on("disconnect")
def ws_disconnect():
    """Client disconnected."""
    session_id = request.sid
    role = request.args.get("role", "")

    if role == "dashboard":
        logger.info("Dashboard client disconnected: %s", session_id)
        return

    logger.info("Web client disconnected: %s", session_id)
    conversation_mgr.end_conversation(session_id)
    mute_state.pop(session_id, None)
    classified_sessions.pop(session_id, None)
    db_end_call(session_id, "completed")
    broadcast_call_ended(session_id)
    broadcast_call_list()


# Dashboard control events
@socketio.on("mute_ai")
def ws_mute_ai(data):
    """Toggle AI mute for a call."""
    call_sid = data.get("call_sid")
    muted = data.get("muted", False)
    if call_sid:
        mute_state[call_sid] = muted
        logger.info("AI %s for call %s", "muted" if muted else "unmuted", call_sid)
        broadcast_ai_status("MUTED" if muted else "ACTIVE")


@socketio.on("drop_call")
def ws_drop_call(data):
    """Drop an active call from dashboard."""
    call_sid = data.get("call_sid")
    if not call_sid:
        return

    logger.info("Dashboard dropping call: %s", call_sid)

    # For Twilio calls, try to end the call via API
    if MODE in ("twilio", "both") and call_sid.startswith("CA"):
        try:
            from twilio.rest import Client
            client = Client(
                os.getenv("TWILIO_ACCOUNT_SID"),
                os.getenv("TWILIO_AUTH_TOKEN"),
            )
            client.calls(call_sid).update(status="completed")
        except Exception as e:
            logger.error("Failed to drop Twilio call: %s", e)

    conversation_mgr.end_conversation(call_sid)
    mute_state.pop(call_sid, None)
    db_end_call(call_sid, "dropped")
    broadcast_call_ended(call_sid)
    broadcast_call_list()


def _convert_webm_to_wav(webm_bytes: bytes) -> bytes:
    """Convert WebM/opus audio to WAV format."""
    from pydub import AudioSegment
    audio = AudioSegment.from_file(io.BytesIO(webm_bytes), format="webm")
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    return wav_buffer.read()


# ===================================================================
#  TWILIO MODE — Bidirectional Media Streams (low-latency)
# ===================================================================

if MODE in ("twilio", "both"):
    from twilio.twiml.voice_response import Connect, Stream, VoiceResponse


@app.route("/voice", methods=["POST"])
def voice():
    """Twilio webhook: incoming call → start a bidirectional media stream."""
    if MODE not in ("twilio", "both"):
        return "Twilio mode not enabled", 404

    call_sid = request.form.get("CallSid", "unknown")
    caller = request.form.get("From", "unknown")
    logger.info("Incoming call from %s (CallSid: %s)", caller, call_sid)

    # Set up conversation, DB record, and dashboard broadcast
    conversation_mgr.get_or_create(call_sid)
    create_call(call_sid, caller=caller, mode="twilio")
    broadcast_call_started(call_sid, caller)
    broadcast_call_list()
    save_message(call_sid, "assistant", GREETING_TEXT)
    broadcast_transcript(call_sid, "ai", GREETING_TEXT)

    # Build TwiML: open a bidirectional media stream to our WebSocket
    response = VoiceResponse()
    connect = Connect()
    stream = Stream(url=f"wss://{request.host}/media-stream")
    stream.parameter(name="caller", value=caller)
    connect.append(stream)
    response.append(connect)

    return Response(str(response), mimetype="application/xml")


@sock.route("/media-stream")
def media_stream(ws):
    """WebSocket endpoint for Twilio bidirectional media streams.

    Audio flows in both directions over this single connection:
      Twilio → mulaw 8kHz → VAD → STT → LLM → Cartesia TTS → mulaw → Twilio
    """
    handler = TwilioStreamHandler(
        ws,
        socketio=socketio,
        conversation_mgr=conversation_mgr,
        mute_state=mute_state,
        greeting_mulaw=_cached_greeting_mulaw,
        ai_detected_mulaw=_cached_ai_detected_mulaw,
    )
    handler.run()


@app.route("/call-status", methods=["POST"])
def call_status():
    """Twilio webhook: called when call status changes."""
    call_sid = request.form.get("CallSid", "unknown")
    status = request.form.get("CallStatus", "unknown")
    logger.info("Call status update (CallSid: %s): %s", call_sid, status)

    if status in ("completed", "failed", "busy", "no-answer", "canceled"):
        conversation_mgr.end_conversation(call_sid)
        mute_state.pop(call_sid, None)

        # Update DB and broadcast
        db_end_call(call_sid, status)
        broadcast_call_ended(call_sid)
        broadcast_call_list()

    return "", 200


# ===================================================================
#  WEB MODE — Serve the browser voice-call UI
# ===================================================================

@app.route("/")
def index():
    """Serve the web voice-call interface."""
    if MODE not in ("web", "both"):
        return "Web mode not enabled. Set MODE=web or MODE=both in .env", 404
    return render_template("index.html")


# ===================================================================
#  REST API — Stats, calls, archive for frontend dashboards
# ===================================================================

@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Aggregate stats for analytics dashboard."""
    stats = get_stats()
    return jsonify(stats)


@app.route("/api/calls", methods=["GET"])
def api_calls():
    """Paginated call history for archive."""
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    calls = get_call_history(limit, offset)
    total = get_total_calls()
    return jsonify({"calls": calls, "total": total, "limit": limit, "offset": offset})


@app.route("/api/calls/<call_id>/transcript", methods=["GET"])
def api_call_transcript(call_id: str):
    """Full transcript + intel for a specific call."""
    call = get_call(call_id)
    if not call:
        return jsonify({"error": "Call not found"}), 404

    messages = get_call_transcript(call_id)
    intel = get_call_intel(call_id)

    return jsonify({
        "call": call,
        "messages": messages,
        "intel": intel,
    })


@app.route("/api/calls/<call_id>/summary", methods=["GET"])
def api_call_summary(call_id: str):
    """AI-generated summary for a call."""
    messages = get_call_transcript(call_id)
    if not messages:
        return jsonify({"error": "No transcript found"}), 404

    intel = get_call_intel(call_id)

    # Build summary prompt
    transcript_text = "\n".join(
        f"{'Scammer' if m['role'] == 'user' else 'AI Dadi'}: {m['content']}"
        for m in messages
    )

    summary_messages = [
        {
            "role": "system",
            "content": (
                "You are a scam analyst. Summarize this scam call transcript. "
                "Include: scammer's tactics, information extracted, how AI wasted their time, "
                "and risk assessment. Keep it concise (3-5 bullet points). Respond in English."
            ),
        },
        {"role": "user", "content": f"Transcript:\n{transcript_text}"},
    ]

    try:
        summary = chat_completion(summary_messages, temperature=0.3)
        return jsonify({"summary": summary, "intel": intel})
    except Exception as e:
        logger.error("Error generating summary: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/calls/<call_id>/analysis", methods=["GET"])
def api_call_analysis(call_id: str):
    """Deep AI analysis of a scam call — returns structured JSON dossier."""
    messages = get_call_transcript(call_id)
    if not messages:
        return jsonify({"error": "No transcript found"}), 404

    call = get_call(call_id)
    intel = get_call_intel(call_id)

    # Build transcript text
    transcript_text = "\n".join(
        f"{'Scammer' if m['role'] == 'user' else 'AI Dadi'}: {m['content']}"
        for m in messages
        if m['role'] != 'system'
    )

    # Include already-extracted intel for context
    intel_context = ""
    if intel:
        intel_context = "\n\nAlready extracted intel from regex:\n" + "\n".join(
            f"- {i['field_name']}: {i['field_value']} (confidence: {i['confidence']})"
            for i in intel
        )

    analysis_prompt = [
        {
            "role": "system",
            "content": (
                "You are a scam call analyst for Karma AI, a system that intercepts scam calls with an AI grandmother persona.\n"
                "Analyze the following scam call transcript and return a JSON object with this EXACT structure (no markdown, no explanation, ONLY valid JSON):\n\n"
                "{\n"
                '  "scammer_profile": {\n'
                '    "name": "name if mentioned, otherwise Unknown",\n'
                '    "organization_claimed": "org they claim to represent",\n'
                '    "phone_number": "if visible",\n'
                '    "location_hints": "any location clues from speech patterns or mentions"\n'
                "  },\n"
                '  "scam_analysis": {\n'
                '    "type": "KYC Fraud / Lottery / Bank Impersonation / Tech Support / Insurance / Refund / OTP / UPI / Other",\n'
                '    "tactics_used": ["list", "of", "tactics"],\n'
                '    "threat_level": "HIGH or MEDIUM or LOW",\n'
                '    "sophistication": "HIGH or MEDIUM or LOW"\n'
                "  },\n"
                '  "extracted_data": {\n'
                '    "upi_ids": [],\n'
                '    "phone_numbers": [],\n'
                '    "bank_accounts": [],\n'
                '    "aadhaar_numbers": [],\n'
                '    "banks_mentioned": []\n'
                "  },\n"
                '  "call_metrics": {\n'
                '    "messages_exchanged": 0,\n'
                '    "scammer_frustration_level": "LOW or MEDIUM or HIGH or EXTREME",\n'
                '    "time_wasted_effectively": true\n'
                "  },\n"
                '  "summary": "2-3 sentence English summary of the call",\n'
                '  "key_moments": ["moment 1", "moment 2", "moment 3"]\n'
                "}\n\n"
                "Rules:\n"
                "- Respond with ONLY the JSON object, nothing else\n"
                "- Fill every field based on the transcript\n"
                "- For missing data, use empty strings or empty arrays\n"
                "- Tactics include: urgency, authority impersonation, fear, fake deadlines, emotional manipulation, technical jargon, etc.\n"
                "- Frustration level: judge from scammer's tone/caps/repetition/threats\n"
                "- Key moments: 3-5 most important events in the call"
            ),
        },
        {
            "role": "user",
            "content": f"Transcript:\n{transcript_text}{intel_context}",
        },
    ]

    import json as json_mod

    try:
        raw = chat_completion(analysis_prompt, temperature=0.2)

        # Parse JSON from LLM response (strip markdown fences if present)
        json_str = raw.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("\n", 1)[1] if "\n" in json_str else json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()

        analysis = json_mod.loads(json_str)

        # Enrich with DB data
        if call:
            analysis["call_metrics"]["duration_seconds"] = call.get("duration_seconds", 0)
            analysis["call_metrics"]["call_mode"] = call.get("mode", "unknown")
            analysis["call_metrics"]["caller_number"] = call.get("caller_number", "unknown")

        analysis["call_metrics"]["messages_exchanged"] = len(
            [m for m in messages if m["role"] != "system"]
        )

        # Merge DB intel into extracted_data
        for item in intel:
            fn = item["field_name"]
            fv = item["field_value"]
            if fn == "upi_id" and fv not in analysis["extracted_data"].get("upi_ids", []):
                analysis["extracted_data"].setdefault("upi_ids", []).append(fv)
            elif fn == "phone_number" and fv not in analysis["extracted_data"].get("phone_numbers", []):
                analysis["extracted_data"].setdefault("phone_numbers", []).append(fv)
            elif fn == "account_number" and fv not in analysis["extracted_data"].get("bank_accounts", []):
                analysis["extracted_data"].setdefault("bank_accounts", []).append(fv)
            elif fn == "aadhaar_number" and fv not in analysis["extracted_data"].get("aadhaar_numbers", []):
                analysis["extracted_data"].setdefault("aadhaar_numbers", []).append(fv)
            elif fn == "bank_mentioned" and fv not in analysis["extracted_data"].get("banks_mentioned", []):
                analysis["extracted_data"].setdefault("banks_mentioned", []).append(fv)

        return jsonify({"analysis": analysis, "intel": intel})

    except json_mod.JSONDecodeError:
        # LLM didn't return valid JSON — return raw text as summary fallback
        return jsonify({"analysis": None, "raw_summary": raw, "intel": intel})
    except Exception as e:
        logger.error("Error generating analysis: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/active-calls", methods=["GET"])
def api_active_calls():
    """List currently active calls."""
    active = get_active_calls()
    return jsonify({"calls": active})


@app.route("/api/calls/<call_id>", methods=["DELETE"])
def api_delete_call(call_id: str):
    """Delete a call and all its messages/intel from the database."""
    call = get_call(call_id)
    if not call:
        return jsonify({"error": "Call not found"}), 404

    delete_call(call_id)
    logger.info("Deleted call %s and all related data", call_id)
    return jsonify({"ok": True, "deleted": call_id})


# ===================================================================
#  Serve frontend dashboard static files
# ===================================================================

@app.route("/dashboard/")
@app.route("/dashboard/<path:filename>")
def serve_frontend(filename="index.html"):
    """Serve frontend files from the frontend directory."""
    return send_from_directory(FRONTEND_DIR, filename)


# ===================================================================
#  Health check
# ===================================================================

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "Karma AI",
        "mode": MODE,
        "active_conversations": len(conversation_mgr.conversations),
        "voice_classifier": "available" if is_classifier_healthy() else "unavailable",
    })


# ===================================================================
#  Run
# ===================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    pinfo = get_provider_info()
    classifier_ok = is_classifier_healthy()
    logger.info("=" * 50)
    logger.info("  KARMA AI — Reverse Scam Call Agent")
    logger.info("  Mode: %s", MODE.upper())
    logger.info("  Twilio: Bidirectional Media Streams (low-latency)")
    logger.info("  LLM: x-ai/grok-4.1-fast (OpenRouter)")
    logger.info("  STT: %s (%s)", pinfo["stt_provider"].upper(), pinfo["stt_model"])
    logger.info("  TTS: %s (%s) [WebSocket streaming]", pinfo["tts_provider"].upper(), pinfo["tts_model"])
    logger.info("  Voice Classifier: %s", "AVAILABLE (port 8000)" if classifier_ok else "UNAVAILABLE (all callers treated as human)")
    logger.info("=" * 50)

    if MODE in ("web", "both"):
        logger.info("Web interface: http://localhost:%d", port)
    if MODE in ("twilio", "both"):
        logger.info("Twilio webhook: %s/voice", BASE_URL)
    logger.info("Dashboard: http://localhost:%d/dashboard/live-calls.html", port)
    logger.info("Analytics: http://localhost:%d/dashboard/analytics.html", port)
    logger.info("Archive:   http://localhost:%d/dashboard/archive.html", port)

    socketio.run(app, host="0.0.0.0", port=port, debug=True, allow_unsafe_werkzeug=True)
