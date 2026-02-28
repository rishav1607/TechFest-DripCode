# Karma AI

**A reverse scam call agent that wastes scammers' time.**

Karma AI impersonates a 75-year-old Indian grandmother ("Kamla Devi") who speaks in Hinglish, pretends to be confused, fumbles OTPs, tells stories about her grandchildren, and keeps scammers on the line as long as possible — extracting their identity information along the way.

Supports **browser voice calls** (Web Mode) and **real phone calls** (Twilio Mode) simultaneously. Includes a real-time dashboard for monitoring live calls, reviewing transcripts, and viewing extracted scammer intelligence.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running](#running)
- [The Dadi Persona](#the-dadi-persona)
- [Voice Classification (AI/Human Detection)](#voice-classification-aihuman-detection)
- [Dashboard](#dashboard)
- [API Reference](#api-reference)
- [Database Schema](#database-schema)
- [Tech Stack](#tech-stack)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## How It Works

### Web Mode (Browser Voice Call)

No phone number needed. Talk to the AI dadi directly from your browser.

```
┌─────────────┐     Socket.IO      ┌──────────────────────────────────┐
│   Browser    │ ◄────────────────► │         Flask Server             │
│              │                    │                                  │
│  Hold mic    │  audio (base64)    │  1. Voice Classification (AI?)   │
│  and speak   │ ──────────────►    │  2. Sarvam STT (saaras:v3)       │
│              │                    │  3. Intel Extraction (regex)      │
│  Audio plays │  audio (base64)    │  4. OpenRouter LLM (streaming)   │
│  back        │ ◄──────────────    │  5. Cartesia TTS (Sonic 3)       │
│              │                    │  6. Dashboard broadcast          │
└─────────────┘                    └──────────────────────────────────┘
```

1. Open `http://localhost:5000` and click **Start Call**
2. Dadi greets you — *"Haaaan? Hello? Kaun bol raha hai?"*
3. Hold the **mic button** (or **Space bar**) and speak
4. Release to send — audio is encoded to WAV entirely in the browser
5. Server transcribes (Sarvam STT), generates response (OpenRouter LLM), converts to speech (Cartesia TTS)
6. Dadi's voice plays back. Repeat.

### Twilio Mode (Phone Call)

Real phone number that scammers can call. Uses **bidirectional Media Streams** for low-latency, real-time audio.

```
┌──────────┐   PSTN    ┌──────────┐  WebSocket   ┌──────────────────────┐
│  Scammer │ ────────► │  Twilio  │ ◄──────────► │    Flask Server      │
│  Phone   │           │  Cloud   │   mulaw 8kHz  │                      │
│          │ ◄──────── │          │    bidir      │  VAD → STT → LLM    │
│          │   Audio   │          │               │  → TTS → mulaw back  │
└──────────┘           └──────────┘               └──────────────────────┘
```

1. Scammer calls the Twilio number
2. Twilio opens a WebSocket Media Stream to the server
3. Raw mulaw audio flows in both directions at 8kHz
4. Server runs the full pipeline: **VAD** (voice activity detection) → **Sarvam STT** → **OpenRouter LLM** (streaming) → **Cartesia TTS** (streaming mulaw) → back to Twilio
5. Cartesia outputs `pcm_mulaw` at 8kHz — byte-for-byte compatible with Twilio, zero transcoding needed
6. Streaming TTS means dadi starts speaking as soon as the first sentence is ready

### Voice Classification Gate

Before any conversation begins, the system collects ~3.5 seconds of the caller's voice and runs it through a fine-tuned **WAV2Vec2** model to detect AI-generated speech. If the caller is classified as AI (robocall), the system announces this and terminates. Only human callers proceed to the full conversation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FLASK BACKEND                              │
│                                                                     │
│  ┌────────────┐   ┌──────────────┐   ┌──────────────┐              │
│  │  Twilio     │   │ Conversation │   │  OpenRouter   │              │
│  │  Media      │──►│   Manager    │──►│  LLM          │              │
│  │  Streams    │   │  (per-call)  │   │  (streaming)  │              │
│  └────────────┘   └──────────────┘   └──────────────┘              │
│        │                 │                   │                       │
│        ▼                 ▼                   ▼                       │
│  ┌────────────┐   ┌──────────────┐   ┌──────────────┐              │
│  │  Sarvam    │   │   SQLite     │   │  Cartesia    │              │
│  │  STT       │   │  Database    │   │  TTS         │              │
│  │ (saaras:v3)│   │ calls/msgs/  │   │ (Sonic 3)    │              │
│  └────────────┘   │    intel     │   └──────────────┘              │
│        │          └──────────────┘          │                       │
│        │                 │                  │                       │
│        ▼                 ▼                  ▼                       │
│  ┌────────────┐   ┌──────────────┐   ┌──────────────┐              │
│  │  Intel     │   │  Socket.IO   │   │  Voice       │              │
│  │  Extractor │   │  Broadcast   │   │  Classifier  │              │
│  │  (regex)   │   │ (dashboard)  │   │  (WAV2Vec2)  │              │
│  └────────────┘   └──────────────┘   └──────────────┘              │
│                          │                                          │
│            ┌─────────────┼──────────────┐                          │
│            ▼             ▼              ▼                           │
│      REST APIs      WebSocket      Static Files                    │
│     /api/stats    /transcript     /dashboard/*                     │
│     /api/calls    /call_started   /                                │
└─────────────────────────────────────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  Live    │ │Analytics │ │ Archive  │
        │  Calls   │ │Dashboard │ │  Page    │
        │Dashboard │ │          │ │          │
        └──────────┘ └──────────┘ └──────────┘
```

---

## Project Structure

```
karmaai2/
├── backend/
│   ├── app.py                    # Main Flask server — routes, Socket.IO, WebSocket
│   ├── twilio_stream.py          # Twilio Media Streams handler (bidirectional pipeline)
│   ├── conversation.py           # ConversationManager + 362-line dadi system prompt
│   ├── llm_service.py            # OpenRouter LLM (streaming chat completion)
│   ├── cartesia_service.py       # Cartesia TTS (WebSocket streaming) + STT
│   ├── sarvam_service.py         # Sarvam AI STT/TTS wrapper
│   ├── speech_service.py         # Provider router (swap Sarvam/Cartesia without code changes)
│   ├── database.py               # SQLite persistence (calls, messages, intel tables)
│   ├── intel_extractor.py        # Regex-based scammer info extraction (UPI, phone, bank, etc.)
│   ├── voice_classifier.py       # AI/human voice classification client
│   ├── start.py                  # Auto-startup (classifier + ngrok + Twilio + Flask)
│   ├── setup_twilio.py           # Programmatic Twilio webhook configuration
│   ├── test_sarvam.py            # API connectivity test
│   ├── requirements.txt          # Python dependencies
│   ├── .env                      # API keys & config (not committed)
│   ├── karma.db                  # SQLite database (auto-created)
│   └── templates/
│       └── index.html            # Web voice call UI (single-file, no build tools)
│
├── frontend/
│   ├── index.html                # Landing page (animated hero, features, FAQ)
│   ├── live-calls.html           # Real-time call monitoring dashboard
│   ├── live-calls.js             # Dashboard Socket.IO logic
│   ├── analytics.html            # Statistics & charts
│   ├── archive.html              # Call history archive
│   ├── style.css                 # Landing page styles
│   ├── dashboard.css             # Dashboard styles
│   ├── script.js                 # Landing page animations (GSAP)
│   └── logo.png                  # Karma logo
│
├── Dataset/
│   ├── api.py                    # FastAPI voice classifier service (port 8000)
│   ├── inference.py              # WAV2Vec2 model loading & prediction
│   ├── train.py                  # Model fine-tuning script
│   ├── collect_dataset.py        # Audio dataset collection
│   ├── analyzers.py              # Multi-signal analysis (WavLM, spectral, prosody)
│   ├── requirements.txt          # ML dependencies (PyTorch, transformers, etc.)
│   ├── dataset/                  # Audio samples (train/test, human/ai)
│   └── model_output/             # Fine-tuned WAV2Vec2 weights (96% accuracy)
│
├── plan.md                       # Implementation roadmap (completed)
└── README.md                     # This file
```

---

## Setup

### Prerequisites

- **Python 3.10+** (tested on 3.13/3.15)
- API keys for:
  - [Sarvam AI](https://dashboard.sarvam.ai) — Hindi/Hinglish speech-to-text
  - [Cartesia](https://cartesia.ai) — text-to-speech (Sonic 3)
  - [OpenRouter](https://openrouter.ai) — LLM (GPT-4o, Groq, etc.)
- **Twilio account** (only for phone mode) — [console.twilio.com](https://console.twilio.com)
- **ngrok** (only for phone mode) — [ngrok.com](https://ngrok.com)
- **NVIDIA GPU** (only for voice classifier) — WAV2Vec2 inference

### 1. Clone & Install Backend

```bash
cd karmaai2/backend

# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create `backend/.env`:

```env
# === Mode ===
MODE=both                    # "web", "twilio", or "both"

# === Speech Providers ===
TTS_PROVIDER=cartesia        # "cartesia" or "sarvam"
STT_PROVIDER=sarvam          # "sarvam" or "cartesia"

# === Sarvam AI (STT) ===
SARVAM_API_KEY=your_sarvam_api_key

# === Cartesia (TTS) ===
CARTESIA_API_KEY=your_cartesia_api_key
CARTESIA_VOICE_ID=your_voice_id    # Hindi female voice

# === OpenRouter (LLM) ===
OPENROUTER_API_KEY=your_openrouter_api_key
LLM_MODEL=openai/gpt-oss-120b     # or x-ai/grok-4.1-fast, openai/gpt-4o, etc.

# === Server ===
PORT=5000

# === Twilio (phone mode only) ===
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_PHONE_NUMBER=+1234567890
BASE_URL=https://xxxx.ngrok-free.app
NGROK_AUTHTOKEN=your_ngrok_token
```

| Variable | Required | Description |
|----------|----------|-------------|
| `MODE` | Yes | `web`, `twilio`, or `both` |
| `SARVAM_API_KEY` | Yes | Sarvam AI API key (STT) |
| `CARTESIA_API_KEY` | Yes | Cartesia API key (TTS) |
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key (LLM) |
| `TTS_PROVIDER` | No | `cartesia` (default) or `sarvam` |
| `STT_PROVIDER` | No | `sarvam` (default) or `cartesia` |
| `LLM_MODEL` | No | OpenRouter model ID (default: `openai/gpt-oss-120b`) |
| `PORT` | No | Server port (default: `5000`) |
| `TWILIO_*` | Twilio only | Twilio credentials and phone number |
| `BASE_URL` | Twilio only | Public ngrok URL for webhooks |
| `NGROK_AUTHTOKEN` | Twilio only | ngrok auth token |

### 3. Set Up Voice Classifier (Optional)

The voice classifier detects AI-generated callers and rejects them. Skip this if you don't need it — the system defaults to assuming human callers when the classifier is unavailable.

```bash
cd karmaai2/Dataset

# Create separate venv (ML dependencies are heavy)
python -m venv venv
venv\Scripts\activate    # or source venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt
```

If `model_output/best_model/` doesn't exist, train the model:

```bash
python train.py    # ~5-10 minutes on RTX 4060
```

### 4. Test API Connectivity (Optional)

```bash
cd karmaai2/backend
python test_sarvam.py
```

---

## Running

### Quick Start — Web Mode Only

```bash
cd karmaai2/backend
python app.py
```

Open **http://localhost:5000**. Click the green button. Hold mic and speak.

### Full Start — With Twilio + Voice Classifier

The auto-startup script handles everything:

```bash
cd karmaai2/backend
python start.py
```

This automatically:
1. Starts the voice classifier API on port 8000
2. Starts an ngrok tunnel on port 5000
3. Updates `BASE_URL` in `.env` with the ngrok URL
4. Configures Twilio webhooks programmatically
5. Launches the Flask server

### Manual Start — Twilio Mode

If you prefer manual control:

```bash
# Terminal 1: Voice classifier (optional)
cd karmaai2/Dataset
venv\Scripts\activate
python api.py

# Terminal 2: ngrok tunnel
ngrok http 5000

# Terminal 3: Configure Twilio webhooks
cd karmaai2/backend
python setup_twilio.py https://xxxx.ngrok-free.app

# Terminal 4: Flask server
cd karmaai2/backend
python app.py
```

### Accessing the Dashboard

Once the server is running:

| URL | Page |
|-----|------|
| `http://localhost:5000` | Web voice call interface |
| `http://localhost:5000/dashboard/` | Landing page |
| `http://localhost:5000/dashboard/live-calls.html` | Live call monitoring |
| `http://localhost:5000/dashboard/analytics.html` | Statistics & charts |
| `http://localhost:5000/dashboard/archive.html` | Call history |

---

## The Dadi Persona

The AI persona is defined in a **362-line system prompt** in `conversation.py`. She is Kamla Devi, a 75-year-old grandmother from Ranchi.

### Personality

- Hard of hearing, easily confused, warm and loving
- Speaks in colloquial Hindi with English code-mixing (natural Hinglish)
- References family: son Rahul, grandson Guddu, daughter-in-law Sunita, neighbor Sharma ji
- Gets excited when offered something — *"Arre waah! Sach mein?!"*
- Goes off on completely irrelevant tangents

### Scam-Baiting Tactics

| Tactic | Example |
|--------|---------|
| **Repeat requests** | *"Kya? Sunai nahi diya beta"* — makes them repeat everything |
| **Wrong OTPs** | Fumbles digits — swaps adjacent numbers, gets "corrected", wastes 2-3 minutes |
| **Chai breaks** | *"Ruko beta, chai rakh ke aati hoon"* — disappears to make tea |
| **Story time** | Launches into long stories about Guddu's exam or Sharma uncle's health |
| **Tech confusion** | *"Mera phone mein ye kaise karte hain?"* — pretends phones are magic |
| **Guilt trips** | *"Beta gussa mat ho, BP badh jaayega"* — when scammer gets frustrated |
| **The "Almost There" loop** | Always one step from completing — *"Bas ho gaya almost..."* — but never finishes |

### Conversation Phases

1. **Trust Building** (first ~2 min) — Warm, cooperative, no tactics
2. **Gentle Extraction** (3-8 min) — Asks scammer's name, bank, branch, phone
3. **Slow Compliance** (8-15 min) — Fumbles numbers, pretends to struggle with apps
4. **Gentle Complications** (15+ min) — Small problems that reset progress

### Safety Rules

- Never gives real OTP, PIN, CVV, or password
- Never confirms a transaction completed
- Never agrees to send money or install an app successfully
- All bank details are fabricated
- Keeps responses to 1-3 short sentences (natural phone style)

---

## Voice Classification (AI/Human Detection)

The `Dataset/` module contains a fine-tuned **WAV2Vec2** model that detects AI-generated speech with **96% accuracy**.

### How It Works

```
Caller audio (3.5 sec)
   ├── wav2vec2 (fine-tuned)          → Primary AI probability
   ├── WavLM embedding analysis       → Hidden state pattern analysis
   ├── Spectral artifact detection    → Frequency/phase anomalies
   └── Prosody analysis               → Pitch jitter, shimmer, rhythm
                    ↓
         Ensemble Classifier (confidence-gated)
                    ↓
         "human" → proceed with call
         "ai"    → announce and terminate
```

### Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | **96.00%** |
| Test F1 Score | **95.99%** |
| Human Precision | 100% |
| AI Recall | 100% |

### Training Data

- **200 samples** (100 human + 100 AI), bilingual (English + Hindi)
- Human audio from [Google FLEURS](https://huggingface.co/datasets/google/fleurs)
- AI audio generated with [Microsoft Edge TTS](https://github.com/rany2/edge-tts)
- Model: `facebook/wav2vec2-base` fine-tuned for 10 epochs on RTX 4060

### Standalone Usage

```bash
cd Dataset
python inference.py path/to/audio.wav
```

### REST API

The classifier runs as a FastAPI service on port 8000:

```bash
# Start
python api.py

# Classify audio file
curl -X POST -F "file=@audio.wav" http://localhost:8000/predict

# Health check
curl http://localhost:8000/health
```

---

## Dashboard

### Live Calls (`live-calls.html`)

Real-time call monitoring via Socket.IO:

- **Active call list** with caller info and duration timer
- **Live transcript** — WhatsApp-style chat bubbles (scammer left, AI right)
- **Intel panel** — displays extracted scammer name, bank, UPI ID, phone number
- **AI status indicator** — CLASSIFYING / DEFENDING / ANALYZING
- **Mute AI** — temporarily silence dadi's responses
- **Drop Call** — terminate a call from the dashboard

### Analytics (`analytics.html`)

- Total calls (24H), detection accuracy, threat level, average latency
- Scam frequency chart (last 7 days)
- Success rate donut chart (neutralized / blocked / escaped)
- Pattern detection alerts

### Archive (`archive.html`)

- Paginated call history with search
- Per-call detail view: full transcript, extracted intel
- AI-generated call summaries
- Deep scam analysis (JSON dossier)

---

## API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /` | GET | Web voice call interface |
| `GET /health` | GET | Server health (status, mode, active calls, classifier status) |
| `GET /dashboard/` | GET | Dashboard landing page |
| `GET /dashboard/<file>` | GET | Dashboard static files |
| `GET /api/stats` | GET | Aggregate statistics |
| `GET /api/calls` | GET | Paginated call history (`?limit=&offset=`) |
| `GET /api/calls/<id>/transcript` | GET | Full transcript + intel for a call |
| `GET /api/calls/<id>/summary` | GET | AI-generated call summary |
| `GET /api/calls/<id>/analysis` | GET | Deep scam analysis (JSON) |
| `GET /api/active-calls` | GET | Currently active calls |
| `DELETE /api/calls/<id>` | DELETE | Delete a call and its data |

### Twilio Webhooks

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /voice` | POST | Incoming call — starts Media Stream |
| `GET /media-stream` | WebSocket | Bidirectional audio streaming |
| `POST /call-status` | POST | Call lifecycle events (hangup cleanup) |

### Socket.IO Events — Web Voice Call

| Direction | Event | Payload |
|-----------|-------|---------|
| Client → Server | `audio_data` | `{audio: "<base64>", format: "wav"}` |
| Client → Server | `end_call` | — |
| Server → Client | `audio_response` | `{audio: "<base64>", text, type}` |
| Server → Client | `transcript` | `{text, role: "user"\|"assistant"}` |
| Server → Client | `processing` | `{stage: "stt"\|"thinking"\|"tts"}` |
| Server → Client | `error` | `{message}` |
| Server → Client | `call_ended` | `{message}` |

### Socket.IO Events — Dashboard

| Direction | Event | Payload |
|-----------|-------|---------|
| Server → Client | `call_started` | `{call_sid, caller, timestamp}` |
| Server → Client | `call_ended` | `{call_sid, duration}` |
| Server → Client | `transcript_message` | `{call_sid, speaker, text, timestamp}` |
| Server → Client | `ai_status` | `{status}` |
| Server → Client | `intel_update` | `{call_sid, scammer_name, ...}` |
| Client → Server | `mute_ai` | `{call_sid, muted}` |
| Client → Server | `drop_call` | `{call_sid}` |

---

## Database Schema

SQLite database (`karma.db`), auto-created on first run.

### `calls`

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | CallSid or session ID |
| `caller_number` | TEXT | Phone number or "web-client" |
| `start_time` | TEXT | ISO 8601 timestamp |
| `end_time` | TEXT | Set on hangup |
| `duration_seconds` | INTEGER | Calculated on end |
| `status` | TEXT | `active`, `completed`, `failed`, `dropped` |
| `mode` | TEXT | `twilio` or `web` |
| `threat_level` | TEXT | Risk assessment (default: `HIGH`) |

### `messages`

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `call_id` | TEXT FK | References `calls(id)` |
| `role` | TEXT | `user` (scammer) or `assistant` (AI) |
| `content` | TEXT | Message text |
| `timestamp` | TEXT | Auto-set |

### `intel`

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment |
| `call_id` | TEXT FK | References `calls(id)` |
| `field_name` | TEXT | `scammer_name`, `upi_id`, `bank_mentioned`, `phone_number`, `account_number`, `aadhaar_number`, `scam_type`, `organization_claimed` |
| `field_value` | TEXT | Extracted value |
| `confidence` | REAL | 0.0–1.0 |
| `timestamp` | TEXT | Auto-set |

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | Flask 3.1 + Flask-SocketIO + Flask-Sock | HTTP, WebSocket, real-time events |
| **LLM** | OpenRouter (GPT-4o / Groq / configurable) | Conversational AI (dadi responses) |
| **STT** | Sarvam AI (saaras:v3) | Hindi/Hinglish speech-to-text |
| **TTS** | Cartesia (Sonic 3) | Text-to-speech with WebSocket streaming |
| **TTS (alt)** | Sarvam AI (bulbul:v3) | Alternative TTS provider |
| **Telephony** | Twilio Media Streams | Real phone call handling |
| **Voice Classifier** | WAV2Vec2 + FastAPI | AI/human voice detection (96% accuracy) |
| **VAD** | webrtcvad (mode 3) | Voice activity detection for utterance segmentation |
| **Database** | SQLite 3 | Call history, transcripts, intelligence |
| **Intel Extraction** | Regex patterns | UPI IDs, phone numbers, banks, scam types |
| **Tunneling** | ngrok | Public webhook URLs for Twilio |
| **Frontend** | Vanilla HTML/CSS/JS | No build tools, single-file voice UI |
| **Dashboard** | HTML/CSS/JS + Socket.IO + GSAP | Real-time monitoring, analytics, archive |
| **Audio** | Web Audio API + pydub + audioop-lts | Browser recording, format conversion |

### Key Design Decisions

- **Provider abstraction**: `speech_service.py` routes between Sarvam and Cartesia — swap providers via env var without code changes
- **Streaming pipeline**: Sentence-level TTS streaming (start speaking before full LLM response completes)
- **Pre-cached greeting**: Greeting audio generated at startup in WAV and mulaw formats for instant playback
- **Fail-open defaults**: Classifier unavailable? Assume human. API fails? Use fallback behavior.
- **One action per turn**: System prompt enforces single action per response to sound natural
- **Conversation windowing**: Keeps system prompt + last 20 messages to prevent token overflow

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Mic not working in browser | Must be on `localhost` or HTTPS. Check browser mic permissions. |
| "Sunai nahi diya" on very short audio | Hold the mic button longer — minimum ~0.3 seconds. |
| Sarvam API errors | Run `python test_sarvam.py`. Check your API key and quota. |
| Twilio not receiving calls | Verify ngrok is running and `BASE_URL` matches. Re-run `setup_twilio.py` or use `start.py`. |
| Audio not playing in browser | Check console for errors. Some browsers block autoplay — click the page first. |
| High latency (5-8 seconds) | Expected — sequential STT + LLM + TTS API calls. Streaming is already enabled. |
| Voice classifier not starting | Ensure `model_output/best_model/` exists. Run `python train.py` if needed. |
| `audioop` import error on Python 3.13+ | Install `audioop-lts` (`pip install audioop-lts`). Already in requirements.txt. |
| `webrtcvad` import error | Falls back to energy-based VAD automatically. Install with `pip install webrtcvad` if you want better accuracy. |
| WebSocket disconnects | Check network stability. Server logs show connection/disconnection events. |
| ngrok URL changes on restart | Use `start.py` (auto-updates) or get a paid ngrok plan for fixed domains. |

---

## License

This project is for educational and research purposes. Use responsibly.
