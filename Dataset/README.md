# AI Voice Detector - Fine-tuned wav2vec2

A multi-signal audio classifier that detects whether a voice is **AI-generated** or **Human**.
Fine-tuned `facebook/wav2vec2-base` + WavLM embedding analysis + spectral artifact detection + prosody analysis.

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **96.00%** |
| Test F1 Score | **95.99%** |
| Best Epoch | 4 / 10 |

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Human | 1.0000 | 0.9200 | 0.9583 |
| AI | 0.9259 | 1.0000 | 0.9615 |

## Project Structure

```
Dataset/
├── venv/                    # Python virtual environment
├── dataset/                 # Audio dataset
│   ├── train/
│   │   ├── human/           # 50 files (25 English + 25 Hindi)
│   │   └── ai/              # 50 files (25 English + 25 Hindi)
│   ├── test/
│   │   ├── human/           # 50 files (25 English + 25 Hindi)
│   │   └── ai/              # 50 files (25 English + 25 Hindi)
│   └── metadata.csv         # File paths, labels, language, split info
├── model_output/
│   ├── best_model/          # Best checkpoint (highest val accuracy)
│   ├── final_model/         # Last epoch checkpoint
│   └── training_summary.json
├── collect_dataset.py       # Downloads human audio + generates AI audio
├── fix_ai.py                # Fills any missing AI samples
├── train.py                 # Fine-tuning script
├── inference.py             # Run predictions on new audio
├── api.py                   # REST API server (FastAPI)
├── analyzers.py             # Multi-signal analyzers (WavLM, spectral, prosody)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Dataset

| Split | Human | AI | Total |
|-------|-------|----|-------|
| Train | 50 (25 EN + 25 HI) | 50 (25 EN + 25 HI) | 100 |
| Test | 50 (25 EN + 25 HI) | 50 (25 EN + 25 HI) | 100 |
| **Total** | **100** | **100** | **200** |

- **Human audio**: Real speech from [Google FLEURS](https://huggingface.co/datasets/google/fleurs) dataset (English + Hindi)
- **AI audio**: Generated using [Microsoft Edge TTS](https://github.com/rany2/edge-tts) with multiple voices
- All audio: 16kHz WAV, mono, normalized

## Setup (Windows)

### Prerequisites

- Python 3.13 (installed at `C:\Python313\python.exe`)
- NVIDIA GPU with CUDA support (tested on RTX 4060 8GB)
- NVIDIA drivers installed

### Step 1: Create Virtual Environment

Open **PowerShell** or **Command Prompt** and navigate to the project:

```powershell
cd Z:\Code\granny\Dataset
```

Create and activate the virtual environment:

```powershell
# Create venv (skip if already exists)
C:\Python313\python.exe -m venv venv

# Activate (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Command Prompt / cmd)
.\venv\Scripts\activate.bat
```

> If PowerShell gives an execution policy error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Step 2: Install Dependencies

```powershell
# Install PyTorch with CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install transformers datasets==3.2.0 accelerate edge-tts soundfile librosa scikit-learn pandas numpy tqdm evaluate
```

> Note: `datasets==3.2.0` is required because newer versions dropped support for the FLEURS dataset loader.

### Step 3: Collect Dataset (optional - already collected)

Skip this if the `dataset/` folder already has 200 files.

```powershell
# Download human audio from FLEURS + generate AI audio with edge-tts
python collect_dataset.py

# Fix any missing AI samples (if edge-tts failed on some)
python fix_ai.py
```

### Step 4: Train the Model (optional - already trained)

Skip this if `model_output/best_model/` already exists.

```powershell
python train.py
```

Training takes about 5-10 minutes on RTX 4060 8GB.

**Training config (optimized for 8GB VRAM):**
- Model: `facebook/wav2vec2-base` (95M params)
- Feature encoder: frozen (only transformer + classifier trained)
- Batch size: 4 with gradient accumulation 4 (effective batch 16)
- Mixed precision: fp16
- Gradient checkpointing: enabled
- Max audio length: 5 seconds
- Epochs: 10
- VRAM usage: ~2-3 GB

## Inference

### Classify a single audio file

```powershell
# Activate venv first
.\venv\Scripts\Activate.ps1

# Run on any audio file
python inference.py path\to\audio.wav
```

### Classify multiple files at once

```powershell
python inference.py file1.wav file2.mp3 file3.wav
```

### Example output

```
Loading model...
Model loaded on cuda

File: test_audio.wav
  Prediction: AI
  Confidence: 97.83%
  Human: 2.17% | AI: 97.83%
```

### Test on dataset samples

```powershell
# Test on a human sample
python inference.py dataset\test\human\human_en_0000.wav

# Test on an AI sample
python inference.py dataset\test\ai\ai_hi_0000.wav

# Test on multiple
python inference.py dataset\test\human\human_hi_0005.wav dataset\test\ai\ai_en_0010.wav
```

### Supported audio formats

- WAV (recommended)
- MP3
- FLAC
- OGG
- Any format supported by torchaudio

Audio is automatically resampled to 16kHz mono if needed.

## REST API

### Start the API server

```powershell
# Activate venv first
.\venv\Scripts\Activate.ps1

# Install API dependencies (one-time)
pip install fastapi uvicorn httpx python-multipart

# Start the server
python api.py
```

Server runs at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Endpoints

#### `POST /predict` — Upload audio file

```powershell
curl -X POST -F "file=@dataset\test\human\human_en_0000.wav" http://localhost:8000/predict
```

#### `POST /predict-url` — Predict from URL (Twilio integration)

```powershell
curl -X POST -H "Content-Type: application/json" -d "{\"url\":\"https://api.twilio.com/recordings/RExxxxx.wav\"}" http://localhost:8000/predict-url
```

#### `GET /health` — Health check

```powershell
curl http://localhost:8000/health
```

### JSON Response (Multi-Signal Analysis)

```json
{
  "prediction": "ai",
  "confidence": 0.6743,
  "probabilities": { "human": 0.3857, "ai": 0.6143 },
  "ensemble": {
    "final_prediction": "ai",
    "final_confidence": 0.6743,
    "ensemble_ai_score": 0.6743,
    "signal_scores": {
      "wav2vec2": 0.6143,
      "wavlm": 0.5847,
      "spectral": 0.17,
      "prosody": 0.6776
    },
    "signal_agreement": 0.75
  },
  "analysis": {
    "wavlm": {
      "temporal_variance": 0.154,
      "entropy": 1.96,
      "layer_divergence": 0.711,
      "ai_score": 0.5847
    },
    "spectral": {
      "spectral_flatness": 0.022,
      "hf_energy_ratio": 0.0,
      "bandwidth_std": 390.76,
      "rolloff_std": 1509.89,
      "ai_score": 0.17
    },
    "prosody": {
      "jitter": 0.045,
      "shimmer": 0.173,
      "f0_range_semitones": 11.5,
      "voiced_ratio": 0.936,
      "ai_score": 0.6776
    }
  }
}
```

### Twilio Integration Example

Use the `/predict-url` endpoint with Twilio's recording URL:

```python
import requests

# After Twilio records a call, get the recording URL
recording_url = "https://api.twilio.com/2010-04-01/Accounts/ACxxx/Recordings/REyyy.wav"

response = requests.post(
    "http://localhost:8000/predict-url",
    json={"url": recording_url}
)
result = response.json()

if result["prediction"] == "ai":
    print(f"AI voice detected! Confidence: {result['confidence']:.2%}")
else:
    print(f"Human voice. Confidence: {result['confidence']:.2%}")
```

## Multi-Signal Analysis Pipeline

The API uses 4 independent analysis signals for robust detection:

```
Audio input
   ├── wav2vec2 (fine-tuned model)     → Primary AI probability
   ├── WavLM embedding analysis        → Hidden state patterns
   ├── Spectral artifact detection     → Frequency/phase anomalies
   └── Prosody analysis                → Pitch jitter, shimmer, rhythm
                    ↓
         Ensemble Classifier (confidence-gated)
                    ↓
              Final decision
```

| Signal | What it detects | Method |
|--------|----------------|--------|
| **wav2vec2** | Learned audio patterns | Fine-tuned neural network (primary) |
| **WavLM** | Embedding entropy, layer divergence | Pre-trained Microsoft WavLM-base |
| **Spectral** | Flatness, bandwidth, rolloff patterns | librosa signal processing (CPU) |
| **Prosody** | Pitch jitter, shimmer, voiced ratio | librosa pitch tracking (CPU) |

The ensemble uses wav2vec2 as the primary decision-maker. Supplementary signals boost confidence when they agree, and can flip borderline predictions only when all 3 unanimously disagree.

## Use in Your Own Code

```python
from inference import load_model, predict

# Load once
model, feature_extractor, device = load_model()

# Predict on any audio file
label, confidence, probs = predict("audio.wav", model, feature_extractor, device)

print(f"Result: {label} ({confidence:.2%})")
print(f"Human probability: {probs[0]:.2%}")
print(f"AI probability: {probs[1]:.2%}")
```

## Model Details

| Property | Value |
|----------|-------|
| Base model | `facebook/wav2vec2-base` |
| Task | Binary classification (Human vs AI) |
| Input | Raw audio waveform (16kHz) |
| Output | Human (0) or AI (1) with confidence |
| Total parameters | 94.4M |
| Trainable parameters | ~68.7M (72.8%) |
| Feature encoder | Frozen (7 CNN layers) |
| Transformer layers | 12 (trainable) |
| Training data | 100 samples (50 human + 50 AI) |
| Test data | 100 samples (50 human + 50 AI) |
| Languages | English, Hindi |

## Tech Stack

- **PyTorch 2.6** with CUDA 12.4
- **Hugging Face Transformers** for wav2vec2
- **edge-tts** for AI voice generation
- **Google FLEURS** for human voice samples
- **torchaudio** for audio loading and resampling
- **Microsoft WavLM** for embedding-based analysis
- **librosa** for spectral and prosody analysis
- **FastAPI + Uvicorn** for REST API server

## Troubleshooting

### "source is not recognized" in PowerShell
Use `.\venv\Scripts\Activate.ps1` instead of `source venv/Scripts/activate`.

### CUDA out of memory
Reduce `max_duration_sec` in `train.py` Config from 5 to 3, or reduce `batch_size` to 2.

### edge-tts fails on some samples
Run `python fix_ai.py` to regenerate failed samples using torchaudio for mp3 conversion.

### Model not found error during inference
Make sure training completed and `model_output/best_model/` exists with model files.

### datasets library error with FLEURS
Use `datasets==3.2.0` specifically. Newer versions dropped loading script support.
