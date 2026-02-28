"""
Inference Script for AI Voice Detector
Load the fine-tuned wav2vec2 model and classify audio as Human or AI.
"""

import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

MODEL_DIR = Path(__file__).parent / "model_output" / "best_model"
TARGET_SR = 16000
MAX_DURATION_SEC = 5
MAX_LENGTH = TARGET_SR * MAX_DURATION_SEC


def load_model(model_dir=MODEL_DIR):
    """Load the fine-tuned model and feature extractor."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return model, feature_extractor, device


def predict(audio_path, model, feature_extractor, device):
    """Predict whether an audio file is Human or AI generated."""
    waveform, sr = torchaudio.load(audio_path)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    # Resample
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
        waveform = resampler(waveform)

    # Truncate / pad
    if waveform.shape[0] > MAX_LENGTH:
        waveform = waveform[:MAX_LENGTH]
    elif waveform.shape[0] < MAX_LENGTH:
        padding = MAX_LENGTH - waveform.shape[0]
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    # Normalize
    waveform = waveform.numpy()
    if np.max(np.abs(waveform)) > 0:
        waveform = waveform / np.max(np.abs(waveform))

    # Feature extraction
    inputs = feature_extractor(
        waveform, sampling_rate=TARGET_SR, return_tensors="pt", padding=False
    )
    input_values = inputs.input_values.to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_values=input_values)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()

    label = model.config.id2label[pred_id]
    return label, confidence, probs[0].cpu().numpy()


def main():
    print("Loading model...")
    model, feature_extractor, device = load_model()
    print(f"Model loaded on {device}\n")

    # If no args, auto-test on a few dataset samples
    if len(sys.argv) < 2:
        print("No audio file provided. Running demo on test set samples...\n")
        test_dir = Path(__file__).parent / "dataset" / "test"
        demo_files = []
        for cls in ["human", "ai"]:
            cls_dir = test_dir / cls
            if cls_dir.exists():
                files = sorted(cls_dir.glob("*.wav"))[:3]
                demo_files.extend(files)
        if not demo_files:
            print("No test files found. Usage: python inference.py <audio_file>")
            sys.exit(1)
        audio_paths = [str(f) for f in demo_files]
    else:
        audio_paths = sys.argv[1:]

    correct = 0
    total = 0
    for audio_path in audio_paths:
        path = Path(audio_path)
        if not path.exists():
            print(f"[!] File not found: {audio_path}")
            continue

        label, confidence, probs = predict(audio_path, model, feature_extractor, device)

        # Determine ground truth from path if available
        ground_truth = ""
        if "/human/" in str(path).replace("\\", "/"):
            ground_truth = "human"
        elif "/ai/" in str(path).replace("\\", "/"):
            ground_truth = "ai"

        status = ""
        if ground_truth:
            is_correct = label == ground_truth
            status = " ✓" if is_correct else " ✗"
            total += 1
            if is_correct:
                correct += 1

        print(f"File: {path.name}")
        print(f"  Prediction: {label.upper()}{status}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Human: {probs[0]:.2%} | AI: {probs[1]:.2%}")
        print()

    if total > 0:
        print(f"Accuracy: {correct}/{total} ({correct/total:.0%})")


if __name__ == "__main__":
    main()
