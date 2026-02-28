"""Fill missing AI English samples using torchaudio for mp3 loading instead of librosa."""

import os
import asyncio
import random
import tempfile
import csv
from pathlib import Path

import edge_tts
import torchaudio
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm

BASE_DIR = Path(__file__).parent / "dataset"
TARGET_SR = 16000

ENGLISH_VOICES = [
    "en-US-GuyNeural", "en-US-JennyNeural", "en-US-AriaNeural",
    "en-US-DavisNeural", "en-US-AndrewNeural", "en-US-ChristopherNeural",
    "en-US-CoraNeural", "en-US-ElizabethNeural",
]

SENTENCES = [
    "The morning sun cast long shadows across the quiet suburban street.",
    "Research shows that regular physical activity improves mental health.",
    "Modern architecture blends functionality with aesthetic beauty seamlessly.",
    "The stock market showed remarkable resilience despite economic uncertainties.",
    "Children learn best through hands on experience and creative exploration.",
    "Digital transformation is reshaping how businesses operate globally.",
    "The national park attracts millions of visitors every single year.",
    "Advances in medical technology have dramatically improved patient outcomes.",
    "Space exploration continues to push the boundaries of human knowledge.",
    "Online education has made learning accessible to people around the world.",
    "Classical music has a profound ability to evoke deep emotional responses.",
    "Public libraries remain vital community resources in the digital age.",
    "Wildlife conservation efforts have helped several endangered species recover.",
    "Renewable energy installations have increased dramatically over the past decade.",
    "Urban planning must consider environmental sustainability and community needs.",
    "Healthcare systems worldwide are adapting to meet changing demographic needs.",
    "Advances in battery technology are making electric vehicles more practical.",
    "Social media platforms continue to evolve and shape public discourse.",
    "Biotechnology innovations are opening new frontiers in disease treatment.",
    "The historical documentary won several international film festival awards.",
    "Weather patterns have become increasingly unpredictable in recent years.",
    "The new railway line will connect previously isolated communities to cities.",
    "Artificial intelligence is being used to improve agricultural productivity.",
    "Ocean pollution poses a serious threat to marine ecosystems globally.",
    "Community gardens provide fresh produce and foster neighborhood connections.",
]


async def generate_one(text, voice, output_path):
    """Generate TTS and convert to wav using torchaudio."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(tmp_path)

        # Use torchaudio instead of librosa for better mp3 support
        waveform, sr = torchaudio.load(tmp_path)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        # Resample
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            waveform = resampler(waveform)

        audio = waveform.numpy().astype(np.float32)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95

        sf.write(str(output_path), audio, TARGET_SR)
        os.unlink(tmp_path)
        return True
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except:
            pass
        return False


async def main():
    random.seed(77)

    for split in ["train", "test"]:
        ai_dir = BASE_DIR / split / "ai"
        existing_en = sorted(ai_dir.glob("ai_en_*.wav"))
        existing_count = len(existing_en)
        needed = 25 - existing_count

        if needed <= 0:
            print(f"[{split}] Already have {existing_count} English AI - OK")
            continue

        print(f"[{split}] Have {existing_count}/25 English AI, generating {needed} more...")

        # Find which indices are missing
        existing_indices = set()
        for f in existing_en:
            idx = int(f.stem.split("_")[-1])
            existing_indices.add(idx)

        generated = 0
        attempt = 0
        random.shuffle(SENTENCES)
        sent_idx = 0

        for idx in range(25):
            if idx in existing_indices:
                continue

            text = SENTENCES[sent_idx % len(SENTENCES)]
            sent_idx += 1
            voice = random.choice(ENGLISH_VOICES)
            output_path = ai_dir / f"ai_en_{idx:04d}.wav"

            # Try up to 3 times
            for retry in range(3):
                success = await generate_one(text, voice, output_path)
                if success:
                    generated += 1
                    break
                voice = random.choice(ENGLISH_VOICES)

        print(f"  Generated {generated}/{needed} missing samples")

    # Rebuild metadata
    print("\nRebuilding metadata...")
    metadata_file = BASE_DIR / "metadata.csv"
    rows = []

    for split in ["train", "test"]:
        split_dir = BASE_DIR / split
        for cls in ["human", "ai"]:
            for f in sorted((split_dir / cls).glob("*.wav")):
                lang = "en" if "_en_" in f.name else "hi"
                rows.append({
                    "file_path": str(f),
                    "file_name": f.name,
                    "label": 0 if cls == "human" else 1,
                    "label_name": cls,
                    "language": lang,
                    "split": split,
                })

    with open(metadata_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "file_name", "label", "label_name", "language", "split"])
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    for split in ["train", "test"]:
        s = [r for r in rows if r["split"] == split]
        h = sum(1 for r in s if r["label"] == 0)
        a = sum(1 for r in s if r["label"] == 1)
        print(f"  [{split.upper()}] Human: {h} | AI: {a} | Total: {len(s)}")

    print(f"\nTotal: {total} files")


if __name__ == "__main__":
    asyncio.run(main())
