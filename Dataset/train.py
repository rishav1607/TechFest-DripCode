"""
Training Script for AI Voice Detector
Fine-tunes wav2vec2-base for binary classification (Human vs AI).
Optimized for RTX 4060 8GB VRAM.

Dataset layout:
  dataset/train/human/  dataset/train/ai/
  dataset/test/human/   dataset/test/ai/
"""

import os
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG - Optimized for RTX 4060 8GB
# ============================================================
class Config:
    # Paths
    dataset_dir = Path(__file__).parent / "dataset"
    metadata_file = dataset_dir / "metadata.csv"
    output_dir = Path(__file__).parent / "model_output"

    # Model
    model_name = "facebook/wav2vec2-base"
    num_labels = 2
    label2id = {"human": 0, "ai": 1}
    id2label = {0: "human", 1: "ai"}

    # Audio
    target_sr = 16000
    max_duration_sec = 5
    max_length = target_sr * max_duration_sec  # 80000 samples

    # Training - optimized for 8GB VRAM
    batch_size = 4
    gradient_accumulation_steps = 4  # effective batch size = 16
    learning_rate = 2e-5
    weight_decay = 0.01
    num_epochs = 10
    warmup_ratio = 0.1
    fp16 = True
    freeze_feature_extractor = True
    gradient_checkpointing = True

    seed = 42


# ============================================================
# DATASET CLASS
# ============================================================
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, feature_extractor, max_length):
        self.file_paths = file_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.max_length = max_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            waveform, sr = torchaudio.load(audio_path)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)

            if sr != Config.target_sr:
                resampler = torchaudio.transforms.Resample(sr, Config.target_sr)
                waveform = resampler(waveform)

            if waveform.shape[0] > self.max_length:
                start = random.randint(0, waveform.shape[0] - self.max_length)
                waveform = waveform[start:start + self.max_length]
            elif waveform.shape[0] < self.max_length:
                padding = self.max_length - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            waveform = waveform.numpy()
            if np.max(np.abs(waveform)) > 0:
                waveform = waveform / np.max(np.abs(waveform))

            inputs = self.feature_extractor(
                waveform,
                sampling_rate=Config.target_sr,
                return_tensors="pt",
                padding=False,
            )
            input_values = inputs.input_values.squeeze(0)

        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            input_values = torch.zeros(self.max_length)

        return {
            "input_values": input_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }


def collate_fn(batch):
    input_values = torch.stack([item["input_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_values": input_values, "labels": labels}


# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, config):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for step, batch in enumerate(pbar):
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)

        if config.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(input_values=input_values, labels=labels)
                loss = outputs.loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(input_values=input_values, labels=labels)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()

        total_loss += loss.item() * config.gradient_accumulation_steps

        if (step + 1) % config.gradient_accumulation_steps == 0:
            if config.fp16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item() * config.gradient_accumulation_steps:.4f}"})

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, device, config):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)

        if config.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(input_values=input_values, labels=labels)
        else:
            outputs = model(input_values=input_values, labels=labels)

        total_loss += outputs.loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, accuracy, f1, all_preds, all_labels


# ============================================================
# MAIN
# ============================================================
def main():
    config = Config()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    print("=" * 60)
    print("  AI VOICE DETECTOR - WAV2VEC2 FINE-TUNING")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Device: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load metadata (has a 'split' column: train / test)
    print(f"\n[*] Loading metadata from {config.metadata_file}")
    df = pd.read_csv(config.metadata_file)

    # Reconstruct paths relative to this project (handles directory moves)
    df["file_path"] = df.apply(
        lambda r: str(config.dataset_dir / r["split"] / r["label_name"] / r["file_name"]),
        axis=1,
    )
    df = df[df["file_path"].apply(os.path.exists)].reset_index(drop=True)

    train_full_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # Split train into train + val (80/20 of training set)
    train_df, val_df = train_test_split(
        train_full_df, test_size=0.2,
        stratify=train_full_df["label"], random_state=config.seed
    )

    print(f"    Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"    Train human: {(train_df['label']==0).sum()} | Train AI: {(train_df['label']==1).sum()}")
    print(f"    Test  human: {(test_df['label']==0).sum()} | Test  AI: {(test_df['label']==1).sum()}")

    # Load model
    print(f"\n[*] Loading model: {config.model_name}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.model_name)

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        label2id=config.label2id,
        id2label=config.id2label,
    )

    if config.freeze_feature_extractor:
        model.freeze_feature_encoder()
        print("    Feature encoder: frozen")

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("    Gradient checkpointing: enabled")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    Params: {trainable:,} trainable / {total:,} total ({trainable/total*100:.1f}%)")

    model.to(device)

    # Datasets & loaders
    train_dataset = AudioDataset(train_df["file_path"].tolist(), train_df["label"].tolist(), feature_extractor, config.max_length)
    val_dataset = AudioDataset(val_df["file_path"].tolist(), val_df["label"].tolist(), feature_extractor, config.max_length)
    test_dataset = AudioDataset(test_df["file_path"].tolist(), test_df["label"].tolist(), feature_extractor, config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )

    total_steps = (len(train_loader) // config.gradient_accumulation_steps) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda") if config.fp16 else None

    print(f"\n[*] Training:")
    print(f"    Effective batch: {config.batch_size} x {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps}")
    print(f"    LR: {config.learning_rate} | Epochs: {config.num_epochs} | fp16: {config.fp16}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    best_epoch = 0

    print("\n" + "=" * 60)
    print("  TRAINING STARTED")
    print("=" * 60)

    for epoch in range(config.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{config.num_epochs} ---")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, config)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device, config)

        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            model.save_pretrained(config.output_dir / "best_model")
            feature_extractor.save_pretrained(config.output_dir / "best_model")
            print(f"  >> Best model saved (val_acc: {val_acc:.4f})")

        if device.type == "cuda":
            print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB / {torch.cuda.memory_reserved()/1024**3:.1f}GB")

    # Save final
    model.save_pretrained(config.output_dir / "final_model")
    feature_extractor.save_pretrained(config.output_dir / "final_model")

    # Test
    print("\n" + "=" * 60)
    print("  TEST SET EVALUATION")
    print("=" * 60)

    best_model = Wav2Vec2ForSequenceClassification.from_pretrained(config.output_dir / "best_model").to(device)
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(best_model, test_loader, device, config)

    print(f"\n  Test Accuracy: {test_acc:.4f}")
    print(f"  Test F1:       {test_f1:.4f}")
    print(f"\n{classification_report(test_labels, test_preds, target_names=['Human', 'AI'], digits=4)}")

    summary = {
        "model": config.model_name,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
    }
    with open(config.output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Best epoch: {best_epoch} | Model saved to: {config.output_dir}")
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
