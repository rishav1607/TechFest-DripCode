"""
Dataset Collection Script for AI Voice Detector
Uses FLEURS (English + Hindi) for human audio, edge-tts for AI audio.

Dataset structure (200 files total):
  TRAIN: 50 human (25 en + 25 hi) + 50 AI (25 en + 25 hi) = 100
  TEST:  50 human (25 en + 25 hi) + 50 AI (25 en + 25 hi) = 100
"""

import os
import asyncio
import random
import csv
import shutil
import tempfile
from pathlib import Path

import edge_tts
import librosa
import soundfile as sf
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).parent / "dataset"
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"
METADATA_FILE = BASE_DIR / "metadata.csv"

TARGET_SR = 16000  # wav2vec2 expects 16kHz

# Per split: 25 english + 25 hindi = 50 per class per split
SAMPLES_PER_LANG_PER_SPLIT = 25

ENGLISH_VOICES = [
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-AriaNeural",
    "en-US-DavisNeural",
    "en-US-AndrewNeural",
    "en-US-ChristopherNeural",
    "en-US-CoraNeural",
    "en-US-ElizabethNeural",
]

HINDI_VOICES = [
    "hi-IN-MadhurNeural",
    "hi-IN-SwaraNeural",
]

# Backup sentences for AI TTS (used if FLEURS transcripts are too short)
ENGLISH_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Artificial intelligence is transforming the way we interact with technology.",
    "She walked through the garden admiring the colorful flowers blooming everywhere.",
    "The weather forecast predicts heavy rainfall throughout the weekend.",
    "Scientists have discovered a new species of butterfly in the Amazon rainforest.",
    "The children played happily in the park while their parents watched from the bench.",
    "Learning a new language opens doors to different cultures and perspectives.",
    "The old library on the corner has been serving the community for over a century.",
    "Music has the power to bring people together regardless of their background.",
    "The sun set behind the mountains painting the sky in shades of orange and purple.",
    "Technology continues to evolve at an unprecedented pace in the modern world.",
    "The chef prepared a delicious meal using fresh ingredients from the local market.",
    "Education is the foundation of a prosperous and progressive society.",
    "The train arrived at the station exactly on time despite the heavy snowfall.",
    "Reading books is one of the best ways to expand your knowledge and vocabulary.",
    "The conference attracted researchers from more than fifty different countries.",
    "She completed the marathon in under four hours setting a personal record.",
    "The documentary explored the impact of climate change on coastal communities.",
    "A healthy breakfast is essential for maintaining energy throughout the day.",
    "The museum exhibition showcased artwork from the Renaissance period.",
    "Communication skills are vital for success in both personal and professional life.",
    "The river flowed peacefully through the valley surrounded by tall green trees.",
    "Innovation drives economic growth and creates new opportunities for employment.",
    "The astronaut described the view of Earth from space as absolutely breathtaking.",
    "Regular exercise and a balanced diet contribute to overall physical wellbeing.",
    "The software update includes several new features and important security patches.",
    "History teaches us valuable lessons about human resilience and determination.",
    "The orchestra performed a beautiful symphony that moved the audience to tears.",
    "Renewable energy sources are becoming increasingly important for our future.",
    "The detective carefully examined the evidence before drawing any conclusions.",
    "The morning sun cast long shadows across the quiet suburban street.",
    "Research shows that regular physical activity improves mental health significantly.",
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
    "Quantum computing promises to revolutionize complex problem solving capabilities.",
    "Healthcare systems worldwide are adapting to meet changing demographic needs.",
    "Advances in battery technology are making electric vehicles more practical.",
    "Social media platforms continue to evolve and shape public discourse.",
    "Biotechnology innovations are opening new frontiers in disease treatment.",
]

HINDI_SENTENCES = [
    "\u092d\u093e\u0930\u0924 \u090f\u0915 \u0935\u093f\u0935\u093f\u0927\u0924\u093e\u0913\u0902 \u0938\u0947 \u092d\u0930\u093e \u0926\u0947\u0936 \u0939\u0948 \u091c\u0939\u093e\u0902 \u0905\u0928\u0947\u0915 \u092d\u093e\u0937\u093e\u090f\u0902 \u092c\u094b\u0932\u0940 \u091c\u093e\u0924\u0940 \u0939\u0948\u0902\u0964",
    "\u0906\u091c \u0915\u093e \u092e\u094c\u0938\u092e \u092c\u0939\u0941\u0924 \u0938\u0941\u0939\u093e\u0935\u0928\u093e \u0939\u0948 \u0914\u0930 \u0906\u0938\u092e\u093e\u0928 \u092c\u093f\u0932\u0915\u0941\u0932 \u0938\u093e\u092b \u0939\u0948\u0964",
    "\u0936\u093f\u0915\u094d\u0937\u093e \u0939\u0930 \u0935\u094d\u092f\u0915\u094d\u0924\u093f \u0915\u093e \u092e\u094c\u0932\u093f\u0915 \u0905\u0927\u093f\u0915\u093e\u0930 \u0939\u0948 \u0914\u0930 \u0907\u0938\u0947 \u0938\u092c\u0915\u094b \u092e\u093f\u0932\u0928\u093e \u091a\u093e\u0939\u093f\u090f\u0964",
    "\u092a\u094d\u0930\u094c\u0926\u094d\u092f\u094b\u0917\u093f\u0915\u0940 \u0928\u0947 \u0939\u092e\u093e\u0930\u0947 \u091c\u0940\u0935\u0928 \u0915\u0947 \u0939\u0930 \u092a\u0939\u0932\u0942 \u0915\u094b \u092c\u0926\u0932 \u0926\u093f\u092f\u093e \u0939\u0948\u0964",
    "\u0938\u094d\u0935\u0938\u094d\u0925 \u091c\u0940\u0935\u0928 \u091c\u0940\u0928\u0947 \u0915\u0947 \u0932\u093f\u090f \u0928\u093f\u092f\u092e\u093f\u0924 \u0935\u094d\u092f\u093e\u092f\u093e\u092e \u0914\u0930 \u0938\u0902\u0924\u0941\u0932\u093f\u0924 \u0906\u0939\u093e\u0930 \u091c\u0930\u0942\u0930\u0940 \u0939\u0948\u0964",
    "\u0939\u093f\u092e\u093e\u0932\u092f \u092a\u0930\u094d\u0935\u0924 \u0936\u094d\u0943\u0902\u0916\u0932\u093e \u0926\u0941\u0928\u093f\u092f\u093e \u0915\u0940 \u0938\u092c\u0938\u0947 \u090a\u0902\u091a\u0940 \u092a\u0930\u094d\u0935\u0924 \u0936\u094d\u0943\u0902\u0916\u0932\u093e \u0939\u0948\u0964",
    "\u0917\u0902\u0917\u093e \u0928\u0926\u0940 \u092d\u093e\u0930\u0924 \u0915\u0940 \u0938\u092c\u0938\u0947 \u092a\u0935\u093f\u0924\u094d\u0930 \u0914\u0930 \u092e\u0939\u0924\u094d\u0935\u092a\u0942\u0930\u094d\u0923 \u0928\u0926\u093f\u092f\u094b\u0902 \u092e\u0947\u0902 \u0938\u0947 \u090f\u0915 \u0939\u0948\u0964",
    "\u092d\u093e\u0930\u0924\u0940\u092f \u0938\u0902\u0938\u094d\u0915\u0943\u0924\u093f \u0905\u092a\u0928\u0940 \u0935\u093f\u0935\u093f\u0927\u0924\u093e \u0914\u0930 \u0938\u092e\u0943\u0926\u094d\u0927\u093f \u0915\u0947 \u0932\u093f\u090f \u0935\u093f\u0936\u094d\u0935 \u092d\u0930 \u092e\u0947\u0902 \u092a\u094d\u0930\u0938\u093f\u0926\u094d\u0927 \u0939\u0948\u0964",
    "\u0915\u0902\u092a\u094d\u092f\u0942\u091f\u0930 \u0935\u093f\u091c\u094d\u091e\u093e\u0928 \u0906\u091c \u0915\u0947 \u0938\u092e\u092f \u092e\u0947\u0902 \u0938\u092c\u0938\u0947 \u0932\u094b\u0915\u092a\u094d\u0930\u093f\u092f \u0935\u093f\u0937\u092f\u094b\u0902 \u092e\u0947\u0902 \u0938\u0947 \u090f\u0915 \u0939\u0948\u0964",
    "\u092a\u0930\u094d\u092f\u093e\u0935\u0930\u0923 \u0915\u0940 \u0930\u0915\u094d\u0937\u093e \u0915\u0930\u0928\u093e \u0939\u092e \u0938\u092c\u0915\u0940 \u091c\u093f\u092e\u094d\u092e\u0947\u0926\u093e\u0930\u0940 \u0939\u0948\u0964",
    "\u092d\u093e\u0930\u0924 \u092e\u0947\u0902 \u0932\u094b\u0915\u0924\u0902\u0924\u094d\u0930 \u0938\u092c\u0938\u0947 \u092c\u0921\u093c\u093e \u0914\u0930 \u0938\u092c\u0938\u0947 \u092e\u091c\u092c\u0942\u0924 \u0939\u0948\u0964",
    "\u0939\u092e\u0947\u0902 \u0905\u092a\u0928\u0947 \u092c\u0941\u091c\u0941\u0930\u094d\u0917\u094b\u0902 \u0915\u093e \u0938\u092e\u094d\u092e\u093e\u0928 \u0915\u0930\u0928\u093e \u091a\u093e\u0939\u093f\u090f \u0915\u094d\u092f\u094b\u0902\u0915\u093f \u0909\u0928\u0915\u093e \u0905\u0928\u0941\u092d\u0935 \u0905\u092e\u0942\u0932\u094d\u092f \u0939\u0948\u0964",
    "\u0935\u093f\u091c\u094d\u091e\u093e\u0928 \u0914\u0930 \u092a\u094d\u0930\u094c\u0926\u094d\u092f\u094b\u0917\u093f\u0915\u0940 \u0928\u0947 \u092e\u093e\u0928\u0935 \u091c\u0940\u0935\u0928 \u0915\u094b \u0938\u0930\u0932 \u0914\u0930 \u0938\u0941\u0935\u093f\u0927\u093e\u091c\u0928\u0915 \u092c\u0928\u093e \u0926\u093f\u092f\u093e \u0939\u0948\u0964",
    "\u092d\u093e\u0930\u0924\u0940\u092f \u0916\u093e\u0928\u093e \u0905\u092a\u0928\u0947 \u0938\u094d\u0935\u093e\u0926 \u0914\u0930 \u092e\u0938\u093e\u0932\u094b\u0902 \u0915\u0947 \u0932\u093f\u090f \u092a\u0942\u0930\u0940 \u0926\u0941\u0928\u093f\u092f\u093e \u092e\u0947\u0902 \u092e\u0936\u0939\u0942\u0930 \u0939\u0948\u0964",
    "\u0915\u093f\u0938\u093e\u0928 \u0939\u092e\u093e\u0930\u0947 \u0926\u0947\u0936 \u0915\u0940 \u0930\u0940\u0922\u093c \u0939\u0948\u0902 \u0914\u0930 \u0909\u0928\u0915\u093e \u092f\u094b\u0917\u0926\u093e\u0928 \u0905\u092e\u0942\u0932\u094d\u092f \u0939\u0948\u0964",
    "\u092a\u0922\u093c\u093e\u0908 \u092e\u0947\u0902 \u092e\u0928 \u0932\u0917\u093e\u0928\u0947 \u0915\u0947 \u0932\u093f\u090f \u090f\u0915\u093e\u0917\u094d\u0930\u0924\u093e \u0914\u0930 \u0905\u0928\u0941\u0936\u093e\u0938\u0928 \u092c\u0939\u0941\u0924 \u091c\u0930\u0942\u0930\u0940 \u0939\u0948\u0964",
    "\u092d\u093e\u0930\u0924 \u0915\u093e \u0905\u0902\u0924\u0930\u093f\u0915\u094d\u0937 \u0915\u093e\u0930\u094d\u092f\u0915\u094d\u0930\u092e \u0926\u0941\u0928\u093f\u092f\u093e \u0915\u0947 \u0938\u092c\u0938\u0947 \u0938\u092b\u0932 \u0915\u093e\u0930\u094d\u092f\u0915\u094d\u0930\u092e\u094b\u0902 \u092e\u0947\u0902 \u0938\u0947 \u090f\u0915 \u0939\u0948\u0964",
    "\u0938\u0902\u0917\u0940\u0924 \u0906\u0924\u094d\u092e\u093e \u0915\u093e \u092d\u094b\u091c\u0928 \u0939\u0948 \u0914\u0930 \u092f\u0939 \u0939\u0930 \u0915\u093f\u0938\u0940 \u0915\u0947 \u091c\u0940\u0935\u0928 \u092e\u0947\u0902 \u0916\u0941\u0936\u0940 \u0932\u093e\u0924\u093e \u0939\u0948\u0964",
    "\u0938\u094d\u0935\u0924\u0902\u0924\u094d\u0930\u0924\u093e \u0926\u093f\u0935\u0938 \u0939\u0930 \u092d\u093e\u0930\u0924\u0940\u092f \u0915\u0947 \u0932\u093f\u090f \u0917\u0930\u094d\u0935 \u0914\u0930 \u0938\u092e\u094d\u092e\u093e\u0928 \u0915\u093e \u0926\u093f\u0928 \u0939\u0948\u0964",
    "\u092f\u094b\u0917 \u0914\u0930 \u0927\u094d\u092f\u093e\u0928 \u0938\u0947 \u0936\u093e\u0930\u0940\u0930\u093f\u0915 \u0914\u0930 \u092e\u093e\u0928\u0938\u093f\u0915 \u0938\u094d\u0935\u093e\u0938\u094d\u0925\u094d\u092f \u0926\u094b\u0928\u094b\u0902 \u092e\u0947\u0902 \u0938\u0941\u0927\u093e\u0930 \u0939\u094b\u0924\u093e \u0939\u0948\u0964",
    "\u092a\u094d\u0930\u0924\u094d\u092f\u0947\u0915 \u0928\u093e\u0917\u0930\u093f\u0915 \u0915\u094b \u0905\u092a\u0928\u0947 \u0905\u0927\u093f\u0915\u093e\u0930\u094b\u0902 \u0914\u0930 \u0915\u0930\u094d\u0924\u0935\u094d\u092f\u094b\u0902 \u0915\u0947 \u092c\u093e\u0930\u0947 \u092e\u0947\u0902 \u091c\u093e\u0928\u0928\u093e \u091a\u093e\u0939\u093f\u090f\u0964",
    "\u092d\u093e\u0930\u0924\u0940\u092f \u0930\u0947\u0932\u0935\u0947 \u0926\u0941\u0928\u093f\u092f\u093e \u0915\u093e \u0938\u092c\u0938\u0947 \u092c\u0921\u093c\u093e \u0930\u0947\u0932\u0935\u0947 \u0928\u0947\u091f\u0935\u0930\u094d\u0915 \u092e\u0947\u0902 \u0938\u0947 \u090f\u0915 \u0939\u0948\u0964",
    "\u0939\u093f\u0902\u0926\u0940 \u092d\u093e\u0937\u093e \u092e\u0947\u0902 \u0938\u093e\u0939\u093f\u0924\u094d\u092f \u0915\u0940 \u090f\u0915 \u0938\u092e\u0943\u0926\u094d\u0927 \u092a\u0930\u0902\u092a\u0930\u093e \u0930\u0939\u0940 \u0939\u0948\u0964",
    "\u0924\u0915\u0928\u0940\u0915\u0940 \u0935\u093f\u0915\u093e\u0938 \u0928\u0947 \u0917\u094d\u0930\u093e\u092e\u0940\u0923 \u092d\u093e\u0930\u0924 \u0915\u0940 \u0924\u0938\u094d\u0935\u0940\u0930 \u092c\u0926\u0932 \u0926\u0940 \u0939\u0948\u0964",
    "\u092d\u093e\u0930\u0924 \u092e\u0947\u0902 \u0935\u093f\u092d\u093f\u0928\u094d\u0928 \u092a\u094d\u0930\u0915\u093e\u0930 \u0915\u0947 \u0924\u094d\u092f\u094b\u0939\u093e\u0930 \u092e\u0928\u093e\u090f \u091c\u093e\u0924\u0947 \u0939\u0948\u0902 \u091c\u094b \u090f\u0915\u0924\u093e \u0915\u093e \u092a\u094d\u0930\u0924\u0940\u0915 \u0939\u0948\u0902\u0964",
    "\u091c\u0932 \u0938\u0902\u0930\u0915\u094d\u0937\u0923 \u0906\u091c \u0915\u0947 \u0938\u092e\u092f \u0915\u0940 \u0938\u092c\u0938\u0947 \u092c\u0921\u093c\u0940 \u0906\u0935\u0936\u094d\u092f\u0915\u0924\u093e \u0939\u0948\u0964",
    "\u092d\u093e\u0930\u0924\u0940\u092f \u0915\u094d\u0930\u093f\u0915\u0947\u091f \u091f\u0940\u092e \u0928\u0947 \u0935\u093f\u0936\u094d\u0935 \u0915\u092a \u092e\u0947\u0902 \u0936\u093e\u0928\u0926\u093e\u0930 \u092a\u094d\u0930\u0926\u0930\u094d\u0936\u0928 \u0915\u093f\u092f\u093e \u0939\u0948\u0964",
    "\u0921\u093f\u091c\u093f\u091f\u0932 \u0907\u0902\u0921\u093f\u092f\u093e \u0905\u092d\u093f\u092f\u093e\u0928 \u0928\u0947 \u0926\u0947\u0936 \u092e\u0947\u0902 \u0924\u0915\u0928\u0940\u0915\u0940 \u0915\u094d\u0930\u093e\u0902\u0924\u093f \u0932\u093e \u0926\u0940 \u0939\u0948\u0964",
    "\u0936\u093f\u0915\u094d\u0937\u093e \u0915\u0947 \u092c\u093f\u0928\u093e \u0915\u093f\u0938\u0940 \u092d\u0940 \u0938\u092e\u093e\u091c \u0915\u093e \u0935\u093f\u0915\u093e\u0938 \u0938\u0902\u092d\u0935 \u0928\u0939\u0940\u0902 \u0939\u0948\u0964",
    "\u092e\u0939\u093e\u0924\u094d\u092e\u093e \u0917\u093e\u0902\u0927\u0940 \u0928\u0947 \u0905\u0939\u093f\u0902\u0938\u093e \u0915\u0947 \u092e\u093e\u0930\u094d\u0917 \u092a\u0930 \u091a\u0932\u0915\u0930 \u0926\u0947\u0936 \u0915\u094b \u0906\u091c\u093e\u0926\u0940 \u0926\u093f\u0932\u093e\u0908\u0964",
    "\u0938\u094d\u0935\u091a\u094d\u091b \u092d\u093e\u0930\u0924 \u0905\u092d\u093f\u092f\u093e\u0928 \u0928\u0947 \u0932\u094b\u0917\u094b\u0902 \u092e\u0947\u0902 \u0938\u094d\u0935\u091a\u094d\u091b\u0924\u093e \u0915\u0947 \u092a\u094d\u0930\u0924\u093f \u091c\u093e\u0917\u0930\u0942\u0915\u0924\u093e \u092c\u0922\u093c\u093e\u0908 \u0939\u0948\u0964",
    "\u092d\u093e\u0930\u0924 \u0915\u093e \u0938\u0902\u0935\u093f\u0927\u093e\u0928 \u0935\u093f\u0936\u094d\u0935 \u0915\u093e \u0938\u092c\u0938\u0947 \u092c\u0921\u093c\u093e \u0932\u093f\u0916\u093f\u0924 \u0938\u0902\u0935\u093f\u0927\u093e\u0928 \u0939\u0948\u0964",
    "\u0906\u092f\u0941\u0930\u094d\u0935\u0947\u0926 \u092d\u093e\u0930\u0924 \u0915\u0940 \u092a\u094d\u0930\u093e\u091a\u0940\u0928 \u091a\u093f\u0915\u093f\u0924\u094d\u0938\u093e \u092a\u0926\u094d\u0927\u0924\u093f \u0939\u0948 \u091c\u094b \u0906\u091c \u092d\u0940 \u092a\u094d\u0930\u093e\u0938\u0902\u0917\u093f\u0915 \u0939\u0948\u0964",
    "\u0928\u0908 \u0936\u093f\u0915\u094d\u0937\u093e \u0928\u0940\u0924\u093f \u0928\u0947 \u092d\u093e\u0930\u0924\u0940\u092f \u0936\u093f\u0915\u094d\u0937\u093e \u092a\u094d\u0930\u0923\u093e\u0932\u0940 \u092e\u0947\u0902 \u092e\u0939\u0924\u094d\u0935\u092a\u0942\u0930\u094d\u0923 \u092c\u0926\u0932\u093e\u0935 \u0915\u093f\u090f \u0939\u0948\u0902\u0964",
    "\u092d\u093e\u0930\u0924 \u092e\u0947\u0902 \u0938\u094c\u0930 \u090a\u0930\u094d\u091c\u093e \u0915\u093e \u0909\u092a\u092f\u094b\u0917 \u0924\u0947\u091c\u0940 \u0938\u0947 \u092c\u0922\u093c \u0930\u0939\u093e \u0939\u0948\u0964",
    "\u0939\u093f\u092e\u093e\u0932\u092f \u0915\u0940 \u091a\u094b\u091f\u093f\u092f\u093e\u0902 \u0926\u0941\u0928\u093f\u092f\u093e \u092d\u0930 \u0915\u0947 \u092a\u0930\u094d\u0935\u0924\u093e\u0930\u094b\u0939\u093f\u092f\u094b\u0902 \u0915\u094b \u0906\u0915\u0930\u094d\u0937\u093f\u0924 \u0915\u0930\u0924\u0940 \u0939\u0948\u0902\u0964",
    "\u092d\u093e\u0930\u0924\u0940\u092f \u0938\u093f\u0928\u0947\u092e\u093e \u0928\u0947 \u0935\u0948\u0936\u094d\u0935\u093f\u0915 \u0938\u094d\u0924\u0930 \u092a\u0930 \u0905\u092a\u0928\u0940 \u092a\u0939\u091a\u093e\u0928 \u092c\u0928\u093e\u0908 \u0939\u0948\u0964",
    "\u0915\u0943\u0924\u094d\u0930\u093f\u092e \u092c\u0941\u0926\u094d\u0927\u093f\u092e\u0924\u094d\u0924\u093e \u092d\u0935\u093f\u0937\u094d\u092f \u0915\u0940 \u0924\u0915\u0928\u0940\u0915 \u0939\u0948 \u091c\u094b \u0939\u0930 \u0915\u094d\u0937\u0947\u0924\u094d\u0930 \u092e\u0947\u0902 \u092c\u0926\u0932\u093e\u0935 \u0932\u093e\u090f\u0917\u0940\u0964",
    "\u0928\u0926\u093f\u092f\u094b\u0902 \u0915\u094b \u092a\u094d\u0930\u0926\u0942\u0937\u0923 \u0938\u0947 \u092c\u091a\u093e\u0928\u093e \u0939\u092e\u093e\u0930\u0940 \u0938\u093e\u092e\u0942\u0939\u093f\u0915 \u091c\u093f\u092e\u094d\u092e\u0947\u0926\u093e\u0930\u0940 \u0939\u0948\u0964",
    "\u092d\u093e\u0930\u0924 \u0935\u093f\u0936\u094d\u0935 \u0915\u093e \u0938\u092c\u0938\u0947 \u092c\u0921\u093c\u093e \u0932\u094b\u0915\u0924\u093e\u0902\u0924\u094d\u0930\u093f\u0915 \u0926\u0947\u0936 \u0939\u0948\u0964",
]


def setup_dirs():
    """Create dataset directories."""
    for split in [TRAIN_DIR, TEST_DIR]:
        for cls in ["human", "ai"]:
            (split / cls).mkdir(parents=True, exist_ok=True)
    print(f"[+] Directory structure created at {BASE_DIR}")


def download_human_audio():
    """Download human audio from FLEURS (English + Hindi), split into train/test."""
    print("\n" + "=" * 60)
    print("DOWNLOADING HUMAN AUDIO FROM FLEURS")
    print("=" * 60)

    transcripts = {"train": {"en": [], "hi": []}, "test": {"en": [], "hi": []}}

    # We need 25 per lang per split = 50 per lang total
    needed_per_lang = SAMPLES_PER_LANG_PER_SPLIT * 2  # train + test

    # --- English ---
    print(f"\n[*] Downloading {needed_per_lang} English human samples...")
    en_dataset = load_dataset("google/fleurs", "en_us", split="train", trust_remote_code=True)
    indices = list(range(len(en_dataset)))
    random.seed(42)
    random.shuffle(indices)
    indices = indices[:needed_per_lang]

    for i, idx in enumerate(tqdm(indices, desc="English human")):
        sample = en_dataset[idx]
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]
        transcript = sample.get("transcription", "")

        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95

        # First half -> train, second half -> test
        if i < SAMPLES_PER_LANG_PER_SPLIT:
            split_dir = TRAIN_DIR
            split_key = "train"
            file_idx = i
        else:
            split_dir = TEST_DIR
            split_key = "test"
            file_idx = i - SAMPLES_PER_LANG_PER_SPLIT

        filename = f"human_en_{file_idx:04d}.wav"
        sf.write(str(split_dir / "human" / filename), audio, TARGET_SR)
        transcripts[split_key]["en"].append(transcript)

    print(f"[+] Saved {needed_per_lang} English human samples (train+test)")

    # --- Hindi ---
    print(f"\n[*] Downloading {needed_per_lang} Hindi human samples...")
    hi_dataset = load_dataset("google/fleurs", "hi_in", split="train", trust_remote_code=True)
    indices = list(range(len(hi_dataset)))
    random.seed(43)
    random.shuffle(indices)
    indices = indices[:needed_per_lang]

    for i, idx in enumerate(tqdm(indices, desc="Hindi human")):
        sample = hi_dataset[idx]
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]
        transcript = sample.get("transcription", "")

        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95

        if i < SAMPLES_PER_LANG_PER_SPLIT:
            split_dir = TRAIN_DIR
            split_key = "train"
            file_idx = i
        else:
            split_dir = TEST_DIR
            split_key = "test"
            file_idx = i - SAMPLES_PER_LANG_PER_SPLIT

        filename = f"human_hi_{file_idx:04d}.wav"
        sf.write(str(split_dir / "human" / filename), audio, TARGET_SR)
        transcripts[split_key]["hi"].append(transcript)

    print(f"[+] Saved {needed_per_lang} Hindi human samples (train+test)")
    return transcripts


async def generate_ai_audio(transcripts):
    """Generate AI audio using edge-tts, matching the same split structure."""
    print("\n" + "=" * 60)
    print("GENERATING AI AUDIO USING EDGE-TTS")
    print("=" * 60)

    for split_name in ["train", "test"]:
        split_dir = TRAIN_DIR if split_name == "train" else TEST_DIR

        # --- English AI ---
        en_texts = transcripts[split_name].get("en", [])
        while len(en_texts) < SAMPLES_PER_LANG_PER_SPLIT:
            en_texts.extend(ENGLISH_SENTENCES)
        en_texts = en_texts[:SAMPLES_PER_LANG_PER_SPLIT]

        print(f"\n[*] Generating {SAMPLES_PER_LANG_PER_SPLIT} English AI ({split_name})...")
        generated = 0
        for i, text in enumerate(tqdm(en_texts, desc=f"AI en ({split_name})")):
            if not text or len(text.strip()) < 5:
                text = random.choice(ENGLISH_SENTENCES)

            voice = random.choice(ENGLISH_VOICES)
            final_path = split_dir / "ai" / f"ai_en_{i:04d}.wav"

            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tmp_path = tmp.name
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(tmp_path)
                audio, sr = librosa.load(tmp_path, sr=TARGET_SR)
                audio = audio.astype(np.float32)
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio)) * 0.95
                sf.write(str(final_path), audio, TARGET_SR)
                generated += 1
                os.unlink(tmp_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        print(f"[+] Generated {generated}/{SAMPLES_PER_LANG_PER_SPLIT} English AI ({split_name})")

        # --- Hindi AI ---
        hi_texts = transcripts[split_name].get("hi", [])
        while len(hi_texts) < SAMPLES_PER_LANG_PER_SPLIT:
            hi_texts.extend(HINDI_SENTENCES)
        hi_texts = hi_texts[:SAMPLES_PER_LANG_PER_SPLIT]

        print(f"\n[*] Generating {SAMPLES_PER_LANG_PER_SPLIT} Hindi AI ({split_name})...")
        generated = 0
        for i, text in enumerate(tqdm(hi_texts, desc=f"AI hi ({split_name})")):
            if not text or len(text.strip()) < 5:
                text = random.choice(HINDI_SENTENCES)

            voice = random.choice(HINDI_VOICES)
            final_path = split_dir / "ai" / f"ai_hi_{i:04d}.wav"

            try:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tmp_path = tmp.name
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(tmp_path)
                audio, sr = librosa.load(tmp_path, sr=TARGET_SR)
                audio = audio.astype(np.float32)
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio)) * 0.95
                sf.write(str(final_path), audio, TARGET_SR)
                generated += 1
                os.unlink(tmp_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        print(f"[+] Generated {generated}/{SAMPLES_PER_LANG_PER_SPLIT} Hindi AI ({split_name})")


def create_metadata():
    """Create metadata CSV covering all splits."""
    print("\n" + "=" * 60)
    print("CREATING METADATA")
    print("=" * 60)

    rows = []

    for split_name, split_dir in [("train", TRAIN_DIR), ("test", TEST_DIR)]:
        # Human
        for f in sorted((split_dir / "human").glob("*.wav")):
            lang = "en" if "_en_" in f.name else "hi"
            rows.append({
                "file_path": str(f),
                "file_name": f.name,
                "label": 0,
                "label_name": "human",
                "language": lang,
                "split": split_name,
            })
        # AI
        for f in sorted((split_dir / "ai").glob("*.wav")):
            lang = "en" if "_en_" in f.name else "hi"
            rows.append({
                "file_path": str(f),
                "file_name": f.name,
                "label": 1,
                "label_name": "ai",
                "language": lang,
                "split": split_name,
            })

    with open(METADATA_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file_path", "file_name", "label", "label_name", "language", "split"
        ])
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    for split in ["train", "test"]:
        s_rows = [r for r in rows if r["split"] == split]
        human = sum(1 for r in s_rows if r["label"] == 0)
        ai = sum(1 for r in s_rows if r["label"] == 1)
        en = sum(1 for r in s_rows if r["language"] == "en")
        hi = sum(1 for r in s_rows if r["language"] == "hi")
        print(f"\n  [{split.upper()}] Total: {len(s_rows)} | Human: {human} | AI: {ai} | EN: {en} | HI: {hi}")

    print(f"\n[+] Metadata: {METADATA_FILE} ({total} total files)")


def validate_dataset():
    """Quick validation."""
    print("\n" + "=" * 60)
    print("VALIDATING DATASET")
    print("=" * 60)

    issues = 0
    total = 0
    durations = []

    for split_dir in [TRAIN_DIR, TEST_DIR]:
        for cls_dir in [split_dir / "human", split_dir / "ai"]:
            for wav_file in sorted(cls_dir.glob("*.wav")):
                total += 1
                try:
                    info = sf.info(str(wav_file))
                    if info.samplerate != TARGET_SR:
                        print(f"  [!] Wrong SR: {wav_file.name} ({info.samplerate})")
                        issues += 1
                    if info.duration < 0.5:
                        print(f"  [!] Too short: {wav_file.name} ({info.duration:.1f}s)")
                        issues += 1
                    durations.append(info.duration)
                except Exception as e:
                    print(f"  [!] Error: {wav_file.name} ({e})")
                    issues += 1

    if durations:
        print(f"\n[+] Validated {total} files, {issues} issues")
        print(f"    Duration: {min(durations):.1f}s - {max(durations):.1f}s (mean {np.mean(durations):.1f}s)")
        print(f"    Total audio: {sum(durations)/60:.1f} minutes")


def main():
    print("=" * 60)
    print("  AI VOICE DETECTOR - DATASET COLLECTION")
    print("  200 files: 50 human + 50 AI (train) / 50 human + 50 AI (test)")
    print("  Each split: 25 English + 25 Hindi per class")
    print("=" * 60)

    random.seed(42)
    setup_dirs()
    transcripts = download_human_audio()
    asyncio.run(generate_ai_audio(transcripts))
    create_metadata()
    validate_dataset()

    print("\n" + "=" * 60)
    print("  DATASET COLLECTION COMPLETE!")
    print("=" * 60)
    print(f"\nDataset: {BASE_DIR}")
    print(f"  train/human/ - 50 files (25 en + 25 hi)")
    print(f"  train/ai/    - 50 files (25 en + 25 hi)")
    print(f"  test/human/  - 50 files (25 en + 25 hi)")
    print(f"  test/ai/     - 50 files (25 en + 25 hi)")
    print(f"\nNext: python train.py")


if __name__ == "__main__":
    main()
