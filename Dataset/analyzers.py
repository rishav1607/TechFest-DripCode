"""
Multi-Signal Audio Analyzers for AI Voice Detection
Provides WavLM embedding analysis, spectral artifact detection,
prosody analysis, and ensemble classification.
"""

import numpy as np
import torch
import torchaudio
import librosa
from transformers import WavLMModel, AutoFeatureExtractor

TARGET_SR = 16000
MAX_DURATION_SEC = 5
MAX_LENGTH = TARGET_SR * MAX_DURATION_SEC


def load_audio_numpy(audio_path: str) -> np.ndarray:
    """Load audio, convert to mono 16kHz, truncate/pad to 5s, normalize.
    Mirrors inference.py preprocessing exactly."""
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
        waveform = resampler(waveform)
    if waveform.shape[0] > MAX_LENGTH:
        waveform = waveform[:MAX_LENGTH]
    elif waveform.shape[0] < MAX_LENGTH:
        padding = MAX_LENGTH - waveform.shape[0]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    waveform = waveform.numpy()
    if np.max(np.abs(waveform)) > 0:
        waveform = waveform / np.max(np.abs(waveform))
    return waveform


# ============================================================
# WavLM Embedding Analyzer
# ============================================================
class WavLMAnalyzer:
    """Analyzes audio using pre-trained WavLM hidden states.
    AI audio tends to have lower temporal variance and entropy."""

    def __init__(self, device: torch.device):
        self.device = device
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "microsoft/wavlm-base"
        )
        self.model = WavLMModel.from_pretrained(
            "microsoft/wavlm-base",
            torch_dtype=torch.float16,
            output_hidden_states=True,
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def analyze(self, audio_np: np.ndarray) -> dict:
        inputs = self.feature_extractor(
            audio_np, sampling_rate=TARGET_SR, return_tensors="pt", padding=False
        )
        input_values = inputs.input_values.to(self.device, dtype=torch.float16)

        outputs = self.model(input_values=input_values)
        last_hidden = outputs.last_hidden_state.float().squeeze(0)  # (T, 768)

        # Temporal variance: mean variance across time for each dimension
        temporal_var = last_hidden.var(dim=0).mean().item()

        # Embedding entropy: discretize hidden values and compute Shannon entropy
        flat_values = last_hidden.cpu().numpy().flatten()
        hist, _ = np.histogram(flat_values, bins=50, density=True)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))

        # Layer divergence: cosine distance between layer 4 and layer 11
        layer4 = outputs.hidden_states[4].float().squeeze(0)
        layer11 = outputs.hidden_states[11].float().squeeze(0)
        cos_sim = torch.nn.functional.cosine_similarity(layer4, layer11, dim=-1)
        layer_divergence = (1.0 - cos_sim.mean()).item()

        # Map features to AI score via sigmoids (calibrated on dataset)
        # Lower entropy → more AI (human ~2.03, ai ~1.88)
        ent_score = 1.0 / (1.0 + np.exp(3.0 * (entropy - 1.96)))
        # Lower layer divergence → more AI (human ~0.730, ai ~0.721)
        div_score = 1.0 / (1.0 + np.exp(50.0 * (layer_divergence - 0.725)))

        ai_score = (ent_score + div_score) / 2.0

        return {
            "temporal_variance": round(float(temporal_var), 6),
            "entropy": round(float(entropy), 4),
            "layer_divergence": round(float(layer_divergence), 6),
            "ai_score": round(float(ai_score), 4),
        }


# ============================================================
# Spectral Artifact Analyzer
# ============================================================
class SpectralArtifactAnalyzer:
    """CPU-based spectral analysis to detect AI audio artifacts.
    Checks for frequency cutoffs, spectral consistency, phase issues."""

    def analyze(self, audio_np: np.ndarray) -> dict:
        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=audio_np)
        flatness_mean = float(np.mean(flatness))

        # STFT for multiple features
        stft = np.abs(librosa.stft(audio_np, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=TARGET_SR, n_fft=2048)

        # High-frequency energy ratio (above 8kHz)
        hf_boundary_idx = np.searchsorted(freqs, 8000)
        total_energy = np.sum(stft ** 2)
        hf_energy = np.sum(stft[hf_boundary_idx:, :] ** 2)
        hf_ratio = float(hf_energy / (total_energy + 1e-10))

        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=audio_np, sr=TARGET_SR)
        bw_mean = float(np.mean(bandwidth))
        bw_std = float(np.std(bandwidth))

        # Spectral rolloff consistency
        rolloff = librosa.feature.spectral_rolloff(y=audio_np, sr=TARGET_SR)
        rolloff_std = float(np.std(rolloff))

        # Phase continuity (2nd-order phase difference)
        stft_complex = librosa.stft(audio_np, n_fft=2048, hop_length=512)
        phase = np.angle(stft_complex)
        if phase.shape[1] > 2:
            phase_diff2 = np.diff(phase, n=2, axis=1)
            phase_discontinuity = float(np.mean(np.abs(phase_diff2)))
        else:
            phase_discontinuity = 0.0

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_np)
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))

        # AI score from spectral features (calibrated on dataset)
        # Higher flatness → more AI (human ~0.047, ai ~0.119)
        flat_score = 1.0 / (1.0 + np.exp(-30.0 * (flatness_mean - 0.08)))
        # Higher bandwidth std → more AI (human ~454, ai ~629)
        bw_score = 1.0 / (1.0 + np.exp(-0.008 * (bw_std - 540)))
        # Higher rolloff std → more AI (human ~1749, ai ~2041)
        rolloff_score = 1.0 / (1.0 + np.exp(-0.005 * (rolloff_std - 1895)))

        ai_score = (flat_score + bw_score + rolloff_score) / 3.0

        return {
            "spectral_flatness": round(flatness_mean, 6),
            "hf_energy_ratio": round(hf_ratio, 6),
            "bandwidth_mean": round(bw_mean, 2),
            "bandwidth_std": round(bw_std, 2),
            "rolloff_std": round(rolloff_std, 2),
            "phase_discontinuity": round(phase_discontinuity, 6),
            "zcr_mean": round(zcr_mean, 6),
            "zcr_std": round(zcr_std, 6),
            "ai_score": round(float(ai_score), 4),
        }


# ============================================================
# Prosody Analyzer
# ============================================================
class ProsodyAnalyzer:
    """CPU-based prosody analysis using pitch tracking.
    Measures jitter, shimmer, and other voice quality features."""

    def analyze(self, audio_np: np.ndarray) -> dict:
        # Extract F0 with probabilistic YIN
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio_np,
            fmin=librosa.note_to_hz('C2'),   # ~65 Hz
            fmax=librosa.note_to_hz('C7'),   # ~2093 Hz
            sr=TARGET_SR,
        )

        voiced_f0 = f0[~np.isnan(f0)]

        if len(voiced_f0) < 5:
            return {
                "jitter": 0.0, "shimmer": 0.0, "f0_range_semitones": 0.0,
                "f0_cv": 0.0, "energy_smoothness": 0.0, "voiced_ratio": 0.0,
                "ai_score": 0.5,
            }

        # Jitter: relative average perturbation of consecutive F0 values
        f0_diffs = np.abs(np.diff(voiced_f0))
        jitter = float(np.mean(f0_diffs) / (np.mean(voiced_f0) + 1e-10))

        # Shimmer: amplitude perturbation
        rms = librosa.feature.rms(y=audio_np, frame_length=2048, hop_length=512)[0]
        amp_diffs = np.abs(np.diff(rms))
        shimmer = float(np.mean(amp_diffs) / (np.mean(rms) + 1e-10))

        # F0 range in semitones
        f0_min = np.min(voiced_f0)
        f0_max = np.max(voiced_f0)
        f0_range_st = float(12.0 * np.log2((f0_max + 1e-10) / (f0_min + 1e-10)))

        # F0 coefficient of variation
        f0_cv = float(np.std(voiced_f0) / (np.mean(voiced_f0) + 1e-10))

        # Energy contour smoothness
        energy_smoothness = float(np.mean(amp_diffs) / (np.mean(rms) + 1e-10))

        # Voiced segment ratio
        voiced_ratio = float(np.sum(~np.isnan(f0)) / len(f0))

        # AI score from prosody features (calibrated on dataset)
        # Higher jitter → more AI (human ~0.040, ai ~0.047)
        jitter_score = 1.0 / (1.0 + np.exp(-100.0 * (jitter - 0.043)))
        # Lower f0_cv → more AI (human ~0.221, ai ~0.208)
        f0cv_score = 1.0 / (1.0 + np.exp(10.0 * (f0_cv - 0.215)))
        # Higher voiced ratio → more AI (human ~0.679, ai ~0.776)
        voiced_score = 1.0 / (1.0 + np.exp(-8.0 * (voiced_ratio - 0.73)))

        ai_score = (jitter_score + f0cv_score + voiced_score) / 3.0

        return {
            "jitter": round(jitter, 6),
            "shimmer": round(shimmer, 6),
            "f0_range_semitones": round(f0_range_st, 2),
            "f0_cv": round(f0_cv, 6),
            "energy_smoothness": round(energy_smoothness, 6),
            "voiced_ratio": round(voiced_ratio, 4),
            "ai_score": round(float(ai_score), 4),
        }


# ============================================================
# Ensemble Classifier
# ============================================================
class EnsembleClassifier:
    """Confidence-gated ensemble: wav2vec2 is primary, other signals
    only influence borderline cases where wav2vec2 is uncertain."""

    WEIGHTS = {
        "wav2vec2": 0.70,
        "wavlm": 0.12,
        "spectral": 0.10,
        "prosody": 0.08,
    }

    def combine(
        self,
        wav2vec2_ai_prob: float,
        wavlm_result: dict,
        spectral_result: dict,
        prosody_result: dict,
    ) -> dict:
        scores = {
            "wav2vec2": wav2vec2_ai_prob,
            "wavlm": wavlm_result["ai_score"],
            "spectral": spectral_result["ai_score"],
            "prosody": prosody_result["ai_score"],
        }

        # wav2vec2 is the trained primary signal — start with its decision
        w2v_pred_ai = wav2vec2_ai_prob > 0.5

        # Count how many supplementary signals agree with wav2vec2
        supp_scores = [scores["wavlm"], scores["spectral"], scores["prosody"]]
        supp_ai_votes = sum(1 for s in supp_scores if s > 0.5)

        # Ensemble strategy: wav2vec2 decision stands, supplementary signals
        # adjust confidence. Override only when ALL 3 supplementary signals
        # strongly disagree (unanimous reversal).
        if w2v_pred_ai:
            # wav2vec2 says AI
            boost = supp_ai_votes * 0.03  # each agreeing signal adds confidence
            final_score = min(0.999, wav2vec2_ai_prob + boost)
        else:
            # wav2vec2 says Human
            boost = (3 - supp_ai_votes) * 0.03  # each agreeing-human signal
            final_score = max(0.001, wav2vec2_ai_prob - boost)

        # Only override wav2vec2 if it's borderline AND all 3 signals disagree
        w2v_confidence = abs(wav2vec2_ai_prob - 0.5)
        if w2v_confidence < 0.05 and supp_ai_votes == 3 and not w2v_pred_ai:
            # wav2vec2 barely says human but all signals say AI — flip to AI
            final_score = 0.55
        elif w2v_confidence < 0.05 and supp_ai_votes == 0 and w2v_pred_ai:
            # wav2vec2 barely says AI but all signals say human — flip to human
            final_score = 0.45

        final_label = "ai" if final_score > 0.5 else "human"
        final_confidence = max(final_score, 1.0 - final_score)

        # Agreement ratio
        ai_votes = sum(1 for s in scores.values() if s > 0.5)
        agreement_ratio = max(ai_votes, 4 - ai_votes) / 4.0

        return {
            "final_prediction": final_label,
            "final_confidence": round(float(final_confidence), 4),
            "ensemble_ai_score": round(float(final_score), 4),
            "signal_scores": {k: round(v, 4) for k, v in scores.items()},
            "signal_weights": {k: round(v, 4) for k, v in self.WEIGHTS.items()},
            "signal_agreement": round(float(agreement_ratio), 2),
        }
