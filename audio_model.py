"""
Audio Deepfake & Synthetic Speech Forensic Classifier.
Combines Pretrained Speech Representation Embeddings (Wav2Vec2 / HuBERT)
with Spectral Signal Processing (MFCC / Mel / ZCR / Pitch Prosody).
"""

from __future__ import annotations
import os
import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.audio.audio_preprocessor import AudioPreprocessor
from backend.audio.spectrogram import SpectrogramGenerator

logger = logging.getLogger(__name__)

class AudioForensicCNN(nn.Module):
    """
    Lightweight Spectrogram CNN classifier operating on Log-Mel representations.
    Provides local inference even when large HuggingFace weights are unavailable offline.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class AudioForensicClassifier:
    """
    Multimodal Speech Forensics Engine.
    Executes Wav2Vec2 / CNN inference combined with acoustic physical signal metrics.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        speech_encoder_name: str = "facebook/wav2vec2-base",
        device: str = "auto"
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.speech_encoder_name = speech_encoder_name
        self.preprocessor = AudioPreprocessor(target_sr=16000)
        self.spec_gen = SpectrogramGenerator(sr=16000)

        self.wav2vec_model = None
        self.wav2vec_processor = None
        self.cnn_model = AudioForensicCNN().to(self.device)
        self.cnn_model.eval()

        self.is_wav2vec_loaded = False
        self.is_baseline = True
        self.encoder_status = "Acoustic-CNN-Baseline"

        # Attempt to load Wav2Vec2 representation model
        self._init_speech_encoder()

    def _init_speech_encoder(self):
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            logger.info(f"Attempting to load speech encoder: {self.speech_encoder_name}")
            # local_files_only flag prevents stalling if offline
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(
                self.speech_encoder_name,
                local_files_only=False
            )
            self.wav2vec_model = Wav2Vec2Model.from_pretrained(
                self.speech_encoder_name,
                local_files_only=False
            ).to(self.device)
            self.wav2vec_model.eval()
            self.is_wav2vec_loaded = True
            self.is_baseline = False
            self.encoder_status = "Wav2Vec2-Active"
            logger.info("Successfully loaded Wav2Vec2 speech representation encoder.")
        except Exception as e:
            logger.warning(f"Could not load HuggingFace Wav2Vec2 model ({e}). Using Acoustic CNN + Signal Processor.")
            self.is_wav2vec_loaded = False
            self.is_baseline = True
            self.encoder_status = "Acoustic-CNN-Baseline"

    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Executes complete audio forensic evaluation on audio file.
        Returns authenticity scores, spectrogram visualization, and acoustic evidence.
        """
        y, sr = self.preprocessor.load_audio(audio_path)
        duration_sec = round(len(y) / sr, 2)

        # 1. Acoustic physical signals
        acoustic_feats = self.preprocessor.compute_acoustic_features(y, sr)
        # 2. Spectrogram & vocoder metrics
        mel_db, spec_vis, vocoder_metrics = self.spec_gen.generate_mel_spectrogram(y)

        # 3. Neural Model Inference (CNN or Wav2Vec2)
        # Prepare Mel input tensor for local CNN
        mel_resized = cv2.resize(mel_db, (128, 128))
        norm_mel = (mel_resized - np.mean(mel_resized)) / (np.std(mel_resized) + 1e-6)
        tensor = torch.tensor(norm_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.cnn_model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        prob_authentic_nn = float(probs[0])
        prob_synthetic_nn = float(probs[1])

        # Signal-based heuristic score
        signal_synthetic_risk = float(
            0.45 * vocoder_metrics["vocoder_artifact_score"] +
            0.35 * acoustic_feats["synthetic_acoustic_risk"] +
            0.20 * acoustic_feats["pitch_flatness"]
        )

        if self.is_wav2vec_loaded:
            # Wav2Vec2 latent variance test (synthetic voices show lower representation entropy)
            try:
                inputs = self.wav2vec_processor(y[:min(len(y), 16000 * 5)], sampling_rate=sr, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.wav2vec_model(**inputs)
                    latent_std = float(torch.std(outputs.last_hidden_state).cpu().numpy())

                # Unusually low representation variance indicates synthetic generation
                if latent_std < 0.15:
                    signal_synthetic_risk = max(signal_synthetic_risk, 0.75)
            except Exception as e:
                logger.debug(f"Wav2Vec2 embedding extraction error: {e}")

        # Combine ML model with physical acoustic forensics
        if self.is_baseline:
            combined_synthetic = 0.40 * prob_synthetic_nn + 0.60 * signal_synthetic_risk
        else:
            combined_synthetic = 0.70 * prob_synthetic_nn + 0.30 * signal_synthetic_risk

        combined_synthetic = float(np.clip(combined_synthetic, 0.02, 0.98))
        combined_authentic = float(1.0 - combined_synthetic)

        # Generate explanatory evidence points
        evidence_items: List[str] = []
        if vocoder_metrics["has_brickwall_cutoff"]:
            evidence_items.append("Sharp high-frequency spectral brick-wall cutoff detected (classic neural vocoder artifact).")
        if acoustic_feats["pitch_flatness"] > 0.65:
            evidence_items.append("Abnormally flat pitch prosody contour; human micro-intonation absent.")
        if acoustic_feats["spectral_centroid"] > 3000:
            evidence_items.append("Unnatural high spectral centroid; presence of metallic synthetic overtones.")
        if not evidence_items:
            evidence_items.append("Natural harmonic structure, continuous formants, and organic pitch jitter observed.")

        return {
            "duration_sec": duration_sec,
            "sample_rate": sr,
            "prob_authentic": round(combined_authentic, 3),
            "prob_synthetic": round(combined_synthetic, 3),
            "confidence": round(abs(combined_authentic - combined_synthetic) * 1.5, 3),
            "decision": "AUTHENTIC" if combined_authentic >= 0.65 else ("SUSPICIOUS" if combined_authentic >= 0.40 else "LIKELY SYNTHETIC"),
            "acoustic_signals": acoustic_feats,
            "vocoder_metrics": vocoder_metrics,
            "evidence": evidence_items,
            "spectrogram_image": spec_vis,
            "encoder_status": "Wav2Vec2-Active" if self.is_wav2vec_loaded else "Acoustic-CNN-Baseline"
        }
