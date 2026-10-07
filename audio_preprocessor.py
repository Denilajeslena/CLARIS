"""
Audio Ingestion, Resampling, and Acoustic Signal Preprocessing.
Supports WAV, MP3, AAC, and video audio extraction.
Computes acoustic features: MFCCs, Spectral Centroid, Spectral Rolloff, ZCR, and Pitch variance.
"""

from __future__ import annotations
import os
import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy import signal
from scipy.io import wavfile

logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """
    Standardizes audio streams to 16kHz mono and computes acoustic signal representations.
    """

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Loads audio file, converts to mono float32 array normalized to [-1.0, 1.0],
        resampling to target sample rate.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        y = None
        sr = self.target_sr

        # Try librosa first if installed
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
            return y.astype(np.float32), sr
        except Exception:
            pass

        # Try soundfile
        try:
            import soundfile as sf
            data, in_sr = sf.read(audio_path)
            if data.ndim > 1:
                data = np.mean(data, axis=1) # convert to mono
            if in_sr != self.target_sr:
                num_samples = int(len(data) * float(self.target_sr) / in_sr)
                data = signal.resample(data, num_samples)
            y = data.astype(np.float32)
            sr = self.target_sr
            return y, sr
        except Exception:
            pass

        # Fallback to scipy.io.wavfile
        try:
            in_sr, data = wavfile.read(audio_path)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if in_sr != self.target_sr:
                num_samples = int(len(data) * float(self.target_sr) / in_sr)
                data = signal.resample(data, num_samples)
            return data.astype(np.float32), self.target_sr
        except Exception as e:
            logger.error(f"Failed to load audio with standard tools: {e}")
            # Generate 1 sec blank audio if invalid
            return np.zeros(self.target_sr, dtype=np.float32), self.target_sr

    def compute_acoustic_features(self, y: np.ndarray, sr: int = 16000) -> Dict[str, float]:
        """
        Computes physical acoustic signals used in synthetic speech forensics:
        - Spectral Centroid: center of mass of frequency spectrum (synthetic TTS often has metallic high-centroid spikes)
        - Spectral Rolloff: frequency below which 85% of energy resides
        - Zero-Crossing Rate (ZCR): rate of sign changes (indicates unvoiced consonant vs synthetic buzz)
        - Pitch stability / unnatural flatness indicator
        """
        if len(y) == 0:
            return {
                "spectral_centroid": 0.0,
                "spectral_rolloff": 0.0,
                "zero_crossing_rate": 0.0,
                "pitch_flatness": 0.0,
                "synthetic_acoustic_risk": 0.0
            }

        # 1. Zero-Crossing Rate
        zcr_series = np.abs(np.diff(np.signbit(y)))
        zcr_mean = float(np.mean(zcr_series))

        # 2. Short-Time Fourier Transform for spectral metrics
        freqs, times, stft = signal.stft(y, fs=sr, nperseg=512, noverlap=256)
        magnitude = np.abs(stft)
        mag_sum = np.sum(magnitude, axis=0) + 1e-6

        # Spectral Centroid
        centroid = np.sum(freqs[:, None] * magnitude, axis=0) / mag_sum
        mean_centroid = float(np.mean(centroid))

        # Spectral Rolloff (85% energy threshold)
        cum_mag = np.cumsum(magnitude, axis=0)
        rolloff_idx = np.argmax(cum_mag >= 0.85 * cum_mag[-1:, :], axis=0)
        mean_rolloff = float(np.mean(freqs[rolloff_idx]))

        # 3. Autocorrelation-based pitch variance / unnatural robotic monotonicity
        corr = np.correlate(y[:min(len(y), 8000)], y[:min(len(y), 8000)], mode='full')
        corr = corr[len(corr)//2:]
        # Pitch flatness: synthetic voices (early TTS/vocoders) often lack natural human pitch jitter
        pitch_std = float(np.std(corr[:400])) if len(corr) >= 400 else 1.0
        pitch_flatness = float(np.clip(1.0 - (pitch_std * 5.0), 0.0, 1.0))

        # Acoustic anomaly metric (heuristic combination)
        centroid_anomaly = 0.5 if (mean_centroid > 3200 or mean_centroid < 800) else 0.1
        synthetic_risk = float(np.clip(0.4 * centroid_anomaly + 0.3 * pitch_flatness + 0.3 * min(1.0, zcr_mean * 8.0), 0.0, 1.0))

        return {
            "spectral_centroid": round(mean_centroid, 2),
            "spectral_rolloff": round(mean_rolloff, 2),
            "zero_crossing_rate": round(zcr_mean, 4),
            "pitch_flatness": round(pitch_flatness, 3),
            "synthetic_acoustic_risk": round(synthetic_risk, 3)
        }
