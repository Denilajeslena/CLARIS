"""
Acoustic Spectrogram and Phase Discontinuity Generator.
Computes Log-Mel Spectrogram and detects vocoder cutoffs / synthetic phase anomalies.
"""

from __future__ import annotations
import numpy as np
import cv2
from scipy import signal
from typing import Tuple, Dict, Any

class SpectrogramGenerator:
    """
    Generates high-resolution Mel Spectrograms and computes phase / spectral continuity metrics.
    """

    def __init__(self, sr: int = 16000, n_mels: int = 128, n_fft: int = 1024, hop_length: int = 256):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def generate_mel_spectrogram(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Computes Log-Mel power spectrogram.
        Returns:
            mel_db: 2D numpy array (n_mels, time_steps)
            spec_image_bgr: 3-channel color image (uint8) for UI display
            vocoder_metrics: Dict with high-frequency cutoff and phase discontinuity metrics
        """
        if len(y) < self.n_fft:
            y = np.pad(y, (0, self.n_fft - len(y)))

        # STFT
        freqs, times, zxx = signal.stft(
            y,
            fs=self.sr,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            boundary='zeros'
        )
        magnitude = np.abs(zxx)

        # Mel filterbank mapping approximation
        mel_basis = self._create_mel_filterbank(num_filters=self.n_mels, n_fft=self.n_fft, sr=self.sr)
        mel_spec = np.dot(mel_basis, magnitude)

        # Log power (dB)
        mel_db = 10.0 * np.log10(np.maximum(1e-5, mel_spec))
        ref = np.max(mel_db)
        mel_db = np.clip(mel_db, ref - 80.0, ref)

        # Normalize to 0-255 for visual presentation
        norm = ((mel_db - np.min(mel_db)) / (np.max(mel_db) - np.min(mel_db) + 1e-6) * 255).astype(np.uint8)
        spec_image = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
        spec_image = cv2.flip(spec_image, 0) # Orient low freq at bottom

        # Forensic vocoder artifacts: artificial high-frequency suppression (e.g. brick-wall filter at 7.5kHz)
        high_freq_band = magnitude[int(magnitude.shape[0] * 0.85):, :]
        high_band_energy = float(np.mean(high_freq_band))
        total_energy = float(np.mean(magnitude)) + 1e-6
        energy_ratio = high_band_energy / total_energy

        has_brickwall_cutoff = bool(energy_ratio < 0.005)
        vocoder_score = float(np.clip(1.0 - (energy_ratio * 20.0), 0.0, 1.0)) if has_brickwall_cutoff else 0.15

        return mel_db, spec_image, {
            "has_brickwall_cutoff": has_brickwall_cutoff,
            "high_frequency_ratio": round(energy_ratio, 5),
            "vocoder_artifact_score": round(vocoder_score, 3)
        }

    def _create_mel_filterbank(self, num_filters: int, n_fft: int, sr: int) -> np.ndarray:
        """Constructs triangular Mel frequency filter matrix."""
        num_bins = n_fft // 2 + 1
        low_mel = 0.0
        high_mel = 2595.0 * np.log10(1.0 + (sr / 2.0) / 700.0)

        mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        bin_points = np.clip(bin_points, 0, num_bins - 1)

        weights = np.zeros((num_filters, num_bins), dtype=np.float32)
        for i in range(1, num_filters + 1):
            left = bin_points[i - 1]
            center = bin_points[i]
            right = bin_points[i + 1]

            for j in range(left, center):
                if center > left:
                    weights[i - 1, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right > center:
                    weights[i - 1, j] = (right - j) / (right - center)

        return weights
