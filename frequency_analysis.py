"""
Frequency-Domain Digital Forensics (2D FFT and DCT Analysis).
Detects periodic upsampling artifacts, checkerboard GAN patterns,
and unnatural high-frequency power spectrum roll-off slopes.
"""

from __future__ import annotations
import numpy as np
import cv2
from scipy import fftpack
from typing import Dict, Any, Tuple

class FrequencyForensics:
    """
    Analyzes 2D Discrete Fourier Transform (FFT) and Discrete Cosine Transform (DCT)
    for generative synthesis and resampling fingerprints.
    """

    def __init__(self, high_pass_cutoff: float = 0.15):
        self.high_pass_cutoff = high_pass_cutoff

    def compute_fft_spectrum(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Computes 2D FFT magnitude spectrum and radial profile.
        Returns:
            magnitude_spectrum_vis: 8-bit image for UI visualization
            radial_profile: 1D array of radially averaged energy
            spectral_anomaly_score: float [0, 1] indicating deviation from natural 1/f law
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Windowing to reduce boundary leakage
        window = np.outer(np.hanning(h), np.hanning(w))
        windowed_gray = gray.astype(float) * window

        # 2D FFT and shift zero-frequency to center
        f = np.fft.fft2(windowed_gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        # Log magnitude for display
        log_magnitude = np.log1p(magnitude)
        norm_mag = cv2.normalize(log_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        mag_vis = norm_mag.astype(np.uint8)

        # Radial profile calculation
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.hypot(x - cx, y - cy).astype(int)

        max_r = min(cy, cx)
        radial_sum = np.bincount(r.ravel(), weights=magnitude.ravel())[:max_r]
        radial_count = np.bincount(r.ravel())[:max_r]
        radial_profile = np.divide(radial_sum, np.maximum(radial_count, 1))

        # Check for unnatural peaks in high frequencies (GAN periodic upsampling artifacts)
        cutoff_idx = int(max_r * self.high_pass_cutoff)
        high_freq_profile = radial_profile[cutoff_idx:]

        if len(high_freq_profile) > 10:
            freqs = np.arange(cutoff_idx, max_r)
            valid = (high_freq_profile > 1e-6) & (freqs > 0)
            if np.sum(valid) > 5:
                log_f = np.log(freqs[valid])
                log_p = np.log(high_freq_profile[valid])
                poly = np.polyfit(log_f, log_p, 1)
                slope = poly[0]  # Natural camera: -1.5 to -2.2

                # AI/GAN images: slope too flat (diffusion ~-0.8) or too steep (GAN ~-3.5)
                slope_dev = abs(slope - (-1.8))
                detrended = log_p - (poly[0] * log_f + poly[1])
                peak_metric = float(np.max(np.abs(detrended)))

                # Increased sensitivity: slope_dev weight raised, peak_metric raised
                anomaly_score = float(np.clip((slope_dev * 0.40) + (peak_metric * 0.45), 0.0, 1.0))
            else:
                anomaly_score = 0.3
        else:
            anomaly_score = 0.3

        return mag_vis, radial_profile, anomaly_score

    def compute_dct_block_metrics(self, img_bgr: np.ndarray) -> float:
        """
        Computes 8x8 block DCT coefficients to detect double JPEG compression or resampling.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape

        h_pad = (h // 8) * 8
        w_pad = (w // 8) * 8
        if h_pad < 8 or w_pad < 8:
            return 0.0

        cropped = gray[:h_pad, :w_pad]
        blocks = cropped.reshape(h_pad // 8, 8, -1, 8).swapaxes(1, 2)

        # Compute 2D DCT on sample of blocks
        sample_blocks = blocks[::2, ::2]
        dct_coeffs = fftpack.dct(fftpack.dct(sample_blocks, axis=-1, norm='ortho'), axis=-2, norm='ortho')

        # High frequency energy ratio
        high_freq_energy = np.mean(np.abs(dct_coeffs[:, :, 4:, 4:]))
        total_energy = np.mean(np.abs(dct_coeffs)) + 1e-5

        ratio = high_freq_energy / total_energy
        dct_anomaly = float(np.clip(ratio * 3.0, 0.0, 1.0))
        return dct_anomaly

    def analyze(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Full frequency inspection returning visualization map and quantitative anomaly metrics.
        """
        mag_vis, radial_profile, fft_anomaly = self.compute_fft_spectrum(img_bgr)
        dct_anomaly = self.compute_dct_block_metrics(img_bgr)

        combined_score = float(0.65 * fft_anomaly + 0.35 * dct_anomaly)

        if combined_score > 0.50:
            evidence_note = "High-frequency periodic spectral spikes detected (indicative of GAN/diffusion upsampling)."
        elif combined_score > 0.30:
            evidence_note = "Minor frequency distribution irregularities detected."
        else:
            evidence_note = "Frequency spectrum conforms to natural optical camera 1/f power law distribution."

        return {
            "frequency_anomaly_score": combined_score,
            "fft_anomaly": fft_anomaly,
            "dct_anomaly": dct_anomaly,
            "evidence_note": evidence_note,
            "spectrum_visualization": mag_vis
        }
