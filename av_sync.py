"""
Cross-Modal Audio-Video Synchronization & Dubbing Mismatch Analyzer.
Correlates mouth vertical aperture / motion dynamics with speech audio energy envelope.
Flags audio-video dubbing latency and identifies desynchronized temporal segments.
"""

from __future__ import annotations
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple
from scipy import signal

class AudioVideoSyncAnalyzer:
    """
    Transparent cross-modal temporal alignment analyzer.
    Extracts visual mouth velocity and compares with acoustic speech envelope.
    """

    def __init__(self, window_sec: float = 1.5, sample_rate: int = 16000):
        self.window_sec = window_sec
        self.sample_rate = sample_rate

    def compute_mouth_motion_series(self, mouth_crops: List[np.ndarray]) -> np.ndarray:
        """
        Calculates frame-to-frame optical flow / intensity change in mouth region.
        """
        if len(mouth_crops) < 2:
            return np.zeros(max(1, len(mouth_crops)))

        velocities = []
        prev_gray = cv2.cvtColor(mouth_crops[0], cv2.COLOR_BGR2GRAY) if mouth_crops[0].ndim == 3 else mouth_crops[0]

        for i in range(1, len(mouth_crops)):
            curr = mouth_crops[i]
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) if curr.ndim == 3 else curr

            # Calculate frame difference in vertical aperture region
            diff = cv2.absdiff(curr_gray, prev_gray)
            motion = float(np.mean(diff))
            velocities.append(motion)
            prev_gray = curr_gray

        velocities.insert(0, velocities[0] if velocities else 0.0)
        motion_arr = np.array(velocities, dtype=np.float32)

        # Normalize to unit variance
        if np.std(motion_arr) > 1e-4:
            motion_arr = (motion_arr - np.mean(motion_arr)) / np.std(motion_arr)
        return motion_arr

    def compute_audio_energy_envelope(self, audio_signal: np.ndarray, num_frames: int, duration_sec: float) -> np.ndarray:
        """
        Extracts speech RMS energy envelope resampled to match video frame count.
        """
        if len(audio_signal) == 0 or num_frames == 0:
            return np.zeros(num_frames)

        # Envelope via Hilbert transform magnitude or squared energy
        analytic = signal.hilbert(audio_signal)
        envelope = np.abs(analytic)

        # Resample envelope to exact video frame count
        resampled_env = signal.resample(envelope, num_frames)
        resampled_env = np.maximum(0, resampled_env)

        if np.std(resampled_env) > 1e-4:
            resampled_env = (resampled_env - np.mean(resampled_env)) / np.std(resampled_env)
        return resampled_env

    def analyze_synchronization(
        self,
        mouth_crops: List[np.ndarray],
        audio_signal: np.ndarray,
        fps: float,
        duration_sec: float
    ) -> Dict[str, Any]:
        """
        Computes normalized cross-correlation between visual speech and acoustic energy.
        Returns:
            sync_score: 0 to 100
            lag_ms: estimated time shift between lips and voice
            mismatch_segments: List of flagged time ranges
        """
        n_frames = len(mouth_crops)
        if n_frames < 6 or len(audio_signal) < self.sample_rate:
            return {
                "sync_score": 75.0,
                "status": "INSUFFICIENT_DATA",
                "estimated_lag_ms": 0.0,
                "mismatch_segments": [],
                "explanation": "Media clip too brief for cross-modal temporal correlation analysis."
            }

        # 1. Feature series
        v_motion = self.compute_mouth_motion_series(mouth_crops)
        a_energy = self.compute_audio_energy_envelope(audio_signal, n_frames, duration_sec)

        # 2. Global cross-correlation
        corr = signal.correlate(v_motion, a_energy, mode='full')
        lags = signal.correlation_lags(len(v_motion), len(a_energy), mode='full')

        best_idx = np.argmax(corr)
        best_lag_frames = lags[best_idx]
        best_lag_sec = best_lag_frames / fps
        best_lag_ms = round(best_lag_sec * 1000.0, 1)

        # Zero-lag correlation coefficient
        zero_lag_corr = float(np.corrcoef(v_motion, a_energy)[0, 1])
        if np.isnan(zero_lag_corr):
            zero_lag_corr = 0.5

        # 3. Local window discrepancy search
        # Find intervals where mouth moves vigorously but audio is silent, or vice versa
        mismatch_segments = []
        win_size = max(4, int(fps * 0.8)) # 0.8s window

        for i in range(0, n_frames - win_size, win_size // 2):
            v_sub = v_motion[i:i + win_size]
            a_sub = a_energy[i:i + win_size]
            sub_corr = float(np.corrcoef(v_sub, a_sub)[0, 1])

            # Discrepancy: negative correlation or stark energy/motion divergence
            if sub_corr < -0.20 or (np.mean(v_sub) > 1.2 and np.mean(a_sub) < -0.6):
                start_sec = round(i / fps, 2)
                end_sec = round((i + win_size) / fps, 2)
                mismatch_segments.append({
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "time_window": f"{int(start_sec//60):02d}:{start_sec%60:04.1f} - {int(end_sec//60):02d}:{end_sec%60:04.1f}",
                    "local_correlation": round(sub_corr, 2),
                    "reason": "Mouth articulatory motion decoupled from acoustic phoneme envelope."
                })

        # Calculate final sync score (0 - 100)
        # Higher positive correlation + lower absolute lag = higher score
        lag_penalty = min(40.0, abs(best_lag_ms) * 0.12)
        corr_component = max(0.0, (zero_lag_corr + 1.0) / 2.0) * 100.0
        mismatch_penalty = min(30.0, len(mismatch_segments) * 7.5)

        raw_sync = corr_component - lag_penalty - mismatch_penalty
        sync_score = float(np.clip(raw_sync, 5.0, 98.0))

        if sync_score >= 70:
            status = "SYNCHRONIZED"
            explanation = f"Speech and lip movements are aligned within natural tolerance (offset: {best_lag_ms} ms)."
        elif sync_score >= 45:
            status = "SUSPICIOUS_DESYNC"
            explanation = f"Noticeable phase misalignment detected between vocal acoustics and lip movement (offset: {best_lag_ms} ms)."
        else:
            status = "SEVERE_MISMATCH"
            explanation = f"Substantial articulatory-acoustic mismatch detected. High likelihood of synthetic dubbing or face reenactment."

        return {
            "sync_score": round(sync_score, 1),
            "status": status,
            "estimated_lag_ms": best_lag_ms,
            "zero_lag_correlation": round(zero_lag_corr, 3),
            "mismatch_segments": mismatch_segments,
            "explanation": explanation
        }
