"""
Compression and Re-encoding Forensics.
Implements Error Level Analysis (ELA) for image JPEG quantization inconsistencies,
detects video frame duplication / frozen frames, and computes compression artifact metrics.
"""

from __future__ import annotations
import os
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io
from typing import Dict, Any, Tuple

class CompressionAnalyzer:
    """
    Evaluates compression artifacts, ELA (Error Level Analysis), and re-encoding clues.
    """

    def __init__(self, ela_resave_quality: int = 90):
        self.ela_quality = ela_resave_quality

    def compute_ela(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculates Error Level Analysis (ELA) on image.
        Detects differences in compression levels between spliced regions and authentic backdrop.
        Returns:
            ela_visual_bgr: Enhanced ELA difference image
            ela_anomaly_score: float [0, 1]
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_orig = Image.fromarray(rgb)

        # Resave in-memory at defined JPEG quality
        buffer = io.BytesIO()
        pil_orig.save(buffer, format='JPEG', quality=self.ela_quality)
        buffer.seek(0)
        pil_resaved = Image.open(buffer)

        # Compute absolute difference
        diff = ImageChops.difference(pil_orig, pil_resaved)

        # Scale difference to highlight high error regions
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max(max_diff, 1)

        diff_enhanced = ImageEnhance.Brightness(diff).enhance(scale * 0.4)
        ela_np = np.array(diff_enhanced)
        ela_bgr = cv2.cvtColor(ela_np, cv2.COLOR_RGB2BGR)

        # Statistical variance in error levels across the image
        # Spliced deepfakes often show pronounced local patches with distinct error variance
        gray_diff = cv2.cvtColor(ela_bgr, cv2.COLOR_BGR2GRAY)
        std_dev = float(np.std(gray_diff))
        ela_score = float(np.clip(std_dev / 40.0, 0.0, 1.0))

        return ela_bgr, round(ela_score, 3)

    def detect_frame_duplication(self, frames_bgr: list[np.ndarray], diff_threshold: float = 1.5) -> Dict[str, Any]:
        """
        Detects repeated/frozen frames in video streams (often caused by frame rate manipulation).
        """
        if len(frames_bgr) < 2:
            return {"duplicated_frame_ratio": 0.0, "duplicate_count": 0}

        dup_count = 0
        for i in range(1, len(frames_bgr)):
            curr_gray = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(frames_bgr[i-1], cv2.COLOR_BGR2GRAY)
            mean_diff = float(np.mean(cv2.absdiff(curr_gray, prev_gray)))

            if mean_diff < diff_threshold:
                dup_count += 1

        dup_ratio = dup_count / (len(frames_bgr) - 1)

        return {
            "duplicate_count": dup_count,
            "duplicated_frame_ratio": round(dup_ratio, 3),
            "is_suspicious_stutter": dup_ratio > 0.25
        }

    def assess_compression_risk(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Provides holistic compression evaluation.
        Clearly notes that compression is contextual evidence, not proof of tampering.
        """
        ela_img, ela_score = self.compute_ela(img_bgr)

        return {
            "ela_score": ela_score,
            "compression_risk_score": round(ela_score * 0.7, 3),
            "forensic_context": (
                "Compression artifacts detected. Note: heavy social media re-encoding "
                "creates high compression noise, but is not definitive proof of malicious manipulation."
            )
        }
