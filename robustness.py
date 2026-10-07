"""
Perturbation and Robustness Evaluation Engine.
Tests forensic model stability against real-world social media degradations:
WhatsApp/Telegram compression, aggressive downscaling, cropping, and noise injection.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, Any, Callable, List
from backend.evaluation.metrics import ForensicMetricsCalculator

class RobustnessEvaluator:
    """
    Simulates real-world transmission channels and measures degradation in forensic accuracy.
    """

    @staticmethod
    def apply_jpeg_compression(img_bgr: np.ndarray, quality: int = 40) -> np.ndarray:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc = cv2.imencode('.jpg', img_bgr, encode_param)
        return cv2.imdecode(enc, 1)

    @staticmethod
    def apply_resizing(img_bgr: np.ndarray, scale: float = 0.5) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def apply_crop(img_bgr: np.ndarray, crop_pct: float = 0.15) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        cy1, cy2 = int(h * crop_pct), int(h * (1.0 - crop_pct))
        cx1, cx2 = int(w * crop_pct), int(w * (1.0 - crop_pct))
        cropped = img_bgr[cy1:cy2, cx1:cx2]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def apply_gaussian_noise(img_bgr: np.ndarray, sigma: float = 12.0) -> np.ndarray:
        noise = np.random.normal(0, sigma, img_bgr.shape)
        noisy = np.clip(img_bgr.astype(float) + noise, 0, 255)
        return noisy.astype(np.uint8)

    def evaluate_model_robustness(
        self,
        test_images: List[np.ndarray],
        ground_truth: List[int],
        predict_fn: Callable[[np.ndarray], float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Runs evaluation across clean images and 4 distortion conditions.
        Returns metrics per perturbation condition.
        """
        conditions = {
            "Original (Clean)": lambda img: img,
            "JPEG Compression (Q=40)": lambda img: self.apply_jpeg_compression(img, 40),
            "Social Media Downscaling (0.5x)": lambda img: self.apply_resizing(img, 0.5),
            "Centric Cropping (15%)": lambda img: self.apply_crop(img, 0.15),
            "Sensor Noise (sigma=12)": lambda img: self.apply_gaussian_noise(img, 12.0)
        }

        results = {}
        for cond_name, transform in conditions.items():
            probs = []
            for img in test_images:
                perturbed = transform(img)
                prob_manip = predict_fn(perturbed)
                probs.append(prob_manip)

            metrics_res = ForensicMetricsCalculator.compute_all_metrics(ground_truth, probs)
            results[cond_name] = metrics_res

        return results
