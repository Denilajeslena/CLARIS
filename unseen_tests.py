"""
Unseen & Out-of-Distribution Manipulation Generalization Benchmarking.
Evaluates forensic classifiers on completely held-out manipulation techniques
(e.g., Diffusion inpainting vs FaceSwap vs NeuralTextures) to measure zero-shot generalization.
"""

from __future__ import annotations
from typing import Dict, Any, List, Callable
import numpy as np
from backend.evaluation.metrics import ForensicMetricsCalculator

class UnseenGeneralizationTester:
    """
    Evaluates detector performance split by known vs held-out unseen manipulation architectures.
    """

    def __init__(self, known_categories: List[str], unseen_categories: List[str]):
        self.known_categories = known_categories
        self.unseen_categories = unseen_categories

    def benchmark_generalization(
        self,
        samples: List[Dict[str, Any]], # [{"image": np.ndarray, "label": int, "category": str}]
        predict_fn: Callable[[np.ndarray], float]
    ) -> Dict[str, Any]:
        """
        Splits dataset into Known vs Unseen manipulation techniques and reports comparative metrics.
        """
        known_y_true, known_preds = [], []
        unseen_y_true, unseen_preds = [], []
        category_breakdown = {}

        for s in samples:
            img = s["image"]
            label = s["label"]
            cat = s["category"]

            pred_prob = predict_fn(img)

            if cat not in category_breakdown:
                category_breakdown[cat] = {"y_true": [], "y_pred": []}
            category_breakdown[cat]["y_true"].append(label)
            category_breakdown[cat]["y_pred"].append(pred_prob)

            if cat in self.known_categories or label == 0:
                known_y_true.append(label)
                known_preds.append(pred_prob)

            if cat in self.unseen_categories or label == 0:
                unseen_y_true.append(label)
                unseen_preds.append(pred_prob)

        # Compute aggregate metrics
        known_metrics = ForensicMetricsCalculator.compute_all_metrics(known_y_true, known_preds)
        unseen_metrics = ForensicMetricsCalculator.compute_all_metrics(unseen_y_true, unseen_preds)

        # Per-category metrics
        per_category = {}
        for cat, data in category_breakdown.items():
            if len(data["y_true"]) >= 2 and len(np.unique(data["y_true"])) > 1:
                per_category[cat] = ForensicMetricsCalculator.compute_all_metrics(data["y_true"], data["y_pred"])
            else:
                mean_p = float(np.mean(data["y_pred"]))
                per_category[cat] = {"mean_predicted_manipulation": round(mean_p, 4), "sample_count": len(data["y_true"])}

        # Generalization drop calculation (ROC-AUC drop)
        generalization_gap = round(known_metrics["roc_auc"] - unseen_metrics["roc_auc"], 4)

        return {
            "known_manipulations": {
                "categories": self.known_categories,
                "metrics": known_metrics
            },
            "unseen_manipulations": {
                "categories": self.unseen_categories,
                "metrics": unseen_metrics
            },
            "generalization_gap_auc": generalization_gap,
            "per_category_breakdown": per_category,
            "verdict": (
                "Robust Generalization Retained" if generalization_gap < 0.12 else
                "Moderate Domain Shift Observed on Novel Generative Techniques"
            )
        }
