"""
Comprehensive Forensic Evaluation Metrics Engine.
Computes ROC-AUC, Balanced Accuracy, F1-Score, Precision, Recall,
False Positive Rate (FPR), and False Negative Rate (FNR).
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Union
from sklearn import metrics

class ForensicMetricsCalculator:
    """
    Computes rigorous scientific metrics for deepfake detection systems.
    Prevents reporting skewed accuracy on imbalanced test sets.
    """

    @staticmethod
    def compute_all_metrics(
        y_true: Union[List[int], np.ndarray],
        y_pred_probs: Union[List[float], np.ndarray],
        threshold: float = 0.50
    ) -> Dict[str, float]:
        """
        Computes full suite of forensic performance indicators.
        y_true: 0 = Authentic, 1 = Manipulated
        y_pred_probs: Probability of manipulation [0, 1]
        """
        y_true = np.array(y_true, dtype=int)
        y_pred_probs = np.array(y_pred_probs, dtype=float)
        y_pred = (y_pred_probs >= threshold).astype(int)

        # Handle edge cases where only 1 class is present
        if len(np.unique(y_true)) < 2:
            return {
                "roc_auc": 0.5,
                "balanced_accuracy": 0.5,
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "fpr": 0.0,
                "fnr": 0.0
            }

        # Confusion Matrix
        # [[TN, FP],
        #  [FN, TP]]
        cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Rates
        fpr = fp / (fp + tn + 1e-7) # False alarm rate (authentic flagged as fake)
        fnr = fn / (fn + tp + 1e-7) # Missed fake rate (manipulation went undetected)
        recall = tp / (tp + fn + 1e-7) # Sensitivity / True Positive Rate
        precision = tp / (tp + fp + 1e-7)
        specificity = tn / (tn + fp + 1e-7)
        balanced_acc = (recall + specificity) / 2.0
        f1 = metrics.f1_score(y_true, y_pred, zero_division=0)

        try:
            roc_auc = metrics.roc_auc_score(y_true, y_pred_probs)
        except Exception:
            roc_auc = 0.5

        return {
            "roc_auc": round(float(roc_auc), 4),
            "balanced_accuracy": round(float(balanced_acc), 4),
            "f1_score": round(float(f1), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "fpr": round(float(fpr), 4),
            "fnr": round(float(fnr), 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        }
