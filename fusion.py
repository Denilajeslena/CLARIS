"""
Multimodal Evidence Fusion and Decision Engine.
Fuses Vision, Audio, Cross-Modal AV Sync, Frequency, and Metadata signals
using Bayesian uncertainty-weighting. Exposes model disagreements transparently.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Optional

class MultimodalFusionEngine:
    """
    Combines heterogeneous forensic probabilities into a calibrated authenticity score (0-100),
    identifying modal conflicts and reporting explicit uncertainty bounds.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "vision": 0.50,
            "audio": 0.20,
            "cross_modal": 0.12,
            "frequency": 0.13,
            "compression": 0.05
        }

    def fuse(
        self,
        vision_res: Optional[Dict[str, Any]] = None,
        audio_res: Optional[Dict[str, Any]] = None,
        cross_modal_res: Optional[Dict[str, Any]] = None,
        freq_res: Optional[Dict[str, Any]] = None,
        metadata_res: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes calibrated fusion across all provided modality results.
        """
        active_modalities = {}
        disagreements = []

        # 1. Vision modality
        if vision_res:
            v_auth = vision_res.get("prob_authentic", vision_res.get("aggregate_authenticity", 0.5))
            v_conf = vision_res.get("confidence", 0.7)
            active_modalities["vision"] = {
                "auth_score": float(v_auth),
                "confidence": float(v_conf),
                "weight": self.weights["vision"]
            }

        # 2. Audio modality
        if audio_res:
            a_auth = audio_res.get("prob_authentic", 0.5)
            a_conf = audio_res.get("confidence", 0.7)
            active_modalities["audio"] = {
                "auth_score": float(a_auth),
                "confidence": float(a_conf),
                "weight": self.weights["audio"]
            }

        # 3. Cross-modal AV sync
        if cross_modal_res:
            sync_val = cross_modal_res.get("sync_score", 70.0) / 100.0
            active_modalities["cross_modal"] = {
                "auth_score": float(sync_val),
                "confidence": 0.75,
                "weight": self.weights["cross_modal"]
            }

        # 4. Frequency domain
        if freq_res:
            freq_anomaly = freq_res.get("frequency_anomaly_score", 0.3)
            f_auth = 1.0 - freq_anomaly
            active_modalities["frequency"] = {
                "auth_score": float(f_auth),
                "confidence": 0.65,
                "weight": self.weights["frequency"]
            }

        # 5. Metadata / Compression
        if metadata_res:
            c_score = metadata_res.get("compression_risk_score", 0.2)
            active_modalities["compression"] = {
                "auth_score": float(1.0 - c_score),
                "confidence": 0.50,
                "weight": self.weights["compression"]
            }

        if not active_modalities:
            return {
                "final_decision": "INSUFFICIENT_DATA",
                "authenticity_score": 50.0,
                "confidence": 0.0,
                "uncertainty": 100.0,
                "disagreement_flag": False
            }

        # Normalize active weights
        total_weight = sum(m["weight"] for m in active_modalities.values())
        weighted_sum = 0.0
        conf_sum = 0.0
        scores_list = []

        for name, m in active_modalities.items():
            norm_w = m["weight"] / total_weight
            weighted_sum += norm_w * m["auth_score"]
            conf_sum += norm_w * m["confidence"]
            scores_list.append((name, m["auth_score"]))

        # Check for modality disagreements (e.g. Vision says real, Audio says synthetic)
        if len(scores_list) > 1:
            max_s = max(s[1] for s in scores_list)
            min_s = min(s[1] for s in scores_list)
            score_spread = max_s - min_s

            if score_spread > 0.40:
                high_mod = [s[0] for s in scores_list if s[1] == max_s][0]
                low_mod = [s[0] for s in scores_list if s[1] == min_s][0]
                disagreements.append(
                    f"Modality conflict: {high_mod.upper()} suggests authenticity ({max_s*100:.0f}%) while {low_mod.upper()} detects manipulation ({min_s*100:.0f}%). Manual expert inspection advised."
                )

        # Pull score down aggressively if ANY modality strongly flags manipulation
        strong_flags = [s[1] for s in scores_list if s[1] < 0.40]
        if strong_flags:
            min_score = min(strong_flags)
            # Weighted pull: stronger the flag, harder the pull
            pull_strength = 0.60 + 0.20 * (1.0 - min_score / 0.40)
            weighted_sum = (1.0 - pull_strength) * weighted_sum + pull_strength * min_score

        authenticity_score = round(float(np.clip(weighted_sum * 100.0, 0.0, 100.0)), 1)
        mean_confidence = round(float(np.clip(conf_sum, 0.20, 0.98)), 2)

        # Uncertainty: higher when models disagree or when close to the 50% border
        decision_margin = abs(authenticity_score - 50.0) / 50.0
        spread_penalty = (max(s[1] for s in scores_list) - min(s[1] for s in scores_list)) if len(scores_list) > 1 else 0.0
        uncertainty_pct = round(float(np.clip((1.0 - decision_margin * 0.7 + spread_penalty * 0.4) * 100.0, 8.0, 92.0)), 1)

        # Decision category — tightened authentic threshold
        if authenticity_score >= 72.0:
            final_decision = "AUTHENTIC"
        elif authenticity_score >= 42.0:
            final_decision = "SUSPICIOUS"
        else:
            final_decision = "LIKELY MANIPULATED"

        return {
            "final_decision": final_decision,
            "authenticity_score": authenticity_score,
            "model_confidence": mean_confidence,
            "uncertainty_percent": uncertainty_pct,
            "disagreement_flag": len(disagreements) > 0,
            "disagreement_notes": disagreements,
            "modal_scores": {k: round(v["auth_score"] * 100.0, 1) for k, v in active_modalities.items()}
        }
