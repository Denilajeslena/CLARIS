"""
Authenticity Scoring and Calibrated Decision Classifier.
Defines risk thresholds, scientific uncertainty quantification,
and forensic disclaimer generation.
"""

from __future__ import annotations
from typing import Dict, Any, List

class ForensicScorer:
    """
    Computes final decision, evidence strength, calibrated uncertainty,
    and formats JSON reports compliant with scientific digital forensic standards.
    """

    def __init__(self, authentic_threshold: float = 68.0, suspicious_threshold: float = 38.0):
        self.auth_thresh = authentic_threshold
        self.susp_thresh = suspicious_threshold

    def calculate_assessment(
        self,
        fusion_result: Dict[str, Any],
        evidence_list: List[Dict[str, Any]],
        media_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Creates full scientific assessment dictionary with legal/forensic caveats.
        """
        score = fusion_result["authenticity_score"]
        conf = fusion_result["model_confidence"]
        uncertainty = fusion_result["uncertainty_percent"]
        decision = fusion_result["final_decision"]

        # Calculate overall evidence strength
        critical_count = sum(1 for e in evidence_list if e.get("severity") == "CRITICAL")
        warning_count = sum(1 for e in evidence_list if e.get("severity") == "WARNING")

        if critical_count >= 2:
            evidence_strength = "STRONG_MANIPULATION_EVIDENCE"
        elif critical_count == 1 or warning_count >= 2:
            evidence_strength = "MODERATE_ANOMALY_EVIDENCE"
        elif warning_count == 1:
            evidence_strength = "WEAK_CONTEXTUAL_SIGNAL"
        else:
            evidence_strength = "CONSISTENT_WITH_AUTHENTIC_CAPTURE"

        scientific_notice = (
            "SCIENTIFIC NOTICE: Deepfake and digital manipulation detection is probabilistic "
            "and based on learned statistical anomalies and signal processing signatures. "
            "Detection systems may exhibit reduced sensitivity toward novel, unseen generative "
            "architectures or heavily recompressed transmissions. Findings should corroborate broader chain-of-custody analysis."
        )

        return {
            "final_decision": decision,
            "authenticity_score": score,
            "model_authenticity_confidence": conf,
            "evidence_strength": evidence_strength,
            "uncertainty_percent": uncertainty,
            "models_disagree": fusion_result.get("disagreement_flag", False),
            "disagreement_notes": fusion_result.get("disagreement_notes", []),
            "modal_scores": fusion_result.get("modal_scores", {}),
            "evidence_count": len(evidence_list),
            "evidence_items": evidence_list,
            "media_metadata": media_metadata,
            "scientific_disclaimer": scientific_notice
        }
