"""
Temporal Inconsistency & Segment Analysis Engine.
Aggregates frame-level forensic predictions using robust statistical estimators
(median, trimmed mean, 90th percentile) and detects temporal jitter and suspicious time windows.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy import stats

class TemporalAnalyzer:
    """
    Analyzes sequence of frame-level forensic scores.
    Detects temporal jitter, flags suspicious time windows, and computes robust aggregation.
    """

    def __init__(self, aggregation_method: str = "trimmed_mean", trim_pct: float = 0.10):
        self.aggregation_method = aggregation_method
        self.trim_pct = trim_pct

    def aggregate_scores(self, frame_scores: List[float]) -> Tuple[float, str]:
        """
        Calculates robust aggregate score using robust statistics.
        Returns:
            aggregated_score: float in [0, 1] (authenticity score)
            explanation: string explaining the aggregation reasoning
        """
        if not frame_scores:
            return 0.5, "No frames available."

        arr = np.array(frame_scores)

        if self.aggregation_method == "median":
            score = float(np.median(arr))
            desc = f"Median filter (score: {score:.2f}) immune to single-frame outliers."
        elif self.aggregation_method == "percentile_10":
            # 10th percentile of authenticity (flags even short injected manipulated clips)
            score = float(np.percentile(arr, 10))
            desc = f"10th percentile lowest-authenticity aggregation ({score:.2f}) targeting localized spliced scenes."
        else: # trimmed_mean default
            score = float(stats.trim_mean(arr, self.trim_pct))
            desc = f"Trimmed mean ({int(self.trim_pct*100)}% tails excluded, score: {score:.2f}) balancing global stability and anomaly sensitivity."

        return score, desc

    def detect_suspicious_segments(
        self,
        timestamps: List[float],
        manip_probs: List[float],
        threshold: float = 0.55,
        min_consecutive_frames: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Finds contiguous temporal windows where manipulation probability exceeds threshold.
        Example output: [{"start": "00:04.2", "end": "00:06.8", "avg_manip": 0.88, "frames": 7}]
        """
        segments = []
        in_segment = False
        seg_start_idx = 0

        for i, prob in enumerate(manip_probs):
            if prob >= threshold:
                if not in_segment:
                    in_segment = True
                    seg_start_idx = i
            else:
                if in_segment:
                    in_segment = False
                    if (i - seg_start_idx) >= min_consecutive_frames:
                        segments.append(self._format_segment(timestamps, manip_probs, seg_start_idx, i - 1))

        # Check trailing segment
        if in_segment and (len(manip_probs) - seg_start_idx) >= min_consecutive_frames:
            segments.append(self._format_segment(timestamps, manip_probs, seg_start_idx, len(manip_probs) - 1))

        return segments

    def _format_segment(self, timestamps: List[float], manip_probs: List[float], start_idx: int, end_idx: int) -> Dict[str, Any]:
        start_t = timestamps[start_idx]
        end_t = timestamps[end_idx]
        mean_p = float(np.mean(manip_probs[start_idx:end_idx + 1]))

        def to_time_str(sec: float) -> str:
            m = int(sec // 60)
            s = sec % 60
            return f"{m:02d}:{s:04.1f}"

        return {
            "start_sec": start_t,
            "end_sec": end_t,
            "time_window": f"{to_time_str(start_t)} - {to_time_str(end_t)}",
            "average_manipulation_prob": round(mean_p, 3),
            "frame_count": (end_idx - start_idx + 1)
        }

    def compute_temporal_jitter(self, frame_scores: List[float]) -> float:
        """
        Calculates frame-to-frame derivative volatility (jerk/flickering).
        High jitter is a classic signature of per-frame deepfakes lacking temporal smoothing.
        """
        if len(frame_scores) < 3:
            return 0.0

        arr = np.array(frame_scores)
        diffs = np.abs(np.diff(arr))
        jitter_score = float(np.mean(diffs))
        return round(float(np.clip(jitter_score * 3.5, 0.0, 1.0)), 3)
