"""
Full Video Forensic Pipeline.
Orchestrates frame extraction, YOLO spatial ROI analysis, face detection,
forensic CNN evaluation, and temporal consistency tracking.
"""

from __future__ import annotations
import os
import logging
from typing import Dict, Any, List, Optional
import numpy as np

from backend.video.frame_extractor import VideoFrameExtractor, VideoFrame
from backend.video.temporal_analyzer import TemporalAnalyzer
from backend.vision.yolo_detector import YOLOSpatialDetector
from backend.vision.face_detector import FaceDetector
from backend.vision.forensic_classifier import ForensicClassifier

logger = logging.getLogger(__name__)

class VideoForensicsPipeline:
    """
    End-to-end video forensic analyzer with cached models and temporal aggregation.
    """

    def __init__(
        self,
        yolo_detector: YOLOSpatialDetector,
        face_detector: FaceDetector,
        forensic_classifier: ForensicClassifier,
        process_every_n_frames: int = 3,
        max_frames: int = 120
    ):
        self.yolo = yolo_detector
        self.face_detector = face_detector
        self.classifier = forensic_classifier
        self.frame_extractor = VideoFrameExtractor(
            process_every_n_frames=process_every_n_frames,
            max_frames=max_frames
        )
        self.temporal_analyzer = TemporalAnalyzer(aggregation_method="trimmed_mean", trim_pct=0.10)

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Executes full video forensics pipeline.
        Returns detailed frame-by-frame timeline, suspicious intervals, and aggregate authenticity score.
        """
        metadata = self.frame_extractor.get_video_info(video_path)
        frames = self.frame_extractor.extract_frames(video_path)

        if not frames:
            raise ValueError(f"Could not extract any valid video frames from {video_path}")

        timeline: List[Dict[str, Any]] = []
        auth_scores: List[float] = []
        manip_scores: List[float] = []
        timestamps: List[float] = []
        face_detected_count = 0

        for f in frames:
            # 1. YOLO spatial context
            rois = self.yolo.detect(f.image_bgr)
            # 2. Face detection
            faces = self.face_detector.detect_faces(f.image_bgr)

            target_crop = None
            is_face = False

            if faces:
                target_crop = faces[0].aligned_face
                is_face = True
                face_detected_count += 1
            elif rois:
                # Use person ROI detected by YOLO
                person_rois = [r for r in rois if r.is_person]
                target_crop = person_rois[0].crop if person_rois else rois[0].crop
            else:
                target_crop = f.image_bgr

            # 3. Forensic classifier prediction
            pred = self.classifier.predict(target_crop)
            auth_p = pred["prob_authentic"]
            manip_p = pred["prob_manipulated"]

            auth_scores.append(auth_p)
            manip_scores.append(manip_p)
            timestamps.append(f.timestamp_sec)

            timeline.append({
                "frame_index": f.frame_index,
                "timestamp_sec": f.timestamp_sec,
                "authenticity_score": round(auth_p, 3),
                "manipulation_score": round(manip_p, 3),
                "confidence": round(pred["confidence"], 3),
                "face_detected": is_face,
                "artifacts": pred["artifacts"]
            })

        # 4. Temporal analysis & aggregation
        agg_auth, agg_desc = self.temporal_analyzer.aggregate_scores(auth_scores)
        jitter = self.temporal_analyzer.compute_temporal_jitter(manip_scores)
        suspicious_segments = self.temporal_analyzer.detect_suspicious_segments(timestamps, manip_scores)

        # Penalize overall authenticity if high temporal jitter is observed
        final_video_auth = float(np.clip(agg_auth - (jitter * 0.15), 0.02, 0.98))

        return {
            "media_info": metadata,
            "total_frames_analyzed": len(frames),
            "face_detection_ratio": round(face_detected_count / len(frames), 2),
            "temporal_jitter": jitter,
            "aggregate_authenticity": round(final_video_auth, 3),
            "aggregate_manipulation": round(1.0 - final_video_auth, 3),
            "aggregation_reasoning": agg_desc,
            "suspicious_segments": suspicious_segments,
            "timeline": timeline
        }
