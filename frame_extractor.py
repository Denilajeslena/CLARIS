"""
Video Frame Extraction and Sampling Engine.
Reads video files via OpenCV, extracts frames at configurable sampling rates,
and timestamps each frame for temporal forensic tracking.
"""

from __future__ import annotations
import cv2
import os
from dataclasses import dataclass
from typing import Generator, List, Tuple, Dict, Any
import numpy as np

@dataclass
class VideoFrame:
    frame_index: int
    timestamp_sec: float
    image_bgr: np.ndarray

class VideoFrameExtractor:
    """
    Handles video ingestion, metadata extraction, and frame-rate adaptive subsampling.
    """

    def __init__(self, process_every_n_frames: int = 3, max_frames: int = 150):
        self.process_every_n_frames = max(1, process_every_n_frames)
        self.max_frames = max_frames

    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Reads stream metadata using OpenCV VideoCapture."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0

        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        cap.release()

        return {
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_sec": round(duration, 2),
            "fourcc": fourcc_str or "H264",
            "file_size_bytes": os.path.getsize(video_path)
        }

    def extract_frames(self, video_path: str) -> List[VideoFrame]:
        """
        Extracts sampled frames with accurate timestamps.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to read video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames: List[VideoFrame] = []
        frame_idx = 0

        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.process_every_n_frames == 0:
                timestamp = frame_idx / fps
                frames.append(VideoFrame(
                    frame_index=frame_idx,
                    timestamp_sec=round(timestamp, 3),
                    image_bgr=frame
                ))

            frame_idx += 1

        cap.release()
        return frames
