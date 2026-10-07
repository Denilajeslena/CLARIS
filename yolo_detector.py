"""
YOLO Spatial Context and Region-of-Interest (ROI) Detector.
Uses Ultralytics YOLO26 (YOLO26L as primary high-accuracy, YOLO26S as fast fallback).
Detects persons, context objects, and bounding boxes for forensic routing.
"""

from __future__ import annotations
import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class DetectedROI:
    bbox: Tuple[int, int, int, int] # x1, y1, x2, y2 (integers)
    confidence: float
    class_id: int
    class_name: str
    is_person: bool
    crop: Optional[np.ndarray] = None

class YOLOSpatialDetector:
    """
    YOLO26 Detector integrating YOLO26L (High-Capacity) and YOLO26S (Fast Fallback).
    Provides spatial bounding boxes, person/scene context, and crops for forensic inspection.
    """

    def __init__(
        self,
        model_size: str = "L",
        model_path: Optional[str] = None,
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        device: str = "auto"
    ):
        self.model_size = model_size.upper()
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.model = None
        self.model_name = f"YOLO26{self.model_size}"
        self.is_loaded = False
        self.is_baseline = False

        self._load_model(model_path)

    def _load_model(self, model_path: Optional[str] = None):
        """Loads Ultralytics YOLO26L / YOLO26S or falls back gracefully."""
        try:
            from ultralytics import YOLO
            import torch

            # Determine weights target
            if model_path and os.path.exists(model_path):
                weights = model_path
            else:
                # Use standard Ultralytics checkpoint name; maps to YOLO26 or fallback YOLOv8/v11 weights
                weights = f"yolo26{self.model_size.lower()}.pt"

            # Auto device resolution
            dev = self.device
            if dev == "auto":
                dev = "cuda:0" if torch.cuda.is_available() else "cpu"

            try:
                self.model = YOLO(weights)
                self.model.to(dev)
                self.is_loaded = True
                self.is_baseline = False
                logger.info(f"Loaded {self.model_name} on {dev}")
            except Exception as e:
                logger.warning(f"Could not load weight {weights}: {e}. Initializing standard fallback YOLO model...")
                # Fallback to general lightweight detector
                self.model = YOLO("yolov8s.pt" if self.model_size == "S" else "yolov8m.pt")
                self.model.to(dev)
                self.is_loaded = True
                self.is_baseline = True

        except Exception as err:
            logger.error(f"Ultralytics YOLO unavailable: {err}. Falling back to OpenCV ROI heuristic.")
            self.model = None
            self.is_loaded = False
            self.is_baseline = True

    def set_model_size(self, size: str):
        """Switch between YOLO26L and YOLO26S at runtime."""
        if size.upper() not in ["L", "S"]:
            raise ValueError(f"Invalid model size {size}. Choose 'L' or 'S'.")
        if self.model_size != size.upper():
            self.model_size = size.upper()
            self.model_name = f"YOLO26{self.model_size}"
            self._load_model(None)

    def detect(self, image_bgr: np.ndarray) -> List[DetectedROI]:
        """
        Runs spatial detection on input image (OpenCV BGR format).
        Returns list of DetectedROI objects with coordinates and person flag.
        """
        h, w = image_bgr.shape[:2]
        rois: List[DetectedROI] = []

        if self.model is not None and self.is_loaded:
            try:
                results = self.model.predict(
                    source=image_bgr,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    verbose=False
                )
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = r.names.get(cls_id, str(cls_id))

                        # Constrain coordinates to image dimensions
                        x1 = max(0, min(coords[0], w - 1))
                        y1 = max(0, min(coords[1], h - 1))
                        x2 = max(x1 + 1, min(coords[2], w))
                        y2 = max(y1 + 1, min(coords[3], h))

                        crop = image_bgr[y1:y2, x1:x2].copy()
                        is_person = (cls_name.lower() == "person" or cls_id == 0)

                        rois.append(DetectedROI(
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            class_id=cls_id,
                            class_name=cls_name,
                            is_person=is_person,
                            crop=crop
                        ))
                return rois
            except Exception as e:
                logger.error(f"Inference error in YOLO: {e}")

        # Baseline OpenCV Fallback ROI extractor if weights not available
        logger.info("Using baseline spatial ROI extractor")
        # Center crop representing primary subject
        pad_x, pad_y = int(w * 0.15), int(h * 0.10)
        x1, y1 = pad_x, pad_y
        x2, y2 = w - pad_x, h - pad_y
        crop = image_bgr[y1:y2, x1:x2].copy()

        rois.append(DetectedROI(
            bbox=(x1, y1, x2, y2),
            confidence=0.85,
            class_id=0,
            class_name="person_roi_baseline",
            is_person=True,
            crop=crop
        ))
        return rois

    def get_summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_size": self.model_size,
            "status": "loaded" if self.is_loaded else "baseline_heuristic",
            "is_baseline": self.is_baseline,
            "conf_thresh": self.conf_thresh
        }
