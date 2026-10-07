"""
Dedicated Deepfake Forensic Classifier.
Implements a dual-stream architecture combining:
1. Deep convolutional feature extractor (EfficientNet / Xception backbone)
2. Spatial artifact analyzer (blending boundaries, local texture anomalies, resampling artifacts, chrominance anomalies)
"""

from __future__ import annotations
import os
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class ForensicBackbone(nn.Module):
    """
    Forensic neural network backbone with hookable activation layer for Grad-CAM.
    Can utilize torchvision EfficientNet-B4 / ResNet50 or Timm Xception.
    """
    def __init__(self, backbone_name: str = "efficientnet_b4", pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        self.backbone_name = backbone_name

        try:
            import torchvision.models as models
            if "efficientnet" in backbone_name.lower():
                weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
                base = models.efficientnet_b4(weights=weights)
                self.features = base.features
                in_features = base.classifier[1].in_features
                self.target_layer = self.features[-1]
            else:
                weights = models.ResNet50_Weights.DEFAULT if pretrained else None
                base = models.resnet50(weights=weights)
                self.features = nn.Sequential(*list(base.children())[:-2])
                in_features = base.fc.in_features
                self.target_layer = self.features[-1]
        except Exception as e:
            logger.warning(f"Failed to load torchvision model with weights ({e}), using custom conv architecture.")
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            in_features = 256
            self.target_layer = self.features[6]

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 128),
            nn.SiLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_map = self.features(x)
        pooled = self.pool(feat_map)
        flattened = torch.flatten(pooled, 1)
        logits = self.classifier(flattened)
        return logits, feat_map


class ForensicClassifier:
    """
    Complete Forensic Classification Engine.
    Executes neural inference + explicit forensic signal tests (blending boundary, texture, color).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        backbone_name: str = "efficientnet_b4",
        device: str = "auto"
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.backbone_name = backbone_name
        self.is_baseline = True
        self.model = ForensicBackbone(backbone_name=backbone_name, pretrained=True).to(self.device)
        self.model.eval()

        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                self.is_baseline = False
                logger.info(f"Loaded trained forensic weights from {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}. Running with pretrained backbone.")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def analyze_patch_artifacts(self, img_bgr: np.ndarray) -> Dict[str, float]:
        h, w = img_bgr.shape[:2]
        if h < 16 or w < 16:
            return {"boundary_anomaly": 0.0, "texture_anomaly": 0.0,
                    "chrominance_anomaly": 0.0, "noise_inconsistency": 0.0, "laplacian_var": 0.0}

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Laplacian variance — AI images are unnaturally smooth (diffusion) or over-sharp (GAN)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(laplacian.var())
        # Real camera photos: 200-900. AI diffusion: 80-200 (too smooth). GAN: 900-2000 (over-sharp).
        texture_anomaly = 0.0
        if lap_var < 180.0:
            texture_anomaly = float(np.clip((180.0 - lap_var) / 180.0, 0.0, 1.0))
        elif lap_var > 900.0:
            texture_anomaly = float(np.clip((lap_var - 900.0) / 800.0, 0.0, 1.0))

        # 2. Noise inconsistency — AI images have unnaturally uniform noise floor across regions
        # Split image into 4 quadrants and compare local noise std deviation
        h2, w2 = h // 2, w // 2
        quads = [
            gray[:h2, :w2], gray[:h2, w2:],
            gray[h2:, :w2], gray[h2:, w2:]
        ]
        quad_noise = [float(np.std(cv2.Laplacian(q.astype(np.float32), cv2.CV_32F))) for q in quads if q.size > 0]
        if len(quad_noise) >= 2:
            noise_cv = float(np.std(quad_noise) / (np.mean(quad_noise) + 1e-5))
            # Real photos have natural noise variation (cv > 0.25). AI images are too uniform (cv < 0.15).
            noise_inconsistency = float(np.clip(1.0 - (noise_cv / 0.25), 0.0, 1.0)) if noise_cv < 0.25 else 0.0
        else:
            noise_inconsistency = 0.0

        # 3. Blending boundary gradient discrepancy
        border_w = max(2, int(w * 0.12))
        border_h = max(2, int(h * 0.12))
        inner = gray[border_h:h-border_h, border_w:w-border_w]
        outer_mask = np.ones((h, w), dtype=bool)
        outer_mask[border_h:h-border_h, border_w:w-border_w] = False
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        inner_grad_mean = float(np.mean(grad_mag[border_h:h-border_h, border_w:w-border_w])) if inner.size > 0 else 1.0
        outer_grad_mean = float(np.mean(grad_mag[outer_mask])) if np.any(outer_mask) else 1.0
        grad_ratio = outer_grad_mean / (inner_grad_mean + 1e-5)
        boundary_anomaly = float(np.clip(abs(grad_ratio - 1.0) / 2.0, 0.0, 1.0))

        # 4. Chrominance discrepancy (YCbCr)
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1].astype(float)
        cb = ycrcb[:, :, 2].astype(float)
        cr_std = float(np.std(cr))
        cb_std = float(np.std(cb))
        color_ratio = abs(cr_std - cb_std) / (cr_std + cb_std + 1e-5)
        chrominance_anomaly = float(np.clip(color_ratio * 1.5, 0.0, 1.0))

        return {
            "boundary_anomaly": boundary_anomaly,
            "texture_anomaly": texture_anomaly,
            "chrominance_anomaly": chrominance_anomaly,
            "noise_inconsistency": noise_inconsistency,
            "laplacian_var": lap_var
        }

    def predict(self, face_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Runs complete forensic inference on BGR face image.
        Returns:
            prob_manipulated: float [0, 1]
            prob_authentic: float [0, 1]
            confidence: float [0, 1]
            artifacts: Dict[str, float]
            logits: Tuple[float, float]
        """
        # Calculate signal artifacts
        artifacts = self.analyze_patch_artifacts(face_bgr)

        # PyTorch Neural forward pass
        img_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, feat_map = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # Model output calibration
        prob_authentic_nn = float(probs[0])
        prob_manipulated_nn = float(probs[1])

        # Heuristic forensic signals — noise_inconsistency is the strongest AI detector
        heuristic_manip_score = (
            0.35 * artifacts["noise_inconsistency"] +
            0.30 * artifacts["texture_anomaly"] +
            0.20 * artifacts["boundary_anomaly"] +
            0.15 * artifacts["chrominance_anomaly"]
        )

        if self.is_baseline:
            # No fine-tuned weights: trust heuristics heavily, nn output is unreliable
            combined_manip = 0.25 * prob_manipulated_nn + 0.75 * heuristic_manip_score
        else:
            combined_manip = 0.75 * prob_manipulated_nn + 0.25 * heuristic_manip_score

        combined_manip = float(np.clip(combined_manip, 0.01, 0.99))
        combined_auth = float(1.0 - combined_manip)

        confidence = float(np.clip(abs(combined_auth - combined_manip) * 1.6, 0.35, 0.99))

        return {
            "prob_authentic": combined_auth,
            "prob_manipulated": combined_manip,
            "confidence": confidence,
            "artifacts": artifacts,
            "is_baseline": self.is_baseline,
            "backbone": self.backbone_name,
            "device": str(self.device)
        }
