"""
Forensic Feature Extractor.
Collects all raw numerical signals from the existing pipeline outputs
into a flat, ordered feature vector for the ML verifier.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np


# Ordered feature names — must stay in sync with MLVerifier
FEATURE_NAMES = [
    # Vision / texture
    "noise_inconsistency",
    "texture_anomaly",
    "boundary_anomaly",
    "chrominance_anomaly",
    "laplacian_var_norm",
    # Neural classifier
    "nn_prob_manipulated",
    "nn_confidence",
    # Frequency domain
    "fft_anomaly",
    "dct_anomaly",
    "frequency_anomaly_score",
    # Compression / ELA
    "ela_score",
    "compression_risk_score",
    # Metadata signals
    "missing_camera_exif",
    "typical_generative_resolution",
    # Temporal (video only — 0 for images)
    "temporal_jitter",
    "suspicious_segment_ratio",
    "face_detection_ratio",
    # Audio (0 when absent)
    "audio_prob_manipulated",
    "audio_pitch_flatness",
    "audio_vocoder_score",
    # AV sync (0 when absent)
    "av_sync_score_inv",   # inverted: high = bad sync
]

NUM_FEATURES = len(FEATURE_NAMES)


def extract_features(
    vision_res: Optional[Dict[str, Any]] = None,
    freq_res: Optional[Dict[str, Any]] = None,
    comp_res: Optional[Dict[str, Any]] = None,
    meta_res: Optional[Dict[str, Any]] = None,
    video_res: Optional[Dict[str, Any]] = None,
    audio_res: Optional[Dict[str, Any]] = None,
    sync_res: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Returns a float32 numpy array of shape (NUM_FEATURES,).
    All missing modalities default to 0.0 (neutral / unknown).
    """
    v = vision_res or {}
    artifacts = v.get("artifacts", {})
    f = freq_res or {}
    c = comp_res or {}
    m = meta_res or {}
    vid = video_res or {}
    aud = audio_res or {}
    sync = sync_res or {}

    lap_var = artifacts.get("laplacian_var", 400.0)
    # Normalise laplacian variance: 0=very smooth(AI), 1=natural, clamp at 2000
    lap_norm = float(np.clip(lap_var / 600.0, 0.0, 2.0)) / 2.0

    # Temporal
    total_frames = max(vid.get("total_frames_analyzed", 1), 1)
    suspicious_frames = sum(
        seg.get("frame_count", 0) for seg in vid.get("suspicious_segments", [])
    )
    suspicious_ratio = float(np.clip(suspicious_frames / total_frames, 0.0, 1.0))

    # Audio
    aud_acoustic = aud.get("acoustic_signals", {})
    aud_vocoder = aud.get("vocoder_metrics", {})

    feat = np.array([
        artifacts.get("noise_inconsistency", 0.0),
        artifacts.get("texture_anomaly", 0.0),
        artifacts.get("boundary_anomaly", 0.0),
        artifacts.get("chrominance_anomaly", 0.0),
        lap_norm,
        v.get("prob_manipulated", 0.0),
        v.get("confidence", 0.5),
        f.get("fft_anomaly", 0.0),
        f.get("dct_anomaly", 0.0),
        f.get("frequency_anomaly_score", 0.0),
        c.get("ela_score", 0.0),
        c.get("compression_risk_score", 0.0),
        0.0 if m.get("camera_hardware_identified", True) else 1.0,
        1.0 if m.get("typical_generative_aspect_ratio", False) else 0.0,
        float(vid.get("temporal_jitter", 0.0)),
        suspicious_ratio,
        float(vid.get("face_detection_ratio", 0.0)),
        float(1.0 - aud.get("prob_authentic", 1.0)),
        float(aud_acoustic.get("pitch_flatness", 0.0)),
        float(aud_vocoder.get("vocoder_artifact_score", 0.0)),
        float(np.clip(1.0 - sync.get("sync_score", 100.0) / 100.0, 0.0, 1.0)),
    ], dtype=np.float32)

    assert len(feat) == NUM_FEATURES, f"Feature count mismatch: {len(feat)} vs {NUM_FEATURES}"
    return feat


def feature_breakdown(feat_vec: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    Maps the raw feature vector to per-category Normal/Suspicious status
    with the raw score for display in the UI.
    """
    f = feat_vec
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    def score(name: str) -> float:
        return float(f[idx[name]])

    def status(val: float, threshold: float = 0.35) -> str:
        return "🔴 Suspicious" if val >= threshold else "🟢 Normal"

    face_texture_score = max(score("noise_inconsistency"), score("texture_anomaly"), score("boundary_anomaly"))
    freq_score = score("frequency_anomaly_score")
    comp_score = max(score("ela_score"), score("compression_risk_score"))
    meta_score = max(score("missing_camera_exif"), score("typical_generative_resolution"))
    temporal_score = max(score("temporal_jitter"), score("suspicious_segment_ratio"))
    audio_score = max(score("audio_prob_manipulated"), score("audio_pitch_flatness"), score("audio_vocoder_score"))
    sync_score = score("av_sync_score_inv")

    return {
        "Face / Texture": {"score": round(face_texture_score, 3), "status": status(face_texture_score)},
        "Frequency (FFT/DCT)": {"score": round(freq_score, 3), "status": status(freq_score, 0.30)},
        "Compression / ELA": {"score": round(comp_score, 3), "status": status(comp_score, 0.40)},
        "Metadata": {"score": round(meta_score, 3), "status": status(meta_score, 0.50)},
        "Temporal (Video)": {"score": round(temporal_score, 3), "status": status(temporal_score, 0.30)},
        "Audio": {"score": round(audio_score, 3), "status": status(audio_score, 0.35)},
        "A/V Sync": {"score": round(sync_score, 3), "status": status(sync_score, 0.35)},
    }
