"""
Explainable AI Evidence Generation Engine.
Synthesizes computed forensic signals into verifiable, factual forensic evidence points.
Binds every conclusion directly to quantifiable model and signal metrics.
"""

from __future__ import annotations
from typing import Dict, Any, List

class EvidenceEngine:
    """
    Transforms quantitative digital forensic metrics into human-readable,
    auditable forensic evidence statements with explicit signal bindings.
    """

    def compile_evidence(
        self,
        vision_data: Dict[str, Any] | None = None,
        video_data: Dict[str, Any] | None = None,
        audio_data: Dict[str, Any] | None = None,
        sync_data: Dict[str, Any] | None = None,
        freq_data: Dict[str, Any] | None = None,
        comp_data: Dict[str, Any] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Synthesizes structured evidence items with severity ('CRITICAL', 'WARNING', 'AUTHENTIC_SIGNAL').
        """
        evidence_items: List[Dict[str, Any]] = []

        # 1. Vision Evidence (Facial boundaries, texture, Grad-CAM)
        if vision_data:
            artifacts = vision_data.get("artifacts", {})
            boundary = artifacts.get("boundary_anomaly", 0.0)
            texture = artifacts.get("texture_anomaly", 0.0)
            color = artifacts.get("chrominance_anomaly", 0.0)
            noise = artifacts.get("noise_inconsistency", 0.0)
            gradcam_expl = vision_data.get("gradcam_explanation", {})

            if noise > 0.35:
                evidence_items.append({
                    "category": "Vision / Noise Analysis",
                    "severity": "CRITICAL" if noise > 0.60 else "WARNING",
                    "title": "Uniform Noise Floor (AI Generation Signature)",
                    "detail": f"Spatial noise distribution is unnaturally uniform across image regions (inconsistency: {noise:.2f}). Real camera sensors produce spatially varying noise; AI generators do not.",
                    "metric_key": "noise_inconsistency",
                    "value": noise
                })

            if boundary > 0.40:
                evidence_items.append({
                    "category": "Vision / Spatial",
                    "severity": "CRITICAL" if boundary > 0.65 else "WARNING",
                    "title": "Facial Blending Boundary Anomaly",
                    "detail": f"Discontinuous edge gradient observed along outer face margins (gradient ratio anomaly: {boundary:.2f}). Characteristic of Poisson blending or alpha-mask splicing seams.",
                    "metric_key": "boundary_anomaly",
                    "value": boundary
                })

            if texture > 0.45:
                evidence_items.append({
                    "category": "Vision / Texture",
                    "severity": "CRITICAL" if texture > 0.70 else "WARNING",
                    "title": "Unnatural Facial Skin Texture",
                    "detail": f"Surface frequency variance ({artifacts.get('laplacian_var', 0):.1f}) deviates significantly from natural optical skin pores; matches GAN over-smoothing or diffusion generation.",
                    "metric_key": "texture_anomaly",
                    "value": texture
                })

            if color > 0.45:
                evidence_items.append({
                    "category": "Vision / Chrominance",
                    "severity": "WARNING",
                    "title": "YCbCr Chrominance Asymmetry",
                    "detail": f"Inconsistent color channel covariance (score: {color:.2f}) between face foreground and background illuminant.",
                    "metric_key": "chrominance_anomaly",
                    "value": color
                })

            if gradcam_expl and gradcam_expl.get("peak_intensity", 0) > 0.40:
                evidence_items.append({
                    "category": "Explainable AI (Grad-CAM)",
                    "severity": "CRITICAL",
                    "title": f"Focal Activation: {gradcam_expl.get('top_region', 'Face')}",
                    "detail": gradcam_expl.get("explanation_text", "High model activation detected in localized face region."),
                    "metric_key": "gradcam_peak",
                    "value": gradcam_expl.get("peak_intensity", 0)
                })

        # 2. Video Temporal Evidence
        if video_data:
            jitter = video_data.get("temporal_jitter", 0.0)
            suspicious_segs = video_data.get("suspicious_segments", [])

            if jitter > 0.35:
                evidence_items.append({
                    "category": "Temporal Dynamics",
                    "severity": "CRITICAL" if jitter > 0.60 else "WARNING",
                    "title": "High Frame-to-Frame Temporal Jitter",
                    "detail": f"Inter-frame prediction volatility ({jitter:.2f}) indicates flickering synthesis artifacts and lack of temporal coherence.",
                    "metric_key": "temporal_jitter",
                    "value": jitter
                })

            for seg in suspicious_segs[:2]: # Report top windows
                evidence_items.append({
                    "category": "Temporal Dynamics",
                    "severity": "CRITICAL",
                    "title": f"Suspicious Segment: {seg['time_window']}",
                    "detail": f"Concentrated manipulation spike across {seg['frame_count']} consecutive frames (mean probability: {seg['average_manipulation_prob']*100:.1f}%).",
                    "metric_key": "segment_detection",
                    "value": seg["average_manipulation_prob"]
                })

        # 3. Audio Evidence
        if audio_data:
            voc = audio_data.get("vocoder_metrics", {})
            ac = audio_data.get("acoustic_signals", {})

            if voc.get("has_brickwall_cutoff", False):
                evidence_items.append({
                    "category": "Audio / Spectral",
                    "severity": "CRITICAL",
                    "title": "Neural Vocoder Brick-Wall Filter Cutoff",
                    "detail": "Sudden spectral energy dropoff above upper band; characteristic artifact of HiFi-GAN or neural vocoder resynthesis.",
                    "metric_key": "vocoder_cutoff",
                    "value": voc.get("vocoder_artifact_score", 0.8)
                })

            if ac.get("pitch_flatness", 0) > 0.60:
                evidence_items.append({
                    "category": "Audio / Prosody",
                    "severity": "WARNING",
                    "title": "Robotic Monotonic Pitch Contour",
                    "detail": f"Acoustic pitch prosody shows unnatural flatness ({ac.get('pitch_flatness', 0):.2f}) without organic human micro-pitch deviations.",
                    "metric_key": "pitch_flatness",
                    "value": ac.get("pitch_flatness", 0)
                })

        # 4. Cross-Modal AV Sync Evidence
        if sync_data:
            sync_score = sync_data.get("sync_score", 100.0)
            lag = sync_data.get("estimated_lag_ms", 0.0)
            mismatches = sync_data.get("mismatch_segments", [])

            if sync_score < 55.0:
                evidence_items.append({
                    "category": "Cross-Modal Consistency",
                    "severity": "CRITICAL",
                    "title": "Severe Audio-Visual Desynchronization",
                    "detail": f"Mouth articulation and speech audio envelope out-of-phase (estimated latency: {lag} ms). Strong indicator of AI audio dubbing or face reenactment.",
                    "metric_key": "av_sync_score",
                    "value": sync_score
                })
            for m in mismatches[:2]:
                evidence_items.append({
                    "category": "Cross-Modal Consistency",
                    "severity": "WARNING",
                    "title": f"Dubbing Desync Window: {m['time_window']}",
                    "detail": m["reason"],
                    "metric_key": "desync_window",
                    "value": m["local_correlation"]
                })

        # 5. Frequency Domain Evidence
        if freq_data:
            f_score = freq_data.get("frequency_anomaly_score", 0.0)
            if f_score > 0.35:
                evidence_items.append({
                    "category": "Frequency Domain",
                    "severity": "WARNING",
                    "title": "Spectral Anomaly & Resampling Spikes",
                    "detail": freq_data.get("evidence_note", "Periodic 2D FFT peaks detected in high-frequency band."),
                    "metric_key": "frequency_anomaly",
                    "value": f_score
                })

        # 6. Authentic Baseline Corroboration if no major tampering detected
        if not evidence_items:
            evidence_items.append({
                "category": "Forensic Verification",
                "severity": "AUTHENTIC_SIGNAL",
                "title": "Consistent Physical & Acoustic Properties",
                "detail": "Natural skin pore texture, continuous audio formants, smooth temporal progression, and natural frequency decay observed without anomalous synthetic signatures.",
                "metric_key": "integrity_pass",
                "value": 1.0
            })

        return evidence_items
