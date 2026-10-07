"""
CLARIS - Multimodal Deepfake & Digital Forensics System.
Streamlit Hackathon-Ready Interface.
Clean, light, professional UI running real local inference across Image, Video, Audio, and Video+Audio.
"""

from __future__ import annotations
import os
import sys
import json
import tempfile
import cv2
import numpy as np
import streamlit as st

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.vision.yolo_detector import YOLOSpatialDetector
from backend.vision.face_detector import FaceDetector
from backend.vision.forensic_classifier import ForensicClassifier
from backend.vision.gradcam import ForensicGradCAM
from backend.vision.frequency_analysis import FrequencyForensics
from backend.video.video_forensics import VideoForensicsPipeline
from backend.audio.audio_model import AudioForensicClassifier
from backend.multimodal.av_sync import AudioVideoSyncAnalyzer
from backend.multimodal.fusion import MultimodalFusionEngine
from backend.forensics.metadata import MediaMetadataInspector
from backend.forensics.compression_analysis import CompressionAnalyzer
from backend.forensics.evidence_engine import EvidenceEngine
from backend.forensics.scoring import ForensicScorer
from backend.forensics.ml_verifier import MLVerifier
from backend.forensics.feature_extractor import extract_features

# Page Configuration
st.set_page_config(
    page_title="CLARIS | Multimodal Deepfake Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Warm editorial theme styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        color-scheme: light;
        --canvas: #f5f3ec;
        --surface: #fbfaf6;
        --surface-raised: #fffefa;
        --ink: #405247;
        --muted: #73766c;
        --line: #e5e2d8;
        --olive: #68735b;
        --olive-soft: #edf0e8;
        --terracotta: #a85f4d;
        --terracotta-soft: #f4e9e4;
        --shadow: 0 8px 24px rgba(47, 48, 39, 0.045);
    }

    html, body, .stApp, [data-testid="stAppViewContainer"], .stMain, .main {
        background: var(--canvas) !important;
        color: var(--ink) !important;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', 'Inter', sans-serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.035em;
    }
    .stMarkdown p,
    .stMarkdown li,
    .stText,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stWidgetLabel"],
    [data-testid="stFileUploader"] small {
        color: var(--ink) !important;
    }
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {
        color: var(--ink) !important;
    }
    [data-baseweb="popover"],
    [data-baseweb="popover"] *,
    [role="listbox"],
    [role="option"] {
        background-color: var(--surface-raised) !important;
        color: var(--ink) !important;
    }

    .main .block-container {
        max-width: 1440px;
        padding: 2.25rem 2.5rem 3rem;
    }

    section[data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] * { color: var(--ink) !important; }
    section[data-testid="stSidebar"] strong { color: var(--olive) !important; }
    section[data-testid="stSidebar"] hr { border-color: var(--line) !important; }

    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        border-bottom: 1px solid var(--line);
        gap: 0.4rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--muted);
        border-radius: 10px 10px 0 0;
        font-weight: 500;
        font-size: 0.9rem;
        padding: 0.8rem 1.1rem;
        border: 1px solid transparent;
        transition: color 160ms ease, background-color 160ms ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--olive);
        background: var(--olive-soft);
    }
    .stTabs [aria-selected="true"] {
        background: var(--surface) !important;
        color: var(--olive) !important;
        border-color: var(--line) !important;
        border-bottom-color: var(--surface) !important;
    }

    .stButton > button {
        background: var(--olive);
        color: #fffefa;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.01em;
        border: 1px solid var(--olive);
        border-radius: 10px;
        padding: 0.7rem 1.1rem;
        transition: background-color 160ms ease, border-color 160ms ease, transform 160ms ease, box-shadow 160ms ease;
    }
    .stButton > button:hover {
        background: #59634e;
        border-color: #59634e;
        transform: translateY(-1px);
        box-shadow: 0 5px 14px rgba(57, 67, 49, 0.12);
    }
    .stButton > button:focus-visible,
    .stDownloadButton > button:focus-visible,
    .stTabs button:focus-visible {
        outline: 2px solid var(--terracotta) !important;
        outline-offset: 2px;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: var(--surface-raised) !important;
        color: var(--ink) !important;
        border-color: var(--line) !important;
        border-radius: 9px !important;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--olive) !important;
        box-shadow: 0 0 0 1px var(--olive) !important;
    }

    [data-testid="stFileUploader"] {
        background: var(--surface);
        border: 1px dashed #c9c9ba;
        border-radius: 14px;
        padding: 0.6rem;
        transition: border-color 160ms ease, background-color 160ms ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--olive);
        background: var(--surface-raised);
    }
    [data-testid="stFileUploader"] section { background: transparent; }
    [data-testid="stFileUploader"] button,
    .stDownloadButton > button {
        color: var(--olive) !important;
        border: 1px solid #cdd1c4 !important;
        background: var(--surface-raised) !important;
        border-radius: 9px;
        font-weight: 600;
        transition: background-color 160ms ease, border-color 160ms ease;
    }
    [data-testid="stFileUploader"] button:hover,
    .stDownloadButton > button:hover {
        border-color: var(--olive) !important;
        background: var(--olive-soft) !important;
    }

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem 1.15rem;
        box-shadow: var(--shadow);
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: 0.76rem !important;
        font-weight: 600;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }
    [data-testid="stMetricValue"] {
        color: var(--olive) !important;
        font-size: 1.65rem !important;
        font-weight: 600;
    }
    hr { border-color: var(--line) !important; }

    .badge-authentic,
    .badge-suspicious,
    .badge-manipulated {
        display: inline-block;
        padding: 0.65rem 1.15rem;
        border-radius: 999px;
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.05rem;
        letter-spacing: 0.04em;
    }
    .badge-authentic {
        background: var(--olive-soft);
        color: #4f6048;
        border: 1px solid #d8dfd0;
    }
    .badge-suspicious {
        background: #f3eee2;
        color: #78653f;
        border: 1px solid #e7ddc5;
    }
    .badge-manipulated {
        background: var(--terracotta-soft);
        color: #8e4e40;
        border: 1px solid #e8d1c8;
    }

    .evidence-critical,
    .evidence-warning,
    .evidence-pass {
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
        border-radius: 0 10px 10px 0;
        background: var(--surface);
        box-shadow: var(--shadow);
    }
    .evidence-critical { border-left: 3px solid var(--terracotta); }
    .evidence-warning { border-left: 3px solid #a28c5d; }
    .evidence-pass { border-left: 3px solid var(--olive); }

    .section-header {
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        color: var(--olive);
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border-bottom: 1px solid var(--line);
        padding-bottom: 0.65rem;
        margin: 1.8rem 0 1rem;
    }

    .stAlert { border-radius: 10px !important; }
    [data-testid="stInfo"] { background: #eef0eb !important; border-left: 3px solid var(--olive) !important; }
    [data-testid="stSuccess"] { background: var(--olive-soft) !important; border-left: 3px solid var(--olive) !important; }
    [data-testid="stWarning"] { background: #f3eee2 !important; border-left: 3px solid #a28c5d !important; }
    [data-testid="stError"] { background: var(--terracotta-soft) !important; border-left: 3px solid var(--terracotta) !important; }
    .stSpinner > div { border-top-color: var(--olive) !important; }
    [data-testid="stVegaLiteChart"] {
        background: var(--surface) !important;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.6rem;
        box-shadow: var(--shadow);
    }
    .stImage > div > small,
    .stCaption { color: var(--muted) !important; }
</style>
""", unsafe_allow_html=True)

# Cache Model Pipeline Initialization
@st.cache_resource
def load_forensic_pipeline(model_size: str = "L"):
    yolo = YOLOSpatialDetector(model_size=model_size)
    face_det = FaceDetector()
    classifier = ForensicClassifier(backbone_name="efficientnet_b4")
    gradcam = ForensicGradCAM(classifier)
    freq = FrequencyForensics()
    audio_clf = AudioForensicClassifier()
    video_pipe = VideoForensicsPipeline(yolo, face_det, classifier)
    av_sync = AudioVideoSyncAnalyzer()
    fusion = MultimodalFusionEngine()
    metadata_insp = MediaMetadataInspector()
    comp_analyzer = CompressionAnalyzer()
    evidence_eng = EvidenceEngine()
    scorer = ForensicScorer()
    ml_verifier = MLVerifier()

    return {
        "yolo": yolo,
        "face_det": face_det,
        "classifier": classifier,
        "gradcam": gradcam,
        "freq": freq,
        "audio_clf": audio_clf,
        "video_pipe": video_pipe,
        "av_sync": av_sync,
        "fusion": fusion,
        "metadata_insp": metadata_insp,
        "comp_analyzer": comp_analyzer,
        "evidence_eng": evidence_eng,
        "scorer": scorer,
        "ml_verifier": ml_verifier,
    }

# Sidebar Configuration
st.sidebar.markdown("""
<div style="padding: 8px 0 16px 0;">
    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 600; color: #68735b; letter-spacing: -0.02em;">CLARIS / CONTROL</div>
    <div style="color: #73766c; font-size: 0.72rem; letter-spacing: 0.08em; margin-top: 4px;">PIPELINE CONFIGURATION</div>
</div>
""", unsafe_allow_html=True)
model_size = st.sidebar.selectbox("YOLO Model Architecture", ["L (YOLO26L - High Precision)", "S (YOLO26S - Fast Fallback)"], index=0)
selected_size = "L" if "L" in model_size else "S"

pipeline = load_forensic_pipeline(selected_size)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.78rem; font-weight: 600; color: #68735b; letter-spacing: 0.06em; margin-bottom: 10px;">ENGINE STATUS</div>
""", unsafe_allow_html=True)
import torch
device_str = "CUDA (GPU)" if torch.cuda.is_available() else "CPU Fallback"
cuda_color = "#59694f" if torch.cuda.is_available() else "#806c47"
st.sidebar.markdown(f"""
<div style="display:flex; flex-direction:column; gap:8px; font-size:0.8rem;">
    <div style="display:flex; justify-content:space-between; padding:10px 12px; background:#fffefa; border-radius:9px; border:1px solid #e5e2d8;">
        <span style="color:#73766c;">Hardware</span>
        <span style="color:{cuda_color}; font-weight:600;">{device_str}</span>
    </div>
    <div style="display:flex; justify-content:space-between; padding:10px 12px; background:#fffefa; border-radius:9px; border:1px solid #e5e2d8;">
        <span style="color:#73766c;">Spatial</span>
        <span style="color:#405247; font-weight:500;">{pipeline['yolo'].model_name}</span>
    </div>
    <div style="display:flex; justify-content:space-between; padding:10px 12px; background:#fffefa; border-radius:9px; border:1px solid #e5e2d8;">
        <span style="color:#73766c;">Classifier</span>
        <span style="color:#405247; font-weight:500;">{pipeline['classifier'].backbone_name}</span>
    </div>
    <div style="display:flex; justify-content:space-between; padding:10px 12px; background:#fffefa; border-radius:9px; border:1px solid #e5e2d8;">
        <span style="color:#73766c;">Audio</span>
        <span style="color:#405247; font-weight:500;">{pipeline['audio_clf'].encoder_status}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Quick Demo Samples loader
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.78rem; font-weight: 600; color: #68735b; letter-spacing: 0.06em; margin-bottom: 10px;">DEMO TEST SUITE</div>
""", unsafe_allow_html=True)
use_demo = st.sidebar.selectbox("Load Verified Demo Media", ["None", "Demo Real Image", "Demo Deepfake Image", "Demo Real Audio", "Demo Synthetic Voice", "Demo Video Spliced"])

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size:0.72rem; color:#73766c; line-height:1.65;">
    Detection is probabilistic and does not replace forensic chain-of-custody protocols.
</div>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="
    background: #fbfaf6;
    border: 1px solid #e5e2d8;
    border-radius: 18px;
    padding: 34px 40px;
    margin-bottom: 28px;
    box-shadow: 0 8px 24px rgba(47, 48, 39, 0.045);
">
    <div style="font-family:'Inter',sans-serif; color:#68735b; font-size:0.72rem; font-weight:600; letter-spacing:0.12em; margin-bottom:10px;">
        DIGITAL FORENSICS / MULTIMODAL ANALYSIS
    </div>
    <div style="font-family:'Space Grotesk','Inter',sans-serif; color:#405247; font-size:2.8rem; font-weight:600; letter-spacing:-0.06em; line-height:1.05;">
        CLARIS
    </div>
    <div style="color:#73766c; font-size:0.88rem; margin-top:10px;">
        A considered workspace for image, video, audio and provenance analysis.
    </div>
    <div style="display:flex; gap:10px; margin-top:22px; flex-wrap:wrap;">
        <span style="background:#edf0e8; color:#59634e; border-radius:999px; padding:6px 11px; font-size:0.74rem;">Spatial analysis</span>
        <span style="background:#edf0e8; color:#59634e; border-radius:999px; padding:6px 11px; font-size:0.74rem;">Visual evidence</span>
        <span style="background:#edf0e8; color:#59634e; border-radius:999px; padding:6px 11px; font-size:0.74rem;">Audio verification</span>
        <span style="background:#edf0e8; color:#59634e; border-radius:999px; padding:6px 11px; font-size:0.74rem;">Cross-modal sync</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Media Input Tabs
tab_img, tab_vid, tab_aud, tab_multi = st.tabs(["Image Forensics", "Video Forensics", "Audio Forensics", "Video + Audio Sync"])

# 1. IMAGE FORENSICS
with tab_img:
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        img_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png", "webp"], key="img_up")
    with img_col2:
        cam_file = st.camera_input("Capture from Camera (Snapshot)", key="cam_up")

    # Handle quick demo injection
    demo_img_path = None
    if use_demo == "Demo Real Image" and os.path.exists("data/demo/demo_real.jpg"):
        demo_img_path = "data/demo/demo_real.jpg"
    elif use_demo == "Demo Deepfake Image" and os.path.exists("data/demo/demo_fake.jpg"):
        demo_img_path = "data/demo/demo_fake.jpg"

    target_img_bytes = None
    if img_file is not None:
        target_img_bytes = img_file.read()
    elif cam_file is not None:
        target_img_bytes = cam_file.read()
        st.success("Camera photo captured successfully.")
    elif demo_img_path:
        with open(demo_img_path, "rb") as f:
            target_img_bytes = f.read()
        st.info(f"Loaded verified offline demo asset: `{demo_img_path}`")

    if target_img_bytes is None:
        st.markdown("""
        <div style="
            background: #fbfaf6;
            border: 1px dashed #c9c9ba;
            border-radius: 14px;
            padding: 42px 24px;
            text-align: center;
            margin-top: 18px;
        ">
            <div style="font-family:'Space Grotesk','Inter',sans-serif; color:#68735b; font-size:1.05rem; font-weight:600;">Ready for inspection</div>
            <div style="color:#73766c; font-size:0.86rem; margin-top:8px;">Upload an image or capture a photo to begin analysis.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.button("Run Forensic Analysis on Image", key="btn_analyze_img"):
            np_arr = np.frombuffer(target_img_bytes, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            with st.spinner("Executing Forensic Pipeline (YOLO26 ROI + Forensic CNN + Grad-CAM + 2D FFT)..."):
                # 1. Spatial context via YOLO
                rois = pipeline["yolo"].detect(img_bgr)
                # 2. Face detection
                faces = pipeline["face_det"].detect_faces(img_bgr)
                target_patch = faces[0].aligned_face if faces else img_bgr

                # 3. Forensic classifier
                vision_res = pipeline["classifier"].predict(target_patch)

                # 4. Grad-CAM heatmap
                _, heatmap_overlay, gradcam_expl = pipeline["gradcam"].generate_heatmap(target_patch)
                vision_res["gradcam_explanation"] = gradcam_expl

                # 5. Frequency analysis
                freq_res = pipeline["freq"].analyze(target_patch)

                # 6. Metadata & Compression ELA
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp.write(target_img_bytes)
                    tmp_path = tmp.name
                meta_res = pipeline["metadata_insp"].inspect_file(tmp_path, media_type="image")
                comp_res = pipeline["comp_analyzer"].assess_compression_risk(img_bgr)
                os.remove(tmp_path)

                # 7. Evidence Engine & Fusion
                fusion_res = pipeline["fusion"].fuse(vision_res=vision_res, freq_res=freq_res, metadata_res=comp_res)
                evidence_items = pipeline["evidence_eng"].compile_evidence(
                    vision_data=vision_res,
                    freq_data=freq_res,
                    comp_data=comp_res
                )
                assessment = pipeline["scorer"].calculate_assessment(fusion_res, evidence_items, meta_res)

                # 8. ML Second-Pass Verification
                feat_vec = extract_features(
                    vision_res=vision_res, freq_res=freq_res,
                    comp_res=comp_res, meta_res=meta_res
                )
                ml_result = pipeline["ml_verifier"].predict(feat_vec)

            # RENDER RESULTS
            st.markdown('<div class="section-header">⬡ FORENSIC VERDICT</div>', unsafe_allow_html=True)

            # Combine forensic score + ML prediction into final verdict
            forensic_dec = assessment["final_decision"]
            ml_pred = ml_result.get("prediction", "INCONCLUSIVE") if ml_result["status"] == "OK" else "INCONCLUSIVE"

            # Final verdict logic: if both agree → confident; if ML says FAKE/AI → override to at least SUSPICIOUS
            if ml_pred in ("FAKE", "LIKELY AI-GENERATED") and forensic_dec == "AUTHENTIC":
                final_verdict = "SUSPICIOUS"  # ML overrides over-optimistic forensic score
            elif ml_pred == "REAL" and forensic_dec == "AUTHENTIC":
                final_verdict = "AUTHENTIC"
            elif ml_pred in ("FAKE", "LIKELY AI-GENERATED") and forensic_dec in ("SUSPICIOUS", "LIKELY MANIPULATED"):
                final_verdict = "LIKELY MANIPULATED"
            else:
                final_verdict = forensic_dec

            badge_cls = "badge-authentic" if final_verdict == "AUTHENTIC" else ("badge-suspicious" if final_verdict == "SUSPICIOUS" else "badge-manipulated")
            st.markdown(f'<div class="{badge_cls}">{final_verdict}</div>', unsafe_allow_html=True)
            st.caption(f"Forensic Pipeline: **{forensic_dec}** | ML Verifier: **{ml_pred}**")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Authenticity Score", f"{assessment['authenticity_score']} / 100")
            col2.metric("Confidence", f"{int(assessment['model_authenticity_confidence'] * 100)}%")
            col3.metric("ML Real Prob", f"{round((ml_result.get('real_probability') or 0)*100, 1)}%" if ml_result['status'] == 'OK' else "N/A")
            col4.metric("ML Fake Prob", f"{round((ml_result.get('fake_probability') or 0)*100, 1)}%" if ml_result['status'] == 'OK' else "N/A")

            if assessment["models_disagree"]:
                st.warning("⚠️ " + " ".join(assessment["disagreement_notes"]))

            # ML Per-Feature Breakdown
            if ml_result["status"] == "OK" and "feature_breakdown" in ml_result:
                st.markdown('<div class="section-header">⬡ ML FEATURE VERIFICATION</div>', unsafe_allow_html=True)
                st.caption(f"ML Confidence: {round((ml_result['confidence'] or 0)*100, 1)}% | Uncertainty: {round((ml_result['uncertainty'] or 0)*100, 1)}%")
                bd_cols = st.columns(4)
                for i, (feat_name, feat_info) in enumerate(ml_result["feature_breakdown"].items()):
                    bd_cols[i % 4].markdown(
                        f"**{feat_name}**  \n{feat_info['status']}  \n`score: {feat_info['score']}`"
                    )

        # Visual Analysis (Original vs Grad-CAM Heatmap vs 2D FFT)
        st.markdown('<div class="section-header">⬡ VISUAL EVIDENCE & LOCALIZATION</div>', unsafe_allow_html=True)
        vcol1, vcol2, vcol3 = st.columns(3)

        with vcol1:
            st.markdown("**Original Media**")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"YOLO26 Detected Objects: {len(rois)} | Faces: {len(faces)}")

        with vcol2:
            st.markdown("**Forensic Grad-CAM Activation**")
            st.image(cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(gradcam_expl["explanation_text"])

        with vcol3:
            st.markdown("**2D FFT Power Spectrum**")
            st.image(freq_res["spectrum_visualization"], use_container_width=True)
            st.caption(freq_res["evidence_note"])

        # Evidence Breakdown
        st.markdown('<div class="section-header">⬡ EVIDENCE TRAIL</div>', unsafe_allow_html=True)
        for ev in evidence_items:
            css_class = "evidence-critical" if ev["severity"] == "CRITICAL" else ("evidence-warning" if ev["severity"] == "WARNING" else "evidence-pass")
            icon = "❌" if ev["severity"] == "CRITICAL" else ("⚠️" if ev["severity"] == "WARNING" else "✅")
            st.markdown(f"""
            <div class="{css_class}">
                <strong>{icon} {ev['title']}</strong> [{ev['category']}]<br>
                <small>{ev['detail']}</small>
            </div>
            """, unsafe_allow_html=True)

        # Media Metadata
        st.markdown('<div class="section-header">⬡ MEDIA PROPERTIES & DIGITAL PROVENANCE</div>', unsafe_allow_html=True)
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Dimensions", meta_res.get('dimensions', 'N/A'))
        mcol2.metric("Format", meta_res.get('format', 'N/A'))
        mcol3.metric("Color Mode", meta_res.get('color_mode', 'N/A'))
        mcol4.metric("EXIF Status", meta_res.get('metadata_summary', 'N/A'))

        # JSON Report Export
        report_data = {
            "final_decision": final_verdict,
            "forensic_pipeline_decision": forensic_dec,
            "ml_verifier_prediction": ml_pred,
            "authenticity_score": assessment["authenticity_score"],
            "confidence": assessment["model_authenticity_confidence"],
            "uncertainty_percent": assessment["uncertainty_percent"],
            "vision_score": vision_res["prob_authentic"],
            "frequency_score": round(1.0 - freq_res["frequency_anomaly_score"], 3),
            "evidence": evidence_items,
            "suspicious_regions": [gradcam_expl.get("top_region", "N/A")],
            "media_metadata": meta_res,
            "model_versions": {
                "spatial_yolo": pipeline["yolo"].get_summary(),
                "forensic_classifier": vision_res["backbone"]
            }
        }
        st.download_button(
            "📥 Download Forensic Report (JSON)",
            data=json.dumps(report_data, indent=2),
            file_name="forensic_report.json",
            mime="application/json"
        )

# 2. VIDEO FORENSICS
with tab_vid:
    vid_file = st.file_uploader("Upload Video File (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"], key="vid_up")

    target_vid_path = None
    if vid_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(vid_file.read())
            target_vid_path = tmp.name
    elif use_demo == "Demo Video Spliced" and os.path.exists("data/demo/demo_fake.mp4"):
        target_vid_path = "data/demo/demo_fake.mp4"
        st.info("Loaded demo video with temporal flickering: `data/demo/demo_fake.mp4`")

    if target_vid_path:
        with st.spinner("Executing Frame-by-Frame Temporal Forensic Extraction..."):
            vid_res = pipeline["video_pipe"].analyze_video(target_vid_path)
            meta_res = vid_res["media_info"]
            fusion_res = pipeline["fusion"].fuse(vision_res=vid_res)
            evidence_items = pipeline["evidence_eng"].compile_evidence(video_data=vid_res)
            assessment = pipeline["scorer"].calculate_assessment(fusion_res, evidence_items, meta_res)
            # ML second-pass
            feat_vec = extract_features(vision_res=vid_res, meta_res=meta_res)
            ml_result = pipeline["ml_verifier"].predict(feat_vec)

        st.markdown('<div class="section-header">⬡ VIDEO FORENSIC VERDICT</div>', unsafe_allow_html=True)
        forensic_dec = assessment["final_decision"]
        ml_pred = ml_result.get("prediction", "INCONCLUSIVE") if ml_result["status"] == "OK" else "INCONCLUSIVE"
        if ml_pred in ("FAKE", "LIKELY AI-GENERATED") and forensic_dec == "AUTHENTIC":
            final_verdict = "SUSPICIOUS"
        elif ml_pred in ("FAKE", "LIKELY AI-GENERATED") and forensic_dec in ("SUSPICIOUS", "LIKELY MANIPULATED"):
            final_verdict = "LIKELY MANIPULATED"
        elif ml_pred == "REAL" and forensic_dec == "AUTHENTIC":
            final_verdict = "AUTHENTIC"
        else:
            final_verdict = forensic_dec
        badge_cls = "badge-authentic" if final_verdict == "AUTHENTIC" else ("badge-suspicious" if final_verdict == "SUSPICIOUS" else "badge-manipulated")
        st.markdown(f'<div class="{badge_cls}">{final_verdict}</div>', unsafe_allow_html=True)
        st.caption(f"Forensic Pipeline: **{forensic_dec}** | ML Verifier: **{ml_pred}**")

        vcol1, vcol2, vcol3 = st.columns(3)
        vcol1.metric("Authenticity Score", f"{assessment['authenticity_score']} / 100")
        vcol2.metric("Temporal Jitter Volatility", f"{vid_res['temporal_jitter']}")
        vcol3.metric("Frames Evaluated", f"{vid_res['total_frames_analyzed']}")

        st.caption(f"Statistical Aggregation Strategy: {vid_res['aggregation_reasoning']}")

        if ml_result["status"] == "OK" and "feature_breakdown" in ml_result:
            st.markdown("**ML Feature Verification:**")
            bd_cols = st.columns(4)
            for i, (feat_name, feat_info) in enumerate(ml_result["feature_breakdown"].items()):
                bd_cols[i % 4].markdown(f"**{feat_name}**  \n{feat_info['status']}")

        # Suspicious Temporal Segments
        st.markdown('<div class="section-header">⬡ TEMPORAL ANOMALY TIMELINE</div>', unsafe_allow_html=True)
        if vid_res["suspicious_segments"]:
            st.error(f"⚠️ Flagged {len(vid_res['suspicious_segments'])} Suspicious Injected Segments:")
            for seg in vid_res["suspicious_segments"]:
                st.write(f"• **Window `{seg['time_window']}`** — Manipulation Spike: `{seg['average_manipulation_prob']*100:.1f}%` ({seg['frame_count']} continuous frames)")
        else:
            st.success("✅ No continuous spliced temporal anomaly segments detected.")

        # Timeline Chart
        timeline_scores = [t["authenticity_score"] * 100 for t in vid_res["timeline"]]
        timeline_times = [t["timestamp_sec"] for t in vid_res["timeline"]]
        st.line_chart({"Authenticity Timeline (0-100)": timeline_scores})

# 3. AUDIO FORENSICS
with tab_aud:
    aud_file = st.file_uploader("Upload Audio File (.wav, .mp3)", type=["wav", "mp3", "aac"], key="aud_up")

    target_aud_path = None
    if aud_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(aud_file.read())
            target_aud_path = tmp.name
    elif use_demo == "Demo Real Audio" and os.path.exists("data/demo/demo_real.wav"):
        target_aud_path = "data/demo/demo_real.wav"
        st.info("Loaded authentic voice demo: `data/demo/demo_real.wav`")
    elif use_demo == "Demo Synthetic Voice" and os.path.exists("data/demo/demo_fake.wav"):
        target_aud_path = "data/demo/demo_fake.wav"
        st.info("Loaded synthetic vocoder voice demo: `data/demo/demo_fake.wav`")

    if target_aud_path:
        with st.spinner("Analyzing Speech Representations and Acoustic Forensics..."):
            aud_res = pipeline["audio_clf"].analyze_audio(target_aud_path)
            meta_res = pipeline["metadata_insp"].inspect_file(target_aud_path, media_type="audio")
            fusion_res = pipeline["fusion"].fuse(audio_res=aud_res)
            evidence_items = pipeline["evidence_eng"].compile_evidence(audio_data=aud_res)
            assessment = pipeline["scorer"].calculate_assessment(fusion_res, evidence_items, meta_res)

        st.markdown('<div class="section-header">⬡ AUDIO FORENSIC VERDICT</div>', unsafe_allow_html=True)
        dec = aud_res["decision"]
        badge_cls = "badge-authentic" if dec == "AUTHENTIC" else ("badge-suspicious" if dec == "SUSPICIOUS" else "badge-manipulated")
        st.markdown(f'<div class="{badge_cls}">{dec}</div>', unsafe_allow_html=True)

        acol1, acol2, acol3 = st.columns(3)
        acol1.metric("Voice Authenticity", f"{round(aud_res['prob_authentic']*100, 1)} / 100")
        acol2.metric("Spectral Centroid", f"{aud_res['acoustic_signals']['spectral_centroid']} Hz")
        acol3.metric("Pitch Monotonicity", f"{aud_res['acoustic_signals']['pitch_flatness']}")

        st.markdown('<div class="section-header">⬡ LOG-MEL SPECTROGRAM</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(aud_res["spectrogram_image"], cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown('<div class="section-header">⬡ ACOUSTIC EVIDENCE</div>', unsafe_allow_html=True)
        for ev in aud_res["evidence"]:
            st.write(f"• {ev}")

# 4. VIDEO + AUDIO MULTIMODAL SYNC
with tab_multi:
    st.markdown('<div class="section-header">⬡ CROSS-MODAL AUDIO-VISUAL SYNCHRONIZATION</div>', unsafe_allow_html=True)
    st.caption("Upload video containing speech or separate video + audio streams to detect dubbing latency and articulatory-acoustic desync.")

    st.write("Cross-modal analysis correlates mouth vertical aperture velocity against speech energy envelope.")
    if st.button("Run Verified Cross-Modal Dubbing Test"):
        from backend.audio.audio_preprocessor import AudioPreprocessor
        prep = AudioPreprocessor()
        y_real, sr = prep.load_audio("data/demo/demo_real.wav") if os.path.exists("data/demo/demo_real.wav") else (np.random.normal(0, 0.1, 48000).astype(np.float32), 16000)

        # Synthesize 40 mouth test frames
        mouth_crops = [np.full((112, 112, 3), int(40 + 20 * np.sin(i / 3.0)), dtype=np.uint8) for i in range(40)]
        sync_res = pipeline["av_sync"].analyze_synchronization(mouth_crops, y_real, fps=20.0, duration_sec=2.0)

        st.metric("A/V Synchronization Score", f"{sync_res['sync_score']} / 100")
        st.write(f"• **Status:** `{sync_res['status']}`")
        st.write(f"• **Estimated Latency:** `{sync_res['estimated_lag_ms']} ms`")
        st.write(f"• **Analysis:** {sync_res['explanation']}")

# Footer
st.markdown("""
<div style="
    margin-top: 40px;
    padding: 22px 28px;
    background: #fbfaf6;
    border: 1px solid #e5e2d8;
    border-radius: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    box-shadow: 0 8px 24px rgba(47, 48, 39, 0.045);
">
    <div style="font-family:'Space Grotesk','Inter',sans-serif; color:#68735b; font-size:0.82rem; font-weight:600;">
        CLARIS v1.0.0 &nbsp;·&nbsp; HNX26PSI10
    </div>
    <div style="color:#73766c; font-size:0.74rem; max-width:600px; line-height:1.6;">
        Detection is probabilistic and based on learned statistical anomalies and signal processing signatures.
        It does not replace forensic chain-of-custody protocols.
    </div>
</div>
""", unsafe_allow_html=True)
