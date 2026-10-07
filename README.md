# CLARIS

### Multimodal Deepfake & Digital Forensics

> **HNX26PSI10 · HackNex 2026**

**CLARIS** is a multimodal digital-forensics platform designed to investigate suspicious **images, videos, and audio**. Instead of relying on a single deepfake classifier, CLARIS combines visual, temporal, acoustic, frequency-domain, synchronization, and metadata signals to produce an **explainable authenticity assessment**.

**Detect the manipulation. Trace the evidence. Explain the decision.**

---

## Why CLARIS?

Deepfake detection is no longer just a binary *real vs fake* problem.

A manipulated video can contain multiple signals:

* Facial and texture inconsistencies
* Abnormal frequency patterns
* Temporal artifacts across frames
* Synthetic or manipulated speech characteristics
* Audio–visual synchronization errors
* Metadata and compression anomalies

CLARIS brings these signals together into a unified **forensic evidence pipeline**.

```text
                    SUSPICIOUS MEDIA
                           │
            ┌──────────────┼──────────────┐
            │              │              │
          IMAGE          VIDEO          AUDIO
            │              │              │
       Visual Forensics  Temporal      Acoustic
       Frequency        Analysis      Analysis
            │              │              │
            └──────────────┼──────────────┘
                           │
                    CROSS-MODAL ANALYSIS
                           │
                 ┌─────────┴─────────┐
                 │                   │
              A/V Sync          Metadata &
              Analysis          Compression
                 │                   │
                 └─────────┬─────────┘
                           │
                    EVIDENCE ENGINE
                           │
                 MULTI-SIGNAL FUSION
                           │
              ┌────────────┴────────────┐
              │                         │
        AUTHENTICITY SCORE       FORENSIC EVIDENCE
              │                         │
              └────────────┬────────────┘
                           │
                  EXPLAINABLE REPORT
```

---

# Key Differentiator

### From Deepfake Detection → Digital Forensic Investigation

| Conventional Detector | CLARIS                               |
| --------------------- | ------------------------------------ |
| Real / Fake           | **Authenticity score + uncertainty** |
| Single modality       | **Multimodal analysis**              |
| Black-box prediction  | **Evidence-backed explanation**      |
| Model output          | **Forensic evidence timeline**       |
| Artifact detection    | **Cross-signal correlation**         |
| Detection only        | **Investigation + reporting**        |

The core novelty of CLARIS is **multimodal forensic evidence fusion** — correlating independent signals rather than trusting a single artifact or model.

---

# Core Capabilities

### 01 · Visual Forensics

Analyze suspicious regions and facial content for manipulation artifacts.

* Face detection and alignment
* Region-of-interest extraction
* Texture and blending analysis
* CNN-based forensic classification
* Grad-CAM explainability
* Frequency-domain analysis

### 02 · Video Forensics

Investigate manipulation across time instead of analyzing a single frame.

* Configurable frame sampling
* Temporal anomaly detection
* Suspicious-segment localization
* Frame duplication analysis
* Temporal consistency analysis

Example:

```text
00:00 ─────── 00:04 ─────── 00:07 ─────── 00:12
                ▲
                │
        Suspicious Segment
        00:04.2 → 00:06.8
```

### 03 · Audio Forensics

Analyze speech and acoustic characteristics for signs of synthetic generation or manipulation.

* 16 kHz preprocessing
* Log-Mel spectrograms
* MFCC features
* Zero-crossing rate
* Spectral characteristics
* Pitch/prosody analysis
* Neural speech representations

### 04 · Audio–Visual Synchronization

A major forensic signal is whether the **mouth movement and speech actually agree**.

CLARIS compares:

```text
Mouth Motion
     │
     ├──────────────┐
     │              │
     ▼              ▼
Optical Motion   Audio Envelope
     │              │
     └──────┬───────┘
            ▼
      Correlation
            │
            ▼
     Sync / Phase Lag
```

This can help identify potential dubbing, lip-sync manipulation, or cross-modal inconsistencies.

### 05 · Frequency-Domain Analysis

Spatial artifacts are not always obvious to the human eye.

CLARIS can inspect frequency-domain characteristics using:

* 2D FFT
* DCT-based analysis
* Spectral energy distribution
* High-frequency anomaly detection
* Periodic artifact analysis

### 06 · Metadata & Compression Forensics

Supporting evidence can also come from the media container itself.

Potential signals include:

* EXIF information
* Camera/device information
* Container metadata
* Codec information
* Resolution and dimensions
* Compression characteristics
* JPEG artifacts
* Frame duplication

> Metadata is treated as **supporting evidence**, not proof of authenticity.

---

# Evidence Engine

The evidence engine combines multiple signals instead of allowing one detector to dominate the final decision.

```text
Visual Evidence ─────┐
                     │
Temporal Evidence ───┤
                     │
Audio Evidence ──────┤
                     ├──► Evidence Fusion ──► Score
A/V Sync ────────────┤                         │
                     │                         ▼
Frequency Evidence ──┤                  Explanation
                     │
Metadata ────────────┘
```

### Example Output

```text
AUTHENTICITY ASSESSMENT
────────────────────────────────

Score              27 / 100
Assessment         HIGH SUSPICION
Confidence         0.86

Evidence

[HIGH] Facial texture inconsistency
[HIGH] A/V synchronization anomaly
[MED]  Frequency-domain irregularity
[MED]  Temporal discontinuity
[LOW]  Metadata inconsistency

Suspicious Segment

00:04.2 ───────── 00:06.8
```

The interface should make it possible for an investigator to understand **why** the system reached its conclusion.

---

# System Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                    CLARIS FORENSIC STUDIO               │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
                    MEDIA INGESTION
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
       IMAGE              VIDEO              AUDIO
          │                  │                  │
          ▼                  ▼                  ▼
    Vision Pipeline    Frame Pipeline     Audio Pipeline
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼
                  CROSS-MODAL ANALYSIS
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
           A/V Sync      Frequency       Metadata
           Analysis       Analysis       Analysis
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                     EVIDENCE ENGINE
                             │
                             ▼
                    SIGNAL FUSION
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
        AUTHENTICITY SCORE         EVIDENCE MAP
                │                         │
                └────────────┬────────────┘
                             ▼
                    FORENSIC REPORT
```

---

# Project Structure

```text
CLARIS/
│
├── configs/
│   └── config.yaml
│
├── backend/
│   ├── vision/
│   │   ├── yolo_detector.py
│   │   ├── face_detector.py
│   │   ├── forensic_classifier.py
│   │   ├── gradcam.py
│   │   └── frequency_analysis.py
│   │
│   ├── video/
│   │   ├── frame_extractor.py
│   │   ├── temporal_analyzer.py
│   │   └── video_forensics.py
│   │
│   ├── audio/
│   │   ├── audio_preprocessor.py
│   │   ├── spectrogram.py
│   │   └── audio_model.py
│   │
│   ├── multimodal/
│   │   ├── av_sync.py
│   │   └── fusion.py
│   │
│   ├── forensics/
│   │   ├── metadata.py
│   │   ├── compression_analysis.py
│   │   ├── evidence_engine.py
│   │   └── scoring.py
│   │
│   └── evaluation/
│       ├── metrics.py
│       ├── robustness.py
│       └── unseen_tests.py
│
├── frontend/
│   └── app.py
│
├── models/
│   ├── yolo26l.pt
│   ├── yolo26s.pt
│   ├── forensic_model/
│   └── audio_model/
│
├── data/
│   ├── real/
│   ├── fake/
│   ├── validation/
│   ├── test/
│   ├── unseen/
│   └── demo/
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── train_forensic.py
│   ├── train_audio.py
│   ├── evaluate.py
│   └── demo_runner.py
│
├── tests/
│   └── test_forensics.py
│
├── requirements.txt
└── README.md
```

---

# Technology Stack

| Layer             | Technology              |
| ----------------- | ----------------------- |
| Frontend          | Streamlit               |
| Computer Vision   | OpenCV                  |
| Object Detection  | YOLO                    |
| Deep Learning     | PyTorch                 |
| Forensic CNN      | EfficientNet / Xception |
| Audio AI          | Wav2Vec2 / CRNN         |
| Signal Processing | NumPy / SciPy           |
| Visualization     | Matplotlib              |
| Explainability    | Grad-CAM                |
| Data Processing   | Pandas                  |
| Configuration     | YAML                    |
| Deployment        | Local / GPU / CPU       |

---

# Hardware Strategy

CLARIS is designed to operate on both GPU and CPU environments.

### GPU Mode

```text
NVIDIA CUDA
     │
     ▼
PyTorch GPU Inference
     │
     ▼
Accelerated Forensic Analysis
```

### CPU Fallback

```text
No CUDA
   │
   ▼
CPU Inference
   │
   ▼
Reduced-Speed Analysis
```

The application should detect the available compute device automatically.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

# Installation

## Requirements

* Python 3.10+
* 8 GB+ RAM recommended
* NVIDIA GPU recommended for faster inference
* CUDA-compatible PyTorch installation for GPU acceleration

### 1. Clone

```bash
git clone <YOUR_REPOSITORY_URL>
cd CLARIS
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Tests

```bash
python tests/test_forensics.py
```

### 5. Launch CLARIS

```bash
streamlit run frontend/app.py
```

---

# Forensic Workflow

```text
01  Upload Media
        ↓
02  Media Validation
        ↓
03  Metadata Extraction
        ↓
04  Frame / Audio Processing
        ↓
05  Modality-Specific Analysis
        ↓
06  Cross-Modal Verification
        ↓
07  Evidence Fusion
        ↓
08  Authenticity Assessment
        ↓
09  Evidence Timeline
        ↓
10  Forensic Report
```

---

# Explainability

CLARIS is designed around **evidence-first AI**.

Instead of:

```text
❌ FAKE
```

the system aims to produce:

```text
┌─────────────────────────────────────┐
│       FORENSIC ASSESSMENT           │
├─────────────────────────────────────┤
│ Authenticity: 27 / 100              │
│ Risk: HIGH                          │
│                                     │
│ Evidence:                           │
│                                     │
│ ● Facial artifact       HIGH        │
│ ● A/V mismatch          HIGH        │
│ ● Frequency anomaly     MEDIUM      │
│ ● Temporal anomaly      MEDIUM      │
│                                     │
│ Suspicious interval:                │
│ 00:04.2 → 00:06.8                   │
└─────────────────────────────────────┘
```

Where possible, visual explanations such as **Grad-CAM heatmaps**, spectrograms, frequency maps, and temporal markers are presented alongside the prediction.

---

# Scientific Integrity

Deepfake detection is highly sensitive to **dataset bias, compression, unseen generators, and distribution shifts**.

CLARIS therefore follows several evaluation principles.

### Identity-Level Data Splitting

Frames from the same source video should never appear across both training and testing datasets.

```text
Source Video A
      │
      └──► TRAIN

Source Video B
      │
      └──► TEST
```

This reduces frame-level data leakage.

### Robustness Testing

Evaluation can include perturbations such as:

* JPEG compression
* Resolution reduction
* Noise
* Re-encoding
* Color changes
* Downscale/upscale operations

### Unseen Generator Testing

Models should also be evaluated against manipulation techniques that were **not present during training**.

Relevant metrics include:

```text
ROC-AUC
F1 Score
Balanced Accuracy
False Positive Rate
False Negative Rate
Precision
Recall
```

---

# Intended Users

CLARIS is designed primarily for:

* **Digital Forensics Investigators**
* **Cybercrime / Police Cyber Cells**
* **Fact-Checking & Journalism Teams**
* **SOC & Corporate Security Teams**
* **Researchers & Academic Labs**
* **Legal and investigative professionals**

The core user is the **digital-forensics investigator** who needs both a prediction and supporting evidence.

---

# Novelty

### Multimodal Evidence Fusion

The primary innovation is not simply detecting deepfakes.

CLARIS treats authenticity assessment as a **forensic evidence-fusion problem**.

```text
                    ┌── Visual
                    │
                    ├── Temporal
                    │
                    ├── Audio
                    │
Media ──────────────┼── A/V Sync
                    │
                    ├── Frequency
                    │
                    └── Metadata
                           │
                           ▼
                    Evidence Fusion
                           │
                           ▼
                 Explainable Assessment
```

This enables the system to identify **conflicting signals** instead of blindly trusting a single classifier.

---

# Responsible Use

CLARIS provides a **probabilistic forensic assessment**, not an absolute declaration of truth.

> Deepfake detection is probabilistic and dependent on learned patterns, signal-processing characteristics, model limitations, and available evidence.

The output should **not replace professional forensic examination, legal procedures, or chain-of-custody protocols**.

---

# Current Status

```text
┌──────────────────────────────────────────┐
│              CLARIS STATUS               │
├──────────────────────────────────────────┤
│ Multimodal Architecture       ✓          │
│ Visual Analysis               ✓          │
│ Audio Analysis                ✓          │
│ Temporal Analysis             ✓          │
│ Frequency Analysis            ✓          │
│ A/V Synchronization           ✓          │
│ Evidence Fusion               ✓          │
│ Explainable Reporting         ✓          │
│ Robustness Evaluation         ◐          │
│ Large-scale Benchmarking      ◐          │
└──────────────────────────────────────────┘
```

> **Note:** Feature status should be updated to reflect the actual implementation in the repository.

---

# HackNex 2026

**Problem ID:** `HNX26PSI10`

**Track:** Multimodal Deepfake & Digital Forensics

**Domains:**

`Computer Vision` · `Generative AI` · `Audio AI` · `Digital Forensics`

### One-line pitch

> **CLARIS transforms deepfake detection from a black-box prediction into an explainable multimodal forensic investigation.**

---

## Team

**CLARIS — HackNex 2026**

Built for **HNX26PSI10 · Multimodal Deepfake & Digital Forensics**

---

<p align="center">

**Detect · Correlate · Explain**

</p>
