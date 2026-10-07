# CLARIS
# ForensicVision: Multimodal Deepfake & Digital Forensics System
### Hackathon Problem ID: HNX26PSI10 (Multimodal Deepfake & Digital Forensics)

**Domain:** Computer Vision | Generative AI | Audio AI | Digital Forensics  
**Status:** Working Local Prototype with Real ML & Signal Processing Pipeline

---

## 1. System Architecture

```
                                  USER MEDIA
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
            IMAGE                   VIDEO                   AUDIO
              │                       │                       │
              │                  OpenCV Video             Audio Loader
              │                  Frame Sampler          (16kHz Resampler)
              │                       │                       │
              └───────────┬───────────┴───────────┬───────────┘
                          │                       │
                       YOLO26L                 Wav2Vec2 /
                      / YOLO26S               Speech CRNN
                          │                       │
                    Spatial ROIs &          Acoustic Features
                     Face Crops            (MFCC, ZCR, Centroid)
                          │                       │
                    Forensic CNN                  │
                   (EfficientNet /             Vocoder
                      Xception)             Discontinuity
                          │                       │
                    Grad-CAM Map                  │
                          │                       │
                          └───────────┬───────────┘
                                      │
                                 Cross-Modal
                              A/V Synchronization
                           (Mouth Flow ↔ Voice Envelope)
                                      │
                            2D FFT / DCT Analysis
                          (Periodic High-Freq Spikes)
                                      │
                            Temporal Aggregation
                           (Trimmed Mean & Jitter)
                                      │
                              EVIDENCE ENGINE
                                      │
                           Bayesian Fusion & Scoring
                             (Score + Uncertainty)
                                      │
                         Explainable Forensic Report
                                      │
                     STREAMLIT UI / FORENSIC STUDIO
```

---

## 2. Exact Project Folder Structure

```
├── configs/
│   └── config.yaml                     # Central configuration (models, thresholds, weights)
├── backend/
│   ├── vision/
│   │   ├── yolo_detector.py            # YOLO26L (High Precision) and YOLO26S (Fast Fallback)
│   │   ├── face_detector.py            # Local face detector, landmark alignment, margin expansion
│   │   ├── forensic_classifier.py      # EfficientNet-B4 / Xception CNN backbone & artifact signals
│   │   ├── gradcam.py                  # PyTorch hook-based Grad-CAM activation heatmap generator
│   │   └── frequency_analysis.py       # 2D Fast Fourier Transform (FFT) & 8x8 block DCT analysis
│   ├── video/
│   │   ├── frame_extractor.py          # OpenCV video reader & configurable subsampling
│   │   ├── temporal_analyzer.py        # Robust trimmed-mean aggregation & suspicious segment locator
│   │   └── video_forensics.py          # End-to-end video pipeline orchestrator
│   ├── audio/
│   │   ├── audio_preprocessor.py       # 16kHz resampler, ZCR, spectral centroid, pitch prosody
│   │   ├── spectrogram.py              # Log-Mel spectrogram generator & vocoder cutoff detector
│   │   └── audio_model.py              # Wav2Vec2 speech encoder + acoustic spectrogram CRNN
│   ├── multimodal/
│   │   ├── av_sync.py                  # Articulatory mouth motion ↔ audio envelope correlation
│   │   └── fusion.py                   # Bayesian multi-signal fusion & modality conflict detection
│   ├── forensics/
│   │   ├── metadata.py                 # EXIF, camera make, container codecs, dimensions
│   │   ├── compression_analysis.py     # JPEG Error Level Analysis (ELA) & frame duplication
│   │   ├── evidence_engine.py          # Explainable AI synthesizer binding metrics to facts
│   │   └── scoring.py                  # Calibrated authenticity scoring (0-100) & uncertainty
│   └── evaluation/
│       ├── metrics.py                  # ROC-AUC, Balanced Accuracy, F1, FPR, FNR calculator
│       ├── robustness.py               # Perturbation tests (JPEG compression, downscaling, noise)
│       └── unseen_tests.py             # Zero-shot generalization tests on novel generative methods
├── frontend/
│   └── app.py                          # Streamlit dark-mode forensic laboratory interface
├── models/
│   ├── yolo26l.pt                      # YOLO26L weights placeholder / checkpoint
│   ├── yolo26s.pt                      # YOLO26S weights placeholder / checkpoint
│   ├── forensic_model/                 # Fine-tuned forensic CNN weights
│   └── audio_model/                    # Audio classifier checkpoints
├── data/
│   ├── real/                           # Authentic reference media
│   ├── fake/                           # Deepfake manipulation media
│   ├── validation/                     # Validation split
│   ├── test/                           # Held-out test split
│   ├── unseen/                         # Novel generative architectures
│   └── demo/                           # Offline verified test media
├── scripts/
│   ├── prepare_dataset.py              # Video identity-level splitting (zero frame leakage)
│   ├── train_forensic.py               # Transfer learning with robustness augmentations
│   ├── train_audio.py                  # Audio classifier fine-tuning script
│   ├── evaluate.py                     # Benchmark runner reporting ROC-AUC and metrics
│   └── demo_runner.py                  # Generates offline verified test media
├── tests/
│   └── test_forensics.py               # Unit & integration test suite
├── requirements.txt                    # Python dependencies
└── README.md                           # Documentation
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10 or 3.11+
- Recommended: NVIDIA GPU with CUDA 11.8 / 12.1 (CPU fallback runs automatically)

```bash
# 1. Clone repository & create virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate verified offline demo media
python scripts/demo_runner.py

# 4. Run automated unit & integration tests
python tests/test_forensics.py
```

---

## 4. Launching the Application

### Streamlit Forensic Dashboard
```bash
streamlit run frontend/app.py
```

The Streamlit UI provides:
- Single-page dark forensic laboratory theme
- Model switching between **YOLO26L** and **YOLO26S**
- Instant verified test suite presets (Deepfake FaceSwap, Authentic Portrait, Spliced Video, Synthetic Voice)
- Side-by-side Visual Evidence: Input Media vs Grad-CAM Activation Heatmap vs 2D FFT Power Spectrum
- Frame-by-frame temporal timeline with flagged suspicious spliced windows (e.g., `00:04.2 - 00:06.8`)
- Log-Mel Spectrogram with neural vocoder brick-wall cutoff indicator
- Cross-modal articulatory lipsync alignment score (0 - 100) and phase lag in milliseconds
- One-click export of `forensic_report.json`

---

## 5. Model System & Hardware Strategy

| Component | Primary Model | Fallback / Fast Model | Role in Forensic Pipeline |
|---|---|---|---|
| **Spatial Context** | **YOLO26L** | **YOLO26S** | Person & scene localization, context bounding boxes, ROI cropping |
| **Face & Landmarks** | OpenCV YuNet DNN | Haar Cascade | 5-point facial landmark alignment, boundary margin expansion |
| **Forensic CNN** | EfficientNet-B4 / Xception | Custom Multi-Stream Conv | Classification, Poisson blending seam & texture artifact detection |
| **Explainability** | Grad-CAM (Target: conv_head) | Spatial Gradient Delta | Gradient-weighted activation heatmaps & anatomical region localization |
| **Frequency Domain** | 2D FFT + 8x8 DCT | Radial Profile | Periodic GAN checkerboard spikes & 1/f spectral roll-off analysis |
| **Speech Forensics** | Wav2Vec2-Base | Acoustic-Mel CRNN | Latent speech representations, vocoder cutoff, pitch monotonicity |
| **Cross-Modal Sync** | Mouth Optical Velocity ↔ Voice Envelope | Zero-Lag Pearson Core | Articulatory-acoustic phase lag & dubbing mismatch detection |

### GPU / CPU Automatic Resolution
- The pipeline detects `torch.cuda.is_available()`.
- If CUDA is present: executes in FP16 on GPU for low-latency inference.
- If GPU is unavailable: automatically falls back to CPU execution without crashing or throwing errors.
- Active hardware status is prominently displayed in the UI header.

---

## 6. Anti-Overfitting & Scientific Integrity

1. **Anti-Data Leakage Splitting:**
   `scripts/prepare_dataset.py` splits data strictly by **Video Identity**, ensuring frames from the same source video NEVER appear in both training and test sets.
2. **Robustness Augmentations:**
   `scripts/train_forensic.py` applies stochastic JPEG compression ($Q \in [35, 85]$), aggressive downscale-upscale resampling, and color jitter to ensure resilience against WhatsApp, Telegram, and social media transcoding.
3. **Unseen Generalization Benchmarking:**
   `backend/evaluation/unseen_tests.py` reports performance splits on known vs held-out novel generative methods (reporting ROC-AUC, Balanced Accuracy, F1, FPR, and FNR).
4. **Probabilistic Disclaimer:**
   The system explicitly states:
   > *"Deepfake detection is probabilistic and based on learned statistical anomalies and signal processing signatures. It does not replace forensic chain-of-custody protocols."*
