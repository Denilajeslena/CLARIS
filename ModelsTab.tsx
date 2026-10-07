import React from "react";
import { Cpu, Layers, Activity, Radio, Eye, Waves, FileCode, CheckCircle2 } from "lucide-react";

interface ModelsTabProps {
  modelSize: "L" | "S";
  onSelectModelSize: (size: "L" | "S") => void;
}

export default function ModelsTab({ modelSize, onSelectModelSize }: ModelsTabProps) {
  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Overview Banner */}
      <div className="space-y-3">
        <div className="text-[11px] font-semibold text-slate-400 tracking-wider uppercase">
          Neural Architecture & Algorithms
        </div>
        <h1 className="font-serif-display text-4xl text-[#0F172A] tracking-tight">
          Forensic Inspection Ensemble
        </h1>
        <p className="text-sm text-slate-600 max-w-2xl leading-relaxed">
          ForensicVision pairs deep convolutional feature representations with classical mathematical signal analysis. Each component serves a distinct role in the chain of custody.
        </p>
      </div>

      {/* Model Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Model 1: YOLO26 */}
        <div className="editorial-card rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Layers className="w-5 h-5 text-teal-700" />
                <h3 className="text-base font-semibold text-[#0F172A]">Ultralytics YOLO26</h3>
              </div>
              <span className="text-[11px] font-mono text-slate-500 uppercase">
                Spatial ROI Context
              </span>
            </div>

            <p className="text-xs text-slate-600 mb-3 leading-relaxed">
              <strong>Forensic Function:</strong> Spatial bounding box localization, human subject detection, and context extraction.
            </p>
            <div className="p-3.5 rounded-xl bg-[#F8FAFD] border border-[#E5E9F0] text-xs text-slate-600 mb-4 leading-relaxed">
              <strong>Technical Role:</strong> YOLO is not used as a standalone deepfake classifier. It isolates high-salience facial and person regions so the dedicated forensic model examines only relevant pixels.
            </div>

            <div className="space-y-2 text-xs font-mono text-slate-500 border-t border-[#E5E9F0] pt-3">
              <div className="flex justify-between">
                <span>Active Model Size:</span>
                <span className="font-bold text-slate-900">YOLO26{modelSize}</span>
              </div>
              <div className="flex justify-between">
                <span>Primary Weights:</span>
                <span className="text-slate-700">models/yolo26l.pt</span>
              </div>
              <div className="flex justify-between">
                <span>Fallback Weights:</span>
                <span className="text-slate-700">models/yolo26s.pt</span>
              </div>
            </div>
          </div>

          <div className="mt-5 pt-4 border-t border-[#E5E9F0] flex items-center justify-between">
            <span className="text-xs text-slate-500">Selected Architecture:</span>
            <div className="flex items-center gap-1 bg-[#F1F4F9] p-1 rounded-xl text-xs">
              <button
                onClick={() => onSelectModelSize("L")}
                className={`px-3 py-1.5 rounded-lg transition cursor-pointer font-medium ${
                  modelSize === "L"
                    ? "bg-white text-slate-900 shadow-sm"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                YOLO26L (High Precision)
              </button>
              <button
                onClick={() => onSelectModelSize("S")}
                className={`px-3 py-1.5 rounded-lg transition cursor-pointer font-medium ${
                  modelSize === "S"
                    ? "bg-white text-slate-900 shadow-sm"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                YOLO26S (Fast Edge)
              </button>
            </div>
          </div>
        </div>

        {/* Model 2: Forensic CNN */}
        <div className="editorial-card rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-teal-700" />
                <h3 className="text-base font-semibold text-[#0F172A]">Forensic CNN (EfficientNet-B4)</h3>
              </div>
              <span className="text-[11px] font-mono text-slate-500 uppercase">
                Primary Classifier
              </span>
            </div>

            <p className="text-xs text-slate-600 mb-3 leading-relaxed">
              <strong>Forensic Function:</strong> Dual-stream classification evaluating Poisson blending boundary gradients and subtle skin texture anomalies.
            </p>
            <p className="text-xs text-slate-500 mb-4 leading-relaxed">
              Combines deep convolutional feature representations with handcrafted forensic metrics (Laplacian variance and YCbCr chrominance covariance).
            </p>

            <div className="space-y-2 text-xs font-mono text-slate-500 border-t border-[#E5E9F0] pt-3">
              <div className="flex justify-between">
                <span>Architecture Backbone:</span>
                <span className="font-bold text-slate-900">tf_efficientnet_b4</span>
              </div>
              <div className="flex justify-between">
                <span>Checkpoint Location:</span>
                <span className="text-slate-700">models/forensic_model/best_checkpoint.pth</span>
              </div>
              <div className="flex justify-between">
                <span>Training Pipeline:</span>
                <span className="text-teal-800">scripts/train_forensic.py</span>
              </div>
            </div>
          </div>

          <div className="mt-5 pt-4 border-t border-[#E5E9F0] text-xs text-slate-500 flex items-center justify-between">
            <span>Runtime Status:</span>
            <span className="text-emerald-700 font-medium flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Multi-Stream Active</span>
            </span>
          </div>
        </div>

        {/* Model 3: Grad-CAM Explainability */}
        <div className="editorial-card rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Eye className="w-5 h-5 text-teal-700" />
                <h3 className="text-base font-semibold text-[#0F172A]">Grad-CAM Spatial Localizer</h3>
              </div>
              <span className="text-[11px] font-mono text-slate-500 uppercase">
                Explainability
              </span>
            </div>

            <p className="text-xs text-slate-600 mb-3 leading-relaxed">
              <strong>Forensic Function:</strong> Computes gradient-weighted class activation heatmaps across the target convolutional head layer.
            </p>
            <p className="text-xs text-slate-500 leading-relaxed mb-4">
              Identifies anatomical regions driving model activation (e.g. mouth seam, eye orbital boundaries, ear-to-jawline interface).
            </p>

            <div className="space-y-2 text-xs font-mono text-slate-500 border-t border-[#E5E9F0] pt-3">
              <div className="flex justify-between">
                <span>Target Layer:</span>
                <span className="font-bold text-slate-900">conv_head / features[-1]</span>
              </div>
              <div className="flex justify-between">
                <span>Intensity Scale:</span>
                <span className="text-slate-700">Jet Colormap [0.0 - 1.0]</span>
              </div>
            </div>
          </div>

          <div className="mt-5 pt-4 border-t border-[#E5E9F0] text-xs text-slate-500 flex items-center justify-between">
            <span>Runtime Status:</span>
            <span className="text-emerald-700 font-medium flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Gradient Hook Ready</span>
            </span>
          </div>
        </div>

        {/* Model 4: Speech Representations & Audio */}
        <div className="editorial-card rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Waves className="w-5 h-5 text-teal-700" />
                <h3 className="text-base font-semibold text-[#0F172A]">Wav2Vec2 + Speech CRNN</h3>
              </div>
              <span className="text-[11px] font-mono text-slate-500 uppercase">
                Acoustic Forensics
              </span>
            </div>

            <p className="text-xs text-slate-600 mb-3 leading-relaxed">
              <strong>Forensic Function:</strong> Pretrained speech representation evaluating phoneme transitions, pitch monotonicity, and vocoder brick-wall cutoffs.
            </p>
            <p className="text-xs text-slate-500 leading-relaxed mb-4">
              Flags phase discontinuities and unnatural high-frequency cutoffs produced by neural vocoders (HiFi-GAN, WaveGlow).
            </p>

            <div className="space-y-2 text-xs font-mono text-slate-500 border-t border-[#E5E9F0] pt-3">
              <div className="flex justify-between">
                <span>Speech Encoder:</span>
                <span className="font-bold text-slate-900">facebook/wav2vec2-base</span>
              </div>
              <div className="flex justify-between">
                <span>Standard Sampling:</span>
                <span className="text-slate-700">16,000 Hz Mono</span>
              </div>
            </div>
          </div>

          <div className="mt-5 pt-4 border-t border-[#E5E9F0] text-xs text-slate-500 flex items-center justify-between">
            <span>Runtime Status:</span>
            <span className="text-emerald-700 font-medium flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Acoustic Core Ready</span>
            </span>
          </div>
        </div>

        {/* Model 5: Cross-Modal AV Sync */}
        <div className="editorial-card rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Radio className="w-5 h-5 text-teal-700" />
                <h3 className="text-base font-semibold text-[#0F172A]">Audio-Visual Synchronization</h3>
              </div>
              <span className="text-[11px] font-mono text-slate-500 uppercase">
                Cross-Modal Dubbing
              </span>
            </div>

            <p className="text-xs text-slate-600 mb-3 leading-relaxed">
              <strong>Forensic Function:</strong> Correlates vertical mouth aperture velocity with speech audio RMS energy using normalized cross-correlation.
            </p>
            <p className="text-xs text-slate-500 leading-relaxed mb-4">
              Identifies phase offsets in milliseconds and flags decoupled temporal windows characteristic of synthetic dubbing.
            </p>

            <div className="space-y-2 text-xs font-mono text-slate-500 border-t border-[#E5E9F0] pt-3">
              <div className="flex justify-between">
                <span>Correlation Method:</span>
                <span className="font-bold text-slate-900">Pearson Cross-Correlation</span>
              </div>
              <div className="flex justify-between">
                <span>Sliding Window:</span>
                <span className="text-slate-700">0.8s - 1.5s Adaptive</span>
              </div>
            </div>
          </div>

          <div className="mt-5 pt-4 border-t border-[#E5E9F0] text-xs text-slate-500 flex items-center justify-between">
            <span>Runtime Status:</span>
            <span className="text-emerald-700 font-medium flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>A/V Correlation Active</span>
            </span>
          </div>
        </div>

        {/* Model 6: 2D FFT & ELA */}
        <div className="editorial-card rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <FileCode className="w-5 h-5 text-teal-700" />
                <h3 className="text-base font-semibold text-[#0F172A]">2D FFT & Error Level Analysis</h3>
              </div>
              <span className="text-[11px] font-mono text-slate-500 uppercase">
                Mathematical Signal
              </span>
            </div>

            <p className="text-xs text-slate-600 mb-3 leading-relaxed">
              <strong>Forensic Function:</strong> Evaluates azimuthal radial energy decay against the natural 1/f power law and measures JPEG quantization variance across recompressed blocks.
            </p>
            <p className="text-xs text-slate-500 leading-relaxed mb-4">
              Provides verifiable, mathematical proof without relying solely on neural network representations.
            </p>

            <div className="space-y-2 text-xs font-mono text-slate-500 border-t border-[#E5E9F0] pt-3">
              <div className="flex justify-between">
                <span>FFT Spectrum Grid:</span>
                <span className="font-bold text-slate-900">128 × 128 Centered Magnitude</span>
              </div>
              <div className="flex justify-between">
                <span>ELA Recompression:</span>
                <span className="text-slate-700">JPEG Quality 85% Matrix Delta</span>
              </div>
            </div>
          </div>

          <div className="mt-5 pt-4 border-t border-[#E5E9F0] text-xs text-slate-500 flex items-center justify-between">
            <span>Runtime Status:</span>
            <span className="text-emerald-700 font-medium flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Mathematical DSP Active</span>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
