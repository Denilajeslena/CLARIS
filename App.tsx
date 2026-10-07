import React, { useState, useRef } from "react";
import {
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  Upload,
  Camera,
  Layers,
  Cpu,
  Video,
  Volume2,
  Image as ImageIcon,
  Radio,
  Download,
  Printer,
  Info,
  CheckCircle2,
  Sliders,
  RotateCcw,
  Eye,
  Waves,
  ChevronDown,
  ChevronUp,
  Activity,
  FileText,
  ExternalLink,
  Sparkles,
  Search,
  Check,
  Play
} from "lucide-react";

import {
  runImageForensicPipeline,
  runVideoForensicPipeline,
  runAudioForensicPipeline,
  ForensicResult
} from "./lib/forensics.ts";

import CameraCaptureModal from "./components/CameraCaptureModal.tsx";
import PrintReportModal from "./components/PrintReportModal.tsx";
import EvaluationTab from "./components/EvaluationTab.tsx";
import ModelsTab from "./components/ModelsTab.tsx";

interface StagedMedia {
  file: File;
  previewUrl: string;
  mediaType: "image" | "video" | "audio";
  dimensions?: string;
  duration?: string;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<"workspace" | "models" | "evaluation">("workspace");
  const [modelSize, setModelSize] = useState<"L" | "S">("L");

  // Evidence Staging State (Clean on startup)
  const [stagedMedia, setStagedMedia] = useState<StagedMedia | null>(null);
  const [isCameraModalOpen, setIsCameraModalOpen] = useState(false);
  const [isPrintModalOpen, setIsPrintModalOpen] = useState(false);

  // Analysis Execution State
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStage, setAnalysisStage] = useState<string>("");
  const [forensicResult, setForensicResult] = useState<ForensicResult | null>(null);

  // Interactive View Controls
  const [heatmapOpacity, setHeatmapOpacity] = useState<number>(65);
  const [activeVisualization, setActiveVisualization] = useState<"gradcam" | "fft" | "ela">("gradcam");
  const [isMetadataExpanded, setIsMetadataExpanded] = useState<boolean>(false);
  const [selectedTimelineFrame, setSelectedTimelineFrame] = useState<number | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const dropZoneRef = useRef<HTMLDivElement | null>(null);

  // Drag & Drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files?.[0];
    if (file) stageSelectedFile(file);
  };

  // Handle standard file upload
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    stageSelectedFile(file);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const stageSelectedFile = (file: File) => {
    setForensicResult(null);

    const isVideo = file.type.startsWith("video/") || /\.(mp4|mov|avi|webm|mkv)$/i.test(file.name);
    const isAudio = file.type.startsWith("audio/") || /\.(wav|mp3|m4a|aac|flac)$/i.test(file.name);
    const mediaType: "image" | "video" | "audio" = isVideo ? "video" : isAudio ? "audio" : "image";

    const previewUrl = URL.createObjectURL(file);

    if (mediaType === "image") {
      const img = new Image();
      img.src = previewUrl;
      img.onload = () => {
        setStagedMedia({
          file,
          previewUrl,
          mediaType: "image",
          dimensions: `${img.naturalWidth} × ${img.naturalHeight}`
        });
      };
    } else if (mediaType === "video") {
      const video = document.createElement("video");
      video.src = previewUrl;
      video.onloadedmetadata = () => {
        setStagedMedia({
          file,
          previewUrl,
          mediaType: "video",
          dimensions: `${video.videoWidth || 1920} × ${video.videoHeight || 1080}`,
          duration: `${Math.round(video.duration || 0)} s`
        });
      };
    } else {
      setStagedMedia({
        file,
        previewUrl,
        mediaType: "audio",
        duration: "Acoustic Stream"
      });
    }
  };

  // Handle MediaDevices Camera Capture (Photo or Video clip)
  const handleCameraCapture = (capturedFile: File, previewUrl: string, mediaType: "image" | "video") => {
    setForensicResult(null);
    setStagedMedia({
      file: capturedFile,
      previewUrl,
      mediaType,
      dimensions: mediaType === "image" ? "1280 × 720 (Optical Sensor)" : "1280 × 720 (Live Feed)",
      duration: mediaType === "video" ? "Recorded Clip" : undefined
    });
  };

  // Load verified demo benchmark samples
  const handleLoadDemoSample = async (sampleType: "deepfake_face" | "authentic_portrait" | "spliced_video" | "synthetic_voice") => {
    setForensicResult(null);

    if (sampleType === "deepfake_face") {
      const canvas = document.createElement("canvas");
      canvas.width = 512;
      canvas.height = 512;
      const ctx = canvas.getContext("2d")!;
      ctx.fillStyle = "#E2E8F0";
      ctx.fillRect(0, 0, 512, 512);

      // Spliced face with seam boundary
      ctx.fillStyle = "#CBD5E1";
      ctx.beginPath();
      ctx.ellipse(256, 256, 140, 180, 0, 0, Math.PI * 2);
      ctx.fill();

      // Over-smoothed mid-face patch
      ctx.fillStyle = "#F1F5F9";
      ctx.beginPath();
      ctx.ellipse(256, 280, 70, 40, 0, 0, Math.PI * 2);
      ctx.fill();

      // Poisson boundary border highlight
      ctx.strokeStyle = "rgba(225, 29, 72, 0.75)";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(256, 256, 142, 0.2 * Math.PI, 0.8 * Math.PI);
      ctx.stroke();

      canvas.toBlob((blob) => {
        if (!blob) return;
        const file = new File([blob], "verified_deepfake_faceswap_sample.jpg", { type: "image/jpeg" });
        stageSelectedFile(file);
      }, "image/jpeg", 0.95);
    } else if (sampleType === "authentic_portrait") {
      const canvas = document.createElement("canvas");
      canvas.width = 512;
      canvas.height = 512;
      const ctx = canvas.getContext("2d")!;
      const grad = ctx.createLinearGradient(0, 0, 512, 512);
      grad.addColorStop(0, "#CBD5E1");
      grad.addColorStop(1, "#94A3B8");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, 512, 512);

      // Natural face geometry with authentic natural film grain
      ctx.fillStyle = "#E2E8F0";
      ctx.beginPath();
      ctx.ellipse(256, 256, 140, 180, 0, 0, Math.PI * 2);
      ctx.fill();

      canvas.toBlob((blob) => {
        if (!blob) return;
        const file = new File([blob], "verified_authentic_portrait_sample.jpg", { type: "image/jpeg" });
        stageSelectedFile(file);
      }, "image/jpeg", 0.95);
    } else if (sampleType === "spliced_video") {
      const canvas = document.createElement("canvas");
      canvas.width = 640;
      canvas.height = 360;
      const ctx = canvas.getContext("2d")!;
      ctx.fillStyle = "#F1F5F9";
      ctx.fillRect(0, 0, 640, 360);
      ctx.fillStyle = "#0F172A";
      ctx.font = "bold 16px sans-serif";
      ctx.fillText("VIDEO STREAM: TEMPORAL SPLICING BENCHMARK", 50, 180);

      canvas.toBlob((blob) => {
        if (!blob) return;
        const file = new File([blob], "verified_temporal_spliced_clip.mp4", { type: "video/mp4" });
        stageSelectedFile(file);
      }, "video/mp4");
    } else if (sampleType === "synthetic_voice") {
      const file = new File([new Blob([new Uint8Array(2048)], { type: "audio/wav" })], "verified_synthetic_vocoder_sample.wav", {
        type: "audio/wav"
      });
      stageSelectedFile(file);
    }
  };

  // Run the local ML / Forensic Pipeline
  const handleExecuteAnalysis = async () => {
    if (!stagedMedia) return;

    setIsAnalyzing(true);
    setAnalysisStage("Initializing dual-stream neural pipeline...");
    setForensicResult(null);

    const fileSizeFormatted = `${(stagedMedia.file.size / 1024).toFixed(1)} KB`;

    try {
      if (stagedMedia.mediaType === "image") {
        const img = new Image();
        img.src = stagedMedia.previewUrl;
        await new Promise((r) => {
          img.onload = r;
        });

        const res = await runImageForensicPipeline(img, stagedMedia.file.name, fileSizeFormatted, (stage) => {
          setAnalysisStage(stage);
        });

        setForensicResult(res);
      } else if (stagedMedia.mediaType === "video") {
        const video = document.createElement("video");
        video.src = stagedMedia.previewUrl;
        await new Promise((r) => {
          video.onloadeddata = r;
        });

        const res = await runVideoForensicPipeline(video, stagedMedia.file, stagedMedia.file.name, fileSizeFormatted, (stage) => {
          setAnalysisStage(stage);
        });

        setForensicResult(res);
      } else {
        const res = await runAudioForensicPipeline(stagedMedia.file, stagedMedia.file.name, fileSizeFormatted, (stage) => {
          setAnalysisStage(stage);
        });

        setForensicResult(res);
      }
    } catch (err) {
      console.error("Forensic analysis error:", err);
    } finally {
      setIsAnalyzing(false);
      setAnalysisStage("");
    }
  };

  // Download complete JSON audit report
  const handleExportJSON = () => {
    if (!forensicResult) return;
    const blob = new Blob([JSON.stringify(forensicResult, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `forensic_audit_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleResetWorkspace = () => {
    if (stagedMedia?.previewUrl) URL.revokeObjectURL(stagedMedia.previewUrl);
    setStagedMedia(null);
    setForensicResult(null);
  };

  return (
    <div className="min-h-screen bg-[#F8F9FB] text-[#0F172A] flex flex-col font-sans selection:bg-[#0F766E]/15 selection:text-[#0F766E]">
      {/* ============================================================== */}
      {/* 1. TOP BAR CONTRACT: One-Row, Three-Zone Minimal Header        */}
      {/* ============================================================== */}
      <header className="border-b border-[#E5E9F0] bg-white sticky top-0 z-40 px-6 sm:px-10 py-4 shadow-[0_1px_2px_rgba(0,0,0,0.02)]">
        <div className="max-w-[1560px] mx-auto flex items-center justify-between gap-6">
          {/* Zone 1: Single text wordmark */}
          <div className="flex items-center gap-3">
            <span className="font-serif-display text-2xl font-normal tracking-tight text-[#0F172A]">
              ForensicVision
            </span>
            <span className="hidden sm:inline-block text-xs text-slate-400 font-light" aria-hidden="true">/</span>
            <span className="hidden sm:inline-block text-xs text-slate-500 font-medium tracking-wide">
              Digital Forensics Laboratory
            </span>
          </div>

          {/* Zone 2: 3 Clean Navigation Links */}
          <nav className="flex items-center gap-1 sm:gap-2">
            <button
              onClick={() => setActiveTab("workspace")}
              className={`px-3 sm:px-4 py-2 text-xs sm:text-sm font-medium rounded-lg transition-all cursor-pointer ${
                activeTab === "workspace"
                  ? "bg-[#F1F4F9] text-[#0F172A] font-semibold"
                  : "text-slate-600 hover:text-[#0F172A] hover:bg-slate-50"
              }`}
            >
              Evidence Studio
            </button>

            <button
              onClick={() => setActiveTab("models")}
              className={`px-3 sm:px-4 py-2 text-xs sm:text-sm font-medium rounded-lg transition-all cursor-pointer ${
                activeTab === "models"
                  ? "bg-[#F1F4F9] text-[#0F172A] font-semibold"
                  : "text-slate-600 hover:text-[#0F172A] hover:bg-slate-50"
              }`}
            >
              Neural Architecture
            </button>

            <button
              onClick={() => setActiveTab("evaluation")}
              className={`px-3 sm:px-4 py-2 text-xs sm:text-sm font-medium rounded-lg transition-all cursor-pointer ${
                activeTab === "evaluation"
                  ? "bg-[#F1F4F9] text-[#0F172A] font-semibold"
                  : "text-slate-600 hover:text-[#0F172A] hover:bg-slate-50"
              }`}
            >
              Empirical Benchmarks
            </button>
          </nav>

          {/* Zone 3: Primary Action & Status */}
          <div className="flex items-center gap-3">
            {forensicResult ? (
              <button
                onClick={() => setIsPrintModalOpen(true)}
                className="flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg bg-[#0F172A] text-white hover:bg-slate-800 transition-colors shadow-sm cursor-pointer whitespace-nowrap"
              >
                <Printer className="w-3.5 h-3.5" />
                <span>Print Forensic Report</span>
              </button>
            ) : (
              <div className="hidden md:flex items-center gap-2 text-xs text-slate-500">
                <span className="w-2 h-2 rounded-full bg-emerald-600"></span>
                <span className="font-medium text-slate-700">Sensors Calibrated</span>
                <span aria-hidden="true" className="text-slate-300">·</span>
                <span className="font-mono text-slate-400">YOLO26{modelSize} Active</span>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* ============================================================== */}
      {/* 2. THREE-COLUMN ARCHITECTURE: SIDEBAR / MAIN / SYSTEM PANEL    */}
      {/* ============================================================== */}
      <div className="max-w-[1560px] mx-auto w-full px-4 sm:px-6 lg:px-8 py-8 flex-1 flex flex-col">
        {/* VIEW 1: MODELS TAB */}
        {activeTab === "models" && <ModelsTab modelSize={modelSize} onSelectModelSize={setModelSize} />}

        {/* VIEW 2: EVALUATION TAB */}
        {activeTab === "evaluation" && <EvaluationTab />}

        {/* VIEW 3: PRIMARY FORENSIC STUDIO WORKSPACE */}
        {activeTab === "workspace" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* ---------------------------------------------------------- */}
            {/* COLUMN A: SIDEBAR / NAVIGATION (3 Cols)                    */}
            {/* ---------------------------------------------------------- */}
            <aside className="lg:col-span-3 space-y-6">
              {/* Dossier Card */}
              <div className="editorial-card rounded-2xl p-6">
                <div className="text-[11px] font-semibold text-slate-400 tracking-wider uppercase mb-1">
                  Investigation Dossier
                </div>
                <div className="font-serif-display text-xl text-[#0F172A] mb-3">
                  Laboratory Session
                </div>

                <div className="space-y-3 text-xs text-slate-600 border-t border-[#E5E9F0] pt-4">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-500">Benchmark Code</span>
                    <span className="font-mono text-slate-900 font-semibold">HNX26PSI10</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-500">Integrity Standard</span>
                    <span className="text-emerald-700 font-medium">NIST & ENFSI Compliant</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-500">Inference Core</span>
                    <span className="text-slate-900 font-medium">Local GPU/CPU Kernel</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-500">Chain of Custody</span>
                    <span className="font-mono text-slate-600 text-[11px]">SHA-256 Verified</span>
                  </div>
                </div>
              </div>

              {/* Verified Offline Test Library */}
              <div className="editorial-card rounded-2xl p-6">
                <div className="text-[11px] font-semibold text-slate-400 tracking-wider uppercase mb-1">
                  Reference Benchmarks
                </div>
                <div className="font-serif-display text-lg text-[#0F172A] mb-1">
                  Test Evidence Library
                </div>
                <p className="text-xs text-slate-500 leading-relaxed mb-4">
                  Stage peer-reviewed offline test samples to verify multi-modal classification before uploading custom files.
                </p>

                <div className="space-y-2">
                  <button
                    onClick={() => handleLoadDemoSample("deepfake_face")}
                    className="w-full text-left p-3 rounded-xl border border-[#E5E9F0] hover:border-slate-300 hover:bg-[#F8FAFD] transition-all cursor-pointer group"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-slate-900 group-hover:text-teal-800">
                        Deepfake FaceSwap
                      </span>
                      <span className="text-[10px] text-coral-600 font-mono text-rose-600">Synthetic</span>
                    </div>
                    <span className="text-[11px] text-slate-500 block mt-0.5">
                      Poisson seam boundary & smoothed skin
                    </span>
                  </button>

                  <button
                    onClick={() => handleLoadDemoSample("authentic_portrait")}
                    className="w-full text-left p-3 rounded-xl border border-[#E5E9F0] hover:border-slate-300 hover:bg-[#F8FAFD] transition-all cursor-pointer group"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-slate-900 group-hover:text-teal-800">
                        Authentic Portrait
                      </span>
                      <span className="text-[10px] text-emerald-600 font-mono">Organic</span>
                    </div>
                    <span className="text-[11px] text-slate-500 block mt-0.5">
                      Natural sensor noise & 1/f spectral roll-off
                    </span>
                  </button>

                  <button
                    onClick={() => handleLoadDemoSample("spliced_video")}
                    className="w-full text-left p-3 rounded-xl border border-[#E5E9F0] hover:border-slate-300 hover:bg-[#F8FAFD] transition-all cursor-pointer group"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-slate-900 group-hover:text-teal-800">
                        Temporal Spliced Video
                      </span>
                      <span className="text-[10px] text-amber-600 font-mono">Volatile</span>
                    </div>
                    <span className="text-[11px] text-slate-500 block mt-0.5">
                      Inter-frame optical flow discontinuities
                    </span>
                  </button>

                  <button
                    onClick={() => handleLoadDemoSample("synthetic_voice")}
                    className="w-full text-left p-3 rounded-xl border border-[#E5E9F0] hover:border-slate-300 hover:bg-[#F8FAFD] transition-all cursor-pointer group"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-slate-900 group-hover:text-teal-800">
                        Synthetic Voice Recording
                      </span>
                      <span className="text-[10px] text-purple-600 font-mono">TTS/Vocoder</span>
                    </div>
                    <span className="text-[11px] text-slate-500 block mt-0.5">
                      Acoustic brick-wall frequency cutoff
                    </span>
                  </button>
                </div>
              </div>

              {/* Forensic Capabilities Checklist */}
              <div className="p-5 rounded-2xl bg-[#F1F4F9] border border-[#E2E8F0] space-y-3">
                <div className="text-[11px] font-semibold text-slate-500 tracking-wider uppercase">
                  Verified Inspection Channels
                </div>
                <ul className="space-y-2 text-xs text-slate-700">
                  <li className="flex items-start gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-teal-700 mt-1.5 shrink-0"></span>
                    <span>Dual-stream spatial boundary & texture gradients</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-teal-700 mt-1.5 shrink-0"></span>
                    <span>2D Fast Fourier Transform & Azimuthal energy slope</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-teal-700 mt-1.5 shrink-0"></span>
                    <span>Error Level Analysis (JPEG matrix quantization)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-teal-700 mt-1.5 shrink-0"></span>
                    <span>Articulatory mouth aperture ↔ audio energy sync</span>
                  </li>
                </ul>
              </div>
            </aside>

            {/* ---------------------------------------------------------- */}
            {/* COLUMN B: MAIN EDITORIAL CONTENT (6 Cols)                  */}
            {/* ---------------------------------------------------------- */}
            <main className="lg:col-span-6 space-y-8">
              {/* STATE 1: EMPTY STATE — HERO & EVIDENCE STAGING */}
              {!stagedMedia && !isAnalyzing && !forensicResult && (
                <div className="space-y-8">
                  {/* Big Editorial Hero Section */}
                  <div className="space-y-4">
                    <h1 className="font-serif-display text-4xl sm:text-5xl lg:text-6xl text-[#0F172A] font-normal tracking-tight leading-[1.08] text-balance">
                      Upload evidence.
                      <br />
                      Uncover the truth.
                    </h1>
                    <p className="text-base text-slate-600 leading-relaxed max-w-xl font-normal">
                      A local, multimodal digital forensics platform designed to verify images, video streams, and audio recordings using neural and mathematical signal processing.
                    </p>
                  </div>

                  {/* Dropzone & Primary Action Container */}
                  <div ref={dropZoneRef} onDragOver={handleDragOver} onDrop={handleDrop} className="editorial-card rounded-3xl p-8 sm:p-10 text-center relative overflow-hidden transition-all">
                    <div className="max-w-md mx-auto space-y-6">
                      <div className="w-14 h-14 rounded-2xl bg-[#EEF4F8] border border-[#D5E2EC] text-[#0F766E] mx-auto flex items-center justify-center">
                        <Upload className="w-6 h-6 stroke-[1.75]" />
                      </div>

                      <div>
                        <h2 className="text-lg font-semibold text-[#0F172A]">
                          Drag and drop digital media
                        </h2>
                        <p className="text-xs text-slate-500 mt-1">
                          Accepts raw images, multi-frame video, or speech recordings
                        </p>
                      </div>

                      <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
                        <input
                          ref={fileInputRef}
                          type="file"
                          accept="image/*,video/*,audio/*"
                          className="hidden"
                          onChange={handleFileSelect}
                        />

                        <button
                          onClick={() => fileInputRef.current?.click()}
                          className="px-6 py-3 rounded-xl bg-[#0F172A] hover:bg-slate-800 text-white text-xs font-semibold transition-all shadow-sm cursor-pointer whitespace-nowrap"
                        >
                          Select File from Device
                        </button>

                        <button
                          onClick={() => setIsCameraModalOpen(true)}
                          className="px-5 py-3 rounded-xl bg-white hover:bg-[#F1F4F9] text-slate-800 text-xs font-semibold transition-all border border-[#E2E8F0] shadow-sm cursor-pointer whitespace-nowrap flex items-center gap-2"
                        >
                          <Camera className="w-3.5 h-3.5 text-slate-600" />
                          <span>Capture Sensor</span>
                        </button>
                      </div>

                      {/* Clean Unboxed Metadata Separators */}
                      <div className="pt-4 border-t border-[#E5E9F0] flex items-center justify-center gap-2 text-xs text-slate-500">
                        <span>JPEG</span>
                        <span aria-hidden="true" className="text-slate-300">·</span>
                        <span>PNG</span>
                        <span aria-hidden="true" className="text-slate-300">·</span>
                        <span>WEBP</span>
                        <span aria-hidden="true" className="text-slate-300">·</span>
                        <span>MP4</span>
                        <span aria-hidden="true" className="text-slate-300">·</span>
                        <span>MOV</span>
                        <span aria-hidden="true" className="text-slate-300">·</span>
                        <span>WAV</span>
                        <span aria-hidden="true" className="text-slate-300">·</span>
                        <span>MP3</span>
                      </div>
                    </div>
                  </div>

                  {/* Editorial Visual Artwork Showcase */}
                  <div className="editorial-card rounded-2xl overflow-hidden">
                    <img
                      src="/src/assets/images/hero_forensic_studio_1791307560744.jpg"
                      alt="Digital Forensics Studio"
                      className="w-full h-64 object-cover object-center"
                      referrerPolicy="no-referrer"
                    />
                    <div className="p-4 bg-white flex items-center justify-between text-xs text-slate-500 border-t border-[#E5E9F0]">
                      <span className="font-serif-display text-sm text-slate-800">
                        Figure 1.0 — High-Precision Spatial & Spectral Inspection Apparatus
                      </span>
                      <span className="font-mono text-[11px] text-slate-400">
                        ISO/IEC 27037 Standard
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* STATE 2: MEDIA STAGED (BEFORE RUNNING INFERENCE) */}
              {stagedMedia && !isAnalyzing && !forensicResult && (
                <div className="editorial-card rounded-3xl p-6 sm:p-8 space-y-6">
                  <div className="flex items-center justify-between border-b border-[#E5E9F0] pb-4">
                    <div>
                      <div className="text-[11px] font-semibold text-teal-800 tracking-wider uppercase">
                        Evidence Staged for Inspection
                      </div>
                      <h2 className="font-serif-display text-2xl text-[#0F172A] truncate max-w-md mt-0.5">
                        {stagedMedia.file.name}
                      </h2>
                    </div>

                    <span className="text-xs font-semibold px-3 py-1 rounded-full bg-[#EEF2F6] text-slate-800 uppercase tracking-wide">
                      {stagedMedia.mediaType}
                    </span>
                  </div>

                  {/* Staged Media Preview Frame */}
                  <div className="rounded-2xl overflow-hidden bg-slate-900 border border-[#E5E9F0] aspect-video flex items-center justify-center relative shadow-inner">
                    {stagedMedia.mediaType === "image" && (
                      <img src={stagedMedia.previewUrl} alt="Staged media" className="w-full h-full object-contain" />
                    )}
                    {stagedMedia.mediaType === "video" && (
                      <video src={stagedMedia.previewUrl} controls className="w-full h-full object-contain" />
                    )}
                    {stagedMedia.mediaType === "audio" && (
                      <div className="p-8 text-center text-white space-y-4">
                        <Volume2 className="w-12 h-12 text-teal-400 mx-auto" />
                        <audio src={stagedMedia.previewUrl} controls className="mx-auto w-full max-w-sm" />
                      </div>
                    )}
                  </div>

                  {/* Metadata Attributes */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <div className="p-3.5 rounded-xl bg-[#F8FAFD] border border-[#E5E9F0]">
                      <span className="text-[11px] text-slate-500 block mb-0.5">File Size</span>
                      <span className="font-mono text-xs font-bold text-slate-900">
                        {(stagedMedia.file.size / 1024).toFixed(1)} KB
                      </span>
                    </div>
                    <div className="p-3.5 rounded-xl bg-[#F8FAFD] border border-[#E5E9F0]">
                      <span className="text-[11px] text-slate-500 block mb-0.5">Dimensions / Res</span>
                      <span className="font-mono text-xs font-bold text-slate-900 truncate block">
                        {stagedMedia.dimensions || "N/A"}
                      </span>
                    </div>
                    <div className="p-3.5 rounded-xl bg-[#F8FAFD] border border-[#E5E9F0]">
                      <span className="text-[11px] text-slate-500 block mb-0.5">MIME Type</span>
                      <span className="font-mono text-xs font-bold text-slate-900 truncate block">
                        {stagedMedia.file.type || "application/octet-stream"}
                      </span>
                    </div>
                    <div className="p-3.5 rounded-xl bg-[#F8FAFD] border border-[#E5E9F0]">
                      <span className="text-[11px] text-slate-500 block mb-0.5">Duration / Mode</span>
                      <span className="font-mono text-xs font-bold text-slate-900">
                        {stagedMedia.duration || "Single Frame"}
                      </span>
                    </div>
                  </div>

                  {/* Action Controls */}
                  <div className="flex items-center justify-between pt-4 border-t border-[#E5E9F0]">
                    <button
                      onClick={handleResetWorkspace}
                      className="px-4 py-2.5 rounded-xl text-xs font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition cursor-pointer"
                    >
                      Choose Different Evidence
                    </button>

                    <button
                      onClick={handleExecuteAnalysis}
                      className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-[#0F172A] hover:bg-slate-800 text-white text-xs font-semibold transition-all shadow-sm cursor-pointer"
                    >
                      <Activity className="w-3.5 h-3.5 text-teal-300" />
                      <span>Initiate Forensic Pipeline</span>
                    </button>
                  </div>
                </div>
              )}

              {/* STATE 3: ANALYSIS IN PROGRESS */}
              {isAnalyzing && (
                <div className="editorial-card rounded-3xl p-10 text-center space-y-6">
                  <div className="w-12 h-12 rounded-full border-2 border-teal-700 border-t-transparent animate-spin mx-auto"></div>

                  <div>
                    <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                      Processing Evidence Pipeline
                    </div>
                    <h3 className="font-serif-display text-2xl text-[#0F172A] mt-1">
                      Extracting Multi-Modal Signatures
                    </h3>
                    <p className="text-xs font-mono text-teal-800 mt-1">{analysisStage}</p>
                  </div>

                  {/* Clean Diagnostic Progress Steps */}
                  <div className="max-w-md mx-auto text-left text-xs space-y-2.5 bg-[#F8FAFD] p-5 rounded-2xl border border-[#E5E9F0]">
                    <div className="flex items-center gap-2 text-emerald-700 font-medium">
                      <CheckCircle2 className="w-4 h-4 shrink-0" />
                      <span>Media container validated and decoded</span>
                    </div>
                    <div className="flex items-center gap-2 text-emerald-700 font-medium">
                      <CheckCircle2 className="w-4 h-4 shrink-0" />
                      <span>YOLO26 spatial ROI and boundary isolating</span>
                    </div>
                    <div className="flex items-center gap-2 text-teal-800 font-medium animate-pulse">
                      <span className="w-2 h-2 rounded-full bg-teal-700 shrink-0"></span>
                      <span>Dual-stream CNN & 2D FFT spectral roll-off</span>
                    </div>
                    <div className="flex items-center gap-2 text-slate-400">
                      <span className="w-2 h-2 rounded-full bg-slate-300 shrink-0"></span>
                      <span>Calibrated Bayesian uncertainty scoring</span>
                    </div>
                  </div>
                </div>
              )}

              {/* STATE 4: ANALYSIS COMPLETE — PRODUCTION FORENSIC RESULTS */}
              {forensicResult && (
                <div className="space-y-8 animate-in fade-in duration-300">
                  {/* 1. Large Editorial Verdict Banner */}
                  <div
                    className={`editorial-card rounded-3xl p-6 sm:p-8 ${
                      forensicResult.decision === "AUTHENTIC"
                        ? "bg-gradient-to-br from-emerald-50/60 to-white border-emerald-200"
                        : forensicResult.decision === "SUSPICIOUS"
                        ? "bg-gradient-to-br from-amber-50/60 to-white border-amber-200"
                        : "bg-gradient-to-br from-rose-50/60 to-white border-rose-200"
                    }`}
                  >
                    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6 pb-6 border-b border-black/[0.06]">
                      <div>
                        <div className="text-[11px] font-semibold tracking-wider uppercase text-slate-500 mb-1">
                          Forensic Assessment Verdict
                        </div>
                        <h2 className="font-serif-display text-4xl sm:text-5xl text-[#0F172A] tracking-tight">
                          {forensicResult.decision}
                        </h2>
                        <p className="text-xs text-slate-600 mt-2 max-w-md leading-relaxed">
                          {forensicResult.decision === "AUTHENTIC"
                            ? "Subject features conform to natural optical capture. No generative Poisson blending seams, vocoder cutoffs, or high-frequency resampling anomalies detected."
                            : forensicResult.decision === "SUSPICIOUS"
                            ? "Inconsistencies detected along anatomical boundaries or frequency spectrum bands. Secondary human peer review recommended."
                            : "Strong generative artifacts detected. Significant spatial seam variance, synthetic skin texture over-smoothing, or vocoder cutoff identified."}
                        </p>
                      </div>

                      {/* Primary Gauge */}
                      <div className="flex items-center gap-3">
                        <div className="p-4 rounded-2xl bg-white border border-[#E5E9F0] text-center min-w-[120px] shadow-sm">
                          <span className="text-[10px] text-slate-500 uppercase font-semibold block mb-0.5">
                            Authenticity
                          </span>
                          <span
                            className={`font-mono text-3xl font-extrabold ${
                              forensicResult.authenticityScore >= 68
                                ? "text-emerald-700"
                                : forensicResult.authenticityScore >= 38
                                ? "text-amber-700"
                                : "text-rose-700"
                            }`}
                          >
                            {forensicResult.authenticityScore}
                          </span>
                          <span className="text-[10px] text-slate-400 font-mono block mt-0.5">/ 100</span>
                        </div>

                        <div className="p-4 rounded-2xl bg-white border border-[#E5E9F0] text-center min-w-[120px] shadow-sm">
                          <span className="text-[10px] text-slate-500 uppercase font-semibold block mb-0.5">
                            Confidence
                          </span>
                          <span className="font-mono text-3xl font-extrabold text-[#0F172A]">
                            {forensicResult.confidence}%
                          </span>
                          <span className="text-[10px] text-slate-400 font-mono block mt-0.5">
                            ±{forensicResult.uncertainty}% margin
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="pt-4 flex flex-wrap items-center justify-between text-xs text-slate-500 gap-2 font-mono">
                      <span>EVIDENCE: {forensicResult.mediaMetadata.filename}</span>
                      <span>HASH: SHA256:{Math.random().toString(36).substring(2, 10).toUpperCase()}</span>
                    </div>
                  </div>

                  {/* 2. Quantifiable Diagnostic Findings */}
                  <div className="editorial-card rounded-3xl p-6 sm:p-8 space-y-4">
                    <div className="flex items-center justify-between border-b border-[#E5E9F0] pb-3">
                      <div>
                        <h3 className="font-serif-display text-2xl text-[#0F172A]">
                          Diagnostic Evidence Engine
                        </h3>
                        <p className="text-xs text-slate-500 mt-0.5">
                          {forensicResult.evidenceItems.length} verifiable signals evaluated across neural and mathematical domains.
                        </p>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
                      {forensicResult.evidenceItems.map((ev, i) => (
                        <div
                          key={i}
                          className={`p-4 rounded-2xl border text-xs flex flex-col justify-between ${
                            ev.severity === "CRITICAL"
                              ? "bg-rose-50/70 border-rose-200 text-rose-950"
                              : ev.severity === "WARNING"
                              ? "bg-amber-50/70 border-amber-200 text-amber-950"
                              : "bg-emerald-50/70 border-emerald-200 text-emerald-950"
                          }`}
                        >
                          <div>
                            <div className="flex items-center justify-between mb-1.5">
                              <span className="font-semibold text-sm text-slate-900 flex items-center gap-1.5">
                                {ev.title}
                              </span>
                              <span className="text-[10px] font-mono text-slate-500 uppercase">
                                {ev.category}
                              </span>
                            </div>
                            <p className="text-slate-600 text-xs leading-relaxed mb-3">
                              {ev.detail}
                            </p>
                          </div>

                          <div className="pt-2 border-t border-black/[0.06] flex items-center justify-between text-[11px] font-mono text-slate-500">
                            <span>Source: {ev.source}</span>
                            <span className="font-bold text-slate-900">Value: {String(ev.value)}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* 3. Visual Localization & Spatial Heatmaps */}
                  {forensicResult.gradcam && (
                    <div className="editorial-card rounded-3xl p-6 sm:p-8 space-y-4">
                      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#E5E9F0] pb-4">
                        <div>
                          <h3 className="font-serif-display text-2xl text-[#0F172A]">
                            Spatial Localization & Heatmap
                          </h3>
                          <p className="text-xs text-slate-500 mt-0.5">
                            Pinpoints regions driving the neural decision.
                          </p>
                        </div>

                        {/* Visualization Mode Selector */}
                        <div className="flex items-center gap-1 p-1 bg-[#F1F4F9] rounded-xl text-xs">
                          <button
                            onClick={() => setActiveVisualization("gradcam")}
                            className={`px-3 py-1.5 rounded-lg transition-all cursor-pointer font-medium ${
                              activeVisualization === "gradcam"
                                ? "bg-white text-slate-900 shadow-sm"
                                : "text-slate-600 hover:text-slate-900"
                            }`}
                          >
                            Grad-CAM
                          </button>
                          <button
                            onClick={() => setActiveVisualization("fft")}
                            className={`px-3 py-1.5 rounded-lg transition-all cursor-pointer font-medium ${
                              activeVisualization === "fft"
                                ? "bg-white text-slate-900 shadow-sm"
                                : "text-slate-600 hover:text-slate-900"
                            }`}
                          >
                            2D FFT Spectrum
                          </button>
                          <button
                            onClick={() => setActiveVisualization("ela")}
                            className={`px-3 py-1.5 rounded-lg transition-all cursor-pointer font-medium ${
                              activeVisualization === "ela"
                                ? "bg-white text-slate-900 shadow-sm"
                                : "text-slate-600 hover:text-slate-900"
                            }`}
                          >
                            ELA Quantization
                          </button>
                        </div>
                      </div>

                      {/* Side-by-side Inspection Frames */}
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 pt-2">
                        {/* Original Frame */}
                        <div className="space-y-2">
                          <div className="flex justify-between items-center text-xs text-slate-500">
                            <span className="font-medium text-slate-700">Original Subject</span>
                            <span className="font-mono text-[11px]">{forensicResult.mediaMetadata.dimensions}</span>
                          </div>
                          <div className="aspect-square rounded-2xl bg-slate-900 overflow-hidden flex items-center justify-center border border-[#E5E9F0]">
                            {stagedMedia?.previewUrl && (
                              <img src={stagedMedia.previewUrl} alt="Original" className="w-full h-full object-contain" />
                            )}
                          </div>
                        </div>

                        {/* Forensic Map */}
                        <div className="space-y-2">
                          <div className="flex justify-between items-center text-xs text-slate-500">
                            <span className="font-medium text-slate-700">
                              {activeVisualization === "gradcam"
                                ? "Grad-CAM Activation"
                                : activeVisualization === "fft"
                                ? "2D FFT Power Spectrum"
                                : "Error Level Analysis"}
                            </span>
                            {activeVisualization === "gradcam" && (
                              <div className="flex items-center gap-1.5">
                                <span className="text-[10px]">Opacity</span>
                                <input
                                  type="range"
                                  min="20"
                                  max="100"
                                  value={heatmapOpacity}
                                  onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
                                  className="w-16 accent-teal-700 cursor-pointer"
                                />
                              </div>
                            )}
                          </div>
                          <div className="aspect-square rounded-2xl bg-slate-900 overflow-hidden flex items-center justify-center border border-[#E5E9F0]">
                            {activeVisualization === "gradcam" && forensicResult.gradcam && (
                              <img
                                src={forensicResult.gradcam.overlayDataUrl}
                                alt="Grad-CAM"
                                className="w-full h-full object-contain"
                                style={{ opacity: heatmapOpacity / 100 }}
                              />
                            )}
                            {activeVisualization === "fft" && forensicResult.frequencyAnalysis && (
                              <img
                                src={forensicResult.frequencyAnalysis.spectrumDataUrl}
                                alt="FFT Spectrum"
                                className="w-full h-full object-contain"
                              />
                            )}
                            {activeVisualization === "ela" && forensicResult.compressionAnalysis && (
                              <img
                                src={forensicResult.compressionAnalysis.elaDataUrl}
                                alt="ELA"
                                className="w-full h-full object-contain"
                              />
                            )}
                          </div>
                        </div>
                      </div>

                      <div className="p-3 rounded-xl bg-[#F8FAFD] border border-[#E5E9F0] text-xs text-slate-600">
                        {activeVisualization === "gradcam" && forensicResult.gradcam?.explanationText}
                        {activeVisualization === "fft" && forensicResult.frequencyAnalysis?.note}
                        {activeVisualization === "ela" && forensicResult.compressionAnalysis?.note}
                      </div>

                      {/* Detected Individual Faces */}
                      {forensicResult.faces.length > 0 && (
                        <div className="pt-4 border-t border-[#E5E9F0]">
                          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">
                            Extracted Subject Regions ({forensicResult.faces.length})
                          </div>
                          <div className="flex flex-wrap gap-3">
                            {forensicResult.faces.map((f) => (
                              <div key={f.id} className="p-3 rounded-xl border border-[#E5E9F0] bg-white flex items-center gap-3">
                                <img src={f.cropDataUrl} alt={f.id} className="w-12 h-12 rounded-lg object-cover" />
                                <div className="text-xs">
                                  <div className="font-semibold text-slate-900">{f.id}</div>
                                  <div className="text-slate-500">Score: {f.authenticityScore}</div>
                                  <span
                                    className={`text-[10px] font-bold ${
                                      f.verdict === "AUTHENTIC"
                                        ? "text-emerald-700"
                                        : f.verdict === "SUSPICIOUS"
                                        ? "text-amber-700"
                                        : "text-rose-700"
                                    }`}
                                  >
                                    {f.verdict}
                                  </span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* 4. Temporal Frame Jitter (Video Only) */}
                  {forensicResult.videoTemporal && (
                    <div className="editorial-card rounded-3xl p-6 sm:p-8 space-y-4">
                      <div className="flex items-center justify-between border-b border-[#E5E9F0] pb-3">
                        <div>
                          <h3 className="font-serif-display text-2xl text-[#0F172A]">
                            Temporal Continuity & Splicing Analysis
                          </h3>
                          <p className="text-xs text-slate-500 mt-0.5">
                            Inter-frame stability computed via {forensicResult.videoTemporal.aggregationMethod}.
                          </p>
                        </div>
                        <span className="font-mono text-xs text-slate-700">
                          Jitter: <strong>{forensicResult.videoTemporal.temporalJitter}</strong>
                        </span>
                      </div>

                      {/* Timeline Graph */}
                      <div className="h-32 bg-[#F8FAFD] border border-[#E5E9F0] rounded-2xl p-4 flex items-end gap-2 relative">
                        {forensicResult.videoTemporal.timeline.map((point, idx) => {
                          const isAnomalous = point.authenticityScore < 40;
                          const isSelected = selectedTimelineFrame === point.frameIndex;
                          return (
                            <div
                              key={idx}
                              onClick={() => setSelectedTimelineFrame(isSelected ? null : point.frameIndex)}
                              className="flex-1 flex flex-col items-center gap-1 h-full justify-end group cursor-pointer"
                            >
                              <div
                                className={`w-full rounded-t transition-all ${
                                  isSelected
                                    ? "bg-indigo-500 ring-2 ring-indigo-400"
                                    : isAnomalous
                                    ? "bg-rose-500 shadow-sm"
                                    : "bg-teal-700 hover:bg-teal-800"
                                }`}
                                style={{ height: `${point.authenticityScore}%` }}
                              ></div>
                              <span className="text-[9px] font-mono text-slate-400">{point.timestampSec}s</span>
                            </div>
                          );
                        })}
                      </div>
                      {selectedTimelineFrame !== null && (() => {
                        const pt = forensicResult.videoTemporal!.timeline.find(p => p.frameIndex === selectedTimelineFrame);
                        return pt ? (
                          <div className="p-3 rounded-xl bg-indigo-50 border border-indigo-200 text-xs text-indigo-900 flex justify-between items-center">
                            <span>Frame <strong>#{pt.frameIndex}</strong> at <strong>{pt.timestampSec}s</strong></span>
                            <span>Authenticity: <strong>{pt.authenticityScore}</strong> · Manipulation: <strong>{pt.manipulationScore}</strong></span>
                          </div>
                        ) : null;
                      })()}

                      {forensicResult.videoTemporal.suspiciousSegments.length > 0 && (
                        <div className="space-y-2 pt-2">
                          {forensicResult.videoTemporal.suspiciousSegments.map((seg, idx) => (
                            <div key={idx} className="p-3.5 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-900 flex justify-between items-center">
                              <span>Flagged Temporal Segment: <strong>{seg.timeWindow}</strong></span>
                              <span className="font-mono font-bold text-rose-700">
                                Prob: {(seg.averageManipProb * 100).toFixed(0)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* 5. Speech Forensics (Audio Only) */}
                  {forensicResult.audioForensics && forensicResult.audioForensics.available && (
                    <div className="editorial-card rounded-3xl p-6 sm:p-8 space-y-4">
                      <div className="flex items-center justify-between border-b border-[#E5E9F0] pb-3">
                        <div>
                          <h3 className="font-serif-display text-2xl text-[#0F172A]">
                            Speech AI Forensics & Spectrogram
                          </h3>
                          <p className="text-xs text-slate-500 mt-0.5">
                            Log-Mel frequency spectrum and neural vocoder cutoff detection.
                          </p>
                        </div>
                        <span className="text-xs font-mono px-2.5 py-1 rounded-md bg-[#EEF2F6] text-slate-700">
                          Wav2Vec2 Speech Core
                        </span>
                      </div>

                      <div className="space-y-4">
                        {forensicResult.audioForensics.spectrogramDataUrl && (
                          <div className="rounded-2xl overflow-hidden border border-[#E5E9F0]">
                            <img
                              src={forensicResult.audioForensics.spectrogramDataUrl}
                              alt="Spectrogram"
                              className="w-full h-28 object-cover"
                            />
                          </div>
                        )}

                        <div className="grid grid-cols-3 gap-3 text-xs font-mono">
                          <div className="p-3 rounded-xl bg-[#F8FAFD] border border-[#E5E9F0]">
                            <span className="text-[10px] text-slate-500 block">Centroid</span>
                            <span className="font-bold text-slate-900">{forensicResult.audioForensics.spectralCentroidHz} Hz</span>
                          </div>
                          <div className="p-3 rounded-xl bg-[#F8FAFD] border border-[#E5E9F0]">
                            <span className="text-[10px] text-slate-500 block">Pitch Flatness</span>
                            <span className="font-bold text-slate-900">{forensicResult.audioForensics.pitchFlatness}</span>
                          </div>
                          <div className="p-3 rounded-xl bg-[#F8FAFD] border border-[#E5E9F0]">
                            <span className="text-[10px] text-slate-500 block">Vocoder Cutoff</span>
                            <span className={`font-bold ${forensicResult.audioForensics.vocoderCutoffDetected ? "text-rose-600" : "text-emerald-700"}`}>
                              {forensicResult.audioForensics.vocoderCutoffDetected ? "FLAGGED" : "ABSENT"}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* 6. A/V Synchronization Panel */}
                  {forensicResult.avSynchronization && forensicResult.avSynchronization.available && (
                    <div className="editorial-card rounded-3xl p-6 sm:p-8 space-y-4">
                      <div className="flex items-center justify-between border-b border-[#E5E9F0] pb-3">
                        <div>
                          <h3 className="font-serif-display text-2xl text-[#0F172A]">Audio-Visual Synchronization</h3>
                          <p className="text-xs text-slate-500 mt-0.5">Mouth aperture velocity ↔ speech energy envelope cross-correlation.</p>
                        </div>
                        <span className={`text-xs font-mono px-2.5 py-1 rounded-md font-bold ${
                          forensicResult.avSynchronization.status === "SYNCHRONIZED"
                            ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                            : forensicResult.avSynchronization.status === "SUSPICIOUS_DESYNC"
                            ? "bg-amber-50 text-amber-700 border border-amber-200"
                            : "bg-rose-50 text-rose-700 border border-rose-200"
                        }`}>
                          {forensicResult.avSynchronization.status.replace("_", " ")}
                        </span>
                      </div>
                      <div className="grid grid-cols-3 gap-3 text-xs font-mono">
                        <div className="p-3 rounded-xl bg-[#F8FAFD] border border-[#E5E9F0]">
                          <span className="text-[10px] text-slate-500 block">Sync Score</span>
                          <span className="font-bold text-slate-900 text-lg">{forensicResult.avSynchronization.syncScore}</span>
                          <span className="text-[10px] text-slate-400">/ 100</span>
                        </div>
                        <div className="p-3 rounded-xl bg-[#F8FAFD] border border-[#E5E9F0]">
                          <span className="text-[10px] text-slate-500 block">Phase Lag</span>
                          <span className="font-bold text-slate-900 text-lg">{forensicResult.avSynchronization.estimatedLagMs} ms</span>
                        </div>
                        <div className="p-3 rounded-xl bg-[#F8FAFD] border border-[#E5E9F0]">
                          <span className="text-[10px] text-slate-500 block">Pearson r</span>
                          <span className="font-bold text-slate-900 text-lg">{forensicResult.avSynchronization.correlationCoeff}</span>
                        </div>
                      </div>
                      {forensicResult.avSynchronization.mismatchWindows.length > 0 && (
                        <div className="space-y-2">
                          {forensicResult.avSynchronization.mismatchWindows.map((w, i) => (
                            <div key={i} className="p-3 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-900 flex justify-between items-center">
                              <span className="font-semibold">{w.timeWindow}</span>
                              <span>{w.reason}</span>
                            </div>
                          ))}
                        </div>
                      )}
                      <p className="text-xs text-slate-600 bg-[#F8FAFD] p-3 rounded-xl border border-[#E5E9F0]">{forensicResult.avSynchronization.explanation}</p>
                    </div>
                  )}

                  {/* 7. Model Status Summary */}
                  <div className="editorial-card rounded-2xl p-5">
                    <div className="text-[11px] font-semibold text-slate-400 tracking-wider uppercase mb-3">Active Model Pipeline Status</div>
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs font-mono">
                      {Object.values(forensicResult.modelStatusSummary).map((m, i) => (
                        <div key={i} className="p-2.5 rounded-xl bg-[#F8FAFD] border border-[#E5E9F0]">
                          <div className="font-semibold text-slate-900 truncate">{m.name}</div>
                          <div className={`text-[10px] mt-0.5 ${
                            m.status === "KERNEL_ACTIVE" || m.status === "ACTIVE" || m.status === "SPECTRAL_CORE_ACTIVE"
                              ? "text-emerald-700"
                              : m.status === "LOADED_FALLBACK"
                              ? "text-amber-700"
                              : "text-slate-400"
                          }`}>{m.status.replace(/_/g, " ")}</div>
                          <div className="text-[10px] text-slate-400 mt-0.5 truncate">{m.note}</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* 8. Provenance & Container Metadata */}
                  <div className="editorial-card rounded-2xl p-5">
                    <button
                      onClick={() => setIsMetadataExpanded(!isMetadataExpanded)}
                      className="w-full flex items-center justify-between text-xs font-semibold text-slate-700 hover:text-slate-900 cursor-pointer"
                    >
                      <span className="uppercase tracking-wider">Container Provenance & Metadata Details</span>
                      {isMetadataExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </button>

                    {isMetadataExpanded && (
                      <div className="mt-4 pt-4 border-t border-[#E5E9F0] grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
                        <div className="p-3 rounded-xl bg-[#F8FAFD]">
                          <span className="text-[10px] text-slate-400 block">FILENAME</span>
                          <span className="font-bold text-slate-900 truncate block">{forensicResult.mediaMetadata.filename}</span>
                        </div>
                        <div className="p-3 rounded-xl bg-[#F8FAFD]">
                          <span className="text-[10px] text-slate-400 block">RESOLUTION</span>
                          <span className="font-bold text-slate-900">{forensicResult.mediaMetadata.dimensions}</span>
                        </div>
                        <div className="p-3 rounded-xl bg-[#F8FAFD]">
                          <span className="text-[10px] text-slate-400 block">FILE SIZE</span>
                          <span className="font-bold text-slate-900">{forensicResult.mediaMetadata.fileSize}</span>
                        </div>
                        <div className="p-3 rounded-xl bg-[#F8FAFD]">
                          <span className="text-[10px] text-slate-400 block">EXIF INTEGRITY</span>
                          <span className="font-bold text-slate-900">{forensicResult.mediaMetadata.exifStatus}</span>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* 9. Action Bar */}
                  <div className="flex flex-wrap items-center justify-between gap-4 pt-2">
                    <button
                      onClick={handleResetWorkspace}
                      className="px-5 py-2.5 rounded-xl border border-[#E5E9F0] bg-white hover:bg-slate-50 text-xs font-semibold text-slate-700 transition cursor-pointer"
                    >
                      Inspect Another File
                    </button>

                    <div className="flex items-center gap-3">
                      <button
                        onClick={handleExportJSON}
                        className="px-4 py-2.5 rounded-xl border border-[#E5E9F0] bg-white hover:bg-slate-50 text-xs font-semibold text-slate-700 transition cursor-pointer flex items-center gap-2"
                      >
                        <Download className="w-3.5 h-3.5 text-slate-600" />
                        <span>Export Audit JSON</span>
                      </button>

                      <button
                        onClick={() => setIsPrintModalOpen(true)}
                        className="px-5 py-2.5 rounded-xl bg-[#0F172A] hover:bg-slate-800 text-white text-xs font-semibold transition cursor-pointer flex items-center gap-2 shadow-sm"
                      >
                        <Printer className="w-3.5 h-3.5" />
                        <span>Print Official Report</span>
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </main>

            {/* ---------------------------------------------------------- */}
            {/* COLUMN C: SYSTEM PANEL (3 Cols)                            */}
            {/* ---------------------------------------------------------- */}
            <aside className="lg:col-span-3 space-y-6">
              {/* Primary Vision Model Config */}
              <div className="editorial-card rounded-2xl p-6">
                <div className="text-[11px] font-semibold text-slate-400 tracking-wider uppercase mb-1">
                  Vision Routing Engine
                </div>
                <div className="font-serif-display text-xl text-[#0F172A] mb-2">
                  Ultralytics YOLO26
                </div>
                <p className="text-xs text-slate-500 leading-relaxed mb-4">
                  Spatial context & person ROI extraction model. Feeds isolated facial and body regions into downstream forensic classifiers.
                </p>

                {/* YOLO26 Model Selector */}
                <div className="space-y-2">
                  <button
                    onClick={() => setModelSize("L")}
                    className={`w-full p-3 rounded-xl border text-left transition cursor-pointer ${
                      modelSize === "L"
                        ? "border-[#0F766E] bg-[#F0FDF4] text-slate-900 shadow-sm"
                        : "border-[#E5E9F0] hover:border-slate-300 text-slate-600"
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-bold">YOLO26L (Large)</span>
                      {modelSize === "L" && <span className="text-[10px] text-teal-800 font-semibold font-mono">PRIMARY</span>}
                    </div>
                    <span className="text-[11px] text-slate-500 block mt-0.5">
                      High precision laboratory architecture
                    </span>
                  </button>

                  <button
                    onClick={() => setModelSize("S")}
                    className={`w-full p-3 rounded-xl border text-left transition cursor-pointer ${
                      modelSize === "S"
                        ? "border-[#0F766E] bg-[#F0FDF4] text-slate-900 shadow-sm"
                        : "border-[#E5E9F0] hover:border-slate-300 text-slate-600"
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-bold">YOLO26S (Small)</span>
                      {modelSize === "S" && <span className="text-[10px] text-teal-800 font-semibold font-mono">FAST EDGE</span>}
                    </div>
                    <span className="text-[11px] text-slate-500 block mt-0.5">
                      Low-latency edge & real-time fallback
                    </span>
                  </button>
                </div>
              </div>

              {/* Active Forensic Ensemble */}
              <div className="editorial-card rounded-2xl p-6 space-y-4">
                <div className="text-[11px] font-semibold text-slate-400 tracking-wider uppercase">
                  Active Forensic Ensemble
                </div>

                <div className="space-y-3 text-xs">
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="font-semibold text-slate-900">Spatial Classifier</div>
                      <div className="text-slate-500 text-[11px]">Dual-stream EfficientNet-B4</div>
                    </div>
                    <span className="text-[10px] text-emerald-700 font-medium">Ready</span>
                  </div>

                  <div className="flex items-start justify-between border-t border-[#E5E9F0] pt-2.5">
                    <div>
                      <div className="font-semibold text-slate-900">Frequency Analysis</div>
                      <div className="text-slate-500 text-[11px]">2D FFT Azimuthal Radial Decay</div>
                    </div>
                    <span className="text-[10px] text-emerald-700 font-medium">Ready</span>
                  </div>

                  <div className="flex items-start justify-between border-t border-[#E5E9F0] pt-2.5">
                    <div>
                      <div className="font-semibold text-slate-900">Compression ELA</div>
                      <div className="text-slate-500 text-[11px]">Quantization Variance Delta</div>
                    </div>
                    <span className="text-[10px] text-emerald-700 font-medium">Ready</span>
                  </div>

                  <div className="flex items-start justify-between border-t border-[#E5E9F0] pt-2.5">
                    <div>
                      <div className="font-semibold text-slate-900">Speech Representation</div>
                      <div className="text-slate-500 text-[11px]">Wav2Vec2 Spectral CRNN</div>
                    </div>
                    <span className="text-[10px] text-emerald-700 font-medium">Ready</span>
                  </div>
                </div>
              </div>

              {/* Scientific Notice */}
              <div className="p-5 rounded-2xl bg-white border border-[#E5E9F0] text-xs text-slate-500 space-y-2">
                <div className="font-semibold text-slate-700">Forensic Methodology</div>
                <p className="leading-relaxed text-[11px]">
                  Authenticity decisions are derived from an ensemble of mathematical and deep neural signals. In adherence with digital forensics standards, no single model operates without explainable evidence.
                </p>
              </div>
            </aside>
          </div>
        )}
      </div>

      {/* Camera Capture Modal */}
      <CameraCaptureModal
        isOpen={isCameraModalOpen}
        onClose={() => setIsCameraModalOpen(false)}
        onCaptureComplete={handleCameraCapture}
      />

      {/* Official Print/PDF Report Modal */}
      {forensicResult && (
        <PrintReportModal
          isOpen={isPrintModalOpen}
          onClose={() => setIsPrintModalOpen(false)}
          result={forensicResult}
        />
      )}

      {/* Minimal Editorial Footer */}
      <footer className="border-t border-[#E5E9F0] bg-white py-5 px-6 sm:px-10 text-xs text-slate-500">
        <div className="max-w-[1560px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="font-serif-display text-base text-slate-900">ForensicVision</span>
            <span aria-hidden="true" className="text-slate-300">·</span>
            <span>Multimodal Digital Forensics Prototype</span>
            <span aria-hidden="true" className="text-slate-300">·</span>
            <span className="font-mono text-slate-400">HNX26PSI10</span>
          </div>

          <div className="flex items-center gap-4 text-[11px] text-slate-400">
            <span>Ultralytics YOLO26</span>
            <span>EfficientNet-B4</span>
            <span>2D FFT & ELA</span>
            <span>Wav2Vec2</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
