import React, { useState, useRef, useEffect } from "react";
import { Camera, Video, Square, RefreshCw, X, AlertCircle, Check, CircleDot } from "lucide-react";

interface CameraCaptureModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCaptureComplete: (file: File, previewUrl: string, mediaType: "image" | "video") => void;
}

export default function CameraCaptureModal({
  isOpen,
  onClose,
  onCaptureComplete
}: CameraCaptureModalProps) {
  const [mode, setMode] = useState<"photo" | "video">("photo");
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);

  // Captured preview state
  const [capturedPhotoUrl, setCapturedPhotoUrl] = useState<string | null>(null);
  const [capturedVideoUrl, setCapturedVideoUrl] = useState<string | null>(null);
  const [capturedBlob, setCapturedBlob] = useState<Blob | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const timerIntervalRef = useRef<number | null>(null);

  // Initialize camera stream when modal opens
  useEffect(() => {
    if (!isOpen) {
      cleanupStream();
      return;
    }

    startCamera();

    return () => {
      cleanupStream();
    };
  }, [isOpen, mode]);

  const startCamera = async () => {
    setCameraError(null);
    setCapturedPhotoUrl(null);
    setCapturedVideoUrl(null);
    setCapturedBlob(null);
    cleanupStream();

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("MediaDevices API is not supported in this browser environment.");
      }

      const constraints: MediaStreamConstraints = {
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: mode === "video"
      };

      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      setStream(mediaStream);

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.play().catch(() => {});
      }
    } catch (err: any) {
      console.error("Camera access error:", err);
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        setCameraError("Camera access permission was denied. Please allow camera access in your browser settings.");
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        setCameraError("No camera hardware sensor was detected on your system.");
      } else {
        setCameraError(err.message || "Unable to acquire camera media stream.");
      }
    }
  };

  const cleanupStream = () => {
    if (timerIntervalRef.current) {
      window.clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
    setIsRecording(false);
    setRecordingSeconds(0);
  };

  // Take high-resolution snapshot photo
  const takeSnapshot = () => {
    if (!videoRef.current) return;
    const v = videoRef.current;
    const w = v.videoWidth || 1280;
    const h = v.videoHeight || 720;

    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(v, 0, 0, w, h);
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        setCapturedBlob(blob);
        setCapturedPhotoUrl(url);
      },
      "image/jpeg",
      0.95
    );
  };

  // Start recording video clip
  const startRecording = () => {
    if (!stream) return;
    recordedChunksRef.current = [];
    setRecordingSeconds(0);

    try {
      const mimeTypes = ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm", "video/mp4"];
      let selectedMime = "";
      for (const m of mimeTypes) {
        if (MediaRecorder.isTypeSupported(m)) {
          selectedMime = m;
          break;
        }
      }

      const recorder = new MediaRecorder(stream, selectedMime ? { mimeType: selectedMime } : undefined);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          recordedChunksRef.current.push(e.data);
        }
      };

      recorder.onstop = () => {
        const videoBlob = new Blob(recordedChunksRef.current, {
          type: selectedMime || "video/webm"
        });
        const url = URL.createObjectURL(videoBlob);
        setCapturedBlob(videoBlob);
        setCapturedVideoUrl(url);
        setIsRecording(false);
        if (timerIntervalRef.current) {
          window.clearInterval(timerIntervalRef.current);
          timerIntervalRef.current = null;
        }
      };

      recorder.start(250);
      setIsRecording(true);

      timerIntervalRef.current = window.setInterval(() => {
        setRecordingSeconds((prev) => {
          if (prev >= 14) {
            stopRecording();
            return 15;
          }
          return prev + 1;
        });
      }, 1000);
    } catch (err: any) {
      console.error("MediaRecorder initiation error:", err);
      setCameraError("Failed to initiate video recorder: " + err.message);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
    }
  };

  const handleConfirmMedia = () => {
    if (!capturedBlob) return;
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);

    if (mode === "photo") {
      const file = new File([capturedBlob], `camera_sensor_capture_${timestamp}.jpg`, {
        type: "image/jpeg"
      });
      onCaptureComplete(file, capturedPhotoUrl!, "image");
    } else {
      const file = new File([capturedBlob], `camera_sensor_clip_${timestamp}.webm`, {
        type: capturedBlob.type || "video/webm"
      });
      onCaptureComplete(file, capturedVideoUrl!, "video");
    }
    onClose();
  };

  const handleRetake = () => {
    if (capturedPhotoUrl) URL.revokeObjectURL(capturedPhotoUrl);
    if (capturedVideoUrl) URL.revokeObjectURL(capturedVideoUrl);
    setCapturedPhotoUrl(null);
    setCapturedVideoUrl(null);
    setCapturedBlob(null);
    startCamera();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm p-4 animate-in fade-in duration-200">
      <div className="relative w-full max-w-2xl bg-white border border-[#E5E9F0] rounded-3xl shadow-2xl overflow-hidden flex flex-col">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-[#E5E9F0] flex items-center justify-between bg-white">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-[#EEF4F8] text-teal-800">
              <Camera className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-serif-display text-lg text-[#0F172A] leading-tight">
                Live Sensor Media Acquisition
              </h3>
              <p className="text-xs text-slate-500">Capture authenticated sensor feed directly into workspace</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Mode Selector */}
        {!capturedPhotoUrl && !capturedVideoUrl && (
          <div className="px-6 py-3 flex items-center justify-between border-b border-[#E5E9F0] bg-[#F8FAFD]">
            <div className="flex items-center gap-1 bg-[#EEF2F6] p-1 rounded-xl text-xs font-medium">
              <button
                onClick={() => {
                  setMode("photo");
                  handleRetake();
                }}
                disabled={isRecording}
                className={`px-3 py-1.5 rounded-lg transition-all cursor-pointer ${
                  mode === "photo"
                    ? "bg-white text-slate-900 font-semibold shadow-sm"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                Photo Still
              </button>

              <button
                onClick={() => {
                  setMode("video");
                  handleRetake();
                }}
                disabled={isRecording}
                className={`px-3 py-1.5 rounded-lg transition-all cursor-pointer ${
                  mode === "video"
                    ? "bg-white text-slate-900 font-semibold shadow-sm"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                Short Video Clip (15s)
              </button>
            </div>

            <div className="text-[11px] font-mono text-slate-500 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-600"></span>
              <span>720p Sensor Ready</span>
            </div>
          </div>
        )}

        {/* Viewfinder Viewport */}
        <div className="relative aspect-video w-full bg-slate-950 flex items-center justify-center overflow-hidden">
          {cameraError ? (
            <div className="p-8 text-center max-w-md bg-white m-4 rounded-2xl shadow-sm border border-[#E5E9F0]">
              <AlertCircle className="w-8 h-8 text-rose-600 mx-auto mb-2" />
              <div className="text-sm font-bold text-slate-900 mb-1">Sensor Initialization Failed</div>
              <p className="text-xs text-slate-500 mb-4">{cameraError}</p>
              <button
                onClick={startCamera}
                className="px-4 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-xs text-slate-800 font-medium transition cursor-pointer inline-flex items-center gap-2"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Retry Permission</span>
              </button>
            </div>
          ) : capturedPhotoUrl ? (
            <img src={capturedPhotoUrl} alt="Captured preview" className="w-full h-full object-contain" />
          ) : capturedVideoUrl ? (
            <video src={capturedVideoUrl} controls autoPlay loop className="w-full h-full object-contain" />
          ) : (
            <>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover -scale-x-100"
              />

              {/* Minimal calibration frame */}
              <div className="absolute inset-10 border border-white/20 rounded-2xl pointer-events-none flex flex-col justify-between p-3">
                <div className="flex justify-between items-center text-[10px] font-mono text-white/60">
                  <span>OPTICAL FRAME</span>
                  <span>16:9 RAW</span>
                </div>
                <div className="flex justify-between items-center text-[10px] font-mono text-white/60">
                  <span>SENSOR EXPOSURE</span>
                  <span>UNCOMPRESSED</span>
                </div>
              </div>

              {/* Recording indicator */}
              {isRecording && (
                <div className="absolute top-4 left-4 bg-white/95 text-rose-700 font-mono text-xs px-3 py-1.5 rounded-full flex items-center gap-2 shadow-sm border border-rose-200">
                  <span className="w-2 h-2 rounded-full bg-rose-600 animate-pulse"></span>
                  <span className="font-bold">REC</span>
                  <span>00:{recordingSeconds < 10 ? `0${recordingSeconds}` : recordingSeconds} / 00:15</span>
                </div>
              )}
            </>
          )}
        </div>

        {/* Modal Controls Bar */}
        <div className="p-5 border-t border-[#E5E9F0] bg-white flex items-center justify-between">
          {capturedPhotoUrl || capturedVideoUrl ? (
            <div className="w-full flex items-center justify-between">
              <button
                onClick={handleRetake}
                className="px-4 py-2 rounded-xl border border-[#E5E9F0] text-xs font-semibold text-slate-600 hover:text-slate-900 transition cursor-pointer"
              >
                Retake
              </button>

              <button
                onClick={handleConfirmMedia}
                className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-[#0F172A] hover:bg-slate-800 text-xs font-semibold text-white transition-all cursor-pointer shadow-sm"
              >
                <Check className="w-4 h-4" />
                <span>Stage for Forensic Analysis</span>
              </button>
            </div>
          ) : (
            <div className="w-full flex items-center justify-between">
              <span className="text-xs text-slate-500">
                {mode === "photo" ? "Align subject inside the frame" : "Up to 15s recorded with audio track"}
              </span>

              {mode === "photo" ? (
                <button
                  onClick={takeSnapshot}
                  disabled={!stream}
                  className="px-6 py-2.5 rounded-xl bg-[#0F172A] hover:bg-slate-800 disabled:opacity-40 text-xs font-semibold text-white transition cursor-pointer shadow-sm flex items-center gap-2"
                >
                  <Camera className="w-4 h-4" />
                  <span>Capture Photo</span>
                </button>
              ) : isRecording ? (
                <button
                  onClick={stopRecording}
                  className="px-6 py-2.5 rounded-xl bg-rose-700 hover:bg-rose-800 text-xs font-semibold text-white transition cursor-pointer shadow-sm flex items-center gap-2"
                >
                  <Square className="w-4 h-4" />
                  <span>Stop Recording</span>
                </button>
              ) : (
                <button
                  onClick={startRecording}
                  disabled={!stream}
                  className="px-6 py-2.5 rounded-xl bg-rose-700 hover:bg-rose-800 disabled:opacity-40 text-xs font-semibold text-white transition cursor-pointer shadow-sm flex items-center gap-2"
                >
                  <CircleDot className="w-4 h-4" />
                  <span>Start Clip Recording</span>
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
