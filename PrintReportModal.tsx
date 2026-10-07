import React from "react";
import { Printer, X, Download, ShieldCheck, ShieldAlert, AlertTriangle, Info } from "lucide-react";
import { ForensicResult } from "../lib/forensics.ts";

interface PrintReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  result: ForensicResult;
}

export default function PrintReportModal({ isOpen, onClose, result }: PrintReportModalProps) {
  if (!isOpen) return null;

  const caseId = `HNX26-CASE-${Math.random().toString(36).substring(2, 8).toUpperCase()}`;
  const now = new Date().toUTCString();

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="forensic-report relative w-full max-w-4xl bg-white text-slate-900 rounded-2xl shadow-2xl overflow-hidden flex flex-col my-8 print:m-0 print:shadow-none print:w-full border border-[#E5E9F0]">
        {/* Modal Controls (Hidden when printing) */}
        <div className="px-6 py-4 bg-white text-slate-900 flex items-center justify-between border-b border-[#E5E9F0] print:hidden">
          <div className="flex items-center gap-2 text-xs">
            <span className="font-serif-display text-lg font-bold text-slate-900">ForensicVision</span>
            <span className="text-slate-400">·</span>
            <span className="text-slate-500 font-medium">Official Laboratory Examination Report</span>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={handlePrint}
              className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-[#0F172A] hover:bg-slate-800 text-white text-xs font-semibold transition-all shadow-sm cursor-pointer"
            >
              <Printer className="w-3.5 h-3.5" />
              <span>Print / Save as PDF</span>
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Printable Report Body */}
        <div className="p-8 sm:p-12 space-y-8 font-sans text-xs text-slate-800 leading-normal bg-white">
          {/* Header */}
          <div className="border-b-2 border-slate-900 pb-5 flex flex-col sm:flex-row justify-between items-start sm:items-end gap-4">
            <div>
              <div className="text-[10px] font-mono tracking-widest text-slate-500 uppercase">
                Digital Media Authenticity & Forensic Inspection
              </div>
              <h1 className="text-2xl font-black text-slate-950 tracking-tight mt-1">
                FORENSIC EXAMINATION REPORT
              </h1>
              <p className="text-xs text-slate-600 font-mono mt-0.5">Problem Benchmark: HNX26PSI10</p>
            </div>

            <div className="text-left sm:text-right font-mono text-[11px] space-y-1">
              <div>
                <span className="text-slate-500">CASE FILE: </span>
                <span className="font-bold text-slate-900">{caseId}</span>
              </div>
              <div>
                <span className="text-slate-500">EXAMINATION DATE: </span>
                <span className="text-slate-700">{now}</span>
              </div>
              <div>
                <span className="text-slate-500">STATUS: </span>
                <span className="font-bold text-slate-900">AUDIT COMPLETE</span>
              </div>
            </div>
          </div>

          {/* Section 1: Evidence Metadata */}
          <div>
            <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-900 border-b border-slate-200 pb-1 mb-3">
              1. Item Under Examination (Chain of Custody)
            </h2>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 bg-slate-50 p-4 rounded border border-slate-200 font-mono text-[11px]">
              <div>
                <span className="text-slate-400 block">FILE IDENTIFIER:</span>
                <strong className="text-slate-900 truncate block">{result.mediaMetadata.filename}</strong>
              </div>
              <div>
                <span className="text-slate-400 block">MEDIA RESOLUTION:</span>
                <strong className="text-slate-900">{result.mediaMetadata.dimensions}</strong>
              </div>
              <div>
                <span className="text-slate-400 block">FILE SIZE:</span>
                <strong className="text-slate-900">{result.mediaMetadata.fileSize}</strong>
              </div>
              <div>
                <span className="text-slate-400 block">EXIF / PROVENANCE:</span>
                <strong className="text-slate-900">{result.mediaMetadata.exifStatus}</strong>
              </div>
            </div>
          </div>

          {/* Section 2: Verdict & Quantitative Indices */}
          <div>
            <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-900 border-b border-slate-200 pb-1 mb-3">
              2. Forensic Assessment Verdict
            </h2>
            <div className="p-5 border-2 rounded flex flex-col sm:flex-row justify-between items-center gap-6 border-slate-900 bg-slate-50/50">
              <div className="flex items-center gap-4">
                {result.decision === "AUTHENTIC" && <ShieldCheck className="w-12 h-12 text-emerald-600 flex-shrink-0" />}
                {result.decision === "SUSPICIOUS" && <AlertTriangle className="w-12 h-12 text-amber-600 flex-shrink-0" />}
                {result.decision === "LIKELY MANIPULATED" && <ShieldAlert className="w-12 h-12 text-rose-600 flex-shrink-0" />}
                {result.decision === "INCONCLUSIVE" && <Info className="w-12 h-12 text-indigo-600 flex-shrink-0" />}

                <div>
                  <div className="text-[10px] font-mono uppercase text-slate-500">EXAMINATION CONCLUSION</div>
                  <div className="text-2xl font-black tracking-wide text-slate-900">{result.decision}</div>
                  <div className="text-[11px] text-slate-600 mt-0.5 max-w-sm">
                    {result.decision === "AUTHENTIC"
                      ? "Features align with natural physical sensor capture without detected generative artifacts."
                      : result.decision === "SUSPICIOUS"
                      ? "Localized spatial or frequency anomalies detected; secondary human verification advised."
                      : "Definitive Poisson blending seams, synthetic skin over-smoothing, or vocoder frequency cutoffs identified."}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-3 text-center font-mono w-full sm:w-auto">
                <div className="p-3 bg-white border border-slate-300 rounded min-w-[85px]">
                  <span className="text-[9px] text-slate-400 block">AUTHENTICITY</span>
                  <span className="text-xl font-black text-slate-900">{result.authenticityScore}</span>
                  <span className="text-[9px] text-slate-400 block">/ 100</span>
                </div>
                <div className="p-3 bg-white border border-slate-300 rounded min-w-[85px]">
                  <span className="text-[9px] text-slate-400 block">CONFIDENCE</span>
                  <span className="text-xl font-black text-slate-900">{result.confidence}%</span>
                  <span className="text-[9px] text-slate-400 block">Calibrated</span>
                </div>
                <div className="p-3 bg-white border border-slate-300 rounded min-w-[85px]">
                  <span className="text-[9px] text-slate-400 block">UNCERTAINTY</span>
                  <span className="text-xl font-black text-slate-900">{result.uncertainty}%</span>
                  <span className="text-[9px] text-slate-400 block">Margin</span>
                </div>
              </div>
            </div>
          </div>

          {/* Section 3: Diagnostic Evidence Matrix */}
          <div>
            <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-900 border-b border-slate-200 pb-1 mb-3">
              3. Diagnostic Evidence Matrix ({result.evidenceItems.length} Verified Signals)
            </h2>
            <div className="border border-slate-200 rounded overflow-hidden">
              <table className="w-full text-left font-sans text-xs">
                <thead className="bg-slate-100 border-b border-slate-200 text-slate-700 font-mono text-[10px]">
                  <tr>
                    <th className="p-2.5">SIGNAL CLASSIFICATION</th>
                    <th className="p-2.5">DIAGNOSTIC EVIDENCE DETAIL</th>
                    <th className="p-2.5">SOURCE ENGINE</th>
                    <th className="p-2.5 text-right">MEASURED VALUE</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200">
                  {result.evidenceItems.map((ev, idx) => (
                    <tr key={idx} className="hover:bg-slate-50">
                      <td className="p-2.5 font-bold">
                        <span
                          className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-mono mr-1.5 ${
                            ev.severity === "CRITICAL"
                              ? "bg-rose-100 text-rose-800"
                              : ev.severity === "WARNING"
                              ? "bg-amber-100 text-amber-800"
                              : "bg-emerald-100 text-emerald-800"
                          }`}
                        >
                          {ev.severity}
                        </span>
                        {ev.title}
                      </td>
                      <td className="p-2.5 text-slate-600 text-[11px]">{ev.detail}</td>
                      <td className="p-2.5 font-mono text-[11px] text-slate-500">{ev.source}</td>
                      <td className="p-2.5 font-mono text-[11px] font-bold text-right text-slate-900">
                        {String(ev.value)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Section 4: Verified Laboratory Pipeline Details */}
          <div>
            <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-900 border-b border-slate-200 pb-1 mb-2">
              4. Verified Algorithmic Pipeline
            </h2>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[10px] font-mono text-slate-600">
              <div className="p-2 border border-slate-200 rounded">
                <span className="text-slate-400 block">PRIMARY VISION:</span>
                <strong>Ultralytics YOLO26</strong>
              </div>
              <div className="p-2 border border-slate-200 rounded">
                <span className="text-slate-400 block">FORENSIC CNN:</span>
                <strong>EfficientNet-B4 Dual-Stream</strong>
              </div>
              <div className="p-2 border border-slate-200 rounded">
                <span className="text-slate-400 block">SPEECH ENCODER:</span>
                <strong>Wav2Vec2 / Acoustic CRNN</strong>
              </div>
              <div className="p-2 border border-slate-200 rounded">
                <span className="text-slate-400 block">CROSS-MODAL A/V:</span>
                <strong>Optical Flow ↔ Energy Envelope</strong>
              </div>
            </div>
          </div>

          {/* Section 5: Legal & Scientific Disclaimer */}
          <div className="border-t border-slate-300 pt-4 text-[10px] text-slate-500 leading-relaxed">
            <strong>SCIENTIFIC & LEGAL DISCLAIMER:</strong> This examination was conducted using probabilistic machine learning architectures and mathematical digital signal processing algorithms. While these techniques identify statistically significant generative and compression anomalies, deepfake detection cannot establish absolute scientific certainty in the absence of verified cryptographic hardware provenance and chain-of-custody documentation.
          </div>

          {/* Sign-off footer */}
          <div className="border-t border-slate-200 pt-4 flex justify-between items-center text-[10px] font-mono text-slate-400">
            <span>FORENSIC VISION v1.0 • HNX26PSI10 LABORATORY ENGINE</span>
            <span>AUTHORIZED FORENSIC AUDIT REPORT</span>
          </div>
        </div>
      </div>
    </div>
  );
}
