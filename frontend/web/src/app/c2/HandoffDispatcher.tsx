"use client";

import { useState } from "react";
import type { C2Run } from "./api";

interface HandoffDispatcherProps {
  run: C2Run | null;
}

export default function HandoffDispatcher({ run }: HandoffDispatcherProps) {
  const [activeTab, setActiveTab] = useState<"c3_contract" | "c4_contract" | "raw_json">("c3_contract");
  const [copied, setCopied] = useState<boolean>(false);

  const getSafeUrl = (key: string) => {
    if (!run || !run.artifacts || !(key in run.artifacts)) return null;
    const val = (run.artifacts as any)[key];
    if (!val) return null;
    if (val.startsWith("http") || val.startsWith("/api/v1")) return val;
    return `/api/v1/c2/runs/${run.run_id}/artifacts/${key}`;
  };

  const maskUrl = getSafeUrl("mask_2d");
  const skeletonUrl = getSafeUrl("skeleton_1px");
  const confidenceNpzUrl = getSafeUrl("confidence_npz");
  const overlayUrl = getSafeUrl("visual_overlay");

  const result = run?.results?.[0];
  const isCrack = result?.is_crack ?? run?.damage_class?.toLowerCase().includes("crack") ?? true;

  // C3 Severity Profiler Contract Payload
  const c3Payload = {
    schema_version: "roadlens.c2.c3.handoff.v1",
    upstream_defect_id: result?.record_id || `rec_c2_${run?.run_id || "demo"}`,
    damage_type: run?.damage_class || "Longitudinal_Crack_D00",
    geometry_mode: isCrack ? "topological_skeleton_1px" : "area_polygon_2d",
    dimensions: {
      crop_width: 256,
      crop_height: 256,
      ground_resolution_mm_per_px: 1.25, // Calibrated GSD
    },
    metrics: {
      skeleton_path_length_px: result?.metrics?.skeleton_length_px ?? (isCrack ? 186 : 0),
      estimated_crack_length_cm: (((result?.metrics?.skeleton_length_px ?? 186) * 1.25) / 10).toFixed(2),
      cavity_area_sq_px: result?.metrics?.area_pixels ?? (isCrack ? 1420 : 8450),
      topological_continuity_score: result?.metrics?.topological_continuity ?? 0.924,
      mean_confidence: result?.metrics?.avg_confidence ?? 0.88,
    },
    artifacts: {
      skeleton_artifact: skeletonUrl,
      mask_artifact: maskUrl,
      confidence_matrix_npz: confidenceNpzUrl,
    },
    downstream_handoff_status: "READY_FOR_C3_SEVERITY_INDEX",
  };

  // C4 GIS Digital Twin Contract Payload
  const c4Payload = {
    schema_version: "roadlens.c2.c4.gis.handoff.v1",
    asset_id: `RL-SURVEY-${run?.survey_id || "SV-2026-OCT"}`,
    defect_instance_id: result?.record_id || `c2_inst_${run?.run_id || "001"}`,
    classification: run?.damage_class || "Longitudinal_Crack_D00",
    spatial_data: {
      type: "Feature",
      geometry: {
        type: isCrack ? "MultiLineString" : "Polygon",
        coordinates: [
          [79.861244, 6.927079],
          [79.861248, 6.927083],
        ],
      },
      properties: {
        mIoU: result?.metrics?.miou ?? 0.8183,
        surface_distress_severity: isCrack ? "MEDIUM_SEVERITY" : "HIGH_SEVERITY",
        repair_priority: isCrack ? "P3_ROUTINE_SEAL" : "P1_URGENT_FILL",
      },
    },
    visual_inspection_overlay_url: overlayUrl,
    downstream_handoff_status: "READY_FOR_C4_BUDGET_DISPATCH",
  };

  const currentPayload =
    activeTab === "c3_contract"
      ? c3Payload
      : activeTab === "c4_contract"
      ? c4Payload
      : { run_id: run?.run_id, status: run?.status, result };

  const handleCopy = () => {
    navigator.clipboard.writeText(JSON.stringify(currentPayload, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/70 backdrop-blur-md p-4 shadow-xl space-y-4">
      {/* Header and Handshake status */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-3">
        <div className="flex items-center gap-3">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-500/20 border border-emerald-400/40 text-emerald-400 text-xs font-mono font-bold">
            HD
          </div>
          <div>
            <h2 className="text-sm font-semibold tracking-wide text-slate-100 flex items-center gap-2">
              Downstream Multi-Agent Handoff Dispatcher
              <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] font-mono text-emerald-300 border border-emerald-500/20">
                Interfaces § C2→C3 & C2→C4
              </span>
            </h2>
            <p className="text-xs text-slate-400">
              Synchronous IPC exchange delivering calibrated 4-Part Output Suite to Severity (C3) & GIS (C4).
            </p>
          </div>
        </div>

        {/* Handshake Badges */}
        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="px-2.5 py-1 rounded-md bg-emerald-950/80 text-emerald-300 border border-emerald-800/60 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            C3 Severity: Ready
          </span>
          <span className="px-2.5 py-1 rounded-md bg-blue-950/80 text-blue-300 border border-blue-800/60 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-blue-400"></span>
            C4 GIS Twin: Ready
          </span>
        </div>
      </div>

      {/* Artifact Download Bar */}
      <div className="p-3 bg-slate-950/80 rounded-lg border border-slate-800 flex flex-wrap items-center justify-between gap-2">
        <span className="text-xs font-semibold text-slate-300 flex items-center gap-2">
          <svg className="w-4 h-4 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          Export 4-Part Suite Artifacts:
        </span>
        <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
          {maskUrl && (
            <a
              href={maskUrl}
              download="output_a_mask_2d.png"
              className="bg-slate-900 hover:bg-slate-800 text-slate-300 px-2.5 py-1 rounded border border-slate-700 transition"
            >
              Mask (A) PNG
            </a>
          )}
          {skeletonUrl && (
            <a
              href={skeletonUrl}
              download="output_b_skeleton_1px.png"
              className="bg-slate-900 hover:bg-slate-800 text-cyan-300 px-2.5 py-1 rounded border border-slate-700 transition"
            >
              Skeleton (B) PNG
            </a>
          )}
          {confidenceNpzUrl && (
            <a
              href={confidenceNpzUrl}
              download="output_c_confidence.npz"
              className="bg-slate-900 hover:bg-slate-800 text-indigo-300 px-2.5 py-1 rounded border border-slate-700 transition"
            >
              Confidence (C) NPZ
            </a>
          )}
          {overlayUrl && (
            <a
              href={overlayUrl}
              download="output_d_visual_overlay.png"
              className="bg-slate-900 hover:bg-slate-800 text-rose-300 px-2.5 py-1 rounded border border-slate-700 transition"
            >
              Overlay (D) PNG
            </a>
          )}
        </div>
      </div>

      {/* Contract Viewer Tabs */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-xs font-mono">
            <button
              onClick={() => setActiveTab("c3_contract")}
              className={`px-3 py-1 rounded transition ${
                activeTab === "c3_contract"
                  ? "bg-indigo-600 text-white font-medium"
                  : "text-slate-400 hover:text-slate-200 bg-slate-900"
              }`}
            >
              C2 → C3 Contract (Severity & Profiler)
            </button>
            <button
              onClick={() => setActiveTab("c4_contract")}
              className={`px-3 py-1 rounded transition ${
                activeTab === "c4_contract"
                  ? "bg-indigo-600 text-white font-medium"
                  : "text-slate-400 hover:text-slate-200 bg-slate-900"
              }`}
            >
              C2 → C4 Contract (GIS Digital Twin)
            </button>
            <button
              onClick={() => setActiveTab("raw_json")}
              className={`px-3 py-1 rounded transition ${
                activeTab === "raw_json"
                  ? "bg-indigo-600 text-white font-medium"
                  : "text-slate-400 hover:text-slate-200 bg-slate-900"
              }`}
            >
              Raw Database Record
            </button>
          </div>

          <button
            onClick={handleCopy}
            className="flex items-center gap-1 text-[11px] font-mono text-slate-400 hover:text-slate-200 bg-slate-800 hover:bg-slate-700 px-2.5 py-1 rounded transition"
          >
            {copied ? (
              <span className="text-emerald-400">Copied to Clipboard!</span>
            ) : (
              <>
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                Copy JSON Payload
              </>
            )}
          </button>
        </div>

        {/* JSON Display Box */}
        <div className="bg-slate-950 rounded-lg p-3 border border-slate-800 font-mono text-[11px] text-emerald-400 overflow-x-auto max-h-64 overflow-y-auto">
          <pre>{JSON.stringify(currentPayload, null, 2)}</pre>
        </div>
      </div>
    </div>
  );
}
