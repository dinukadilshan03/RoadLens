/* eslint-disable @next/next/no-img-element */
"use client";

import { useState } from "react";
import type { FrameSummary, ObservationRecord } from "./api";

interface DetectionViewerProps {
  runId: string;
  frames: FrameSummary[];
  records: ObservationRecord[];
  selectedRecordId: string | null;
  onSelectRecord: (recordId: string | null) => void;
}

export default function DetectionViewer({
  runId,
  frames,
  records,
  selectedRecordId,
  onSelectRecord,
}: DetectionViewerProps) {
  const [currentFrameIdx, setCurrentFrameIdx] = useState(0);
  const [showAnnotated, setShowAnnotated] = useState(true);

  if (!frames || frames.length === 0) {
    return (
      <div className="flex h-96 items-center justify-center rounded-xl border border-dashed border-slate-300 bg-slate-50 text-center text-sm text-slate-500">
        No frame results available for this run.
      </div>
    );
  }

  const currentFrame = frames[currentFrameIdx] || frames[0];
  const frameRecords = records.filter((r) => r.frame_idx === currentFrame.frame_idx);

  const annotatedUrl = `/api/v1/c1/runs/${runId}/artifacts/annotated/${currentFrame.annotated_frame}`;
  const rawUrl = `/api/v1/c1/runs/${runId}/artifacts/inputs/${currentFrame.source_media}`;

  return (
    <div className="flex flex-col rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      {/* Header controls */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-800">
            Inspection Canvas
          </span>
          <span className="rounded bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-600">
            {currentFrame.width} × {currentFrame.height}px
          </span>
          <span className="rounded bg-blue-50 px-2 py-0.5 text-[11px] font-semibold text-blue-700">
            {frameRecords.length} {frameRecords.length === 1 ? "Defect" : "Defects"} Detected
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Toggle Annotated vs Clean */}
          <div className="inline-flex rounded-lg border border-slate-200 bg-slate-50 p-0.5 text-xs">
            <button
              type="button"
              onClick={() => setShowAnnotated(true)}
              className={`rounded-md px-3 py-1 font-semibold transition-all ${
                showAnnotated
                  ? "bg-white text-blue-700 shadow-sm"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Annotated (Boxes)
            </button>
            <button
              type="button"
              onClick={() => setShowAnnotated(false)}
              className={`rounded-md px-3 py-1 font-semibold transition-all ${
                !showAnnotated
                  ? "bg-white text-blue-700 shadow-sm"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Raw Clean Image
            </button>
          </div>

          {/* Video Frame Selector (if multi-frame video) */}
          {frames.length > 1 && (
            <div className="flex items-center gap-1.5 text-xs">
              <button
                type="button"
                disabled={currentFrameIdx === 0}
                onClick={() => setCurrentFrameIdx((i) => Math.max(0, i - 1))}
                className="rounded border border-slate-200 px-2 py-1 font-medium disabled:opacity-40"
              >
                ← Prev
              </button>
              <span className="font-semibold text-slate-700">
                {currentFrameIdx + 1} / {frames.length}
              </span>
              <button
                type="button"
                disabled={currentFrameIdx === frames.length - 1}
                onClick={() => setCurrentFrameIdx((i) => Math.min(frames.length - 1, i + 1))}
                className="rounded border border-slate-200 px-2 py-1 font-medium disabled:opacity-40"
              >
                Next →
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Frame Canvas */}
      <div className="relative mt-3 flex items-center justify-center overflow-hidden rounded-lg border border-slate-200 bg-slate-900">
        <img
          src={showAnnotated ? annotatedUrl : rawUrl}
          alt={`Road inspection frame ${currentFrame.frame_idx}`}
          className="max-h-[540px] w-auto max-w-full object-contain"
          onError={(e) => {
            // Fallback to annotated if raw input path is structured differently
            if (!showAnnotated) {
              (e.currentTarget as HTMLImageElement).src = annotatedUrl;
            }
          }}
        />

        {/* Informative overlay pill */}
        <div className="absolute bottom-2 left-2 flex items-center gap-2 rounded bg-black/75 px-2.5 py-1 text-[11px] text-white backdrop-blur-sm">
          <span>{currentFrame.source_media}</span>
          <span>•</span>
          <span>YOLOv8m B0 Inference</span>
          <span>•</span>
          <span>{showAnnotated ? "Bounding Boxes Active" : "Original View"}</span>
        </div>
      </div>

      {/* Quick summary below canvas */}
      <div className="mt-2.5 flex flex-wrap items-center justify-between text-xs text-slate-500">
        <div>
          <span>Distress findings: </span>
          {frameRecords.length === 0 ? (
            <span className="font-medium text-emerald-600">No distress detected above confidence threshold.</span>
          ) : (
            <span className="inline-flex flex-wrap items-center gap-1.5 font-medium text-slate-700">
              {frameRecords.map((r) => {
                const isSel = selectedRecordId === r.record_id;
                return (
                  <button
                    key={r.record_id}
                    type="button"
                    onClick={() => onSelectRecord(isSel ? null : r.record_id)}
                    className={`rounded px-1.5 py-0.5 text-[11px] font-semibold transition-colors cursor-pointer ${
                      isSel ? "bg-blue-600 text-white" : "bg-slate-100 text-slate-700 hover:bg-slate-200"
                    }`}
                  >
                    {r.display_name} ({(r.confidence * 100).toFixed(1)}%)
                  </button>
                );
              })}
            </span>
          )}
        </div>
        <div className="text-[11px]">
          Click any card on the right to inspect exact bounding box coordinates & C2 crop.
        </div>
      </div>
    </div>
  );
}
