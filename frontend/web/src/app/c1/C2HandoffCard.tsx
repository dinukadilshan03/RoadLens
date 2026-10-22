/* eslint-disable @next/next/no-img-element */
"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import type { ObservationRecord } from "./api";

interface C2HandoffCardProps {
  records: ObservationRecord[];
  runId?: string;
  surveyId?: string;
}

export default function C2HandoffCard({ records, runId, surveyId }: C2HandoffCardProps) {
  const router = useRouter();
  const [batchBusy, setBatchBusy] = useState(false);
  const [batchError, setBatchError] = useState<string | null>(null);

  const recordsWithCrops = records.filter((r) => r.crop_url);

  if (recordsWithCrops.length === 0) {
    return null;
  }

  const effectiveRunId = runId || recordsWithCrops[0]?.external_run_id;

  const handleBatchHandoff = async () => {
    if (!effectiveRunId) return;
    setBatchBusy(true);
    setBatchError(null);
    try {
      const fd = new FormData();
      fd.append("c1_run_id", effectiveRunId);
      const res = await fetch("/api/v1/c2/runs/from-c1", {
        method: "POST",
        body: fd,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.message || `Batch handoff failed (${res.status})`);
      }
      const data = await res.json();
      const firstC2RunId = data.runs?.[0]?.run_id;
      const firstObsId = data.runs?.[0]?.c1_observation_id;
      const params = new URLSearchParams();
      params.set("c1_run_id", effectiveRunId);
      if (firstObsId) params.set("c1_observation_id", firstObsId);
      if (firstC2RunId) params.set("c2_run_id", firstC2RunId);
      router.push(`/c2?${params.toString()}`);
    } catch (e) {
      setBatchError(e instanceof Error ? e.message : String(e));
      setBatchBusy(false);
    }
  };

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-indigo-100 text-[11px] font-bold text-indigo-700">
              C2
            </span>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800">
              Downstream C2 Integration Feed (Defect ROI Crops & Transforms)
            </h3>
          </div>
          <p className="mt-0.5 text-[11px] text-slate-500">
            C1 isolates defect bounding boxes into cropped image artifacts to feed Component 2 (Noise-Resilient Segmentation).
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded bg-emerald-50 px-2 py-0.5 text-[10px] font-semibold text-emerald-700 border border-emerald-200">
            Contract: docs/interfaces.md § C1→C2 Ready
          </span>
          {effectiveRunId && (
            <button
              type="button"
              disabled={batchBusy}
              onClick={handleBatchHandoff}
              className="inline-flex items-center gap-1 rounded-md bg-indigo-600 px-2.5 py-1 text-[11px] font-bold text-white hover:bg-indigo-700 transition-colors disabled:opacity-50 cursor-pointer shadow-2xs"
            >
              {batchBusy
                ? "Dispatching to C2..."
                : `Segment All in C2 (${recordsWithCrops.length}) →`}
            </button>
          )}
        </div>
      </div>

      {batchError && (
        <div className="mt-2 rounded border border-red-200 bg-red-50 px-3 py-1.5 text-[11px] text-red-700">
          {batchError}
        </div>
      )}

      {/* Grid of cropped defect ROIs */}
      <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
        {recordsWithCrops.map((r, idx) => (
          <div
            key={r.record_id}
            className="flex flex-col justify-between rounded-lg border border-slate-200 bg-slate-50 p-2 text-center hover:border-indigo-300 transition-colors"
          >
            <div>
              <div className="flex h-24 items-center justify-center overflow-hidden rounded bg-white border border-slate-100">
                <img
                  src={r.crop_url}
                  alt={`C2 ROI Crop ${idx + 1}`}
                  className="max-h-full max-w-full object-contain"
                />
              </div>
              <div className="mt-1.5 text-left">
                <span className="block truncate text-[11px] font-bold text-slate-800">
                  {r.display_name}
                </span>
                <span className="block text-[10px] font-mono text-slate-500">
                  {r.crop_transform.width}×{r.crop_transform.height}px @ ({r.crop_transform.origin_xy[0]}, {r.crop_transform.origin_xy[1]})
                </span>
              </div>
            </div>

            <Link
              href={`/c2?c1_run_id=${encodeURIComponent(r.external_run_id)}&c1_observation_id=${encodeURIComponent(r.record_id)}&auto_run=1`}
              className="mt-2 inline-flex w-full items-center justify-center rounded border border-indigo-200 bg-indigo-50 px-2 py-1 text-[10px] font-bold text-indigo-700 hover:bg-indigo-600 hover:text-white transition-colors"
            >
              Segment in C2 →
            </Link>
          </div>
        ))}
      </div>

      {/* Contract Snippet */}
      <div className="mt-3 rounded bg-slate-900 p-2.5 font-mono text-[10px] text-slate-200 overflow-x-auto">
        <div className="text-slate-400 mb-1">{"// Exchange payload passed to Component 2 (Segmentation Engine):"}</div>
        <pre className="text-emerald-400">
{JSON.stringify(
  recordsWithCrops.slice(0, 2).map((r) => ({
    c1_run_id: r.external_run_id,
    c1_observation_id: r.record_id,
    defect_id: r.defect_id,
    survey_id: r.survey_id || surveyId,
    damage_class: r.damage_class,
    confidence: r.confidence,
    bbox: r.bbox,
    crop_artifact: r.crop_artifact,
    crop_url: r.crop_url,
    crop_transform: r.crop_transform,
  })),
  null,
  2
)}
        </pre>
      </div>
    </div>
  );
}
