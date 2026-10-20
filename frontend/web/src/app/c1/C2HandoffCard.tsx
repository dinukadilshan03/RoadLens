/* eslint-disable @next/next/no-img-element */
"use client";

import type { ObservationRecord } from "./api";

interface C2HandoffCardProps {
  records: ObservationRecord[];
}

export default function C2HandoffCard({ records }: C2HandoffCardProps) {
  const recordsWithCrops = records.filter((r) => r.crop_url);

  if (recordsWithCrops.length === 0) {
    return null;
  }

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-blue-100 text-[11px] font-bold text-blue-700">
              C2
            </span>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800">
              Downstream C2 Integration Feed (Defect ROI Crops & Transforms)
            </h3>
          </div>
          <p className="mt-0.5 text-[11px] text-slate-500">
            C1 isolates defect bounding boxes into cropped image artifacts to feed Component 2 (Crack Segmentation).
          </p>
        </div>
        <span className="rounded bg-emerald-50 px-2 py-0.5 text-[10px] font-semibold text-emerald-700 border border-emerald-200">
          Contract: docs/interfaces.md § C1→C2 Ready
        </span>
      </div>

      {/* Grid of cropped defect ROIs */}
      <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {recordsWithCrops.map((r, idx) => (
          <div
            key={r.record_id}
            className="flex flex-col rounded-lg border border-slate-200 bg-slate-50 p-2 text-center"
          >
            <div className="flex h-24 items-center justify-center overflow-hidden rounded bg-white">
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
        ))}
      </div>

      {/* Contract Snippet */}
      <div className="mt-3 rounded bg-slate-900 p-2.5 font-mono text-[10px] text-slate-200 overflow-x-auto">
        <div className="text-slate-400 mb-1">{"// Exchange payload passed to Component 2 (Crack Segmentation):"}</div>
        <pre className="text-emerald-400">
{JSON.stringify(
  recordsWithCrops.slice(0, 2).map((r) => ({
    observation_id: r.record_id,
    damage_class: r.damage_class,
    crop_artifact: r.crop_artifact,
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
