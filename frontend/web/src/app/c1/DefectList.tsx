/* eslint-disable @next/next/no-img-element */
"use client";

import {
  CLASS_BG_LIGHT,
  CLASS_BORDER,
  CLASS_COLORS,
  CLASS_LABELS,
  type ObservationRecord,
} from "./api";

interface DefectListProps {
  records: ObservationRecord[];
  selectedRecordId: string | null;
  onSelectRecord: (recordId: string | null) => void;
}

export default function DefectList({
  records,
  selectedRecordId,
  onSelectRecord,
}: DefectListProps) {
  if (records.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center rounded-xl border border-slate-200 bg-white p-8 text-center shadow-sm">
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-emerald-50 text-emerald-600">
          <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        </div>
        <h3 className="mt-3 text-sm font-bold text-slate-800">Clear Pavement Surface</h3>
        <p className="mt-1 text-xs text-slate-500 max-w-xs">
          No pavement distress detected above the confidence threshold.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-100 pb-3">
        <div>
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800">
            Detected Pavement Defects ({records.length})
          </h3>
          <p className="text-[11px] text-slate-500">
            Localized observations with bounding box coordinates and confidence scores.
          </p>
        </div>
      </div>

      <div className="mt-3 flex flex-col gap-2.5 max-h-[500px] overflow-y-auto pr-1 c1-scroll">
        {records.map((rec) => {
          const isSelected = selectedRecordId === rec.record_id;
          const color = CLASS_COLORS[rec.damage_class] ?? "#3b82f6";
          const bgLight = CLASS_BG_LIGHT[rec.damage_class] ?? "#eff6ff";
          const borderColor = CLASS_BORDER[rec.damage_class] ?? "#bfdbfe";
          const [x1, y1, x2, y2] = rec.bbox;
          const w = Math.round(x2 - x1);
          const h = Math.round(y2 - y1);
          const confPercent = Math.round(rec.confidence * 100);

          return (
            <div
              key={rec.record_id}
              onClick={() => onSelectRecord(isSelected ? null : rec.record_id)}
              className={`rounded-lg border p-3 cursor-pointer transition-all ${
                isSelected
                  ? "border-blue-500 bg-blue-50/40 shadow-sm ring-1 ring-blue-500"
                  : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/50"
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span
                    className="h-3 w-3 rounded-full shrink-0"
                    style={{ backgroundColor: color }}
                  />
                  <div>
                    <span className="text-xs font-bold text-slate-900">
                      {CLASS_LABELS[rec.damage_class] ?? rec.display_name}
                    </span>
                    <span className="ml-1.5 rounded border px-1.5 py-0.2 text-[10px] font-semibold"
                      style={{ backgroundColor: bgLight, borderColor: borderColor, color: color }}
                    >
                      Class {rec.class_id}
                    </span>
                  </div>
                </div>

                <div className="text-right">
                  <span className="text-xs font-extrabold text-slate-900">{confPercent}%</span>
                  <span className="block text-[10px] text-slate-400 font-medium">confidence</span>
                </div>
              </div>

              {/* Confidence Progress Bar */}
              <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${confPercent}%`,
                    backgroundColor: color,
                  }}
                />
              </div>

              {/* Geometry & Coordinates metadata */}
              <div className="mt-2.5 flex flex-wrap items-center justify-between border-t border-slate-100 pt-2 text-[11px] text-slate-500">
                <div className="flex items-center gap-2">
                  <span>
                    BBox:{" "}
                    <code className="rounded bg-slate-100 px-1 py-0.5 font-mono text-[10px] text-slate-800">
                      [{x1.toFixed(0)}, {y1.toFixed(0)}, {x2.toFixed(0)}, {y2.toFixed(0)}]
                    </code>
                  </span>
                  <span>•</span>
                  <span>{w} × {h}px</span>
                </div>

                {rec.crop_url && (
                  <span className="font-semibold text-blue-600 hover:text-blue-800">
                    {isSelected ? "Hide Crop ▲" : "View C2 Crop ▼"}
                  </span>
                )}
              </div>

              {/* Expanded C2 Crop Preview when Selected */}
              {isSelected && rec.crop_url && (
                <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-2.5">
                  <div className="text-[11px] font-bold text-slate-700">
                    Downstream C2 Defect ROI Crop ({rec.crop_artifact}):
                  </div>
                  <div className="mt-1.5 flex items-center gap-3">
                    <img
                      src={rec.crop_url}
                      alt={`Defect ${rec.record_id} crop`}
                      className="h-20 w-auto rounded border border-slate-300 bg-white object-contain"
                    />
                    <div className="text-[10px] text-slate-600 space-y-1">
                      <div>
                        <span className="font-semibold text-slate-700">Origin (x,y):</span> [
                        {rec.crop_transform.origin_xy[0]}, {rec.crop_transform.origin_xy[1]}]
                      </div>
                      <div>
                        <span className="font-semibold text-slate-700">Crop Size:</span>{" "}
                        {rec.crop_transform.width} × {rec.crop_transform.height}px
                      </div>
                      <div className="text-slate-500">
                        Supplied to Component 2 for binary crack mask & skeleton segmentation.
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
