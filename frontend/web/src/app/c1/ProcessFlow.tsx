"use client";

import { useState } from "react";
import type { RunStatus } from "./api";

interface ProcessStep {
  id: number;
  key: string;
  title: string;
  shortLabel: string;
  badge: string;
  description: string;
  technicalNote: string;
}

const STEPS: ProcessStep[] = [
  {
    id: 1,
    key: "ingest",
    title: "1. Media Ingestion",
    shortLabel: "Input Media",
    badge: "Images & Video",
    description: "Accepts high-resolution road photographs or continuous smartphone dashcam recording.",
    technicalNote: "Supports .jpg, .png, .mp4. Continuous video feeds are parsed into presentation timestamps without dropping frames.",
  },
  {
    id: 2,
    key: "preprocess",
    title: "2. Frame Standardization",
    shortLabel: "Preprocessing",
    badge: "800x800 Normalization",
    description: "Resolution normalization and letterbox padding preserving native crack aspect ratios.",
    technicalNote: "800x800 input resolution was selected over standard 640 to retain fine pixel features of hairline cracks.",
  },
  {
    id: 3,
    key: "inference",
    title: "3. Base Model Inference",
    shortLabel: "YOLOv8m (B0)",
    badge: "25.8M Parameters",
    description: "Deep convolutional feature extraction trained on RDD2022 (38,269 international images).",
    technicalNote: "YOLOv8m backbone with PAN-FPN neck. Validated at 66.2% mAP@50 overall and 78.4% specifically on Potholes (D40).",
  },
  {
    id: 4,
    key: "localize",
    title: "4. Distress Localization",
    shortLabel: "Bounding Boxes",
    badge: "4 Distress Classes",
    description: "Multi-class bounding box regression with non-maximum suppression (NMS) and confidence scoring.",
    technicalNote: "Outputs exact coordinates [x_min, y_min, x_max, y_max], class classification (D00, D10, D20, D40), and score in [0, 1].",
  },
  {
    id: 5,
    key: "handoff",
    title: "5. Downstream C2 Feed",
    shortLabel: "C2 Defect Crops",
    badge: "C1 → C2 Pipeline",
    description: "Automated defect ROI cropping with spatial transform coordinates for Component 2.",
    technicalNote: "Packages cropped distress patches with pixel transforms {origin_xy, width, height} to feed C2 Crack Segmentation directly.",
  },
];

interface ProcessFlowProps {
  status: RunStatus | null;
  stage: string | null;
  totalDefects?: number;
}

export default function ProcessFlow({ status, stage, totalDefects }: ProcessFlowProps) {
  const [selectedStep, setSelectedStep] = useState<number | null>(null);

  // Determine which step is currently active
  const getStepStatus = (stepId: number): "idle" | "active" | "completed" => {
    if (!status) return "idle";
    if (status === "succeeded") return "completed";
    if (status === "failed" || status === "cancelled") return "idle";

    if (stage === "extracting_frames") {
      if (stepId <= 2) return stepId === 2 ? "active" : "completed";
      return "idle";
    }
    if (stage === "running_inference") {
      if (stepId < 3) return "completed";
      if (stepId === 3) return "active";
      return "idle";
    }
    if (stage === "writing_outputs") {
      if (stepId < 4) return "completed";
      return "active";
    }
    if (stage === "done") return "completed";

    return stepId === 1 ? "active" : "idle";
  };

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
        <div>
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800">
            System Process Flow (Input → YOLOv8m Inference → C2 Feed)
          </h2>
          <p className="text-xs text-slate-500">
            Component 1 lifecycle: from raw pavement capture to defect localization and downstream C2 handoff.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {status === "running" && (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-blue-50 px-2.5 py-1 text-xs font-semibold text-blue-700">
              <span className="h-2 w-2 rounded-full bg-blue-600 c1-pulsing" />
              Processing: {stage ?? "In Progress"}
            </span>
          )}
          {status === "succeeded" && (
            <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-semibold text-emerald-700">
              <svg className="h-3.5 w-3.5 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
              </svg>
              Pipeline Complete ({totalDefects ?? 0} defects localized)
            </span>
          )}
        </div>
      </div>

      {/* 5-Stage Process Cards */}
      <div className="mt-4 grid grid-cols-1 gap-2.5 sm:grid-cols-2 lg:grid-cols-5">
        {STEPS.map((step) => {
          const s = getStepStatus(step.id);
          const isSelected = selectedStep === step.id;

          let badgeStyle = "bg-slate-100 text-slate-600 border-slate-200";
          let circleStyle = "bg-slate-100 text-slate-500 border-slate-300";
          let cardStyle = "border-slate-200 bg-slate-50/50 hover:bg-slate-50";

          if (s === "completed") {
            badgeStyle = "bg-emerald-50 text-emerald-700 border-emerald-200";
            circleStyle = "bg-emerald-600 text-white border-emerald-600";
            cardStyle = "border-emerald-200 bg-emerald-50/30";
          } else if (s === "active") {
            badgeStyle = "bg-blue-50 text-blue-700 border-blue-200";
            circleStyle = "bg-blue-600 text-white border-blue-600 c1-pulsing";
            cardStyle = "border-blue-400 bg-blue-50/30 shadow-sm";
          }

          if (isSelected) {
            cardStyle += " ring-2 ring-blue-500";
          }

          return (
            <button
              key={step.id}
              type="button"
              onClick={() => setSelectedStep(isSelected ? null : step.id)}
              className={`flex flex-col justify-between rounded-lg border p-3.5 text-left transition-all cursor-pointer ${cardStyle}`}
            >
              <div>
                <div className="flex items-center justify-between gap-1.5">
                  <span className={`flex h-6 w-6 items-center justify-center rounded-full border text-xs font-bold ${circleStyle}`}>
                    {s === "completed" ? "✓" : step.id}
                  </span>
                  <span className={`rounded border px-1.5 py-0.5 text-[10px] font-semibold tracking-wide ${badgeStyle}`}>
                    {step.badge}
                  </span>
                </div>
                <h3 className="mt-2 text-xs font-bold text-slate-900">{step.title}</h3>
                <p className="mt-1 line-clamp-2 text-[11px] text-slate-600 leading-snug">
                  {step.description}
                </p>
              </div>
              <div className="mt-2.5 flex items-center justify-between border-t border-slate-200/60 pt-2 text-[10px] text-slate-500">
                <span className="font-medium text-slate-700">{step.shortLabel}</span>
                <span className="text-blue-600 font-semibold">{isSelected ? "Close ↑" : "Notes →"}</span>
              </div>
            </button>
          );
        })}
      </div>

      {/* Expandable Scientific Details Note for Selected Step */}
      {selectedStep !== null && (
        <div className="mt-3 rounded-lg border border-blue-200 bg-blue-50/60 p-3 text-xs text-blue-900">
          <div className="flex items-start justify-between gap-2">
            <div>
              <span className="font-bold text-blue-950">
                {STEPS[selectedStep - 1].title} · Examiner Reference Note:
              </span>{" "}
              <span className="text-blue-900">{STEPS[selectedStep - 1].technicalNote}</span>
            </div>
            <button
              type="button"
              onClick={() => setSelectedStep(null)}
              className="text-xs font-bold text-blue-700 hover:text-blue-900"
            >
              ✕
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
