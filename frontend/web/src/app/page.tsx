import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
      {/* Top Banner */}
      <header className="border-b border-slate-200 bg-white shadow-2xs">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-6">
          <div className="flex items-center gap-3">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600 text-sm font-black text-white shadow-xs">
              RL
            </span>
            <div>
              <span className="text-base font-extrabold tracking-tight text-slate-900 block leading-tight">
                RoadLens Platform
              </span>
              <span className="text-[11px] text-slate-500 font-medium">
                AI Framework for Pavement Distress & Roughness Management
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="rounded-full bg-emerald-50 border border-emerald-200 px-3 py-1 text-xs font-semibold text-emerald-700">
              ● API & Models Active
            </span>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="mx-auto max-w-5xl px-4 py-12 sm:px-6 space-y-10">
        {/* Hero Section */}
        <div className="text-center space-y-3">
          <span className="inline-flex items-center rounded-full bg-blue-50 border border-blue-200 px-3 py-1 text-xs font-bold text-blue-700 uppercase tracking-wider">
            SLIIT Research Demonstration Suite · 2026
          </span>
          <h1 className="text-3xl font-black tracking-tight text-slate-900 sm:text-4xl">
            Autonomous Road Condition Assessment
          </h1>
          <p className="mx-auto max-w-2xl text-sm text-slate-600 leading-relaxed">
            Select a research component below to explore live inference, model evaluation metrics,
            and inter-component pipeline data feeds.
          </p>
        </div>

        {/* Component Navigation Cards Grid */}
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
          {/* Component 1 Card */}
          <div className="flex flex-col justify-between rounded-2xl border-2 border-blue-500/80 bg-white p-6 shadow-sm hover:shadow-md transition-all">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600 text-xs font-bold text-white shadow-xs">
                  C1
                </span>
                <span className="rounded-full bg-blue-50 border border-blue-200 px-2.5 py-0.5 text-xs font-bold text-blue-700">
                  Featured Viva Demo
                </span>
              </div>

              <div>
                <h2 className="text-lg font-bold text-slate-900">
                  Road Damage Detection & Downstream Feed
                </h2>
                <div className="text-xs font-medium text-slate-400 mt-0.5">
                  Member: IT23262690 · Base Model (B0) Phase
                </div>
              </div>

              <p className="text-xs text-slate-600 leading-relaxed">
                Autonomous distress localization using YOLOv8m (RDD2022 international benchmark).
                Detects Potholes (D40), Alligator Cracks (D20), Longitudinal Cracks (D00), and Transverse Cracks (D10)
                with calibrated confidence and generates cropped defect ROIs to feed Component 2 (Crack Segmentation).
              </p>

              <div className="flex flex-wrap gap-1.5 pt-1">
                <span className="rounded bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-700">
                  YOLOv8m Base Model
                </span>
                <span className="rounded bg-emerald-50 text-emerald-700 border border-emerald-200 px-2 py-0.5 text-[11px] font-bold">
                  66.2% mAP@50
                </span>
                <span className="rounded bg-red-50 text-red-700 border border-red-200 px-2 py-0.5 text-[11px] font-bold">
                  78.4% Pothole Recall
                </span>
                <span className="rounded bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-700">
                  C2 Defect Crops
                </span>
              </div>
            </div>

            <div className="mt-6 pt-4 border-t border-slate-100">
              <Link
                href="/c1"
                className="inline-flex w-full items-center justify-center rounded-lg bg-blue-600 px-4 py-2.5 text-xs font-bold text-white hover:bg-blue-700 transition-colors shadow-xs"
              >
                Launch C1 Damage Detection Demo →
              </Link>
            </div>
          </div>

          {/* Component 4 Card */}
          <div className="flex flex-col justify-between rounded-2xl border border-slate-200 bg-white p-6 shadow-sm hover:shadow-md transition-all">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-800 text-xs font-bold text-white shadow-xs">
                  C4
                </span>
                <span className="rounded-full bg-slate-100 border border-slate-200 px-2.5 py-0.5 text-xs font-bold text-slate-700">
                  Sensor Pipeline
                </span>
              </div>

              <div>
                <h2 className="text-lg font-bold text-slate-900">
                  Road Roughness & Ride Quality Assessment
                </h2>
                <div className="text-xs font-medium text-slate-400 mt-0.5">
                  Member: IT23222472 · Motion Sensor Branch
                </div>
              </div>

              <p className="text-xs text-slate-600 leading-relaxed">
                Estimates pavement roughness from vehicle vibration telemetry (IMU + GPS).
                Produces segment-level condition assessments (Good, Regular, Bad)
                and diagnostic window metrics along surveyed road coordinates.
              </p>

              <div className="flex flex-wrap gap-1.5 pt-1">
                <span className="rounded bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-700">
                  IMU Accelerometer
                </span>
                <span className="rounded bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-700">
                  XGBoost Classifier
                </span>
                <span className="rounded bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-700">
                  GeoJSON Segments
                </span>
              </div>
            </div>

            <div className="mt-6 pt-4 border-t border-slate-100">
              <Link
                href="/c4"
                className="inline-flex w-full items-center justify-center rounded-lg border border-slate-300 bg-slate-50 px-4 py-2.5 text-xs font-bold text-slate-800 hover:bg-slate-100 transition-colors"
              >
                Launch C4 Roughness Demo →
              </Link>
            </div>
          </div>
        </div>

        {/* Architecture Pipeline Summary Callout */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 text-xs text-slate-600 space-y-2">
          <div className="font-bold text-slate-800 uppercase tracking-wider text-[11px]">
            Platform Research Architecture:
          </div>
          <p>
            • <strong>C1 (Road Damage Detection)</strong> localizes distress bounding boxes and outputs defect ROI crops.
          </p>
          <p>
            • <strong>C2 (Crack Segmentation)</strong> consumes C1 ROI crops to produce pixel-level binary distress masks and skeletons.
          </p>
          <p>
            • <strong>C3 (Severity & Measurement)</strong> calculates physical metric dimensions (area, depth, volume) from C2 masks.
          </p>
          <p>
            • <strong>C4 (Roughness Assessment)</strong> independently monitors motion telemetry to provide road segment quality scores.
          </p>
        </div>
      </main>
    </div>
  );
}
