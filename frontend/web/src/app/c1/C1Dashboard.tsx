/* eslint-disable @next/next/no-img-element */
"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import C2HandoffCard from "./C2HandoffCard";
import DefectList from "./DefectList";
import DetectionViewer from "./DetectionViewer";
import ProcessFlow from "./ProcessFlow";
import {
  ApiError,
  CLASS_COLORS,
  CLASS_LABELS,
  PRESET_SAMPLES,
  api,
  isActive,
  type DamageClass,
  type ModelInfo,
  type ObservationRecord,
  type Run,
  type SamplePreset,
} from "./api";

function errText(e: unknown): string {
  return e instanceof ApiError || e instanceof Error ? e.message : String(e);
}

export default function C1Dashboard() {
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [activeRun, setActiveRun] = useState<Run | null>(null);
  const [records, setRecords] = useState<ObservationRecord[]>([]);
  const [selectedRecordId, setSelectedRecordId] = useState<string | null>(null);

  const [confThreshold, setConfThreshold] = useState<number>(0.25);
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadType, setUploadType] = useState<"preset" | "custom">("preset");

  // 1. Initial Load: Fetch Model Info and Recent Runs
  useEffect(() => {
    let ignore = false;
    api.model()
      .then((m) => {
        if (!ignore) setModel(m);
      })
      .catch((e) => {
        if (!ignore) setError(`Could not load C1 model info: ${errText(e)}`);
      });

    api.runs()
      .then((r) => {
        if (!ignore && r.length > 0) {
          setRuns(r);
          setActiveRun(r[0]);
        }
      })
      .catch(() => {
        // Ignored on initial cold start
      });

    return () => {
      ignore = true;
    };
  }, []);

  // 2. Fetch Records when Active Run Changes & Succeeds
  useEffect(() => {
    if (!activeRun || activeRun.status !== "succeeded") {
      return;
    }
    let ignore = false;
    api.records(activeRun.run_id)
      .then((recs) => {
        if (!ignore) setRecords(recs);
      })
      .catch((e) => {
        if (!ignore) setError(`Error loading records: ${errText(e)}`);
      });

    return () => {
      ignore = true;
    };
  }, [activeRun]);

  // 3. Polling for Active Run
  useEffect(() => {
    if (!activeRun || !isActive(activeRun.status)) return;
    const runId = activeRun.run_id;
    const interval = setInterval(async () => {
      try {
        const updated = await api.run(runId);
        setActiveRun(updated);
        setRuns((prev) => prev.map((r) => (r.run_id === updated.run_id ? updated : r)));
        if (!isActive(updated.status)) {
          clearInterval(interval);
          setBusy(false);
        }
      } catch {
        // transient poll error
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [activeRun]);

  // 4. Action: Run Preset Demo Sample
  const handleRunPreset = useCallback(
    async (preset: SamplePreset) => {
      setBusy(true);
      setError(null);
      setSelectedRecordId(null);
      setRecords([]);
      try {
        const run = await api.demo(preset.key, confThreshold);
        setActiveRun(run);
        setRuns((prev) => [run, ...prev]);
      } catch (e) {
        setError(`Failed to trigger preset run: ${errText(e)}`);
        setBusy(false);
      }
    },
    [confThreshold]
  );

  // 5. Action: Custom Upload
  const handleCustomUpload = useCallback(
    async (e: React.FormEvent<HTMLFormElement>) => {
      e.preventDefault();
      const form = e.currentTarget;
      const data = new FormData(form);

      const files = data.getAll("files") as File[];
      if (!files || files.length === 0 || files[0].size === 0) {
        setError("Please choose at least one image or video file.");
        return;
      }

      setBusy(true);
      setError(null);
      setSelectedRecordId(null);
      setRecords([]);
      try {
        const run = await api.upload(data);
        setActiveRun(run);
        setRuns((prev) => [run, ...prev]);
        form.reset();
      } catch (e) {
        setError(`Upload failed: ${errText(e)}`);
        setBusy(false);
      }
    },
    []
  );

  const summary = activeRun?.summary;

  return (
    <div className="c1-theme min-h-screen bg-slate-50 text-slate-900 pb-16 font-sans">
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-200 bg-white shadow-xs">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="flex items-center gap-2 text-xs font-semibold text-slate-500 hover:text-slate-800 transition-colors"
            >
              <span>← RoadLens Platform</span>
            </Link>
            <span className="text-slate-300">|</span>
            <div className="flex items-center gap-2">
              <span className="flex h-6 w-6 items-center justify-center rounded bg-blue-600 text-xs font-bold text-white shadow-xs">
                C1
              </span>
              <span className="text-sm font-bold text-slate-900">Road Damage Detection</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <span className="hidden sm:inline-flex items-center gap-1.5 rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700">
              <span className="h-2 w-2 rounded-full bg-emerald-500" />
              {model?.model_version ?? "YOLOv8m Base Model Operational"}
            </span>
            <span className="rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-mono text-slate-600">
              {model ? `${model.input_resolution}×${model.input_resolution}` : "800×800"} Res
            </span>
          </div>
        </div>
      </header>

      {/* Center road divider line */}
      <div className="c1-centre-line w-full" />

      {/* Main Content Container */}
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 space-y-6">
        {/* Academic Presentation Banner */}
        <div className="rounded-xl border border-blue-200 bg-gradient-to-r from-blue-50/80 via-white to-blue-50/40 p-5 shadow-xs">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="space-y-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className="rounded bg-blue-600 px-2 py-0.5 text-[11px] font-bold uppercase tracking-wider text-white">
                  Research Component 1
                </span>
                <span className="text-xs font-medium text-slate-500">
                  IT23262690 · Pavement Distress Localization & Temporal Tracking
                </span>
              </div>
              <h1 className="text-xl font-extrabold tracking-tight text-slate-900 sm:text-2xl">
                Base Model Distress Detection & Downstream Feed
              </h1>
              <p className="max-w-3xl text-xs text-slate-600 leading-relaxed">
                Trained on international benchmark datasets (RDD2022 across 6 countries). Localizes 4 core
                distress classes (Potholes, Alligator Cracks, Longitudinal Cracks, Transverse Cracks),
                generates calibrated confidence scores, and feeds isolated defect ROIs to Component 2 (Crack Segmentation).
              </p>
            </div>

            {/* Model Provenance Stats Box */}
            <div className="flex flex-wrap gap-2 lg:shrink-0">
              <div className="rounded-lg border border-slate-200 bg-white p-2.5 shadow-2xs">
                <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                  Overall Benchmark
                </div>
                <div className="text-base font-extrabold text-blue-700">66.2% mAP@50</div>
                <div className="text-[10px] text-slate-500">
                  {model ? `SHA: ${model.weights_sha256.slice(0, 8)}` : "YOLOv8m (25.8M params)"}
                </div>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-2.5 shadow-2xs">
                <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                  Pothole Detection
                </div>
                <div className="text-base font-extrabold text-red-600">78.4% AP@50</div>
                <div className="text-[10px] text-slate-500">Class D40 High Recall</div>
              </div>
              <div className="rounded-lg border border-slate-200 bg-white p-2.5 shadow-2xs">
                <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                  Milestone Status
                </div>
                <div className="text-sm font-bold text-emerald-700">PP1 Ready (50%)</div>
                <div className="text-[10px] text-slate-500">PP2: Sri Lankan Domain Adapt</div>
              </div>
            </div>
          </div>
        </div>

        {/* Process Flow Visualizer (The Core Viva Story: Input -> Inference -> Outputs) */}
        <ProcessFlow
          status={activeRun?.status ?? null}
          stage={activeRun?.stage ?? null}
          totalDefects={records.length}
        />

        {/* Global Error Banner */}
        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-3.5 text-xs text-red-700 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="font-bold">Error:</span> {error}
            </div>
            <button
              type="button"
              onClick={() => setError(null)}
              className="text-xs font-bold text-red-900 hover:underline"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Interactive Control Panel: 1-Click Preset Demos vs Custom Upload */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3">
            <div>
              <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800">
                Interactive Demonstration Controls
              </h2>
              <p className="text-xs text-slate-500">
                Choose a curated 1-click test road sample or upload your own image/video recording.
              </p>
            </div>

            {/* Upload Mode Selector */}
            <div className="inline-flex rounded-lg border border-slate-200 bg-slate-50 p-0.5 text-xs">
              <button
                type="button"
                onClick={() => setUploadType("preset")}
                className={`rounded-md px-3 py-1.5 font-bold transition-all cursor-pointer ${
                  uploadType === "preset"
                    ? "bg-white text-blue-700 shadow-2xs"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                1-Click Preset Samples
              </button>
              <button
                type="button"
                onClick={() => setUploadType("custom")}
                className={`rounded-md px-3 py-1.5 font-bold transition-all cursor-pointer ${
                  uploadType === "custom"
                    ? "bg-white text-blue-700 shadow-2xs"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                Custom Media Upload
              </button>
            </div>
          </div>

          {/* Preset Buttons Grid */}
          {uploadType === "preset" ? (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {PRESET_SAMPLES.map((preset) => (
                <div
                  key={preset.key}
                  className="c1-card flex flex-col justify-between overflow-hidden rounded-lg border border-slate-200 bg-white"
                >
                  <div className="relative h-28 w-full bg-slate-100 overflow-hidden">
                    <img
                      src={preset.imageUrl}
                      alt={preset.title}
                      className="h-full w-full object-cover"
                    />
                    <span className="absolute top-2 right-2 rounded bg-black/75 px-1.5 py-0.5 text-[10px] font-bold text-white backdrop-blur-xs">
                      {preset.badge}
                    </span>
                  </div>

                  <div className="p-3 flex flex-col flex-1 justify-between">
                    <div>
                      <h4 className="text-xs font-bold text-slate-900">{preset.title}</h4>
                      <p className="mt-1 text-[11px] text-slate-500 line-clamp-2">
                        {preset.description}
                      </p>
                      <span className="mt-2 block text-[10px] font-semibold text-blue-600">
                        {preset.distressTypes}
                      </span>
                    </div>

                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => handleRunPreset(preset)}
                      className="mt-3 inline-flex w-full items-center justify-center rounded-md bg-blue-600 px-3 py-1.5 text-xs font-bold text-white hover:bg-blue-700 transition-colors disabled:opacity-50 cursor-pointer shadow-2xs"
                    >
                      {busy ? "Processing..." : "▶ Run Inference"}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            /* Custom Upload Form */
            <form onSubmit={handleCustomUpload} className="space-y-4">
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                <div className="sm:col-span-2">
                  <label className="block text-xs font-bold text-slate-700 mb-1">
                    Upload Road Media File
                  </label>
                  <input
                    name="files"
                    type="file"
                    accept=".jpg,.jpeg,.png,.webp,.bmp,.mp4,.avi,.mov"
                    multiple
                    required
                    className="block w-full text-xs text-slate-600 file:mr-3 file:rounded-md file:border-0 file:bg-blue-50 file:px-3 file:py-2 file:text-xs file:font-bold file:text-blue-700 hover:file:bg-blue-100 border border-slate-200 rounded-md p-1.5"
                  />
                  <p className="mt-1 text-[11px] text-slate-500">
                    Supports high-res photographs or road video recordings.
                  </p>
                </div>

                <div>
                  <label className="block text-xs font-bold text-slate-700 mb-1">
                    Survey / Road Route Identifier
                  </label>
                  <input
                    name="survey_id"
                    type="text"
                    placeholder="e.g. Galle-Road-A2"
                    className="w-full rounded-md border border-slate-200 px-3 py-1.5 text-xs text-slate-900 placeholder:text-slate-400"
                  />
                </div>
              </div>

              <div className="flex items-center justify-between border-t border-slate-100 pt-3">
                <div className="flex items-center gap-3">
                  <span className="text-xs font-bold text-slate-700">Sensitivity Threshold:</span>
                  <input
                    type="range"
                    min="0.10"
                    max="0.80"
                    step="0.05"
                    value={confThreshold}
                    onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                    className="w-32 cursor-pointer"
                  />
                  <span className="font-mono text-xs font-bold text-blue-700">
                    {Math.round(confThreshold * 100)}%
                  </span>
                </div>

                <button
                  type="submit"
                  disabled={busy}
                  className="rounded-md bg-blue-600 px-5 py-2 text-xs font-bold text-white hover:bg-blue-700 transition-colors disabled:opacity-50 cursor-pointer shadow-xs"
                >
                  {busy ? "Processing Inference..." : "Upload & Run Detection"}
                </button>
              </div>
            </form>
          )}

          {/* Quick sensitivity slider for preset mode as well */}
          {uploadType === "preset" && (
            <div className="flex flex-wrap items-center justify-between border-t border-slate-100 pt-3 text-xs text-slate-600">
              <div className="flex items-center gap-3">
                <span className="font-bold text-slate-700">Confidence Threshold (Filter NMS):</span>
                <input
                  type="range"
                  min="0.10"
                  max="0.80"
                  step="0.05"
                  value={confThreshold}
                  onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                  className="w-32 cursor-pointer"
                />
                <span className="font-mono font-bold text-blue-700">
                  {Math.round(confThreshold * 100)}%
                </span>
                <span className="text-[11px] text-slate-500">
                  (Adjusts minimum confidence required to flag distress)
                </span>
              </div>

              {activeRun && (
                <div className="text-[11px] text-slate-500">
                  Active Run ID: <code className="font-mono text-slate-800">{activeRun.run_id.slice(0, 8)}</code>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Results Showcase: Two-Column Responsive Layout */}
        {activeRun?.status === "succeeded" && summary && (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
            {/* Left Column (7 cols): Frame Inspection Canvas & Metrics */}
            <div className="space-y-6 lg:col-span-7">
              <DetectionViewer
                runId={activeRun.run_id}
                frames={summary.frames}
                records={records}
                selectedRecordId={selectedRecordId}
                onSelectRecord={setSelectedRecordId}
              />

              {/* Summary Metrics Grid */}
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <div className="rounded-lg border border-slate-200 bg-white p-3.5 shadow-2xs">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                    Defects Detected
                  </div>
                  <div className="mt-1 text-2xl font-black text-slate-900">
                    {summary.total_defects_detected}
                  </div>
                  <div className="text-[10px] text-slate-500">Across {summary.total_frames_processed} frames</div>
                </div>

                <div className="rounded-lg border border-slate-200 bg-white p-3.5 shadow-2xs">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                    Average Confidence
                  </div>
                  <div className="mt-1 text-2xl font-black text-blue-700">
                    {Math.round(summary.average_confidence * 100)}%
                  </div>
                  <div className="text-[10px] text-slate-500">Peak: {Math.round(summary.max_confidence * 100)}%</div>
                </div>

                <div className="rounded-lg border border-slate-200 bg-white p-3.5 shadow-2xs">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                    Severe Potholes
                  </div>
                  <div className="mt-1 text-2xl font-black text-red-600">
                    {summary.counts_by_class["Pothole_D40"] ?? 0}
                  </div>
                  <div className="text-[10px] text-slate-500">Class D40 Depressions</div>
                </div>

                <div className="rounded-lg border border-slate-200 bg-white p-3.5 shadow-2xs">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                    Fatigue / Cracks
                  </div>
                  <div className="mt-1 text-2xl font-black text-amber-600">
                    {(summary.counts_by_class["Alligator_Crack_D20"] ?? 0) +
                      (summary.counts_by_class["Longitudinal_Crack_D00"] ?? 0) +
                      (summary.counts_by_class["Transverse_Crack_D10"] ?? 0)}
                  </div>
                  <div className="text-[10px] text-slate-500">Linear & Interconnected</div>
                </div>
              </div>

              {/* Class Distribution Bar */}
              <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-2xs">
                <div className="text-xs font-bold uppercase tracking-wider text-slate-700 mb-2">
                  Distress Classification Breakdown
                </div>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  {(["Pothole_D40", "Alligator_Crack_D20", "Longitudinal_Crack_D00", "Transverse_Crack_D10"] as DamageClass[]).map(
                    (cname) => {
                      const count = summary.counts_by_class[cname] ?? 0;
                      const color = CLASS_COLORS[cname];
                      return (
                        <div key={cname} className="flex items-center gap-2 rounded border border-slate-100 bg-slate-50 p-2">
                          <span className="h-3 w-3 rounded-full shrink-0" style={{ backgroundColor: color }} />
                          <div className="min-w-0 flex-1">
                            <span className="block truncate text-[11px] font-bold text-slate-800">
                              {CLASS_LABELS[cname]}
                            </span>
                            <span className="text-[10px] text-slate-500">{count} occurrences</span>
                          </div>
                        </div>
                      );
                    }
                  )}
                </div>
              </div>
            </div>

            {/* Right Column (5 cols): Defect Inspector List & Downstream C2 Feed */}
            <div className="space-y-6 lg:col-span-5">
              <DefectList
                records={records}
                selectedRecordId={selectedRecordId}
                onSelectRecord={setSelectedRecordId}
              />

              <C2HandoffCard records={records} />
            </div>
          </div>
        )}

        {/* Recent Runs History Section */}
        {runs.length > 0 && (
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 mb-3">
              Run History & Provenance ({runs.length} runs recorded)
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-600">
                <thead className="border-b border-slate-200 bg-slate-50 text-[10px] font-bold uppercase tracking-wider text-slate-500">
                  <tr>
                    <th className="px-3 py-2">Run ID</th>
                    <th className="px-3 py-2">Survey Identifier</th>
                    <th className="px-3 py-2">Source</th>
                    <th className="px-3 py-2">Status</th>
                    <th className="px-3 py-2">Defects Found</th>
                    <th className="px-3 py-2">Avg Confidence</th>
                    <th className="px-3 py-2">Timestamp</th>
                    <th className="px-3 py-2">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-medium">
                  {runs.slice(0, 8).map((r) => {
                    const isCurrent = activeRun?.run_id === r.run_id;
                    return (
                      <tr
                        key={r.run_id}
                        className={`hover:bg-slate-50/80 transition-colors ${
                          isCurrent ? "bg-blue-50/50" : ""
                        }`}
                      >
                        <td className="px-3 py-2.5 font-mono text-slate-800 font-bold">
                          {r.run_id.slice(0, 8)}
                        </td>
                        <td className="px-3 py-2.5 text-slate-700">{r.survey_id}</td>
                        <td className="px-3 py-2.5">
                          <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-semibold text-slate-700">
                            {r.source}
                          </span>
                        </td>
                        <td className="px-3 py-2.5">
                          <span
                            className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${
                              r.status === "succeeded"
                                ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                                : r.status === "running"
                                ? "bg-blue-50 text-blue-700 border border-blue-200"
                                : "bg-slate-100 text-slate-600"
                            }`}
                          >
                            {r.status}
                          </span>
                        </td>
                        <td className="px-3 py-2.5 font-bold text-slate-900">
                          {r.summary?.total_defects_detected ?? "-"}
                        </td>
                        <td className="px-3 py-2.5 font-bold text-blue-700">
                          {r.summary ? `${Math.round(r.summary.average_confidence * 100)}%` : "-"}
                        </td>
                        <td className="px-3 py-2.5 text-slate-400 text-[11px]">
                          {new Date(r.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                        </td>
                        <td className="px-3 py-2.5">
                          <button
                            type="button"
                            onClick={() => setActiveRun(r)}
                            className="rounded border border-slate-200 bg-white px-2 py-1 text-[11px] font-semibold text-blue-600 hover:bg-blue-50 transition-colors cursor-pointer"
                          >
                            {isCurrent ? "Active" : "Inspect →"}
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
