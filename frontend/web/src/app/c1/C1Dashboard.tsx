"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import DefectMap from "./DefectMap";
import {
  ApiError,
  C1_API,
  CLASS_COLORS,
  CLASS_LABELS,
  api,
  isActive,
  type DefectCollection,
  type DefectProperties,
  type ModelInfo,
  type Run,
} from "./api";

const STAGES = ["extract_frames", "predict_b0", "crop_rois", "export_contracts", "done"];

const STATUS_STYLE: Record<string, string> = {
  queued: "bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300",
  running: "bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300",
  succeeded: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
  failed: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300",
  cancelled: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
};

function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${STATUS_STYLE[status] ?? ""}`}>
      {status}
    </span>
  );
}

function fmtDate(iso: string) {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function errText(e: unknown) {
  return e instanceof ApiError || e instanceof Error ? e.message : String(e);
}

// ---------------------------------------------------------------------------------------------
function UploadForm({ onCreated }: { onCreated: (run: Run) => void }) {
  const [inputType, setInputType] = useState<"video" | "images">("video");
  const [busy, setBusy] = useState(false);
  const [conf, setConf] = useState(0.25);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = e.currentTarget;
    const data = new FormData(form);

    for (const [k, v] of Array.from(data.entries())) {
      if (v instanceof File && v.size === 0) data.delete(k);
    }
    if (!String(data.get("survey_id") ?? "").trim()) data.delete("survey_id");

    setBusy(true);
    setError(null);
    try {
      onCreated(await api.upload(data));
      form.reset();
    } catch (err) {
      setError(errText(err));
    } finally {
      setBusy(false);
    }
  }

  const fileCls =
    "block w-full text-xs file:mr-3 file:rounded-md file:border-0 file:bg-zinc-100 file:px-3 file:py-1.5 " +
    "file:text-xs file:font-medium hover:file:bg-zinc-200 dark:file:bg-zinc-800 dark:hover:file:bg-zinc-700";

  return (
    <form onSubmit={submit} className="flex flex-col gap-3">
      <label className="flex flex-col gap-1 text-xs font-medium">
        Survey Title <span className="font-normal text-zinc-500">(optional)</span>
        <input
          name="survey_id"
          placeholder="e.g. Galle-Road-Survey-Pilot"
          maxLength={128}
          className="rounded-md border border-zinc-300 bg-transparent px-2.5 py-1.5 text-sm font-normal dark:border-zinc-700"
        />
      </label>

      <div className="flex gap-4 text-xs">
        <label className="flex items-center gap-1.5">
          <input
            type="radio"
            checked={inputType === "video"}
            onChange={() => setInputType("video")}
          />
          Road Video (.mp4)
        </label>
        <label className="flex items-center gap-1.5">
          <input
            type="radio"
            checked={inputType === "images"}
            onChange={() => setInputType("images")}
          />
          Image Batch (.jpg, .png)
        </label>
      </div>

      {inputType === "video" ? (
        <label className="flex flex-col gap-1 text-xs font-medium">
          Survey Video File
          <span className="font-normal text-zinc-500">Continuous smartphone road recording</span>
          <input
            name="video"
            type="file"
            accept="video/mp4,video/quicktime,video/*"
            required
            className={fileCls}
          />
        </label>
      ) : (
        <label className="flex flex-col gap-1 text-xs font-medium">
          Road Survey Images
          <span className="font-normal text-zinc-500">Multiple image frames</span>
          <input
            name="images"
            type="file"
            accept="image/*"
            multiple
            required
            className={fileCls}
          />
        </label>
      )}

      <label className="flex flex-col gap-1 text-xs font-medium">
        GPS Trace CSV <span className="font-normal text-zinc-500">optional · timestamp_ms, latitude, longitude</span>
        <input name="gps" type="file" accept=".csv,text/csv" className={fileCls} />
      </label>

      <div className="flex flex-col gap-1 text-xs font-medium">
        <div className="flex justify-between">
          <span>Confidence Threshold</span>
          <span className="text-zinc-500 font-normal">{(conf * 100).toFixed(0)}%</span>
        </div>
        <input
          name="conf"
          type="range"
          min="0.10"
          max="0.80"
          step="0.05"
          value={conf}
          onChange={(e) => setConf(parseFloat(e.target.value))}
          className="accent-zinc-900 dark:accent-zinc-100"
        />
      </div>

      {error && (
        <p className="rounded-md bg-red-50 p-2 text-xs text-red-700 dark:bg-red-950 dark:text-red-300">
          {error}
        </p>
      )}

      <button
        type="submit"
        disabled={busy}
        className="rounded-md bg-zinc-900 px-3 py-2 text-sm font-medium text-white hover:bg-zinc-700 disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-300"
      >
        {busy ? "Uploading Survey…" : "Run B0 Road Damage Detection"}
      </button>
    </form>
  );
}

// ---------------------------------------------------------------------------------------------
function StatCard({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
      <p className="text-[11px] uppercase tracking-wide text-zinc-500">{label}</p>
      <p className="mt-0.5 text-lg font-semibold">{value}</p>
      {hint && <p className="text-[10px] text-zinc-500">{hint}</p>}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------
export default function C1Dashboard() {
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState<Run | null>(null);
  const [defects, setDefects] = useState<DefectCollection | null>(null);
  const [selectedDefect, setSelectedDefect] = useState<DefectProperties | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [demoBusy, setDemoBusy] = useState(false);
  const loadedDefectsFor = useRef<string | null>(null);

  const refreshRuns = useCallback(async () => {
    try {
      setRuns(await api.runs());
      setApiError(null);
    } catch (e) {
      setApiError(errText(e));
    }
  }, []);

  useEffect(() => {
    api.model().then(setModel).catch((e) => setApiError(errText(e)));
    api.runs().then((r) => {
      setRuns(r);
      if (r.length > 0) {
        setSelected(r[0]);
      }
    }).catch((e) => setApiError(errText(e)));
  }, []);

  // Poll selected run if active; load defects when succeeded
  const selectedId = selected?.run_id;
  const selectedStatus = selected?.status;
  useEffect(() => {
    if (!selectedId || !selectedStatus) return;
    if (selectedStatus === "succeeded" && loadedDefectsFor.current !== selectedId) {
      loadedDefectsFor.current = selectedId;
      api.defects(selectedId).then(setDefects).catch(() => setDefects(null));
    }
    if (!isActive(selectedStatus)) return;

    const t = setInterval(async () => {
      try {
        const r = await api.run(selectedId);
        setSelected(r);
        if (!isActive(r.status)) refreshRuns();
      } catch {
        // keep polling
      }
    }, 1200);
    return () => clearInterval(t);
  }, [selectedId, selectedStatus, refreshRuns]);

  function selectRun(run: Run) {
    if (run.run_id !== selectedId) {
      setDefects(null);
      setSelectedDefect(null);
      loadedDefectsFor.current = null;
    }
    setSelected(run);
  }

  function handleRunCreated(run: Run) {
    selectRun(run);
    refreshRuns();
  }

  async function runDemo() {
    setDemoBusy(true);
    try {
      const demoRun = await api.demo();
      handleRunCreated(demoRun);
    } catch (e) {
      setApiError(errText(e));
    } finally {
      setDemoBusy(false);
    }
  }

  const summary = selected?.summary;
  const stageIdx = selected?.stage ? STAGES.indexOf(selected.stage) : -1;
  const totalDefects = summary?.total_defects_detected ?? 0;

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-1 flex-col gap-6 px-4 py-6 lg:flex-row">
      {/* ------------------------------------------------------------- Left Sidebar */}
      <aside className="flex w-full shrink-0 flex-col gap-6 lg:w-80">
        <div>
          <div className="flex items-center gap-2">
            <p className="text-xs font-semibold uppercase tracking-wider text-blue-600 dark:text-blue-400">
              RoadLens · Component 1
            </p>
            <span className="rounded-full bg-blue-100 px-2 py-0.5 text-[10px] font-semibold text-blue-800 dark:bg-blue-950 dark:text-blue-300">
              50% Milestone
            </span>
          </div>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight">Road Damage Detection</h1>
          <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-400">
            Automated road distress localization using YOLOv8m Base Model (B0), frame extraction, and geolocated mapping.
          </p>

          <div className="mt-3 flex flex-col gap-1.5 text-xs">
            {model ? (
              <span
                className="inline-flex items-center gap-1.5 rounded-full bg-emerald-100 px-2.5 py-1 font-medium text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300"
                title={`Weights SHA: ${model.weights_sha256}`}
              >
                <span className="h-2 w-2 rounded-full bg-emerald-600"></span>
                {model.model_version} · mAP@50 66.2%
              </span>
            ) : (
              <span className="text-zinc-500">Loading model info…</span>
            )}
          </div>
        </div>

        {/* Upload Card */}
        <section className="rounded-xl border border-zinc-200 p-4 dark:border-zinc-800">
          <h2 className="mb-2 text-sm font-semibold">New Road Survey</h2>
          <UploadForm onCreated={handleRunCreated} />

          <div className="mt-4 border-t border-zinc-200 pt-3 dark:border-zinc-800">
            <button
              onClick={runDemo}
              disabled={demoBusy}
              className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm font-medium hover:bg-zinc-50 disabled:opacity-50 dark:border-zinc-700 dark:hover:bg-zinc-900"
            >
              {demoBusy ? "Starting Demo Survey…" : "Try Sample Survey (Colombo)"}
            </button>
            <p className="mt-1.5 text-[11px] text-zinc-500">
              Deterministic Colombo Galle Road survey with potholes & crack instances.
            </p>
          </div>
        </section>

        {/* Recent Runs */}
        <section className="flex min-h-0 flex-col">
          <div className="mb-2 flex items-center justify-between">
            <h2 className="text-sm font-semibold">Recent Surveys</h2>
            <button onClick={refreshRuns} className="text-xs text-zinc-500 hover:text-zinc-800">
              Refresh
            </button>
          </div>
          {apiError && (
            <p className="mb-2 rounded-md bg-red-50 p-2 text-xs text-red-700 dark:bg-red-950 dark:text-red-300">
              {apiError}
            </p>
          )}
          {runs.length === 0 && !apiError && (
            <p className="text-xs text-zinc-500">No surveys processed yet.</p>
          )}
          <ul className="flex flex-col gap-1.5 overflow-y-auto lg:max-h-[35vh]">
            {runs.map((r) => (
              <li key={r.run_id}>
                <button
                  onClick={() => selectRun(r)}
                  className={`flex w-full items-center gap-2 rounded-lg border px-3 py-2 text-left text-sm ${
                    r.run_id === selectedId
                      ? "border-zinc-900 dark:border-zinc-100 bg-zinc-50/50 dark:bg-zinc-900/50"
                      : "border-zinc-200 hover:border-zinc-400 dark:border-zinc-800"
                  }`}
                >
                  <span className="min-w-0 flex-1">
                    <span className="block truncate font-medium text-xs">{r.survey_id}</span>
                    <span className="block text-[10px] text-zinc-500">
                      {fmtDate(r.created_at)}
                      {r.summary ? ` · ${r.summary.total_defects_detected} defects` : ""}
                    </span>
                  </span>
                  <StatusBadge status={r.status} />
                </button>
              </li>
            ))}
          </ul>
        </section>
      </aside>

      {/* ------------------------------------------------------------- Main Content Area */}
      <main className="flex min-w-0 flex-1 flex-col gap-5">
        {selected && (
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-lg font-semibold">{selected.survey_id}</h2>
            <StatusBadge status={selected.status} />
            <span className="ml-auto text-xs text-zinc-500">{fmtDate(selected.created_at)}</span>
          </div>
        )}

        {/* Progress Bar */}
        {selected && isActive(selected.status) && (
          <div className="flex flex-col gap-2 rounded-lg border border-blue-200 bg-blue-50/50 p-3 dark:border-blue-900 dark:bg-blue-950/20">
            <div className="flex justify-between text-xs font-medium text-blue-900 dark:text-blue-300">
              <span>Processing stage: {selected.stage || "initializing"}</span>
              {selected.progress.total > 0 && (
                <span>
                  {selected.progress.completed} / {selected.progress.total}
                </span>
              )}
            </div>
            <div className="flex gap-1">
              {STAGES.slice(0, 4).map((st, i) => (
                <div key={st} className="flex-1">
                  <div
                    className={`h-1.5 rounded-full ${
                      i <= stageIdx ? "bg-blue-600" : "bg-zinc-200 dark:bg-zinc-800"
                    }`}
                  />
                  <p className="mt-1 text-[10px] text-zinc-500 capitalize">{st.replace("_", " ")}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Summary Metric Stats */}
        {summary && (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <StatCard label="Frames Analyzed" value={String(summary.total_frames_processed)} />
            <StatCard label="Defects Detected" value={String(summary.total_defects_detected)} />
            <StatCard
              label="Avg Confidence"
              value={`${(summary.average_confidence * 100).toFixed(1)}%`}
              hint="Confidence threshold >= 25%"
            />
            <StatCard
              label="B0 Baseline mAP"
              value={`${(summary.b0_benchmark_mAP50 * 100).toFixed(1)}%`}
              hint="Trained on RDD2022"
            />
          </div>
        )}

        {/* Class Distribution Bar */}
        {summary && totalDefects > 0 && (
          <div className="rounded-xl border border-zinc-200 p-3 dark:border-zinc-800">
            <p className="mb-1 text-xs font-medium text-zinc-500">Damage Classification Breakdown</p>
            <div className="flex h-3 overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
              {Object.entries(summary.counts_by_class).map(([cname, count]) => {
                if (count === 0) return null;
                const pct = (count / totalDefects) * 100;
                return (
                  <div
                    key={cname}
                    style={{ width: `${pct}%`, backgroundColor: CLASS_COLORS[cname] }}
                    title={`${CLASS_LABELS[cname]}: ${count} (${pct.toFixed(0)}%)`}
                  />
                );
              })}
            </div>
            <div className="mt-2 flex flex-wrap gap-4 text-xs">
              {Object.entries(summary.counts_by_class).map(([cname, count]) => (
                <span key={cname} className="flex items-center gap-1.5">
                  <span
                    className="inline-block h-2.5 w-2.5 rounded-sm"
                    style={{ backgroundColor: CLASS_COLORS[cname] }}
                  />
                  {CLASS_LABELS[cname]}: {count}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Interactive Map View */}
        <div className="h-[46vh] min-h-[340px]">
          <DefectMap
            data={defects}
            selectedId={selectedDefect?.defect_id}
            onSelect={(d) => setSelectedDefect(d)}
          />
        </div>

        {/* Selected Defect Detail & C2 ROI Crop Handoff (FR15) */}
        {selectedDefect && selected && (
          <section className="rounded-xl border-2 border-blue-500/40 bg-zinc-50/50 p-4 dark:border-blue-500/30 dark:bg-zinc-900/50">
            <div className="flex items-center justify-between pb-3 border-b border-zinc-200 dark:border-zinc-800">
              <div>
                <span className="text-xs uppercase tracking-wider font-semibold text-blue-600 dark:text-blue-400">
                  Observation Inspector · FR14 / FR15
                </span>
                <h3 className="text-base font-semibold">{selectedDefect.display_name}</h3>
              </div>
              <button
                onClick={() => setSelectedDefect(null)}
                className="text-xs text-zinc-500 hover:text-zinc-800"
              >
                Close ✕
              </button>
            </div>

            <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Annotated Full Frame */}
              <div>
                <p className="text-xs font-medium text-zinc-600 dark:text-zinc-400 mb-1.5">
                  Annotated Frame Overlay (B0 Localization)
                </p>
                <div className="relative aspect-video w-full overflow-hidden rounded-lg border border-zinc-300 dark:border-zinc-700 bg-black">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={api.artifactUrl(selected.run_id, "frames", selectedDefect.annotated_frame)}
                    alt="Annotated detection frame"
                    className="h-full w-full object-contain"
                  />
                </div>
              </div>

              {/* Cropped Defect ROI (Ready for C2 Segmentation) */}
              <div className="flex flex-col">
                <div className="flex items-center justify-between mb-1.5">
                  <p className="text-xs font-medium text-zinc-600 dark:text-zinc-400">
                    Cropped Defect ROI
                  </p>
                  <span className="rounded bg-purple-100 px-1.5 py-0.5 text-[10px] font-semibold text-purple-800 dark:bg-purple-950 dark:text-purple-300">
                    Upstream Input to C2
                  </span>
                </div>

                <div className="flex items-center justify-center h-48 w-full overflow-hidden rounded-lg border border-zinc-300 dark:border-zinc-700 bg-zinc-950 p-2">
                  {selectedDefect.crop_image ? (
                    /* eslint-disable-next-line @next/next/no-img-element */
                    <img
                      src={api.artifactUrl(selected.run_id, "crops", selectedDefect.crop_image)}
                      alt="Cropped defect region"
                      className="max-h-full max-w-full object-contain rounded"
                    />
                  ) : (
                    <span className="text-xs text-zinc-500">Crop unavailable</span>
                  )}
                </div>

                <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                  <div className="rounded bg-white p-2 border border-zinc-200 dark:bg-zinc-800 dark:border-zinc-700">
                    <span className="text-[10px] text-zinc-500 block">Confidence</span>
                    <span className="font-semibold text-sm">
                      {(selectedDefect.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="rounded bg-white p-2 border border-zinc-200 dark:bg-zinc-800 dark:border-zinc-700">
                    <span className="text-[10px] text-zinc-500 block">Frame Timestamp</span>
                    <span className="font-semibold text-sm">
                      {(selectedDefect.timestamp_ms / 1000).toFixed(1)}s
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Detected Defects Table (FR14) */}
        {defects && defects.features.length > 0 && (
          <section className="rounded-xl border border-zinc-200 p-4 dark:border-zinc-800">
            <h3 className="mb-2 text-sm font-semibold">
              Detected Defects ({defects.features.length})
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-zinc-200 text-left text-zinc-500 dark:border-zinc-800">
                    <th className="p-2">Defect ID</th>
                    <th className="p-2">Damage Category</th>
                    <th className="p-2">Confidence</th>
                    <th className="p-2">Time</th>
                    <th className="p-2">Bounding Box</th>
                    <th className="p-2">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {defects.features.map((feat) => {
                    const p = feat.properties;
                    const isSelected = selectedDefect?.defect_id === p.defect_id;
                    return (
                      <tr
                        key={p.defect_id}
                        onClick={() => setSelectedDefect(p)}
                        className={`cursor-pointer border-b border-zinc-100 hover:bg-zinc-50 dark:border-zinc-800/50 dark:hover:bg-zinc-900 ${
                          isSelected ? "bg-blue-50/60 dark:bg-blue-950/40" : ""
                        }`}
                      >
                        <td className="p-2 font-mono text-zinc-600 dark:text-zinc-400">
                          {p.defect_id}
                        </td>
                        <td className="p-2 font-medium">
                          <span className="flex items-center gap-1.5">
                            <span
                              className="h-2 w-2 rounded-full"
                              style={{ backgroundColor: CLASS_COLORS[p.damage_class] }}
                            />
                            {p.display_name}
                          </span>
                        </td>
                        <td className="p-2 tabular-nums">{(p.confidence * 100).toFixed(1)}%</td>
                        <td className="p-2 tabular-nums">{(p.timestamp_ms / 1000).toFixed(1)}s</td>
                        <td className="p-2 font-mono text-[11px] text-zinc-500">
                          [{p.bbox.map((v) => Math.round(v)).join(", ")}]
                        </td>
                        <td className="p-2">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedDefect(p);
                            }}
                            className="text-xs text-blue-600 underline dark:text-blue-400"
                          >
                            Inspect ROI
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Contract Exports */}
            {selected && selected.status === "succeeded" && (
              <div className="mt-4 flex flex-wrap gap-4 border-t border-zinc-200 pt-3 text-xs dark:border-zinc-800">
                <a
                  className="text-blue-600 underline dark:text-blue-400"
                  href={`${C1_API}/runs/${selected.run_id}/defects`}
                  target="_blank"
                  rel="noreferrer"
                >
                  Download GeoJSON (Map Layer)
                </a>
                <a
                  className="text-blue-600 underline dark:text-blue-400"
                  href={`${C1_API}/runs/${selected.run_id}/records`}
                  target="_blank"
                  rel="noreferrer"
                >
                  Download C1-C2 Records (JSONL Contract)
                </a>
              </div>
            )}
          </section>
        )}

        {/* 50% Milestone & Research Context Card */}
        <section className="rounded-xl border border-zinc-200 bg-zinc-50/50 p-4 dark:border-zinc-800 dark:bg-zinc-900/30">
          <h3 className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">
            C1 Research Evaluation: Base Model B0 vs. Next 50% Roadmap
          </h3>
          <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-400">
            For Progress Presentation 1 (50% Milestone), the baseline detector (B0) is validated on the international RDD2022 dataset.
            The remaining 50% addresses domain shift on local Sri Lankan roads through temporal consistency pseudo-labeling.
          </p>

          <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* B0 Performance Table */}
            <div className="rounded-lg border border-zinc-200 bg-white p-3 dark:border-zinc-800 dark:bg-zinc-900">
              <p className="text-xs font-semibold text-zinc-700 dark:text-zinc-300 mb-2">
                B0 Baseline Validation Metrics (RDD2022)
              </p>
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b text-zinc-500">
                    <th className="py-1 text-left">Class</th>
                    <th className="py-1 text-right">AP@50</th>
                    <th className="py-1 text-right">Precision</th>
                    <th className="py-1 text-right">Recall</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800">
                  <tr>
                    <td className="py-1">Pothole (D40)</td>
                    <td className="py-1 text-right font-semibold text-emerald-600">78.4%</td>
                    <td className="py-1 text-right">70.4%</td>
                    <td className="py-1 text-right">75.0%</td>
                  </tr>
                  <tr>
                    <td className="py-1">Alligator Crack (D20)</td>
                    <td className="py-1 text-right font-semibold">67.2%</td>
                    <td className="py-1 text-right">69.0%</td>
                    <td className="py-1 text-right">61.0%</td>
                  </tr>
                  <tr>
                    <td className="py-1">Longitudinal Crack (D00)</td>
                    <td className="py-1 text-right font-semibold">60.3%</td>
                    <td className="py-1 text-right">64.5%</td>
                    <td className="py-1 text-right">56.7%</td>
                  </tr>
                  <tr>
                    <td className="py-1">Transverse Crack (D10)</td>
                    <td className="py-1 text-right font-semibold">58.7%</td>
                    <td className="py-1 text-right">62.0%</td>
                    <td className="py-1 text-right">55.2%</td>
                  </tr>
                  <tr className="font-semibold bg-zinc-50 dark:bg-zinc-800/50">
                    <td className="py-1">Overall Detector</td>
                    <td className="py-1 text-right text-emerald-600">66.2%</td>
                    <td className="py-1 text-right">66.5%</td>
                    <td className="py-1 text-right">62.0%</td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Next 50% Roadmap */}
            <div className="rounded-lg border border-zinc-200 bg-white p-3 dark:border-zinc-800 dark:bg-zinc-900">
              <p className="text-xs font-semibold text-zinc-700 dark:text-zinc-300 mb-2">
                Planned Second 50% Scope (Progress 2 & Final)
              </p>
              <ul className="flex flex-col gap-1.5 text-xs text-zinc-600 dark:text-zinc-400">
                <li className="flex items-start gap-1.5">
                  <span className="text-blue-500 font-bold">•</span>
                  <span><strong>WBS 4.0:</strong> Complete local Sri Lankan road dataset annotation & route splitting.</span>
                </li>
                <li className="flex items-start gap-1.5">
                  <span className="text-blue-500 font-bold">•</span>
                  <span><strong>WBS 6.0–7.0:</strong> Evaluate Few-Label (B1) & Image Pseudo-Label (B2) Baselines.</span>
                </li>
                <li className="flex items-start gap-1.5">
                  <span className="text-blue-500 font-bold">•</span>
                  <span><strong>WBS 8.0–9.0:</strong> Multi-frame Temporal Association & Reliability Scoring Mechanism (FR7, FR8).</span>
                </li>
                <li className="flex items-start gap-1.5">
                  <span className="text-blue-500 font-bold">•</span>
                  <span><strong>WBS 10.0 & 14.0:</strong> Temporal-Guided Domain Adaptation & Duplicate Defect Consolidation (FR9).</span>
                </li>
              </ul>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
