"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import SegmentMap, { type MapStyle } from "./SegmentMap";
import {
  ApiError,
  C4_API,
  CLASS_COLORS,
  CLASS_LABELS,
  api,
  isActive,
  type LabelCheck,
  type ModelInfo,
  type RoughnessClass,
  type Run,
  type SegmentCollection,
} from "./api";

const CLASSES: RoughnessClass[] = ["good", "regular", "bad"];
const STAGES = ["validate", "sync", "features", "predict", "segments", "done"];

const STATUS_STYLE: Record<string, string> = {
  queued: "bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300",
  running: "bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300",
  succeeded: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
  failed: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300",
  cancelled: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  rejected: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300",
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
  const [split, setSplit] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = e.currentTarget;
    const data = new FormData(form);
    for (const [k, v] of Array.from(data.entries())) {
      if (v instanceof File && v.size === 0) data.delete(k);   // unused optional file inputs
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
        Survey name <span className="font-normal text-zinc-500">(optional)</span>
        <input
          name="survey_id"
          placeholder="e.g. galle-road-morning"
          maxLength={128}
          className="rounded-md border border-zinc-300 bg-transparent px-2.5 py-1.5 text-sm font-normal dark:border-zinc-700"
        />
      </label>
      <div className="flex gap-4 text-xs">
        <label className="flex items-center gap-1.5">
          <input type="radio" checked={!split} onChange={() => setSplit(false)} /> One IMU file
        </label>
        <label className="flex items-center gap-1.5">
          <input type="radio" checked={split} onChange={() => setSplit(true)} /> Separate acc + gyro
        </label>
      </div>
      {split ? (
        <>
          <label className="flex flex-col gap-1 text-xs font-medium">
            Accelerometer CSV
            <input key="acc" name="acc" type="file" accept=".csv,text/csv" required className={fileCls} />
          </label>
          <label className="flex flex-col gap-1 text-xs font-medium">
            Gyroscope CSV
            <input key="gyro" name="gyro" type="file" accept=".csv,text/csv" required className={fileCls} />
          </label>
        </>
      ) : (
        <label className="flex flex-col gap-1 text-xs font-medium">
          IMU CSV <span className="font-normal text-zinc-500">timestamp_ms, ax…az (m/s²), gx…gz (rad/s)</span>
          <input key="imu" name="imu" type="file" accept=".csv,text/csv" required className={fileCls} />
        </label>
      )}
      <label className="flex flex-col gap-1 text-xs font-medium">
        GPS CSV <span className="font-normal text-zinc-500">timestamp_ms, latitude, longitude, accuracy_m, speed_mps</span>
        <input name="gps" type="file" accept=".csv,text/csv" required className={fileCls} />
      </label>
      <label className="flex flex-col gap-1 text-xs font-medium">
        Sensor manifest <span className="font-normal text-zinc-500">JSON, must include &quot;gravity_included&quot;: true</span>
        <input name="manifest" type="file" accept=".json,application/json" required className={fileCls} />
      </label>
      <label className="flex flex-col gap-1 text-xs font-medium">
        Reference labels <span className="font-normal text-zinc-500">optional · timestamp_ms, road_condition — to compare with the model</span>
        <input name="labels" type="file" accept=".csv,text/csv" className={fileCls} />
      </label>
      {error && <p className="rounded-md bg-red-50 p-2 text-xs text-red-700 dark:bg-red-950 dark:text-red-300">{error}</p>}
      <button
        type="submit"
        disabled={busy}
        className="rounded-md bg-zinc-900 px-3 py-2 text-sm font-medium text-white hover:bg-zinc-700 disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-300"
      >
        {busy ? "Uploading…" : "Upload and assess"}
      </button>
    </form>
  );
}

// ---------------------------------------------------------------------------------------------
function Evaluation({ lc }: { lc: LabelCheck }) {
  if (!lc.windows || !lc.confusion_matrix || !lc.per_class) {
    return <p className="rounded-md bg-amber-50 p-3 text-xs text-amber-900 dark:bg-amber-950 dark:text-amber-200">{lc.note}</p>;
  }
  const cm = lc.confusion_matrix.rows_true_cols_pred;
  const max = Math.max(...cm.flat(), 1);
  return (
    <div className="flex flex-col gap-3 rounded-lg border border-zinc-200 p-4 dark:border-zinc-800">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <h3 className="text-sm font-semibold">Model vs reference labels</h3>
        <span className="text-[11px] text-zinc-500">{lc.windows} windows · {lc.method}</span>
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        <Stat label="Accuracy" value={`${(100 * (lc.accuracy ?? 0)).toFixed(1)}%`} />
        <Stat label="Macro-F1" value={(lc.macro_f1 ?? 0).toFixed(3)} />
        {CLASSES.map((c) => (
          <Stat key={c} label={`F1 ${CLASS_LABELS[c]}`} value={lc.per_class![c].f1.toFixed(3)} hint={`${lc.per_class![c].support} windows`} />
        ))}
      </div>
      <div className="overflow-x-auto">
        <table className="text-xs">
          <thead>
            <tr>
              <th className="p-1.5 text-left font-normal text-zinc-500">Actual ↓ / Predicted →</th>
              {CLASSES.map((c) => (
                <th key={c} className="p-1.5 font-medium">{CLASS_LABELS[c]}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {CLASSES.map((t, i) => (
              <tr key={t}>
                <th className="p-1.5 text-left font-medium">{CLASS_LABELS[t]}</th>
                {CLASSES.map((p, j) => (
                  <td
                    key={p}
                    className="min-w-16 p-1.5 text-center tabular-nums"
                    style={{
                      background: i === j ? `rgba(26,152,80,${0.15 + 0.6 * (cm[i][j] / max)})` : cm[i][j] ? `rgba(215,48,39,${0.1 + 0.6 * (cm[i][j] / max)})` : undefined,
                    }}
                  >
                    {cm[i][j]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------
function RunDetails({ run, segments }: { run: Run; segments: SegmentCollection | null }) {
  const s = run.summary;
  const total = s ? CLASSES.reduce((a, c) => a + (s.segments.seconds_by_class[c] ?? 0), 0) : 0;
  const stageIdx = run.stage ? STAGES.indexOf(run.stage) : -1;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-lg font-semibold">{run.survey_id}</h2>
        <StatusBadge status={run.status} />
        {run.model_kind === "demo" && (
          <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800 dark:bg-amber-950 dark:text-amber-300">
            demo model · mock results
          </span>
        )}
        <span className="ml-auto text-xs text-zinc-500">{fmtDate(run.created_at)}</span>
      </div>

      {isActive(run.status) && (
        <div className="flex flex-col gap-2">
          <div className="flex gap-1">
            {STAGES.slice(0, 5).map((st, i) => (
              <div key={st} className="flex-1">
                <div className={`h-1.5 rounded-full ${i <= stageIdx ? "bg-blue-600" : "bg-zinc-200 dark:bg-zinc-800"}`} />
                <p className="mt-1 text-[10px] text-zinc-500">{st}</p>
              </div>
            ))}
          </div>
          <button
            onClick={() => api.cancel(run.run_id).catch(() => undefined)}
            className="self-start text-xs text-zinc-500 underline hover:text-zinc-800"
          >
            Cancel run
          </button>
        </div>
      )}

      {run.error && (
        <p className="rounded-md bg-red-50 p-3 text-sm text-red-800 dark:bg-red-950 dark:text-red-300">
          {run.error.message}
        </p>
      )}

      {s && (
        <>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat label="Segments" value={String(s.segments.count)} />
            <Stat label="Trip length" value={`${Math.round(s.input.duration_s)} s`} />
            <Stat label="Sensor rate" value={`${Math.round(s.input.native_rate_hz)} Hz`} />
            <Stat label="Not assessed" value={`${s.segments.unassessed_windows} windows`} hint="slower than 1 m/s" />
          </div>
          <div>
            <p className="mb-1.5 text-xs font-medium text-zinc-500">Share of assessed road (by time)</p>
            <div className="flex h-3 overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
              {CLASSES.map((c) => {
                const v = s.segments.seconds_by_class[c] ?? 0;
                return total > 0 && v > 0 ? (
                  <div key={c} style={{ width: `${(100 * v) / total}%`, background: CLASS_COLORS[c] }} title={`${CLASS_LABELS[c]} ${v}s`} />
                ) : null;
              })}
            </div>
            <div className="mt-2 flex flex-wrap gap-4 text-xs">
              {CLASSES.map((c) => (
                <span key={c} className="flex items-center gap-1.5">
                  <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: CLASS_COLORS[c] }} />
                  {CLASS_LABELS[c]}: {s.segments.by_class[c] ?? 0} segments ·{" "}
                  {total > 0 ? Math.round((100 * (s.segments.seconds_by_class[c] ?? 0)) / total) : 0}%
                </span>
              ))}
            </div>
          </div>
        </>
      )}

      {s?.label_check && <Evaluation lc={s.label_check} />}

      {run.warnings.length > 0 && (
        <ul className="flex flex-col gap-1 rounded-md bg-amber-50 p-3 text-xs text-amber-900 dark:bg-amber-950 dark:text-amber-200">
          {run.warnings.map((w) => (
            <li key={w}>⚠ {w}</li>
          ))}
        </ul>
      )}

      {run.status === "succeeded" && (
        <div className="flex flex-wrap gap-3 text-xs">
          <a className="underline" href={`${C4_API}/runs/${run.run_id}/records`} target="_blank" rel="noreferrer">
            Segment records (JSON)
          </a>
          <a className="underline" href={`${C4_API}/runs/${run.run_id}/segments`} target="_blank" rel="noreferrer">
            GeoJSON
          </a>
          <a className="underline" href={`${C4_API}/runs/${run.run_id}/windows.csv`}>
            Window details (CSV)
          </a>
          {segments && <span className="text-zinc-500">{segments.features.length} segments on map</span>}
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
      <p className="text-[11px] uppercase tracking-wide text-zinc-500">{label}</p>
      <p className="mt-0.5 text-lg font-semibold">{value}</p>
      {hint && <p className="text-[10px] text-zinc-500">{hint}</p>}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------
const NO_LABEL = "#9ca3af";

function useMapStyle(mode: "predicted" | "actual" | "match", lc: LabelCheck | undefined): MapStyle | undefined {
  const classLegend = CLASSES.map((c) => ({ label: CLASS_LABELS[c], color: CLASS_COLORS[c] }));
  const truth = (id: string) => lc?.segments?.[id]?.true_class ?? null;
  const tip = (id: string, pred: RoughnessClass) => {
    const t = truth(id);
    return t ? `Model: <b>${CLASS_LABELS[pred]}</b> · Actual: <b>${CLASS_LABELS[t]}</b>` : "No reference label";
  };
  if (!lc?.windows || mode === "predicted") {
    return lc?.windows
      ? { color: (f) => CLASS_COLORS[f.properties.roughness_class], tooltip: (f) => tip(f.properties.record_id, f.properties.roughness_class), legend: classLegend }
      : undefined;
  }
  if (mode === "actual") {
    return {
      color: (f) => {
        const t = truth(f.properties.record_id);
        return t ? CLASS_COLORS[t] : NO_LABEL;
      },
      tooltip: (f) => tip(f.properties.record_id, f.properties.roughness_class),
      legend: [...classLegend, { label: "No label", color: NO_LABEL }],
    };
  }
  return {
    color: (f) => {
      const t = truth(f.properties.record_id);
      return !t ? NO_LABEL : t === f.properties.roughness_class ? "#1a9850" : "#d73027";
    },
    tooltip: (f) => tip(f.properties.record_id, f.properties.roughness_class),
    legend: [
      { label: "Correct", color: "#1a9850" },
      { label: "Wrong", color: "#d73027" },
      { label: "No label", color: NO_LABEL },
    ],
  };
}

// ---------------------------------------------------------------------------------------------
export default function C4Dashboard() {
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState<Run | null>(null);
  const [segments, setSegments] = useState<SegmentCollection | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [demoBusy, setDemoBusy] = useState(false);
  const loadedSegmentsFor = useRef<string | null>(null);
  const [mapMode, setMapMode] = useState<"predicted" | "actual" | "match">("predicted");
  const labelCheck = selected?.summary?.label_check;
  const mapStyle = useMapStyle(mapMode, labelCheck);

  const refreshRuns = useCallback(async () => {
    try {
      setRuns(await api.runs());
      setApiError(null);
    } catch (e) {
      setApiError(errText(e));
    }
  }, []);

  useEffect(() => {
    api.model().then(setModel).catch((e) => setModelError(errText(e)));
    api
      .runs()
      .then((r) => {
        setRuns(r);
        setApiError(null);
      })
      .catch((e) => setApiError(errText(e)));
  }, []);

  // Poll the selected run while it is queued or running; load its segments once it succeeds.
  const selectedId = selected?.run_id;
  const selectedStatus = selected?.status;
  useEffect(() => {
    if (!selectedId || !selectedStatus) return;
    if (selectedStatus === "succeeded" && loadedSegmentsFor.current !== selectedId) {
      loadedSegmentsFor.current = selectedId;
      api.segments(selectedId).then(setSegments).catch(() => setSegments(null));
    }
    if (!isActive(selectedStatus)) return;
    const t = setInterval(async () => {
      try {
        const r = await api.run(selectedId);
        setSelected(r);
        if (!isActive(r.status)) refreshRuns();
      } catch {
        /* keep polling */
      }
    }, 1200);
    return () => clearInterval(t);
  }, [selectedId, selectedStatus, refreshRuns]);

  function select(run: Run) {
    if (run.run_id !== selectedId) {
      setSegments(null);
      loadedSegmentsFor.current = null;
    }
    setSelected(run);
  }

  function created(run: Run) {
    select(run);
    refreshRuns();
  }

  async function runDemo() {
    setDemoBusy(true);
    try {
      created(await api.demo());
    } catch (e) {
      setApiError(errText(e));
    } finally {
      setDemoBusy(false);
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-1 flex-col gap-6 px-4 py-6 lg:flex-row">
      {/* ------------------------------------------------------------- side panel */}
      <aside className="flex w-full shrink-0 flex-col gap-6 lg:w-80">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-zinc-500">RoadLens · Component 4</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight">Road roughness</h1>
          <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
            Upload a smartphone recording (accelerometer, gyroscope, GPS) to classify the road as good, regular or bad.
          </p>
          <div className="mt-3 text-xs">
            {model ? (
              <span
                className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 font-medium ${
                  model.kind === "demo"
                    ? "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300"
                    : "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300"
                }`}
                title={`SHA-256 ${model.model_sha256}`}
              >
                {model.kind === "demo" ? "Demo model (synthetic)" : "Production model"} · {model.model_version}
                {model.lovo_macro_f1 != null && ` · LOVO F1 ${model.lovo_macro_f1.toFixed(3)}`}
              </span>
            ) : modelError ? (
              <span className="text-red-600">Model: {modelError}</span>
            ) : (
              <span className="text-zinc-500">Loading model…</span>
            )}
          </div>
        </div>

        <section className="rounded-xl border border-zinc-200 p-4 dark:border-zinc-800">
          <h2 className="mb-3 text-sm font-semibold">New survey</h2>
          <UploadForm onCreated={created} />
          <div className="mt-4 border-t border-zinc-200 pt-3 dark:border-zinc-800">
            <button
              onClick={runDemo}
              disabled={demoBusy}
              className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm hover:bg-zinc-50 disabled:opacity-50 dark:border-zinc-700 dark:hover:bg-zinc-900"
            >
              {demoBusy ? "Starting…" : "Try the sample drive"}
            </button>
            <p className="mt-1.5 text-[11px] text-zinc-500">75 s synthetic drive in Colombo: good → regular → bad → good.</p>
          </div>
        </section>

        <section className="flex min-h-0 flex-col">
          <div className="mb-2 flex items-center justify-between">
            <h2 className="text-sm font-semibold">Recent runs</h2>
            <button onClick={refreshRuns} className="text-xs text-zinc-500 hover:text-zinc-800">
              Refresh
            </button>
          </div>
          {apiError && (
            <p className="mb-2 rounded-md bg-red-50 p-2 text-xs text-red-700 dark:bg-red-950 dark:text-red-300">
              Backend: {apiError}
            </p>
          )}
          {runs.length === 0 && !apiError && <p className="text-xs text-zinc-500">No runs yet.</p>}
          <ul className="flex flex-col gap-1.5 overflow-y-auto lg:max-h-[40vh]">
            {runs.map((r) => (
              <li key={r.run_id}>
                <button
                  onClick={() => select(r)}
                  className={`flex w-full items-center gap-2 rounded-lg border px-3 py-2 text-left text-sm ${
                    r.run_id === selectedId
                      ? "border-zinc-900 dark:border-zinc-100"
                      : "border-zinc-200 hover:border-zinc-400 dark:border-zinc-800"
                  }`}
                >
                  <span className="min-w-0 flex-1">
                    <span className="block truncate font-medium">{r.survey_id}</span>
                    <span className="block text-[11px] text-zinc-500">
                      {fmtDate(r.created_at)} · {r.source}
                      {r.summary ? ` · ${r.summary.segments.count} segments` : ""}
                    </span>
                  </span>
                  <StatusBadge status={r.status} />
                </button>
              </li>
            ))}
          </ul>
        </section>
      </aside>

      {/* ------------------------------------------------------------- main area */}
      <main className="flex min-w-0 flex-1 flex-col gap-4">
        {labelCheck?.windows ? (
          <div className="flex items-center gap-1 self-start rounded-lg border border-zinc-200 p-1 text-xs dark:border-zinc-800">
            {(["predicted", "actual", "match"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setMapMode(m)}
                className={`rounded-md px-3 py-1.5 ${mapMode === m ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900" : "hover:bg-zinc-100 dark:hover:bg-zinc-900"}`}
              >
                {m === "predicted" ? "Model prediction" : m === "actual" ? "Actual label" : "Correct / wrong"}
              </button>
            ))}
          </div>
        ) : null}
        <div className="h-[55vh] min-h-[360px]">
          <SegmentMap data={segments} style={mapStyle} />
        </div>
        <section className="rounded-xl border border-zinc-200 p-4 dark:border-zinc-800">
          {selected ? (
            <RunDetails run={selected} segments={segments} />
          ) : (
            <p className="text-sm text-zinc-500">
              Upload a survey or try the sample drive. Results appear on the map, coloured by road condition.
            </p>
          )}
        </section>
      </main>
    </div>
  );
}
