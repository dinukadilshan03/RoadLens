"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import RoadStrip from "./RoadStrip";
import SegmentMap, { type MapStyle } from "./SegmentMap";
import {
  ApiError,
  C4_API,
  CLASS_COLORS,
  CLASS_LABELS,
  NO_LABEL_COLOR,
  api,
  isActive,
  type LabelCheck,
  type ModelInfo,
  type RoughnessClass,
  type Run,
  type RunStatus,
  type SegmentCollection,
} from "./api";
import { fmtDuration, fmtKm, roadProfile, type RoadProfile } from "./road";

const CLASSES: RoughnessClass[] = ["good", "regular", "bad"];

const STAGES: { key: string; label: string }[] = [
  { key: "validate", label: "Checking the files" },
  { key: "sync", label: "Aligning the sensors" },
  { key: "features", label: "Measuring vibration" },
  { key: "predict", label: "Classifying the road" },
  { key: "segments", label: "Building road segments" },
];

const STATUS_TEXT: Record<RunStatus, string> = {
  queued: "Waiting",
  running: "Processing",
  succeeded: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
  rejected: "Rejected",
};

function fmtDate(iso: string) {
  return new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function errText(e: unknown) {
  return e instanceof ApiError || e instanceof Error ? e.message : String(e);
}

const btnPrimary =
  "inline-flex items-center justify-center rounded-md bg-[var(--c4-sign)] px-4 py-2.5 text-sm font-semibold text-[var(--c4-sign-ink)] hover:brightness-110 disabled:opacity-50";
const btnQuiet =
  "inline-flex items-center justify-center rounded-md border border-[var(--c4-line)] px-4 py-2.5 text-sm font-medium hover:bg-[var(--c4-bg)] disabled:opacity-50";

function Notice({ tone, children }: { tone: "danger" | "warn"; children: React.ReactNode }) {
  return (
    <div
      role={tone === "danger" ? "alert" : undefined}
      className={`rounded-md border-l-4 px-3 py-2.5 text-sm ${
        tone === "danger" ? "border-[var(--c4-bad)] bg-[var(--c4-danger-bg)]" : "border-[var(--c4-regular)] bg-[var(--c4-warn-bg)]"
      }`}
    >
      {children}
    </div>
  );
}

// --------------------------------------------------------------------------------------------- upload
function FileSlot({ name, title, hint, accept, required }: { name: string; title: string; hint: string; accept: string; required?: boolean }) {
  const [file, setFile] = useState<string | null>(null);
  return (
    <label className="block cursor-pointer">
      <input
        name={name}
        type="file"
        accept={accept}
        required={required}
        className="peer sr-only"
        onChange={(e) => setFile(e.currentTarget.files?.[0]?.name ?? null)}
      />
      <span
        className={`flex items-center gap-3 rounded-md border px-3 py-2.5 peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-[var(--c4-sign)] ${
          file ? "border-[var(--c4-good)] bg-[var(--c4-surface)]" : "border-dashed border-[var(--c4-line)] hover:border-[var(--c4-muted)]"
        }`}
      >
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-semibold">
            {title}
            {!required && <span className="font-normal text-[var(--c4-muted)]"> (optional)</span>}
          </span>
          <span className={`block text-xs text-[var(--c4-muted)] ${file ? "truncate" : ""}`}>{file ?? hint}</span>
        </span>
        <span className="shrink-0 text-xs font-semibold text-[var(--c4-sign)]">{file ? "Change" : "Choose file"}</span>
      </span>
    </label>
  );
}

function UploadForm({ onCreated, onDemo, demoBusy }: { onCreated: (run: Run) => void; onDemo: () => void; demoBusy: boolean }) {
  const [split, setSplit] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formKey, setFormKey] = useState(0);

  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const data = new FormData(e.currentTarget);
    for (const [k, v] of Array.from(data.entries())) {
      if (v instanceof File && v.size === 0) data.delete(k); // unused optional file inputs
    }
    if (!String(data.get("survey_id") ?? "").trim()) data.delete("survey_id");
    setBusy(true);
    setError(null);
    try {
      onCreated(await api.upload(data));
      setFormKey((k) => k + 1); // clears the chosen files
    } catch (err) {
      setError(errText(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form key={formKey} onSubmit={submit} className="flex flex-col gap-3">
      <label className="flex flex-col gap-1">
        <span className="text-sm font-semibold">Survey name <span className="font-normal text-[var(--c4-muted)]">(optional)</span></span>
        <input
          name="survey_id"
          placeholder="Galle Road, morning"
          maxLength={128}
          className="rounded-md border border-[var(--c4-line)] bg-[var(--c4-surface)] px-3 py-2 text-sm placeholder:text-[var(--c4-muted)]"
        />
      </label>

      <fieldset className="flex min-w-0 flex-col gap-2">
        <legend className="mb-1.5 text-sm font-semibold">Motion sensors</legend>
        <div className="mb-1 flex rounded-md border border-[var(--c4-line)] p-0.5 text-xs" role="radiogroup">
          {[
            [false, "One file"],
            [true, "Two files"],
          ].map(([v, text]) => (
            <button
              key={String(v)}
              type="button"
              role="radio"
              aria-checked={split === v}
              onClick={() => setSplit(v as boolean)}
              className={`flex-1 rounded px-2 py-1.5 ${split === v ? "bg-[var(--c4-ink)] text-[var(--c4-bg)]" : "text-[var(--c4-muted)]"}`}
            >
              {text as string}
            </button>
          ))}
        </div>
        {split ? (
          <>
            <FileSlot key="acc" name="acc" title="Accelerometer" hint="CSV in m/s², gravity included" accept=".csv,text/csv" required />
            <FileSlot key="gyro" name="gyro" title="Gyroscope" hint="CSV in rad/s" accept=".csv,text/csv" required />
          </>
        ) : (
          <FileSlot key="imu" name="imu" title="Accelerometer and gyroscope" hint="CSV with acceleration (m/s²) and rotation (rad/s)" accept=".csv,text/csv" required />
        )}
      </fieldset>
      <FileSlot name="gps" title="GPS track" hint="CSV with latitude, longitude and speed" accept=".csv,text/csv" required />
      <FileSlot name="manifest" title="Sensor manifest" hint="JSON describing the phone and its sensors" accept=".json,application/json" required />
      <FileSlot name="labels" title="Reference labels" hint="CSV of known road condition, to check the model" accept=".csv,text/csv" />

      {error && <Notice tone="danger">{error}</Notice>}
      <button type="submit" disabled={busy} className={btnPrimary}>
        {busy ? "Uploading…" : "Assess the road"}
      </button>
      <button type="button" onClick={onDemo} disabled={demoBusy} className={btnQuiet}>
        {demoBusy ? "Starting…" : "Try a sample drive"}
      </button>
      <p className="text-xs text-[var(--c4-muted)]">The sample is a 75-second synthetic drive in Colombo.</p>
    </form>
  );
}

// --------------------------------------------------------------------------------------------- survey list
function ConditionBar({ run, className = "h-1.5" }: { run: Run; className?: string }) {
  const s = run.summary?.segments.seconds_by_class;
  const total = s ? CLASSES.reduce((a, c) => a + (s[c] ?? 0), 0) : 0;
  if (!s || !total) return <div className={`${className} bg-[var(--c4-line)]`} />;
  return (
    <div className={`flex ${className} overflow-hidden`}>
      {CLASSES.map((c) => (s[c] ? <div key={c} style={{ width: `${(100 * s[c]) / total}%`, background: CLASS_COLORS[c] }} /> : null))}
    </div>
  );
}

function SurveyList({ runs, selectedId, onSelect }: { runs: Run[]; selectedId?: string; onSelect: (r: Run) => void }) {
  if (!runs.length) return <p className="text-sm text-[var(--c4-muted)]">No surveys yet. Upload a recording to start.</p>;
  return (
    <ul className="flex flex-col">
      {runs.map((r) => {
        const on = r.run_id === selectedId;
        return (
          <li key={r.run_id}>
            <button
              onClick={() => onSelect(r)}
              aria-current={on ? "true" : undefined}
              className={`w-full border-l-4 py-2.5 pl-3 pr-2 text-left ${
                on ? "border-[var(--c4-sign)] bg-[var(--c4-surface)]" : "border-transparent hover:bg-[var(--c4-surface)]"
              }`}
            >
              <span className="flex items-baseline gap-2">
                <span className="min-w-0 flex-1 truncate text-sm font-semibold">{r.survey_id}</span>
                {r.status !== "succeeded" && (
                  <span className={`shrink-0 text-xs ${r.status === "failed" || r.status === "rejected" ? "text-[var(--c4-bad)]" : "text-[var(--c4-muted)]"}`}>
                    {STATUS_TEXT[r.status]}
                  </span>
                )}
              </span>
              <span className="mt-0.5 block text-xs text-[var(--c4-muted)]">
                {fmtDate(r.created_at)}
                {r.summary ? `, ${fmtDuration(r.summary.input.duration_s)}` : ""}
              </span>
              {r.status === "succeeded" && <ConditionBar run={r} className="mt-2 h-1" />}
            </button>
          </li>
        );
      })}
    </ul>
  );
}

// --------------------------------------------------------------------------------------------- results
function Progress({ run }: { run: Run }) {
  const idx = run.stage ? STAGES.findIndex((s) => s.key === run.stage) : -1;
  return (
    <section className="flex flex-col gap-4 rounded-md border border-[var(--c4-line)] bg-[var(--c4-surface)] p-5">
      <p className="text-base font-semibold">{run.status === "queued" ? "Waiting to start" : "Assessing the road"}</p>
      <ol className="flex flex-col gap-2">
        {STAGES.map((s, i) => {
          const state = i < idx ? "done" : i === idx ? "now" : "next";
          return (
            <li key={s.key} className="flex items-center gap-3 text-sm">
              <span
                className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${
                  state === "done"
                    ? "bg-[var(--c4-good)] text-white"
                    : state === "now"
                      ? "bg-[var(--c4-sign)] text-[var(--c4-sign-ink)]"
                      : "border border-[var(--c4-line)] text-[var(--c4-muted)]"
                }`}
              >
                {state === "done" ? "✓" : i + 1}
              </span>
              <span className={state === "next" ? "text-[var(--c4-muted)]" : ""}>{s.label}</span>
            </li>
          );
        })}
      </ol>
      <button onClick={() => api.cancel(run.run_id).catch(() => undefined)} className="self-start text-sm text-[var(--c4-muted)] underline">
        Cancel
      </button>
    </section>
  );
}

function ConditionSummary({ profile, run }: { profile: RoadProfile; run: Run }) {
  const total = profile.totalKm || 1;
  return (
    <section aria-label="Road condition totals" className="grid grid-cols-3 border-y border-[var(--c4-line)]">
      {CLASSES.map((c, i) => (
        <div key={c} className={`px-3 py-4 sm:px-4 ${i ? "border-l border-[var(--c4-line)]" : ""}`}>
          <div className="flex items-center gap-2 text-sm font-semibold">
            <span className="inline-block h-3 w-3" style={{ background: CLASS_COLORS[c] }} />
            {CLASS_LABELS[c]}
          </div>
          <p className="mt-1 text-2xl font-bold tracking-tight sm:text-3xl">
            {fmtKm(profile.kmByClass[c])}
            <span className="ml-1 text-base font-semibold text-[var(--c4-muted)]">km</span>
          </p>
          <p className="text-sm text-[var(--c4-muted)]">
            {Math.round((100 * profile.kmByClass[c]) / total)}% of the route, {run.summary?.segments.by_class[c] ?? 0} segments
          </p>
        </div>
      ))}
    </section>
  );
}

function Evaluation({ lc }: { lc: LabelCheck }) {
  if (!lc.windows || !lc.confusion_matrix || !lc.per_class) return <Notice tone="warn">{lc.note}</Notice>;
  const cm = lc.confusion_matrix.rows_true_cols_pred;
  const rowSum = cm.map((r) => r.reduce((a, b) => a + b, 0) || 1);
  return (
    <section className="flex flex-col gap-4">
      <div>
        <h3 className="text-base font-semibold">How the model compares with the reference labels</h3>
        <p className="text-sm text-[var(--c4-muted)]">
          Checked on {lc.windows} two-second windows where the label is clear and the car was moving straight.
        </p>
      </div>
      <div className="flex flex-wrap gap-x-10 gap-y-3">
        <div>
          <p className="text-3xl font-bold">{(100 * (lc.accuracy ?? 0)).toFixed(1)}%</p>
          <p className="text-sm text-[var(--c4-muted)]">windows classified correctly</p>
        </div>
        <div>
          <p className="text-3xl font-bold">{(lc.macro_f1 ?? 0).toFixed(3)}</p>
          <p className="text-sm text-[var(--c4-muted)]">macro-F1 (all three classes weighted equally)</p>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full max-w-xl border-collapse text-sm">
          <caption className="pb-2 text-left text-sm text-[var(--c4-muted)]">
            Rows are the reference label, columns the model&apos;s answer. Shading shows the share of each row.
          </caption>
          <thead>
            <tr className="text-[var(--c4-muted)]">
              <th className="p-2 text-left font-normal">Reference</th>
              <th colSpan={3} scope="colgroup" className="p-2 text-center font-normal">Model&apos;s answer</th>
              <th className="p-2" />
            </tr>
            <tr>
              <th className="p-2" />
              {CLASSES.map((c) => (
                <th key={c} scope="col" className="p-2 text-center font-semibold">{CLASS_LABELS[c]}</th>
              ))}
              <th scope="col" className="p-2 text-right font-semibold">Correct</th>
            </tr>
          </thead>
          <tbody>
            {CLASSES.map((t, i) => (
              <tr key={t} className="border-t border-[var(--c4-line)]">
                <th scope="row" className="whitespace-nowrap p-2 text-left font-semibold">
                  <span className="mr-2 inline-block h-2.5 w-2.5" style={{ background: CLASS_COLORS[t] }} />
                  {CLASS_LABELS[t]}
                </th>
                {CLASSES.map((p, j) => {
                  const share = cm[i][j] / rowSum[i];
                  const colour = i === j ? "46,157,91" : "210,63,47";
                  return (
                    <td key={p} className="p-2 text-center" style={{ background: cm[i][j] ? `rgba(${colour},${0.08 + 0.5 * share})` : undefined }}>
                      {cm[i][j]}
                    </td>
                  );
                })}
                <td className="p-2 text-right text-[var(--c4-muted)]">{Math.round(100 * lc.per_class![t].recall)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function mapStyleFor(mode: "predicted" | "actual" | "match", lc: LabelCheck | undefined): MapStyle | undefined {
  const classLegend = CLASSES.map((c) => ({ label: CLASS_LABELS[c], color: CLASS_COLORS[c] }));
  const truth = (id: string) => lc?.segments?.[id]?.true_class ?? null;
  const tip: MapStyle["tooltip"] = (f) => {
    const t = truth(f.properties.record_id);
    return t ? `Reference label: <b>${CLASS_LABELS[t]}</b>` : "No reference label";
  };
  if (!lc?.windows) return undefined;
  if (mode === "predicted") return { color: (f) => CLASS_COLORS[f.properties.roughness_class], tooltip: tip, legend: classLegend };
  if (mode === "actual") {
    return {
      color: (f) => {
        const t = truth(f.properties.record_id);
        return t ? CLASS_COLORS[t] : NO_LABEL_COLOR;
      },
      tooltip: tip,
      legend: [...classLegend, { label: "No label", color: NO_LABEL_COLOR }],
    };
  }
  return {
    color: (f) => {
      const t = truth(f.properties.record_id);
      return !t ? NO_LABEL_COLOR : t === f.properties.roughness_class ? CLASS_COLORS.good : CLASS_COLORS.bad;
    },
    tooltip: tip,
    legend: [
      { label: "Agrees", color: CLASS_COLORS.good },
      { label: "Disagrees", color: CLASS_COLORS.bad },
      { label: "No label", color: NO_LABEL_COLOR },
    ],
  };
}

function RunHeader({ run, profile }: { run: Run; profile: RoadProfile | null }) {
  const s = run.summary;
  const facts = [
    fmtDate(run.created_at),
    s && `${fmtDuration(s.input.duration_s)} recorded`,
    profile && `${fmtKm(profile.totalKm)} km assessed`,
    s && `sensors at ${Math.round(s.input.native_rate_hz)} Hz`,
  ].filter(Boolean);
  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-2xl font-bold tracking-tight">{run.survey_id}</h2>
        {run.status !== "succeeded" && (
          <span className="rounded-full border border-[var(--c4-line)] px-2.5 py-0.5 text-xs font-semibold">{STATUS_TEXT[run.status]}</span>
        )}
        {run.model_kind === "demo" && (
          <span className="rounded-full bg-[var(--c4-warn-bg)] px-2.5 py-0.5 text-xs font-semibold">Demo model, results are not real</span>
        )}
      </div>
      <p className="text-sm text-[var(--c4-muted)]">{facts.join(", ")}</p>
    </div>
  );
}

// --------------------------------------------------------------------------------------------- page
export default function C4Dashboard() {
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState<Run | null>(null);
  const [segments, setSegments] = useState<SegmentCollection | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [demoBusy, setDemoBusy] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [hovered, setHovered] = useState<string | null>(null);
  const loadedSegmentsFor = useRef<string | null>(null);
  const [mapMode, setMapMode] = useState<"predicted" | "actual" | "match">("predicted");
  const labelCheck = selected?.summary?.label_check;
  const mapStyle = useMemo(() => mapStyleFor(mapMode, labelCheck), [mapMode, labelCheck]);
  const profile = useMemo(() => roadProfile(segments, labelCheck), [segments, labelCheck]);

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
        if (!r.length) setShowUpload(true);
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
      setHovered(null);
      setMapMode("predicted");
      loadedSegmentsFor.current = null;
    }
    setSelected(run);
  }

  function created(run: Run) {
    select(run);
    setShowUpload(false);
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

  const done = selected?.status === "succeeded";

  return (
    <div className="c4 flex min-h-screen flex-1 flex-col">
      {/* ------------------------------------------------------------------ header */}
      <header className="bg-[var(--c4-header)] text-[var(--c4-header-ink)]">
        <div className="mx-auto flex w-full max-w-[90rem] flex-wrap items-end justify-between gap-x-6 gap-y-1 px-4 pb-3 pt-4 sm:px-6">
          <div>
            <p className="text-sm opacity-75">RoadLens</p>
            <h1 className="text-2xl font-bold tracking-tight">Road condition from a phone</h1>
          </div>
          <p className="text-sm opacity-80" title={model ? `Model file SHA-256 ${model.model_sha256}` : undefined}>
            {model
              ? `${model.kind === "demo" ? "Demo model" : "Model"} ${model.model_version}${model.lovo_macro_f1 != null ? `, F1 ${model.lovo_macro_f1.toFixed(3)} on an unseen vehicle` : ""}`
              : modelError
                ? `Model unavailable: ${modelError}`
                : "Loading model…"}
          </p>
        </div>
        <div className="c4-centre-line" aria-hidden />
      </header>

      <div className="mx-auto flex w-full max-w-[90rem] flex-1 flex-col gap-6 px-4 py-6 sm:px-6 lg:flex-row lg:gap-8">
        {/* ---------------------------------------------------------------- surveys */}
        <aside className="flex w-full shrink-0 flex-col gap-4 lg:w-80">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold">Surveys</h2>
            <button
              onClick={() => setShowUpload((v) => !v)}
              aria-expanded={showUpload}
              className={showUpload ? "text-sm font-semibold text-[var(--c4-muted)] underline" : btnPrimary + " !py-1.5"}
            >
              {showUpload ? "Close" : "New survey"}
            </button>
          </div>
          {showUpload && (
            <div className="rounded-md border border-[var(--c4-line)] bg-[var(--c4-surface)] p-4">
              <UploadForm onCreated={created} onDemo={runDemo} demoBusy={demoBusy} />
            </div>
          )}
          {apiError && (
            <Notice tone="danger">
              The server did not respond: {apiError}.{" "}
              <button onClick={refreshRuns} className="font-semibold underline">Try again</button>
            </Notice>
          )}
          <div className="max-h-72 overflow-y-auto lg:max-h-[calc(100vh-12rem)]">
            <SurveyList runs={runs} selectedId={selectedId} onSelect={select} />
          </div>
        </aside>

        {/* ---------------------------------------------------------------- selected survey */}
        <main className="flex min-w-0 flex-1 flex-col gap-6">
          {selected ? <RunHeader run={selected} profile={profile} /> : (
            <div className="flex flex-col gap-1">
              <h2 className="text-2xl font-bold tracking-tight">Choose a survey</h2>
              <p className="max-w-[65ch] text-sm text-[var(--c4-muted)]">
                Record a drive with a phone fixed to the dashboard, then upload the motion and GPS files. Each stretch of
                road is classed as good, regular or bad and drawn on the map.
              </p>
            </div>
          )}

          {selected && isActive(selected.status) && <Progress run={selected} />}
          {selected?.error && <Notice tone="danger">{selected.error.message}</Notice>}

          <div className="h-[52vh] min-h-[340px]">
            <SegmentMap data={segments} style={mapStyle} highlight={hovered} onHover={setHovered}>
              {labelCheck?.windows ? (
                <div className="flex rounded-md border border-[var(--c4-line)] bg-[var(--c4-surface)] p-0.5 text-xs" role="radiogroup" aria-label="Map colours">
                  {(["predicted", "actual", "match"] as const).map((m) => (
                    <button
                      key={m}
                      role="radio"
                      aria-checked={mapMode === m}
                      onClick={() => setMapMode(m)}
                      className={`rounded px-3 py-1.5 ${mapMode === m ? "bg-[var(--c4-ink)] text-[var(--c4-bg)]" : "hover:bg-[var(--c4-bg)]"}`}
                    >
                      {m === "predicted" ? "Model" : m === "actual" ? "Reference" : "Agreement"}
                    </button>
                  ))}
                </div>
              ) : null}
            </SegmentMap>
          </div>

          {done && profile && (
            <>
              <RoadStrip profile={profile} showReference={!!labelCheck?.windows} hovered={hovered} onHover={setHovered} />
              <ConditionSummary profile={profile} run={selected!} />
            </>
          )}

          {done && labelCheck && <Evaluation lc={labelCheck} />}

          {selected && selected.warnings.length > 0 && (
            <Notice tone="warn">
              <ul className="flex list-disc flex-col gap-1 pl-4">
                {selected.warnings.map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            </Notice>
          )}

          {done && (
            <footer className="flex flex-wrap gap-x-5 gap-y-1 border-t border-[var(--c4-line)] pt-3 text-sm text-[var(--c4-muted)]">
              <span>Download:</span>
              <a className="underline hover:text-[var(--c4-ink)]" href={`${C4_API}/runs/${selected!.run_id}/segments`} target="_blank" rel="noreferrer">
                Road segments (GeoJSON)
              </a>
              <a className="underline hover:text-[var(--c4-ink)]" href={`${C4_API}/runs/${selected!.run_id}/records`} target="_blank" rel="noreferrer">
                Segment records (JSON)
              </a>
              <a className="underline hover:text-[var(--c4-ink)]" href={`${C4_API}/runs/${selected!.run_id}/windows.csv`}>
                Window details (CSV)
              </a>
              {selected!.summary && selected!.summary.segments.unassessed_windows > 0 && (
                <span className="basis-full">
                  {selected!.summary.segments.unassessed_windows} windows were skipped because the car was stopped or slower than 1 m/s.
                </span>
              )}
            </footer>
          )}
        </main>
      </div>
    </div>
  );
}
