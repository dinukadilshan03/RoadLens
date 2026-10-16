// Client for the C4 road-roughness API (backend: /api/v1/c4, see ml/c4/api/router.py).

export const C4_API = "/api/v1/c4";

export type RoughnessClass = "good" | "regular" | "bad";
export type RunStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled" | "rejected";

export interface RunError {
  code: string;
  message: string;
  retryable?: boolean;
}

export interface RunSummary {
  segments: {
    count: number;
    by_class: Record<RoughnessClass, number>;
    seconds_by_class: Record<RoughnessClass, number>;
    unassessed_windows: number;
    null_geometry: number;
  };
  input: {
    imu_rows: number;
    gps_rows: number;
    native_rate_hz: number;
    duration_s: number;
    speed_source: string;
  };
  label_check?: LabelCheck;
}

export interface LabelCheck {
  windows: number;
  accuracy?: number;
  macro_f1?: number;
  per_class?: Record<RoughnessClass, { precision: number; recall: number; f1: number; support: number }>;
  confusion_matrix?: { labels: RoughnessClass[]; rows_true_cols_pred: number[][] };
  segments: Record<string, { true_class: RoughnessClass | null; share: number }>;
  method?: string;
  note?: string;
}

export interface Run {
  run_id: string;
  survey_id: string;
  source: "upload" | "demo";
  status: RunStatus;
  stage: string | null;
  progress?: { completed: number | null; total: number | null } | null;
  model_kind?: "demo" | "production" | null;
  model_version?: string | null;
  created_at: string;
  finished_at: string | null;
  error: RunError | null;
  warnings: string[];
  summary: RunSummary | null;
}

export interface ModelInfo {
  kind: "demo" | "production";
  source_kind: "mock" | "inference";
  model_name: string;
  model_version: string;
  model_sha256: string;
  n_features: number;
  lovo_macro_f1: number | null;
}

export interface SegmentFeature {
  type: "Feature";
  geometry: { type: "LineString"; coordinates: [number, number][] } | null;
  properties: {
    record_id: string;
    interval_start_ms: number;
    interval_end_ms: number;
    roughness_class: RoughnessClass;
    confidence: number | null;
  };
}

export interface SegmentCollection {
  type: "FeatureCollection";
  features: SegmentFeature[];
}

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${C4_API}${path}`, { cache: "no-store", ...init });
  if (!res.ok) {
    let code = "http_error";
    let message = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      code = body.code ?? code;
      message = body.message ?? body.detail ?? message;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, code, message);
  }
  return res.json() as Promise<T>;
}

export const api = {
  model: () => request<ModelInfo>("/model"),
  runs: () => request<{ runs: Run[] }>("/runs?limit=30").then((r) => r.runs),
  run: (id: string) => request<Run>(`/runs/${id}`),
  segments: (id: string) => request<SegmentCollection>(`/runs/${id}/segments`),
  demo: () => request<Run>("/runs/demo", { method: "POST" }),
  cancel: (id: string) => request<Run>(`/runs/${id}/cancel`, { method: "POST" }),
  upload: (form: FormData) => request<Run>("/runs", { method: "POST", body: form }),
};

export const CLASS_COLORS: Record<RoughnessClass, string> = {
  good: "#1a9850",
  regular: "#e6a100",
  bad: "#d73027",
};

export const CLASS_LABELS: Record<RoughnessClass, string> = {
  good: "Good",
  regular: "Regular",
  bad: "Bad",
};

export const isActive = (s: RunStatus) => s === "queued" || s === "running";
