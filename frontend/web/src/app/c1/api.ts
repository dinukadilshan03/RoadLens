/**
 * Client API for Component 1 (Road Damage Detection & Mapping).
 */

export const C1_API = "/api/v1/c1";

export interface SupportedClass {
  id: number;
  name: string;
  display: string;
  color: string;
}

export interface B0Benchmark {
  overall: {
    mAP50: number;
    mAP50_95: number;
    precision: number;
    recall: number;
  };
  per_class: Record<string, { ap50: number; precision: number; recall: number }>;
}

export interface ModelInfo {
  model_version: string;
  milestone: string;
  weights_loaded: boolean;
  weights_sha256: string;
  weights_path: string;
  input_resolution: string;
  supported_classes: SupportedClass[];
  b0_benchmark: B0Benchmark;
  next_50_scope: string[];
}

export interface RunSummary {
  model_version: string;
  weights_sha256: string;
  total_frames_processed: number;
  total_defects_detected: number;
  counts_by_class: Record<string, number>;
  average_confidence: number;
  b0_benchmark_mAP50: number;
}

export interface Run {
  run_id: string;
  survey_id: string;
  source: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  stage: string | null;
  progress: { completed: number; total: number };
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: { code: string; message: string } | null;
  warnings: string[];
  summary: RunSummary | null;
}

export interface DefectProperties {
  defect_id: string;
  damage_class: string;
  display_name: string;
  confidence: number;
  timestamp_ms: number;
  frame_idx: number;
  bbox: [number, number, number, number];
  crop_image: string;
  annotated_frame: string;
}

export interface DefectFeature {
  type: "Feature";
  geometry: {
    type: "Point";
    coordinates: [number, number]; // [lng, lat]
  };
  properties: DefectProperties;
}

export interface DefectCollection {
  type: "FeatureCollection";
  features: DefectFeature[];
}

export const CLASS_COLORS: Record<string, string> = {
  Longitudinal_Crack_D00: "#2563eb",
  Transverse_Crack_D10: "#059669",
  Alligator_Crack_D20: "#d97706",
  Pothole_D40: "#dc2626",
};

export const CLASS_LABELS: Record<string, string> = {
  Longitudinal_Crack_D00: "Longitudinal (D00)",
  Transverse_Crack_D10: "Transverse (D10)",
  Alligator_Crack_D20: "Alligator (D20)",
  Pothole_D40: "Pothole (D40)",
};

export class ApiError extends Error {
  code: string;
  details: Record<string, unknown>;
  constructor(message: string, code = "api_error", details = {}) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.details = details;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    let code = "http_error";
    let details = {};
    try {
      const err = await res.json();
      msg = err.message || msg;
      code = err.code || code;
      details = err.details || {};
    } catch {
      // not JSON
    }
    throw new ApiError(msg, code, details);
  }
  return res.json();
}

export function isActive(status: string): boolean {
  return status === "queued" || status === "running";
}

export const api = {
  model: () => request<ModelInfo>(`${C1_API}/model`),
  runs: () => request<Run[]>(`${C1_API}/runs`),
  run: (runId: string) => request<Run>(`${C1_API}/runs/${runId}`),
  upload: (data: FormData) =>
    request<Run>(`${C1_API}/runs`, {
      method: "POST",
      body: data,
    }),
  demo: () =>
    request<Run>(`${C1_API}/runs/demo`, {
      method: "POST",
    }),
  defects: (runId: string) => request<DefectCollection>(`${C1_API}/runs/${runId}/defects`),
  artifactUrl: (runId: string, kind: "frames" | "crops" | "inputs", filename: string) =>
    `${C1_API}/runs/${runId}/artifacts/${kind}/${filename}`,
};
