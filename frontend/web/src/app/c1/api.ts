// Client for the C1 Road Damage Detection API (backend: /api/v1/c1, see ml/c1/api/router.py).

export const C1_API = "/api/v1/c1";

export type DamageClass =
  | "Longitudinal_Crack_D00"
  | "Transverse_Crack_D10"
  | "Alligator_Crack_D20"
  | "Pothole_D40";

export type RunStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled" | "rejected";

export interface ClassDefinition {
  id: number;
  name: DamageClass;
  display_name: string;
  color: string;
  b0_ap50: number;
}

export interface ModelValidationMetrics {
  dataset: string;
  architecture: string;
  overall: {
    precision: number;
    recall: number;
    mAP50: number;
    mAP50_95: number;
  };
  per_class_ap50: Record<DamageClass, number>;
}

export interface ModelInfo {
  component: string;
  model_version: string;
  weights_ready: boolean;
  weights_filename: string;
  weights_size_mb: number;
  weights_sha256: string;
  input_resolution: number;
  classes: ClassDefinition[];
  validation_metrics: ModelValidationMetrics;
}

export interface FrameSummary {
  frame_idx: number;
  source_media: string;
  width: number;
  height: number;
  timestamp_ms: number | null;
  annotated_frame: string;
  defect_count: number;
}

export interface RunSummary {
  producer: {
    model_version: string;
    weights_sha256: string;
    code_version: string;
    config_version: string;
  };
  conf_threshold: number;
  iou_threshold: number;
  imgsz: number;
  total_frames_processed: number;
  frames_with_defects: number;
  total_defects_detected: number;
  counts_by_class: Record<DamageClass, number>;
  average_confidence: number;
  max_confidence: number;
  frames: FrameSummary[];
  b0_benchmark_mAP50: number;
}

export interface RunError {
  code: string;
  message: string;
  retryable?: boolean;
}

export interface Run {
  run_id: string;
  survey_id: string;
  source: "upload" | "demo";
  status: RunStatus;
  stage: string | null;
  progress?: { completed: number | null; total: number | null } | null;
  model_version?: string | null;
  weights_sha256?: string | null;
  conf_threshold: number;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  error?: RunError | null;
  warnings: string[];
  summary?: RunSummary | null;
}

export interface CropTransform {
  origin_xy: [number, number];
  width: number;
  height: number;
  source_frame_size: [number, number];
}

export interface ObservationRecord {
  schema_version: string;
  component: string;
  record_id: string;
  defect_id: string;
  external_run_id: string;
  survey_id: string;
  source_media: string;
  frame_idx: number;
  frame_dimensions: { width: number; height: number };
  timestamp_ms: number | null;
  damage_class: DamageClass;
  class_id: number;
  display_name: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x_min, y_min, x_max, y_max]
  annotated_frame: string;
  crop_artifact: string | null;
  crop_transform: CropTransform;
  location: { latitude: number; longitude: number } | null;
  location_null_reason?: string | null;
  annotated_url?: string;
  crop_url?: string;
}

export interface SamplePreset {
  key: string;
  title: string;
  badge: string;
  description: string;
  distressTypes: string;
  imageUrl: string;
}

export const PRESET_SAMPLES: SamplePreset[] = [
  {
    key: "pothole_alligator",
    title: "Pothole & Fatigue Crack",
    badge: "Severe Distress",
    description: "Structural depression accompanied by interconnected alligator crack networks.",
    distressTypes: "Pothole (D40) · Alligator (D20)",
    imageUrl: "/samples/c1/sample_pothole_alligator.jpg",
  },
  {
    key: "potholes",
    title: "Deep Pothole Cluster",
    badge: "High Hazard",
    description: "Multiple sharp asphalt surface depressions with prominent depth shadow.",
    distressTypes: "Potholes (D40) x2 · Transverse (D10)",
    imageUrl: "/samples/c1/sample_potholes.jpg",
  },
  {
    key: "alligator",
    title: "Alligator Fatigue Cracks",
    badge: "Base Failure",
    description: "Hexagonal pattern of fatigue cracking caused by heavy repeated axle load.",
    distressTypes: "Alligator Crack (D20)",
    imageUrl: "/samples/c1/sample_alligator.jpg",
  },
  {
    key: "longitudinal",
    title: "Longitudinal Linear Crack",
    badge: "Joint Separation",
    description: "Linear crack running parallel to road centerline along the pavement seam.",
    distressTypes: "Longitudinal Crack (D00)",
    imageUrl: "/samples/c1/sample_longitudinal.jpg",
  },
];

export const CLASS_COLORS: Record<DamageClass, string> = {
  Longitudinal_Crack_D00: "#2563eb", // Blue
  Transverse_Crack_D10: "#d97706",   // Amber
  Alligator_Crack_D20: "#ea580c",    // Orange
  Pothole_D40: "#dc2626",            // Red
};

export const CLASS_BG_LIGHT: Record<DamageClass, string> = {
  Longitudinal_Crack_D00: "#eff6ff",
  Transverse_Crack_D10: "#fffbeb",
  Alligator_Crack_D20: "#fff7ed",
  Pothole_D40: "#fef2f2",
};

export const CLASS_BORDER: Record<DamageClass, string> = {
  Longitudinal_Crack_D00: "#bfdbfe",
  Transverse_Crack_D10: "#fde68a",
  Alligator_Crack_D20: "#fed7aa",
  Pothole_D40: "#fecaca",
};

export const CLASS_LABELS: Record<DamageClass, string> = {
  Longitudinal_Crack_D00: "Longitudinal Crack (D00)",
  Transverse_Crack_D10: "Transverse Crack (D10)",
  Alligator_Crack_D20: "Alligator Crack (D20)",
  Pothole_D40: "Pothole (D40)",
};

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public details?: Record<string, unknown>
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${C1_API}${path}`, { cache: "no-store", ...init });
  if (!res.ok) {
    let code = "http_error";
    let message = `Request failed (${res.status})`;
    let details: Record<string, unknown> = {};
    try {
      const body = await res.json();
      code = body.code ?? code;
      message = body.message ?? body.detail ?? message;
      details = body.details ?? details;
    } catch {
      // non-JSON
    }
    throw new ApiError(res.status, code, message, details);
  }
  return res.json() as Promise<T>;
}

export const isActive = (s: RunStatus) => s === "queued" || s === "running";

export const api = {
  model: () => request<ModelInfo>("/model"),
  runs: () => request<{ runs: Run[] }>("/runs?limit=30").then((r) => r.runs),
  run: (id: string) => request<Run>(`/runs/${id}`),
  records: (id: string) =>
    request<{ records: ObservationRecord[] }>(`/runs/${id}/records`).then((r) => r.records),
  cancel: (id: string) => request<Run>(`/runs/${id}/cancel`, { method: "POST" }),
  upload: (form: FormData) => request<Run>("/runs", { method: "POST", body: form }),
  demo: (sampleKey: string, confThreshold = 0.25) => {
    const fd = new FormData();
    fd.append("sample_key", sampleKey);
    fd.append("conf_threshold", confThreshold.toString());
    return request<Run>("/runs/demo", { method: "POST", body: fd });
  },
  artifactUrl: (runId: string, kind: "annotated" | "crops", filename: string) =>
    `${C1_API}/runs/${runId}/artifacts/${kind}/${filename}`,
};
