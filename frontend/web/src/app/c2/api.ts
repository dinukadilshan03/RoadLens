// Client for the C2 Noise-Resilient Road Damage Segmentation API (/api/v1/c2).

export const C2_API = "/api/v1/c2";

export interface TriPillarTargets {
  pillar_1_spatial: { target_mIoU: string; target_f1: string };
  pillar_2_continuity: { target_topological_continuity: string };
  pillar_3_hardware: { target_vram: string; target_latency: string };
}

export interface C2ModelInfo {
  component: string;
  name: string;
  architecture: string;
  weights_ready: boolean;
  cgan_sensor_ready: boolean;
  segmenter_ready: boolean;
  cgan_weights_sha256: string;
  segmenter_weights_sha256: string;
  input_resolution: [number, number];
  tri_pillar_targets: TriPillarTargets;
  output_suite: {
    output_a: string;
    output_b: string;
    output_c: string;
    output_d: string;
  };
}

export interface C1CropItem {
  run_id: string;
  record_id: string;
  defect_id: string;
  damage_class: string;
  display_name: string;
  confidence: number;
  crop_artifact: string;
  crop_transform: {
    origin_xy: [number, number];
    width: number;
    height: number;
  };
  created_at: string;
}

export interface C2Artifacts {
  input_rgb: string;
  pseudo_thermal: string;
  mask_2d: string;
  skeleton_1px: string;
  confidence_npz: string;
  confidence_map: string;
  visual_overlay: string;
}

export interface C2RunMetrics {
  area_pixels: number;
  skeleton_length_px: number;
  avg_confidence: number;
  miou: number;
  f1_score: number;
  topological_continuity: number;
  latency_ms: number;
  vram_mb: number;
}

export interface C2ResultRecord {
  schema_version: string;
  component: string;
  record_id: string;
  run_id: string;
  survey_id: string;
  upstream: {
    c1_run_id?: string | null;
    c1_observation_id?: string | null;
  };
  damage_class: string;
  is_crack: boolean;
  dimensions: {
    width: number;
    height: number;
    original_width: number;
    original_height: number;
  };
  alpha_weight: number;
  metrics: C2RunMetrics;
  artifacts: C2Artifacts;
  handoff_ready: {
    c3_severity: boolean;
    c4_gis_twin: boolean;
  };
}

export interface C2Run {
  run_id: string;
  survey_id: string;
  source: "upload" | "demo" | "c1_handoff";
  c1_run_id?: string | null;
  c1_observation_id?: string | null;
  damage_class: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled" | "rejected";
  stage?: string | null;
  progress?: { completed: number | null; total: number | null } | null;
  alpha_score?: number | null;
  latency_ms?: number | null;
  vram_mb?: number | null;
  artifacts?: C2Artifacts | null;
  summary?: Record<string, any> | null;
  results?: C2ResultRecord[];
  error?: { code: string; message: string } | null;
  warnings?: string[];
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
}

export interface BenchmarkPreset {
  id: string;
  title: string;
  damageClass: string;
  noiseType: string;
  description: string;
  thumbnail: string;
  expectedAlpha: number;
}

export const BENCHMARK_PRESETS: BenchmarkPreset[] = [
  {
    id: "shadow_crack",
    title: "Longitudinal Crack under Shadows",
    damageClass: "Longitudinal_Crack_D00",
    noiseType: "Tree Canopy Shadow",
    description: "Evaluates cGAN moisture synthesis eliminating dark tree shadows that deceive RGB models.",
    thumbnail: "/samples/c2/sample_shadow_crack.jpg",
    expectedAlpha: 0.35,
  },
  {
    id: "glare_alligator",
    title: "Alligator Crack under Glare",
    damageClass: "Alligator_Crack_D20",
    noiseType: "Tropical Solar Glare",
    description: "Evaluates dynamic Alpha gate preserving narrow hairline cracks (<3mm) washed out by solar glare.",
    thumbnail: "/samples/c2/sample_glare_alligator.jpg",
    expectedAlpha: 0.28,
  },
  {
    id: "drainage_grate",
    title: "Metallic Drainage Grate (OOD)",
    damageClass: "Longitudinal_Crack_D00",
    noiseType: "Out-Of-Distribution Grate",
    description: "Tests DMA-Net Fig. 8c scenario: proves thermal cross-fusion ignores metal grid covers without false alarms.",
    thumbnail: "/samples/c2/sample_drainage_grate.jpg",
    expectedAlpha: 0.85,
  },
  {
    id: "pothole_water",
    title: "Water-Filled Asphalt Pothole",
    damageClass: "Pothole_D40",
    noiseType: "Puddle Moisture Cavity",
    description: "Emulates cool radiometric thermal void for 2D area delineation to feed C3 3D volume calculations.",
    thumbnail: "/samples/c2/sample_pothole_water.jpg",
    expectedAlpha: 0.76,
  },
];

export const api = {
  async model(): Promise<C2ModelInfo> {
    const res = await fetch(`${C2_API}/model`);
    if (!res.ok) throw new Error(`Failed to load model info: ${res.statusText}`);
    return res.json();
  },

  async runs(): Promise<C2Run[]> {
    const res = await fetch(`${C2_API}/runs`);
    if (!res.ok) throw new Error(`Failed to list C2 runs: ${res.statusText}`);
    return res.json();
  },

  async run(runId: string): Promise<C2Run> {
    const res = await fetch(`${C2_API}/runs/${runId}`);
    if (!res.ok) throw new Error(`Failed to load C2 run ${runId}: ${res.statusText}`);
    return res.json();
  },

  async c1Crops(): Promise<C1CropItem[]> {
    const res = await fetch(`${C2_API}/c1-available-crops`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.crops || [];
  },

  async triggerRun(formData: FormData): Promise<{ run_id: string; status: string }> {
    const res = await fetch(`${C2_API}/runs`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ message: res.statusText }));
      throw new Error(err.message || "Failed to trigger C2 run");
    }
    return res.json();
  },

  artifactUrl(runId: string, artifactType: keyof C2Artifacts): string {
    return `${C2_API}/runs/${runId}/artifacts/${artifactType}`;
  },
};
