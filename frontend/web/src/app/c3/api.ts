export interface Defect {
  defect_id: string;
  source_bbox_xyxy: number[];
  raw_axial_depression_map?: number[][];
  geometry_raw: {
    d95_raw: number;
    mean_raw: number;
    max_raw: number;
    source_pothole_area_px: number;
  } | null;
  reliability_status: string;
  reliability: { input_qa_flags: string[]; action: string };
  reasons: string[];
  measurement_notes: string;
  pixel_measurements: {
    footprint_area_px2: number;
    crack: Record<string, unknown> | null;
  };
}
export interface Analysis {
  bundle_version: string;
  frame_width: number;
  frame_height: number;
  defects: Defect[];
  overall_status: string;
}
export async function analyzeSurface(
  image: File,
  mask: File,
  crack?: File,
): Promise<Analysis> {
  const form = new FormData();
  form.append("image", image);
  form.append("mask", mask);
  if (crack) form.append("crack_mask", crack);
  const response = await fetch("/api/v1/c3/analyze", {
    method: "POST",
    body: form,
  });
  const body = await response.json();
  if (!response.ok)
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : "Analysis failed. Check your inputs and try again.",
    );
  return body;
}
