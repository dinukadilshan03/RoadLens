export interface RelativeSurface {
  x: number[];
  y: number[];
  z: number[][];
  depression: number[][];
  aspect_ratio: { x: number; y: number; z: number };
  has_depression: boolean;
  physical_scale_verified: false;
}
export interface Defect {
  crack_network?: {
    connected_regions: number;
    regions?: { region_id: number; bbox_xyxy: number[]; overlay: string; pixels: number }[];
    branches: { branch_id: number; bbox_xyxy: number[]; overlay: string; pixels: number }[];
    complexity: { label: string; explanation: string };
    regional_visual_findings: { region_id: number; prediction: { visual_crack_severity?: string; review_required?: boolean } | null }[];
  };
  source_group?: "pothole" | "crack";
  source_kind?: "photo" | "mask";
  source_width?: number;
  source_height?: number;
  defect_id: string;
  damage_type: "pothole" | "crack";
  source_bbox_xyxy: number[];
  raw_axial_depression_map?: number[][];
  display_chart?: { profiles: number[][]; distribution: number[] };
  relative_surface?: RelativeSurface | null;
  geometry_raw: { max_raw: number } | null;
  visuals: {
    outline: string;
    centerline?: string;
    width?: string;
    depression?: string;
  };
  presentation: {
    label: string;
    severity: string;
    assessment: string;
    explanation: string;
    shape: string;
    review_reasons: string[];
    physical_severity: string;
    measurement: string;
    measurement_note: string;
    reliability: string;
    recommendation: string;
  };
}
export interface Analysis {
  input_warnings: { damage_type: string; message: string }[];
  frame_width: number;
  frame_height: number;
  defects: Defect[];
  summary: string;
}
export type FieldMeasurements = Record<
  string,
  Record<string, string | number | boolean>
>;
function inputs(
  image: File | undefined,
  mask?: File,
  crack?: File,
  fields?: FieldMeasurements,
  crackImage?: File,
  crackMatchesImage = false,
) {
  const form = new FormData();
  form.append("crack_matches_image", String(crackMatchesImage));
  if (image) form.append("image", image);
  if (crackImage) form.append("crack_image", crackImage);
  if (mask) form.append("mask", mask);
  if (crack) form.append("crack_mask", crack);
  if (fields && Object.keys(fields).length)
    form.append("field_measurements", JSON.stringify(fields));
  return form;
}
async function request(
  path: string,
  image: File | undefined,
  mask?: File,
  crack?: File,
  fields?: FieldMeasurements,
  crackImage?: File,
  crackMatchesImage = false,
) {
  const response = await fetch(`/api/v1/c3/${path}`, {
    method: "POST",
    body: inputs(image, mask, crack, fields, crackImage, crackMatchesImage),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : "Unable to analyze. Check the image and marked areas, then try again.",
    );
  }
  return response;
}
export async function analyzeSurface(
  image: File | undefined,
  mask?: File,
  crack?: File,
  fields?: FieldMeasurements,
  crackImage?: File,
  crackMatchesImage = false,
): Promise<Analysis> {
  return (await request("assess", image, mask, crack, fields, crackImage, crackMatchesImage)).json();
}
export async function downloadReport(
  image: File | undefined,
  mask?: File,
  crack?: File,
  fields?: FieldMeasurements,
  crackImage?: File,
  crackMatchesImage = false,
) {
  const blob = await (
    await request("assessment-report", image, mask, crack, fields, crackImage, crackMatchesImage)
  ).blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "roadlens-road-damage.pdf";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
