// Distance along the surveyed road ("chainage") for the segment features of a run.
import type { LabelCheck, RoughnessClass, SegmentCollection } from "./api";

export interface RoadPiece {
  id: string;
  fromKm: number;
  toKm: number;
  startMs: number;
  endMs: number;
  predicted: RoughnessClass;
  actual: RoughnessClass | null;
  confidence: number | null;
}

export interface RoadProfile {
  pieces: RoadPiece[];
  totalKm: number;
  kmByClass: Record<RoughnessClass, number>;
}

function haversineKm([lon1, lat1]: [number, number], [lon2, lat2]: [number, number]) {
  const r = Math.PI / 180;
  const a =
    Math.sin(((lat2 - lat1) * r) / 2) ** 2 +
    Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.sin(((lon2 - lon1) * r) / 2) ** 2;
  return 12742 * Math.asin(Math.sqrt(a));
}

export function roadProfile(data: SegmentCollection | null, labels?: LabelCheck): RoadProfile | null {
  if (!data?.features.length) return null;
  const feats = [...data.features].sort((a, b) => a.properties.interval_start_ms - b.properties.interval_start_ms);
  const kmByClass: Record<RoughnessClass, number> = { good: 0, regular: 0, bad: 0 };
  const pieces: RoadPiece[] = [];
  let km = 0;
  for (const f of feats) {
    const c = f.geometry?.coordinates ?? [];
    let len = 0;
    for (let i = 1; i < c.length; i++) len += haversineKm(c[i - 1], c[i]);
    const p = f.properties;
    pieces.push({
      id: p.record_id,
      fromKm: km,
      toKm: km + len,
      startMs: p.interval_start_ms,
      endMs: p.interval_end_ms,
      predicted: p.roughness_class,
      actual: labels?.segments?.[p.record_id]?.true_class ?? null,
      confidence: p.confidence,
    });
    kmByClass[p.roughness_class] += len;
    km += len;
  }
  return { pieces, totalKm: km, kmByClass };
}

export function fmtKm(km: number) {
  return km >= 10 ? km.toFixed(1) : km.toFixed(2);
}

export function fmtDuration(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m ? `${m} min ${s} s` : `${s} s`;
}
