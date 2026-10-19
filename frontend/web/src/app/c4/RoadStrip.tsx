"use client";

// Straight-line diagram: the surveyed road drawn as a straight strip by distance (km), the way road
// agencies chart pavement condition. One row for the model, and one for the reference labels when given.
import { CLASS_COLORS, CLASS_LABELS, NO_LABEL_COLOR } from "./api";
import { fmtKm, type RoadPiece, type RoadProfile } from "./road";

function ticks(totalKm: number) {
  const steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50];
  const step = steps.find((s) => totalKm / s <= 8) ?? 100;
  const out: number[] = [];
  for (let k = 0; k <= totalKm + 1e-9; k += step) out.push(+k.toFixed(3));
  return out;
}

function describe(p: RoadPiece) {
  const conf = p.confidence == null ? "" : `, model confidence ${Math.round(p.confidence * 100)}%`;
  const ref = p.actual ? ` Reference label: ${CLASS_LABELS[p.actual]}.` : "";
  return `km ${fmtKm(p.fromKm)} to ${fmtKm(p.toKm)}: ${CLASS_LABELS[p.predicted]}${conf}.${ref}`;
}

function Row({
  label,
  profile,
  colour,
  hovered,
  onHover,
}: {
  label: string;
  profile: RoadProfile;
  colour: (p: RoadPiece) => string;
  hovered: string | null;
  onHover: (id: string | null) => void;
}) {
  const { pieces, totalKm } = profile;
  const active = hovered ? pieces.find((p) => p.id === hovered) : undefined;
  return (
    <div className="grid grid-cols-[5.5rem_1fr] items-center gap-3">
      <span className="text-sm text-[var(--c4-muted)]">{label}</span>
      <div className="c4-unassessed relative h-7 rounded-[3px]" onMouseLeave={() => onHover(null)}>
        {pieces.map((p) => (
          <div
            key={p.id}
            onMouseEnter={() => onHover(p.id)}
            className="absolute inset-y-0"
            style={{
              left: `${(100 * p.fromKm) / totalKm}%`,
              width: `max(1px, ${(100 * (p.toKm - p.fromKm)) / totalKm}%)`,
              background: colour(p),
            }}
          />
        ))}
        {active && (
          <div
            aria-hidden
            className="pointer-events-none absolute inset-y-0 outline outline-2 outline-[var(--c4-ink)]"
            style={{
              left: `${(100 * active.fromKm) / totalKm}%`,
              width: `max(3px, ${(100 * (active.toKm - active.fromKm)) / totalKm}%)`,
            }}
          />
        )}
      </div>
    </div>
  );
}

export default function RoadStrip({
  profile,
  showReference,
  hovered,
  onHover,
}: {
  profile: RoadProfile;
  showReference: boolean;
  hovered: string | null;
  onHover: (id: string | null) => void;
}) {
  const active = profile.pieces.find((p) => p.id === hovered) ?? null;
  const t = ticks(profile.totalKm);
  return (
    <figure className="flex flex-col gap-2.5" aria-label="Road condition along the surveyed route, by distance">
      <figcaption className="flex min-h-[1.5rem] flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <span className="text-base font-semibold">Condition along the route</span>
        <span className="text-sm text-[var(--c4-muted)]" aria-live="polite">
          {active ? describe(active) : "Point at the strip or the map to inspect a stretch of road."}
        </span>
      </figcaption>
      <Row label="Model" profile={profile} colour={(p) => CLASS_COLORS[p.predicted]} hovered={hovered} onHover={onHover} />
      {showReference && (
        <Row
          label="Reference"
          profile={profile}
          colour={(p) => (p.actual ? CLASS_COLORS[p.actual] : NO_LABEL_COLOR)}
          hovered={hovered}
          onHover={onHover}
        />
      )}
      <div className="grid grid-cols-[5.5rem_1fr] gap-3">
        <span />
        <div className="relative h-5 text-xs text-[var(--c4-muted)]">
          {t.map((k) => (
            <span
              key={k}
              className="absolute top-0 flex -translate-x-1/2 flex-col items-center whitespace-nowrap"
              style={{ left: `${(100 * k) / profile.totalKm}%` }}
            >
              <span className="h-1.5 w-px bg-[var(--c4-muted)]" />
              {k} km
            </span>
          ))}
        </div>
      </div>
    </figure>
  );
}
