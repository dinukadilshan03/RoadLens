"use client";
import { useState } from "react";
import type { FieldMeasurements } from "../api";

export default function FieldInspection({
  id,
  busy,
  saved,
  onSave,
}: {
  id: string;
  busy: boolean;
  saved?: FieldMeasurements[string];
  onSave: (field: FieldMeasurements[string] | undefined) => Promise<void>;
}) {
  const [depth, setDepth] = useState(String(saved?.deepest_point_mm || ""));
  const [uncertainty, setUncertainty] = useState(
    String(saved?.depth_uncertainty_mm ?? ""),
  );
  const [method, setMethod] = useState(
    String(saved?.measurement_method || "STRAIGHTEDGE_RULER"),
  );
  const [asphalt, setAsphalt] = useState(
    saved?.road_surface_type === "ASPHALT_CONCRETE",
  );
  const [size, setSize] = useState(
    saved?.fhwa_150mm_inscribed_circle_confirmed === true,
  );
  const [reference, setReference] = useState(
    saved?.healthy_road_reference_verified === true,
  );
  return (
    <details className="c3-research">
      <summary>Add a physical field inspection for this pothole</summary>
      <p>
        Enter measurements taken on site. Confirm the conditions below so the
        existing pothole assessment policy can determine whether they are
        suitable. This does not calibrate the estimated image shape.
      </p>
      <form
        className="c3-field-form"
        onSubmit={(e) => {
          e.preventDefault();
          void onSave({
            source: "MANUAL_FIELD_MEASUREMENT",
            deepest_point_mm: Number(depth),
            ...(uncertainty !== ""
              ? { depth_uncertainty_mm: Number(uncertainty) }
              : {}),
            measurement_method: method,
            road_surface_type: asphalt ? "ASPHALT_CONCRETE" : "UNCONFIRMED",
            fhwa_150mm_inscribed_circle_confirmed: size,
            healthy_road_reference_verified: reference,
          });
        }}
      >
        <label htmlFor={`${id}-depth`}>
          Measured deepest point (mm)
          <input
            id={`${id}-depth`}
            type="number"
            min="0.01"
            step="any"
            required
            value={depth}
            disabled={busy}
            onChange={(e) => setDepth(e.target.value)}
          />
        </label>
        <label htmlFor={`${id}-uncertainty`}>
          Measurement uncertainty (mm, optional)
          <input
            id={`${id}-uncertainty`}
            type="number"
            min="0"
            step="any"
            value={uncertainty}
            disabled={busy}
            onChange={(e) => setUncertainty(e.target.value)}
          />
        </label>
        <label htmlFor={`${id}-method`}>
          Inspection method
          <select
            id={`${id}-method`}
            value={method}
            disabled={busy}
            onChange={(e) => setMethod(e.target.value)}
          >
            <option value="STRAIGHTEDGE_RULER">Straightedge and ruler</option>
            <option value="ROAD_REFERENCE_DEPTH_GAUGE">
              Road reference depth gauge
            </option>
          </select>
        </label>
        <label>
          <input
            type="checkbox"
            checked={asphalt}
            disabled={busy}
            onChange={(e) => setAsphalt(e.target.checked)}
          />{" "}
          The surface is asphalt concrete.
        </label>
        <label>
          <input
            type="checkbox"
            checked={size}
            disabled={busy}
            onChange={(e) => setSize(e.target.checked)}
          />{" "}
          A circle at least 150 mm across fits inside the pothole opening.
        </label>
        <label>
          <input
            type="checkbox"
            checked={reference}
            disabled={busy}
            onChange={(e) => setReference(e.target.checked)}
          />{" "}
          Depth was measured from the surrounding healthy road surface.
        </label>
        <button disabled={busy} type="submit">
          Apply inspection measurement
        </button>
        {saved && (
          <button
            disabled={busy}
            type="button"
            onClick={() => void onSave(undefined)}
          >
            Remove inspection measurement
          </button>
        )}
      </form>
    </details>
  );
}
