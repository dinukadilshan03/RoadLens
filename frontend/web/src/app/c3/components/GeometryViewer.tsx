/* eslint-disable @next/next/no-img-element -- User-provided images are displayed without remote optimization. */
"use client";
import { useEffect, useRef, useState } from "react";
import type { Defect } from "../api";

export default function GeometryViewer({
  defect,
  image,
  width,
  height,
}: {
  defect: Defect;
  image: string;
  width: number;
  height: number;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [opacity, setOpacity] = useState(65);
  const [row, setRow] = useState(96);
  const map = defect.raw_axial_depression_map;
  const max = defect.geometry_raw?.max_raw || 1;
  useEffect(() => {
    const context = canvas.current?.getContext("2d");
    if (!context || !map?.length) return;
    const pixels = context.createImageData(256, 192);
    map.forEach((line, y) =>
      line.forEach((value, x) => {
        const t = Math.min(1, value / max),
          i = (y * 256 + x) * 4;
        pixels.data[i] = Math.round(70 + 185 * t);
        pixels.data[i + 1] = Math.round(160 - 105 * t);
        pixels.data[i + 2] = Math.round(150 - 110 * t);
        pixels.data[i + 3] = value > 0 ? 255 : 0;
      }),
    );
    context.putImageData(pixels, 0, 0);
  }, [map, max]);
  if (!map?.length) return <p>No prediction map available for this defect.</p>;
  const values = map[row] || map[0];
  const points = values
    .map((v, x) => `${(x * 600) / 255},${145 - (v / max) * 130}`)
    .join(" ");
  const bins = Array(16).fill(0) as number[];
  map
    .flat()
    .filter((v) => v > 0)
    .forEach((v) => bins[Math.min(15, Math.floor((v / max) * 16))]++);
  const peak = Math.max(1, ...bins);
  return (
    <>
      <div className="c3-visual-grid">
        <figure>
          <div className="c3-image" style={{ aspectRatio: `${width}/${height}` }}>
            <img src={image} alt="Original uploaded road" />
          </div>
          <figcaption>
            <span>01 / Source image</span>
            <b>Original pothole</b>
          </figcaption>
        </figure>
        <figure>
          <div
            className="c3-image c3-interactive"
            style={{ aspectRatio: `${width}/${height}` }}
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId);
              const r = e.currentTarget.getBoundingClientRect();
              setRow(
                Math.max(
                  0,
                  Math.min(
                    191,
                    Math.floor(((e.clientY - r.top) / r.height) * 192),
                  ),
                ),
              );
            }}
            onPointerMove={(e) => {
              if (!e.buttons) return;
              const r = e.currentTarget.getBoundingClientRect();
              setRow(
                Math.max(
                  0,
                  Math.min(
                    191,
                    Math.floor(((e.clientY - r.top) / r.height) * 192),
                  ),
                ),
              );
            }}
          >
            <img src={image} alt="Road aligned with predicted depression" />
            <canvas
              ref={canvas}
              width={256}
              height={192}
              style={{ opacity: opacity / 100 }}
              aria-label="Model-produced RAW depression heatmap"
            />
            <span
              className="c3-scanline"
              style={{ top: `${(row / 192) * 100}%` }}
            />
            <span className="c3-map-badge">CRDL · RAW</span>
          </div>
          <figcaption>
            <span>02 / Trained prediction</span>
            <b>Predicted depression map</b>
          </figcaption>
        </figure>
      </div>
      <div className="c3-controls">
        <label>
          Heatmap opacity{" "}
          <input
            type="range"
            min="0"
            max="100"
            value={opacity}
            onChange={(e) => setOpacity(+e.target.value)}
          />
          {opacity}%
        </label>
        <span className="c3-legend">
          Lower <i /> Higher relative depression
        </span>
      </div>
      <div className="c3-profile">
        <div>
          <h3>Interactive depression profile</h3>
          <p>
            Click or drag vertically on the map to inspect a horizontal
            cross-section.
          </p>
        </div>
        <label>
          Profile row · {row}
          <input
            aria-label="Profile row"
            type="range"
            min="0"
            max="191"
            value={row}
            onChange={(e) => setRow(+e.target.value)}
          />
        </label>
        <svg
          viewBox="0 0 620 180"
          role="img"
          aria-label={`Predicted RAW depression profile at native map row ${row}`}
        >
          <path
            d="M0 15H600 M0 80H600 M0 145H600"
            stroke="#ced6c5"
            fill="none"
          />
          <polygon points={`0,145 ${points} 600,145`} fill="#75927830" />
          <polyline
            points={points}
            fill="none"
            stroke="#436c59"
            strokeWidth="2"
          />
          <text x="0" y="172">
            0
          </text>
          <text x="510" y="172">
            256 native pixels
          </text>
          <text x="5" y="12">
            {max.toFixed(2)} RAW
          </text>
        </svg>
      </div>
      <details className="c3-research">
        <summary>Research view · distribution & model evidence</summary>
        <p>
          Native 192 × 256 optical-axis residual. The visualization stretches to
          the full image; measurements use the native prediction.
        </p>
        <div
          className="c3-histogram"
          role="img"
          aria-label="Distribution of positive predicted RAW depression values"
        >
          {bins.map((v, i) => (
            <div
              key={i}
              style={{ height: `${Math.max(1, (v / peak) * 100)}%` }}
              title={`${((max * i) / 16).toFixed(2)}–${((max * (i + 1)) / 16).toFixed(2)} RAW: ${v} map pixels`}
            />
          ))}
        </div>
        <p>
          Distribution: 0 to {max.toFixed(2)} RAW. Zero-valued background
          excluded.
        </p>
        <p>{defect.measurement_notes}</p>
        {defect.pixel_measurements.crack && (
          <pre>{JSON.stringify(defect.pixel_measurements.crack, null, 2)}</pre>
        )}
        <p>
          RGB-D reference reconstruction is unavailable for ordinary RGB
          uploads.
        </p>
      </details>
    </>
  );
}
