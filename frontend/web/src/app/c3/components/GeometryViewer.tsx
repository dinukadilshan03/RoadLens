/* eslint-disable @next/next/no-img-element -- User-provided images are displayed without remote optimization. */
"use client";
import { useState } from "react";
import dynamic from "next/dynamic";
import type { Defect } from "../api";

const RelativeSurfaceViewer = dynamic(() => import("./RelativeSurfaceViewer"), {
  ssr: false,
  loading: () => <p className="c3-muted">Loading interactive surface…</p>,
});

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
  const [branchId, setBranchId] = useState(0);
  const branch = defect.crack_network?.branches.find((item) => item.branch_id === branchId);
  const [opacity, setOpacity] = useState(65);
  const [row, setRow] = useState(() =>
    Math.min(
      191,
      Math.floor(
        ((defect.source_bbox_xyxy[1] + defect.source_bbox_xyxy[3]) /
          2 /
          height) *
          192,
      ),
    ),
  );
  const map = defect.display_chart?.profiles;
  if (defect.damage_type === "crack")
    return (
      <>
        <p className="c3-muted">
          This view shows the location and shape of the identified crack.
        </p>
        {defect.crack_network && <>
          <label>Inspect a branch <select value={branchId} onChange={(event) => setBranchId(Number(event.target.value))}>
            <option value={0}>Entire network</option>
            {defect.crack_network.branches.map((item) => <option key={item.branch_id} value={item.branch_id}>Branch {item.branch_id}</option>)}
          </select></label>
          {branch && <div className="c3-image" style={{ aspectRatio: `${width}/${height}` }}>
            <img src={image} alt="Crack network" />
            <img src={branch.overlay} alt={`Highlighted branch ${branch.branch_id}`} style={{
              left: `${branch.bbox_xyxy[0] / width * 100}%`, top: `${branch.bbox_xyxy[1] / height * 100}%`,
              width: `${(branch.bbox_xyxy[2] - branch.bbox_xyxy[0]) / width * 100}%`,
              height: `${(branch.bbox_xyxy[3] - branch.bbox_xyxy[1]) / height * 100}%`,
            }} />
          </div>}
        </>}
        <div className="c3-visual-grid">
          {[
            { title: defect.source_kind === "mask" ? "Uploaded crack mask" : "Original crack road image", layer: undefined },
            { title: "Highlighted crack area", layer: defect.visuals.outline },
            { title: "Extracted crack outline", layer: defect.visuals.outline },
            { title: "Crack centerline", layer: defect.visuals.centerline },
            { title: "Visible width variation", layer: defect.visuals.width },
          ].map((view, i) => (
            <figure key={view.title}>
              <div
                className="c3-image"
                style={{ aspectRatio: `${width}/${height}` }}
              >
                {i < 2 && (
                  <img
                    src={image}
                    alt={
                      i === 0
                        ? (defect.source_kind === "mask" ? "Uploaded crack mask" : "Original crack road image")
                        : "Road with highlighted crack"
                    }
                  />
                )}
                {view.layer && <img src={view.layer} alt={view.title} />}
              </div>
              <figcaption>
                <b>{view.title}</b>
                {view.title === "Visible width variation" && <div className="c3-width-key">
                  <div className="c3-width-scale" aria-hidden="true" />
                  <div className="c3-width-labels"><strong>Blue / cool: narrower</strong><strong>Orange?red / warm: wider</strong></div>
                  <p>Colors show visible crack width, not damage severity.</p>
                </div>}
              </figcaption>
            </figure>
          ))}
        </div>

      </>
    );
  if (!map?.length)
    return <p>No depression map is available for this marked area.</p>;
  const values = map[row] || map[0];
  const points = values
    .map((v, x) => `${(x * 600) / 255},${15 + v * 130}`)
    .join(" ");
  const bins = defect.display_chart?.distribution || [];
  const peak = Math.max(1, ...bins);
  return (
    <>
      <div className="c3-visual-grid">
        <figure>
          <div
            className="c3-image"
            style={{ aspectRatio: `${width}/${height}` }}
          >
            <img src={image} alt="Road photograph" />
            <img src={defect.visuals.outline} alt="Highlighted pothole area" />
          </div>
          <figcaption>
            <b>Highlighted pothole area</b>
          </figcaption>
        </figure>
        <figure>
          <div
            className="c3-image"
            style={{ aspectRatio: `${width}/${height}` }}
          >
            <img
              src={defect.visuals.depression}
              alt="Predicted depression color map"
            />
          </div>
          <figcaption>
            <b>Predicted depression color map</b>
          </figcaption>
        </figure>
        <figure>
          <div
            className="c3-image"
            style={{ aspectRatio: `${width}/${height}` }}
          >
            <img src={image} alt="Original uploaded road" />
          </div>
          <figcaption>
            <span>01 / Source image</span>
            <b>Original road image</b>
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
            <img
              src={defect.visuals.depression}
              style={{ opacity: opacity / 100 }}
              alt="Estimated road depression pattern"
            />
            <span
              className="c3-scanline"
              style={{ top: `${(row / 192) * 100}%` }}
            />
            <span className="c3-map-badge">Estimated shape</span>
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
          <h3>Road Depression Profile</h3>
          <p>
            Explore how the estimated road surface changes across the selected
            pothole. Click or drag vertically on the image to move the profile
            line.
          </p>
        </div>
        <label>
          Profile position: {Math.round((row / 191) * 100)}%
          <input
            aria-label="Profile position"
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
          aria-label="Healthy reference level and estimated damaged road profile at the selected image position"
        >
          <path
            d="M0 15H600 M0 80H600 M0 145H600"
            stroke="#ced6c5"
            fill="none"
          />
          <line x1="0" y1="15" x2="600" y2="15" stroke="#7c8486" strokeWidth="2" strokeDasharray="8 5" />
          <polygon points={`0,15 ${points} 600,15`} fill="#75927830" />
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
            Position across image
          </text>
          <text x="5" y="12">
            Relative depression
          </text>
        </svg>
        <div className="c3-profile-legend" aria-label="Road profile legend">
          <span><i className="c3-reference-line" />Healthy reference level</span>
          <span><i className="c3-damaged-line" />Estimated damaged profile</span>
        </div>
      </div>
      <h3>Distribution of estimated depression</h3>
      <div
        className="c3-histogram"
        role="img"
        aria-label="Distribution of estimated relative depression, shallower to deeper"
      >
        {bins.map((v, i) => (
          <div
            key={i}
            style={{ height: `${Math.max(1, (v / peak) * 100)}%` }}
          />
        ))}
      </div>
      <p className="c3-muted">
        Shallower to deeper estimated regions. Colors illustrate the estimated
        shape of the road depression. Warmer colors indicate areas predicted to
        be deeper relative to the surrounding road.
      </p>
      {defect.relative_surface ? (
        <RelativeSurfaceViewer surface={defect.relative_surface} />
      ) : (
        <p className="c3-muted">
          A 3D surface is not available for this marked area at the model’s
          image resolution.
        </p>
      )}
    </>
  );
}
