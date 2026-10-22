/* eslint-disable @next/next/no-img-element -- Uploaded photo and exact crack-mask overlays. */
"use client";

import { useState } from "react";
import type { Defect } from "../api";

export default function CrackAreaResults({ defect, image }: { defect: Defect; image: string }) {
  const findings = defect.crack_network?.regional_visual_findings || [];
  const [selected, setSelected] = useState(findings[0]?.region_id);
  const finding = findings.find((item) => item.region_id === selected);
  const region = defect.crack_network?.regions?.find((item) => item.region_id === selected);
  const width = defect.source_width || 1, height = defect.source_height || 1;
  const box = region?.bbox_xyxy;
  const prediction = finding?.prediction;
  const review = !!prediction?.review_required || prediction?.visual_crack_severity === "NEEDS_REVIEW";
  const label = !prediction ? "Unavailable" : review ? "Needs review" : prediction.visual_crack_severity?.toLowerCase() || "Unavailable";
  return <details className="c3-regional-findings">
    <summary>View individual area results</summary>
    <p>Select an area to see its location highlighted in red.</p>
    <div className="c3-area-buttons" role="group" aria-label="Choose a crack area">
      {findings.map((item) => <button key={item.region_id} aria-pressed={selected === item.region_id}
        onClick={() => setSelected(item.region_id)}>Area {item.region_id}</button>)}
    </div>
    {image && <div className="c3-image c3-area-image" style={{ aspectRatio: `${width}/${height}` }}>
      <img src={image} alt={`Crack photograph showing selected area ${selected}`} />
      {region?.overlay && box && <img src={region.overlay} alt={`Area ${selected} crack mask highlighted in red`} style={{
        left: `${box[0] / width * 100}%`, top: `${box[1] / height * 100}%`,
        width: `${(box[2] - box[0]) / width * 100}%`, height: `${(box[3] - box[1]) / height * 100}%`,
      }} />}
      {box && <span className="c3-area-box" style={{
        left: `${box[0] / width * 100}%`, top: `${box[1] / height * 100}%`,
        width: `${(box[2] - box[0]) / width * 100}%`, height: `${(box[3] - box[1]) / height * 100}%`,
      }} />}
      <span className="c3-area-caption">Area {selected} — highlighted in red</span>
    </div>}
    {!region && <p>Re-run analysis to show this area on the image.</p>}
    <div className={review || !prediction ? "c3-review-reason" : "c3-area-result"} aria-live="polite">
      <strong>Area {selected}: {label}</strong>
      {review && <p><strong>Why:</strong> The model could not confidently choose between low and high visual severity for this area. Review the red-marked area in the photo.</p>}
      {!prediction && <p><strong>Why:</strong> The visual assessment could not be completed for this area. Try analyzing again with its matching road photo.</p>}
    </div>
  </details>;
}
