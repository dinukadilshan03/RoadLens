"use client";

import CrackAreaResults from "./CrackAreaResults";
﻿import type { Analysis, Defect } from "../api";

function severityText(defect: Defect) {
  if (defect.source_kind === "mask") return "Photo needed";
  if (defect.presentation.severity === "Needs Review") return "Needs review";
  return defect.presentation.severity;
}

function nextStep(defect: Defect) {
  if (defect.source_kind === "mask") return "Add the road photo that matches this crack mask.";
  if (defect.presentation.reliability === "Insufficient input") return "Upload a closer, sharper photo and its matching mask.";
  if (defect.presentation.severity === "Needs Review") return "Check the marked area and arrange a closer inspection.";
  return "Use this result to plan an on-site inspection.";
}

function reviewPoints(defect: Defect) {
  const points = new Set<string>();
  if (defect.source_kind === "mask") points.add("The mask shows the crack shape. A matching photo is needed to assess its appearance.");
  if (defect.presentation.reliability === "Insufficient input") points.add("The marked area is too small for a clear result.");
  const reasons = defect.presentation.review_reasons.join(" ");
  if (reasons.includes("touching the edge")) points.add("Some damage may continue beyond the edge of the photo.");
  if (defect.source_kind !== "mask" && reasons.includes("aggregation")) points.add("The crack model assesses individual areas. A reliable severity rating for the whole connected network has not yet been established.");
  if (reasons.includes("unavailable")) points.add("The visual severity assessment is currently unavailable.");
  if (reasons.includes("could not choose")) points.add("The photo does not give a clear severity result.");
  if (!points.size) points.add(defect.presentation.severity === "Needs Review"
    ? "Check that the marked area matches the damage in the photo."
    : "Image analysis is complete. Confirm the condition during an on-site inspection.");
  return [...points];
}

export default function AssessmentSummary({ result, mode, imageFor }: { result?: Analysis; mode: "findings" | "guidance"; imageFor?: (defect: Defect) => string }) {
  if (!result) return <p className="c3-muted">Your {mode === "findings" ? "findings" : "next steps"} will appear after analysis.</p>;
  return <div className="c3-summary-grid">
    {result.defects.map((defect) => <article className="c3-summary-card" key={defect.defect_id}>
      <header className="c3-summary-heading">
        <h3>{defect.presentation.label}</h3>
        {mode === "findings" && <span className="c3-result-badge" data-level={severityText(defect).toLowerCase()}>{severityText(defect)}</span>}
      </header>
      {mode === "findings" ? <ul className="c3-summary-points">
        <li><strong>{defect.damage_type === "crack" ? "Crack shape" : "Surface shape"}:</strong>{" "}
          {defect.damage_type === "crack" ? "Visible crack paths and width changes analyzed." : "Estimated pothole depression mapped."}</li>
        {defect.crack_network && <>
          <li><strong>Coverage:</strong> {defect.crack_network.connected_regions} connected {defect.crack_network.connected_regions === 1 ? "region" : "regions"}, assessed as one network.</li>
          <li><strong>Pattern complexity:</strong> {defect.crack_network.complexity.label === "Needs Review" ? "Not available yet" : defect.crack_network.complexity.label}.</li>
        </>}
        {defect.source_kind !== "mask" && <li><strong>Visual severity:</strong> {severityText(defect)}.</li>}
        {defect.presentation.measurement && <li>{defect.presentation.measurement}</li>}
      </ul> : <>
        <ul className="c3-summary-points">{reviewPoints(defect).map((point) => <li key={point}>{point}</li>)}</ul>
        <div className="c3-next-step"><strong>Next step</strong><p>{nextStep(defect)}</p></div>
      </>}
      {mode === "findings" && defect.damage_type === "crack" && (defect.presentation.severity === "Needs Review" || defect.source_kind === "mask") && <aside className="c3-review-reason">
        <strong>Why review is needed</strong>
        <ul className="c3-summary-points">{reviewPoints(defect).map((point) => <li key={point}>{point}</li>)}</ul>
      </aside>}
      {mode === "findings" && !!defect.crack_network?.regional_visual_findings.length &&
        <CrackAreaResults defect={defect} image={imageFor?.(defect) || ""} />}

    </article>)}
  </div>;
}
