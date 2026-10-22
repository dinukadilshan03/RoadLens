/* eslint-disable @next/next/no-img-element -- Server-provided research figures. */
"use client";

import { useState } from "react";

type Evidence = {
  gate: { frozen_dev_mae_raw: Record<string, number>; frozen_dev_n: number };
  evaluation: { test_count: number; test_macro_f1: number; test_accuracy: number };
};

export default function ResearchEvidence() {
  const [evidence, setEvidence] = useState<Evidence>();
  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(false);
  const [opened, setOpened] = useState(false);
  async function load() {
    if (evidence || loading) return;
    setLoading(true);
    setError(false);
    try {
      const responses = await Promise.all([
        fetch("/api/v1/c3/research/gate"),
        fetch("/api/v1/c3/research/pothole-evaluation"),
      ]);
      if (responses.some((response) => !response.ok)) throw new Error();
      const [gate, evaluation] = await Promise.all(responses.map((response) => response.json()));
      setEvidence({ gate, evaluation });
    } catch {
      setError(true);
    } finally {
      setLoading(false);
    }
  }
  return (
    <details className="c3-panel" onToggle={(event) => {
      if (event.currentTarget.open) setOpened(true);
    }}>
      <summary>Research View — C3-28 learned surface and RGB-D comparison</summary>
      <p className="c3-muted">
        Explore the saved C3-28 experimental reconstruction example. These figures
        show a research pothole, separate from the selected uploaded pothole.
      </p>
      {opened && <>
        <h3>From the road image to the estimated surface</h3>
        <figure className="c3-research-figure">
          <img loading="lazy" src="/api/v1/c3/research/stages"
            alt="C3-28 research example: RGB photograph, pothole mask, RGB-D fitted reference, learned healthy reference, CRDL depression and combined surface" />
          <figcaption>
            Actual C3-28 six-stage research output: compare the RGB-D fitted healthy
            reference (C) with the learned healthy reference (D), then inspect
            the estimated depression (E) and combined relative surface (F).
          </figcaption>
        </figure>
        <h3>Healthy reference and damaged surface</h3>
        <figure className="c3-research-figure">
          <img loading="lazy" src="/api/v1/c3/research/comparison"
            alt="C3-28 experimental learned healthy-road surface beside its combined estimated damaged surface" />
          <figcaption>
            The saved example shows a sloping relative healthy-road reference on
            the left and a localized depression in the estimated damaged surface
            on the right. The learned model can represent tilt and smooth curvature
            in image coordinates. This describes the research example, not a
            measured slope or curvature of your uploaded road.
          </figcaption>
        </figure>
        <details className="c3-research-evaluation" onToggle={(event) => {
          if (event.currentTarget.open) void load();
        }}>
          <summary>Model evaluation and research context</summary>
          <p className="c3-muted">
            The default pothole view retains the flat zero-level healthy reference.
            C3-28 remains experimental; its surfaces use relative RAW units.
          </p>
          {loading && <p>Loading evaluation…</p>}
          {error && <p role="alert">Evaluation is unavailable. Close and reopen this section to retry.</p>}
          {evidence && <>
            <h3>C3-22 pothole visual severity</h3>
            <p>Held-out test: {evidence.evaluation.test_count} images; macro F1 {evidence.evaluation.test_macro_f1.toFixed(3)};
              accuracy {evidence.evaluation.test_accuracy.toFixed(3)}. These are dataset evaluation results.</p>
            <h3>C3-28 frozen-development comparison</h3>
            <p>{evidence.gate.frozen_dev_n} samples. Lower relative error is better.</p>
            <table>
              <thead><tr><th>Reference</th><th>Mean absolute error (RAW)</th></tr></thead>
              <tbody>{[
                ["Trained C3-28", "trained_ref_mae_raw"],
                ["Flat baseline", "flat_mae_raw"],
                ["Training-mean baseline", "train_mean_mae_raw"],
              ].map(([label, key]) => <tr key={key}><td>{label}</td><td>{evidence.gate.frozen_dev_mae_raw[key].toFixed(3)}</td></tr>)}</tbody>
            </table>
            <p>The trained reference did not outperform the training-mean baseline;
              live deployment remains disabled. Independent scene validation and
              metric depth calibration remain unestablished.</p>
          </>}
        </details>
      </>}
    </details>
  );
}
