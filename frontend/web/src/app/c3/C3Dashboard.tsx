/* eslint-disable @next/next/no-img-element -- Local uploaded images use browser data URLs. */
"use client";
import Image from "next/image";
import Link from "next/link";
import { useEffect, useState } from "react";
import { analyzeSurface, type Analysis } from "./api";
import GeometryViewer from "./components/GeometryViewer";

function usePreview(file?: File) {
  const [url, setUrl] = useState("");
  useEffect(() => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setUrl(String(reader.result));
    reader.readAsDataURL(file);
    return () => reader.abort();
  }, [file]);
  return file ? url : "";
}
export default function C3Dashboard() {
  const [image, setImage] = useState<File>();
  const [mask, setMask] = useState<File>();
  const [crack, setCrack] = useState<File>();
  const [result, setResult] = useState<Analysis>();
  const [selected, setSelected] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const imageUrl = usePreview(image),
    maskUrl = usePreview(mask);
  const defect = result?.defects[selected];
  function invalidate() {
    setResult(undefined);
    setError("");
    setSelected(0);
  }
  async function analyze() {
    if (!image || !mask) return;
    setBusy(true);
    setError("");
    setResult(undefined);
    try {
      setResult(await analyzeSurface(image, mask, crack));
      setSelected(0);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to analyze image");
    } finally {
      setBusy(false);
    }
  }
  function exportReport() {
    const blob = new Blob([JSON.stringify(result, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob),
      a = document.createElement("a");
    a.href = url;
    a.download = "roadlens-surface-analysis.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const stages = ["Input", "Mask", "Geometry", "Analysis", "Results"];
  return (
    <div className="c3-workspace">
      <header className="c3-header">
        <Link href="/" aria-label="RoadLens home">
          <Image
            src="/logo.png"
            width={210}
            height={70}
            alt="RoadLens"
            style={{ height: "auto" }}
          />
        </Link>
        <nav aria-label="Main navigation">
          <Link href="/">Home</Link>
          <Link href="/c1">C1 · Detect</Link>
          <Link href="/c2">C2 · Segment</Link>
          <Link href="/c3" aria-current="page">
            C3 · Measure
          </Link>
          <Link href="/c4">C4 · Assess</Link>
        </nav>
        <span className="c3-pill">SURFACE INTELLIGENCE</span>
      </header>
      <main className="c3-main">
        <section className="c3-intro">
          <div>
            <p className="c3-eyebrow">
              COMPONENT 03 / A CLOSER LOOK BENEATH THE SURFACE
            </p>
            <h1>
              Road Surface
              <br />
              <em>Intelligence.</em>
            </h1>
            <p>
              Adaptive surface reconstruction & geometric depression analysis.
              <br />
              From a road image to a more informed surface review.
            </p>
          </div>
          <div className="c3-intro-note">
            <span>03</span>
            <p>
              See the shape.
              <br />
              Understand the limits.
            </p>
            <small>
              Trained CRDL inference
              <br />
              RAW geometry · Pixel measurements
            </small>
          </div>
        </section>
        <section className="c3-workflow" aria-label="Analysis workflow">
          <div>
            <b>Analysis workflow</b>
            <span>5 processing stages</span>
          </div>
          <ol>
            {stages.map((stage, i) => (
              <li
                key={stage}
                className={
                  result
                    ? "complete"
                    : busy && i === 2
                      ? "active"
                      : i === 0
                        ? "active"
                        : ""
                }
              >
                <span>{result ? "✓" : i + 1}</span>
                {stage}
              </li>
            ))}
          </ol>
        </section>
        <section className="c3-panel">
          <div className="c3-section-title">
            <div>
              <p className="c3-eyebrow">01 — ROAD IMAGE INPUT</p>
              <h2>Bring the surface into view.</h2>
            </div>
            <span className="c3-pill">Manual now · Automatic later</span>
          </div>
          <p className="c3-muted">
            Upload a road image and a pothole mask. Masks are automatically resized
            and converted to binary. Use light damage regions on a dark background.
          </p>
          <div className="c3-upload-grid">
            {[
              {
                title: "Upload road image",
                hint: "Common image formats · up to 15 MB",
                file: image,
                accept: "image/*",
                set: setImage,
              },
              {
                title: "Upload pothole mask",
                hint: "Image mask · automatically prepared",
                file: mask,
                accept: "image/*",
                set: setMask,
              },
              {
                title: "Crack segmentation mask",
                hint: "Optional · automatically resized",
                file: crack,
                accept: "image/*",
                set: setCrack,
              },
            ].map((item, i) => (
              <label className="c3-upload" key={item.title}>
                <span className="c3-upload-icon">{i === 0 ? "↥" : "⊞"}</span>
                <b>{item.title}</b>
                <span>{item.file?.name || item.hint}</span>
                <input
                  aria-label={item.title}
                  type="file"
                  accept={item.accept}
                  disabled={busy}
                  onChange={(e) => {
                    item.set(e.target.files?.[0]);
                    invalidate();
                  }}
                />
              </label>
            ))}
          </div>
          {imageUrl && (
            <div className="c3-input-preview">
              <figure>
                <img src={imageUrl} alt="Selected road input" />
                <figcaption>Road image</figcaption>
              </figure>
              {maskUrl && (
                <figure>
                  <img src={maskUrl} alt="Selected binary segmentation mask" />
                  <figcaption>Pothole segmentation</figcaption>
                </figure>
              )}
            </div>
          )}
          <div className="c3-submit">
            <p>
              Physical scale is unverified.
              <br />
              <span>Results support review in RAW and pixel units.</span>
            </p>
            <button disabled={!image || !mask || busy} onClick={analyze}>
              {busy ? "Analyzing with trained CRDL…" : "Analyze Road Surface"}{" "}
              <span>↗</span>
            </button>
          </div>
          <div aria-live="polite">
            {busy && (
              <p className="c3-notice">
                Inputs are being validated and processed by the geometry model.
                Analysis and results will appear when inference completes.
              </p>
            )}
            {error && (
              <p className="c3-error" role="alert">
                {error}
              </p>
            )}
          </div>
        </section>
        <section className="c3-panel" aria-busy={busy}>
          <div className="c3-section-title">
            <div>
              <p className="c3-eyebrow">02 — VISUAL GEOMETRY ANALYSIS</p>
              <h2>A clearer picture of the depression.</h2>
            </div>
            {result && (
              <label>
                Defect{" "}
                <select
                  value={selected}
                  onChange={(e) => setSelected(+e.target.value)}
                >
                  {result.defects.map((d, i) => (
                    <option value={i} key={d.defect_id}>
                      {d.defect_id}
                    </option>
                  ))}
                </select>
              </label>
            )}
          </div>
          {defect && result ? (
            <>
              <div className="c3-segmentation">
                <h3>Segmentation & bounding box</h3>
                <div
                  className="c3-segment-frame"
                  style={{
                    aspectRatio: `${result.frame_width}/${result.frame_height}`,
                  }}
                >
                  <img
                    src={imageUrl}
                    alt="Uploaded road with supplied segmentation"
                  />
                  {maskUrl && (
                    <img
                      className="c3-mask-layer"
                      src={maskUrl}
                      alt="Aligned supplied binary mask overlay"
                    />
                  )}
                  <span
                    style={{
                      left: `${(defect.source_bbox_xyxy[0] / result.frame_width) * 100}%`,
                      top: `${(defect.source_bbox_xyxy[1] / result.frame_height) * 100}%`,
                      width: `${((defect.source_bbox_xyxy[2] - defect.source_bbox_xyxy[0]) / result.frame_width) * 100}%`,
                      height: `${((defect.source_bbox_xyxy[3] - defect.source_bbox_xyxy[1]) / result.frame_height) * 100}%`,
                    }}
                  />
                </div>
                <p>
                  Supplied mask · bounding box derived from nonzero pixels.
                  Segmentation is provided by your upload.
                </p>
              </div>
              <GeometryViewer
                key={defect.defect_id}
                defect={defect}
                image={imageUrl}
                width={result.frame_width}
                height={result.frame_height}
              />
            </>
          ) : (
            <div className="c3-empty">
              <span>⌁</span>
              <h3>The details will take shape here.</h3>
              <p>
                Analyze your road image to explore the actual depression map,
                <br />
                linked profile and geometric measurements.
              </p>
              <small>
                Source image → Supplied segmentation → Trained inference
              </small>
            </div>
          )}
        </section>
        <section className="c3-panel c3-results">
          <div className="c3-section-title">
            <div>
              <p className="c3-eyebrow">03 — FINAL ANALYSIS RESULTS</p>
              <h2>Measurements with context.</h2>
            </div>
            {result && (
              <div className="c3-export">
                <button onClick={exportReport}>Export JSON ↓</button>
                <button onClick={() => window.print()}>Print report ↗</button>
              </div>
            )}
          </div>
          <div className="c3-metrics">
            {[
              {
                name: "D95 depth",
                value: defect?.geometry_raw?.d95_raw,
                unit: "RAW",
                help: "95th percentile of predicted depression within the model-space pothole mask. 95% of values are at or below this depth.",
              },
              {
                name: "Mean depth",
                value: defect?.geometry_raw?.mean_raw,
                unit: "RAW",
                help: "Average predicted depression within the model-space pothole mask.",
              },
              {
                name: "Maximum depth",
                value: defect?.geometry_raw?.max_raw,
                unit: "RAW",
                help: "Largest predicted depression value within the model-space pothole mask.",
              },
              {
                name: "Footprint",
                value: defect?.pixel_measurements.footprint_area_px2,
                unit: "px²",
                help: "Number of nonzero pixels in the supplied full-resolution pothole mask. This is not physical area.",
              },
            ].map((m) => (
              <article key={m.name}>
                <div>
                  {m.name}
                  <details>
                    <summary aria-label={`Explain ${m.name}`}>ⓘ</summary>
                    <p>{m.help}</p>
                  </details>
                </div>
                <strong>
                  {m.value == null
                    ? "—"
                    : m.value.toLocaleString(undefined, {
                        maximumFractionDigits: 2,
                      })}
                </strong>
                <span>{m.unit}</span>
              </article>
            ))}
          </div>
          <div className="c3-notice">
            <b>Review only — physical scale unverified</b>
            <p>
              Model-predicted physical severity is unavailable. Measurements are
              provided in RAW and pixel units.
            </p>
          </div>
          {defect && (
            <details className="c3-research">
              <summary>Reliability & provenance</summary>
              <p>{defect.reliability_status.replaceAll("_", " ")}</p>
              <ul>
                {[...defect.reasons, ...defect.reliability.input_qa_flags].map(
                  (reason) => (
                    <li key={reason}>
                      {reason.replaceAll("_", " ").toLowerCase()}
                    </li>
                  ),
                )}
              </ul>
              <p>Model bundle: {result?.bundle_version}</p>
            </details>
          )}
        </section>
        <footer className="c3-footer">
          <Link href="/">← Back to RoadLens</Link>
          <p>Road damage. Surface detail. A more informed assessment.</p>
          <span>COMPONENT 03</span>
        </footer>
      </main>
    </div>
  );
}
