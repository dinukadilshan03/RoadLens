/* eslint-disable @next/next/no-img-element -- Local uploaded images and backend visualizations. */
"use client";
import Image from "next/image";
import Link from "next/link";
import { useEffect, useState } from "react";
import {
  analyzeSurface,
  downloadReport,
  type Analysis,
} from "./api";
import GeometryViewer from "./components/GeometryViewer";
import ResearchEvidence from "./components/ResearchEvidence";
import AssessmentSummary from "./components/AssessmentSummary";

function usePreview(file?: File) {
  const [preview, setPreview] = useState<{ file: File; url: string }>();
  useEffect(() => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setPreview({ file, url: String(reader.result) });
    reader.readAsDataURL(file);
    return () => reader.abort();
  }, [file]);
  return preview?.file === file ? preview?.url || "" : "";
}
export default function C3Dashboard() {
  const [image, setImage] = useState<File>();
  const [mask, setMask] = useState<File>();
  const [crack, setCrack] = useState<File>();
  const [crackImage, setCrackImage] = useState<File>();
  const [crackMatchesImage, setCrackMatchesImage] = useState(false);
  const [result, setResult] = useState<Analysis>();
  const [selected, setSelected] = useState(0);
  const [busy, setBusy] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState("");
  const [dimensions, setDimensions] = useState<Record<string, { width: number; height: number }>>({});
  const imageUrl = usePreview(image),
    maskUrl = usePreview(mask),
    crackUrl = usePreview(crack),
    crackImageUrl = usePreview(crackImage);
  const defect = result?.defects[selected];
  const sourceWidth = defect?.source_width || result?.frame_width || 1;
  const sourceHeight = defect?.source_height || result?.frame_height || 1;
  const sourceImage = defect?.source_group === "crack" ? crackImageUrl || (defect?.source_kind === "photo" ? imageUrl : crackUrl) : imageUrl;
  const sourceDefects = result?.defects.filter((d) => d.source_group === defect?.source_group) || [];
  function invalidate() {
    setResult(undefined);
    setError("");
    setSelected(0);
  }
  async function analyze() {
    if (!(image && mask) && !crack) return;
    setBusy(true);
    setError("");
    setResult(undefined);
    try {
      setResult(await analyzeSurface(image, mask, crack, undefined, crackImage, crackMatchesImage));
      setSelected(0);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to analyze");
    } finally {
      setBusy(false);
    }
  }
  async function exportReport() {
    if (!result) return;
    setExporting(true);
    setError("");
    try {
      await downloadReport(image, mask, crack, undefined, crackImage, crackMatchesImage);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to create report");
    } finally {
      setExporting(false);
    }
  }
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
          <Link href="/c1">C1 - Detect</Link>
          <Link href="/c3" aria-current="page">
            C3 - Damage assessment
          </Link>
          <Link href="/c4">C4 - Assess</Link>
        </nav>
        <span className="c3-pill">SURFACE INTELLIGENCE</span>
      </header>
      <main className="c3-main">
        <section className="c3-intro">
          <div>
            <p className="c3-eyebrow">
              COMPONENT 03 / DAMAGE ASSESSMENT
            </p>
            <h1>
              See the damage.
              <br />
              <em>Understand the surface.</em>
            </h1>
            <p>
              Explore potholes and cracks through clear visual analysis.
              <br />
              Make your next road inspection more informed.
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
              Damage severity and surface shape
              <br />
              from your photo and marked areas.
            </small>
          </div>
        </section>
        <section className="c3-panel">
          <p className="c3-eyebrow">01 / ROAD IMAGE INPUT</p>
          <h2>Examine Your Road Surface</h2>
          <p className="c3-muted">
            Upload one road photograph and its crack mask for one network assessment.
            Pothole masks are optional. For a mask from a different photograph,
            supply its matching crack photograph. A mask is a black-and-white image with
            the damage marked in white. Use PNG masks.
          </p>
          <div className="c3-upload-grid">
            {[
              {
                title: "Road photograph",
                file: image,
                set: setImage,
                hint: "Pairs with damage masks from this photograph - up to 15 MB",
              },
              {
                title: "Pothole damage mask",
                file: mask,
                set: setMask,
                hint: "Must match the pothole photograph",
              },
              {
                title: "Crack damage mask",
                file: crack,
                set: setCrack,
                hint: "One mask is analyzed as one complete crack network",
              },
            ].map((item) => (
              <label className="c3-upload" key={item.title}>
                <b>{item.title}</b>
                <span>{item.file?.name || item.hint}</span>
                <input
                  type="file"
                  accept="image/*"
                  aria-label={item.title}
                  disabled={busy || exporting}
                  onChange={(e) => {
                    item.set(e.target.files?.[0]);
                    invalidate();
                  }}
                />
                {item.file && (
                  <button
                    disabled={busy || exporting}
                    onClick={(e) => {
                      e.preventDefault();
                      item.set(undefined);
                      invalidate();
                    }}
                  >
                    Remove
                  </button>
                )}
              </label>
            ))}
          </div>
          {crack && image && mask && <label className="c3-notice">
            <input type="checkbox" checked={crackMatchesImage} disabled={busy || exporting}
              onChange={(event) => { setCrackMatchesImage(event.target.checked); invalidate(); }} />
            The crack mask belongs to the road photograph above
          </label>}
          {crack && <details className="c3-notice">
            <summary>Use a different matching crack photograph (optional)</summary>
            <p>The crack mask already provides shape analysis. The road photograph above is used automatically for crack-only input. With a pothole mask too, confirm that both masks belong to that photograph. Otherwise add the matching crack photograph here. Multi-branch networks require review; regional predictions do not establish overall severity.</p>
            <label>Crack road photograph
              <input type="file" accept="image/*" disabled={busy || exporting} onChange={(event) => { setCrackImage(event.target.files?.[0]); invalidate(); }} />
            </label>
            {crackImage && <button disabled={busy || exporting} onClick={() => { setCrackImage(undefined); invalidate(); }}>Remove crack photo</button>}
          </details>}
          <div className="c3-input-preview">
            {[
              { url: imageUrl, label: "Road photograph", photo: "" },
              { url: maskUrl, label: "Pothole marked area", photo: imageUrl },
              { url: crackUrl, label: "Crack mask", photo: crackImageUrl || ((crackMatchesImage || !mask) ? imageUrl : "") },
              { url: crackImageUrl, label: "Crack photograph", photo: "" },
            ]
              .filter((p) => p.url)
              .map((p) => (
                <figure key={p.label}>
                  <img src={p.url} alt={p.label} onLoad={(event) => {
                    const { naturalWidth: width, naturalHeight: height } = event.currentTarget;
                    setDimensions((current) => current[p.url]?.width === width && current[p.url]?.height === height
                      ? current : { ...current, [p.url]: { width, height } });
                  }} />
                  <figcaption>{p.label}{dimensions[p.url] && ` (${dimensions[p.url].width} x ${dimensions[p.url].height} pixels)`}</figcaption>
                  {p.photo && dimensions[p.url] && dimensions[p.photo] &&
                    (dimensions[p.url].width !== dimensions[p.photo].width || dimensions[p.url].height !== dimensions[p.photo].height) &&
                    <p className="c3-error">This mask has a different size from the photo. Replace it with the matching mask. Other valid masks can still be analyzed.</p>}
                </figure>
              ))}
          </div>
          <div className="c3-submit">
            <p>
              Analyze a road photograph with its crack mask, a mask alone, or pothole and crack inputs together.
              <br />
              <span>Each crack mask produces one network assessment.</span>
            </p>
            <button
              disabled={(!(image && mask) && !crack) || busy || exporting}
              onClick={analyze}
            >
              {busy ? "Analyzing road surface..." : "Analyze Road Surface"}
            </button>
          </div>
          <div aria-live="polite">
            {result?.input_warnings.map((warning) => <p className="c3-notice" key={warning.damage_type} role="status">{warning.message} Results below cover only the masks that were analyzed.</p>)}
            {busy && (
              <p className="c3-notice">
                Checking the image and analyzing the supplied damaged areas...
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
          <p className="c3-eyebrow">02 / DAMAGE IDENTIFICATION</p>
          <h2>Damage Found on Your Road</h2>
          {result && defect ? (
            <>
              <p className="c3-muted">
                Select a marked area to explore its findings. Potholes are
                outlined in amber; cracks in blue.
              </p>
              <div
                className="c3-defect-tabs"
                role="group"
                aria-label="Select damaged area"
              >
                {result.defects.map((d, i) => (
                  <button
                    key={d.defect_id}
                    aria-pressed={selected === i}
                    disabled={busy || exporting}
                    onClick={() => setSelected(i)}
                  >
                    {d.presentation.label}
                  </button>
                ))}
              </div>
              <div
                className="c3-image"
                style={{
                  aspectRatio: `${sourceWidth}/${sourceHeight}`,
                }}
              >
                <img
                  src={sourceImage}
                  alt={defect.source_kind === "mask" ? "Separate crack mask with identified damage" : "Road photograph with identified damage"}
                />
                {sourceDefects.map((d) => (
                  <button
                    className="c3-region"
                    key={d.defect_id}
                    aria-label={`Select ${d.presentation.label}`}
                    disabled={busy || exporting}
                    onClick={() => setSelected(result.defects.findIndex((item) => item.defect_id === d.defect_id))}
                    style={{
                      left: `${(d.source_bbox_xyxy[0] / sourceWidth) * 100}%`,
                      top: `${(d.source_bbox_xyxy[1] / sourceHeight) * 100}%`,
                      width: `${((d.source_bbox_xyxy[2] - d.source_bbox_xyxy[0]) / sourceWidth) * 100}%`,
                      height: `${((d.source_bbox_xyxy[3] - d.source_bbox_xyxy[1]) / sourceHeight) * 100}%`,
                      borderColor:
                        d.damage_type === "pothole" ? "#dda328" : "#2e85dc",
                      background: defect.defect_id === d.defect_id ? "#ffffff15" : "transparent",
                    }}
                  >
                    <span>{d.presentation.label}</span>
                  </button>
                ))}
                {sourceDefects.map((d) => (
                  <img
                    key={d.defect_id}
                    className="c3-outline"
                    src={d.visuals.outline}
                    alt=""
                  />
                ))}
              </div>
            </>
          ) : (
            <div className="c3-empty">
              <h3>Your marked areas will appear here.</h3>
              <p>Upload your image and damage masks to begin.</p>
            </div>
          )}
        </section>
        <section className="c3-panel">
          <p className="c3-eyebrow">03 / VISUAL DAMAGE ANALYSIS</p>
          <h2>Understand the Damage</h2>
          {defect && result ? (
            <>
              <h3>{defect.presentation.label}</h3>
              <GeometryViewer
                key={defect.defect_id}
                defect={defect}
                image={sourceImage}
                width={sourceWidth}
                height={sourceHeight}
              />
            </>
          ) : (
            <p className="c3-muted">
              Explore the selected damage after analysis.
            </p>
          )}
        </section>
        {defect?.damage_type === "pothole" && <ResearchEvidence key={defect.defect_id} />}
        <section className="c3-panel">
          <p className="c3-eyebrow">04 / DAMAGE ASSESSMENT RESULTS</p>
          <h2>Your Road Damage Results</h2>
          <AssessmentSummary result={result} mode="findings" imageFor={(item) =>
            item.source_group === "crack" ? crackImageUrl || (item.source_kind === "photo" ? imageUrl : crackUrl) : imageUrl
          } />
        </section>
        <section className="c3-panel">
          <p className="c3-eyebrow">05 / REVIEW AND NEXT STEPS</p>
          <h2>What Should You Do Next?</h2>
          <AssessmentSummary result={result} mode="guidance" />
        </section>
        <section className="c3-panel">
          <p className="c3-eyebrow">06 / DOWNLOAD REPORT</p>
          <h2>Your Road Damage Report</h2>
          <p className="c3-muted">
            {result?.summary ||
              "Your report will include separate findings, visual analysis and inspection guidance for every damaged area."}
          </p>
          <button
            disabled={!result || busy || exporting}
            onClick={exportReport}
          >
            {exporting ? "Preparing PDF..." : "Download PDF Report"}
          </button>
        </section>
        <footer className="c3-footer">
          <Link href="/">Back to RoadLens</Link>
          <p>Road damage. Surface detail. A more informed assessment.</p>
          <span>COMPONENT 03</span>
        </footer>
      </main>
    </div>
  );
}
