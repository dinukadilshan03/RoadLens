"use client";

import { useEffect, useRef, useState } from "react";
import type * as Plotly from "plotly.js";
import type { RelativeSurface } from "../api";

const perspective = { x: 1.65, y: -1.8, z: 1.25 };
const top = { x: 0.01, y: 0.01, z: 3 };

export default function RelativeSurfaceViewer({
  surface,
}: {
  surface: RelativeSurface;
}) {
  const chart = useRef<HTMLDivElement>(null);
  const library = useRef<typeof Plotly | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">(
    "loading",
  );

  useEffect(() => {
    const node = chart.current;
    if (!node) return;
    let cancelled = false;
    let observer: ResizeObserver | undefined;
    let plotly: typeof Plotly | undefined;
    async function draw() {
      const probe = document.createElement("canvas");
      const gl = probe.getContext("webgl");
      if (!gl) throw new Error("WebGL is unavailable");
      gl.getExtension("WEBGL_lose_context")?.loseContext();
      const plotlyModule = await import("plotly.js-gl3d-dist-min");
      if (cancelled) return;
      plotly = plotlyModule.default;
      library.current = plotly;
      let compact = node!.clientWidth < 480;
      // Plotly's published types omit surface-specific color properties.
      const trace: Partial<Plotly.PlotData> & {
        surfacecolor: number[][];
        cmin: number;
        cmax: number;
      } = {
        type: "surface",
        x: surface.x,
        y: surface.y,
        z: surface.z,
        surfacecolor: surface.depression,
        cmin: 0,
        cmax: 1,
        colorscale: [
          [0, "#d4d8ce"], [0.003, "#d4d8ce"], [0.004, "#5b9991"],
          [0.2, "#57ad9b"], [0.45, "#d7bb69"], [0.7, "#ed8a43"], [1, "#cc3830"],
        ],
        showscale: false,
        hovertemplate:
          "Across image: %{x:.1f}%<br>Along image: %{y:.1f}%<br>Estimated relative depression: %{surfacecolor:.0%}<extra></extra>",
      };
      await plotly.newPlot(
        node!,
        [trace, {
          type: "surface",
          name: "Healthy reference level",
          x: surface.x,
          y: surface.y,
          z: surface.z.map((row) => row.map(() => 0)),
          opacity: 0.22,
          colorscale: [[0, "#92999b"], [1, "#92999b"]],
          showscale: false,
          hovertemplate: "Healthy reference level: 0<extra></extra>",
        }],
        {
          autosize: true,
          margin: { l: 8, r: 8, t: 12, b: 8 },
          paper_bgcolor: "#f7f7f0",
          font: { family: "Arial, sans-serif", color: "#263a35", size: 10 },
          scene: {
            domain: { x: [0.07, 0.93], y: [0.06, 0.97] },
            xaxis: {
              title: { text: compact ? "" : "Across (%)" },
              showticklabels: !compact,
              nticks: 4,
              gridcolor: "#dae0d3",
              showbackground: false,
              zeroline: false,
            },
            yaxis: {
              title: { text: compact ? "" : "Along (%)" },
              showticklabels: !compact,
              nticks: 4,
              autorange: "reversed",
              gridcolor: "#dae0d3",
              showbackground: false,
              zeroline: false,
            },
            zaxis: {
              title: { text: compact ? "" : "Relative level" },
              showticklabels: !compact,
              range: [-1.14, 0.04],
              tickvals: [-1, 0],
              ticktext: ["Deeper", "Road reference"],
              gridcolor: "#dae0d3",
              zeroline: false,
            },
            aspectmode: "manual",
            aspectratio: surface.aspect_ratio,
            camera: { eye: perspective, up: { x: 0, y: 0, z: 1 } },
            dragmode: "orbit",
          },
          uirevision: "keep",
        },
        {
          responsive: true,
          displayModeBar: false,
          displaylogo: false,
          scrollZoom: true,
        },
      );
      if (cancelled) {
        plotly.purge(node!);
        return;
      }
      observer = new ResizeObserver(() => {
        try {
          plotly?.Plots.resize(node!);
          const narrow = node!.clientWidth < 480;
          if (narrow !== compact) {
            compact = narrow;
            const labels: Partial<Plotly.Layout> & Record<string, unknown> = {
              "scene.xaxis.title.text": narrow ? "" : "Across (%)",
              "scene.yaxis.title.text": narrow ? "" : "Along (%)",
              "scene.zaxis.title.text": narrow ? "" : "Relative level",
              "scene.xaxis.showticklabels": !narrow,
              "scene.yaxis.showticklabels": !narrow,
              "scene.zaxis.showticklabels": !narrow,
            };
            void plotly?.relayout(node!, labels).catch(() => {});
          }
        } catch {
          /* The panel may be unmounting. */
        }
      });
      observer.observe(node!);
      setStatus("ready");
    }
    void draw().catch(() => {
      if (!cancelled) setStatus("error");
    });
    return () => {
      cancelled = true;
      observer?.disconnect();
      plotly?.purge(node);
      library.current = null;
    };
  }, [surface]);

  async function changeView(eye: typeof perspective) {
    if (!chart.current || !library.current) return;
    const update: Partial<Plotly.Layout> & {
      "scene.camera": Partial<Plotly.Camera>;
    } = {
      "scene.camera": { eye, up: { x: 0, y: 0, z: 1 } },
    };
    try {
      await library.current.relayout(chart.current, update);
    } catch {
      setStatus("error");
    }
  }

  return (
    <section className="c3-surface" aria-labelledby="c3-surface-title">
      <div className="c3-section-title">
        <div>
          <p className="c3-eyebrow">EXPLORE THE ESTIMATED SHAPE</p>
          <h3 id="c3-surface-title">Estimated Road Depression — 3D View</h3>
        </div>
        <span className="c3-pill">Relative shape</span>
      </div>
      <p className="c3-muted">
        Drag to rotate, scroll or pinch to zoom, and hover to explore the
        selected pothole.
      </p>
      <div className="c3-surface-actions">
        <button
          disabled={status !== "ready"}
          onClick={() => void changeView(perspective)}
        >
          Perspective view
        </button>
        <button
          disabled={status !== "ready"}
          onClick={() => void changeView(top)}
        >
          Top view
        </button>
      </div>
      <div
        ref={chart}
        className="c3-surface-chart"
        role="img"
        aria-label="Interactive estimated pothole depression surface. The surface descends below the surrounding road reference."
        style={{ display: status === "error" ? "none" : undefined }}
      />
      {status === "ready" && (
        <div
          className="c3-surface-legend"
          aria-label="Relative depression color scale"
        >
          <span>Shallower</span>
          <i />
          <span>Deeper</span>
        </div>
      )}
      <p className="c3-muted">
        The 3D shape follows the depression map in the same image coordinates.
        Cooler areas are shallower and warmer areas are deeper.
        The translucent gray plane marks the healthy-reference level; the colored surface shows the estimated pothole depression.
      </p>
      <div aria-live="polite">
        {status === "loading" && (
          <p className="c3-muted">Loading interactive surface…</p>
        )}
        {status === "error" && (
          <p className="c3-notice">
            The interactive 3D view is unavailable in this browser. You can
            still use the image views and depression profile above, and the PDF
            includes a static 3D view.
          </p>
        )}
      </div>
      {!surface.has_depression && (
        <p className="c3-muted">
          The model did not estimate a depression in this marked area. The
          surface is shown flat.
        </p>
      )}

    </section>
  );
}
