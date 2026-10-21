"use client";

// Road-segment map. Leaflet is loaded from a CDN at runtime, so no npm dependency is added.
// Base map: OpenStreetMap tiles, toned down in c4.css so the condition colours stand out.
import { useEffect, useRef, useState } from "react";
import { CLASS_COLORS, CLASS_LABELS, type SegmentCollection, type SegmentFeature } from "./api";

export interface MapStyle {
  color: (f: SegmentFeature) => string;
  tooltip?: (f: SegmentFeature) => string;
  legend: { label: string; color: string }[];
}

const DEFAULT_STYLE: MapStyle = {
  color: (f) => CLASS_COLORS[f.properties.roughness_class],
  legend: (Object.keys(CLASS_COLORS) as (keyof typeof CLASS_COLORS)[]).map((c) => ({ label: CLASS_LABELS[c], color: CLASS_COLORS[c] })),
};

const LEAFLET_VERSION = "1.9.4";
const LEAFLET_CSS = `https://unpkg.com/leaflet@${LEAFLET_VERSION}/dist/leaflet.css`;
const LEAFLET_JS = `https://unpkg.com/leaflet@${LEAFLET_VERSION}/dist/leaflet.js`;

/* eslint-disable @typescript-eslint/no-explicit-any */
type Leaflet = any;
let leafletPromise: Promise<Leaflet> | null = null;

function loadLeaflet(): Promise<Leaflet> {
  const w = window as any;
  if (w.L) return Promise.resolve(w.L);
  if (!leafletPromise) {
    leafletPromise = new Promise((resolve, reject) => {
      const css = document.createElement("link");
      css.rel = "stylesheet";
      css.href = LEAFLET_CSS;
      document.head.appendChild(css);
      const js = document.createElement("script");
      js.src = LEAFLET_JS;
      js.async = true;
      js.onload = () => resolve(w.L);
      js.onerror = () => {
        leafletPromise = null;
        reject(new Error("The map could not be loaded. Check the internet connection and reload the page."));
      };
      document.head.appendChild(js);
    });
  }
  return leafletPromise;
}

function fmtTime(ms: number) {
  const s = Math.round(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

const WEIGHT = 6;

export default function SegmentMap({
  data,
  style = DEFAULT_STYLE,
  highlight = null,
  onHover,
  children,
}: {
  data: SegmentCollection | null;
  style?: MapStyle;
  highlight?: string | null;
  onHover?: (id: string | null) => void;
  children?: React.ReactNode;
}) {
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<Leaflet>(null);
  const layer = useRef<Leaflet>(null);
  const lines = useRef<Map<string, Leaflet>>(new Map());
  const hoverCb = useRef(onHover);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const fitted = useRef<SegmentCollection | null>(null);

  useEffect(() => {
    hoverCb.current = onHover;
  }, [onHover]);

  useEffect(() => {
    let cancelled = false;
    loadLeaflet()
      .then((L) => {
        if (cancelled || !el.current || map.current) return;
        map.current = L.map(el.current, { zoomControl: true }).setView([7.8731, 80.7718], 7);
        // OpenStreetMap tiles; c4.css tones them down so the condition colours stand out.
        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
          maxZoom: 19,
          attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        }).addTo(map.current);
        layer.current = L.layerGroup().addTo(map.current);
        setError(null);
        setReady(true);
      })
      .catch((e: Error) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const L = (window as any).L;
    if (!L || !map.current || !layer.current) return;
    layer.current.clearLayers();
    lines.current.clear();
    const feats = (data?.features ?? []).filter((f) => f.geometry);
    if (!feats.length) return;
    const bounds = L.latLngBounds([]);
    for (const f of feats) {
      const p = f.properties;
      const latlngs = f.geometry!.coordinates.map(([lon, lat]) => [lat, lon]);
      const line = L.polyline(latlngs, { color: style.color(f), weight: WEIGHT, opacity: 0.95, lineCap: "butt" });
      const conf = p.confidence == null ? "" : `, ${Math.round(p.confidence * 100)}% confident`;
      line.bindTooltip(
        `<b>${CLASS_LABELS[p.roughness_class]}</b>${conf}<br/>Recorded at ${fmtTime(p.interval_start_ms)}` +
          (style.tooltip ? `<br/>${style.tooltip(f)}` : ""),
        { sticky: true },
      );
      line.on("mouseover", () => hoverCb.current?.(p.record_id));
      line.on("mouseout", () => hoverCb.current?.(null));
      line.addTo(layer.current);
      lines.current.set(p.record_id, line);
      bounds.extend(line.getBounds());
    }
    if (fitted.current !== data) {
      map.current.fitBounds(bounds, { padding: [36, 36], maxZoom: 18 }); // only when new data arrives
      fitted.current = data;
    }
  }, [data, ready, style]);

  // Emphasise the stretch that is pointed at, here or on the road strip.
  useEffect(() => {
    lines.current.forEach((line, id) => {
      const on = highlight === id;
      line.setStyle({ weight: on ? WEIGHT + 6 : WEIGHT, opacity: highlight && !on ? 0.35 : 0.95 });
      if (on) line.bringToFront();
    });
  }, [highlight, data, style, ready]);

  return (
    <div className="relative h-full min-h-[340px] w-full overflow-hidden rounded-md border border-[var(--c4-line)]">
      <div ref={el} className="absolute inset-0 z-0" role="region" aria-label="Map of the surveyed road, coloured by condition" />
      {error && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-[var(--c4-surface)] p-6 text-center text-sm text-[var(--c4-muted)]">
          {error}
        </div>
      )}
      {children && <div className="absolute right-3 top-3 z-[500]">{children}</div>}
      <div className="absolute bottom-3 left-3 z-[500] flex gap-3 rounded-md border border-[var(--c4-line)] bg-[var(--c4-surface)] px-3 py-1.5 text-xs">
        {style.legend.map((l) => (
          <span key={l.label} className="flex items-center gap-1.5">
            <span className="inline-block h-1.5 w-5" style={{ background: l.color }} />
            {l.label}
          </span>
        ))}
      </div>
    </div>
  );
}
