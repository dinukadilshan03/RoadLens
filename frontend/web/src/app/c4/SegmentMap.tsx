"use client";

// Road-segment map. Leaflet is loaded from a CDN at runtime, so no npm dependency is added;
// tiles come from OpenStreetMap.
import { useEffect, useRef, useState } from "react";
import { CLASS_COLORS, CLASS_LABELS, type SegmentCollection } from "./api";

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
        reject(new Error("Map library could not be loaded (no internet connection?)."));
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

export default function SegmentMap({ data }: { data: SegmentCollection | null }) {
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<Leaflet>(null);
  const layer = useRef<Leaflet>(null);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    loadLeaflet()
      .then((L) => {
        if (cancelled || !el.current || map.current) return;
        map.current = L.map(el.current, { zoomControl: true }).setView([7.8731, 80.7718], 7);
        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
          maxZoom: 19,
          attribution: "&copy; OpenStreetMap contributors",
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
    const feats = (data?.features ?? []).filter((f) => f.geometry);
    if (!feats.length) return;
    const bounds = L.latLngBounds([]);
    for (const f of feats) {
      const p = f.properties;
      const latlngs = f.geometry!.coordinates.map(([lon, lat]) => [lat, lon]);
      const line = L.polyline(latlngs, {
        color: CLASS_COLORS[p.roughness_class],
        weight: 7,
        opacity: 0.9,
        lineCap: "round",
      });
      const conf = p.confidence == null ? "–" : `${Math.round(p.confidence * 100)}%`;
      line.bindTooltip(
        `<b>${CLASS_LABELS[p.roughness_class]}</b> · confidence ${conf}<br/>` +
          `${fmtTime(p.interval_start_ms)} – ${fmtTime(p.interval_end_ms)}`,
        { sticky: true },
      );
      line.addTo(layer.current);
      bounds.extend(line.getBounds());
    }
    map.current.fitBounds(bounds, { padding: [30, 30], maxZoom: 18 });
  }, [data, ready]);

  return (
    <div className="relative h-full min-h-[360px] w-full overflow-hidden rounded-xl border border-zinc-200 dark:border-zinc-800">
      <div ref={el} className="absolute inset-0 z-0" aria-label="Map of road segments" />
      {error && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-zinc-50/90 p-6 text-center text-sm text-zinc-600 dark:bg-zinc-900/90 dark:text-zinc-300">
          {error}
        </div>
      )}
      <div className="absolute bottom-3 left-3 z-[500] flex gap-3 rounded-lg bg-white/95 px-3 py-2 text-xs shadow dark:bg-zinc-900/95">
        {(Object.keys(CLASS_COLORS) as (keyof typeof CLASS_COLORS)[]).map((c) => (
          <span key={c} className="flex items-center gap-1.5">
            <span className="inline-block h-1.5 w-5 rounded-full" style={{ background: CLASS_COLORS[c] }} />
            {CLASS_LABELS[c]}
          </span>
        ))}
      </div>
    </div>
  );
}
