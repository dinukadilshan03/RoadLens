"use client";

import { useEffect, useRef, useState } from "react";
import {
  CLASS_COLORS,
  CLASS_LABELS,
  type DefectCollection,
  type DefectProperties,
} from "./api";

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
        reject(new Error("Map library could not be loaded (offline?)."));
      };
      document.head.appendChild(js);
    });
  }
  return leafletPromise;
}

export default function DefectMap({
  data,
  selectedId,
  onSelect,
}: {
  data: DefectCollection | null;
  selectedId?: string | null;
  onSelect?: (defect: DefectProperties) => void;
}) {
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
        const m = L.map(el.current, {
          center: [6.895, 79.854], // Colombo Galle Road default
          zoom: 14,
          zoomControl: true,
        });
        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
          attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
          maxZoom: 19,
        }).addTo(m);
        map.current = m;
        setReady(true);
      })
      .catch((e) => setError(e.message));

    return () => {
      cancelled = true;
      if (map.current) {
        map.current.remove();
        map.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!ready || !map.current) return;
    const L = (window as any).L;
    if (layer.current) {
      map.current.removeLayer(layer.current);
      layer.current = null;
    }
    if (!data || data.features.length === 0) return;

    const group = L.featureGroup();

    data.features.forEach((feat) => {
      const [lng, lat] = feat.geometry.coordinates;
      const props = feat.properties;
      const color = CLASS_COLORS[props.damage_class] || "#ef4444";
      const isSelected = selectedId === props.defect_id;

      const marker = L.circleMarker([lat, lng], {
        radius: isSelected ? 9 : 7,
        fillColor: color,
        color: isSelected ? "#ffffff" : "#1e293b",
        weight: isSelected ? 3 : 1.5,
        opacity: 1,
        fillOpacity: 0.85,
      });

      marker.bindTooltip(
        `<b>${props.display_name}</b><br/>Confidence: ${(props.confidence * 100).toFixed(0)}%<br/>Time: ${(props.timestamp_ms / 1000).toFixed(1)}s`,
        { direction: "top", offset: [0, -5] }
      );

      marker.on("click", () => {
        if (onSelect) onSelect(props);
      });

      marker.addTo(group);
    });

    group.addTo(map.current);
    layer.current = group;

    try {
      const bounds = group.getBounds();
      if (bounds.isValid()) {
        map.current.fitBounds(bounds, { padding: [40, 40], maxZoom: 17 });
      }
    } catch {
      // bounds fit ignore
    }
  }, [ready, data, selectedId, onSelect]);

  return (
    <div className="relative h-full w-full overflow-hidden rounded-xl border border-zinc-200 dark:border-zinc-800">
      {error && (
        <div className="flex h-full items-center justify-center bg-red-50 p-4 text-xs text-red-700 dark:bg-red-950 dark:text-red-300">
          {error}
        </div>
      )}
      <div ref={el} className="h-full w-full" />

      {/* Map Legend */}
      <div className="absolute bottom-3 right-3 z-[1000] rounded-lg border border-zinc-200/80 bg-white/90 p-2.5 shadow-sm backdrop-blur-sm dark:border-zinc-700/80 dark:bg-zinc-900/90 text-xs">
        <p className="mb-1.5 font-semibold text-zinc-700 dark:text-zinc-200">Road Distress Classes</p>
        <div className="flex flex-col gap-1 text-[11px]">
          {Object.entries(CLASS_COLORS).map(([cname, color]) => (
            <div key={cname} className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-zinc-600 dark:text-zinc-300">{CLASS_LABELS[cname] || cname}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
