/* eslint-disable @next/next/no-img-element */
"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import type { C2Run } from "./api";

interface SplitLensSliderProps {
  run: C2Run | null;
}

export default function SplitLensSlider({ run }: SplitLensSliderProps) {
  const [sliderPosition, setSliderPosition] = useState<number>(50); // percentage 0-100
  const [targetLayer, setTargetLayer] = useState<"visual_overlay" | "pseudo_thermal" | "skeleton_1px">("visual_overlay");
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const artifacts = run?.artifacts;
  const getSafeUrl = (key: string) => {
    if (!run || !artifacts || !(key in artifacts)) return null;
    const val = (artifacts as any)[key];
    if (!val) return null;
    if (val.startsWith("http") || val.startsWith("/api/v1")) return val;
    return `/api/v1/c2/runs/${run.run_id}/artifacts/${key}`;
  };

  const inputUrl = getSafeUrl("input_rgb");
  const targetUrl = getSafeUrl(targetLayer);

  const handleMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(percentage);
  }, []);

  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    if (e.touches.length > 0) {
      handleMove(e.touches[0].clientX);
    }
  }, [handleMove]);

  useEffect(() => {
    const handleGlobalMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        handleMove(e.clientX);
      }
    };
    const handleGlobalMouseUp = () => {
      if (isDragging) {
        setIsDragging(false);
      }
    };

    if (isDragging) {
      window.addEventListener("mousemove", handleGlobalMouseMove);
      window.addEventListener("mouseup", handleGlobalMouseUp);
    }
    return () => {
      window.removeEventListener("mousemove", handleGlobalMouseMove);
      window.removeEventListener("mouseup", handleGlobalMouseUp);
    };
  }, [isDragging, handleMove]);

  return (
    <div className="space-y-4">
      {/* Header and comparison mode selection */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-1">
        <div className="flex items-center gap-3">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-pink-500/20 border border-pink-400/40 text-pink-400 text-xs font-mono font-bold">
            SL
          </div>
          <div>
            <h2 className="text-sm font-semibold tracking-wide text-slate-100">
              Interactive Split-Lens Cross-Modal Comparator
            </h2>
            <p className="text-xs text-slate-400">
              Slide horizontally to verify noise suppression, tree shadow cancellation, and crack boundary adherence.
            </p>
          </div>
        </div>

        {/* Compare Target Selector */}
        <div className="flex items-center gap-1.5 bg-slate-900 border border-slate-800 p-1 rounded-lg text-xs font-mono">
          <span className="text-slate-400 px-2 text-[11px]">Compare with:</span>
          <button
            onClick={() => setTargetLayer("visual_overlay")}
            className={`px-2.5 py-1 rounded transition text-[11px] ${
              targetLayer === "visual_overlay"
                ? "bg-rose-600 text-white font-medium"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Red Overlay
          </button>
          <button
            onClick={() => setTargetLayer("pseudo_thermal")}
            className={`px-2.5 py-1 rounded transition text-[11px] ${
              targetLayer === "pseudo_thermal"
                ? "bg-rose-600 text-white font-medium"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Thermal cGAN
          </button>
          <button
            onClick={() => setTargetLayer("skeleton_1px")}
            className={`px-2.5 py-1 rounded transition text-[11px] ${
              targetLayer === "skeleton_1px"
                ? "bg-rose-600 text-white font-medium"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            1px Skeleton
          </button>
        </div>
      </div>

      {/* Main Split Lens Frame */}
      <div
        ref={containerRef}
        onMouseDown={() => setIsDragging(true)}
        onTouchMove={handleTouchMove}
        className="split-lens-container relative w-full h-[380px] sm:h-[480px] bg-black/90 border border-slate-800 cursor-ew-resize overflow-hidden flex items-center justify-center select-none"
      >
        {inputUrl && targetUrl ? (
          <>
            {/* Background Layer: Target (e.g. Red Overlay or Thermal) */}
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <img
                src={targetUrl}
                alt="Target Segmentation Layer"
                className="max-h-full max-w-full object-contain"
              />
            </div>

            {/* Foreground Layer: Input RGB clipped to sliderPosition */}
            <div
              className="absolute inset-0 flex items-center justify-center pointer-events-none overflow-hidden"
              style={{
                clipPath: `polygon(0 0, ${sliderPosition}% 0, ${sliderPosition}% 100%, 0 100%)`,
              }}
            >
              <img
                src={inputUrl}
                alt="Original Optical Crop"
                className="max-h-full max-w-full object-contain"
              />
            </div>

            {/* Draggable Divider Line */}
            <div
              className="split-lens-divider"
              style={{ left: `${sliderPosition}%` }}
            >
              <div className="split-lens-handle">
                <svg className="w-4 h-4 text-slate-900" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M8 9l4-4 4 4m0 6l-4 4-4-4" transform="rotate(90 12 12)" />
                </svg>
              </div>
            </div>

            {/* Labels overlay */}
            <div className="absolute top-4 left-4 bg-slate-950/80 backdrop-blur-sm border border-slate-700/60 rounded px-2.5 py-1 text-[11px] font-mono text-slate-200 pointer-events-none">
              ◀ Optical RGB (Noisy)
            </div>
            <div className="absolute top-4 right-4 bg-slate-950/80 backdrop-blur-sm border border-slate-700/60 rounded px-2.5 py-1 text-[11px] font-mono text-slate-200 pointer-events-none">
              {targetLayer === "visual_overlay"
                ? "Segmented Overlay ▶"
                : targetLayer === "pseudo_thermal"
                ? "Pseudo-Thermal cGAN ▶"
                : "1-px Skeleton ▶"}
            </div>

            {/* Current Position Marker */}
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-slate-950/80 border border-slate-800 rounded-full px-3 py-0.5 text-[10px] font-mono text-slate-400 pointer-events-none">
              Split: {Math.round(sliderPosition)}%
            </div>
          </>
        ) : (
          <div className="text-center p-8 text-slate-500">
            <svg className="w-12 h-12 mx-auto mb-3 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 7v8a2 2 0 002 2h6M8 7V5a2 2 0 012-2h4.586a1 1 0 01.707.293l4.414 4.414a1 1 0 01.293.707V15a2 2 0 01-2 2h-2M8 7H6a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2v-2" />
            </svg>
            <p className="text-sm font-medium text-slate-400">No active segmentation run loaded</p>
            <p className="text-xs text-slate-500 mt-1">Select an upstream C1 crop or a benchmark preset from above.</p>
          </div>
        )}
      </div>
    </div>
  );
}
