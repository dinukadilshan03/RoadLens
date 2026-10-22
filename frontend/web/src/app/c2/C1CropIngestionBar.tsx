/* eslint-disable @next/next/no-img-element */
"use client";

import Link from "next/link";
import { useState, useRef, useEffect } from "react";
import { BENCHMARK_PRESETS, type BenchmarkPreset, type C1CropItem } from "./api";

interface C1CropIngestionBarProps {
  c1Crops: C1CropItem[];
  selectedCropId: string | null;
  activeC1RunFilter?: string | null;
  onChangeC1RunFilter?: (runId: string | null) => void;
  onSelectC1Crop: (crop: C1CropItem) => void;
  onBatchSegmentC1Run?: (c1RunId: string) => void;
  onSelectPreset: (preset: BenchmarkPreset) => void;
  onUploadFile: (file: File) => void;
  isLoading: boolean;
  defaultTab?: "c1_feed" | "presets" | "upload";
}

export default function C1CropIngestionBar({
  c1Crops,
  selectedCropId,
  activeC1RunFilter,
  onChangeC1RunFilter,
  onSelectC1Crop,
  onBatchSegmentC1Run,
  onSelectPreset,
  onUploadFile,
  isLoading,
  defaultTab,
}: C1CropIngestionBarProps) {
  const [activeTab, setActiveTab] = useState<"c1_feed" | "presets" | "upload">(
    defaultTab || "presets"
  );
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (defaultTab) {
      setActiveTab(defaultTab);
    }
  }, [defaultTab]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onUploadFile(e.target.files[0]);
    }
  };

  const uniqueC1RunIds = Array.from(new Set(c1Crops.map((c) => c.run_id).filter(Boolean)));
  const filteredCrops = activeC1RunFilter
    ? c1Crops.filter((c) => c.run_id === activeC1RunFilter)
    : c1Crops;

  const getCropImageUrl = (crop: C1CropItem) => {
    if (crop.crop_url) return crop.crop_url;
    if (
      crop.crop_artifact.startsWith("http") ||
      crop.crop_artifact.startsWith("/api/v1")
    ) {
      return crop.crop_artifact;
    }
    return `/api/v1/c1/runs/${crop.run_id}/artifacts/crops/${crop.crop_artifact}`;
  };

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/70 backdrop-blur-md p-4 shadow-xl">
      {/* Tab switchers & pipeline status */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-3 mb-3">
        <div className="flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded-md bg-blue-500/20 text-blue-400 font-mono text-xs font-bold border border-blue-500/30">
            IN
          </div>
          <span className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Defect Ingestion & Pipeline Feeder
          </span>
          <span className="hidden sm:inline-block text-slate-500 text-xs">|</span>
          <span className="hidden sm:inline-block text-[11px] text-slate-400">
            Interfaces § C1→C2 Contract Compliant
          </span>
        </div>

        {/* Ingestion Source Switcher */}
        <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs font-mono">
          <button
            onClick={() => setActiveTab("presets")}
            className={`px-3 py-1 rounded transition text-[11px] ${
              activeTab === "presets"
                ? "bg-indigo-600 text-white font-medium shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Research Benchmark Presets (4)
          </button>
          <button
            onClick={() => setActiveTab("c1_feed")}
            className={`px-3 py-1 rounded transition text-[11px] flex items-center gap-1.5 ${
              activeTab === "c1_feed"
                ? "bg-indigo-600 text-white font-medium shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Upstream C1 Crops
            <span className="rounded-full bg-slate-800 px-1.5 py-0.2 text-[9px] text-slate-300">
              {c1Crops.length}
            </span>
          </button>
          <button
            onClick={() => setActiveTab("upload")}
            className={`px-3 py-1 rounded transition text-[11px] ${
              activeTab === "upload"
                ? "bg-indigo-600 text-white font-medium shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Custom Upload
          </button>
        </div>
      </div>

      {/* Tab 1: Research Benchmark Presets */}
      {activeTab === "presets" && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {BENCHMARK_PRESETS.map((preset) => (
            <div
              key={preset.id}
              onClick={() => !isLoading && onSelectPreset(preset)}
              className="group relative flex flex-col justify-between p-3 rounded-lg border border-slate-800/90 bg-slate-950/60 hover:border-indigo-500/60 hover:bg-slate-900/90 transition-all cursor-pointer shadow-sm hover:shadow-indigo-500/10"
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-indigo-300 border border-slate-700">
                    {preset.noiseType}
                  </span>
                  <span className="text-[10px] font-mono text-slate-500">
                    α ≈ {preset.expectedAlpha}
                  </span>
                </div>

                <div className="relative h-28 w-full rounded overflow-hidden bg-slate-900 border border-slate-800/60 mb-2.5">
                  <img
                    src={preset.thumbnail}
                    alt={preset.title}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-slate-950/90 via-transparent to-transparent"></div>
                  <div className="absolute bottom-1.5 left-2 right-2 text-[11px] font-semibold text-slate-200 truncate">
                    {preset.title}
                  </div>
                </div>

                <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed">
                  {preset.description}
                </p>
              </div>

              <div className="mt-3 pt-2 border-t border-slate-850 flex items-center justify-between text-[10px]">
                <span className="font-mono text-slate-500">{preset.damageClass}</span>
                <span className="font-semibold text-indigo-400 group-hover:text-indigo-300 flex items-center gap-1">
                  Run Segmenter ▶
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Tab 2: Upstream C1 Live Feed Crops */}
      {activeTab === "c1_feed" && (
        <div className="space-y-3">
          {uniqueC1RunIds.length > 0 && (
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-[11px] text-slate-400">Filter by C1 Run:</span>
                <button
                  type="button"
                  onClick={() => onChangeC1RunFilter?.(null)}
                  className={`rounded px-2 py-0.5 text-[10px] border transition ${
                    !activeC1RunFilter
                      ? "border-indigo-500 bg-indigo-600/20 text-indigo-200"
                      : "border-slate-800 bg-slate-950 text-slate-400 hover:text-slate-200"
                  }`}
                >
                  All Runs ({c1Crops.length})
                </button>
                {uniqueC1RunIds.slice(0, 6).map((rId) => (
                  <button
                    key={rId}
                    type="button"
                    onClick={() => onChangeC1RunFilter?.(rId)}
                    className={`rounded px-2 py-0.5 text-[10px] border transition ${
                      activeC1RunFilter === rId
                        ? "border-indigo-500 bg-indigo-600/20 text-indigo-200"
                        : "border-slate-800 bg-slate-950 text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {rId.slice(0, 8)}
                  </button>
                ))}
              </div>

              {onBatchSegmentC1Run && (activeC1RunFilter || uniqueC1RunIds[0]) && (
                <button
                  type="button"
                  disabled={isLoading}
                  onClick={() =>
                    onBatchSegmentC1Run(activeC1RunFilter || uniqueC1RunIds[0])
                  }
                  className="rounded-md bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white px-2.5 py-1 text-[11px] font-semibold transition cursor-pointer"
                >
                  Segment All in Run ({(activeC1RunFilter || uniqueC1RunIds[0]).slice(0, 8)}) ▶
                </button>
              )}
            </div>
          )}

          {filteredCrops.length === 0 ? (
            <div className="text-center py-8 text-slate-500 text-xs">
              <svg className="w-8 h-8 mx-auto mb-2 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
              No upstream C1 defect crops detected yet.
              <p className="text-slate-600 mt-1">
                <Link href="/c1" className="text-indigo-400 hover:underline">
                  Run a road inspection in Component 1 →
                </Link>{" "}
                or switch to Benchmark Presets above.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3 max-h-64 overflow-y-auto pr-1">
              {filteredCrops.map((crop) => {
                const isSelected = selectedCropId === crop.record_id;
                const isSegmented = Boolean(crop.latest_c2_run_id);
                return (
                  <div
                    key={crop.record_id}
                    onClick={() => !isLoading && onSelectC1Crop(crop)}
                    className={`flex flex-col justify-between p-2 rounded-lg border transition cursor-pointer ${
                      isSelected
                        ? "border-indigo-500 bg-indigo-950/30 ring-1 ring-indigo-500"
                        : "border-slate-850 bg-slate-950/60 hover:border-slate-700 hover:bg-slate-900"
                    }`}
                  >
                    <div>
                      <div className="relative h-24 w-full rounded overflow-hidden bg-slate-900 mb-2 border border-slate-800">
                        <img
                          src={getCropImageUrl(crop)}
                          alt={crop.display_name}
                          className="w-full h-full object-contain"
                        />
                        {isSegmented && (
                          <span className="absolute top-1 right-1 rounded bg-emerald-950/90 border border-emerald-500/40 px-1.5 py-0.5 text-[9px] font-mono font-bold text-emerald-300">
                            Segmented ✓
                          </span>
                        )}
                      </div>
                      <div className="text-[11px] font-semibold text-slate-200 truncate">
                        {crop.display_name}
                      </div>
                      <div className="text-[10px] font-mono text-emerald-400 mt-0.5">
                        Conf: {(crop.confidence * 100).toFixed(1)}%
                      </div>
                      <div className="text-[9px] font-mono text-slate-500 truncate mt-0.5">
                        {crop.crop_transform.width}×{crop.crop_transform.height}px · ({crop.crop_transform.origin_xy[0]},{crop.crop_transform.origin_xy[1]})
                      </div>
                    </div>
                    <div className="mt-1.5 pt-1 border-t border-slate-800/80 flex items-center justify-between text-[9px] font-mono text-slate-400">
                      <span className="truncate">{crop.survey_id || crop.run_id.slice(0, 8)}</span>
                      <span className="text-indigo-400 font-semibold">Segment ▶</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Tab 3: Custom Upload */}
      {activeTab === "upload" && (
        <div
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-slate-700 hover:border-indigo-500/80 rounded-xl p-8 text-center cursor-pointer transition bg-slate-950/40 hover:bg-slate-900/50"
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileChange}
          />
          <svg className="w-10 h-10 mx-auto mb-2 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          <p className="text-sm font-semibold text-slate-200">
            Drop road surface crop here or click to browse
          </p>
          <p className="text-xs text-slate-400 mt-1">
            Supports PNG, JPG, WEBP. Cropped road distress image will be auto-normalized to 256×256 px.
          </p>
        </div>
      )}
    </div>
  );
}
