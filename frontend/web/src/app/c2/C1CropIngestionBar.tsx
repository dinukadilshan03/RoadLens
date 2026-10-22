/* eslint-disable @next/next/no-img-element */
"use client";

import { useState, useRef } from "react";
import { BENCHMARK_PRESETS, type BenchmarkPreset, type C1CropItem } from "./api";

interface C1CropIngestionBarProps {
  c1Crops: C1CropItem[];
  selectedCropId: string | null;
  onSelectC1Crop: (crop: C1CropItem) => void;
  onSelectPreset: (preset: BenchmarkPreset) => void;
  onUploadFile: (file: File) => void;
  isLoading: boolean;
}

export default function C1CropIngestionBar({
  c1Crops,
  selectedCropId,
  onSelectC1Crop,
  onSelectPreset,
  onUploadFile,
  isLoading,
}: C1CropIngestionBarProps) {
  const [activeTab, setActiveTab] = useState<"c1_feed" | "presets" | "upload">("presets");
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onUploadFile(e.target.files[0]);
    }
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
        <div>
          {c1Crops.length === 0 ? (
            <div className="text-center py-8 text-slate-500 text-xs">
              <svg className="w-8 h-8 mx-auto mb-2 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
              No upstream C1 defect crops detected yet in PostgreSQL database.
              <p className="text-slate-600 mt-1">Run a survey in Component 1 or switch to Benchmark Presets above.</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3 max-h-60 overflow-y-auto pr-1">
              {c1Crops.map((crop) => {
                const isSelected = selectedCropId === crop.record_id;
                return (
                  <div
                    key={crop.record_id}
                    onClick={() => !isLoading && onSelectC1Crop(crop)}
                    className={`flex flex-col p-2 rounded-lg border transition cursor-pointer ${
                      isSelected
                        ? "border-indigo-500 bg-indigo-950/30"
                        : "border-slate-850 bg-slate-950/60 hover:border-slate-700 hover:bg-slate-900"
                    }`}
                  >
                    <div className="relative h-24 w-full rounded overflow-hidden bg-slate-900 mb-2 border border-slate-800">
                      <img
                        src={crop.crop_artifact}
                        alt={crop.display_name}
                        className="w-full h-full object-contain"
                      />
                    </div>
                    <div className="text-[11px] font-semibold text-slate-200 truncate">
                      {crop.display_name}
                    </div>
                    <div className="text-[10px] font-mono text-emerald-400 mt-0.5">
                      Conf: {(crop.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="text-[9px] font-mono text-slate-500 truncate mt-0.5">
                      {crop.crop_transform.width}×{crop.crop_transform.height}px
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
