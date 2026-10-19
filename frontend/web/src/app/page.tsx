import Link from "next/link";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col bg-zinc-50 font-sans dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100">
      <header className="border-b border-zinc-200 bg-white/80 px-6 py-4 backdrop-blur dark:border-zinc-800 dark:bg-zinc-900/80">
        <div className="mx-auto flex max-w-7xl items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-600 text-white font-bold text-lg shadow-sm">
              RL
            </div>
            <div>
              <span className="font-bold tracking-tight text-lg">RoadLens</span>
              <span className="ml-2 text-xs text-zinc-500">Sri Lankan Road Assessment Platform</span>
            </div>
          </div>
          <div className="flex items-center gap-4 text-xs font-medium">
            <Link
              href="/c1"
              className="rounded-md bg-blue-50 px-3 py-1.5 text-blue-700 hover:bg-blue-100 dark:bg-blue-950 dark:text-blue-300"
            >
              Component 1 (Damage)
            </Link>
            <Link
              href="/c4"
              className="rounded-md bg-zinc-100 px-3 py-1.5 text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300"
            >
              Component 4 (Roughness)
            </Link>
          </div>
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-7xl flex-1 flex-col px-6 py-10">
        <div className="max-w-3xl">
          <div className="inline-flex items-center gap-2 rounded-full border border-blue-200 bg-blue-50/60 px-3 py-1 text-xs font-semibold text-blue-800 dark:border-blue-900 dark:bg-blue-950/40 dark:text-blue-300">
            Progress Presentation 1 (50% Milestone)
          </div>
          <h1 className="mt-4 text-4xl font-bold tracking-tight sm:text-5xl">
            An AI Framework for Sri Lankan Road Management
          </h1>
          <p className="mt-4 text-base text-zinc-600 dark:text-zinc-400 leading-relaxed">
            Automated road condition assessment using smartphone video, AI object detection, pixel segmentation,
            3D surface reconstruction, and multi-sensor roughness analytics.
          </p>
        </div>

        {/* Component Grid */}
        <div className="mt-10 grid grid-cols-1 gap-6 md:grid-cols-2">
          {/* Component 1 (Primary / Active Demo) */}
          <div className="group relative flex flex-col justify-between rounded-2xl border-2 border-blue-500/80 bg-white p-6 shadow-sm transition hover:shadow-md dark:border-blue-500/60 dark:bg-zinc-900">
            <div>
              <div className="flex items-center justify-between">
                <span className="rounded-full bg-blue-100 px-2.5 py-0.5 text-xs font-semibold text-blue-800 dark:bg-blue-950 dark:text-blue-300">
                  Component 1 · Lead
                </span>
                <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400">
                  ● 50% Milestone Ready
                </span>
              </div>
              <h2 className="mt-3 text-xl font-semibold tracking-tight">
                Road Damage Detection & Mapping
              </h2>
              <p className="mt-2 text-xs text-zinc-600 dark:text-zinc-400 leading-normal">
                Detects potholes and cracks from smartphone survey video using a YOLOv8m Base Model (B0).
                Extracts frame coordinates, extracts damage bounding boxes, generates C2 defect ROI crops, and maps defects to GPS locations.
              </p>

              <div className="mt-4 flex flex-wrap gap-2 text-[11px]">
                <span className="rounded bg-zinc-100 px-2 py-1 font-medium text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
                  Base Model B0 (YOLOv8m)
                </span>
                <span className="rounded bg-zinc-100 px-2 py-1 font-medium text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
                  mAP@50: 66.2%
                </span>
                <span className="rounded bg-zinc-100 px-2 py-1 font-medium text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
                  Potholes (D40): 78.4%
                </span>
                <span className="rounded bg-zinc-100 px-2 py-1 font-medium text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
                  C2 Defect Crop Export
                </span>
              </div>
            </div>

            <div className="mt-6 border-t border-zinc-100 pt-4 dark:border-zinc-800">
              <Link
                href="/c1"
                className="inline-flex w-full items-center justify-center rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-blue-500"
              >
                Launch C1 Detection Dashboard →
              </Link>
            </div>
          </div>

          {/* Component 4 */}
          <div className="group relative flex flex-col justify-between rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm transition hover:shadow-md dark:border-zinc-800 dark:bg-zinc-900">
            <div>
              <div className="flex items-center justify-between">
                <span className="rounded-full bg-zinc-100 px-2.5 py-0.5 text-xs font-semibold text-zinc-800 dark:bg-zinc-800 dark:text-zinc-300">
                  Component 4
                </span>
                <span className="text-xs text-zinc-500">Parallel Sensor Branch</span>
              </div>
              <h2 className="mt-3 text-xl font-semibold tracking-tight">
                Multi-Sensor Road Roughness Assessment
              </h2>
              <p className="mt-2 text-xs text-zinc-600 dark:text-zinc-400 leading-normal">
                Processes smartphone IMU (accelerometer, gyroscope) and GPS metadata to classify continuous road segments into Good, Regular, and Bad roughness.
              </p>

              <div className="mt-4 flex flex-wrap gap-2 text-[11px]">
                <span className="rounded bg-zinc-100 px-2 py-1 font-medium text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
                  IMU / GPS Sensor Processing
                </span>
                <span className="rounded bg-zinc-100 px-2 py-1 font-medium text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
                  Segment Road Classification
                </span>
              </div>
            </div>

            <div className="mt-6 border-t border-zinc-100 pt-4 dark:border-zinc-800">
              <Link
                href="/c4"
                className="inline-flex w-full items-center justify-center rounded-lg border border-zinc-300 px-4 py-2.5 text-sm font-semibold text-zinc-900 hover:bg-zinc-50 dark:border-zinc-700 dark:text-zinc-100 dark:hover:bg-zinc-800"
              >
                Launch C4 Roughness Dashboard →
              </Link>
            </div>
          </div>

          {/* Component 2 Placeholder */}
          <div className="rounded-2xl border border-dashed border-zinc-300 bg-zinc-50/50 p-6 dark:border-zinc-800 dark:bg-zinc-900/20">
            <span className="rounded-full bg-zinc-200 px-2 py-0.5 text-xs font-medium text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">
              Component 2 · Downstream
            </span>
            <h3 className="mt-3 text-lg font-semibold text-zinc-700 dark:text-zinc-300">
              Noise-Resilient Pavement Segmentation
            </h3>
            <p className="mt-1 text-xs text-zinc-500">
              Consumes cropped defect regions (ROIs) from C1 to generate pixel-level damage masks and crack skeletons.
            </p>
          </div>

          {/* Component 3 Placeholder */}
          <div className="rounded-2xl border border-dashed border-zinc-300 bg-zinc-50/50 p-6 dark:border-zinc-800 dark:bg-zinc-900/20">
            <span className="rounded-full bg-zinc-200 px-2 py-0.5 text-xs font-medium text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">
              Component 3 · Downstream
            </span>
            <h3 className="mt-3 text-lg font-semibold text-zinc-700 dark:text-zinc-300">
              Surface Reconstruction & Geometric Severity
            </h3>
            <p className="mt-1 text-xs text-zinc-500">
              Consumes C2 masks and geometric depth data to estimate defect depth, area, and volumetric severity.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
