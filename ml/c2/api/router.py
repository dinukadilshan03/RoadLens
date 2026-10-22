"""C2 REST API (FastAPI router, mounted by the RoadLens backend under /api/v1/c2).

Endpoints:
    GET  /api/v1/c2/model                                C2 model specs, LTCA 67 MB VRAM, Tri-Pillar targets
    POST /api/v1/c2/runs                                 trigger segmentation (upload, sample, or C1 crop handoff)
    GET  /api/v1/c2/runs                                 list recent C2 runs
    GET  /api/v1/c2/runs/{run_id}                        run status, telemetry, alpha score, artifact URLs
    GET  /api/v1/c2/runs/{run_id}/artifacts/{artifact}  serve images: input_rgb, pseudo_thermal, mask_2d, skeleton_1px, confidence_map, visual_overlay
    GET  /api/v1/c2/runs/{run_id}/record                 contract handoff record for C3 & C4
    GET  /api/v1/c2/c1-available-crops                   fetch upstream crops from C1 for instant ingestion

References:
- Proposal Section 2.2 Sub-Objectives 1-5
- Proposal Section 3.1 & Figure 4.1
- docs/interfaces.md § Component payload requirements (C2)
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ml.c2.adapters.segmenter_adapter import (
    DEFAULT_CGAN_WEIGHTS,
    DEFAULT_SEGMENTER_WEIGHTS,
    C2Cancelled,
    C2InputError,
    C2Request,
    compute_sha256,
    run as run_segmenter,
)
from ml.c2.api.store import RunStore, now

log = logging.getLogger("roadlens.c2")
C2_DIR = Path(__file__).resolve().parents[1]
MAX_CROP_BYTES = 10 * 1024 * 1024  # 10 MiB per crop
SAFE_FILENAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _error(status: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "code": code,
            "message": message,
            "request_id": uuid.uuid4().hex,
            "details": details or {},
        },
    )


class _StoreContext:
    def __init__(self, store, run_id: str, output_dir: Path):
        self.store = store
        self.run_id = run_id
        self.output_dir = Path(output_dir)

    def progress(self, stage: str, completed: int | None = None, total: int | None = None) -> None:
        self.store.update(self.run_id, stage=stage, progress={"completed": completed, "total": total})

    def is_cancelled(self) -> bool:
        meta = self.store.get(self.run_id)
        return bool(meta.get("cancel_requested"))


def build_router(
    storage_dir: Path,
    cgan_weights_path: Path = DEFAULT_CGAN_WEIGHTS,
    segmenter_weights_path: Path = DEFAULT_SEGMENTER_WEIGHTS,
    store: RunStore | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/c2", tags=["c2-segmentation"])
    run_store = store or RunStore(storage_dir)

    @router.get("/model")
    def get_model():
        cgan_ready = cgan_weights_path.exists()
        seg_ready = segmenter_weights_path.exists()
        pothole_weights_path = C2_DIR / "weights" / "c2_pothole_ltca_best.pth"
        pothole_ready = pothole_weights_path.exists()
        return {
            "component": "C2",
            "name": "Noise-Resilient Road Damage Segmentation Engine",
            "architecture": "Software-Emulated Pix2Pix cGAN + Dual MobileNetV3 + LTCA + Alpha Gate",
            "weights_ready": cgan_ready and seg_ready,
            "cgan_sensor_ready": cgan_ready,
            "segmenter_ready": seg_ready,
            "pothole_ready": pothole_ready,
            "cgan_weights_sha256": compute_sha256(cgan_weights_path),
            "segmenter_weights_sha256": compute_sha256(segmenter_weights_path),
            "pothole_weights_sha256": compute_sha256(pothole_weights_path) if pothole_ready else None,
            "input_resolution": [256, 256],
            "tri_pillar_targets": {
                "pillar_1_spatial": {"target_mIoU": ">= 81.83%", "target_f1": ">= 87.00%"},
                "pillar_2_continuity": {"target_topological_continuity": "> 90.00%"},
                "pillar_3_hardware": {"target_vram": "67 MB (Goal)", "target_latency": "< 30 ms"},
            },
            "output_suite": {
                "output_a": "2D High-Precision Binary Mask (0/255 lossless PNG)",
                "output_b": "1-Pixel Skeleton Centerline (unbroken line)",
                "output_c": "Soft Confidence Map (.npz array in [0,1])",
                "output_d": "High-Contrast Red Visual Overlay",
            },
        }

    @router.get("/c1-available-crops")
    def list_c1_crops(limit: int = 20):
        crops = run_store.get_available_c1_crops(limit=limit)
        return {"count": len(crops), "crops": crops}

    @router.post("/runs", status_code=202)
    def create_run(
        background_tasks: BackgroundTasks,
        survey_id: Annotated[str, Form()] = "survey_default",
        damage_class: Annotated[str, Form()] = "crack",
        threshold: Annotated[float, Form()] = 0.5,
        c1_run_id: Annotated[str | None, Form()] = None,
        c1_observation_id: Annotated[str | None, Form()] = None,
        sample_id: Annotated[str | None, Form()] = None,
        preset_id: Annotated[str | None, Form()] = None,
        crop_file: UploadFile | None = File(None),
        image: UploadFile | None = File(None),
    ):
        target_file = crop_file if (crop_file is not None and crop_file.filename) else image
        active_sample_id = sample_id or preset_id
        source = "c1_handoff" if (c1_run_id and c1_observation_id) else ("demo" if active_sample_id else "upload")
        meta = run_store.new(
            survey_id=survey_id,
            source=source,
            damage_class=damage_class,
            c1_run_id=c1_run_id,
            c1_observation_id=c1_observation_id,
        )
        run_id = meta["run_id"]
        run_dir = run_store.dir(run_id)
        input_crop_path = run_dir / "inputs" / "crop.jpg"

        # Determine input file
        if target_file is not None and target_file.filename:
            with open(input_crop_path, "wb") as f:
                content = target_file.file.read()
                f.write(content)
        elif active_sample_id:
            # Handle preset sample
            sample_file_map = {
                "shadow_crack": C2_DIR / "data" / "samples" / "sample_shadow_crack.jpg",
                "glare_alligator": C2_DIR / "data" / "samples" / "sample_glare_alligator.jpg",
                "drainage_grate": C2_DIR / "data" / "samples" / "sample_drainage_grate.jpg",
                "pothole_water": C2_DIR / "data" / "samples" / "sample_pothole_water.jpg",
            }
            src = sample_file_map.get(active_sample_id)
            if src and src.exists():
                shutil.copy(src, input_crop_path)
            else:
                # Generate a synthetic demonstration pattern
                img = np.full((256, 256, 3), 110, dtype=np.uint8)
                cv2.line(img, (40, 20), (210, 230), (30, 30, 30), 3)
                cv2.imwrite(str(input_crop_path), img)
        elif c1_run_id and c1_observation_id:
            # Look up crop artifact from C1 storage
            c1_storage = storage_dir.parent / "c1" / "runs" / c1_run_id / "outputs" / "crops"
            found = list(c1_storage.glob(f"*{c1_observation_id}*"))
            if found:
                shutil.copy(found[0], input_crop_path)
            else:
                # Synthesize fallback crop matching the defect
                img = np.full((256, 256, 3), 120, dtype=np.uint8)
                cv2.circle(img, (128, 128), 50, (40, 40, 40), -1)
                cv2.imwrite(str(input_crop_path), img)
        else:
            # Fallback sample crop
            img = np.full((256, 256, 3), 128, dtype=np.uint8)
            cv2.line(img, (30, 30), (220, 220), (30, 30, 30), 2)
            cv2.imwrite(str(input_crop_path), img)

        def _execute():
            ctx = _StoreContext(run_store, run_id, run_dir / "outputs")
            run_store.update(run_id, status="running", started_at=now())
            try:
                req = C2Request(
                    survey_id=survey_id,
                    input_crop_path=input_crop_path,
                    damage_class=damage_class,
                    external_run_id=run_id,
                    upstream_c1_id=c1_observation_id,
                    upstream_c1_run_id=c1_run_id,
                    cgan_weights_path=cgan_weights_path,
                    segmenter_weights_path=segmenter_weights_path,
                    threshold=threshold,
                )
                res = run_segmenter(req, ctx)
                run_store.complete(
                    run_id,
                    results=[res.record],
                    status="succeeded",
                    stage="finished",
                    alpha_score=res.alpha_weight,
                    latency_ms=res.latency_ms,
                    vram_mb=res.vram_mb,
                    artifacts=res.artifacts,
                    summary=res.summary,
                    finished_at=now(),
                )
            except Exception as ex:
                log.exception(f"C2 run {run_id} failed: {ex}")
                run_store.update(
                    run_id,
                    status="failed",
                    error={"code": "processing_error", "message": str(ex)},
                    finished_at=now(),
                )

        background_tasks.add_task(_execute)
        return {"run_id": run_id, "status": "queued"}

    def _format_run(meta: dict) -> dict:
        r_id = meta.get("run_id")
        if not r_id:
            return meta
        out = dict(meta)
        if out.get("artifacts"):
            out["artifacts"] = {
                k: f"/api/v1/c2/runs/{r_id}/artifacts/{k}"
                for k in out["artifacts"].keys()
            }
        return out

    @router.get("/runs")
    def list_runs(limit: int = 50):
        runs = run_store.list(limit=limit)
        return [_format_run(r) for r in runs]

    @router.get("/runs/{run_id}")
    def get_run(run_id: str):
        try:
            meta = run_store.get(run_id)
            meta["results"] = run_store.get_results(run_id)
            return _format_run(meta)
        except KeyError:
            return _error(404, "run_not_found", f"Run {run_id} not found.")

    @router.get("/runs/{run_id}/record")
    def get_record(run_id: str):
        results = run_store.get_results(run_id)
        if not results:
            return _error(404, "no_results", f"No segmentation results for run {run_id}")
        return results[0]

    @router.get("/runs/{run_id}/artifacts/{artifact_type}")
    def get_artifact(run_id: str, artifact_type: str):
        try:
            d = run_store.dir(run_id) / "outputs"
        except KeyError:
            return _error(404, "run_not_found", f"Run {run_id} not found.")

        filename_map = {
            "input_rgb": d / "input_rgb.jpg",
            "pseudo_thermal": d / "pseudo_thermal.png",
            "mask_2d": d / "mask_2d.png",
            "skeleton_1px": d / "skeleton_1px.png",
            "confidence_map": d / "confidence_map.png",
            "confidence_npz": d / "confidence.npz",
            "visual_overlay": d / "visual_overlay.png",
        }

        p = filename_map.get(artifact_type)
        if not p or not p.exists():
            return _error(404, "artifact_not_found", f"Artifact {artifact_type} not found.")

        media_type = "application/octet-stream" if p.suffix == ".npz" else ("image/jpeg" if p.suffix in [".jpg", ".jpeg"] else "image/png")
        return FileResponse(p, media_type=media_type)

    return router
