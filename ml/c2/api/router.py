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

import json
import logging
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ml.c2.adapters.segmenter_adapter import (
    DEFAULT_CGAN_WEIGHTS,
    DEFAULT_SEGMENTER_WEIGHTS,
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
    c1_storage_dir: Path | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/c2", tags=["c2-segmentation"])
    c1_root = Path(
        c1_storage_dir
        or getattr(store, "c1_root", None)
        or os.environ.get("C1_STORAGE_DIR")
        or (C2_DIR.parent / "c1" / "runs")
    )
    run_store = store or RunStore(storage_dir, c1_storage_dir=c1_root)
    if hasattr(run_store, "c1_root") and c1_storage_dir is not None:
        run_store.c1_root = c1_root

    def _resolve_c1_handoff(c1_run_id: str, c1_observation_id: str) -> tuple[dict | None, Path | None]:
        obs = None
        if hasattr(run_store, "get_c1_observation"):
            obs = run_store.get_c1_observation(c1_run_id, c1_observation_id)

        crops_dir = c1_root / c1_run_id / "outputs" / "crops"
        crop_path: Path | None = None
        if crops_dir.is_dir():
            if obs and obs.get("crop_artifact"):
                candidate = crops_dir / Path(str(obs["crop_artifact"])).name
                if candidate.is_file():
                    crop_path = candidate
            if crop_path is None:
                direct = crops_dir / f"{c1_observation_id}_crop.jpg"
                if direct.is_file():
                    crop_path = direct
            if crop_path is None:
                matches = sorted(crops_dir.glob(f"*{c1_observation_id}*"))
                if matches:
                    crop_path = matches[0]
        return obs, crop_path

    def _schedule_c2_execution(
        background_tasks: BackgroundTasks,
        *,
        run_id: str,
        run_dir: Path,
        survey_id: str,
        input_crop_path: Path,
        damage_class: str,
        threshold: float,
        c1_run_id: str | None = None,
        c1_observation_id: str | None = None,
        c1_obs: dict | None = None,
    ) -> None:
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
                    upstream_c1_defect_id=c1_obs.get("defect_id") if c1_obs else None,
                    upstream_c1_confidence=c1_obs.get("confidence") if c1_obs else None,
                    upstream_c1_bbox=c1_obs.get("bbox") if c1_obs else None,
                    upstream_c1_crop_transform=c1_obs.get("crop_transform") if c1_obs else None,
                    upstream_c1_frame_dimensions=c1_obs.get("frame_dimensions") if c1_obs else None,
                    upstream_c1_frame_idx=c1_obs.get("frame_idx") if c1_obs else None,
                    upstream_c1_source_media=c1_obs.get("source_media") if c1_obs else None,
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
    def list_c1_crops(limit: int = 50, c1_run_id: str | None = None):
        crops = run_store.get_available_c1_crops(limit=max(1, min(limit, 200)), c1_run_id=c1_run_id)
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
        is_c1_handoff = bool(c1_run_id and c1_observation_id)
        source = "c1_handoff" if is_c1_handoff else ("demo" if active_sample_id else "upload")

        c1_obs: dict | None = None
        c1_crop_path: Path | None = None
        if is_c1_handoff and c1_run_id and c1_observation_id:
            c1_obs, c1_crop_path = _resolve_c1_handoff(c1_run_id, c1_observation_id)
            if c1_obs:
                if survey_id in ("survey_default", "", "SURVEY-C1-FEED", c1_run_id):
                    survey_id = c1_obs.get("survey_id") or survey_id
                if damage_class in ("crack", "") or c1_obs.get("damage_class"):
                    damage_class = c1_obs.get("damage_class") or damage_class
            if c1_crop_path is None and (target_file is None or not target_file.filename):
                return _error(
                    404,
                    "c1_crop_not_found",
                    f"C1 crop for observation '{c1_observation_id}' in run '{c1_run_id}' was not found.",
                    {"c1_run_id": c1_run_id, "c1_observation_id": c1_observation_id},
                )

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

        # Determine input file: prefer authoritative C1 crop on disk when c1_handoff is active
        if is_c1_handoff and c1_crop_path is not None and c1_crop_path.is_file():
            shutil.copy(c1_crop_path, input_crop_path)
        elif target_file is not None and target_file.filename:
            with open(input_crop_path, "wb") as f:
                content = target_file.file.read()
                f.write(content)
        elif active_sample_id:
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
                img = np.full((256, 256, 3), 110, dtype=np.uint8)
                cv2.line(img, (40, 20), (210, 230), (30, 30, 30), 3)
                cv2.imwrite(str(input_crop_path), img)
        else:
            img = np.full((256, 256, 3), 128, dtype=np.uint8)
            cv2.line(img, (30, 30), (220, 220), (30, 30, 30), 2)
            cv2.imwrite(str(input_crop_path), img)

        _schedule_c2_execution(
            background_tasks,
            run_id=run_id,
            run_dir=run_dir,
            survey_id=survey_id,
            input_crop_path=input_crop_path,
            damage_class=damage_class,
            threshold=threshold,
            c1_run_id=c1_run_id,
            c1_observation_id=c1_observation_id,
            c1_obs=c1_obs,
        )
        return {
            "run_id": run_id,
            "status": "queued",
            "source": source,
            "survey_id": survey_id,
            "damage_class": damage_class,
            "c1_run_id": c1_run_id,
            "c1_observation_id": c1_observation_id,
        }

    @router.post("/runs/from-c1", status_code=202)
    def create_runs_from_c1(
        background_tasks: BackgroundTasks,
        c1_run_id: Annotated[str, Form(description="Upstream C1 run_id")],
        observation_ids: Annotated[
            str | None,
            Form(description="Optional comma-separated or JSON array of C1 observation record_ids"),
        ] = None,
        threshold: Annotated[float, Form()] = 0.5,
    ):
        available = run_store.get_available_c1_crops(limit=200, c1_run_id=c1_run_id)
        if not available:
            return _error(
                404,
                "c1_crops_not_found",
                f"No C1 defect crops found for C1 run '{c1_run_id}'.",
                {"c1_run_id": c1_run_id},
            )

        selected_ids: set[str] | None = None
        if observation_ids:
            raw = observation_ids.strip()
            if raw.startswith("["):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        selected_ids = {str(x).strip() for x in parsed if str(x).strip()}
                except json.JSONDecodeError:
                    return _error(422, "invalid_input", "observation_ids must be a valid JSON list or comma-separated string.")
            else:
                selected_ids = {part.strip() for part in raw.split(",") if part.strip()}

        targets = [c for c in available if selected_ids is None or c.get("record_id") in selected_ids]
        if not targets:
            return _error(
                404,
                "c1_observations_not_found",
                "None of the requested observation_ids matched available crops in the C1 run.",
                {"c1_run_id": c1_run_id},
            )

        created_runs: list[dict] = []
        survey_id = targets[0].get("survey_id") or f"survey-{c1_run_id[:8]}"

        for obs in targets:
            obs_id = obs["record_id"]
            _, crop_path = _resolve_c1_handoff(c1_run_id, obs_id)
            if crop_path is None or not crop_path.is_file():
                continue

            obs_survey_id = obs.get("survey_id") or survey_id
            obs_damage_class = obs.get("damage_class") or "crack"
            meta = run_store.new(
                survey_id=obs_survey_id,
                source="c1_handoff",
                damage_class=obs_damage_class,
                c1_run_id=c1_run_id,
                c1_observation_id=obs_id,
            )
            r_id = meta["run_id"]
            r_dir = run_store.dir(r_id)
            input_crop_path = r_dir / "inputs" / "crop.jpg"
            shutil.copy(crop_path, input_crop_path)

            _schedule_c2_execution(
                background_tasks,
                run_id=r_id,
                run_dir=r_dir,
                survey_id=obs_survey_id,
                input_crop_path=input_crop_path,
                damage_class=obs_damage_class,
                threshold=threshold,
                c1_run_id=c1_run_id,
                c1_observation_id=obs_id,
                c1_obs=obs,
            )
            created_runs.append(
                {
                    "run_id": r_id,
                    "c1_run_id": c1_run_id,
                    "c1_observation_id": obs_id,
                    "defect_id": obs.get("defect_id"),
                    "damage_class": obs_damage_class,
                    "status": "queued",
                }
            )

        if not created_runs:
            return _error(
                404,
                "c1_crop_files_missing",
                f"Crop files on disk were not found for C1 run '{c1_run_id}'.",
                {"c1_run_id": c1_run_id},
            )

        return {
            "c1_run_id": c1_run_id,
            "survey_id": survey_id,
            "count": len(created_runs),
            "runs": created_runs,
        }

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
