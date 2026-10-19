"""Component 1 (Road Damage Detection & Mapping) REST API.

Mounted by backend under /api/v1 -> /api/v1/c1/...
Endpoints:
    GET  /api/v1/c1/model                     B0 Model metadata & RDD2022 validation metrics
    POST /api/v1/c1/runs                      Upload road video or images + optional GPS CSV
    POST /api/v1/c1/runs/demo                 Run sample Colombo road survey fixture
    GET  /api/v1/c1/runs                      List recent detection runs
    GET  /api/v1/c1/runs/{run_id}             Run status, stage, progress, and summary
    GET  /api/v1/c1/runs/{run_id}/defects     GeoJSON FeatureCollection for Leaflet map
    GET  /api/v1/c1/runs/{run_id}/records     RoadLens v1 contract JSON records for C2
    GET  /api/v1/c1/runs/{run_id}/artifacts/{kind}/{filename}
                                              Annotated frame overlays or C2 cropped ROIs
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import traceback
import uuid
from pathlib import Path
from typing import Annotated, List, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ..adapters.detector import (
    B0_VALIDATION_METRICS,
    CLASS_DISPLAY,
    CLASS_HEX_COLORS,
    CLASS_NAMES,
    C1Detector,
)
from .store import C1RunStore, now

logger = logging.getLogger("roadlens.c1.api")

C1_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = C1_DIR / "weights" / "base_model_b0.pt"
FIXTURE_DIR = C1_DIR / "fixtures" / "sample_survey"


def _error(status: int, code: str, message: str, details: Optional[dict] = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "code": code,
            "message": message,
            "request_id": uuid.uuid4().hex,
            "details": details or {},
        },
    )


def build_c1_router(
    storage_dir: Optional[Path | str] = None,
    weights_path: Optional[Path | str] = None,
    store: Optional[C1RunStore] = None,
) -> APIRouter:
    storage_path = Path(storage_dir or os.environ.get("C1_STORAGE_DIR") or C1_DIR / "runs")
    weights_file = Path(weights_path or os.environ.get("C1_WEIGHTS_PATH") or WEIGHTS_PATH)

    run_store = store or C1RunStore(storage_path)
    detector = C1Detector(weights_file)
    one_at_a_time = threading.Semaphore(1)

    router = APIRouter(prefix="/c1", tags=["C1 Road Damage Detection"])
    router.store = run_store  # type: ignore[attr-defined]

    @router.get("/model")
    def get_model_info():
        return {
            "model_version": "YOLOv8m-RDD2022-B0",
            "milestone": "50% Progress (Base Model B0 & Single-Frame Localization)",
            "weights_loaded": detector.is_ready,
            "weights_sha256": detector.sha256,
            "weights_path": str(weights_file.name),
            "input_resolution": "800x800",
            "supported_classes": [
                {"id": k, "name": v, "display": CLASS_DISPLAY[v], "color": CLASS_HEX_COLORS[v]}
                for k, v in CLASS_NAMES.items()
            ],
            "b0_benchmark": B0_VALIDATION_METRICS,
            "next_50_scope": [
                "Sri Lankan target dataset completion & route splitting",
                "Few-label fine-tuning (B1) & image-based pseudo-labeling (B2)",
                "Temporal detection association across sequential video frames (FR7)",
                "Temporal consistency & reliability scoring mechanism (FR8)",
                "Duplicate damage consolidation (FR9)",
            ],
        }

    def execute_run(run_id: str, input_files: List[Path], gps_path: Optional[Path], conf: float):
        out_dir = run_store.dir(run_id) / "outputs"
        with one_at_a_time:
            run_store.update(run_id, status="running", started_at=now())

            def progress_cb(stage: str, completed: Optional[int], total: Optional[int]):
                run_store.update(
                    run_id,
                    stage=stage,
                    progress={"completed": completed or 0, "total": total or 0},
                )

            try:
                results = detector.run_detection(
                    input_files=input_files,
                    output_dir=out_dir,
                    gps_csv_path=gps_path,
                    progress_cb=progress_cb,
                    conf_thresh=conf,
                )
                run_store.update(
                    run_id,
                    status="succeeded",
                    stage="done",
                    finished_at=now(),
                    summary=results["summary"],
                )
            except Exception as e:
                logger.error(f"C1 run {run_id} failed: {e}\n{traceback.format_exc()}")
                run_store.update(
                    run_id,
                    status="failed",
                    finished_at=now(),
                    error={"code": "inference_failed", "message": str(e)},
                )

    @router.post("/runs")
    def create_run(
        background_tasks: BackgroundTasks,
        survey_id: Annotated[Optional[str], Form()] = None,
        conf: Annotated[float, Form()] = 0.25,
        video: Annotated[Optional[UploadFile], File()] = None,
        images: Annotated[Optional[List[UploadFile]], File()] = None,
        gps: Annotated[Optional[UploadFile], File()] = None,
    ):
        name = (survey_id or "").strip() or f"survey-{uuid.uuid4().hex[:6]}"
        run_meta = run_store.new(survey_id=name, source="upload")
        run_id = run_meta["run_id"]
        in_dir = run_store.dir(run_id) / "inputs"

        saved_inputs: List[Path] = []
        gps_path: Optional[Path] = None

        # 1. Save video if provided
        if video and video.filename and video.size:
            v_path = in_dir / video.filename
            with open(v_path, "wb") as f:
                shutil.copyfileobj(video.file, f)
            saved_inputs.append(v_path)

        # 2. Save images if provided
        if images:
            for img in images:
                if img.filename and img.size:
                    i_path = in_dir / img.filename
                    with open(i_path, "wb") as f:
                        shutil.copyfileobj(img.file, f)
                    saved_inputs.append(i_path)

        # 3. Save GPS if provided
        if gps and gps.filename and gps.size:
            gps_path = in_dir / gps.filename
            with open(gps_path, "wb") as f:
                shutil.copyfileobj(gps.file, f)

        if not saved_inputs:
            return _error(422, "missing_input", "Please provide a road video file or at least one road image.")

        background_tasks.add_task(execute_run, run_id, saved_inputs, gps_path, conf)
        return run_meta

    @router.post("/runs/demo")
    def create_demo_run(background_tasks: BackgroundTasks):
        """Executes a sample survey run on Colombo Galle Road fixture."""
        run_meta = run_store.new(survey_id="Sample Drive: Colombo Galle Road", source="demo")
        run_id = run_meta["run_id"]
        in_dir = run_store.dir(run_id) / "inputs"

        sample_imgs_dir = FIXTURE_DIR / "images"
        sample_gps = FIXTURE_DIR / "colombo_galle_road_gps.csv"

        saved_inputs = []
        if sample_imgs_dir.exists():
            for p in sorted(sample_imgs_dir.glob("*.jpg")):
                dest = in_dir / p.name
                shutil.copy2(p, dest)
                saved_inputs.append(dest)

        dest_gps = in_dir / "gps.csv"
        if sample_gps.exists():
            shutil.copy2(sample_gps, dest_gps)
            gps_arg = dest_gps
        else:
            gps_arg = None

        background_tasks.add_task(execute_run, run_id, saved_inputs, gps_arg, 0.25)
        return run_meta

    @router.get("/runs")
    def list_runs():
        return run_store.list()

    @router.get("/runs/{run_id}")
    def get_run(run_id: str):
        try:
            return run_store.get(run_id)
        except KeyError:
            return _error(404, "run_not_found", f"Run {run_id} does not exist.")

    @router.get("/runs/{run_id}/defects")
    def get_defects_geojson(run_id: str):
        try:
            return run_store.defects_geojson(run_id)
        except KeyError:
            return _error(404, "run_not_found", f"Run {run_id} does not exist.")

    @router.get("/runs/{run_id}/records")
    def get_records(run_id: str):
        try:
            return run_store.records(run_id)
        except KeyError:
            return _error(404, "run_not_found", f"Run {run_id} does not exist.")

    @router.get("/runs/{run_id}/artifacts/{kind}/{filename}")
    def get_artifact(run_id: str, kind: str, filename: str):
        try:
            p = run_store.artifact_path(run_id, kind, filename)
            if not p or not p.is_file():
                return _error(404, "artifact_not_found", f"Artifact {kind}/{filename} not found.")
            media_type = "image/jpeg" if filename.lower().endswith((".jpg", ".jpeg")) else "application/octet-stream"
            return FileResponse(p, media_type=media_type)
        except KeyError:
            return _error(404, "run_not_found", f"Run {run_id} does not exist.")

    return router
