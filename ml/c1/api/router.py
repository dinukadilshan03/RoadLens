"""
C1 REST API (FastAPI router, mounted by the RoadLens backend under /api/v1).

    GET  /api/v1/c1/model                                model info (version, sha256, classes, B0 metrics)
    POST /api/v1/c1/runs                                 upload road image(s) or video -> 202 Accepted
    GET  /api/v1/c1/runs                                 recent C1 runs
    GET  /api/v1/c1/runs/{run_id}                        status, stage, progress, summary, warnings, error
    POST /api/v1/c1/runs/{run_id}/cancel                 cooperative cancellation
    GET  /api/v1/c1/runs/{run_id}/records                contract observation records (damage_class, bbox, confidence)
    GET  /api/v1/c1/runs/{run_id}/artifacts/{kind}/{fn}  annotated frames (kind=annotated) or C2 crops (kind=crops)

Settings (environment variables):
    C1_STORAGE_DIR   run folders (default: ml/c1/runs, git-ignored; use a volume in Docker)
    C1_WEIGHTS_PATH  trained YOLOv8m weights (default: ml/c1/weights/base_model_b0.pt)

Errors use the platform format {code, message, request_id, details}:
    422 invalid input, 404 unknown run/artifact, 409 run not finished, 413 file too large, 503 model unavailable.
"""
import logging
import os
import re
import shutil
import threading
import traceback
import uuid
from pathlib import Path
from typing import Annotated

SAMPLE_FILES: dict[str, str] = {
    "pothole_alligator": "sample_pothole_alligator.jpg",
    "potholes": "sample_potholes.jpg",
    "alligator": "sample_alligator.jpg",
    "longitudinal": "sample_longitudinal.jpg",
}

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ..adapters.detector import (
    DEFAULT_WEIGHTS_PATH,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    C1Cancelled,
    C1InputError,
    C1Request,
    get_model_info,
    run,
)
from .store import RunStore, now

log = logging.getLogger("roadlens.c1")
C1_DIR = Path(__file__).resolve().parents[1]
MAX_IMAGE_BYTES = 25 * 1024 * 1024         # 25 MiB per image (per docs/architecture.md)
MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024   # 2 GiB per video (per docs/architecture.md)
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
    """RunContext that reports progress into the run store and honours cancel requests."""

    def __init__(self, store, run_id: str, output_dir: Path):
        self.store = store
        self.run_id = run_id
        self.output_dir = Path(output_dir)

    def progress(self, stage: str, completed: int | None = None, total: int | None = None) -> None:
        self.store.update(self.run_id, stage=stage, progress={"completed": completed, "total": total})

    def is_cancelled(self) -> bool:
        return bool(self.store.get(self.run_id).get("cancel_requested"))


def _sanitize_filename(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name or fallback).name)
    return cleaned or fallback


def _save_upload(upload: UploadFile, path: Path, limit: int) -> int:
    size = 0
    with open(path, "wb") as f:
        while chunk := upload.file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                raise OverflowError(upload.filename or path.name)
            f.write(chunk)
    if size == 0:
        raise ValueError(f"'{upload.filename or path.name}' is empty.")
    return size


def build_router(storage_dir=None, weights_path=None, store=None) -> APIRouter:
    """store: any object with the RunStore interface (default: file-based RunStore in storage_dir)."""
    storage_dir = Path(storage_dir or os.environ.get("C1_STORAGE_DIR") or C1_DIR / "runs")
    weights_path = Path(weights_path or os.environ.get("C1_WEIGHTS_PATH") or DEFAULT_WEIGHTS_PATH)
    store = store or RunStore(storage_dir)
    one_at_a_time = threading.Semaphore(1)  # serialize heavy YOLO inference jobs

    r = APIRouter(prefix="/c1", tags=["C1 road damage detection"])
    r.store = store

    def execute(run_id: str, req_kwargs: dict) -> None:
        d = store.dir(run_id)
        with one_at_a_time:
            if store.get(run_id).get("cancel_requested"):
                store.update(run_id, status="cancelled", finished_at=now())
                return
            store.update(run_id, status="running", started_at=now())
            try:
                res = run(
                    C1Request(weights_path=str(weights_path), external_run_id=run_id, **req_kwargs),
                    _StoreContext(store, run_id, d / "outputs"),
                )
                store.complete(
                    run_id,
                    res.records,
                    status="succeeded",
                    stage="done",
                    finished_at=now(),
                    model_version=res.summary["producer"]["model_version"],
                    weights_sha256=res.summary["producer"]["weights_sha256"],
                    summary=res.summary,
                    warnings=res.warnings,
                    artifacts=res.artifacts,
                )
            except C1InputError as exc:
                store.update(
                    run_id,
                    status="failed",
                    finished_at=now(),
                    error={"code": "invalid_input", "message": str(exc), "retryable": False},
                )
            except C1Cancelled as exc:
                store.update(
                    run_id,
                    status="cancelled",
                    finished_at=now(),
                    error={"code": "cancelled", "message": str(exc), "retryable": False},
                )
            except LookupError as exc:
                store.update(
                    run_id,
                    status="failed",
                    finished_at=now(),
                    error={"code": "model_unavailable", "message": str(exc), "retryable": True},
                )
            except Exception as exc:  # noqa: BLE001
                log.error("C1 run %s failed:\n%s", run_id, traceback.format_exc())
                store.update(
                    run_id,
                    status="failed",
                    finished_at=now(),
                    error={
                        "code": "internal_error",
                        "message": f"Internal processing error ({type(exc).__name__}).",
                        "retryable": True,
                    },
                )

    def public(meta: dict) -> dict:
        return {k: v for k, v in meta.items() if k != "artifacts"}

    def finished(run_id: str) -> JSONResponse | None:
        try:
            meta = store.get(run_id)
        except KeyError:
            return _error(404, "not_found", "Unknown C1 run.", {"run_id": run_id})
        if meta["status"] != "succeeded":
            return _error(
                409,
                "run_not_succeeded",
                f"Run is {meta['status']}; results are not available.",
                {"status": meta["status"]},
            )
        return None

    @r.get("/model")
    def model_info():
        info = get_model_info(weights_path)
        if not info["weights_ready"]:
            return _error(
                503,
                "model_unavailable",
                f"C1 base model weights not found at {weights_path}.",
                {"expected_path": str(weights_path)},
            )
        return info

    @r.post("/runs", status_code=202)
    def create_run(
        background: BackgroundTasks,
        files: Annotated[
            list[UploadFile],
            File(description="One or more road images (.jpg, .png, .webp) or video files (.mp4, .avi, .mov)"),
        ],
        conf_threshold: Annotated[float, Form(description="Detection confidence threshold in [0.01, 0.99]")] = 0.25,
        survey_id: Annotated[str | None, Form(description="Optional survey/recording identifier")] = None,
    ):
        if not (0.01 <= conf_threshold <= 0.99):
            return _error(422, "invalid_input", "conf_threshold must be between 0.01 and 0.99.")
        if not files:
            return _error(422, "invalid_input", "Upload at least one image or video file.")
        if not weights_path.is_file():
            return _error(
                503,
                "model_unavailable",
                f"C1 base model weights not found at {weights_path}.",
            )

        # Validate extensions before creating a run
        allowed_exts = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
        for up in files:
            ext = Path(up.filename or "").suffix.lower()
            if ext not in allowed_exts:
                return _error(
                    422,
                    "invalid_input",
                    f"Unsupported media file '{up.filename}'. Supported extensions: {sorted(allowed_exts)}.",
                )

        survey_id = (survey_id or "").strip()[:128] or f"survey-{uuid.uuid4().hex[:8]}"
        meta = store.new(survey_id, "upload", conf_threshold=conf_threshold)
        inputs_dir = store.dir(meta["run_id"]) / "inputs"
        media_paths: list[str] = []

        try:
            for idx, up in enumerate(files):
                ext = Path(up.filename or "").suffix.lower()
                limit = MAX_VIDEO_BYTES if ext in VIDEO_EXTENSIONS else MAX_IMAGE_BYTES
                safe_name = _sanitize_filename(up.filename or f"input_{idx}{ext}", f"input_{idx}{ext}")
                dest = inputs_dir / f"{idx:03d}_{safe_name}"
                _save_upload(up, dest, limit)
                media_paths.append(str(dest))
        except OverflowError as exc:
            store.update(
                meta["run_id"],
                status="rejected",
                error={"code": "too_large", "message": f"File '{exc}' exceeds maximum allowed size."},
            )
            return _error(413, "too_large", "Images must be at most 25 MiB and videos at most 2 GiB.")
        except ValueError as exc:
            store.update(
                meta["run_id"],
                status="rejected",
                error={"code": "invalid_input", "message": str(exc)},
            )
            return _error(422, "invalid_input", str(exc))

        background.add_task(
            execute,
            meta["run_id"],
            {
                "survey_id": survey_id,
                "media_paths": media_paths,
                "conf_threshold": conf_threshold,
            },
        )
        return public(meta)

    @r.post("/runs/demo", status_code=202)
    def create_demo_run(
        background: BackgroundTasks,
        sample_key: Annotated[str, Form(description="Sample key: pothole_alligator, potholes, alligator, longitudinal")] = "pothole_alligator",
        conf_threshold: Annotated[float, Form(description="Detection confidence threshold")] = 0.25,
    ):
        if not (0.01 <= conf_threshold <= 0.99):
            return _error(422, "invalid_input", "conf_threshold must be between 0.01 and 0.99.")
        if not weights_path.is_file():
            return _error(
                503,
                "model_unavailable",
                f"C1 base model weights not found at {weights_path}.",
            )
        filename = SAMPLE_FILES.get(sample_key, "sample_pothole_alligator.jpg")
        sample_path = C1_DIR / "fixtures" / "samples" / filename
        if not sample_path.is_file():
            return _error(404, "sample_not_found", f"Demo sample '{sample_key}' ({filename}) not found.")

        survey_id = f"demo-{sample_key}"
        meta = store.new(survey_id, "demo", conf_threshold=conf_threshold)
        inputs_dir = store.dir(meta["run_id"]) / "inputs"
        dest = inputs_dir / filename
        shutil.copyfile(sample_path, dest)

        background.add_task(
            execute,
            meta["run_id"],
            {
                "survey_id": survey_id,
                "media_paths": [str(dest)],
                "conf_threshold": conf_threshold,
            },
        )
        return public(meta)

    @r.get("/runs")
    def list_runs(limit: int = 50):
        return {"runs": [public(m) for m in store.list(max(1, min(limit, 200)))]}

    @r.get("/runs/{run_id}")
    def get_run(run_id: str):
        try:
            return public(store.get(run_id))
        except KeyError:
            return _error(404, "not_found", "Unknown C1 run.", {"run_id": run_id})

    @r.post("/runs/{run_id}/cancel", status_code=202)
    def cancel_run(run_id: str):
        try:
            meta = store.get(run_id)
        except KeyError:
            return _error(404, "not_found", "Unknown C1 run.", {"run_id": run_id})
        if meta["status"] not in ("queued", "running"):
            return _error(409, "not_cancellable", f"Run is already {meta['status']}.")
        return public(store.update(run_id, cancel_requested=True))

    @r.get("/runs/{run_id}/records")
    def records(run_id: str):
        if err := finished(run_id):
            return err
        recs = store.records(run_id)
        enriched = []
        for rec in recs:
            item = dict(rec)
            if item.get("annotated_frame"):
                item["annotated_url"] = f"/api/v1/c1/runs/{run_id}/artifacts/annotated/{item['annotated_frame']}"
            if item.get("crop_artifact"):
                item["crop_url"] = f"/api/v1/c1/runs/{run_id}/artifacts/crops/{item['crop_artifact']}"
            enriched.append(item)
        return {"records": enriched}

    @r.get("/runs/{run_id}/artifacts/{kind}/{filename}")
    def get_artifact(run_id: str, kind: str, filename: str):
        if kind not in ("annotated", "crops"):
            return _error(404, "not_found", f"Unknown artifact kind '{kind}'.")
        if not SAFE_FILENAME.match(filename) or ".." in filename:
            return _error(422, "invalid_input", "Invalid artifact filename.")
        if err := finished(run_id):
            return err
        path = store.dir(run_id) / "outputs" / kind / filename
        if not path.is_file():
            return _error(404, "not_found", "Artifact file not found.", {"filename": filename})
        return FileResponse(path, media_type="image/jpeg", filename=filename)

    return r

