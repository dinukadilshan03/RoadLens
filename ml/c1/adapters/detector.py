"""
Component 1 (C1) Road Damage Detection Adapter.

Wraps the trained YOLOv8m base model (base_model_b0.pt, trained on RDD2022 4 core classes)
to produce standardized RoadLens C1 detection records:
    - damage_class  (Longitudinal_Crack_D00, Transverse_Crack_D10, Alligator_Crack_D20, Pothole_D40)
    - bbox          ([x_min, y_min, x_max, y_max] in source frame pixel coordinates)
    - confidence    (detection score in [0.0, 1.0])
    - annotated_frame and cropped defect ROIs (crop_artifact + crop_transform) for downstream C2.

Metadata fields (timestamp_ms, location, location_null_reason) are included as nullable fields
per docs/interfaces.md and docs/data-plan.md so visual detection works standalone without GPS
and seamlessly accepts GPS/timestamp enrichment.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
from PIL import Image, ImageDraw

logger = logging.getLogger("roadlens.c1.detector")

C1_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_PATH = C1_DIR / "weights" / "base_model_b0.pt"

CLASS_NAMES: dict[int, str] = {
    0: "Longitudinal_Crack_D00",
    1: "Transverse_Crack_D10",
    2: "Alligator_Crack_D20",
    3: "Pothole_D40",
}

CLASS_DISPLAY: dict[str, str] = {
    "Longitudinal_Crack_D00": "Longitudinal Crack",
    "Transverse_Crack_D10": "Transverse Crack",
    "Alligator_Crack_D20": "Alligator Crack",
    "Pothole_D40": "Pothole",
}

CLASS_HEX_COLORS: dict[str, str] = {
    "Longitudinal_Crack_D00": "#3b82f6",  # Blue
    "Transverse_Crack_D10": "#eab308",    # Amber
    "Alligator_Crack_D20": "#f97316",     # Orange
    "Pothole_D40": "#ef4444",             # Red
}

B0_VALIDATION_METRICS: dict[str, Any] = {
    "dataset": "RDD2022 Cleaned (38,269 images across 6 countries)",
    "architecture": "YOLOv8m (25.8M parameters, 800x800 resolution)",
    "overall": {
        "precision": 0.684,
        "recall": 0.619,
        "mAP50": 0.662,
        "mAP50_95": 0.378,
    },
    "per_class_ap50": {
        "Longitudinal_Crack_D00": 0.603,
        "Transverse_Crack_D10": 0.587,
        "Alligator_Crack_D20": 0.672,
        "Pothole_D40": 0.784,
    },
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class C1InputError(ValueError):
    """Raised when input media or parameters fail C1 validation."""


class C1Cancelled(RuntimeError):
    """Raised when a running C1 job is cooperatively cancelled."""


class RunContext(Protocol):
    output_dir: Path

    def progress(self, stage: str, completed: int | None = None, total: int | None = None) -> None: ...

    def is_cancelled(self) -> bool: ...


@dataclass
class C1Request:
    survey_id: str
    external_run_id: str
    media_paths: list[str]
    weights_path: str = str(DEFAULT_WEIGHTS_PATH)
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 800
    max_video_frames: int = 40


@dataclass
class C1Result:
    records: list[dict[str, Any]]
    summary: dict[str, Any]
    artifacts: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, Any] = {}
_SHA_CACHE: dict[str, str] = {}


def compute_sha256(path: Path) -> str:
    key = str(path.resolve())
    if key in _SHA_CACHE:
        return _SHA_CACHE[key]
    if not path.is_file():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _SHA_CACHE[key] = digest
    return digest


def _get_yolo_model(weights_path: Path):
    if not weights_path.is_file():
        raise LookupError(f"C1 base model weights not found at {weights_path}")
    key = str(weights_path.resolve())
    with _MODEL_LOCK:
        if key not in _MODEL_CACHE:
            from ultralytics import YOLO

            logger.info("Loading YOLOv8m B0 model from %s", weights_path)
            _MODEL_CACHE[key] = YOLO(str(weights_path))
        return _MODEL_CACHE[key]


def get_model_info(weights_path: str | Path = DEFAULT_WEIGHTS_PATH) -> dict[str, Any]:
    wpath = Path(weights_path)
    exists = wpath.is_file()
    sha = compute_sha256(wpath) if exists else "missing"
    size_mb = round(wpath.stat().st_size / (1024 * 1024), 2) if exists else 0.0
    return {
        "component": "C1",
        "model_version": "YOLOv8m-RDD2022-B0",
        "weights_ready": exists,
        "weights_filename": wpath.name,
        "weights_size_mb": size_mb,
        "weights_sha256": sha,
        "input_resolution": 800,
        "classes": [
            {
                "id": cid,
                "name": cname,
                "display_name": CLASS_DISPLAY[cname],
                "color": CLASS_HEX_COLORS[cname],
                "b0_ap50": B0_VALIDATION_METRICS["per_class_ap50"][cname],
            }
            for cid, cname in CLASS_NAMES.items()
        ],
        "validation_metrics": B0_VALIDATION_METRICS,
    }


class C1Detector:
    """Executes YOLOv8m (B0) road damage detection on uploaded images or video frames."""

    def __init__(self, weights_path: str | Path = DEFAULT_WEIGHTS_PATH):
        self.weights_path = Path(weights_path)
        if not self.weights_path.is_file():
            raise LookupError(f"C1 model weights not found at {self.weights_path}")
        self.sha256 = compute_sha256(self.weights_path)

    def run(self, request: C1Request, context: RunContext) -> C1Result:
        if not (0.01 <= request.conf_threshold <= 0.99):
            raise C1InputError("conf_threshold must be between 0.01 and 0.99.")
        if not request.media_paths:
            raise C1InputError("At least one image or video file is required.")

        output_dir = Path(context.output_dir)
        raw_frames_dir = output_dir / "frames_raw"
        annotated_dir = output_dir / "annotated"
        crops_dir = output_dir / "crops"
        for d in (raw_frames_dir, annotated_dir, crops_dir):
            d.mkdir(parents=True, exist_ok=True)

        warnings: list[str] = []
        context.progress("extracting_frames", completed=0, total=len(request.media_paths))
        extracted_frames = self._prepare_frames(
            request.media_paths, raw_frames_dir, request.max_video_frames, warnings, context
        )
        if not extracted_frames:
            raise C1InputError("No readable road images or video frames could be decoded.")

        model = _get_yolo_model(self.weights_path)
        records: list[dict[str, Any]] = []
        frame_summaries: list[dict[str, Any]] = []
        total_frames = len(extracted_frames)

        producer = {
            "model_version": "YOLOv8m-RDD2022-B0",
            "weights_sha256": self.sha256[:16],
            "code_version": "1.0.0",
            "config_version": f"b0-imgsz{request.imgsz}-conf{request.conf_threshold:.2f}",
        }

        for idx, frame_info in enumerate(extracted_frames):
            if context.is_cancelled():
                raise C1Cancelled("C1 run cancelled by user.")
            context.progress("running_inference", completed=idx, total=total_frames)

            frame_path = Path(frame_info["path"])
            preds = self._predict_frame(
                model=model,
                frame_path=frame_path,
                conf_threshold=request.conf_threshold,
                iou_threshold=request.iou_threshold,
                imgsz=request.imgsz,
            )

            annotated_name = f"frame_{idx:04d}_annotated.jpg"
            annotated_path = annotated_dir / annotated_name
            frame_records, width, height = self._annotate_and_crop(
                frame_path=frame_path,
                annotated_save_path=annotated_path,
                crops_dir=crops_dir,
                preds=preds,
                frame_idx=idx,
                source_media=frame_info["source_media"],
                timestamp_ms=frame_info["timestamp_ms"],
                survey_id=request.survey_id,
                external_run_id=request.external_run_id,
                producer=producer,
            )
            records.extend(frame_records)
            frame_summaries.append({
                "frame_idx": idx,
                "source_media": frame_info["source_media"],
                "width": width,
                "height": height,
                "timestamp_ms": frame_info["timestamp_ms"],
                "annotated_frame": annotated_name,
                "defect_count": len(frame_records),
            })

        context.progress("writing_outputs", completed=total_frames, total=total_frames)
        summary = self._build_summary(
            request=request,
            producer=producer,
            frame_summaries=frame_summaries,
            records=records,
        )

        records_jsonl_path = output_dir / "c1_records.jsonl"
        with open(records_jsonl_path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(rec, allow_nan=False) + "\n" for rec in records)

        summary_json_path = output_dir / "c1_run_summary.json"
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, allow_nan=False)

        artifacts = {
            "records_jsonl": "c1_records.jsonl",
            "summary_json": "c1_run_summary.json",
            "annotated_frames": [fs["annotated_frame"] for fs in frame_summaries],
            "crop_images": [r["crop_artifact"] for r in records if r.get("crop_artifact")],
        }
        return C1Result(records=records, summary=summary, artifacts=artifacts, warnings=warnings)

    def _prepare_frames(
        self,
        media_paths: list[str],
        raw_frames_dir: Path,
        max_video_frames: int,
        warnings: list[str],
        context: RunContext,
    ) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        for m_idx, raw_path in enumerate(media_paths):
            if context.is_cancelled():
                raise C1Cancelled("C1 run cancelled during frame preparation.")
            p = Path(raw_path)
            if not p.is_file():
                warnings.append(f"Input file {p.name} does not exist and was skipped.")
                continue
            ext = p.suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                try:
                    with Image.open(p) as im:
                        im.verify()
                except Exception as exc:
                    raise C1InputError(f"Corrupted or unreadable image file '{p.name}': {exc}") from exc
                frames.append({
                    "path": str(p),
                    "source_media": p.name,
                    "timestamp_ms": None,
                })
            elif ext in VIDEO_EXTENSIONS:
                video_frames = self._extract_video_frames(p, raw_frames_dir, max_video_frames, len(frames))
                if not video_frames:
                    raise C1InputError(f"Could not decode video frames from '{p.name}'.")
                frames.extend(video_frames)
            else:
                raise C1InputError(
                    f"Unsupported media extension '{ext}' for '{p.name}'. "
                    f"Supported images: {sorted(IMAGE_EXTENSIONS)}; videos: {sorted(VIDEO_EXTENSIONS)}."
                )
            context.progress("extracting_frames", completed=m_idx + 1, total=len(media_paths))
        return frames

    @staticmethod
    def _extract_video_frames(
        video_path: Path,
        save_dir: Path,
        max_frames: int,
        start_offset: int,
    ) -> list[dict[str, Any]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, total_frames // max_frames) if total_frames > max_frames else max(1, int(fps // 2))

        extracted: list[dict[str, Any]] = []
        frame_pos = 0
        saved_count = 0

        while cap.isOpened() and saved_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_pos % step == 0:
                pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_ms = int(pos_msec if pos_msec and pos_msec > 0 else (frame_pos / fps) * 1000)
                fname = f"raw_{start_offset + saved_count:04d}.jpg"
                fpath = save_dir / fname
                cv2.imwrite(str(fpath), frame)
                extracted.append({
                    "path": str(fpath),
                    "source_media": video_path.name,
                    "timestamp_ms": timestamp_ms,
                })
                saved_count += 1
            frame_pos += 1

        cap.release()
        return extracted

    @staticmethod
    def _predict_frame(
        model: Any,
        frame_path: Path,
        conf_threshold: float,
        iou_threshold: float,
        imgsz: int,
    ) -> list[dict[str, Any]]:
        results = model.predict(
            source=str(frame_path),
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False,
        )
        detections: list[dict[str, Any]] = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = [float(c) for c in boxes.xyxy[i].tolist()]
                class_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")
                detections.append({
                    "class_id": cls_id,
                    "damage_class": class_name,
                    "display_name": CLASS_DISPLAY.get(class_name, class_name),
                    "confidence": round(conf, 4),
                    "bbox_xyxy": xyxy,
                })
        return detections

    @staticmethod
    def _annotate_and_crop(
        frame_path: Path,
        annotated_save_path: Path,
        crops_dir: Path,
        preds: list[dict[str, Any]],
        frame_idx: int,
        source_media: str,
        timestamp_ms: int | None,
        survey_id: str,
        external_run_id: str,
        producer: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int, int]:
        img = Image.open(frame_path).convert("RGB")
        orig_img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size

        records: list[dict[str, Any]] = []
        for pred in preds:
            raw_x1, raw_y1, raw_x2, raw_y2 = pred["bbox_xyxy"]
            x1 = round(max(0.0, min(float(w), raw_x1)), 1)
            y1 = round(max(0.0, min(float(h), raw_y1)), 1)
            x2 = round(max(0.0, min(float(w), raw_x2)), 1)
            y2 = round(max(0.0, min(float(h), raw_y2)), 1)

            if x2 <= x1 or y2 <= y1:
                continue

            record_id = f"obs_{uuid.uuid4().hex[:12]}"
            defect_id = f"def_{uuid.uuid4().hex[:8]}"
            cname = pred["damage_class"]
            display_name = pred["display_name"]
            conf = pred["confidence"]
            color = CLASS_HEX_COLORS.get(cname, "#ffffff")

            # Draw bounding box and confidence badge on the annotated copy
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = f"{display_name} ({conf * 100:.1f}%)"
            badge_w = len(label) * 7 + 8
            badge_y1 = max(0.0, y1 - 20.0)
            draw.rectangle([x1, badge_y1, min(float(w), x1 + badge_w), badge_y1 + 20.0], fill=color)
            draw.text((x1 + 4, badge_y1 + 3), label, fill="#ffffff")

            # Extract clean (unannotated) defect ROI crop for downstream C2 Crack Segmentation
            crop_name: str | None = f"{record_id}_crop.jpg"
            crop_w = round(x2 - x1, 1)
            crop_h = round(y2 - y1, 1)
            if crop_w >= 4 and crop_h >= 4:
                crop_box = (int(x1), int(y1), max(int(x1) + 1, int(x2)), max(int(y1) + 1, int(y2)))
                crop = orig_img.crop(crop_box)
                crop.save(crops_dir / crop_name, quality=92)
            else:
                crop_name = None

            crop_transform = {
                "origin_xy": [x1, y1],
                "width": crop_w,
                "height": crop_h,
                "source_frame_size": [w, h],
            }

            records.append({
                "schema_version": "1.0",
                "component": "C1",
                "record_id": record_id,
                "defect_id": defect_id,
                "external_run_id": external_run_id,
                "survey_id": survey_id,
                "source_media": source_media,
                "frame_idx": frame_idx,
                "frame_dimensions": {"width": w, "height": h},
                "timestamp_ms": timestamp_ms,
                "damage_class": cname,
                "class_id": pred["class_id"],
                "display_name": display_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "annotated_frame": annotated_save_path.name,
                "crop_artifact": crop_name,
                "crop_transform": crop_transform,
                "location": None,
                "location_null_reason": "metadata_not_supplied",
                "producer": producer,
                "source_kind": "inference",
                "upstream": None,
            })

        img.save(annotated_save_path, quality=90)
        return records, w, h

    @staticmethod
    def _build_summary(
        request: C1Request,
        producer: dict[str, Any],
        frame_summaries: list[dict[str, Any]],
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        counts_by_class: dict[str, int] = {cname: 0 for cname in CLASS_NAMES.values()}
        for rec in records:
            cname = rec["damage_class"]
            counts_by_class[cname] = counts_by_class.get(cname, 0) + 1

        confidences = [r["confidence"] for r in records]
        avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        max_conf = round(max(confidences), 4) if confidences else 0.0
        frames_with_defects = sum(1 for fs in frame_summaries if fs["defect_count"] > 0)

        return {
            "producer": producer,
            "conf_threshold": request.conf_threshold,
            "iou_threshold": request.iou_threshold,
            "imgsz": request.imgsz,
            "total_frames_processed": len(frame_summaries),
            "frames_with_defects": frames_with_defects,
            "total_defects_detected": len(records),
            "counts_by_class": counts_by_class,
            "average_confidence": avg_conf,
            "max_confidence": max_conf,
            "frames": frame_summaries,
            "b0_benchmark_mAP50": B0_VALIDATION_METRICS["overall"]["mAP50"],
        }


def run(request: C1Request, context: RunContext) -> C1Result:
    """Trusted inference adapter entry point matching docs/interfaces.md."""
    detector = C1Detector(weights_path=request.weights_path)
    return detector.run(request, context)
