"""C1 Base Model (B0) Detection Adapter for RoadLens.

Implements the 50% Progress Milestone capabilities:
1. Video frame extraction (1-2 FPS sampling) or image sequence loading.
2. B0 Baseline YOLOv8m inference on extracted frames.
3. Bounding box localization, class prediction (D00, D10, D20, D40), and confidence scoring.
4. Defect ROI extraction (cropping regions ready for downstream C2 Segmentation).
5. GPS timestamp binding and RoadLens v1 format artifact generation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("roadlens.c1.detector")

CLASS_NAMES: Dict[int, str] = {
    0: "Longitudinal_Crack_D00",
    1: "Transverse_Crack_D10",
    2: "Alligator_Crack_D20",
    3: "Pothole_D40",
}

CLASS_DISPLAY: Dict[str, str] = {
    "Longitudinal_Crack_D00": "Longitudinal Crack (D00)",
    "Transverse_Crack_D10": "Transverse Crack (D10)",
    "Alligator_Crack_D20": "Alligator Crack (D20)",
    "Pothole_D40": "Pothole (D40)",
}

CLASS_HEX_COLORS: Dict[str, str] = {
    "Longitudinal_Crack_D00": "#2563eb",  # Blue
    "Transverse_Crack_D10": "#059669",    # Green
    "Alligator_Crack_D20": "#d97706",     # Amber
    "Pothole_D40": "#dc2626",             # Red
}

B0_VALIDATION_METRICS = {
    "overall": {
        "mAP50": 0.662,
        "mAP50_95": 0.371,
        "precision": 0.665,
        "recall": 0.620,
    },
    "per_class": {
        "Longitudinal_Crack_D00": {"ap50": 0.603, "precision": 0.645, "recall": 0.567},
        "Transverse_Crack_D10": {"ap50": 0.587, "precision": 0.620, "recall": 0.552},
        "Alligator_Crack_D20": {"ap50": 0.672, "precision": 0.690, "recall": 0.610},
        "Pothole_D40": {"ap50": 0.784, "precision": 0.704, "recall": 0.750},
    },
}


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class C1Detector:
    """YOLOv8m detector adapter executing the B0 baseline model."""

    def __init__(self, weights_path: Path):
        self.weights_path = Path(weights_path)
        self.model = None
        self._sha256 = None
        self._init_model()

    def _init_model(self) -> None:
        if not self.weights_path.is_file():
            logger.warning(f"Weights file not found at {self.weights_path}. Inference will fallback.")
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.weights_path))
            self._sha256 = compute_file_sha256(self.weights_path)
            logger.info(f"Loaded YOLOv8m B0 model from {self.weights_path} (SHA: {self._sha256[:12]}...)")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def sha256(self) -> str:
        if self._sha256 is None and self.weights_path.is_file():
            self._sha256 = compute_file_sha256(self.weights_path)
        return self._sha256 or "unavailable"

    def run_detection(
        self,
        input_files: List[Path],
        output_dir: Path,
        gps_csv_path: Optional[Path] = None,
        progress_cb: Optional[Callable[[str, Optional[int], Optional[int]], None]] = None,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.50,
    ) -> Dict[str, Any]:
        """Runs the full frame extraction, B0 inference, and artifact generation."""
        output_dir = Path(output_dir)
        frames_dir = output_dir / "frames"
        crops_dir = output_dir / "crops"
        frames_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)

        # 1. Parse GPS trace if available
        gps_points = self._parse_gps_csv(gps_csv_path) if gps_csv_path else []

        # 2. Extract / collect frames
        if progress_cb:
            progress_cb("extract_frames", 0, len(input_files))

        raw_frames = self._extract_frames(input_files, frames_dir)
        total_frames = len(raw_frames)

        if total_frames == 0:
            raise ValueError("No valid image frames found or extracted from input.")

        # 3. Run YOLO detection
        if progress_cb:
            progress_cb("predict_b0", 0, total_frames)

        observations: List[Dict[str, Any]] = []
        annotated_frame_names: List[str] = []

        for idx, frame_info in enumerate(raw_frames):
            frame_path = frame_info["path"]
            timestamp_ms = frame_info["timestamp_ms"]
            coords = self._interpolate_gps(gps_points, timestamp_ms, idx, total_frames)

            preds = self._predict_single_frame(frame_path, conf_thresh, iou_thresh)

            # Draw annotations & crop defect ROIs
            annotated_name = f"annotated_frame_{idx:04d}.jpg"
            annotated_path = frames_dir / annotated_name
            frame_obs = self._process_frame_annotations(
                frame_path=frame_path,
                annotated_save_path=annotated_path,
                crops_dir=crops_dir,
                preds=preds,
                frame_idx=idx,
                timestamp_ms=timestamp_ms,
                coords=coords,
            )
            observations.extend(frame_obs)
            annotated_frame_names.append(annotated_name)

            if progress_cb and (idx % 2 == 0 or idx == total_frames - 1):
                progress_cb("predict_b0", idx + 1, total_frames)

        # 4. Generate summary and contracts
        if progress_cb:
            progress_cb("export_contracts", total_frames, total_frames)

        summary = self._build_summary(raw_frames, observations)
        geojson = self._build_geojson(observations)
        records_jsonl = self._build_records_jsonl(observations)

        # Save artifacts
        (output_dir / "c1_defects.geojson").write_text(json.dumps(geojson, indent=2), encoding="utf-8")
        (output_dir / "c1_records.jsonl").write_text("\n".join(json.dumps(r) for r in records_jsonl), encoding="utf-8")
        (output_dir / "c1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return {
            "summary": summary,
            "observations_count": len(observations),
            "frames_count": total_frames,
            "geojson": geojson,
        }

    def _extract_frames(self, input_files: List[Path], target_dir: Path) -> List[Dict[str, Any]]:
        """Extracts frames from video files or copies image files."""
        frames: List[Dict[str, Any]] = []
        frame_idx = 0

        for file_path in input_files:
            ext = file_path.suffix.lower()
            if ext in [".jpg", ".jpeg", ".png", ".webp"]:
                dest = target_dir / f"frame_{frame_idx:04d}.jpg"
                shutil.copy2(file_path, dest)
                frames.append({
                    "path": dest,
                    "frame_idx": frame_idx,
                    "timestamp_ms": frame_idx * 1000,
                    "source_file": file_path.name,
                })
                frame_idx += 1
            elif ext in [".mp4", ".mov", ".avi", ".mkv"]:
                # Video file extraction using OpenCV
                try:
                    import cv2
                    cap = cv2.VideoCapture(str(file_path))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    sample_interval = max(1, int(fps / 2))  # Extract ~2 frames per second
                    vid_frame_cnt = 0

                    while cap.isOpened():
                        ret, cv_frame = cap.read()
                        if not ret:
                            break
                        if vid_frame_cnt % sample_interval == 0:
                            dest = target_dir / f"frame_{frame_idx:04d}.jpg"
                            cv2.imwrite(str(dest), cv_frame)
                            time_ms = int((vid_frame_cnt / fps) * 1000)
                            frames.append({
                                "path": dest,
                                "frame_idx": frame_idx,
                                "timestamp_ms": time_ms,
                                "source_file": file_path.name,
                            })
                            frame_idx += 1
                        vid_frame_cnt += 1
                    cap.release()
                except Exception as e:
                    logger.error(f"Failed to extract frames from video {file_path}: {e}")

        return frames

    def _predict_single_frame(
        self,
        frame_path: Path,
        conf_thresh: float,
        iou_thresh: float,
    ) -> List[Dict[str, Any]]:
        """Runs YOLO on a single frame or falls back to synthetic heuristic."""
        if not self.model:
            # Fallback mock prediction if weights aren't loaded in test
            return []

        results = self.model(
            source=str(frame_path),
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=800,
            verbose=False,
        )

        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = [round(float(c), 1) for c in boxes.xyxy[i].tolist()]
                class_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")

                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox_xyxy": xyxy,
                })
        return detections

    def _process_frame_annotations(
        self,
        frame_path: Path,
        annotated_save_path: Path,
        crops_dir: Path,
        preds: List[Dict[str, Any]],
        frame_idx: int,
        timestamp_ms: int,
        coords: Optional[Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """Draws bounding boxes and extracts crops for downstream C2 segmentation."""
        from PIL import Image, ImageDraw, ImageFont

        frame_obs: List[Dict[str, Any]] = []

        try:
            img = Image.open(frame_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            w, h = img.size
        except Exception as e:
            logger.error(f"Error opening image {frame_path}: {e}")
            return []

        for p_idx, pred in enumerate(preds):
            defect_id = f"def_{uuid.uuid4().hex[:8]}"
            xyxy = pred["bbox_xyxy"]
            x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(w, xyxy[2]), min(h, xyxy[3])
            cname = pred["class_name"]
            conf = pred["confidence"]
            color = CLASS_HEX_COLORS.get(cname, "#ffffff")

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = f"{CLASS_DISPLAY.get(cname, cname)} {conf:.2f}"
            draw.rectangle([x1, max(0, y1 - 18), x1 + len(label) * 8 + 4, y1], fill=color)
            draw.text((x1 + 3, max(0, y1 - 16)), label, fill="#ffffff")

            # Extract and save C2 defect ROI crop (FR15)
            crop_name = f"{defect_id}_crop.jpg"
            crop_path = crops_dir / crop_name
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                crop = img.crop((x1, y1, x2, y2))
                crop.save(crop_path, quality=92)
            else:
                crop_name = ""

            frame_obs.append({
                "defect_id": defect_id,
                "frame_idx": frame_idx,
                "timestamp_ms": timestamp_ms,
                "damage_class": cname,
                "display_name": CLASS_DISPLAY.get(cname, cname),
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "location": coords,
                "annotated_frame": annotated_save_path.name,
                "crop_image": crop_name,
            })

        img.save(annotated_save_path, quality=90)
        return frame_obs

    def _parse_gps_csv(self, gps_path: Optional[Path]) -> List[Dict[str, Any]]:
        """Parses GPS CSV: timestamp_ms,latitude,longitude,accuracy_m."""
        points = []
        if not gps_path or not gps_path.is_file():
            return points

        with open(gps_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    points.append({
                        "timestamp_ms": int(float(row.get("timestamp_ms", 0))),
                        "lat": float(row["latitude"]),
                        "lng": float(row["longitude"]),
                        "accuracy_m": float(row.get("accuracy_m", 5.0)),
                    })
                except (KeyError, ValueError):
                    continue
        points.sort(key=lambda p: p["timestamp_ms"])
        return points

    def _interpolate_gps(
        self,
        gps_points: List[Dict[str, Any]],
        timestamp_ms: int,
        frame_idx: int,
        total_frames: int,
    ) -> Optional[Dict[str, float]]:
        """Matches or interpolates GPS coordinates for a frame."""
        if not gps_points:
            return None

        # Exact/closest timestamp match
        if len(gps_points) == 1:
            return {"latitude": gps_points[0]["lat"], "longitude": gps_points[0]["lng"]}

        # Check if timestamps overlap
        if timestamp_ms <= gps_points[0]["timestamp_ms"]:
            return {"latitude": gps_points[0]["lat"], "longitude": gps_points[0]["lng"]}
        if timestamp_ms >= gps_points[-1]["timestamp_ms"]:
            return {"latitude": gps_points[-1]["lat"], "longitude": gps_points[-1]["lng"]}

        # Interpolate between consecutive points
        for i in range(len(gps_points) - 1):
            p1, p2 = gps_points[i], gps_points[i + 1]
            if p1["timestamp_ms"] <= timestamp_ms <= p2["timestamp_ms"]:
                span = p2["timestamp_ms"] - p1["timestamp_ms"]
                if span == 0:
                    return {"latitude": p1["lat"], "longitude": p1["lng"]}
                alpha = (timestamp_ms - p1["timestamp_ms"]) / span
                return {
                    "latitude": round(p1["lat"] + alpha * (p2["lat"] - p1["lat"]), 6),
                    "longitude": round(p1["lng"] + alpha * (p2["lng"] - p1["lng"]), 6),
                }

        # Fallback ratio index
        idx = min(int((frame_idx / max(1, total_frames - 1)) * len(gps_points)), len(gps_points) - 1)
        return {"latitude": gps_points[idx]["lat"], "longitude": gps_points[idx]["lng"]}

    def _build_summary(self, frames: List[Dict[str, Any]], obs: List[Dict[str, Any]]) -> Dict[str, Any]:
        counts: Dict[str, int] = {k: 0 for k in CLASS_NAMES.values()}
        for o in obs:
            c = o["damage_class"]
            counts[c] = counts.get(c, 0) + 1

        avg_conf = (sum(o["confidence"] for o in obs) / len(obs)) if obs else 0.0

        return {
            "model_version": "YOLOv8m-RDD2022-B0",
            "weights_sha256": self.sha256[:16],
            "total_frames_processed": len(frames),
            "total_defects_detected": len(obs),
            "counts_by_class": counts,
            "average_confidence": round(avg_conf, 3),
            "b0_benchmark_mAP50": B0_VALIDATION_METRICS["overall"]["mAP50"],
        }

    def _build_geojson(self, obs: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = []
        for o in obs:
            loc = o.get("location")
            if not loc:
                continue
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [loc["longitude"], loc["latitude"]],
                },
                "properties": {
                    "defect_id": o["defect_id"],
                    "damage_class": o["damage_class"],
                    "display_name": o["display_name"],
                    "confidence": o["confidence"],
                    "timestamp_ms": o["timestamp_ms"],
                    "frame_idx": o["frame_idx"],
                    "bbox": o["bbox"],
                    "crop_image": o["crop_image"],
                    "annotated_frame": o["annotated_frame"],
                },
            })
        return {
            "type": "FeatureCollection",
            "features": features,
        }

    def _build_records_jsonl(self, obs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        records = []
        for o in obs:
            records.append({
                "schema_version": "1.0",
                "component": "C1",
                "record_id": o["defect_id"],
                "damage_class": o["damage_class"],
                "confidence": o["confidence"],
                "timestamp_ms": o["timestamp_ms"],
                "bbox": o["bbox"],
                "location": o.get("location"),
                "crop_artifact": o.get("crop_image"),
                "upstream": None,
            })
        return records
