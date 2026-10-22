"""File-based run store for C2 API runs (offline fallback and unit tests).

Layout:
    <run_id>/run.json                  metadata, stage, progress, alpha_score, latency, vram, summary
    <run_id>/inputs/crop.jpg           ingested C1 crop or uploaded crop
    <run_id>/outputs/mask_2d.png       Output A: 2D Binary Mask
    <run_id>/outputs/skeleton_1px.png   Output B: 1-Pixel Skeleton Centerline
    <run_id>/outputs/confidence.npz    Output C: Soft Confidence Map
    <run_id>/outputs/visual_overlay.png Output D: Red Visual Overlay
    <run_id>/outputs/pseudo_thermal.png Synthesized Pseudo-Thermal Map
    <run_id>/outputs/c2_result.json    Handoff contract record for C3 and C4
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

RUN_ID = re.compile(r"^[0-9a-f]{32}$")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class RunStore:
    def __init__(self, root: str | Path, c1_storage_dir: str | Path | None = None):
        candidate_root = Path(root)
        fallback_root = Path(__file__).resolve().parents[1] / "runs"
        try:
            candidate_root.mkdir(parents=True, exist_ok=True)
            test_dir = candidate_root / ".write_check"
            test_dir.mkdir(exist_ok=True)
            test_dir.rmdir()
            self.root = candidate_root
        except OSError:
            fallback_root.mkdir(parents=True, exist_ok=True)
            self.root = fallback_root
        default_c1 = os.environ.get("C1_STORAGE_DIR") or (Path(__file__).resolve().parents[2] / "c1" / "runs")
        self.c1_root = Path(c1_storage_dir or default_c1)
        self._lock = threading.Lock()

    def new(
        self,
        survey_id: str,
        source: str,
        damage_class: str = "crack",
        c1_run_id: str | None = None,
        c1_observation_id: str | None = None,
    ) -> dict:
        run_id = uuid.uuid4().hex
        d = self.root / run_id
        (d / "inputs").mkdir(parents=True, exist_ok=True)
        (d / "outputs").mkdir(parents=True, exist_ok=True)

        meta = {
            "run_id": run_id,
            "survey_id": survey_id,
            "source": source,
            "c1_run_id": c1_run_id,
            "c1_observation_id": c1_observation_id,
            "damage_class": damage_class,
            "status": "queued",
            "stage": None,
            "progress": None,
            "model_version": None,
            "weights_sha256": None,
            "alpha_score": None,
            "latency_ms": None,
            "vram_mb": None,
            "summary": None,
            "warnings": [],
            "error": None,
            "artifacts": None,
            "cancel_requested": False,
            "created_at": now(),
            "started_at": None,
            "finished_at": None,
        }
        self._write(run_id, meta)
        return meta

    def dir(self, run_id: str) -> Path:
        if not RUN_ID.match(run_id or ""):
            raise KeyError(run_id)
        d = self.root / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get(self, run_id: str) -> dict:
        p = self.dir(run_id) / "run.json"
        if not p.exists():
            raise KeyError(run_id)
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def update(self, run_id: str, **fields) -> dict:
        with self._lock:
            meta = self.get(run_id)
            meta.update(fields)
            self._write(run_id, meta)
            return meta

    def list(self, limit: int = 50) -> list[dict]:
        runs = []
        if not self.root.exists():
            return runs
        for d in self.root.iterdir():
            if d.is_dir() and RUN_ID.match(d.name) and (d / "run.json").exists():
                try:
                    runs.append(self.get(d.name))
                except (OSError, json.JSONDecodeError):
                    continue
        runs.sort(key=lambda m: m.get("created_at") or "", reverse=True)
        return runs[:limit]

    def complete(self, run_id: str, results: list[dict], **fields) -> dict:
        p = self.dir(run_id) / "outputs" / "c2_results.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return self.update(run_id, **fields)

    def get_results(self, run_id: str) -> list[dict]:
        p = self.dir(run_id) / "outputs" / "c2_results.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return []

    def get_record(self, run_id: str) -> dict | None:
        p = self.dir(run_id) / "outputs" / "c2_results.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
                elif isinstance(data, dict):
                    return data
        meta = self.get(run_id)
        results = meta.get("results")
        if results and len(results) > 0:
            return results[0]
        return None

    @staticmethod
    def _enrich_c1_crop(row: dict) -> dict:
        item = dict(row)
        r_id = item.get("run_id") or item.get("external_run_id")
        item["run_id"] = r_id
        crop_art = item.get("crop_artifact")
        if crop_art and r_id:
            if str(crop_art).startswith("http") or str(crop_art).startswith("/api/v1"):
                item["crop_url"] = str(crop_art)
            else:
                item["crop_url"] = f"/api/v1/c1/runs/{r_id}/artifacts/crops/{crop_art}"
        ann_frame = item.get("annotated_frame")
        if ann_frame and r_id:
            if str(ann_frame).startswith("http") or str(ann_frame).startswith("/api/v1"):
                item["annotated_url"] = str(ann_frame)
            else:
                item["annotated_url"] = f"/api/v1/c1/runs/{r_id}/artifacts/annotated/{ann_frame}"
        return item

    def get_available_c1_crops(self, limit: int = 50, c1_run_id: str | None = None) -> list[dict]:
        if not self.c1_root.exists():
            return []

        c2_by_obs: dict[str, str] = {}
        for r in self.list(limit=200):
            if r.get("status") == "succeeded" and r.get("c1_observation_id"):
                obs_id = r["c1_observation_id"]
                if obs_id not in c2_by_obs:
                    c2_by_obs[obs_id] = r["run_id"]

        run_dirs = []
        if c1_run_id:
            candidate = self.c1_root / c1_run_id
            if candidate.is_dir():
                run_dirs.append(candidate)
        else:
            for d in self.c1_root.iterdir():
                if d.is_dir() and RUN_ID.match(d.name):
                    run_dirs.append(d)
            run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        items: list[dict] = []
        for d in run_dirs:
            jsonl_path = d / "outputs" / "c1_records.jsonl"
            run_json_path = d / "run.json"
            created_at = None
            if run_json_path.is_file():
                try:
                    with open(run_json_path, encoding="utf-8") as rf:
                        created_at = json.load(rf).get("created_at")
                except Exception:
                    pass
            if not jsonl_path.is_file():
                continue
            try:
                with open(jsonl_path, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        if not rec.get("crop_artifact"):
                            continue
                        rec["run_id"] = rec.get("external_run_id") or d.name
                        rec["created_at"] = rec.get("created_at") or created_at or now()
                        rec["latest_c2_run_id"] = c2_by_obs.get(rec.get("record_id", ""))
                        items.append(self._enrich_c1_crop(rec))
                        if len(items) >= limit:
                            return items
            except Exception:
                continue
        return items

    def get_c1_observation(self, c1_run_id: str, c1_observation_id: str) -> dict | None:
        crops = self.get_available_c1_crops(limit=500, c1_run_id=c1_run_id)
        for c in crops:
            if c.get("record_id") == c1_observation_id:
                return c
        return None

    def _write(self, run_id: str, meta: dict) -> None:
        p = self.dir(run_id) / "run.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
