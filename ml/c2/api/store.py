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
import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

RUN_ID = re.compile(r"^[0-9a-f]{32}$")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class RunStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
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

    def get_available_c1_crops(self, limit: int = 20) -> list[dict]:
        return []

    def _write(self, run_id: str, meta: dict) -> None:
        p = self.dir(run_id) / "run.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
