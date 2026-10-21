"""
File-based run store for C1 API runs (used in unit tests or when PostgreSQL is not configured).

Layout (one folder per run, under C1_STORAGE_DIR):
    <run_id>/run.json                  status, stage, timestamps, survey, model, summary, warnings, error
    <run_id>/inputs/                   uploaded road images or video files
    <run_id>/outputs/annotated/        annotated frames with bounding boxes + confidence badges
    <run_id>/outputs/crops/            cropped defect ROIs for downstream C2 Crack Segmentation
    <run_id>/outputs/c1_records.jsonl  contract observation records (schema 1.0)
    <run_id>/outputs/c1_run_summary.json
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
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def new(self, survey_id: str, source: str, conf_threshold: float = 0.25) -> dict:
        run_id = uuid.uuid4().hex
        d = self.root / run_id
        (d / "inputs").mkdir(parents=True)
        (d / "outputs" / "annotated").mkdir(parents=True)
        (d / "outputs" / "crops").mkdir(parents=True)
        meta = {
            "run_id": run_id,
            "survey_id": survey_id,
            "source": source,
            "status": "queued",
            "stage": None,
            "progress": None,
            "model_version": None,
            "weights_sha256": None,
            "conf_threshold": conf_threshold,
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
        if not (d / "run.json").exists():
            raise KeyError(run_id)
        return d

    def get(self, run_id: str) -> dict:
        with open(self.dir(run_id) / "run.json", encoding="utf-8") as f:
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

    def complete(self, run_id: str, records: list[dict], **fields) -> dict:
        """Mark a run finished; records are already written to outputs/c1_records.jsonl."""
        return self.update(run_id, **fields)

    def records(self, run_id: str) -> list[dict]:
        p = self.dir(run_id) / "outputs" / "c1_records.jsonl"
        if not p.is_file():
            return []
        with open(p, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _write(self, run_id: str, meta: dict) -> None:
        d = self.root / run_id
        tmp = d / f"run.json.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, allow_nan=False)
        os.replace(tmp, d / "run.json")
