"""File-based run store for C1 API runs.

Directory Layout:
    ml/c1/runs/<run_id>/
        run.json              # status, stage, timestamps, survey, model, summary, warnings, error
        inputs/               # uploaded video / images, optional gps.csv
        outputs/              # c1_defects.geojson, c1_records.jsonl, c1_summary.json
            frames/           # annotated frames with bounding box overlays
            crops/            # cropped defect ROIs ready for C2
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class C1RunStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def new(self, survey_id: str, source: str = "upload") -> Dict[str, Any]:
        run_id = uuid.uuid4().hex
        d = self.root / run_id
        (d / "inputs").mkdir(parents=True, exist_ok=True)
        (d / "outputs").mkdir(parents=True, exist_ok=True)
        (d / "outputs" / "frames").mkdir(parents=True, exist_ok=True)
        (d / "outputs" / "crops").mkdir(parents=True, exist_ok=True)

        meta = {
            "run_id": run_id,
            "survey_id": survey_id,
            "source": source,
            "status": "queued",
            "stage": None,
            "progress": {"completed": 0, "total": 0},
            "created_at": now(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "warnings": [],
            "summary": None,
        }
        self._write(run_id, meta)
        return meta

    def dir(self, run_id: str) -> Path:
        if not RUN_ID_PATTERN.match(run_id or ""):
            raise KeyError(run_id)
        d = self.root / run_id
        if not (d / "run.json").exists():
            raise KeyError(run_id)
        return d

    def get(self, run_id: str) -> Dict[str, Any]:
        with open(self.dir(run_id) / "run.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def update(self, run_id: str, **fields: Any) -> Dict[str, Any]:
        with self._lock:
            meta = self.get(run_id)
            meta.update(fields)
            self._write(run_id, meta)
            return meta

    def list(self, limit: int = 50) -> List[Dict[str, Any]]:
        runs = []
        if not self.root.exists():
            return []
        for d in self.root.iterdir():
            if RUN_ID_PATTERN.match(d.name) and (d / "run.json").exists():
                try:
                    runs.append(self.get(d.name))
                except (OSError, json.JSONDecodeError):
                    continue
        runs.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        return runs[:limit]

    def defects_geojson(self, run_id: str) -> Dict[str, Any]:
        path = self.dir(run_id) / "outputs" / "c1_defects.geojson"
        if not path.exists():
            return {"type": "FeatureCollection", "features": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def records(self, run_id: str) -> List[Dict[str, Any]]:
        path = self.dir(run_id) / "outputs" / "c1_records.jsonl"
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def artifact_path(self, run_id: str, kind: str, filename: str) -> Optional[Path]:
        """Resolves path to an artifact: kind in ['frames', 'crops', 'inputs']."""
        d = self.dir(run_id)
        if kind == "frames":
            p = d / "outputs" / "frames" / filename
        elif kind == "crops":
            p = d / "outputs" / "crops" / filename
        elif kind == "inputs":
            p = d / "inputs" / filename
        else:
            return None
        return p if p.is_file() else None

    def _write(self, run_id: str, meta: Dict[str, Any]) -> None:
        d = self.root / run_id
        tmp = d / f"run.json.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, allow_nan=False)
        os.replace(tmp, d / "run.json")
