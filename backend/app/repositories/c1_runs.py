"""
PostgreSQL persistence for Component 1 (road damage detection) runs and observation records.

Tables (created on first use with CREATE TABLE IF NOT EXISTS):
    c1_runs          one row per C1 run: survey, source, status, stage, progress, timestamps,
                     model_version, weights_sha256, conf_threshold, summary, warnings, error, cancel_requested
    c1_observations  one row per detected road defect observation (contract record schema 1.0):
                     damage_class, class_id, display_name, confidence, bbox ([x_min,y_min,x_max,y_max]),
                     frame_idx, frame_dimensions, annotated_frame, crop_artifact, crop_transform,
                     and nullable timestamp_ms / location / location_null_reason (ready for metadata phase).

Uploaded images/videos and output image artifacts stay on disk (C1_STORAGE_DIR/<run_id>/inputs|outputs);
their metadata and structured observation records are stored in PostgreSQL.
"""
from __future__ import annotations

import re
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from backend.app.database import connection_params

SCHEMA = """
CREATE TABLE IF NOT EXISTS c1_runs (
    run_id            TEXT PRIMARY KEY,
    survey_id         TEXT NOT NULL,
    source            TEXT NOT NULL CHECK (source IN ('upload', 'demo')),
    status            TEXT NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled', 'rejected')),
    stage             TEXT,
    progress          JSONB,
    model_version     TEXT,
    weights_sha256    TEXT,
    conf_threshold    DOUBLE PRECISION NOT NULL DEFAULT 0.25,
    summary           JSONB,
    warnings          JSONB NOT NULL DEFAULT '[]'::jsonb,
    error             JSONB,
    artifacts         JSONB,
    cancel_requested  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at        TIMESTAMPTZ,
    finished_at       TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS c1_runs_created_idx ON c1_runs (created_at DESC);
CREATE INDEX IF NOT EXISTS c1_runs_survey_idx ON c1_runs (survey_id);

CREATE TABLE IF NOT EXISTS c1_observations (
    id                   BIGSERIAL PRIMARY KEY,
    run_id               TEXT NOT NULL REFERENCES c1_runs (run_id) ON DELETE CASCADE,
    record_id            TEXT NOT NULL UNIQUE,
    defect_id            TEXT NOT NULL,
    survey_id            TEXT NOT NULL,
    source_media         TEXT NOT NULL,
    frame_idx            INTEGER NOT NULL,
    frame_dimensions     JSONB NOT NULL,
    timestamp_ms         BIGINT,
    damage_class         TEXT NOT NULL CHECK (damage_class IN (
                             'Longitudinal_Crack_D00',
                             'Transverse_Crack_D10',
                             'Alligator_Crack_D20',
                             'Pothole_D40'
                         )),
    class_id             INTEGER NOT NULL CHECK (class_id BETWEEN 0 AND 3),
    display_name         TEXT NOT NULL,
    confidence           DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    bbox                 JSONB NOT NULL,
    annotated_frame      TEXT NOT NULL,
    crop_artifact        TEXT,
    crop_transform       JSONB,
    location             JSONB,
    location_null_reason TEXT,
    producer             JSONB NOT NULL,
    source_kind          TEXT NOT NULL CHECK (source_kind IN ('import', 'inference', 'mock')),
    record               JSONB NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS c1_observations_run_idx ON c1_observations (run_id, frame_idx);
CREATE INDEX IF NOT EXISTS c1_observations_class_idx ON c1_observations (damage_class);
"""

RUN_ID = re.compile(r"^[0-9a-f]{32}$")
RUN_COLUMNS = (
    "run_id",
    "survey_id",
    "source",
    "status",
    "stage",
    "progress",
    "model_version",
    "weights_sha256",
    "conf_threshold",
    "summary",
    "warnings",
    "error",
    "artifacts",
    "cancel_requested",
    "created_at",
    "started_at",
    "finished_at",
)
JSON_COLUMNS = {"progress", "summary", "warnings", "error", "artifacts"}
TIME_COLUMNS = {"created_at", "started_at", "finished_at"}


def _iso(v):
    return v.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z") if v else None


def _parse_time(v):
    if v is None or isinstance(v, datetime):
        return v
    return datetime.fromisoformat(str(v))


class PostgresRunStore:
    """Same interface as ml.c1.api.store.RunStore, with run state and observations in PostgreSQL."""

    def __init__(self, storage_dir, connect=None):
        self.root = Path(storage_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._connect = connect or (lambda: psycopg.connect(**connection_params(), row_factory=dict_row))
        self._schema_ready = False
        self._lock = threading.Lock()

    @contextmanager
    def _conn(self):
        if not self._schema_ready:
            with self._lock:
                if not self._schema_ready:
                    with self._connect() as c:
                        c.execute(SCHEMA)
                    self._schema_ready = True
        with self._connect() as c:
            yield c

    @staticmethod
    def _row(r):
        return {k: (_iso(r[k]) if k in TIME_COLUMNS else r[k]) for k in RUN_COLUMNS}

    # --------------------------------------------------------------- RunStore interface
    def new(self, survey_id: str, source: str, conf_threshold: float = 0.25) -> dict:
        run_id = uuid.uuid4().hex
        d = self.root / run_id
        (d / "inputs").mkdir(parents=True, exist_ok=True)
        (d / "outputs" / "annotated").mkdir(parents=True, exist_ok=True)
        (d / "outputs" / "crops").mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            r = c.execute(
                "INSERT INTO c1_runs (run_id, survey_id, source, status, conf_threshold) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING *",
                (run_id, survey_id, source, "queued", conf_threshold),
            ).fetchone()
        return self._row(r)

    def dir(self, run_id: str) -> Path:
        self.get(run_id)  # validates run_id and verifies the run exists in PostgreSQL
        d = self.root / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get(self, run_id: str) -> dict:
        if not RUN_ID.match(run_id or ""):
            raise KeyError(run_id)
        with self._conn() as c:
            r = c.execute("SELECT * FROM c1_runs WHERE run_id = %s", (run_id,)).fetchone()
        if r is None:
            raise KeyError(run_id)
        return self._row(r)

    def update(self, run_id: str, **fields) -> dict:
        return self._update(None, run_id, fields)

    def list(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM c1_runs ORDER BY created_at DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def complete(self, run_id: str, records: list[dict], **fields) -> dict:
        """Insert all observation records and mark the run finished in one transaction."""
        with self._conn() as c, c.transaction():
            c.execute("DELETE FROM c1_observations WHERE run_id = %s", (run_id,))
            if records:
                with c.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO c1_observations ("
                        "run_id, record_id, defect_id, survey_id, source_media, frame_idx, frame_dimensions, "
                        "timestamp_ms, damage_class, class_id, display_name, confidence, bbox, "
                        "annotated_frame, crop_artifact, crop_transform, location, location_null_reason, "
                        "producer, source_kind, record"
                        ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        [
                            (
                                run_id,
                                r["record_id"],
                                r["defect_id"],
                                r["survey_id"],
                                r["source_media"],
                                r["frame_idx"],
                                Jsonb(r["frame_dimensions"]),
                                r.get("timestamp_ms"),
                                r["damage_class"],
                                r["class_id"],
                                r["display_name"],
                                r["confidence"],
                                Jsonb(r["bbox"]),
                                r["annotated_frame"],
                                r.get("crop_artifact"),
                                Jsonb(r["crop_transform"]) if r.get("crop_transform") else None,
                                Jsonb(r["location"]) if r.get("location") else None,
                                r.get("location_null_reason"),
                                Jsonb(r["producer"]),
                                r["source_kind"],
                                Jsonb(r),
                            )
                            for r in records
                        ],
                    )
            return self._update(c, run_id, fields)

    def records(self, run_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT record FROM c1_observations WHERE run_id = %s ORDER BY frame_idx, id",
                (run_id,),
            ).fetchall()
        return [r["record"] for r in rows]

    # --------------------------------------------------------------- helpers
    def _update(self, conn, run_id: str, fields: dict) -> dict:
        fields = {k: v for k, v in fields.items() if k in RUN_COLUMNS and k != "run_id"}
        if not fields:
            return self.get(run_id)
        sets, vals = [], []
        for k, v in fields.items():
            sets.append(f"{k} = %s")
            vals.append(
                Jsonb(v)
                if k in JSON_COLUMNS and v is not None
                else _parse_time(v)
                if k in TIME_COLUMNS
                else v
            )
        sql = f"UPDATE c1_runs SET {', '.join(sets)} WHERE run_id = %s RETURNING *"
        if conn is None:
            with self._conn() as c:
                r = c.execute(sql, (*vals, run_id)).fetchone()
        else:
            r = conn.execute(sql, (*vals, run_id)).fetchone()
        if r is None:
            raise KeyError(run_id)
        return self._row(r)
