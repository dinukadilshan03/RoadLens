"""
PostgreSQL persistence for Component 4 (road roughness) runs and their segment results.

Tables (created on first use with CREATE TABLE IF NOT EXISTS; move to the project's migration tool
when one is added):

    c4_runs      one row per C4 run: survey, source (upload/demo), status, stage, progress, timestamps,
                 model kind/version, summary, warnings, error, cancel flag
    c4_segments  one row per road segment result (contract record schema 1.0): interval, roughness class,
                 confidence, class probabilities, quality, sensor, provenance and GeoJSON LineString

Uploaded CSV files and output artifacts stay on disk (C4_STORAGE_DIR/<run_id>/inputs|outputs);
only their relative names are stored here. Geometry is stored as GeoJSON (JSONB) because the current
Compose database is plain PostgreSQL; switch to geometry(LineString, 4326) when PostGIS is enabled.
"""
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
CREATE TABLE IF NOT EXISTS c4_runs (
    run_id            TEXT PRIMARY KEY,
    survey_id         TEXT NOT NULL,
    source            TEXT NOT NULL CHECK (source IN ('upload', 'demo')),
    status            TEXT NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled', 'rejected')),
    stage             TEXT,
    progress          JSONB,
    model_kind        TEXT,
    model_version     TEXT,
    summary           JSONB,
    warnings          JSONB NOT NULL DEFAULT '[]'::jsonb,
    error             JSONB,
    artifacts         JSONB,
    cancel_requested  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at        TIMESTAMPTZ,
    finished_at       TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS c4_runs_created_idx ON c4_runs (created_at DESC);
CREATE INDEX IF NOT EXISTS c4_runs_survey_idx ON c4_runs (survey_id);

CREATE TABLE IF NOT EXISTS c4_segments (
    id                   BIGSERIAL PRIMARY KEY,
    run_id               TEXT NOT NULL REFERENCES c4_runs (run_id) ON DELETE CASCADE,
    record_id            TEXT NOT NULL UNIQUE,
    survey_id            TEXT NOT NULL,
    interval_start_ms    BIGINT NOT NULL,
    interval_end_ms      BIGINT NOT NULL CHECK (interval_end_ms > interval_start_ms),
    roughness_class      TEXT NOT NULL CHECK (roughness_class IN ('good', 'regular', 'bad')),
    confidence           DOUBLE PRECISION CHECK (confidence BETWEEN 0 AND 1),
    class_probabilities  JSONB,
    quality              JSONB,
    sensor               JSONB,
    producer             JSONB NOT NULL,
    source_kind          TEXT NOT NULL CHECK (source_kind IN ('import', 'inference', 'mock')),
    geometry             JSONB,
    geometry_null_reason TEXT,
    record               JSONB NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS c4_segments_run_idx ON c4_segments (run_id, interval_start_ms);
CREATE INDEX IF NOT EXISTS c4_segments_class_idx ON c4_segments (roughness_class);
"""

RUN_ID = re.compile(r'^[0-9a-f]{32}$')
RUN_COLUMNS = ('run_id', 'survey_id', 'source', 'status', 'stage', 'progress', 'model_kind', 'model_version',
               'summary', 'warnings', 'error', 'artifacts', 'cancel_requested', 'created_at', 'started_at',
               'finished_at')
JSON_COLUMNS = {'progress', 'summary', 'warnings', 'error', 'artifacts'}
TIME_COLUMNS = {'created_at', 'started_at', 'finished_at'}


def _iso(v):
    return v.astimezone(UTC).isoformat(timespec='seconds').replace('+00:00', 'Z') if v else None


def _parse_time(v):
    if v is None or isinstance(v, datetime):
        return v
    return datetime.fromisoformat(str(v))


class PostgresRunStore:
    """Same interface as ml.c4.api.store.RunStore, with run state and results in PostgreSQL."""

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
    def new(self, survey_id, source):
        run_id = uuid.uuid4().hex
        d = self.root / run_id
        (d / 'inputs').mkdir(parents=True)
        (d / 'outputs').mkdir()
        with self._conn() as c:
            r = c.execute('INSERT INTO c4_runs (run_id, survey_id, source, status) VALUES (%s, %s, %s, %s) '
                          'RETURNING *', (run_id, survey_id, source, 'queued')).fetchone()
        return self._row(r)

    def dir(self, run_id):
        self.get(run_id)                                    # validates the id and that the run exists
        return self.root / run_id

    def get(self, run_id):
        if not RUN_ID.match(run_id or ''):
            raise KeyError(run_id)
        with self._conn() as c:
            r = c.execute('SELECT * FROM c4_runs WHERE run_id = %s', (run_id,)).fetchone()
        if r is None:
            raise KeyError(run_id)
        return self._row(r)

    def update(self, run_id, **fields):
        return self._update(None, run_id, fields)

    def list(self, limit=50):
        with self._conn() as c:
            rows = c.execute('SELECT * FROM c4_runs ORDER BY created_at DESC LIMIT %s', (limit,)).fetchall()
        return [self._row(r) for r in rows]

    def complete(self, run_id, records, **fields):
        """Insert all segment records and mark the run finished in one transaction."""
        with self._conn() as c, c.transaction():
            c.execute('DELETE FROM c4_segments WHERE run_id = %s', (run_id,))
            with c.cursor() as cur:
                cur.executemany(
                    'INSERT INTO c4_segments (run_id, record_id, survey_id, interval_start_ms, interval_end_ms, '
                    'roughness_class, confidence, class_probabilities, quality, sensor, producer, source_kind, '
                    'geometry, geometry_null_reason, record) '
                    'VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)',
                    [(run_id, r['record_id'], r['survey_id'], r['interval_start_ms'], r['interval_end_ms'],
                      r['roughness_class'], r.get('confidence'), Jsonb(r.get('class_probabilities')),
                      Jsonb(r.get('quality')), Jsonb(r.get('sensor')), Jsonb(r['producer']), r['source_kind'],
                      Jsonb(r['geometry']) if r.get('geometry') else None, r.get('geometry_null_reason'),
                      Jsonb(r)) for r in records])
            return self._update(c, run_id, fields)

    def segments_geojson(self, run_id):
        with self._conn() as c:
            rows = c.execute('SELECT record_id, interval_start_ms, interval_end_ms, roughness_class, confidence, '
                             'geometry FROM c4_segments WHERE run_id = %s ORDER BY interval_start_ms',
                             (run_id,)).fetchall()
        return {'type': 'FeatureCollection', 'features': [
            {'type': 'Feature', 'geometry': r['geometry'],
             'properties': {k: r[k] for k in ('record_id', 'interval_start_ms', 'interval_end_ms',
                                              'roughness_class', 'confidence')}} for r in rows]}

    def records(self, run_id):
        with self._conn() as c:
            rows = c.execute('SELECT record FROM c4_segments WHERE run_id = %s ORDER BY interval_start_ms',
                             (run_id,)).fetchall()
        return [r['record'] for r in rows]

    # --------------------------------------------------------------- helpers
    def _update(self, conn, run_id, fields):
        fields = {k: v for k, v in fields.items() if k in RUN_COLUMNS and k != 'run_id'}
        if not fields:
            return self.get(run_id)
        sets, vals = [], []
        for k, v in fields.items():
            sets.append(f'{k} = %s')
            vals.append(Jsonb(v) if k in JSON_COLUMNS and v is not None else _parse_time(v) if k in TIME_COLUMNS else v)
        sql = f'UPDATE c4_runs SET {", ".join(sets)} WHERE run_id = %s RETURNING *'   # keys are whitelisted
        if conn is None:
            with self._conn() as c:
                r = c.execute(sql, (*vals, run_id)).fetchone()
        else:
            r = conn.execute(sql, (*vals, run_id)).fetchone()
        if r is None:
            raise KeyError(run_id)
        return self._row(r)

