"""
File-based run store for C4 API runs (first milestone; no shared database tables yet).

Layout (one folder per run, under C4_STORAGE_DIR):
    <run_id>/run.json       status, stage, timestamps, survey, model, summary, warnings, error
    <run_id>/inputs/        uploaded imu.csv / acc.csv / gyro.csv, gps.csv, manifest.json
    <run_id>/outputs/       adapter artifacts (c4_segments.jsonl, .geojson, c4_windows.csv, c4_run_summary.json)

When the platform's database schema exists, the worker can commit the records from outputs/ and this
store can be replaced without changing the API responses.
"""
import json
import os
import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

RUN_ID = re.compile(r'^[0-9a-f]{32}$')


def now():
    return datetime.now(UTC).isoformat(timespec='seconds').replace('+00:00', 'Z')


class RunStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def new(self, survey_id, source):
        run_id = uuid.uuid4().hex
        d = self.root / run_id
        (d / 'inputs').mkdir(parents=True)
        (d / 'outputs').mkdir()
        meta = {'run_id': run_id, 'survey_id': survey_id, 'source': source, 'status': 'queued',
                'stage': None, 'created_at': now(), 'started_at': None, 'finished_at': None,
                'error': None, 'warnings': [], 'summary': None}
        self._write(run_id, meta)
        return meta

    def dir(self, run_id):
        if not RUN_ID.match(run_id or ''):
            raise KeyError(run_id)
        d = self.root / run_id
        if not (d / 'run.json').exists():
            raise KeyError(run_id)
        return d

    def get(self, run_id):
        with open(self.dir(run_id) / 'run.json') as f:
            return json.load(f)

    def update(self, run_id, **fields):
        with self._lock:
            meta = self.get(run_id)
            meta.update(fields)
            self._write(run_id, meta)
            return meta

    def list(self, limit=50):
        runs = []
        for d in self.root.iterdir():
            if RUN_ID.match(d.name) and (d / 'run.json').exists():
                try:
                    runs.append(self.get(d.name))
                except (OSError, json.JSONDecodeError):
                    continue
        runs.sort(key=lambda m: m['created_at'], reverse=True)
        return runs[:limit]

    def complete(self, run_id, records, **fields):
        """Mark a run finished; the records are already in outputs/c4_segments.jsonl."""
        return self.update(run_id, **fields)

    def segments_geojson(self, run_id):
        return json.loads((self.dir(run_id) / 'outputs' / 'c4_segments.geojson').read_text())

    def records(self, run_id):
        with open(self.dir(run_id) / 'outputs' / 'c4_segments.jsonl') as f:
            return [json.loads(line) for line in f if line.strip()]

    def _write(self, run_id, meta):
        d = self.root / run_id
        tmp = d / f'run.json.{os.getpid()}.{threading.get_ident()}.tmp'
        with open(tmp, 'w') as f:
            json.dump(meta, f, indent=2, allow_nan=False)
        os.replace(tmp, d / 'run.json')
