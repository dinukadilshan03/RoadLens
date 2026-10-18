"""
C4 REST API (FastAPI router, mounted by the RoadLens backend under /api/v1).

    GET  /api/v1/c4/model                    model in use (version, checksum, demo or production)
    POST /api/v1/c4/runs                     upload IMU (or acc + gyro), GPS, manifest (+ optional labels) -> 202
    POST /api/v1/c4/runs/demo                run the synthetic fixture (no upload needed)
    GET  /api/v1/c4/runs                     recent runs
    GET  /api/v1/c4/runs/{run_id}            status, stage, summary, warnings, error
    POST /api/v1/c4/runs/{run_id}/cancel     cooperative cancellation (between stages)
    GET  /api/v1/c4/runs/{run_id}/segments   GeoJSON FeatureCollection of road segments (map)
    GET  /api/v1/c4/runs/{run_id}/records    contract records (schema 1.0)
    GET  /api/v1/c4/runs/{run_id}/windows.csv  per-window diagnostics

Mount in backend/app/main.py:
    from ml.c4.api import router as c4_router
    app.include_router(c4_router, prefix="/api/v1")

Settings (environment variables):
    C4_STORAGE_DIR   run folders (default: ml/c4/runs, git-ignored; use a volume in Docker)
    C4_MODEL_DIR     production model folder (default: ml/c4/models/production)
    C4_ALLOW_DEMO    "1" (default) uses / builds the synthetic demo model when no production model exists;
                     its results are marked source_kind = "mock". Set "0" in production.

Errors use the platform format {code, message, request_id, details}: 422 invalid input, 404 unknown run,
409 run not finished, 413 file too large, 503 no model available.
"""
import json
import logging
import os
import threading
import traceback
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ..roughness.adapter import C4Cancelled, C4Request, run
from ..roughness.evaluate import evaluate, read_labels
from ..roughness.inputs import C4InputError
from .store import RunStore, now

log = logging.getLogger('roadlens.c4')
C4_DIR = Path(__file__).resolve().parents[1]
FIXTURE_DIR = C4_DIR / 'fixtures' / 'synthetic_drive'
MAX_CSV_BYTES = 250 * 1024 * 1024
MAX_JSON_BYTES = 10 * 1024 * 1024
MODEL_FILES = ('c4_model_portable.json', 'c4_deployment_model.joblib')


def _error(status, code, message, details=None):
    return JSONResponse(status_code=status, content={'code': code, 'message': message,
                                                     'request_id': uuid.uuid4().hex, 'details': details or {}})


class _ModelResolver:
    def __init__(self, model_dir, allow_demo, demo_dir):
        self.model_dir, self.allow_demo, self.demo_dir = Path(model_dir), allow_demo, Path(demo_dir)
        self._lock = threading.Lock()

    @staticmethod
    def _has_model(d):
        return any((d / f).exists() for f in MODEL_FILES)

    def resolve(self):
        """(model_dir, kind) or raises LookupError."""
        if self._has_model(self.model_dir):
            return self.model_dir, 'production'
        if not self.allow_demo:
            raise LookupError(f'No C4 model found in {self.model_dir}.')
        with self._lock:
            if not self._has_model(self.demo_dir):
                from ..scripts.make_demo_model import build
                log.warning('No production C4 model; building the synthetic demo model in %s', self.demo_dir)
                build(str(self.demo_dir), str(FIXTURE_DIR))
        return self.demo_dir, 'demo'


class _StoreContext:
    """RunContext that reports progress into the run store and honours cancel requests."""

    def __init__(self, store, run_id, output_dir):
        self.store, self.run_id, self.output_dir = store, run_id, Path(output_dir)

    def progress(self, stage, completed=None, total=None):
        self.store.update(self.run_id, stage=stage, progress={'completed': completed, 'total': total})

    def is_cancelled(self):
        return bool(self.store.get(self.run_id).get('cancel_requested'))


def _save_upload(upload, path, limit):
    size = 0
    with open(path, 'wb') as f:
        while chunk := upload.file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                raise OverflowError(upload.filename)
            f.write(chunk)
    if size == 0:
        raise ValueError(f'{upload.filename or "file"} is empty.')
    return size


def build_router(storage_dir=None, model_dir=None, allow_demo=None, demo_dir=None, store=None):
    """store: any object with the RunStore interface (default: file-based RunStore in storage_dir)."""
    storage_dir = storage_dir or os.environ.get('C4_STORAGE_DIR') or C4_DIR / 'runs'
    model_dir = model_dir or os.environ.get('C4_MODEL_DIR') or C4_DIR / 'models' / 'production'
    if allow_demo is None:
        allow_demo = os.environ.get('C4_ALLOW_DEMO', '1') != '0'
    demo_dir = demo_dir or C4_DIR / 'models' / 'demo'
    store = store or RunStore(storage_dir)
    models = _ModelResolver(model_dir, allow_demo, demo_dir)
    one_at_a_time = threading.Semaphore(1)          # inference is CPU-bound; queue runs in order
    r = APIRouter(prefix='/c4', tags=['C4 road roughness'])
    r.store = store

    def execute(run_id, req_kwargs, labels_path=None):
        d = store.dir(run_id)
        with one_at_a_time:
            if store.get(run_id).get('cancel_requested'):
                store.update(run_id, status='cancelled', finished_at=now())
                return
            store.update(run_id, status='running', started_at=now())
            try:
                mdir, kind = models.resolve()
                labels = read_labels(labels_path) if labels_path else None      # validate before inference
                res = run(C4Request(model_dir=str(mdir), external_run_id=run_id, **req_kwargs),
                          _StoreContext(store, run_id, d / 'outputs'))
                summary = {k: res.summary[k] for k in ('segments', 'input', 'processing', 'model')}
                if labels is not None:
                    import pandas as pd
                    windows = pd.read_csv(d / 'outputs' / res.artifacts['windows_csv'])
                    summary['label_check'] = evaluate(windows, res.records, labels)
                store.complete(run_id, res.records, status='succeeded', stage='done', finished_at=now(),
                               model_kind=kind, model_version=res.summary['producer']['model_version'],
                               summary=summary, warnings=res.warnings, artifacts=res.artifacts)
            except C4InputError as exc:
                store.update(run_id, status='failed', finished_at=now(),
                             error={'code': 'invalid_input', 'message': str(exc), 'retryable': False})
            except C4Cancelled as exc:
                store.update(run_id, status='cancelled', finished_at=now(),
                             error={'code': 'cancelled', 'message': str(exc), 'retryable': False})
            except LookupError as exc:
                store.update(run_id, status='failed', finished_at=now(),
                             error={'code': 'model_unavailable', 'message': str(exc), 'retryable': True})
            except Exception as exc:  # noqa: BLE001 - any failure must end the run as failed, with a log
                log.error('C4 run %s failed:\n%s', run_id, traceback.format_exc())
                store.update(run_id, status='failed', finished_at=now(),
                             error={'code': 'internal_error', 'message': f'Internal processing error ({type(exc).__name__}).',
                                    'retryable': True})

    def public(meta):
        return {k: v for k, v in meta.items() if k != 'artifacts'}

    def finished(run_id):
        try:
            meta = store.get(run_id)
        except KeyError:
            return _error(404, 'not_found', 'Unknown C4 run.', {'run_id': run_id})
        if meta['status'] != 'succeeded':
            return _error(409, 'run_not_succeeded', f'Run is {meta["status"]}; results are not available.',
                          {'status': meta['status']})
        return None

    @r.get('/model')
    def model_info():
        try:
            mdir, kind = models.resolve()
        except LookupError as exc:
            return _error(503, 'model_unavailable', str(exc))
        from ..roughness.adapter import _predictor
        info = _predictor(str(mdir)).info()
        return {'kind': kind, 'source_kind': 'mock' if kind == 'demo' else 'inference', **info}

    @r.post('/runs', status_code=202)
    def create_run(            # sync: FastAPI runs it in a worker thread, so file I/O does not block the loop
            background: BackgroundTasks,
            gps: Annotated[UploadFile, File(description='GPS CSV: timestamp_ms,latitude,longitude,accuracy_m,speed_mps')],
            imu: Annotated[UploadFile | None, File(description='IMU CSV: timestamp_ms,ax_mps2,...,gz_rads')] = None,
            acc: Annotated[UploadFile | None, File(description='Accelerometer CSV (with gyro, instead of imu)')] = None,
            gyro: Annotated[UploadFile | None, File(description='Gyroscope CSV (with acc, instead of imu)')] = None,
            manifest: Annotated[UploadFile | None, File(description='Sensor manifest JSON file')] = None,
            manifest_json: Annotated[str | None, Form(description='Sensor manifest as a JSON string')] = None,
            labels: Annotated[UploadFile | None, File(description='Optional reference labels CSV '
                                                      '(timestamp_ms,road_condition) to evaluate the result')] = None,
            survey_id: Annotated[str | None, Form()] = None):
        if not imu and not (acc and gyro):
            return _error(422, 'invalid_input', 'Upload an IMU CSV, or both an accelerometer CSV and a gyroscope CSV.')
        if not manifest and not manifest_json:
            return _error(422, 'invalid_input', 'A sensor manifest is required (file "manifest" or field "manifest_json").')
        raw = manifest.file.read(MAX_JSON_BYTES + 1) if manifest else manifest_json.encode()
        if len(raw) > MAX_JSON_BYTES:
            return _error(413, 'too_large', 'Manifest exceeds 10 MiB.')
        try:
            man = json.loads(raw)
        except ValueError as exc:
            return _error(422, 'invalid_input', f'Manifest is not valid JSON ({exc}).')
        if not isinstance(man, dict):
            return _error(422, 'invalid_input', 'Manifest must be a JSON object.')

        survey_id = (survey_id or '').strip()[:128] or f'survey-{uuid.uuid4().hex[:8]}'
        meta = store.new(survey_id, 'upload')
        d = store.dir(meta['run_id']) / 'inputs'
        paths = {}
        try:
            for name, up in (('imu', imu), ('acc', acc), ('gyro', gyro), ('gps', gps), ('labels', labels)):
                if up is not None:
                    p = d / f'{name}.csv'
                    _save_upload(up, p, MAX_CSV_BYTES)
                    paths[f'{name}_path'] = str(p)
        except OverflowError:
            store.update(meta['run_id'], status='rejected', error={'code': 'too_large', 'message': 'CSV exceeds 250 MiB.'})
            return _error(413, 'too_large', 'Each CSV file must be at most 250 MiB.')
        except ValueError as exc:
            store.update(meta['run_id'], status='rejected', error={'code': 'invalid_input', 'message': str(exc)})
            return _error(422, 'invalid_input', str(exc))
        if 'imu_path' in paths:
            paths.pop('acc_path', None), paths.pop('gyro_path', None)
        (d / 'manifest.json').write_text(json.dumps(man, indent=2))
        labels_path = paths.pop('labels_path', None)
        background.add_task(execute, meta['run_id'], {'survey_id': survey_id, 'manifest': man, **paths}, labels_path)
        return public(meta)

    @r.post('/runs/demo', status_code=202)
    def create_demo_run(background: BackgroundTasks):
        meta = store.new('demo-synthetic-drive', 'demo')
        man = json.loads((FIXTURE_DIR / 'manifest.json').read_text())
        background.add_task(execute, meta['run_id'], {
            'survey_id': meta['survey_id'], 'manifest': man,
            'imu_path': str(FIXTURE_DIR / 'imu.csv'), 'gps_path': str(FIXTURE_DIR / 'gps.csv')})
        return public(meta)

    @r.get('/runs')
    def list_runs(limit: int = 50):
        return {'runs': [public(m) for m in store.list(max(1, min(limit, 200)))]}

    @r.get('/runs/{run_id}')
    def get_run(run_id: str):
        try:
            return public(store.get(run_id))
        except KeyError:
            return _error(404, 'not_found', 'Unknown C4 run.', {'run_id': run_id})

    @r.post('/runs/{run_id}/cancel', status_code=202)
    def cancel_run(run_id: str):
        try:
            meta = store.get(run_id)
        except KeyError:
            return _error(404, 'not_found', 'Unknown C4 run.', {'run_id': run_id})
        if meta['status'] not in ('queued', 'running'):
            return _error(409, 'not_cancellable', f'Run is already {meta["status"]}.')
        return public(store.update(run_id, cancel_requested=True))

    @r.get('/runs/{run_id}/segments')
    def segments(run_id: str):
        if err := finished(run_id):
            return err
        return JSONResponse(store.segments_geojson(run_id), media_type='application/geo+json')

    @r.get('/runs/{run_id}/records')
    def records(run_id: str):
        if err := finished(run_id):
            return err
        return {'records': store.records(run_id)}

    @r.get('/runs/{run_id}/windows.csv')
    def windows(run_id: str):
        if err := finished(run_id):
            return err
        return FileResponse(store.dir(run_id) / 'outputs' / 'c4_windows.csv', media_type='text/csv', filename=f'c4_windows_{run_id}.csv')

    return r
