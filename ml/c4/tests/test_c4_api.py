"""API tests for the C4 router (FastAPI TestClient; background tasks run before the response returns)."""
import json
from pathlib import Path

import pytest

pytest.importorskip('fastapi')
pytest.importorskip('multipart')
pytest.importorskip('httpx')
pytest.importorskip('xgboost')

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ml.c4.api.router import build_router

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures' / 'synthetic_drive'


@pytest.fixture(scope='module')
def client(tmp_path_factory):
    base = tmp_path_factory.mktemp('api')
    app = FastAPI()
    app.include_router(build_router(storage_dir=base / 'runs', model_dir=base / 'none',
                                    allow_demo=True, demo_dir=base / 'demo'), prefix='/api/v1')
    return TestClient(app)


def upload(client, **override):
    files = {'imu': ('imu.csv', (FIXTURE / 'imu.csv').read_bytes(), 'text/csv'),
             'gps': ('gps.csv', (FIXTURE / 'gps.csv').read_bytes(), 'text/csv'),
             'manifest': ('manifest.json', (FIXTURE / 'manifest.json').read_bytes(), 'application/json')}
    files.update(override)
    files = {k: v for k, v in files.items() if v is not None}
    return client.post('/api/v1/c4/runs', files=files, data={'survey_id': 'test-survey'})


def test_model_info_uses_demo_model(client):
    r = client.get('/api/v1/c4/model')
    assert r.status_code == 200
    body = r.json()
    assert body['kind'] == 'demo' and body['source_kind'] == 'mock' and len(body['model_sha256']) == 64


def test_upload_run_and_results(client):
    r = upload(client)
    assert r.status_code == 202
    run_id = r.json()['run_id']
    meta = client.get(f'/api/v1/c4/runs/{run_id}').json()
    assert meta['status'] == 'succeeded', meta
    assert meta['survey_id'] == 'test-survey' and meta['stage'] == 'done'
    assert meta['summary']['segments']['count'] > 0

    gj = client.get(f'/api/v1/c4/runs/{run_id}/segments')
    assert gj.status_code == 200 and gj.headers['content-type'].startswith('application/geo+json')
    feats = gj.json()['features']
    assert feats and all(f['properties']['roughness_class'] in {'good', 'regular', 'bad'} for f in feats)

    recs = client.get(f'/api/v1/c4/runs/{run_id}/records').json()['records']
    assert len(recs) == len(feats)
    assert all(x['external_run_id'] == run_id and x['source_kind'] == 'mock' for x in recs)

    csv = client.get(f'/api/v1/c4/runs/{run_id}/windows.csv')
    assert csv.status_code == 200 and csv.text.startswith('window_start_ms')

    runs = client.get('/api/v1/c4/runs').json()['runs']
    assert any(x['run_id'] == run_id for x in runs)


def test_demo_run(client):
    r = client.post('/api/v1/c4/runs/demo')
    assert r.status_code == 202
    meta = client.get(f'/api/v1/c4/runs/{r.json()["run_id"]}').json()
    assert meta['status'] == 'succeeded' and meta['source'] == 'demo'


def test_invalid_input_becomes_failed_run(client):
    bad = (FIXTURE / 'manifest.json').read_text().replace('"gravity_included": true', '"gravity_included": false')
    r = upload(client, manifest=('manifest.json', bad.encode(), 'application/json'))
    assert r.status_code == 202
    meta = client.get(f'/api/v1/c4/runs/{r.json()["run_id"]}').json()
    assert meta['status'] == 'failed'
    assert meta['error']['code'] == 'invalid_input' and 'gravity_included' in meta['error']['message']
    res = client.get(f'/api/v1/c4/runs/{meta["run_id"]}/segments')
    assert res.status_code == 409 and res.json()['code'] == 'run_not_succeeded'


def test_request_validation_errors(client):
    r = upload(client, imu=None)
    assert r.status_code == 422 and r.json()['code'] == 'invalid_input' and 'request_id' in r.json()
    r = upload(client, manifest=('manifest.json', b'[1,2]', 'application/json'))
    assert r.status_code == 422
    r = upload(client, gps=('gps.csv', b'', 'text/csv'))
    assert r.status_code == 422


def test_separate_acc_gyro_and_manifest_string(client):
    lines = (FIXTURE / 'imu.csv').read_text().splitlines()
    acc = '\n'.join(','.join(l.split(',')[:4]) for l in lines)
    gyr = '\n'.join(','.join([l.split(',')[0]] + l.split(',')[4:]) for l in lines)
    files = {'acc': ('acc.csv', acc.encode(), 'text/csv'), 'gyro': ('gyro.csv', gyr.encode(), 'text/csv'),
             'gps': ('gps.csv', (FIXTURE / 'gps.csv').read_bytes(), 'text/csv')}
    r = client.post('/api/v1/c4/runs', files=files, data={'manifest_json': (FIXTURE / 'manifest.json').read_text()})
    assert r.status_code == 202
    assert client.get(f'/api/v1/c4/runs/{r.json()["run_id"]}').json()['status'] == 'succeeded'


def test_unknown_and_malformed_run_ids(client):
    assert client.get('/api/v1/c4/runs/' + '0' * 32).status_code == 404
    r = client.get('/api/v1/c4/runs/..%2F..%2Fetc')
    assert r.status_code == 404
    assert client.post('/api/v1/c4/runs/' + 'f' * 32 + '/cancel').status_code == 404


def test_no_model_and_demo_disabled(tmp_path):
    app = FastAPI()
    app.include_router(build_router(storage_dir=tmp_path / 'runs', model_dir=tmp_path / 'none', allow_demo=False),
                       prefix='/api/v1')
    c = TestClient(app)
    r = c.get('/api/v1/c4/model')
    assert r.status_code == 503 and r.json()['code'] == 'model_unavailable'
    run_id = c.post('/api/v1/c4/runs/demo').json()['run_id']
    meta = c.get(f'/api/v1/c4/runs/{run_id}').json()
    assert meta['status'] == 'failed' and meta['error']['code'] == 'model_unavailable'
    json.dumps(meta)
