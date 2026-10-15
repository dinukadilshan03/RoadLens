"""Contract tests for the C4 adapter (synthetic fixture + demo model; no real data or weights needed)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('scipy')
pytest.importorskip('sklearn')
pytest.importorskip('xgboost')

import itertools

from ml.c4.roughness.adapter import C4Cancelled, C4Request, LocalRunContext, run
from ml.c4.roughness.inputs import C4InputError
from ml.c4.scripts.make_demo_model import build

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures' / 'synthetic_drive'
CLASSES = {'good', 'regular', 'bad'}


@pytest.fixture(scope='session')
def model_dir(tmp_path_factory):
    return build(str(tmp_path_factory.mktemp('model')), str(FIXTURE))


def manifest():
    return json.loads((FIXTURE / 'manifest.json').read_text())


def request(model_dir, **kw):
    args = {'survey_id': 'survey-1', 'imu_path': str(FIXTURE / 'imu.csv'), 'gps_path': str(FIXTURE / 'gps.csv'),
                'manifest': manifest(), 'model_dir': model_dir, 'external_run_id': 'run-1'}
    args.update(kw)
    return C4Request(**args)


@pytest.fixture(scope='session')
def result(model_dir, tmp_path_factory):
    return run(request(model_dir), LocalRunContext(tmp_path_factory.mktemp('out')))


def test_records_follow_contract(result):
    assert result.records
    for r in result.records:
        assert r['schema_version'] == '1.0' and r['component'] == 'C4'
        assert r['survey_id'] == 'survey-1' and r['external_run_id'] == 'run-1'
        assert set(r['producer']) == {'model_version', 'code_version', 'config_version'}
        assert r['source_kind'] == 'mock'                                    # demo model
        assert r['roughness_class'] in CLASSES
        assert isinstance(r['interval_start_ms'], int) and isinstance(r['interval_end_ms'], int)
        assert 0 <= r['interval_start_ms'] < r['interval_end_ms']
        assert r['confidence'] is None or 0 <= r['confidence'] <= 1
        if r['geometry'] is None:
            assert r['geometry_null_reason']
        else:
            coords = r['geometry']['coordinates']
            assert r['geometry']['type'] == 'LineString' and len(coords) >= 2
            assert all(-180 <= x <= 180 and -90 <= y <= 90 and (x, y) != (0, 0) for x, y in coords)
    json.dumps(result.records, allow_nan=False)                              # no NaN / Infinity


def test_segments_do_not_overlap_and_respect_max_length(result):
    iv = sorted((r['interval_start_ms'], r['interval_end_ms']) for r in result.records)
    assert all(a[1] <= b[0] for a, b in itertools.pairwise(iv))
    assert all(e - s <= 10_000 for s, e in iv)


def test_stationary_start_is_not_assessed(result):
    assert min(r['interval_start_ms'] for r in result.records) >= 10_000
    assert result.summary['processing']['noise_profile_subtracted'] is True


def test_demo_model_recovers_simulated_road_types(result):
    truth = pd.read_csv(FIXTURE / 'truth.csv')
    hit = tot = 0
    for r in result.records:
        mid = (r['interval_start_ms'] + r['interval_end_ms']) / 2
        road = truth.road_type[(truth.start_ms <= mid) & (mid < truth.end_ms)].iat[0]
        tot += 1
        hit += road == r['roughness_class']
    assert hit / tot >= 0.75


def test_artifacts_are_relative_paths(result):
    assert set(result.artifacts) == {'segments_jsonl', 'segments_geojson', 'windows_csv', 'summary_json'}
    assert all(not Path(rel).is_absolute() and '..' not in rel for rel in result.artifacts.values())


def test_deterministic(model_dir, tmp_path):
    a = run(request(model_dir), LocalRunContext(tmp_path / 'a')).records
    b = run(request(model_dir), LocalRunContext(tmp_path / 'b')).records
    assert a == b
    lines = (tmp_path / 'a' / 'c4_segments.jsonl').read_text().splitlines()
    assert [json.loads(x) for x in lines] == a
    gj = json.loads((tmp_path / 'a' / 'c4_segments.geojson').read_text())
    assert gj['type'] == 'FeatureCollection' and len(gj['features']) == len(a)


def test_separate_acc_and_gyro_streams(model_dir, tmp_path):
    imu = pd.read_csv(FIXTURE / 'imu.csv')
    imu[['timestamp_ms', 'ax_mps2', 'ay_mps2', 'az_mps2']].to_csv(tmp_path / 'acc.csv', index=False)
    g = imu[['timestamp_ms', 'gx_rads', 'gy_rads', 'gz_rads']].copy()
    g['timestamp_ms'] += 3                                                   # different clock phase
    g.to_csv(tmp_path / 'gyro.csv', index=False)
    res = run(request(model_dir, imu_path=None, acc_path=str(tmp_path / 'acc.csv'),
                      gyro_path=str(tmp_path / 'gyro.csv')), LocalRunContext(tmp_path / 'out'))
    assert res.records


def test_missing_speed_is_derived_from_positions(model_dir, tmp_path):
    gps = pd.read_csv(FIXTURE / 'gps.csv')
    gps['speed_mps'] = np.nan
    gps.to_csv(tmp_path / 'gps.csv', index=False)
    res = run(request(model_dir, gps_path=str(tmp_path / 'gps.csv')), LocalRunContext(tmp_path / 'out'))
    assert res.summary['input']['speed_source'] == 'derived_from_positions'
    assert res.records


def test_gps_gap_is_not_bridged(model_dir, tmp_path):
    gps = pd.read_csv(FIXTURE / 'gps.csv')
    gps = gps[(gps.timestamp_ms < 35_000) | (gps.timestamp_ms > 50_000)]
    gps.to_csv(tmp_path / 'gps.csv', index=False)
    res = run(request(model_dir, gps_path=str(tmp_path / 'gps.csv')), LocalRunContext(tmp_path / 'out'))
    for r in res.records:
        assert r['interval_end_ms'] <= 36_000 or r['interval_start_ms'] >= 49_000
    assert any('GPS gap' in w for w in res.warnings)


@pytest.mark.parametrize('change, message', [
    (lambda imu, gps, m: m.pop('gravity_included'), 'gravity_included'),
    (lambda imu, gps, m: imu.drop(columns='gz_rads', inplace=True), 'missing required column'),
    (lambda imu, gps, m: imu.__setitem__(['ax_mps2', 'ay_mps2', 'az_mps2'],
                                         imu[['ax_mps2', 'ay_mps2', 'az_mps2']] / 9.80665), 'units'),
    (lambda imu, gps, m: gps.__setitem__('latitude', 0.0) or gps.__setitem__('longitude', 0.0), 'GPS'),
])
def test_invalid_input_rejected(model_dir, tmp_path, change, message):
    imu, gps, m = pd.read_csv(FIXTURE / 'imu.csv'), pd.read_csv(FIXTURE / 'gps.csv'), manifest()
    change(imu, gps, m)
    imu.to_csv(tmp_path / 'imu.csv', index=False)
    gps.to_csv(tmp_path / 'gps.csv', index=False)
    with pytest.raises(C4InputError, match=message):
        run(request(model_dir, imu_path=str(tmp_path / 'imu.csv'), gps_path=str(tmp_path / 'gps.csv'), manifest=m),
            LocalRunContext(tmp_path / 'out'))


def test_low_sampling_rate_rejected(model_dir, tmp_path):
    imu = pd.read_csv(FIXTURE / 'imu.csv').iloc[::4]                        # 25 Hz
    imu.to_csv(tmp_path / 'imu.csv', index=False)
    with pytest.raises(C4InputError, match='sampling rate'):
        run(request(model_dir, imu_path=str(tmp_path / 'imu.csv')), LocalRunContext(tmp_path / 'out'))


def test_cancellation(model_dir, tmp_path):
    class Cancelled(LocalRunContext):
        def is_cancelled(self):
            return len(self.stages) >= 2
    with pytest.raises(C4Cancelled):
        run(request(model_dir), Cancelled(tmp_path / 'out'))
