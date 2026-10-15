"""Tests for the model export and PVS conversion scripts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('sklearn')
xgb = pytest.importorskip('xgboost')
joblib = pytest.importorskip('joblib')

from ml.c4.roughness import c4_pipeline as c4
from ml.c4.roughness.predictor import Predictor
from ml.c4.scripts.export_portable_model import export
from ml.c4.scripts.pvs_to_contract_csv import convert

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures' / 'synthetic_drive'


def test_export_reproduces_joblib_pipeline(tmp_path):
    feats = [f'f{i}' for i in range(6)]
    rng = np.random.default_rng(1)
    X, y = rng.standard_normal((300, 6)).astype(np.float32), rng.integers(0, 3, 300)
    pipe = c4.make_model('XGB')
    pipe.set_params(xgbclassifier__n_estimators=20, xgbclassifier__n_jobs=1)
    pipe.fit(X, y)
    bundle = {'model': pipe, 'model_name': 'XGB', 'features': feats, 'speed_slopes': {}, 'speed_ref_log': 2.0,
              'config': dict(c4.CFG), 'sensor': 'left_dashboard', 'channels': c4.ROBUST_CH,
              'classes': c4.CLASSES, 'lovo_macro_f1': 0.828}
    joblib.dump(bundle, tmp_path / 'c4_deployment_model.joblib')
    export(str(tmp_path / 'c4_deployment_model.joblib'), str(tmp_path / 'portable'))
    p = Predictor(str(tmp_path / 'portable'))
    assert p.source == 'portable' and len(p.checksum) == 64
    assert np.allclose(p.model.predict_proba(X), pipe.predict_proba(X), atol=1e-5)
    j = Predictor(str(tmp_path))                                   # joblib bundle loads as well
    assert j.source == 'joblib'


def test_pvs_conversion_writes_contract_files(tmp_path):
    imu = pd.read_csv(FIXTURE / 'imu.csv')
    gps = pd.read_csv(FIXTURE / 'gps.csv')
    df = pd.DataFrame({'timestamp': 1.5e9 + imu.timestamp_ms / 1000})
    for s in c4.SENSORS:                                           # PVS layout: every sensor, gyro in deg/s
        for k, cols in (('acc', ['ax_mps2', 'ay_mps2', 'az_mps2']), ('gyro', ['gx_rads', 'gy_rads', 'gz_rads'])):
            for c, src in zip(c4.sensor_cols(s, k), cols):
                df[c] = np.rad2deg(imu[src]) if k == 'gyro' else imu[src]
    for c in ('latitude', 'longitude', 'speed_mps'):
        df[c if c != 'speed_mps' else 'speed'] = np.interp(imu.timestamp_ms, gps.timestamp_ms, gps[c])
    df['road_condition'] = 'good'
    df.to_csv(tmp_path / 'synced.csv', index=False)
    out = convert(1, str(tmp_path / 'contract'), str(tmp_path / 'synced.csv'))
    c_imu = pd.read_csv(Path(out) / 'imu.csv')
    assert list(c_imu.columns) == ['timestamp_ms', 'ax_mps2', 'ay_mps2', 'az_mps2', 'gx_rads', 'gy_rads', 'gz_rads']
    assert np.allclose(c_imu.gx_rads, imu.gx_rads, atol=1e-4)
    c_gps = pd.read_csv(Path(out) / 'gps.csv')
    assert list(c_gps.columns) == ['timestamp_ms', 'latitude', 'longitude', 'accuracy_m', 'speed_mps']
    assert json.loads((Path(out) / 'manifest.json').read_text())['gravity_included'] is True


def test_portable_reproduces_float64_per_step_scaling(tmp_path):
    """Models trained with scikit-learn <= 1.7 scale float32 input step by step with float64 parameters."""
    from ml.c4.roughness.predictor import PortableXGB
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 20)).astype(np.float32)
    y = rng.integers(0, 3, 2000)
    mean, scale = rng.standard_normal(20) * 3.3, np.abs(rng.standard_normal(20)) * 1.7 + 0.1
    Z = X.copy()
    Z -= mean
    Z /= scale
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, n_jobs=1).fit(Z, y)
    clf.get_booster().save_model(str(tmp_path / 'b.json'))
    meta = {'impute_median': [0.0] * 20, 'scale_mean': mean.tolist(), 'scale_std': scale.tolist(),
            'scale_arithmetic': 'float64_per_step'}
    assert np.abs(PortableXGB(meta, str(tmp_path / 'b.json')).predict_proba(X) - clf.predict_proba(Z)).max() == 0
