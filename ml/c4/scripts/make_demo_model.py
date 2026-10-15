"""
Build a small DEMO model from the synthetic fixture, in the portable format.

The demo model exists only so that the adapter, tests and the platform can be exercised end to end
without the real model weights. It is marked "demo": true, so every record it produces has
source_kind = "mock". It must never be used to assess real roads.

Usage:  python -m ml.c4.scripts.make_demo_model --out ml/c4/models/demo
"""
import argparse
import os
import tempfile

import numpy as np
import pandas as pd

from ml.c4.roughness import c4_pipeline as c4
from ml.c4.roughness.inputs import check_manifest, read_gps, read_imu, synchronise
from ml.c4.roughness.predictor import Predictor, save_portable
from ml.c4.scripts.make_fixture import make_fixture

CHANNELS = c4.ROBUST_CH
STATS = ['t__{ch}__std', 't__{ch}__rms', 't__{ch}__peak', 't__{ch}__iqr', 'f__{ch}__bprel_3-8',
         'f__{ch}__bprel_8-16', 'f__{ch}__bprel_16-30', 'f__{ch}__entropy']


def build(out_dir, fixture_dir=None):
    import xgboost as xgb
    fixture_dir = fixture_dir or make_fixture(tempfile.mkdtemp(prefix='c4fx_'))
    sensor = 'left_dashboard'
    features = [s.format(ch=f'{sensor}__{c}') for c in CHANNELS for s in STATS]
    meta = {'model_name': 'c4-demo-xgb', 'model_version': 'demo-1', 'demo': True, 'features': features,
            'speed_slopes': {}, 'speed_ref_log': float(np.log(9.0)), 'config': dict(c4.CFG),
            'sensor': sensor, 'channels': CHANNELS, 'classes': c4.CLASSES, 'lovo_macro_f1': None,
            'notes': 'Synthetic demo model for tests. NOT for real road assessment.'}
    pred = Predictor.from_objects(meta)
    imu = read_imu(os.path.join(fixture_dir, 'imu.csv'))
    gps = read_gps(os.path.join(fixture_dir, 'gps.csv'))
    check_manifest({'gravity_included': True})
    track, _ = synchronise(imu, gps)
    A, F, _, _ = pred.extract(track)
    X, moving = pred._normalise(F, A)
    truth = pd.read_csv(os.path.join(fixture_dir, 'truth.csv'))
    mid = (A.t_start.to_numpy() + 1.0) * 1000
    road = np.array([truth.road_type[(truth.start_ms <= m) & (m < truth.end_ms)].iat[0] for m in mid])
    use = moving & (road != 'stationary')
    y = np.array([c4.CLASS_ID[r] for r in road[use]])
    Xu = X.to_numpy()[use]
    clf = xgb.XGBClassifier(n_estimators=40, max_depth=3, learning_rate=0.3, subsample=1.0,
                            random_state=0, n_jobs=1, objective='multi:softprob')
    clf.fit(Xu, y)
    k = Xu.shape[1]
    return save_portable(out_dir, meta, clf.get_booster(), np.zeros(k), np.zeros(k), np.ones(k))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'models', 'demo'))
    p.add_argument('--fixture', default=None)
    a = p.parse_args()
    print('Demo model written to', build(a.out, a.fixture))
