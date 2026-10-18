"""
Score the app (adapter) output of a PVS trip against its labels, the same way as the research evaluation.

For every 2 s window the app assessed (moving, no manoeuvre), the majority PVS label of that window is used
when it covers >= 80 % of the window (label purity, as in training). Macro-F1 over good/regular/bad.

Usage:
    python -m ml.c4.scripts.score_app_run --trip <folder with imu.csv gps.csv manifest.json labels.csv> \
        --model-dir <model folder> [--out <output folder>]
"""
import argparse
import json
import os
import tempfile

import pandas as pd

from ml.c4.roughness.adapter import C4Request, LocalRunContext, run
from ml.c4.roughness.evaluate import evaluate, read_labels


def score(trip_dir, model_dir, out_dir=None):
    out_dir = out_dir or tempfile.mkdtemp(prefix='c4score_')
    with open(os.path.join(trip_dir, 'manifest.json')) as f:
        manifest = json.load(f)
    res = run(C4Request(survey_id=os.path.basename(os.path.normpath(trip_dir)),
                        imu_path=os.path.join(trip_dir, 'imu.csv'), gps_path=os.path.join(trip_dir, 'gps.csv'),
                        manifest=manifest, model_dir=model_dir), LocalRunContext(out_dir))
    windows = pd.read_csv(os.path.join(out_dir, 'c4_windows.csv'))
    ev = evaluate(windows, res.records, read_labels(os.path.join(trip_dir, 'labels.csv')))
    ev.pop('segments')
    return {'segments': res.summary['segments']['count'], 'model': res.summary['model']['model_version'], **ev}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--trip', required=True)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    print(json.dumps(score(a.trip, a.model_dir, a.out), indent=2))
