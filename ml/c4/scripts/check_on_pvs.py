"""
Run the adapter on a PVS trip converted by pvs_to_contract_csv.py and compare the segments with the labels.

The comparison is per 1 s slice: the predicted class of the segment covering the slice centre versus the
majority PVS label (worst side) in that slice. Note: a trip that was used to train the model gives an
optimistic score; use a held-out vehicle (e.g. a model trained without that vehicle) for a fair check.
"""
import argparse
import json
import os
import tempfile

import numpy as np
import pandas as pd

from ml.c4.roughness.adapter import C4Request, LocalRunContext, run


def check(contract_dir, model_dir, out_dir=None):
    from sklearn.metrics import classification_report, f1_score
    with open(os.path.join(contract_dir, 'manifest.json')) as f:
        manifest = json.load(f)
    out_dir = out_dir or tempfile.mkdtemp(prefix='c4chk_')
    res = run(C4Request(survey_id='pvs-check', imu_path=os.path.join(contract_dir, 'imu.csv'),
                        gps_path=os.path.join(contract_dir, 'gps.csv'), manifest=manifest, model_dir=model_dir),
              LocalRunContext(out_dir, verbose=True))
    lab = pd.read_csv(os.path.join(contract_dir, 'labels.csv')).dropna()
    lab['sec'] = lab.timestamp_ms // 1000
    truth = lab.groupby('sec').road_condition.agg(lambda s: s.value_counts().idxmax())
    rows = []
    for r in res.records:
        for s in range(int(np.ceil(r['interval_start_ms'] / 1000)), int(r['interval_end_ms'] // 1000)):
            if s in truth.index:
                rows.append((truth[s], r['roughness_class']))
    y, p = zip(*rows)
    print(classification_report(y, p, labels=['good', 'regular', 'bad'], zero_division=0))
    print('macro-F1 per second:', round(f1_score(y, p, average='macro', labels=['good', 'regular', 'bad']), 4))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--contract', required=True)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    check(a.contract, a.model_dir, a.out)
