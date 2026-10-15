"""
Export the deployment model from Notebook 07 into the portable format used by the adapter.

The joblib bundle stores a scikit-learn pipeline (median imputer -> standard scaler -> XGBClassifier)
that can only be loaded with the same scikit-learn/xgboost versions as the Colab session that created it.
The portable format keeps the same model as plain JSON: imputer medians, scaler mean/std and the
xgboost booster (xgboost's own JSON format). Predictions are checked to be identical after export.

Run in Colab (Drive mounted):
    python -m ml.c4.scripts.export_portable_model \
        --bundle /content/drive/MyDrive/PVS_Model_Results_v2/c4_deployment_model.joblib \
        --out /content/drive/MyDrive/PVS_Model_Results_v2/portable
Then copy the two files from --out into ml/c4/models/production/ (not committed to Git).
"""
import argparse
import os

import joblib
import numpy as np


def export(bundle_path, out_dir):
    from ml.c4.roughness.predictor import PortableXGB, save_portable, scale_arithmetic
    b = joblib.load(bundle_path)
    pipe = b['model']
    steps = dict(pipe.named_steps)
    imp, sc, clf = steps['simpleimputer'], steps['standardscaler'], steps['xgbclassifier']
    meta = {k: v for k, v in b.items() if k != 'model'}
    meta.setdefault('model_version', f"{b.get('model_name', 'XGB')}-c7-lovo{b.get('lovo_macro_f1') or 0:.3f}")
    mode = scale_arithmetic(imp, sc, len(b['features']))
    print('Scaler arithmetic:', mode)
    save_portable(out_dir, meta, clf.get_booster(), imp.statistics_, sc.mean_, sc.scale_, mode)

    import json
    with open(os.path.join(out_dir, 'c4_model_portable.json')) as f:
        pm = json.load(f)
    port = PortableXGB(pm, os.path.join(out_dir, pm['booster_file']))
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, len(b['features']))).astype(np.float32)
    X[rng.random(X.shape) < 0.02] = np.nan
    d = np.abs(pipe.predict_proba(X) - port.predict_proba(X)).max()
    print(f'Max probability difference joblib vs portable: {d:.2e}')
    if d > 1e-5:
        raise SystemExit('Portable export does not reproduce the joblib model.')
    print('Portable model written to', out_dir)
    return out_dir


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    export(a.bundle, a.out)
