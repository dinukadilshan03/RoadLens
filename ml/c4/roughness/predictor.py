"""
Cloud inference for Component 4 (FR-17).

The deployed model is the smartphone model (C7) exported by Notebook 07: one dashboard-like sensor,
four orientation-invariant channels, 40 cross-vehicle stable features. For every uploaded trip the
same steps as in Notebooks 03-05 are applied, using the shared c4_pipeline module:

    cleaning and segmentation -> glitch handling, gravity removal, 40 Hz low-pass, invariant channels
    -> 2 s windows -> trip noise profile from stationary windows -> time + PSD features
    -> speed normalisation (slopes learned in training) -> per-trip robust normalisation
    -> selected features -> classifier

Two model formats are supported:
  * the original joblib bundle (c4_deployment_model.joblib), which needs the same scikit-learn /
    xgboost versions as the Colab session that created it;
  * a portable bundle (c4_model_portable.json + c4_model_portable.xgb.json) created by
    scripts/export_portable_model.py, which works with any recent xgboost version.
"""
import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import c4_pipeline as c4
from .inputs import C4InputError

CLASSES = ['good', 'regular', 'bad']


@dataclass
class TripResult:
    windows: pd.DataFrame
    report: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


class PortableXGB:
    """Median imputation + standard scaling + xgboost booster, rebuilt from the portable export.

    The scaling reproduces the float32 rounding of the scikit-learn version that trained the model
    (meta['scale_arithmetic']): 'float64_per_step' (scikit-learn <= 1.7: X -= mean_; X /= scale_ with float64
    parameters) or 'float32_params' (newer: parameters cast to float32 first). One rounding step can move a
    value across a tree split, so this keeps the predictions identical to the trained pipeline.
    """

    def __init__(self, meta, booster_path):
        import xgboost as xgb
        self.fill = np.asarray(meta['impute_median'], float)
        self.mean = np.asarray(meta['scale_mean'], float)
        scale = np.asarray(meta['scale_std'], float)
        self.scale = np.where(scale == 0, 1, scale)
        self.arithmetic = meta.get('scale_arithmetic', 'float32_params')
        if self.arithmetic not in ('float64_per_step', 'float32_params'):
            raise ValueError(f'Unknown scale_arithmetic {self.arithmetic!r} in the model file.')
        self.booster = xgb.Booster()
        self.booster.load_model(booster_path)

    def transform(self, X):
        f32 = np.float32
        X = np.asarray(X, f32)
        X = np.where(np.isfinite(X), X, self.fill.astype(f32))
        if self.arithmetic == 'float64_per_step':
            X = (X - self.mean).astype(f32)
            return (X / self.scale).astype(f32)
        return (X - self.mean.astype(f32)) / self.scale.astype(f32)

    def predict_proba(self, X):
        import xgboost as xgb
        return self.booster.predict(xgb.DMatrix(self.transform(X)))


class Predictor:
    def __init__(self, model_dir):
        portable = os.path.join(model_dir, 'c4_model_portable.json')
        bundle_path = os.path.join(model_dir, 'c4_deployment_model.joblib')
        if os.path.exists(portable):
            with open(portable) as f:
                meta = json.load(f)
            self.model = PortableXGB(meta, os.path.join(model_dir, meta['booster_file']))
            self.source = 'portable'
        elif os.path.exists(bundle_path):
            import joblib
            meta = joblib.load(bundle_path)
            self.model = meta['model']
            self.source = 'joblib'
        else:
            raise FileNotFoundError(
                f'No model found in {model_dir}. Copy c4_deployment_model.joblib (Notebook 07) or the portable '
                'export (scripts/export_portable_model.py) into this folder.')
        files = [portable, os.path.join(model_dir, meta['booster_file'])] if self.source == 'portable' else [bundle_path]
        checksum = hashlib.sha256(''.join(file_sha256(f) for f in files).encode()).hexdigest()
        self._setup(meta, checksum)

    @classmethod
    def from_objects(cls, meta, model=None, checksum='0' * 64, source='memory'):
        """Predictor from an in-memory bundle (model=None: feature extraction only, used to build models)."""
        self = cls.__new__(cls)
        self.model, self.source = model, source
        self._setup(meta, checksum)
        return self

    def _setup(self, meta, checksum):
        self.meta, self.checksum = meta, checksum
        self.version = str(meta.get('model_version') or f"{meta.get('model_name', 'model')}-{checksum[:12]}")
        self.cfg = {**c4.CFG, **meta['config']}            # identical settings to training
        self.features = list(meta['features'])
        self.slopes = meta.get('speed_slopes') or {}
        self.ref = meta.get('speed_ref_log')
        self.sensor = meta.get('sensor', self.cfg['DEPLOY_SENSOR'])
        self.cfg['MANEUVER_SENSOR'] = self.sensor
        self.channels = list(meta.get('channels', c4.ROBUST_CH))
        self.classes = list(meta.get('classes', CLASSES))
        self.name = meta.get('model_name', 'model')
        self.demo = bool(meta.get('demo', False))
        self.cols = {k: c4.sensor_cols(self.sensor, k) for k in ('acc', 'gyro')}
        c4.CFG.update(self.cfg)

    # ------------------------------------------------------------------ info
    def info(self):
        return {'model_name': self.name, 'model_version': self.version, 'model_sha256': self.checksum,
                'format': self.source, 'n_features': len(self.features),
                'sensor': self.sensor, 'channels': self.channels, 'classes': self.classes,
                'lovo_macro_f1': self.meta.get('lovo_macro_f1'), 'demo_model': self.demo,
                'window_sec': c4.CFG['WINDOW_SEC'], 'step_sec': c4.CFG['STEP_SEC'],
                'min_speed_mps': c4.CFG['MIN_SPEED']}

    # ------------------------------------------------------------------ steps
    def _to_pipeline_frame(self, df):
        """Map the phone channels onto the column names used in training."""
        f = pd.DataFrame({'timestamp': df.timestamp.to_numpy()})
        for k in ('acc', 'gyro'):
            for a, col in zip('xyz', self.cols[k]):
                f[col] = df[f'{k}_{a}'].to_numpy()
        for c in ('latitude', 'longitude', 'speed'):
            f[c] = df[c].to_numpy()
        f['accuracy_m'] = df['accuracy_m'].to_numpy() if 'accuracy_m' in df else np.nan
        f['label'] = df['label'].to_numpy() if 'label' in df else -1
        return f

    def _clean(self, df):
        """Same rules as c4.clean_trip, for the single deployed sensor."""
        sens = self.cols['acc'] + self.cols['gyro']
        df[sens] = df[sens].interpolate(limit=c4.CFG['MAX_INTERP_GAP'], limit_area='inside')
        dt = np.diff(df.timestamp.to_numpy(), prepend=df.timestamp.iat[0])
        med = np.median(dt[1:])
        bad = df[sens].isna().any(axis=1).to_numpy() | ~np.isfinite(df.speed.to_numpy())
        gap = dt > c4.CFG['GAP_FACTOR'] * med
        df['seg'] = np.cumsum(gap | bad)
        df = df[~bad]
        keep = df.groupby('seg')['seg'].transform('size') >= c4.win_len() + 2 * c4.CFG['FS']
        out = df[keep].reset_index(drop=True)
        return out, int(out['seg'].nunique())

    def _signals_and_windows(self, df):
        n = len(df)
        proc_names = [f'{self.sensor}__{c}' for c in c4.PROC_CH]
        PROC = np.zeros((n, len(proc_names)), np.float32)
        YAW, LONG = np.zeros(n), np.zeros(n)
        W, S = c4.win_len(), c4.step_len()
        recs, stat = [], []
        for idx in df.groupby('seg', sort=False).indices.values():
            seg = df.iloc[idx]
            _, p, _ = c4.process_sensor(seg, self.sensor)
            PROC[idx] = p
            YAW[idx], LONG[idx] = c4.maneuver_signals(seg)
            a, b = idx[0], idx[-1] + 1
            spd, lab, acc = df.speed.to_numpy(), df.label.to_numpy(), df.accuracy_m.to_numpy()
            for st in range(a, b - W + 1, S):
                sl = slice(st, st + W)
                sp = spd[sl]
                if sp.max() < c4.CFG['STOP_SPEED'] and (st - a) % W == 0:
                    stat.append(st)
                l = lab[sl]; l = l[l >= 0]
                maj, pur = -1, 0.0
                if len(l):
                    cnt = np.bincount(l, minlength=3); maj = int(cnt.argmax()); pur = cnt[maj] / W
                recs.append((st, int(df.seg.iat[st]), float(df.timestamp.iat[st]), float(df.latitude.iat[st + W // 2]),
                             float(df.longitude.iat[st + W // 2]), float(np.nanmax(acc[sl])) if np.isfinite(acc[sl]).any()
                             else np.nan, float(sp.mean()), float(sp.std()),
                             float(sp.mean() * np.abs(YAW[sl]).mean()), float(np.abs(LONG[sl]).mean()), maj, pur))
        A = pd.DataFrame(recs, columns=['start', 'seg', 't_start', 'latitude', 'longitude', 'accuracy_m',
                                        'speed_mean', 'speed_std',
                                        'lat_acc', 'long_acc', 'label', 'purity'])
        return PROC, proc_names, A, np.array(stat, int)

    def _normalise(self, F, A):
        X = F.copy()
        ls = np.log(np.clip(A.speed_mean.to_numpy(), 0.5, None))
        for c in self.features:                            # speed normalisation (training slopes)
            if c in self.slopes:
                X[c] = c4._as_log(F, c) - self.slopes[c] * (ls - self.ref)
        moving = (A.speed_mean >= c4.CFG['MIN_SPEED']).to_numpy()
        ref = X.loc[moving, self.features]                 # per-trip robust normalisation (label-free)
        med, iqr = ref.median(), (ref.quantile(.75) - ref.quantile(.25)).replace(0, np.nan)
        Xn = ((X[self.features] - med) / iqr).fillna(0)
        return Xn.astype(np.float32), moving

    # ------------------------------------------------------------------ main
    def extract(self, phone_df):
        """Cleaning, noise handling, windows and features. Returns (windows A, features F, report, warnings)."""
        c4.CFG.update(self.cfg)
        warnings = []
        df, n_segments = self._clean(self._to_pipeline_frame(phone_df))
        if len(df) < c4.win_len():
            raise C4InputError('No continuous recording long enough for analysis (at least 4 s of IMU data '
                               'with GPS speed is needed).')
        PROC, names, A, stat = self._signals_and_windows(df)
        noise = c4.noise_profile(PROC, stat)
        if noise is None:
            warnings.append(f'Fewer than 3 stationary windows ({len(stat)}); the vehicle noise profile could not be '
                            'estimated and was not subtracted. Stopping briefly with the engine running at the start '
                            'of a survey fixes this.')
        chans = [names.index(f'{self.sensor}__{c}') for c in self.channels]
        F = c4.compute_features(PROC[:, chans], [names[i] for i in chans], A.start.to_numpy(),
                                noise=None if noise is None else noise[:, chans])
        F['s__speed_mean'], F['s__speed_std'] = A.speed_mean.to_numpy(), A.speed_std.to_numpy()
        report = {'segments': n_segments, 'windows': len(A), 'stationary_windows': len(stat),
                  'noise_profile_subtracted': noise is not None}
        return A, F, report, warnings

    def predict(self, phone_df):
        if self.model is None:
            raise RuntimeError('This predictor has no model (feature extraction only).')
        A, F, report, warnings = self.extract(phone_df)
        missing = [c for c in self.features if c not in F.columns]
        if missing:
            raise RuntimeError(f'Model expects features that the pipeline did not produce: {missing[:5]}')
        X, moving = self._normalise(F, A)
        if moving.sum() == 0:
            raise C4InputError('The vehicle never moved faster than 1 m/s; no road-condition assessment is possible.')
        if moving.sum() < 30:
            warnings.append(f'Only {int(moving.sum())} moving windows (< 30 s of driving above '
                            f'{c4.CFG["MIN_SPEED"]} m/s); per-trip normalisation is less reliable for short surveys.')

        proba = np.full((len(A), len(self.classes)), np.nan)
        proba[moving] = self.model.predict_proba(X.to_numpy()[moving])
        A['assessed'] = moving
        A['manoeuvre'] = ((A.lat_acc > c4.CFG['LAT_ACC_MAX']) | (A.long_acc > c4.CFG['LONG_ACC_MAX'])).to_numpy()
        for i, c in enumerate(self.classes):
            A[f'p_{c}'] = proba[:, i]
        filled = np.where(np.isnan(proba), -1, proba)
        A['road_condition'] = np.where(moving, np.array(self.classes)[filled.argmax(1)], None)
        A['confidence'] = np.where(moving, filled.max(1), np.nan)

        report.update(assessed_windows=int(moving.sum()), manoeuvre_windows=int((A.manoeuvre & A.assessed).sum()))
        lab_ok = A.assessed & (A.label >= 0) & (A.purity >= c4.CFG['MIN_PURITY'])
        if lab_ok.sum() > 0:
            from sklearn.metrics import accuracy_score, f1_score
            y = np.array(self.classes)[A.label[lab_ok].to_numpy()]
            p = A.road_condition[lab_ok].to_numpy()
            report['label_check'] = {'windows': int(lab_ok.sum()),
                                     'accuracy': round(float(accuracy_score(y, p)), 4),
                                     'macro_f1': round(float(f1_score(y, p, average='macro', labels=self.classes,
                                                                      zero_division=0)), 4)}
        return TripResult(windows=A.drop(columns=['start', 'purity']), report=report, warnings=warnings)


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def scale_arithmetic(imputer, scaler, n_features, seed=1):
    """Which float32 rounding this fitted StandardScaler uses (see PortableXGB)."""
    Xs = np.random.default_rng(seed).standard_normal((500, n_features)).astype(np.float32)
    ref = scaler.transform(imputer.transform(Xs))
    per_step = Xs.copy()
    per_step -= scaler.mean_
    per_step /= scaler.scale_
    if np.array_equal(per_step, ref):
        return 'float64_per_step'
    if np.array_equal((Xs - scaler.mean_.astype(np.float32)) / scaler.scale_.astype(np.float32), ref):
        return 'float32_params'
    raise ValueError('The scaler arithmetic of this model could not be reproduced.')


def save_portable(out_dir, meta, booster, impute_median, scale_mean, scale_std, arithmetic='float32_params'):
    """Write the portable model format read by Predictor: c4_model_portable.json + c4_model_portable.xgb.json."""
    os.makedirs(out_dir, exist_ok=True)
    booster_file = 'c4_model_portable.xgb.json'
    booster.save_model(os.path.join(out_dir, booster_file))
    m = {k: v for k, v in meta.items() if k != 'model'}
    m.update(booster_file=booster_file, impute_median=[float(v) for v in impute_median],
             scale_mean=[float(v) for v in scale_mean], scale_std=[float(v) for v in scale_std],
             scale_arithmetic=arithmetic)
    with open(os.path.join(out_dir, 'c4_model_portable.json'), 'w') as f:
        json.dump(m, f, indent=1, default=_plain)
    return out_dir


def _plain(o):
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'{type(o).__name__} is not JSON serialisable')
