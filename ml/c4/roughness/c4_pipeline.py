"""
c4_pipeline.py - shared pipeline for Component 4
Multi-Sensor ML Road Roughness Assessment and Cross-Vehicle-Driver Generalization
Sandanayaka S D I K - IT23222472

Used by notebooks 02-09. Upload this file to  MyDrive/PVS_Code/c4_pipeline.py
"""
import os
import re
import glob
import time
import warnings

import numpy as np
import pandas as pd
from scipy import signal, stats

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# 1. Paths and configuration
# =============================================================================
DRIVE = os.environ.get('C4_DRIVE', '/content/drive/MyDrive')


def raw_dir(n):        return os.path.join(DRIVE, f'PVS{n}')
def synced_path(n):    return os.path.join(DRIVE, f'PVS{n}', 'Synced_Data', f'PVS{n}_synchronized.csv')


PROCESSED_DIR    = os.path.join(DRIVE, 'PVS_Processed')
FEATURE_DIR      = os.path.join(DRIVE, 'PVS_Feature_Datasets_v2')
RESULTS_DIR      = os.path.join(DRIVE, 'PVS_Model_Results_v2')
OLD_FEATURE_DIR  = os.path.join(DRIVE, 'PVS_Feature_Datasets')        # features from the earlier pipeline

CFG = dict(
    FS=100.0,
    WINDOW_SEC=2.0, STEP_SEC=1.0,          # 2 s windows, 50 % overlap
    LABEL_MODE='worst',                    # 'worst' (road_condition), 'left' or 'right'
    MIN_PURITY=0.8, MIN_LABELED=0.5,
    MIN_SPEED=1.0, STOP_SPEED=0.3,         # m/s
    GAP_FACTOR=3.0, MAX_INTERP_GAP=5,
    GRAVITY_HZ=0.3, LOWPASS_HZ=40.0,
    HAMPEL_HALF=2, HAMPEL_K=10.0, HAMPEL_MAX_RUN=2, HAMPEL_SCALE_SEC=1.0, HAMPEL_SINGLE_AXIS=True,
    HAMPEL_REPLACE=False, ACC_FULL_SCALE=156.9, GYRO_FULL_SCALE=34.9,   # MPU-9250: 16 g, 2000 deg/s
    LAT_ACC_MAX=1.5, LONG_ACC_MAX=1.0,     # m/s^2
    BANDS=[(0.5, 3), (3, 8), (8, 16), (16, 30), (30, 45)],
    NPERSEG=128,
    N_STABLE_FEATURES=40, REDUNDANCY_CORR=0.95,
    ACC_UNIT='auto', GYRO_UNIT='auto', SPEED_UNIT='auto',
    MANEUVER_SENSOR='left_dashboard',
    DEPLOY_SENSOR='left_dashboard',
    RANDOM_SEED=42,
)

SIDES = ['left', 'right']
PLACEMENTS = ['dashboard', 'above_suspension', 'below_suspension']
SENSORS = [f'{s}_{p}' for s in SIDES for p in PLACEMENTS]
DASHBOARD = ['left_dashboard', 'right_dashboard']

RAW_CH    = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
AXIS_CH   = ['acc_dx', 'acc_dy', 'acc_dz', 'gyro_fx', 'gyro_fy', 'gyro_fz']
ROBUST_CH = ['acc_vert', 'acc_horiz', 'acc_mag', 'gyro_mag']
PROC_CH   = AXIS_CH + ROBUST_CH

RAW_COLS  = [f'{s}__{c}' for s in SENSORS for c in RAW_CH]
PROC_COLS = [f'{s}__{c}' for s in SENSORS for c in PROC_CH]

CLASSES = ['good', 'regular', 'bad']
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
PVS_META = {
    1: ('VW Saveiro', 'Driver 1', 1), 2: ('VW Saveiro', 'Driver 1', 2), 3: ('VW Saveiro', 'Driver 1', 3),
    4: ('Fiat Bravo', 'Driver 2', 1), 5: ('Fiat Bravo', 'Driver 2', 2), 6: ('Fiat Bravo', 'Driver 2', 3),
    7: ('Fiat Palio', 'Driver 3', 1), 8: ('Fiat Palio', 'Driver 3', 2), 9: ('Fiat Palio', 'Driver 3', 3),
}
SPEED_COLS = ['s__speed_mean', 's__speed_std']


CONFIG_FILE = os.path.join(PROCESSED_DIR, 'config_used.json')


def save_config():
    """Store the configuration used in notebook 03 so that notebooks 04-09 use exactly the same values."""
    import json
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(CFG, f, indent=2)
    print('Configuration saved ->', CONFIG_FILE)


def _load_config():
    import json
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            CFG.update(json.load(f))


_load_config()


def win_len():  return int(round(CFG['WINDOW_SEC'] * CFG['FS']))
def step_len(): return max(1, int(round(CFG['STEP_SEC'] * CFG['FS'])))
def nyq():      return CFG['FS'] / 2
def bands():    return [(lo, min(hi, 0.9 * nyq())) for lo, hi in CFG['BANDS'] if lo < 0.9 * nyq()]


def sensor_cols(sensor, kind):
    """Synced-CSV column names for one sensor, e.g. left_acc_x_dashboard."""
    side, placement = sensor.split('_', 1)
    return [f'{side}_{kind}_{a}_{placement}' for a in 'xyz']


def ensure_dirs():
    for d in (PROCESSED_DIR, FEATURE_DIR, RESULTS_DIR):
        os.makedirs(d, exist_ok=True)


# =============================================================================
# 2. Loading
# =============================================================================
def find_csv_file(pvs_path, filename):
    """Locate a raw PVS CSV file, also when it is nested in a folder of the same name."""
    direct = os.path.join(pvs_path, filename)
    if os.path.isfile(direct):
        return direct
    if os.path.isdir(direct):
        nested = os.path.join(direct, filename)
        if os.path.isfile(nested):
            return nested
    matches = glob.glob(os.path.join(pvs_path, '**', filename), recursive=True)
    matches = [m for m in matches if os.path.isfile(m)]
    if matches:
        return matches[0]
    raise FileNotFoundError(f'{filename} not found inside {pvs_path}')


def load_raw_label_extras(n, expected_rows):
    """Speed-bump flags and side-specific labels from the raw dataset_labels.csv (row-aligned with the synced file)."""
    try:
        lab = pd.read_csv(find_csv_file(raw_dir(n), 'dataset_labels.csv'))
    except FileNotFoundError:
        print(f'  PVS{n}: raw dataset_labels.csv not found -> speed bumps cannot be excluded')
        return None
    if len(lab) != expected_rows:
        print(f'  PVS{n}: raw labels have {len(lab)} rows, synced data {expected_rows} -> extras skipped')
        return None
    bump_cols = [c for c in lab.columns if 'speed_bump' in c.lower() and not c.lower().startswith('no_')]
    out = pd.DataFrame({'bump': lab[bump_cols].sum(axis=1).values > 0 if bump_cols else False})
    for side in SIDES:
        cols = [f'{k}_road_{side}' for k in CLASSES]
        if all(c in lab.columns for c in cols):
            out[f'label_{side}'] = np.where(lab[cols].sum(axis=1) > 0, lab[cols].values.argmax(1), -1)
    return out


def load_trip(n):
    df = pd.read_csv(synced_path(n))
    extras = load_raw_label_extras(n, len(df))
    df['bump'] = extras['bump'].values if extras is not None else False
    if CFG['LABEL_MODE'] == 'worst':
        df['label'] = df['road_condition'].map(CLASS_ID).fillna(-1).astype(int)
    else:
        col = f"label_{CFG['LABEL_MODE']}"
        if extras is None or col not in extras:
            raise ValueError(f'LABEL_MODE={CFG["LABEL_MODE"]} needs the raw labels file for PVS{n}')
        df['label'] = extras[col].values
    v, d, s = PVS_META[n]
    df['pvs'], df['vehicle'], df['driver'], df['scenario'] = n, v, d, s
    return df


def detect_units(df):
    acc = df[sensor_cols('left_dashboard', 'acc')].values
    gyro = df[sensor_cols('left_dashboard', 'gyro')].values
    acc_med = np.nanmedian(np.linalg.norm(acc, axis=1))
    gyro_p = np.nanpercentile(np.abs(gyro), 99.5)
    sp95 = np.nanpercentile(df['speed'].dropna(), 95)
    u = dict(
        acc=CFG['ACC_UNIT'] if CFG['ACC_UNIT'] != 'auto' else ('g' if acc_med < 3 else 'm/s2'),
        gyro=CFG['GYRO_UNIT'] if CFG['GYRO_UNIT'] != 'auto' else ('deg/s' if gyro_p > 3 else 'rad/s'),
        speed=CFG['SPEED_UNIT'] if CFG['SPEED_UNIT'] != 'auto' else ('km/h' if sp95 > 35 else 'm/s'),
        evidence=dict(acc_median_norm=round(float(acc_med), 3), gyro_p99_5=round(float(gyro_p), 3),
                      speed_p95=round(float(sp95), 3)))
    return u


def harmonise_units(df, units):
    for s in SENSORS:
        if units['acc'] == 'g':
            df[sensor_cols(s, 'acc')] *= 9.80665
        if units['gyro'] == 'deg/s':
            df[sensor_cols(s, 'gyro')] = np.deg2rad(df[sensor_cols(s, 'gyro')])
    if units['speed'] == 'km/h':
        df['speed'] = df['speed'] / 3.6
    return df


# =============================================================================
# 3. Cleaning and segmentation
# =============================================================================
def all_sensor_cols():
    return [c for s in SENSORS for k in ('acc', 'gyro') for c in sensor_cols(s, k)]


def clean_trip(df):
    sens = all_sensor_cols()
    n0 = len(df)
    df = df.sort_values('timestamp').drop_duplicates('timestamp', keep='first').reset_index(drop=True)
    n_dup = n0 - len(df)
    df[sens] = df[sens].interpolate(limit=CFG['MAX_INTERP_GAP'], limit_area='inside')
    df['speed'] = df['speed'].interpolate(limit_area='inside').bfill().ffill()
    dt = np.diff(df['timestamp'].values, prepend=df['timestamp'].values[0])
    med = np.median(dt[1:])
    bad = df[sens].isna().any(axis=1).values | ~np.isfinite(df['speed'].values)
    gap = dt > CFG['GAP_FACTOR'] * med
    df['seg'] = np.cumsum(gap | bad)
    df = df[~bad]
    keep = df.groupby('seg')['seg'].transform('size') >= win_len() + 2 * CFG['FS']
    df = df[keep].reset_index(drop=True)
    df['segment_id'] = df['pvs'].astype(str) + '_' + df['seg'].astype(str)
    report = dict(rows_in=n0, duplicates=n_dup, gaps=int(gap.sum()), bad_rows=int(bad.sum()),
                  rows_out=len(df), segments=df['segment_id'].nunique())
    return df.drop(columns='seg'), report


# =============================================================================
# 4. Noise handling (signal level)
# =============================================================================
def _sos(kind, cutoff, order=4):
    return signal.butter(order, cutoff, kind, fs=CFG['FS'], output='sos')


def _unit(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def hampel(x):
    """Replace isolated (<= HAMPEL_MAX_RUN samples) outliers > HAMPEL_K robust SD from the local median."""
    half, k = CFG['HAMPEL_HALF'], CFG['HAMPEL_K']
    scale_win = int(CFG['HAMPEL_SCALE_SEC'] * CFG['FS']) | 1
    s = pd.DataFrame(x)
    med = s.rolling(2 * half + 1, center=True, min_periods=1).median()
    dev = (s - med).abs()
    scale = 1.4826 * dev.rolling(scale_win, center=True, min_periods=1).median()
    floor = np.maximum(1.4826 * np.median(dev.values, axis=0), 1e-3 * np.std(x, axis=0) + 1e-9)
    out = dev.values > k * np.maximum(scale.values, floor)
    if CFG.get('HAMPEL_SINGLE_AXIS', True):
        # A road impact excites several axes of a sensor at the same time; a sensor glitch affects one axis.
        # Outliers that coincide (within +/-1 sample) with an outlier on another axis of the same
        # accelerometer or gyroscope are therefore kept as genuine road events.
        from scipy.ndimage import maximum_filter1d
        near = maximum_filter1d(out.astype(np.uint8), size=3, axis=0).astype(bool)
        for g in (slice(0, 3), slice(3, 6)):
            if out.shape[1] >= g.stop:
                n_axes = near[:, g].sum(1)
                out[:, g] &= (n_axes == 1)[:, None]
    spike = np.zeros_like(out)
    for j in range(out.shape[1]):
        o = out[:, j]
        if o.any():
            run_id = np.cumsum(np.r_[True, o[1:] != o[:-1]])
            run_len = np.bincount(run_id)[run_id]
            spike[:, j] = o & (run_len <= CFG['HAMPEL_MAX_RUN'])
    return np.where(spike, med.values, x), spike


def process_sensor(seg, sensor):
    """Raw channels and noise-handled channels for one sensor in one continuous segment."""
    A = seg[sensor_cols(sensor, 'acc')].values.astype(float)
    G = seg[sensor_cols(sensor, 'gyro')].values.astype(float)
    raw = np.hstack([A, G])
    x, spike = hampel(raw)                                            # 1. outliers (diagnostic)
    if not CFG.get('HAMPEL_REPLACE', False):
        # On PVS the statistical outliers occur mainly on Bad roads, i.e. they are road impacts, not glitches
        # (Notebook 03, Section 4). They are therefore kept; only physically impossible values
        # (at or beyond the sensor full-scale range) are treated as glitches and replaced by the local median.
        lim = np.array([CFG.get('ACC_FULL_SCALE', 16 * 9.80665)] * 3 + [CFG.get('GYRO_FULL_SCALE', np.deg2rad(2000))] * 3)
        sat = np.abs(raw) >= 0.98 * lim
        x = raw.copy()
        if sat.any():
            med = signal.medfilt(raw, kernel_size=[5, 1])
            x[sat] = med[sat]
    A2, G2 = x[:, :3], x[:, 3:]
    grav = signal.sosfiltfilt(_sos('low', CFG['GRAVITY_HZ']), A2, axis=0)   # 2. gravity / drift
    lp = _sos('low', min(CFG['LOWPASS_HZ'], 0.9 * nyq()))
    dyn = signal.sosfiltfilt(lp, A2 - grav, axis=0)                   # 3. low-pass
    Gf = signal.sosfiltfilt(lp, G2, axis=0)
    gu = _unit(grav)                                                  # 4. orientation-invariant channels
    vert = (dyn * gu).sum(1)
    horiz = np.linalg.norm(dyn - vert[:, None] * gu, axis=1)
    proc = np.column_stack([dyn, Gf, vert, horiz, np.linalg.norm(dyn, axis=1), np.linalg.norm(Gf, axis=1)])
    return raw, proc, spike.any(1)


def maneuver_signals(seg):
    """Yaw rate (turning) and longitudinal acceleration (braking/accelerating) for manoeuvre detection."""
    s = CFG['MANEUVER_SENSOR']
    A = seg[sensor_cols(s, 'acc')].values.astype(float)
    G = seg[sensor_cols(s, 'gyro')].values.astype(float)
    gu = _unit(signal.sosfiltfilt(_sos('low', CFG['GRAVITY_HZ']), A, axis=0))
    yaw = signal.sosfiltfilt(_sos('low', 1.0, 2), (G * gu).sum(1))
    v = signal.sosfiltfilt(_sos('low', 0.5, 2), seg['speed'].values.astype(float))
    return yaw, np.gradient(v) * CFG['FS']


def process_trip(df):
    n = len(df)
    RAW = np.zeros((n, len(RAW_COLS)), np.float32)
    PROC = np.zeros((n, len(PROC_COLS)), np.float32)
    SPIKE = np.zeros((n, len(SENSORS)), bool)
    YAW, LONG = np.zeros(n), np.zeros(n)
    for _, idx in df.groupby('segment_id', sort=False).indices.items():
        seg = df.iloc[idx]
        for k, s in enumerate(SENSORS):
            r, p, sp = process_sensor(seg, s)
            RAW[idx, 6 * k:6 * k + 6] = r
            PROC[idx, 10 * k:10 * k + 10] = p
            SPIKE[idx, k] = sp
        YAW[idx], LONG[idx] = maneuver_signals(seg)
    return RAW, PROC, SPIKE, YAW, LONG


# =============================================================================
# 5. Windowing
# =============================================================================
def make_windows(df, YAW, LONG):
    """All windows of a trip (moving or not); thresholds are applied later so they can be varied."""
    W, S = win_len(), step_len()
    lab, bump, spd = df['label'].values, df['bump'].values, df['speed'].values
    lat, lon, ts = df['latitude'].values, df['longitude'].values, df['timestamp'].values
    recs, stat = [], []
    n = int(df['pvs'].iat[0]); v, d, sc = PVS_META[n]
    for sid, idx in df.groupby('segment_id', sort=False).indices.items():
        a, b = idx[0], idx[-1] + 1
        for st in range(a, b - W + 1, S):
            sl = slice(st, st + W)
            sp = spd[sl]
            if sp.max() < CFG['STOP_SPEED'] and (st - a) % W == 0:
                stat.append(st)
            l = lab[sl]; l = l[l >= 0]
            maj, pur = -1, 0.0
            if len(l):
                c = np.bincount(l, minlength=3); maj = int(c.argmax()); pur = c[maj] / W
            recs.append((n, v, d, sc, st, sid, maj, pur, len(l) / W, bool(bump[sl].any()),
                         sp.mean(), sp.std(), sp.mean() * np.abs(YAW[sl]).mean(), np.abs(LONG[sl]).mean(),
                         lat[st + W // 2], lon[st + W // 2], ts[st]))
    A = pd.DataFrame(recs, columns=['pvs', 'vehicle', 'driver', 'scenario', 'start', 'segment_id', 'maj_label',
                                    'purity', 'labeled_frac', 'bump', 'speed_mean', 'speed_std', 'lat_acc',
                                    'long_acc', 'lat', 'lon', 't_start'])
    return A, np.array(stat, int)


def save_trip(n, RAW, PROC, SPIKE, stat_starts, df):
    np.savez(os.path.join(PROCESSED_DIR, f'PVS{n}_signals.npz'), RAW=RAW, PROC=PROC, SPIKE=SPIKE,
             STAT=stat_starts, label=df['label'].values.astype(np.int8))


def load_trip_signals(n):
    return np.load(os.path.join(PROCESSED_DIR, f'PVS{n}_signals.npz'))


# =============================================================================
# 6. Features
# =============================================================================
def windows_of(arr, starts):
    return arr[np.asarray(starts)[:, None] + np.arange(win_len())]


def welch(X):
    return signal.welch(X, fs=CFG['FS'], nperseg=min(CFG['NPERSEG'], win_len()), axis=1)


def noise_profile(PROC, stat_starts, min_windows=3):
    if len(stat_starts) < min_windows:
        return None
    return np.median(welch(windows_of(PROC, stat_starts).astype(np.float64))[1], axis=0)


def compute_features(arr, col_names, starts, noise=None, chunk=500):
    out = []
    for c0 in range(0, len(starts), chunk):
        X = windows_of(arr, starts[c0:c0 + chunk]).astype(np.float64)
        mean, std = X.mean(1), X.std(1)
        rms, peak = np.sqrt((X ** 2).mean(1)), np.abs(X).max(1)
        q75, q25 = np.percentile(X, [75, 25], axis=1)
        tf = {'mean': mean, 'std': std, 'rms': rms, 'iqr': q75 - q25, 'peak': peak,
              'range': X.max(1) - X.min(1), 'skew': stats.skew(X, axis=1),
              'kurt': stats.kurtosis(X, axis=1), 'crest': peak / (rms + 1e-9)}
        fr, P = welch(X)
        if noise is not None:
            P = np.clip(P - noise[None], 0, None)
        d = fr[1] - fr[0]
        use = fr >= 0.5
        tot = P[:, use].sum(1) * d + 1e-12
        ff = {}
        for lo, hi in bands():
            m = (fr >= lo) & (fr < hi)
            bp = P[:, m].sum(1) * d
            ff[f'bpabs_{lo:g}-{hi:g}'] = np.log10(bp + 1e-12)
            ff[f'bprel_{lo:g}-{hi:g}'] = bp / tot
        ff['logpow'] = np.log10(tot)
        Pu = P[:, use]
        ff['domfreq'] = fr[use][Pu.argmax(1)]
        ff['centroid'] = (fr[use][None, :, None] * Pu).sum(1) / (Pu.sum(1) + 1e-12)
        pn = Pu / (Pu.sum(1, keepdims=True) + 1e-12)
        ff['entropy'] = -(pn * np.log2(pn + 1e-12)).sum(1) / np.log2(Pu.shape[1])
        cols = {}
        for dom, dct in (('t', tf), ('f', ff)):
            for name, val in dct.items():
                for i, cn in enumerate(col_names):
                    cols[f'{dom}__{cn}__{name}'] = val[:, i].astype(np.float32)
        out.append(pd.DataFrame(cols))
    return pd.concat(out, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def parse(col):
    """'t__left_dashboard__acc_vert__rms' -> ('t', 'left_dashboard', 'acc_vert', 'rms')."""
    p = col.split('__')
    return (p[0], None, None, p[1]) if len(p) == 2 else tuple(p)


# =============================================================================
# 7. Dataset object with thresholds
# =============================================================================
class C4Data:
    def __init__(self, A, F_RAW, F_PROC):
        self.A = A.reset_index(drop=True)
        for c in self.A.columns:                      # plain numpy object arrays (pandas/pyarrow string dtypes)
            if not (pd.api.types.is_numeric_dtype(self.A[c]) or pd.api.types.is_bool_dtype(self.A[c])):
                self.A[c] = self.A[c].astype(object)
        self.F_RAW, self.F_PROC = F_RAW.reset_index(drop=True), F_PROC.reset_index(drop=True)
        for F in (self.F_RAW, self.F_PROC):
            F['s__speed_mean'] = self.A['speed_mean'].values
            F['s__speed_std'] = self.A['speed_std'].values
        self.set_thresholds()

    def set_thresholds(self, min_purity=None, min_speed=None, lat_acc_max=None, long_acc_max=None):
        A = self.A
        self.th = dict(min_purity=min_purity or CFG['MIN_PURITY'], min_speed=min_speed or CFG['MIN_SPEED'],
                       lat_acc_max=lat_acc_max or CFG['LAT_ACC_MAX'], long_acc_max=long_acc_max or CFG['LONG_ACC_MAX'])
        ok = (A.purity >= self.th['min_purity']) & (A.labeled_frac >= CFG['MIN_LABELED'])
        self.y = np.where(ok, A.maj_label, -1).astype(int)
        self.U = (A.speed_mean >= self.th['min_speed']).values                          # moving windows
        self.maneuver = ((A.lat_acc > self.th['lat_acc_max']) | (A.long_acc > self.th['long_acc_max'])).values
        self.L = self.U & (self.y >= 0) & ~A.bump.values                               # usable labelled windows
        self.E = self.L & ~self.maneuver                                               # common evaluation set
        self._splits = None
        return self

    def summary(self):
        A = self.A
        return pd.DataFrame({
            'all windows': A.groupby('vehicle').size(),
            'moving': pd.Series(self.U).groupby(A.vehicle.values).sum(),
            'labelled usable (L)': pd.Series(self.L).groupby(A.vehicle.values).sum(),
            'evaluation set (E)': pd.Series(self.E).groupby(A.vehicle.values).sum(),
            'bump windows': (A.bump & self.U).groupby(A.vehicle).sum(),
            'manoeuvre %': (pd.Series(self.maneuver & self.L).groupby(A.vehicle.values).sum()
                            / pd.Series(self.L).groupby(A.vehicle.values).sum() * 100).round(1)})

    # ---------------------------------------------------------------- splits
    def splits(self):
        if self._splits is not None:
            return self._splits
        A = self.A
        Lidx = np.where(self.L)[0]
        tr, te = train_test_split(Lidx, test_size=0.2, stratify=self.y[Lidx], random_state=CFG['RANDOM_SEED'])
        m = lambda idx: np.isin(np.arange(len(A)), idx)
        veh, sc = sorted(A.vehicle.unique()), sorted(A.scenario.unique())
        self._splits = {
            'Random': [('random 80/20', m(tr), m(te))],
            'LOVO': [(f'test={v}', (A.vehicle != v).values, (A.vehicle == v).values) for v in veh],
            'LORO': [(f'test=route{r}', (A.scenario != r).values, (A.scenario == r).values) for r in sc],
            'LOVO+LORO': [(f'test={v}/route{r}', ((A.vehicle != v) & (A.scenario != r)).values,
                           ((A.vehicle == v) & (A.scenario == r)).values) for v in veh for r in sc],
        }
        return self._splits


# =============================================================================
# 8. Vehicle-driver dependency handling (feature level)
# =============================================================================
AMP_STATS = ('std', 'rms', 'iqr', 'peak', 'range')
MAG_CH = ('acc_horiz', 'acc_mag', 'gyro_mag')


def is_amp(col):
    dom, _, ch, st = parse(col)
    if dom == 't':
        return st in AMP_STATS or (st == 'mean' and ch in MAG_CH)
    return dom == 'f' and (st == 'logpow' or st.startswith('bpabs'))


def _as_log(X, c):
    v = X[c].values.astype(float)
    return v if c.startswith('f__') else np.log(np.clip(v, 0, None) + 1e-9)


def _log_speed(data):
    return np.log(np.clip(data.A.speed_mean.values, 0.5, None))


def fit_speed_slopes(data, X, tr):
    """log(feature) ~ b * log(speed), fitted within class on training windows only."""
    ls, y = _log_speed(data)[tr], data.y[tr]
    xs = ls - pd.Series(ls).groupby(y).transform('mean').values
    den = (xs ** 2).sum()
    slopes = {}
    for c in [c for c in X.columns if is_amp(c)]:
        v = _as_log(X, c)[tr]
        ok = np.isfinite(v)
        vy = v - pd.Series(v).groupby(y).transform('mean').values
        slopes[c] = float((xs[ok] * vy[ok]).sum() / (xs[ok] ** 2).sum()) if den > 0 and ok.any() else 0.0
    return slopes, float(np.mean(ls))


def apply_speed_norm(data, X, slopes, ref):
    X = X.copy()
    ls = _log_speed(data)
    for c, b in slopes.items():
        X[c] = (_as_log(X, c) - b * (ls - ref)).astype(np.float32)
    return X.drop(columns=[c for c in SPEED_COLS if c in X.columns])


def trip_normalize(data, X):
    """Robust per-trip standardisation using label-free statistics of all moving windows of the trip."""
    cols = [c for c in X.columns if not c.startswith('s__')]
    pvs = data.A.pvs.values
    ref = X.loc[data.U, cols].groupby(pvs[data.U])
    med, q25, q75 = ref.median(), ref.quantile(.25), ref.quantile(.75)
    iqr = (q75 - q25).replace(0, np.nan)
    Xn = X.copy()
    Xn[cols] = ((X[cols].values - med.loc[pvs].values) / iqr.loc[pvs].values).astype(np.float32)
    Xn[cols] = Xn[cols].fillna(0)
    return Xn


def stable_select(data, X, tr, k=None):
    """Features that separate the classes consistently in every training vehicle, with redundancy removal."""
    k = k or CFG['N_STABLE_FEATURES']
    y, veh = data.y[tr], data.A.vehicle.values[tr]
    Xtr = X.iloc[tr]
    Xtr = Xtr.fillna(Xtr.median())
    vehicles = [v for v in np.unique(veh) if len(np.unique(y[veh == v])) >= 2]
    common = sorted(set.intersection(*[set(np.unique(y[veh == v])) for v in vehicles]))
    ranks, orders = [], []
    for v in vehicles:
        m = veh == v
        Fv = np.nan_to_num(f_classif(Xtr[m], y[m])[0])
        ranks.append(stats.rankdata(Fv) / len(Fv))
        orders.append(Xtr[m].groupby(y[m]).median().reindex(common).rank().values)
    score = np.min(ranks, axis=0)
    consistent = (np.all([(o == orders[0]).all(axis=0) for o in orders], axis=0)
                  if len(orders) > 1 else np.ones(len(score), bool))
    order = np.lexsort((-score, ~consistent))
    cand = [X.columns[i] for i in order[:4 * k]]
    corr = Xtr[cand].corr().abs().fillna(0)
    sel = []
    for c in cand:
        if all(corr.at[c, s] < CFG['REDUNDANCY_CORR'] for s in sel):
            sel.append(c)
        if len(sel) == k:
            break
    return sel


# =============================================================================
# 9. Configurations
# =============================================================================
def make_config(src, sensors, channels, drop, man=False, spd=False, trip=False, sel=False, k=None):
    return dict(src=src, sensors=list(sensors), channels=list(channels), drop=drop,
                man=man, spd=spd, trip=trip, sel=sel, k=k)


def ablation_configs(sensors=DASHBOARD, deploy_sensor=None):
    deploy_sensor = deploy_sensor or CFG['DEPLOY_SENSOR']
    return {
        'C0 Baseline (raw)':            make_config('raw',  sensors, RAW_CH,   'bprel'),
        'C1 + Noise handling':          make_config('proc', sensors, AXIS_CH,  'bprel'),
        'C2 + Invariant repr.':         make_config('proc', sensors, PROC_CH,  'bpabs'),
        'C3 + Manoeuvre filter':        make_config('proc', sensors, PROC_CH,  'bpabs', man=True),
        'C4 + Speed normalisation':     make_config('proc', sensors, PROC_CH,  'bpabs', man=True, spd=True),
        'C5 + Per-trip normalisation':  make_config('proc', sensors, PROC_CH,  'bpabs', man=True, spd=True, trip=True),
        'C6 + Stable selection (FULL)': make_config('proc', sensors, PROC_CH,  'bpabs', man=True, spd=True, trip=True, sel=True),
        'C7 Deployment (phone)':        make_config('proc', [deploy_sensor], ROBUST_CH, 'bpabs', man=True, spd=True, trip=True, sel=True),
    }


def config_columns(data, cfg):
    F = data.F_RAW if cfg['src'] == 'raw' else data.F_PROC
    cols = []
    for c in F.columns:
        dom, sen, ch, st = parse(c)
        if dom == 's' or (sen in cfg['sensors'] and ch in cfg['channels'] and not st.startswith(cfg['drop'])):
            cols.append(c)
    return F[cols]


def prepare(data, cfg, tr):
    X, info = config_columns(data, cfg), {}
    if cfg['spd']:
        slopes, ref = fit_speed_slopes(data, X, tr)
        X = apply_speed_norm(data, X, slopes, ref)
        info.update(slopes=slopes, ref=ref)
    if cfg['trip']:
        X = trip_normalize(data, X)
    if cfg['sel']:
        sel = stable_select(data, X, tr, cfg.get('k'))
        X = X[sel]
        info['selected'] = sel
    return X, info


def train_index(data, cfg, train_mask):
    m = train_mask & data.L
    if cfg['man']:
        m = m & ~data.maneuver
    return np.where(m)[0]


# =============================================================================
# 10. Models and experiments
# =============================================================================
def make_model(name):
    from xgboost import XGBClassifier
    pre = [SimpleImputer(strategy='median'), StandardScaler()]
    seed = CFG['RANDOM_SEED']
    if name == 'SVM':
        clf = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
    elif name == 'RF':
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight='balanced',
                                     n_jobs=-1, random_state=seed)
    elif name == 'XGB':
        clf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8,
                            colsample_bytree=0.8, objective='multi:softprob', num_class=3,
                            eval_metric='mlogloss', n_jobs=-1, random_state=seed)
    else:
        raise ValueError(name)
    return make_pipeline(*pre, clf)


def fit(name, X, y):
    m = make_model(name)
    if name == 'XGB':
        m.fit(X, y, xgbclassifier__sample_weight=compute_sample_weight('balanced', y))
    else:
        m.fit(X, y)
    return m


def metrics(y, p):
    f1c = f1_score(y, p, average=None, labels=[0, 1, 2], zero_division=0)
    return dict(accuracy=accuracy_score(y, p),
                precision=precision_score(y, p, average='macro', zero_division=0),
                recall=recall_score(y, p, average='macro', zero_division=0),
                macro_f1=f1_score(y, p, average='macro', zero_division=0),
                f1_good=f1c[0], f1_regular=f1c[1], f1_bad=f1c[2],
                recall_bad=recall_score(y, p, labels=[2], average='macro', zero_division=0))


def run(data, name, cfg, split_names, models, keep_predictions=False, verbose=True):
    rows, cms, preds = [], {}, []
    splits = data.splits()
    for sname in split_names:
        for fname, trm, tem in splits[sname]:
            tr = train_index(data, cfg, trm)
            te = np.where(tem & data.E)[0]
            if len(te) == 0 or len(np.unique(data.y[tr])) < 3:
                continue
            t0 = time.time()
            X, info = prepare(data, cfg, tr)
            for mname in models:
                model = fit(mname, X.values[tr], data.y[tr])
                p = model.predict(X.values[te])
                rows.append({'config': name, 'split': sname, 'fold': fname, 'model': mname,
                             'n_train': len(tr), 'n_test': len(te), 'n_features': X.shape[1],
                             **metrics(data.y[te], p)})
                cms[(name, sname, fname, mname)] = confusion_matrix(data.y[te], p, labels=[0, 1, 2])
                if keep_predictions:
                    allm = np.where(tem & data.U)[0]
                    preds.append(pd.DataFrame({'row': allm, 'config': name, 'split': sname, 'fold': fname,
                                               'model': mname, 'pred': model.predict(X.values[allm])}))
            if verbose:
                f1s = ', '.join(f"{r['model']}={r['macro_f1']:.3f}" for r in rows[-len(models):])
                print(f'  {name[:30]:30s} {sname:9s} {fname:24s} {f1s}  ({time.time() - t0:.0f}s)')
    out = pd.DataFrame(rows)
    if keep_predictions:
        return out, cms, (pd.concat(preds, ignore_index=True) if preds else pd.DataFrame())
    return out, cms


def degradation_table(df, order=None):
    g = df.groupby(['config', 'model', 'split'])['macro_f1']
    t = g.mean().unstack('split')
    if 'LOVO' in t:
        t['LOVO_std'] = g.std().unstack('split')['LOVO']
    if 'Random' in t and 'LOVO' in t:
        t['Degradation'] = t['Random'] - t['LOVO']
    t = t.round(3)
    if order is not None:
        t = t.reindex(order, level=0)
    return t


def vehicle_identifiability(data, cfg, n_max=4000):
    """Balanced accuracy of predicting the vehicle-driver from the features (trip-grouped CV). Chance = 0.33."""
    X, _ = prepare(data, {**cfg, 'sel': False}, np.where(data.L)[0])
    idx = np.where(data.E)[0]
    rng = np.random.default_rng(CFG['RANDOM_SEED'])
    idx = rng.choice(idx, min(n_max, len(idx)), replace=False)
    if len(np.unique(data.A.pvs.values[idx])) < 3:
        return np.nan
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=CFG['RANDOM_SEED'])
    m = make_pipeline(SimpleImputer(strategy='median'),
                      RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=CFG['RANDOM_SEED']))
    return float(cross_val_score(m, X.values[idx], data.A.vehicle.values[idx], groups=data.A.pvs.values[idx],
                                 cv=cv, scoring='balanced_accuracy').mean())


def load_data():
    A = pd.read_parquet(os.path.join(FEATURE_DIR, 'windows.parquet'))
    F_RAW = pd.read_parquet(os.path.join(FEATURE_DIR, 'features_raw.parquet'))
    F_PROC = pd.read_parquet(os.path.join(FEATURE_DIR, 'features_processed.parquet'))
    return C4Data(A, F_RAW, F_PROC)


def check_setup():
    ok = os.path.exists(synced_path(1))
    print('Drive folder      :', DRIVE)
    print('Synced PVS1 found :', ok)
    print('Processed folder  :', PROCESSED_DIR, os.path.exists(PROCESSED_DIR))
    print('Feature folder    :', FEATURE_DIR, os.path.exists(FEATURE_DIR))
    print('Results folder    :', RESULTS_DIR, os.path.exists(RESULTS_DIR))
    return ok
