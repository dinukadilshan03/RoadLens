"""
Compare app predictions with reference labels (e.g. the PVS labels of a test trip).

Labels file (optional upload, never needed for inference):
    labels.csv   timestamp_ms,road_condition        road_condition in good/regular/bad or empty

Window level (the metric reported in the research): every assessed, non-manoeuvre 2 s window whose
majority label covers >= 80 % of the window (label purity, as in training) is compared with the predicted
class. Segment level (for the map): each segment gets the majority label of its interval and its share.
"""
import numpy as np
import pandas as pd

from .inputs import C4InputError

CLASSES = ['good', 'regular', 'bad']


def read_labels(path):
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise C4InputError(f'Labels CSV could not be read ({exc}).') from exc
    if not {'timestamp_ms', 'road_condition'} <= set(df.columns):
        raise C4InputError('Labels CSV needs the columns timestamp_ms,road_condition.')
    df = df[['timestamp_ms', 'road_condition']].copy()
    df['timestamp_ms'] = pd.to_numeric(df.timestamp_ms, errors='coerce')
    df['road_condition'] = df.road_condition.astype('string').str.strip().str.lower()
    bad = df.road_condition.notna() & ~df.road_condition.isin(CLASSES)
    if bad.any():
        raise C4InputError(f'Labels CSV contains unknown classes {sorted(df.road_condition[bad].unique())[:5]}; '
                           'use good, regular or bad.')
    df = df.dropna(subset=['timestamp_ms']).sort_values('timestamp_ms')
    if df.road_condition.notna().sum() == 0:
        raise C4InputError('Labels CSV contains no labels.')
    return df.reset_index(drop=True)


def _majority(t, lab, start_ms, end_ms):
    i0, i1 = np.searchsorted(t, start_ms), np.searchsorted(t, end_ms)
    s = pd.Series(lab[i0:i1]).dropna()
    if s.empty:
        return None, 0.0, 0
    vc = s.value_counts()
    return str(vc.index[0]), float(vc.iloc[0] / max(i1 - i0, 1)), int(i1 - i0)


def evaluate(windows, records, labels, window_ms=2000, min_purity=0.8):
    """windows: c4_windows.csv table; records: segment records; labels: read_labels() output."""
    t = labels.timestamp_ms.to_numpy(float)
    lab = labels.road_condition.to_numpy(object)
    rate = 1000 / np.median(np.diff(t)) if len(t) > 1 else 100
    need = 0.5 * window_ms / 1000 * rate                       # at least half of the window labelled
    y_true, y_pred = [], []
    W = windows[windows.assessed.astype(bool) & ~windows.manoeuvre.astype(bool)]
    for r in W.itertuples():
        c, share, n = _majority(t, lab, r.window_start_ms, r.window_start_ms + window_ms)
        if c is None or n < need or share < min_purity:
            continue
        y_true.append(c)
        y_pred.append(r.road_condition)
    seg = {}
    for rec in records:
        c, share, _ = _majority(t, lab, rec['interval_start_ms'], rec['interval_end_ms'])
        seg[rec['record_id']] = {'true_class': c, 'share': round(share, 3)}
    if not y_true:
        return {'windows': 0, 'segments': seg,
                'note': 'No assessed window overlaps the labels with enough purity; check the label timestamps.'}
    yt, yp = np.array(y_true), np.array(y_pred)
    cm = [[int(((yt == a) & (yp == b)).sum()) for b in CLASSES] for a in CLASSES]
    per = {}
    for c in CLASSES:
        tp = int(((yt == c) & (yp == c)).sum())
        p_n, t_n = int((yp == c).sum()), int((yt == c).sum())
        prec = tp / p_n if p_n else 0.0
        rec_ = tp / t_n if t_n else 0.0
        per[c] = {'precision': round(prec, 4), 'recall': round(rec_, 4),
                  'f1': round(2 * prec * rec_ / (prec + rec_), 4) if prec + rec_ else 0.0, 'support': t_n}
    present = [c for c in CLASSES if per[c]['support'] or (yp == c).any()]
    return {
        'windows': len(yt),
        'accuracy': round(float((yt == yp).mean()), 4),
        'macro_f1': round(float(np.mean([per[c]['f1'] for c in present])), 4),
        'per_class': per,
        'confusion_matrix': {'labels': CLASSES, 'rows_true_cols_pred': cm},
        'segments': seg,
        'method': f'2 s windows, assessed and without manoeuvre, label purity >= {min_purity:.0%}',
    }
