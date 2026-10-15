"""
Window policy: from overlapping 2 s window predictions to non-overlapping road segments.

The model classifies 2 s windows every 1 s (50 % overlap), as in training. To give every moment of the
survey at most one class, each window is represented by its central 1 s slice
[t_start + 0.5 s, t_start + 1.5 s). With a 1 s step these slices tile the recording without overlap.

Consecutive slices are merged into one segment while they
  * belong to the same continuous recording part (no IMU/GPS gap between them),
  * are adjacent in time,
  * have the same predicted class, and
  * the segment is shorter than max_segment_s.

Windows below the minimum speed are not assessed and produce no segment; they are counted in the summary.
Each segment has a LineString built from the interpolated GPS track inside its time interval. If fewer than
two distinct positions are available the geometry is null and geometry_null_reason says why.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SegmentPolicy:
    window_s: float = 2.0
    step_s: float = 1.0
    max_segment_s: float = 10.0
    geometry_step_s: float = 1.0       # one LineString vertex per second (plus both end points)
    low_accuracy_m: float = 25.0
    coord_decimals: int = 7


def _slices(W, policy):
    pad = (policy.window_s - policy.step_s) / 2
    s = W.copy()
    s['slice_start'] = s.t_start + pad
    s['slice_end'] = s.slice_start + policy.step_s
    return s


def merge_windows(W, policy=None):
    """W: assessed window table from Predictor. Returns one row per segment (times in seconds)."""
    policy = policy or SegmentPolicy()
    S = _slices(W[W.assessed].sort_values('t_start'), policy).reset_index(drop=True)
    if S.empty:
        return pd.DataFrame()
    new = np.ones(len(S), bool)
    start = S.slice_start.iat[0]
    for i in range(1, len(S)):
        adjacent = abs(S.slice_start.iat[i] - S.slice_end.iat[i - 1]) < 1e-6
        same = (S.road_condition.iat[i] == S.road_condition.iat[i - 1]) and (S.seg.iat[i] == S.seg.iat[i - 1])
        if adjacent and same and S.slice_end.iat[i] - start <= policy.max_segment_s + 1e-6:
            new[i] = False
        else:
            start = S.slice_start.iat[i]
    S['segment_no'] = np.cumsum(new) - 1
    g = S.groupby('segment_no', sort=True)
    seg = pd.DataFrame({
        'start_s': g.slice_start.min(), 'end_s': g.slice_end.max(),
        'roughness_class': g.road_condition.first(),
        'confidence': g.confidence.mean(),
        'n_windows': g.size(),
        'speed_mean_mps': g.speed_mean.mean(),
        'manoeuvre_fraction': g.manoeuvre.mean(),
        'gps_accuracy_max_m': g.accuracy_m.max(),
        **{f'p_{c}': g[f'p_{c}'].mean() for c in ('good', 'regular', 'bad') if f'p_{c}' in S},
    }).reset_index(drop=True)
    return seg


def linestring(track, start_s, end_s, policy=None):
    """GeoJSON LineString ([lon, lat]) of the GPS track between start_s and end_s, or (None, reason)."""
    policy = policy or SegmentPolicy()
    t = track.timestamp.to_numpy()
    m = (t >= start_s) & (t <= end_s)
    if not m.any():
        return None, 'no_gps_in_interval'
    sub = track.loc[m, ['timestamp', 'longitude', 'latitude']].dropna()
    if sub.empty:
        return None, 'gps_gap'
    ts = sub.timestamp.to_numpy()
    keep = np.unique(np.r_[0, np.searchsorted(ts, np.arange(ts[0], ts[-1], policy.geometry_step_s)), len(ts) - 1])
    pts = np.round(sub[['longitude', 'latitude']].to_numpy()[keep], policy.coord_decimals)
    pts = pts[np.r_[True, np.any(np.diff(pts, axis=0) != 0, axis=1)]]          # drop repeated vertices
    if len(pts) < 2:
        return None, 'stationary_or_single_position'
    return {'type': 'LineString', 'coordinates': pts.tolist()}, None
