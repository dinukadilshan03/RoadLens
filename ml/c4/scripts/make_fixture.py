"""
Deterministic synthetic contract fixture for C4 (no real survey data is committed).

A 75 s drive with a phone lying on the dashboard, tilted so that no device axis is vertical:
    0-10 s   stationary, engine idling (noise-profile windows)
    10-75 s  driving at 7-11 m/s over three road types: good (smooth), regular (moderate), bad (strong impacts)

Files written to the output folder (contract formats from docs/interfaces.md):
    imu.csv        timestamp_ms,ax_mps2,ay_mps2,az_mps2,gx_rads,gy_rads,gz_rads   (100 Hz)
    gps.csv        timestamp_ms,latitude,longitude,accuracy_m,speed_mps            (1 Hz)
    manifest.json  sensor manifest
    truth.csv      start_ms,end_ms,road_type (what was simulated; for tests only)

Usage:  python -m ml.c4.scripts.make_fixture --out ml/c4/fixtures/synthetic_drive
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

FS = 100
PLAN = [(0, 10, 'stationary'), (10, 30, 'good'), (30, 45, 'regular'), (45, 60, 'bad'), (60, 75, 'good')]
AMP = {'stationary': 0.0, 'good': 0.12, 'regular': 0.45, 'bad': 1.3}
START = (6.9271, 79.8612)          # Colombo


def _rotation(roll, pitch):
    cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return Ry @ Rx


def make_fixture(out_dir, seed=0):
    rng = np.random.default_rng(seed)
    n = PLAN[-1][1] * FS
    t = np.arange(n) / FS
    kind = np.empty(n, object)
    for a, b, k in PLAN:
        kind[a * FS:b * FS] = k
    moving = kind != 'stationary'

    speed = np.where(moving, 9 + 2 * np.sin(2 * np.pi * t / 23), 0.0)
    ramp = np.clip((t - 10) / 3, 0, 1)
    speed *= ramp

    # vertical road vibration in the vehicle frame: band-limited noise + impacts, scaled by road type
    amp = np.array([AMP[k] for k in kind])
    vib = np.convolve(rng.standard_normal(n), np.hanning(7) / np.hanning(7).sum(), 'same')
    vib = vib / vib.std() * amp * np.sqrt(np.maximum(speed, 0.5) / 9)
    bad = kind == 'bad'
    hits = np.flatnonzero(bad & (rng.random(n) < 0.02))
    for h in hits:
        L = min(15, n - h)
        vib[h:h + L] += rng.choice([-1, 1]) * 6 * np.exp(-np.arange(L) / 4) * np.sin(np.arange(L) * 1.3)
    idle = 0.03 * np.sin(2 * np.pi * 28 * t)                       # engine idle line, all the time
    acc_v = np.c_[0.08 * vib + 0.02 * rng.standard_normal(n),
                  0.10 * vib + 0.02 * rng.standard_normal(n),
                  9.80665 + vib + idle + 0.02 * rng.standard_normal(n)]
    gyro_v = np.c_[0.04 * vib, 0.03 * vib, 0.01 * np.sin(2 * np.pi * t / 30) * moving] + 0.002 * rng.standard_normal((n, 3))

    R = _rotation(np.deg2rad(8), np.deg2rad(-14))                 # phone tilted on the dashboard
    acc, gyro = acc_v @ R, gyro_v @ R

    imu = pd.DataFrame({'timestamp_ms': np.round(t * 1000).astype(int)})
    for i, c in enumerate(['ax_mps2', 'ay_mps2', 'az_mps2']):
        imu[c] = np.round(acc[:, i], 4)
    for i, c in enumerate(['gx_rads', 'gy_rads', 'gz_rads']):
        imu[c] = np.round(gyro[:, i], 5)

    dist = np.cumsum(speed) / FS
    heading = np.deg2rad(35)
    lat = START[0] + dist * np.cos(heading) / 111_320
    lon = START[1] + dist * np.sin(heading) / (111_320 * np.cos(np.deg2rad(START[0])))
    g = np.arange(0, n, FS)
    gps = pd.DataFrame({'timestamp_ms': imu.timestamp_ms.to_numpy()[g] + 37,      # GPS clock not aligned to IMU
                        'latitude': np.round(lat[g], 7), 'longitude': np.round(lon[g], 7),
                        'accuracy_m': np.round(4 + rng.random(len(g)) * 2, 1),
                        'speed_mps': np.round(speed[g], 2)})

    os.makedirs(out_dir, exist_ok=True)
    imu.to_csv(os.path.join(out_dir, 'imu.csv'), index=False)
    gps.to_csv(os.path.join(out_dir, 'gps.csv'), index=False)
    pd.DataFrame([(a * 1000, b * 1000, k) for a, b, k in PLAN], columns=['start_ms', 'end_ms', 'road_type']) \
        .to_csv(os.path.join(out_dir, 'truth.csv'), index=False)
    manifest = {'schema_version': '1.0', 'device': 'synthetic-phone', 'platform': 'synthetic',
                'axis_frame': 'device', 'gravity_included': True, 'sample_rate_hz': FS,
                'gps_rate_hz': 1, 'clock_origin': 'capture_start', 'mount': 'dashboard'}
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    return out_dir


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'synthetic_drive'))
    p.add_argument('--seed', type=int, default=0)
    print('Fixture written to', make_fixture(p.parse_args().out, p.parse_args().seed))
