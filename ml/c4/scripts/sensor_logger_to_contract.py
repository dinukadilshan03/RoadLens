"""
Convert a Sensor Logger (Android/iOS app by Kelvin Choi) export into the C4 contract files.

Input: the unzipped export folder (or the .zip itself) containing
    Metadata.csv            recording epoch time, device, sampling periods
    TotalAcceleration.csv   accelerometer INCLUDING gravity, m/s^2   (time ns, seconds_elapsed, z, y, x)
    Gyroscope.csv           angular rate, rad/s                       (time ns, seconds_elapsed, z, y, x)
    Location.csv            GPS fixes: latitude, longitude, horizontalAccuracy (m), speed (m/s)

Output (same formats as docs/interfaces.md):
    acc.csv   timestamp_ms,ax_mps2,ay_mps2,az_mps2
    gyro.csv  timestamp_ms,gx_rads,gy_rads,gz_rads
    gps.csv   timestamp_ms,latitude,longitude,accuracy_m,speed_mps
    manifest.json

timestamp_ms is relative to the recording start (Metadata "recording epoch time"); samples before the
start are dropped. "Accelerometer.csv" from Sensor Logger is gravity-removed (linear acceleration) and is
therefore NOT used.

Usage:  python -m ml.c4.scripts.sensor_logger_to_contract <export folder or .zip> --out <folder>
"""
import argparse
import json
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd


def _read(folder, name, required=True):
    p = os.path.join(folder, name)
    if not os.path.exists(p):
        if required:
            raise FileNotFoundError(f'{name} not found in {folder}. Enable this sensor in Sensor Logger.')
        return None
    return pd.read_csv(p)


def convert(src, out_dir):
    if zipfile.is_zipfile(src):
        tmp = tempfile.mkdtemp(prefix='sensorlogger_')
        with zipfile.ZipFile(src) as z:
            names = [n for n in z.namelist() if n.endswith('.csv') and '..' not in n and not n.startswith('/')]
            z.extractall(tmp, members=names)
        src = tmp
    meta = _read(src, 'Metadata.csv').iloc[0]
    t0_ms = float(meta['recording epoch time'])

    def rel_ms(df):
        return np.round(df['time'].astype('float64') / 1e6 - t0_ms).astype('int64')

    acc = _read(src, 'TotalAcceleration.csv')
    gyr = _read(src, 'Gyroscope.csv')
    loc = _read(src, 'Location.csv')

    a = pd.DataFrame({'timestamp_ms': rel_ms(acc), 'ax_mps2': acc.x, 'ay_mps2': acc.y, 'az_mps2': acc.z})
    g = pd.DataFrame({'timestamp_ms': rel_ms(gyr), 'gx_rads': gyr.x, 'gy_rads': gyr.y, 'gz_rads': gyr.z})
    speed = loc['speed'].where(loc['speed'] >= 0) if 'speed' in loc else np.nan
    acc_m = loc['horizontalAccuracy'] if 'horizontalAccuracy' in loc else np.nan
    p = pd.DataFrame({'timestamp_ms': rel_ms(loc), 'latitude': loc.latitude, 'longitude': loc.longitude,
                      'accuracy_m': acc_m, 'speed_mps': speed})
    a, g, p = (d[d.timestamp_ms >= 0].drop_duplicates('timestamp_ms') for d in (a, g, p))

    os.makedirs(out_dir, exist_ok=True)
    a.to_csv(os.path.join(out_dir, 'acc.csv'), index=False)
    g.to_csv(os.path.join(out_dir, 'gyro.csv'), index=False)
    p.to_csv(os.path.join(out_dir, 'gps.csv'), index=False)
    rate = 1000 / np.median(np.diff(a.timestamp_ms)) if len(a) > 1 else None
    manifest = {
        'schema_version': '1.0', 'device': str(meta.get('device name', 'unknown')),
        'platform': f"{meta.get('platform', '')} {meta.get('platform version', '')}".strip(),
        'app': f"Sensor Logger {meta.get('appVersion', '')}".strip(),
        'axis_frame': 'android_device' if str(meta.get('platform')) == 'android' else 'device',
        'gravity_included': True, 'sample_rate_hz': round(rate, 1) if rate else None,
        'clock_origin': 'recording_start', 'capture_start_at_epoch_ms': int(t0_ms),
        'timezone': str(meta.get('recording timezone', '')),
    }
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    return {'out_dir': out_dir, 'acc_rows': len(a), 'gyro_rows': len(g), 'gps_rows': len(p),
            'duration_s': round((a.timestamp_ms.iloc[-1] - a.timestamp_ms.iloc[0]) / 1000, 1), 'rate_hz': manifest['sample_rate_hz']}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('src', help='Sensor Logger export folder or .zip')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    print(json.dumps(convert(a.src, a.out), indent=2))
