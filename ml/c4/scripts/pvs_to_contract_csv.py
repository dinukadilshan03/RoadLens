"""
Convert one synchronised PVS trip into the C4 contract files, so the adapter can be checked on real data.

Only the left dashboard sensor is exported (the deployed smartphone-like position). GPS is reduced to
1 Hz fixes, as a phone would record. The PVS road labels are written to labels.csv for evaluation only;
they are not part of the contract.

Usage (Colab, after mounting Drive):
    python -m ml.c4.scripts.pvs_to_contract_csv --pvs 7 --out /content/pvs7_contract
then
    python -m ml.c4.scripts.check_on_pvs --contract /content/pvs7_contract --model-dir <model folder>
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from ml.c4.roughness import c4_pipeline as c4


def convert(n, out_dir, synced_csv=None):
    df = pd.read_csv(synced_csv or c4.synced_path(n))
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    units = c4.detect_units(df)
    df = c4.harmonise_units(df, units)
    t0 = df.timestamp.iat[0]
    ms = np.round((df.timestamp - t0) * 1000).astype('int64')
    acc, gyr = c4.sensor_cols('left_dashboard', 'acc'), c4.sensor_cols('left_dashboard', 'gyro')
    imu = pd.DataFrame({'timestamp_ms': ms})
    for src, dst in zip(acc + gyr, ['ax_mps2', 'ay_mps2', 'az_mps2', 'gx_rads', 'gy_rads', 'gz_rads']):
        imu[dst] = df[src].round(5)
    # one GPS fix per second (the PVS GPS is 1 Hz, repeated on the 100 Hz grid)
    sec = (ms // 1000)
    first = ~sec.duplicated()
    gps = pd.DataFrame({'timestamp_ms': ms[first], 'latitude': df.latitude[first].round(7),
                        'longitude': df.longitude[first].round(7), 'accuracy_m': np.nan,
                        'speed_mps': df.speed[first].round(3)})
    lab = df['road_condition'] if 'road_condition' in df else pd.Series(np.nan, index=df.index)
    os.makedirs(out_dir, exist_ok=True)
    imu.dropna().to_csv(os.path.join(out_dir, 'imu.csv'), index=False)
    gps.to_csv(os.path.join(out_dir, 'gps.csv'), index=False)
    pd.DataFrame({'timestamp_ms': ms, 'road_condition': lab}).to_csv(os.path.join(out_dir, 'labels.csv'), index=False)
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump({'schema_version': '1.0', 'device': f'PVS{n} MPU-9250 left dashboard', 'axis_frame': 'device',
                   'gravity_included': True, 'sample_rate_hz': 100, 'gps_rate_hz': 1,
                   'clock_origin': 'capture_start', 'source_units': units}, f, indent=2, default=str)
    return out_dir


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pvs', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--synced-csv', default=None, help='path of PVS<n>_synchronized.csv (default: Drive layout)')
    a = p.parse_args()
    print('Contract files written to', convert(a.pvs, a.out, a.synced_csv))
