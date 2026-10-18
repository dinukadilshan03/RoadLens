"""
C4 raw-sensor ingestion: contract validation, synchronisation and resampling.

Input files follow docs/interfaces.md ("Raw GPS CSV" and "Raw IMU CSV"):

    IMU  : timestamp_ms,ax_mps2,ay_mps2,az_mps2,gx_rads,gy_rads,gz_rads
           (or two files: timestamp_ms,ax_mps2,ay_mps2,az_mps2  and  timestamp_ms,gx_rads,gy_rads,gz_rads)
    GPS  : timestamp_ms,latitude,longitude,accuracy_m,speed_mps        (accuracy_m and speed_mps nullable)

timestamp_ms is an integer, relative to the survey capture start, and must use the same clock for all files.
The sensor manifest must declare that the accelerometer includes gravity: the orientation-invariant
channels used by the model are computed relative to the gravity direction. Any device axis frame is accepted.

The output is one table on a uniform 100 Hz grid (the sampling rate used in training) with acceleration,
angular rate, interpolated position and speed. Samples inside IMU gaps, and positions/speeds inside
GPS gaps longer than the configured tolerance, are left empty (NaN) rather than bridged.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

IMU_ACC = ['ax_mps2', 'ay_mps2', 'az_mps2']
IMU_GYRO = ['gx_rads', 'gy_rads', 'gz_rads']
GPS_COLS = ['timestamp_ms', 'latitude', 'longitude', 'accuracy_m', 'speed_mps']


class C4InputError(ValueError):
    """Invalid or unsupported input. Not retryable; the message is safe to show to the user."""


@dataclass
class IngestConfig:
    target_fs: float = 100.0          # Hz, sampling rate of the trained model
    min_native_fs: float = 50.0       # Hz, below this the 0.5-45 Hz bands cannot be estimated
    warn_native_fs: float = 90.0
    max_imu_gap_s: float = 0.2        # longer IMU gaps are not interpolated
    max_gps_gap_s: float = 5.0        # longer GPS gaps leave position and speed unknown
    low_accuracy_m: float = 25.0      # GPS fixes worse than this are flagged in segment quality
    gravity_check_range: tuple = (7.0, 12.5)   # m/s^2, median |acc| when gravity is included


@dataclass
class IngestReport:
    imu_rows: int = 0
    gps_rows: int = 0
    native_rate_hz: float = 0.0
    duration_s: float = 0.0
    imu_gaps: int = 0
    gps_gaps: int = 0
    speed_source: str = 'gps_speed_mps'
    warnings: list = field(default_factory=list)


def _read_csv(path, required, name):
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise C4InputError(f'{name}: file could not be read as CSV ({exc}).') from exc
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise C4InputError(f'{name}: missing required column(s) {missing}; found {list(df.columns)}.')
    for c in required:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def read_imu(imu_path=None, acc_path=None, gyro_path=None):
    """Return one IMU table (timestamp_ms + 6 channels) from a combined file or separate acc/gyro files."""
    if imu_path:
        imu = _read_csv(imu_path, ['timestamp_ms'] + IMU_ACC + IMU_GYRO, 'IMU CSV')
    elif acc_path and gyro_path:
        acc = _read_csv(acc_path, ['timestamp_ms'] + IMU_ACC, 'Accelerometer CSV')
        gyr = _read_csv(gyro_path, ['timestamp_ms'] + IMU_GYRO, 'Gyroscope CSV')
        acc = acc.dropna().sort_values('timestamp_ms').drop_duplicates('timestamp_ms')
        gyr = gyr.dropna().sort_values('timestamp_ms').drop_duplicates('timestamp_ms')
        if len(gyr) < 2:
            raise C4InputError('Gyroscope CSV contains fewer than 2 valid rows.')
        imu = acc.copy()
        for c in IMU_GYRO:                         # gyroscope onto the accelerometer clock
            imu[c] = np.interp(acc.timestamp_ms, gyr.timestamp_ms, gyr[c],
                               left=np.nan, right=np.nan)
    else:
        raise C4InputError('Provide an IMU CSV, or both an accelerometer CSV and a gyroscope CSV.')
    imu = imu[['timestamp_ms'] + IMU_ACC + IMU_GYRO].dropna()
    return imu.sort_values('timestamp_ms').drop_duplicates('timestamp_ms').reset_index(drop=True)


def read_gps(gps_path):
    gps = _read_csv(gps_path, ['timestamp_ms', 'latitude', 'longitude'], 'GPS CSV')
    for c in ('accuracy_m', 'speed_mps'):
        gps[c] = pd.to_numeric(gps[c], errors='coerce') if c in gps else np.nan
    gps = gps[GPS_COLS]
    ok = gps.timestamp_ms.notna() & gps.latitude.between(-90, 90) & gps.longitude.between(-180, 180)
    ok &= ~((gps.latitude == 0) & (gps.longitude == 0))            # missing fix written as 0,0
    return gps[ok].sort_values('timestamp_ms').drop_duplicates('timestamp_ms').reset_index(drop=True)


def check_manifest(manifest):
    """The manifest fields C4 depends on (docs/interfaces.md: axis frame, gravity, rate, device, clock)."""
    m = dict(manifest or {})
    if m.get('gravity_included') is not True:
        raise C4InputError('Sensor manifest must state "gravity_included": true. C4 needs the raw accelerometer '
                           '(including gravity) to find the vertical direction; "linear acceleration" is not supported.')
    return m


def _segment_speed_from_positions(t, lat, lon):
    """Speed (m/s) between consecutive distinct GPS fixes (haversine), assigned to interval midpoints."""
    la, lo = np.radians(lat), np.radians(lon)
    a = np.sin(np.diff(la) / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(np.diff(lo) / 2) ** 2
    d = 2 * 6371008.8 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    dt = np.diff(t)
    return (t[:-1] + t[1:]) / 2, d / np.where(dt > 0, dt, np.nan)


def _interp_with_gaps(grid, t, v, max_gap):
    """Linear interpolation that leaves NaN where the nearest source samples are more than max_gap apart."""
    ok = np.isfinite(v)
    t, v = t[ok], v[ok]
    if len(t) < 2:
        return np.full(len(grid), np.nan)
    out = np.interp(grid, t, v, left=np.nan, right=np.nan)
    i = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    out[(t[i] - t[i - 1]) > max_gap] = np.nan
    return out


def synchronise(imu, gps, cfg=None):
    """Resample IMU to the model rate and attach position and speed. Returns (table, report)."""
    cfg = cfg or IngestConfig()
    rep = IngestReport(imu_rows=len(imu), gps_rows=len(gps))
    if len(imu) < 10:
        raise C4InputError('IMU data contains fewer than 10 valid rows.')
    if len(gps) < 2:
        raise C4InputError('GPS data contains fewer than 2 valid fixes; road segments cannot be located.')

    t = imu.timestamp_ms.to_numpy(float) / 1000.0
    dt = np.diff(t)
    rep.native_rate_hz = float(1 / np.median(dt))
    rep.duration_s = float(t[-1] - t[0])
    if rep.native_rate_hz < cfg.min_native_fs:
        raise C4InputError(f'IMU sampling rate is {rep.native_rate_hz:.0f} Hz; at least {cfg.min_native_fs:.0f} Hz '
                           '(ideally 100 Hz or more) is required because the model uses vibration up to 45 Hz.')
    if rep.native_rate_hz < cfg.warn_native_fs:
        rep.warnings.append(f'IMU sampling rate is {rep.native_rate_hz:.0f} Hz (< 100 Hz); vibration above '
                            f'{rep.native_rate_hz / 2:.0f} Hz is missing and predictions may be less reliable.')

    g = np.median(np.linalg.norm(imu[IMU_ACC].to_numpy(), axis=1))
    lo, hi = cfg.gravity_check_range
    if not lo <= g <= hi:
        raise C4InputError(f'Median acceleration magnitude is {g:.2f} m/s^2, but about 9.81 m/s^2 is expected with '
                           'gravity included. Check the units (m/s^2, not g) and that raw acceleration is supplied.')
    gyro_p = float(np.percentile(np.abs(imu[IMU_GYRO].to_numpy()), 99.5))
    if gyro_p > 20:
        rep.warnings.append(f'Angular rate reaches {gyro_p:.1f} rad/s; check that the gyroscope is in rad/s, not deg/s.')

    grid = np.arange(t[0], t[-1], 1 / cfg.target_fs)
    out = pd.DataFrame({'timestamp': grid})
    for src, dst in zip(IMU_ACC + IMU_GYRO, ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']):
        out[dst] = _interp_with_gaps(grid, t, imu[src].to_numpy(float), cfg.max_imu_gap_s)
    rep.imu_gaps = int(np.sum(dt > cfg.max_imu_gap_s))
    if rep.imu_gaps:
        rep.warnings.append(f'{rep.imu_gaps} IMU gap(s) longer than {cfg.max_imu_gap_s} s; '
                            'the recording was split at these gaps.')

    tg = gps.timestamp_ms.to_numpy(float) / 1000.0
    out['latitude'] = _interp_with_gaps(grid, tg, gps.latitude.to_numpy(float), cfg.max_gps_gap_s)
    out['longitude'] = _interp_with_gaps(grid, tg, gps.longitude.to_numpy(float), cfg.max_gps_gap_s)
    out['accuracy_m'] = _interp_with_gaps(grid, tg, gps.accuracy_m.to_numpy(float), cfg.max_gps_gap_s)
    rep.gps_gaps = int(np.sum(np.diff(tg) > cfg.max_gps_gap_s))
    if rep.gps_gaps:
        rep.warnings.append(f'{rep.gps_gaps} GPS gap(s) longer than {cfg.max_gps_gap_s} s; position and speed are '
                            'unknown there and those parts are not assessed.')

    if gps.speed_mps.notna().mean() >= 0.5:
        out['speed'] = _interp_with_gaps(grid, tg, gps.speed_mps.to_numpy(float), cfg.max_gps_gap_s)
    else:
        rep.speed_source = 'derived_from_positions'
        rep.warnings.append('speed_mps is missing; speed was derived from consecutive GPS positions.')
        tm, v = _segment_speed_from_positions(tg, gps.latitude.to_numpy(float), gps.longitude.to_numpy(float))
        v = pd.Series(v).rolling(3, center=True, min_periods=1).median().to_numpy()
        out['speed'] = _interp_with_gaps(grid, tm, v, cfg.max_gps_gap_s)
    if float(np.nanpercentile(out.speed, 95)) > 60:
        raise C4InputError('speed_mps exceeds 60 m/s for more than 5 % of the trip; check that speed is in m/s.')
    return out, rep
