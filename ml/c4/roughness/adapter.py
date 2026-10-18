"""
C4 trusted inference adapter (docs/interfaces.md, "Trusted inference adapter boundary").

    run(request: C4Request, context: RunContext) -> C4Result

The adapter only reads the input files named in the request and writes files into context.output_dir.
It never touches the database: the orchestration service uploads the artifacts and commits the records.

Stages (reported through context.progress):
    1 validate   - manifest and CSV contract, units, sampling rate
    2 sync       - IMU resampled to 100 Hz, GPS position/speed interpolated within the gap tolerance
    3 features   - noise handling, 2 s windows, invariant-channel features (same code as training)
    4 predict    - speed + per-trip normalisation, classifier on moving windows
    5 segments   - window policy (central 1 s slices, merged up to 10 s), LineStrings, records

Output artifacts (relative to output_dir):
    c4_segments.jsonl      one result record per segment (the payload committed by the platform)
    c4_segments.geojson    the same segments as a FeatureCollection (map preview / export)
    c4_windows.csv         every 2 s window with probabilities (diagnostics, not committed as records)
    c4_run_summary.json    input report, warnings, counts, model provenance

Failure behaviour:
    C4InputError  - the input is invalid or cannot be assessed (not retryable; message is user-safe)
    C4Cancelled   - context.is_cancelled() became true between stages
    other errors  - internal failure; the worker should mark the run failed and keep the log
"""
import argparse
import json
import math
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from . import __version__
from .inputs import (
    C4InputError,
    IngestConfig,
    check_manifest,
    read_gps,
    read_imu,
    synchronise,
)
from .predictor import Predictor
from .segments import SegmentPolicy, linestring, merge_windows

SCHEMA_VERSION = '1.0'
COMPONENT = 'C4'
CONFIG_VERSION = 'c4-ingest-1.0+segments-1.0'


class C4Cancelled(RuntimeError):
    """The run was cancelled through the run context."""


class RunContext(Protocol):
    output_dir: Path

    def progress(self, stage: str, completed: int | None = None, total: int | None = None) -> None: ...

    def is_cancelled(self) -> bool: ...


@dataclass
class LocalRunContext:
    """Simple context for scripts and tests."""
    output_dir: Path
    verbose: bool = False
    stages: list = field(default_factory=list)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def progress(self, stage, completed=None, total=None):
        self.stages.append((stage, completed, total))
        if self.verbose:
            print(f'[C4] {stage} {completed}/{total}' if total else f'[C4] {stage}')

    def is_cancelled(self):
        return False


@dataclass
class C4Request:
    survey_id: str
    gps_path: str
    manifest: dict
    model_dir: str
    imu_path: str | None = None
    acc_path: str | None = None
    gyro_path: str | None = None
    external_run_id: str | None = None
    upstream: list = field(default_factory=list)          # C4 is independent of C1-C3; normally empty
    ingest: IngestConfig = field(default_factory=IngestConfig)
    policy: SegmentPolicy = field(default_factory=SegmentPolicy)


@dataclass
class C4Result:
    records: list
    artifacts: dict                      # name -> path relative to output_dir
    summary: dict
    warnings: list


_predictors = {}


def _predictor(model_dir):
    key = os.path.abspath(model_dir)
    if key not in _predictors:
        _predictors[key] = Predictor(key)
    return _predictors[key]


def _num(v, nd=4):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), nd)


def _ms(seconds):
    return round(seconds * 1000)


def run(request: C4Request, context: RunContext) -> C4Result:
    def stage(name, done=None, total=5):
        if context.is_cancelled():
            raise C4Cancelled(f'Cancelled before stage "{name}".')
        context.progress(name, done, total)

    stage('validate', 0)
    manifest = check_manifest(request.manifest)
    imu = read_imu(request.imu_path, request.acc_path, request.gyro_path)
    gps = read_gps(request.gps_path)

    stage('sync', 1)
    track, ingest = synchronise(imu, gps, request.ingest)
    warnings = list(ingest.warnings)
    declared = manifest.get('sample_rate_hz')
    if declared and abs(float(declared) - ingest.native_rate_hz) > 0.1 * float(declared):
        warnings.append(f'Manifest declares {declared} Hz but timestamps show {ingest.native_rate_hz:.1f} Hz; '
                        'the measured rate was used.')

    stage('features', 2)
    predictor = _predictor(request.model_dir)
    stage('predict', 3)
    trip = predictor.predict(track)
    warnings += trip.warnings
    W = trip.windows

    stage('segments', 4)
    seg = merge_windows(W, request.policy)
    run_id = request.external_run_id or str(uuid.uuid4())
    producer = {'model_version': predictor.version, 'code_version': __version__, 'config_version': CONFIG_VERSION}
    source_kind = 'mock' if predictor.demo else 'inference'
    records, features = [], []
    for i, r in enumerate(seg.itertuples(index=False)):
        geom, reason = linestring(track, r.start_s, r.end_s, request.policy)
        acc = _num(r.gps_accuracy_max_m, 1)
        rec = {
            'schema_version': SCHEMA_VERSION, 'component': COMPONENT, 'external_run_id': run_id,
            'survey_id': request.survey_id, 'record_id': f'{run_id}:seg:{i:05d}',
            'producer': producer, 'source_kind': source_kind, 'upstream': list(request.upstream),
            'interval_start_ms': _ms(r.start_s), 'interval_end_ms': _ms(r.end_s),
            'geometry': geom, 'geometry_null_reason': reason,
            'roughness_class': r.roughness_class, 'confidence': _num(r.confidence),
            'class_probabilities': {c: _num(getattr(r, f'p_{c}')) for c in predictor.classes},
            'quality': {
                'n_windows': int(r.n_windows), 'speed_mean_mps': _num(r.speed_mean_mps, 2),
                'manoeuvre_fraction': _num(r.manoeuvre_fraction, 3), 'gps_accuracy_max_m': acc,
                'low_gps_accuracy': bool(acc is not None and acc > request.policy.low_accuracy_m),
                'noise_profile_subtracted': bool(trip.report.get('noise_profile_subtracted')),
            },
            'sensor': {'device': manifest.get('device'), 'axis_frame': manifest.get('axis_frame'),
                       'native_rate_hz': round(ingest.native_rate_hz, 1), 'model_rate_hz': request.ingest.target_fs,
                       'speed_source': ingest.speed_source},
        }
        records.append(rec)
        features.append({'type': 'Feature', 'geometry': geom,
                         'properties': {k: rec[k] for k in ('record_id', 'interval_start_ms', 'interval_end_ms',
                                                            'roughness_class', 'confidence')}})

    out = Path(context.output_dir)
    with open(out / 'c4_segments.jsonl', 'w') as f:
        for rec in records:
            f.write(json.dumps(rec, allow_nan=False) + '\n')
    with open(out / 'c4_segments.geojson', 'w') as f:
        json.dump({'type': 'FeatureCollection', 'features': features}, f, allow_nan=False)
    Wout = W.copy()
    Wout.insert(0, 'window_start_ms', (Wout.pop('t_start') * 1000).round().astype('int64'))
    Wout.drop(columns=[c for c in ('seg', 'label') if c in Wout]).to_csv(out / 'c4_windows.csv', index=False, float_format='%.10g')

    counts = {c: int((seg.roughness_class == c).sum()) if len(seg) else 0 for c in predictor.classes}
    dur = {c: round(float((seg.end_s - seg.start_s)[seg.roughness_class == c].sum()), 1) if len(seg) else 0.0
           for c in predictor.classes}
    summary = {
        'schema_version': SCHEMA_VERSION, 'component': COMPONENT, 'external_run_id': run_id,
        'survey_id': request.survey_id, 'producer': producer, 'source_kind': source_kind,
        'model': {k: v for k, v in predictor.info().items()},
        'input': {'imu_rows': ingest.imu_rows, 'gps_rows': ingest.gps_rows,
                  'native_rate_hz': round(ingest.native_rate_hz, 2), 'duration_s': round(ingest.duration_s, 1),
                  'imu_gaps': ingest.imu_gaps, 'gps_gaps': ingest.gps_gaps, 'speed_source': ingest.speed_source},
        'processing': trip.report,
        'segments': {'count': len(records), 'by_class': counts, 'seconds_by_class': dur,
                     'unassessed_windows': int((~W.assessed).sum()),
                     'null_geometry': sum(r['geometry'] is None for r in records)},
        'policy': vars(request.policy), 'ingest_config': {k: v for k, v in vars(request.ingest).items()},
        'warnings': warnings,
    }
    with open(out / 'c4_run_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
    context.progress('done', 5, 5)
    return C4Result(records=records, warnings=warnings, summary=summary,
                    artifacts={'segments_jsonl': 'c4_segments.jsonl', 'segments_geojson': 'c4_segments.geojson',
                               'windows_csv': 'c4_windows.csv', 'summary_json': 'c4_run_summary.json'})


def main(argv=None):
    p = argparse.ArgumentParser(description='Run the C4 road-roughness adapter on contract CSV files.')
    p.add_argument('--gps', required=True)
    p.add_argument('--imu')
    p.add_argument('--acc')
    p.add_argument('--gyro')
    p.add_argument('--manifest', required=True, help='sensor manifest JSON file')
    p.add_argument('--model-dir', required=True)
    p.add_argument('--out', default=None)
    p.add_argument('--survey-id', default='local-survey')
    a = p.parse_args(argv)
    with open(a.manifest) as f:
        manifest = json.load(f)
    out = a.out or tempfile.mkdtemp(prefix='c4_')
    req = C4Request(survey_id=a.survey_id, gps_path=a.gps, imu_path=a.imu, acc_path=a.acc, gyro_path=a.gyro,
                    manifest=manifest, model_dir=a.model_dir)
    try:
        res = run(req, LocalRunContext(out, verbose=True))
    except C4InputError as exc:
        print('INPUT ERROR:', exc)
        return 2
    print(json.dumps(res.summary['segments'], indent=2))
    for w in res.warnings:
        print('WARNING:', w)
    print('Output folder:', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
