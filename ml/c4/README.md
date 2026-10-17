# C4 – Road roughness from smartphone sensors

Component 4 classifies road surface roughness as **good / regular / bad** from a phone's accelerometer,
gyroscope and GPS, using a model that was validated on unseen vehicle–driver pairs (leave-one-vehicle-out).
This folder is the C4 **trusted inference adapter** described in `docs/architecture.md` and
`docs/interfaces.md`. It reads raw sensor files, runs the trained model, and returns contract records
plus output files. It never writes to the database; the platform's worker uploads the outputs and
commits the records.

```
ml/c4/
  api/               FastAPI router (upload, status, GeoJSON, records) + file-based run store
  roughness/
    adapter.py       run(request, context) -> C4Result, plus a CLI
    inputs.py        CSV contract checks, units, synchronisation, resampling to 100 Hz
    predictor.py     model loading (portable / joblib), features, normalisation, classification
    segments.py      window policy: 2 s windows -> non-overlapping road segments + LineStrings
    c4_pipeline.py   the research pipeline shared with the notebooks (noise handling and features)
  scripts/
    make_fixture.py           deterministic synthetic drive in the contract format
    make_demo_model.py        small demo model for tests (results are marked "mock")
    export_portable_model.py  joblib bundle from Notebook 07 -> portable model (run in Colab)
    pvs_to_contract_csv.py    PVS trip -> contract CSVs (real-data check)
    check_on_pvs.py           run the adapter on a converted PVS trip and score it against the labels
  fixtures/synthetic_drive/   contract fixture used by the tests
  models/                     model weights (not in Git, see models/README.md)
  tests/                      contract tests
  requirements.in / .lock     dependencies (locked for Python 3.11)
```

## Quick start

```bash
uv pip install -r ml/c4/requirements.lock          # Python 3.11, repo virtual environment
python -m pytest ml/c4/tests -q                     # 17 tests, about 5 s

python -m ml.c4.scripts.make_demo_model --out ml/c4/models/demo
python -m ml.c4.roughness.adapter \
    --imu ml/c4/fixtures/synthetic_drive/imu.csv \
    --gps ml/c4/fixtures/synthetic_drive/gps.csv \
    --manifest ml/c4/fixtures/synthetic_drive/manifest.json \
    --model-dir ml/c4/models/demo --out c4_out
```

From Python (the worker):

```python
from ml.c4.roughness.adapter import C4Request, LocalRunContext, run

result = run(C4Request(survey_id=survey_id, imu_path=imu, gps_path=gps, manifest=manifest,
                       model_dir=model_dir, external_run_id=run_id),
             context)              # any object with output_dir, progress(stage, done, total), is_cancelled()
result.records     # list of C4 result records (also in c4_segments.jsonl)
result.artifacts   # {'segments_jsonl': 'c4_segments.jsonl', 'segments_geojson': ..., 'windows_csv': ..., 'summary_json': ...}
```

## REST API

`ml/c4/api` is a FastAPI router. The backend mounts it under `/api/v1` with PostgreSQL persistence
(`backend/app/routes/c4.py` and `backend/app/repositories/c4_runs.py`). The web page is at
`http://localhost:3000/c4` (`frontend/web/src/app/c4`).

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/c4/model` | Model in use: version, SHA-256, `demo` or `production` |
| POST | `/api/v1/c4/runs` | Multipart upload → `202` + run. Fields: `gps` (required), `imu` or `acc` + `gyro`, `manifest` file or `manifest_json`, optional `survey_id` |
| POST | `/api/v1/c4/runs/demo` | Run on the synthetic fixture (no upload needed) |
| GET | `/api/v1/c4/runs` | Recent runs |
| GET | `/api/v1/c4/runs/{run_id}` | Status (`queued`/`running`/`succeeded`/`failed`/`cancelled`), stage, summary, warnings, error |
| POST | `/api/v1/c4/runs/{run_id}/cancel` | Cancel (checked between stages) |
| GET | `/api/v1/c4/runs/{run_id}/segments` | GeoJSON FeatureCollection for the map |
| GET | `/api/v1/c4/runs/{run_id}/records` | Contract records (schema 1.0) |
| GET | `/api/v1/c4/runs/{run_id}/windows.csv` | Per-window diagnostics |

How runs are handled:

* Runs execute in the background, one at a time, so poll `GET /runs/{run_id}`.
* Errors use the platform format `{code, message, request_id, details}`:
  * 422 invalid input
  * 404 unknown run
  * 409 results requested before the run succeeded
  * 413 file too large (CSV 250 MiB, manifest 10 MiB)
  * 503 no model
* Invalid sensor data does not fail the request. It produces a run with `status: failed` and
  `error.code: invalid_input`, and the message is safe to show to the user.

Settings, read from environment variables:

* `C4_STORAGE_DIR`: run folders. The default is `ml/c4/runs`, which is git-ignored; in Docker, mount a
  volume here.
* `C4_MODEL_DIR`: production model folder. The default is `ml/c4/models/production`.
* `C4_ALLOW_DEMO`:
  * `1` (default): when there is no production model, the synthetic demo model is used or built.
    Results are then marked `source_kind: "mock"`.
  * `0`: use this in production.

Storage:

* **PostgreSQL** (backend), with two tables:
  * `c4_runs`: status, stage, model, summary, warnings and error of each run.
  * `c4_segments`: one row per road segment result, in the contract record format, with interval,
    class, confidence, quality and GeoJSON geometry.
  The tables are created on first use. Geometry is stored as GeoJSON because the Compose database is
  plain PostgreSQL; switch the column to `geometry(LineString, 4326)` when PostGIS is enabled.
* **Files**: the uploaded CSVs and output artifacts live in `C4_STORAGE_DIR/<run_id>/`. In Compose
  this is the `c4_data` volume.
* The router also works without a database (`build_router()` uses a file store), which is what the
  `ml/c4` tests use.

This is the first milestone. Execution uses FastAPI background tasks, one run at a time. When the
platform worker exists, it calls the same `ml.c4.roughness.adapter.run(...)`.

The backend Docker image installs `ml/c4/requirements.lock` (see `backend/Dockerfile`).

## Input contract

As defined in `docs/interfaces.md`:

| File | Columns | Notes |
|---|---|---|
| IMU CSV | `timestamp_ms,ax_mps2,ay_mps2,az_mps2,gx_rads,gy_rads,gz_rads` | or two files: acc (`timestamp_ms,ax_mps2,ay_mps2,az_mps2`) and gyro (`timestamp_ms,gx_rads,gy_rads,gz_rads`) |
| GPS CSV | `timestamp_ms,latitude,longitude,accuracy_m,speed_mps` | `accuracy_m` and `speed_mps` may be empty |
| Manifest | JSON | must contain `"gravity_included": true`; also `device`, `axis_frame`, `sample_rate_hz`, `clock_origin` |

* `timestamp_ms` is an integer, relative to the survey capture start, and uses the same clock in all files.
* Accelerometer values must be in m/s² **including gravity** (Android `TYPE_ACCELEROMETER`, not
  `TYPE_LINEAR_ACCELERATION`). The gravity direction is needed for the orientation-invariant channels,
  so any device axis frame and any phone orientation is accepted.
* The IMU must be sampled at ≥ 50 Hz. 100 Hz or more is recommended, because the model was trained at
  100 Hz. Below 90 Hz a warning is added.
* For the vehicle noise profile, start each survey with a few seconds stationary with the engine running.

## Processing

This is the same processing as the research notebooks, documented there in detail:

1. **Ingestion** (`inputs.py`, config `c4-ingest-1.0`)
   * Check units: median |acc| must be about 9.8 m/s². Warn if the gyroscope looks like deg/s.
   * Put separate gyroscope data on the accelerometer clock.
   * Resample to 100 Hz. IMU gaps longer than 0.2 s are not interpolated.
   * Interpolate GPS position and speed onto the IMU timeline. GPS gaps longer than 5 s are **not
     bridged**: position and speed stay unknown there, and that part is not assessed.
   * If `speed_mps` is missing, derive speed from consecutive positions (haversine distance).
2. **Noise handling** (`c4_pipeline.process_sensor`)
   * Replace full-scale glitches.
   * Remove gravity with a 0.3 Hz filter.
   * Apply a 40 Hz zero-phase low-pass filter.
   * Compute orientation-invariant channels: vertical, horizontal, |acc| and |gyro|.
   * Subtract the trip's stationary noise profile (PSD subtraction).
3. **Windows and features**: 2 s windows with a 1 s step. Time-domain and spectral features are computed
   on the invariant channels.
4. **Driver/vehicle-dependency handling**
   * Speed normalisation, using slopes learned in training. Raw speed is not a model input.
   * Per-trip robust median/IQR normalisation over the trip's moving windows. This is label-free.
   * Only the cross-vehicle stable features selected in training are used.
5. **Classification** of every window with speed ≥ 1 m/s. Slower windows are not assessed.

## Window policy and output

The model classifies overlapping 2 s windows every 1 s. Each window is represented by its **central
1 s slice** `[t+0.5 s, t+1.5 s)`, so the slices tile the recording without overlap. Consecutive slices
are merged into one segment while all of these hold:

* they have the same class;
* they are adjacent;
* they are in the same continuous recording part;
* the segment is ≤ 10 s long.

Segment confidence and class probabilities are the means over the segment's windows.

Every record in `c4_segments.jsonl` contains:

* **Common identity fields:**
  * `schema_version` `"1.0"`
  * `component` `"C4"`
  * `external_run_id`, `survey_id`, `record_id`
  * `producer` (`model_version`, `code_version`, `config_version`)
  * `source_kind` (`inference`, or `mock` for the demo model)
  * `upstream`, which is empty because C4 does not depend on C1–C3
* **Interval:** `interval_start_ms`, `interval_end_ms`, as integer ms on the survey clock.
* **`geometry`:** a GeoJSON LineString `[lon, lat]`, at 1 vertex per second. It is `null` with a
  `geometry_null_reason` when fewer than two positions exist. `[0,0]` is never used.
* **Result:** `roughness_class` (`good`/`regular`/`bad`), `confidence`, `class_probabilities`.
* **`quality`:**
  * number of windows
  * mean speed
  * manoeuvre fraction (turning or braking windows)
  * worst GPS accuracy and a `low_gps_accuracy` flag (> 25 m)
  * whether the noise profile was subtracted
* **`sensor`:** device, axis frame, native and model rate, speed source.

The other outputs are:

* `c4_segments.geojson`: the same segments for map preview.
* `c4_windows.csv`: every window with its probabilities, for diagnostics.
* `c4_run_summary.json`:
  * input report
  * counts per class and seconds per class
  * unassessed windows
  * warnings
  * model provenance
  * configuration

The results are **segment assessments of roughness class**, not IRI values.

## Failure behaviour

| Situation | Behaviour |
|---|---|
| Missing columns, unreadable CSV, no gravity in the manifest, wrong units, IMU < 50 Hz, fewer than 2 GPS fixes, no movement ≥ 1 m/s, recording too short | `C4InputError`. Not retryable; the message is safe to show to the user |
| `context.is_cancelled()` becomes true | `C4Cancelled`, raised between stages |
| GPS gaps, IMU gaps, no stationary start, short survey, low IMU rate, derived speed | The run succeeds; a warning is added to `result.warnings` and the summary |
| Model files missing | `FileNotFoundError` (deployment problem) |
| Anything else | Internal error; the worker marks the run as failed and keeps the log |

## Model deployment

The production model is the smartphone model (C7) from Notebook 07. It is an XGBoost model with
40 stable features, LOVO macro-F1 0.828.

1. In Colab, with Drive mounted and this repository's `ml` folder on the path, run:
   ```bash
   python -m ml.c4.scripts.export_portable_model \
       --bundle /content/drive/MyDrive/PVS_Model_Results_v2/c4_deployment_model.joblib \
       --out /content/drive/MyDrive/PVS_Model_Results_v2/portable
   ```
   The script verifies that the portable model gives the same probabilities as the joblib bundle.
2. Copy the two files into the deployment model folder, for example `ml/c4/models/production/`.
   Do not commit them; the weights come from the trusted asset store.
3. The model version recorded in results includes the SHA-256 of the model files.

To check the model on real data, convert a PVS trip with `scripts/pvs_to_contract_csv.py` and score
it with `scripts/check_on_pvs.py`. A trip that was used for training gives an optimistic score.

## Notes

* `c4_pipeline.py` is kept identical to the version used by the research notebooks, so that training
  and inference use the same code. For this reason it is not reformatted to the repository lint rules.
* Open item from `docs/interfaces.md`, "C4 sensor coordinate conventions / window policy": proposed
  above. Any axis frame is accepted because of the orientation-invariant channels; the only
  requirement is that gravity is included.
