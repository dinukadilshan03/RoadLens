# RoadLens exchange contracts

Proposed v1 platform contract, 7 October 2026. This document defines implementation requirements; executable schemas and endpoints are not yet present.

## Formats

| Data | Canonical format | Rules |
|---|---|---|
| API / small import manifest | UTF-8 `.json` | `schema_version: "1.0"`; strict required fields, no NaN/Infinity |
| Large result tables | `.jsonl` | One typed result per line, linked from manifest; same record validators |
| Survey video | `.mp4`, H.264 | Validate codec, duration and decode; record presentation timestamps, not frame_index/fps assumptions |
| RGB images/crops | `.jpg` or `.png` | Persist orientation-normalized width/height and coordinate transform |
| Binary masks/skeletons | Grayscale lossless `.png` | Values 0/255, 0 background; skeleton required only for relevant crack output |
| Confidence/depth/depression arrays | `.npz` | Numeric arrays only, `allow_pickle=False`, declared shape/dtype/units; bounds on decompressed bytes |
| GPS / raw sensors | UTF-8 `.csv` | Headers, explicit units and capture-relative timestamps; no implicit clock conversions |
| Internal large sensor tables | `.parquet` | Optional normalized derivative, original CSV retained |
| Map / spatial export | `.geojson` | WGS84 `[longitude, latitude]`; numeric measurements remain properties |
| Code / notebooks | `.py` / `.ipynb` | Python for deployable code; notebooks for experiments, never survey imports |

## Common identity and provenance

Every result has `schema_version`, `component`, `external_run_id`, `survey_id`, `record_id`, `producer` (model_version, code_version, config_version), `source_kind` (import/inference/mock), and explicit `upstream` references. Use UUIDs for platform IDs. External IDs are scoped to the imported run, never globally trusted. API maps manifest file IDs to server-owned artifact IDs after upload; reject arbitrary URLs and filesystem paths.

An import manifest declares component runs, files (file_id, kind, relative display name, SHA-256, byte_size, media_type), and inline records or result-file references. Begin an import under an already-created survey. Idempotency scope is project + survey + client key; a repeated key with different content returns conflict. Multi-component bundles validate C1, then C2, then C3; C4 independently. Commit the whole requested bundle atomically. Individual C2/C3 imports can refer to previously committed compatible runs.

All times use integer `timestamp_ms` relative to survey capture start; `capture_start_at` is nullable UTC ISO 8601. Preserve offset/drift and timestamp source metadata. Unknown absolute time must stay unknown. GPS points carry timestamp_ms, latitude, longitude, and optional accuracy_m. Define interpolation tolerances in a versioned ingestion config; never bridge a large GPS gap silently. Location may be null, with a reason; do not substitute `[0,0]`.

## Component payload requirements

| Component | Required payload semantics |
|---|---|
| C1 | external_damage_id; observation ID; frame/artifact reference; timestamp_ms; damage_class; confidence in [0,1]; bbox; optional location/accuracy and temporal reliability; optional crop artifact with transform |
| C2 | Exact upstream C1 run and observation/defect; mask artifact; pixel dimensions and transform; skeleton for crack output when supported; optional confidence array or scalar with declared meaning |
| C3 | Exact C2 result reference; defect type; applicable nullable measurements; units; reliability; review_required; metric_valid; calibration/model scale provenance; severity and severity_scheme_version if produced |
| C4 | Segment reference or supplied valid LineString; interval_start_ms/end_ms; roughness_class in good/regular/bad; optional confidence; sensor/quality and model provenance |

C1 bounding boxes use `[x_min,y_min,x_max,y_max]`, pixels in the orientation-normalized source frame, top-left origin, exclusive upper bounds. Reject negative/out-of-frame/zero-area boxes. Crop coordinates and masks use their declared image space; store crop origin, scale and any rotation/transform so overlays and measurements remain aligned. Never silently resize masks for C3 input.

C3 SI fields: `depth_m`, `area_m2`, `volume_m3`, `length_m`, `width_mean_m`, `width_max_m`. Only applicable values are supplied. Unknown/unreliable physical values are null with a reason; zero means an actual zero estimate. A normalized depression field cannot be labeled metres without established metric scale. UI respects metric_valid and review_required. The platform does not calculate scientific severity thresholds.

C4 outputs are segment assessments, not per-defect measurements and not IRI unless the member explicitly supplies and validates an IRI estimate. Segment geometry and time interval must belong to the survey; segment creation can be part of import. The roughness branch remains valid without any video.

Raw GPS CSV: `timestamp_ms,latitude,longitude,accuracy_m,speed_mps` (last two nullable). Raw IMU CSV: `timestamp_ms,ax_mps2,ay_mps2,az_mps2,gx_rads,gy_rads,gz_rads`; accelerometer/gyro may be uploaded in separate streams if sampling differs. Manifest declares axis frame, gravity inclusion, sample rate, device and clock origin. Synchronization, filtering and resampling belong to the C4 adapter. A plain CSV with unknown axes/units is not inference-ready.

## API surface

All domain paths begin `/api/v1`. Existing `/health` can remain. Next.js currently strips `/api` through its rewrite: update routing deliberately so `/api/v1` is forwarded unchanged, and verify local and production behavior together.

| Method/path | Purpose |
|---|---|
| POST /auth/login; POST /auth/logout; GET /auth/me | Web session lifecycle |
| GET/POST /surveys; GET /surveys/{id} | Survey list/create/detail |
| POST /surveys/{id}/uploads | Allocate upload session and permitted artifacts |
| PUT /uploads/{id}/files/{file_id} | Stream one file to storage, with bounded size |
| POST /uploads/{id}/finalize | Queue verification; return 202 + job ID |
| POST /surveys/{id}/imports/validate | Validate manifest, return immutable preview ID or structured errors |
| POST /surveys/{id}/imports | Commit verified preview asynchronously; 202 + job ID |
| POST /surveys/{id}/runs | Request eligible inference branches and versions; added in inference milestone |
| GET /jobs/{id}; POST /jobs/{id}/cancel | State/progress/error and cooperative cancellation |
| GET /surveys/{id}/runs | Immutable run history and provenance |
| PUT /surveys/{id}/result-selection | Select compatible run set; audited |
| GET /defects; GET /defects/{id} | Filtered, cursor-paginated observations/results |
| GET /segments; GET /segments/{id} | Roughness assessments |
| GET /map | Bounded GeoJSON by viewport, filters and selected runs |
| POST /reviews | Append reviewer decision tied to exact result version |
| GET /artifacts/{id}/download | Authorized download or short-lived URL |
| POST /exports; GET /exports/{id} | Async CSV/GeoJSON export with run/filter snapshot |

All mutation IDs are checked against the authenticated project membership. API errors include `code`, safe `message`, `request_id` and field/record-level `details`. Use 422 for validation, 409 for incompatible state/idempotency conflict, 413 for upload limits, 403/404 for access policy, and 202 for accepted jobs. Job progress includes stage and completed/total units when known; avoid fabricated percentages.

Preview tokens bind manifest digest, validated artifacts and target upstream runs. Commit rechecks authorization, artifact availability and dependencies. Stale or changed content requires validation again. Browser retries must reuse the idempotency key for the same action.

## Trusted inference adapter boundary

Proposed Python entry point: `run(request: ComponentRequest, context: RunContext) -> ComponentResult`. Request contains validated input references, exact upstream versions and config. Context provides managed local input paths, a temporary output directory, progress reporting and cancellation checks. Results contain validated records plus relative output artifacts. The orchestration service uploads outputs and commits database rows; research code does not directly modify shared tables.

Each component must ship a deterministic small contract fixture, documented preprocessing, dependency lock, model checksum/version and failure behavior. Adapters may be executed in isolated worker images when their ML dependencies conflict. The same v1 payload can later cross an HTTP/container boundary without changing browser code.
