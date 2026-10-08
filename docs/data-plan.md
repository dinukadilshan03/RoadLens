# Upload, metadata and mapping plan

Planning draft, 8 October 2026. PostgreSQL is selected by the user. Basic Docker database setup and backend connectivity are implemented. The application schema and remaining workflow decisions are still being planned.

This draft revises the survey-first assumptions in architecture.md and interfaces.md. Those documents remain historical design proposals; their mandatory projects/accounts/survey creation and survey_id contracts must be reconciled before implementation.

## Confirmed scope

- PostgreSQL is the application database. PostGIS remains optional pending spatial-query requirements.
- First release is a single shared demo without accounts.
- Users upload individual road photos or videos without creating a route or survey.
- GPS, capture time and sensor data are optional. Missing metadata must not block visual analysis.
- Mobile recording is required later, using the same storage and processing contracts.
- Map placement requires usable location evidence. A file without location still has a results page.
- C1-C3 visual analysis and C4 sensor-based roughness are separate branches with independent eligibility and status.

## User workflow

1. Select a photo or video. An internal recording ID is created automatically; the user need not name an inspection.
2. Upload and validate the file; display extracted metadata and its source.
3. Optionally attach a GPS or sensor file, supply a capture time, or add a location.
4. For video tracks, preview path, clock alignment and coverage before using them for mapping. A manually supplied single point is an approximate recording location, not a track for every frame.
5. Run supported analysis. Show unavailable branches and the missing inputs; never fabricate sensor-based roughness from an ordinary photo.
6. View results beside the media. Located results also appear on the map; unlocated results remain in the list.
7. Correct metadata or rerun analysis while retaining the earlier inputs and run version.

A later mobile client creates the same recording, uploads video chunks/photos and telemetry, and supplies clock mappings. Offline upload retries reuse stable client IDs to avoid duplicates. Live streaming is outside this plan.

## Logical records

These are data responsibilities, not a requirement to build every table immediately.

| Record | Responsibility | Timing |
|---|---|---|
| Recording | Groups related media and telemetry; optional title and capture time; created automatically | First upload milestone |
| File | Storage key, original name, verified type, size, checksum, upload state, media duration/dimensions and role | First upload milestone |
| Metadata revision | Extracted values and user corrections, provenance and selected version | Metadata milestone |
| Telemetry stream | GPS/IMU type, source file, units, axes, clock origin, alignment and validation state | GPS/sensor milestone |
| GPS sample | Stream ID, sample time, latitude/longitude and optional accuracy | Mapping milestone |
| Processing run | Component/model/config versions, exact input and metadata revisions, status and errors | Analysis milestone |
| Result | Run reference, frame time or time interval, component payload, evidence references and location quality | Analysis milestone |

An upload is a transfer operation; a recording is the logical grouping. This prevents a future video plus GPS file, or mobile video split into chunks, from becoming unrelated inspections. A photo simply has one recording with one media file. Sensor-only recordings can use the same model when C4 is introduced.

Start with a small schema and introduce responsibilities as their milestone is implemented. Do not build users, memberships, route catalogs, model registries or review systems for the first demo.

## Time alignment

Maintain three distinct notions of time:

- Upload time: server receipt time; always known and never used as capture time.
- Capture time: optional real-world time, stored in UTC only when timezone/offset is known. Preserve original values and uncertainty if a file has a timezone-less date.
- Recording time: elapsed time from a recording's chosen origin. Use an integer microsecond representation internally and explicit units in public contracts.

Each media file records its start offset within the recording. Video events use presentation timestamps relative to the file start, not frame number divided by average FPS. This accommodates variable-frame-rate video and future chunks.

Each telemetry stream preserves original timestamps and maps them onto recording time. Initially allow a known offset; later support a calibrated scale for clock drift:

`recording_time = scale * (source_time - source_origin) + offset`

Save the mapping method, parameters, uncertainty and revision. Mobile implementations must verify how platform camera and sensor clocks relate; starting both APIs at roughly the same moment is not proof of synchronization.

For companion uploads, accept documented time units and clock origin. If alignment cannot be established automatically, require an explicit offset for frame-level mapping. A drawn A-to-B path does not establish speed or frame positions. Trimming a video changes the alignment and must be recorded.

## GPS and map behavior

- Use WGS84 coordinates; named latitude/longitude fields in input and longitude-first coordinates in GeoJSON.
- Validate finite values, coordinate ranges, sample ordering, duplicates and declared accuracy units. Unknown is null, never an automatic zero coordinate.
- Keep the original track and normalized samples. Do not silently replace suspicious readings.
- Map paths are derived from usable samples. Break paths across large gaps; do not connect disconnected sections as if recorded continuously.
- Match a detection's recording time to a valid GPS sample, or interpolate between valid nearby bracketing samples under a versioned gap/accuracy policy. Do not extrapolate beyond coverage.
- Initial gap, jump and accuracy thresholds must be calibrated with representative recordings before acceptance; they are not scientific constants.
- Label location method: recorded camera GPS, interpolated camera GPS, manual approximate point, or unknown. Retain source samples, alignment revision and supplied accuracy.
- Camera position is an approximate observation position, not the exact defect coordinate. Exact defect geolocation requires additional positioning/camera geometry work.
- Repeated observations of one defect in a video should reference a consolidated defect identity from C1. Do not count every frame as a new pothole. Cross-recording deduplication is deferred.
- C4 roughness is an interval/segment overlay, separate from C1-C3 defect markers and severity.
- Clicking a marker seeks to its evidence timestamp. Playing a video can move a camera-position marker where GPS coverage exists.
- Map responses use viewport filters, result caps and simplified display tracks. Retain full data for analysis. Location corrections regenerate derived map positions without rerunning visual detection unnecessarily.

## Sensor data

Keep original recordings as files. Normalize into a documented file representation for processing; do not put every high-frequency IMU reading into database rows by default.

A stream declares accelerometer/gyroscope units, axis orientation, whether gravity is included, sampling information, device and timestamps. Separate streams are allowed when sensors have different rates. Preserve gaps and report invalid or missing fields.

For scale illustration only: one hour at 1 Hz is 3,600 GPS samples; one hour at 100 Hz is 360,000 IMU samples. Queryable GPS rows can be useful for map/time lookup while large IMU arrays remain files. Actual capture rates are still undecided.

Validate a first supported companion-file contract before adding adapters for other formats. CSV with explicit headers/units/time origin is a candidate; GPX and device-specific embedded telemetry are optional import adapters, not guaranteed support for arbitrary files. Embedded metadata availability must be tested on sample devices.

## Storage and consistency

Database: recording/file metadata, stream descriptions, GPS samples, run state, result summaries and relationships.

File storage: original images/videos, raw sensor files, masks, crops and large arrays. Use private local storage for the demo behind a storage interface; cloud object storage can implement that interface later.

Validate uploaded content before marking it ready. Interrupted uploads remain incomplete. Associate attached files explicitly with a recording rather than guessing from filenames or filesystem modification dates.

Runs reference immutable input versions. Retried upload/processing requests must not create duplicate visible results. Failed processing preserves original inputs for retry; each component reports its own state.

Deleting a recording must account for its files, telemetry and derived outputs and any active job. Define retention before importing real data. Backups must cover both database and media, with a restore check.

Shared demo means all visitors within the deployment boundary have the same access. Keep it local or access-restricted; public anonymous uploads and sensitive GPS data are a different scope requiring limits and an access policy. No account system is planned for this milestone.

## Database decision criteria

| Candidate | When it fits | Tradeoff for this plan |
|---|---|---|
| SQLite | One-machine demo, modest writes, simple map display | Very simple setup; only one writer at a time, less convenient if API and workers write concurrently or spatial queries grow |
| PostgreSQL | Shared application, concurrent processing, relational provenance | One additional service, but fits recordings/files/runs/results and supports indexed JSONB for variable metadata |
| PostgreSQL + PostGIS | Nearby-defect, route-distance and spatial filtering become core requirements | PostGIS is an extension in the same database, not a second database service; spatial types and migrations need care |
| MongoDB | A document-oriented model is the dominant requirement | Supports geographic queries, but flexible metadata alone does not justify choosing it over relational records here |

Displaying coordinates on a map does not by itself require PostGIS. Ask whether the server needs to answer distance, nearest-road, inside-area and large viewport queries. If yes, PostGIS becomes more valuable. Ordinary latitude/longitude columns are sufficient for a small initial display.

Decision, 8 October 2026: use PostgreSQL. The comparison above records the alternatives considered, not an open database decision. PostGIS remains conditional on confirmed spatial queries. Do not introduce a time-series database, Redis or an extra storage service merely because sensor files exist.

Docker is a deployment choice independent of the database model. The proposed local setup extends existing Compose with a pinned PostgreSQL image, persistent named volume, health check and environment configuration. Pin versions after compatibility verification, not in this planning draft.

## Milestones and checks

1. Upload foundation: recording/file records and local storage. Check photo/video upload, restart persistence, malformed input and interrupted transfers.
2. Metadata: extraction, optional corrections, GPS/sensor attachment and clock alignment. Check unknown dates, missing timezone, out-of-order samples, invalid coordinates, trimmed video and mismatched clocks.
3. Mapping: track preview, missing-coverage handling and timed markers. Use a known synthetic track to verify matching and map click-to-seek. Check that gaps are not bridged and camera location is labeled approximate.
4. Analysis: integrate each component as available. Check independent branch eligibility, failed runs, repeat requests, version provenance and repeated observations of the same defect.
5. Mobile capture: offline recording and upload through the same contracts. Check clock alignment on real devices, chunk offsets, interrupted/retried upload and loss of GPS.

Do not claim physical positioning accuracy from synthetic tests; validate that separately against field references. Use small fixtures to test software behavior.

## Decisions still needed before finalizing

- Is the demo local to one laptop, shared on a team network, or hosted for remote access?
- Approximate video duration, recordings per day, concurrent processing and retention requirements.
- First map milestone: show paths/markers only, or also nearby defects, area filters and road-level aggregation?
- Available sample devices and GPS/IMU files; supported first import format and achievable time alignment.
- Whether the first processing milestone uses research outputs, real inference or explicitly labeled mock fixtures.
- C1 consolidation behavior, C3 metric/calibration requirements and C4 sensor/window requirements from component owners.

These decisions affect scope and sizing. Unavailable research outputs do not justify invented values or final schemas for unconfirmed payloads.

## Technical references

- SQLite deployment/concurrency guidance: https://www.sqlite.org/whentouse.html
- PostgreSQL JSONB: https://www.postgresql.org/docs/17/datatype-json.html
- PostGIS distance queries: https://postgis.net/docs/ST_DWithin.html
- MongoDB geospatial support: https://www.mongodb.com/docs/manual/geospatial-queries/
- Docker persistent volumes: https://docs.docker.com/engine/storage/volumes/
