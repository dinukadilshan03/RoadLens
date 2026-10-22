# Component 3 surface workspace

Open `/c3` in the existing Next.js frontend. The dashboard contains image input, damage identification, visual analysis, assessment results, severity/reliability, and PDF reporting. Uploads stay in the current browser session; C3 does not persist files or run history.

## Independent dashboard inputs

The first two upload boxes form a pothole RGB/mask pair. The third is an independent crack mask in its own coordinate system. The crack mask can have different dimensions from the pothole photo and can be analyzed without any photo. An optional separate crack photo enables the trained C3-27 RGB visual classifier; that photo must match the crack mask. A mask alone supplies morphology only and is never sent to either RGB classifier.

The dashboard uses `POST /api/v1/c3/assess` and `POST /api/v1/c3/assessment-report` with optional `image`, `mask`, `crack_mask` and `crack_image`. A pothole photo/mask pair or a crack mask is required. Returned defects include `source_group`, `source_width`, `source_height`, `source_kind` and `analysis_mode`. The viewer selects each finding's own source image and coordinates. Combined PDFs include a separate overview per source, followed by the matching defect pages. Mask-only crack findings show shape analysis with an invitation to add a crack photo for visual severity. Pothole 3D visualization is unchanged.

The following shared-frame endpoints remain available for existing C1/C2 clients.

## Inputs and processing

Manual masks are checked independently. If one mask is invalid but another is valid, analysis continues for the valid mask and returns explicit `input_warnings` identifying the skipped damage type. The dashboard and PDF show these messages. If no mask is valid, the request returns an actionable error. Size mismatch messages include the mask and photo dimensions; masks from another photograph must be analyzed with that photograph. The app does not stretch an unrelated mask onto the selected road photo. Upload previews show pixel dimensions before analysis.

`POST /api/v1/c3/analyze` accepts multipart `image`, optional `mask` (potholes), and optional `crack_mask`. At least one damage mask is required. Images are EXIF-oriented before validation. Masks must contain binary grayscale values (0/1 or 0/255), both foreground and background, and exactly match the road photograph dimensions and coordinates. No resizing or photographic thresholding is performed. Use lossless PNG masks. Limits: 15 MB per file, 12 megapixels, minimum 8 pixels per side, and 20 connected damaged regions per request. Matching dimensions cannot establish that the user selected the correct photograph; users must verify the visible overlay.

For future C1/C2 integration, send `image` and a `detections` JSON list instead of manual masks. Each record has a unique `defect_id`, `damage_type` (`pothole` or `crack`, default pothole for existing clients), integer exclusive `bbox_xyxy`, `mask_space` (`frame` or `bbox`), and binary `mask`. Frame masks must fit the bounding box; bbox masks must exactly match its dimensions. Existing associated full-frame `crack_mask` records are separated into independent crack findings. Connected components use 8-neighbor connectivity and receive deterministic identifiers. Manual single-region IDs are `pothole` and `crack`; multiple regions use e.g. `pothole-pothole-1`. Use the returned identifiers to attach field records.

Both manual and automatic records use the same normalization and analysis service. Potholes retain the original trained CRDL checkpoint, inference modules and existing measurement/reliability policy. Cracks use the existing pixel geometry algorithm, skeletonization and local distance-based width visualization, independently of model availability. There is no artificial crack fallback. Actual C1/C2 orchestration remains future integration work.

Responses retain internal bridge measurements for reproducibility and add backend-generated `presentation`, PNG `visuals`, normalized profile data and distribution bins. Ordinary UI panels show plain-language findings rather than technical measurements or internal flags. The depression line moves with the selected image row and graph; opacity is adjustable. Width colors represent image-relative geometry, never measured millimeters.

## Physical evidence

The dashboard requires only RGB and separately identified pothole/crack masks. Physical severity is not established by the image. For existing API clients, optional multipart `field_measurements` remains supported, but it is never required by the UI or used to alter AI visual severity. Valid evidence-gated field labels are reported separately as physical severity; threshold-ambiguous measurements remain unclassified. Crack field assessment is not enabled.

## PDF reports

`POST /api/v1/c3/report` accepts the same inputs and field records, validates and recomputes findings, and returns an attachment with `application/pdf`. It contains an overview with labeled damage locations and a separate illustrated page for every defect, measurements with sources, severity status and physical limitations. The dashboard exposes only **Download PDF Report**. No user-facing JSON/CSV exports are provided. Research artifacts remain available locally; the dashboard does not display a Research Evidence section.

## Validation

The backend container contains the model dependencies. Run `python -m pytest tests/test_c3_api.py` in a development environment with those dependencies and pytest/httpx installed. Tests exercise the real checkpoint, pothole/crack/both inputs, multiple defects, invalid/photographic/misaligned masks, automatic frame/ROI inputs, model failure, field evidence and PDF pages. Run `npx tsc --noEmit`, `npm run lint` and `npm run build` in `frontend/web`.

Usability studies, physical field evaluation and automated C1/C2 orchestration require the corresponding people, measurements and integration inputs; they are not implied by image analysis.


## Interactive relative 3D surface

The `build_crdl_3d_visualization.py` reference is adapted in `backend/app/services/c3_surface.py`. Every real pothole prediction is paired with its own full-frame mask, resized to the native model grid with nearest-neighbor sampling. The surface is zero outside the mask, cropped with a 24-cell road margin, and sampled every second cell. Sampling falls back to the native crop when decimation would erase all predicted depression. Native statistics use the complete, undecimated paired region. Invalid nonfinite or negative predictions are rejected; a region that disappears at model resolution has no surface, and zero depression produces a flat surface.

`relative_surface` contains backend-prepared x/y image-position percentages, negative normalized z values, relative colors, display aspect ratio, and internal native statistics. Normalization changes only the display scale; it does not establish physical coordinates or field depth. Longest-axis scaling fixes extreme portrait/landscape aspect ratios in the reference. The reference's example file paths, sample identifier, developer HTML output and technical badges are not used in the application.

The pothole visual results lazily load the Plotly GL3D surface bundle on the client. Users can rotate, zoom, hover, and switch between perspective and top views. The viewer uses the original model-grid display proportions, visible depth relief and multicolor depression scale. Crack results do not show a pothole surface. WebGL or bundle failures leave the existing 2D views and profile available. PDF reports add one static Matplotlib 3D page per available pothole surface, using the same prepared geometry and colors. There are no new user-facing image or HTML export actions.

Additional tests in `tests/test_c3_surface.py` verify paired-mask isolation, depression direction, flat output, missing native-grid regions, decimation fallback and invalid predictions.

The interactive viewer also overlays a translucent conceptual healthy-road plane at relative zero using the same image coordinates. There is no vertical exaggeration control. The 2D profile and 3D surface both descend below the road reference. The plane does not use the deployment-disabled C3-28 learned reconstruction model. Geometry remains image-relative.

## Trained-model bundles

Imported `CRDL_Component3_Actual_Trained_Models_and_Backend.zip` into `ml/c3` with the original `geometry`, `pothole_visual_severity`, `crack_visual_severity`, `integration` and `research_only` namespaces. Top-level provenance and setup files are in `bundle_evidence`. All 45 manifest-listed files passed SHA-256 verification. The imported geometry checkpoint and Python modules match the existing root C3 bundle byte-for-byte; existing imports remain compatible.

C3-22 supplies experimental Low / Moderate / High pothole visual severity. C3-27 supplies Low / High crack visual severity using RGB-only inference and its supplied temperature and review threshold. The original inference preprocessing is preserved. Fusion is disabled; its joblib file remains under research_only. C3-07 development length and C3-28 reconstruction models are preserved as research artifacts, not production measurements.

Input review flags or classifier abstention produce Needs Review. Missing classifier models do not invent a class or block available geometry. Insufficient geometric detail is reported as Insufficient input. The UI and PDF show no uncalibrated score percentages. Geometry describes CRDL relative depression concentration or measured skeleton topology; it never supplies severity thresholds or millimetres.

`GET /api/v1/c3/model` reports geometry readiness and each visual model's availability. `GET /api/v1/c3/research/{artifact}` exposes only four allowlisted research artifacts: stages, comparison, gate and pothole-evaluation. These read-only research endpoints are retained for existing clients; the Research Evidence UI section has been removed. No research model is executed by this endpoint.

Model weights and `.joblib` artifacts are ignored by Git. Docker includes only the root CRDL and two selected visual severity checkpoints; research checkpoints and joblib artifacts remain excluded. Keep the trained files locally or provision them into the matching bundle folders when deploying. The backend Docker dependency installation includes torchvision; for a local C3 environment install the dependencies in `ml/c3/bundle_evidence/requirements_component3.txt`.

## Simplified assessment presentation

The dashboard and PDF show AI visual severity, estimated shape, review status and inspection guidance. Repeated physical-measurement notices, unavailable physical-severity labels and experimental-classifier wording have been removed from presentation. Internal physical evidence fields and model provenance are preserved for API compatibility. The 3D surface retains the depression map values and uses the original horizontal display proportions, multicolor depression scale and depth relief; it does not claim a recovered physical aspect ratio.


### Whole crack network assessment

Each supplied crack mask is one analysis unit, including disconnected regions.
Explicit JSON crack instances remain separate. Original pixel morphology is retained;
secondary cleanup removes isolated pixels and branch inspection omits one-pixel
terminal stubs. Regions and branches are inspection aids, not duplicate defects.
C3-27 evaluates RGB regions (at most 20 non-isolated connected regions, with coverage
reported); branched or disconnected networks retain Needs Review, with no averaging
or maximum-class aggregation. Physical severity remains separate.

`/assess` and `/assessment-report` use `image` automatically for crack-only uploads.
With both pothole and crack masks, `crack_matches_image=true` confirms shared source;
otherwise supply `crack_image` or receive mask-only analysis. Dimensions alone cannot
prove source identity. A supplied crack photograph takes precedence.

C3-26 reference cohort/scoring artifacts are absent from this checkout. Complexity
therefore reports Needs Review with an explicit explanation until those artifacts
are supplied; arbitrary Lower/Moderate/Higher cutoffs must not be substituted.
