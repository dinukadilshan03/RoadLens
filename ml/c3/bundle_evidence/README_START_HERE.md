# CRDL Component 3 — real saved checkpoint and inference files

These files were copied from your saved C3-20, C3-22, C3-27, C3-07 and C3-28 Google Drive experiment artifacts. The source model checkpoints are not recreated and have not been retrained or replaced.

## Install into your existing VS Code repository

1. Extract this ZIP.
2. Copy `backend/component3/` into your repository's `backend/` folder (merge carefully; back up similarly named existing files first).
3. Install the Python dependencies listed in `requirements_component3.txt` within the backend's environment.
4. From the extracted bundle root, run `python smoke_test_models.py`. This checks cold-load/inference on synthetic input and **does not** measure model accuracy.
5. Wire the output into your existing image + mask API / UI code. The included C3-23 adapter demonstrates how C3-20 and C3-22 can run on the same pothole.

**Main models**:
- `backend/component3/geometry/selected_model.pt` — selected C3-14-trained road-relative pothole depression, C3-20 release wrapper.
- `backend/component3/pothole_visual_severity/visual_severity_model.pt` — C3-22 pothole visual Low/Moderate/High.
- `backend/component3/crack_visual_severity/crack_visual_low_high_model.pt` — C3-27 crack visual Low/High; uncertain inputs may require review.

**Crack morphology** is provided by `geometry/geometry_measurements.py` and the C3-20 measurement/integration helpers; C3-07 experimental trained crack-length regression is under `research_only/` because it does not establish accurate camera-independent physical lengths. C3-26 cohort-percentile complexity analysis is exploratory and not a trained crack-severity predictor.

**C3-28 auxiliary healthy-road learned reference** resides under `research_only/C3-28_reference_NOT_DEPLOYABLE/`. Its frozen development RAW MAE was **43.2821**, versus training-mean **42.7588** (lower is better). The frozen-development gate **FAILED**; do not enable this checkpoint in the production API. The existing CRDL 3D relative visualizations can continue without it. An annotated reconstruction evidence figure and corrected research PDF are included for the examiner view.

**C3-27 fusion** (`research_only/C3-27_morphology_fusion_NOT_SELECTED/`) must NOT be copied into the live `crack_visual_severity` folder by default. On the matching polygon test subset, RGB-only macro-F1=0.731 and RGB+morphology macro-F1=0.7092. The shipped inference service uses its better-supported RGB-only mode unless fusion file is explicitly placed alongside it.

## Units, confidence and UI constraints

Only RGB image + aligned pothole and/or crack segmentation masks are required at inference. There is no verified automatic physical depth, millimetre scale, physical volume or engineering severity from these images. C3-20 contains an optional *separate* field-reference severity method, but it is not required for RGB+mask inference. Pothole and crack visual severity are **different experimental classifiers**; do not combine their classes or confuse visual risk with measured physical severity.

The model scores and C3-28 development error tolerance do not provide certified per-image physical confidence. Use appropriate `Needs Review` / unverified labels. UI export can be PDF-only; internal model API/JSON data is for developer integration.

## Source provenance

C3-20: `C3-20_severity_ready_bundle.zip`, created 2026-10-21. C3-22: `C3-22_visual_severity_experimental_bundle.zip`, created 2026-10-22. C3-27 source files from `C3-27_crack_visual_severity_20261022T090859Z`. C3-07 trained crack-length files from `C3-07_section09_SUPERVISED_LENGTH_ABLATION_20261020T164116_613524Z`. C3-28 correction source `C3-28_reference_shape_confidence_20261022T100842Z`.

The package is a verified collection of files and includes a diagnostic smoke test. It is **not a tested drop-in replacement for your whole frontend/backend repository**. Confirm your backend API routes, C1/C2 mask handoff, physical-warning UI, and PDF generator in your existing application.
