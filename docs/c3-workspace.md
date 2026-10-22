# Component 3 surface workspace

Open `/c3` in the existing Next.js frontend. C3 has its own route, styles, API client and visualization components under `frontend/web/src/app/c3/`. It uses the shared frontend/backend containers, matching C1 and C4.

Run the stack with `docker compose up --build`. The backend loads `ml/c3/selected_model.pt` once during startup and verifies the bundle's checkpoint and architecture hashes. The selected checkpoint is explicitly included in the Docker context; experimental checkpoints and datasets remain excluded. `scikit-image` is installed for the existing geometry measurement service.

`GET /api/v1/c3/model` reports readiness. `POST /api/v1/c3/analyze` accepts multipart `image`, `mask`, and optional `crack_mask` in image formats supported by Pillow (including JPEG, PNG, WebP and BMP). Manual masks are converted to grayscale, thresholded at 127 (0/1 masks retain their foreground), and resized to the EXIF-oriented road image using nearest-neighbor sampling. Use light foreground on dark background; resizing assumes the mask covers the same full scene. The manual adapter derives an exclusive XYXY bounding box. Limits: 15 MB per uploaded file, 12 megapixels, minimum 8 pixels per side.

For future automatic integration, send multipart `image` plus a `detections` JSON list instead of manual masks. Each record follows the existing Component3SeverityBridge contract: `defect_id`, integer `bbox_xyxy`, `mask_space` (`frame` or `bbox`), binary `mask`, optional `crack_mask`, and optional confidence fields. Maximum 20 defects per request. Both modes execute the same service. The frontend supports defect selection; automated C1/C2 orchestration remains a future team integration.

Responses retain the bridge's per-defect geometry, reliability, severity abstention and provenance, plus native 192×256 maps and coordinate metadata. Maps use full-frame coordinates. No physical measurements, calibrated uncertainty or physical severity are inferred by the frontend. The profile samples a selectable horizontal map row; the histogram includes positive map values. Research view shows pixel crack measurements when a crack mask is supplied. Ordinary RGB uploads do not provide a healthy-road RGB-D reconstruction.

Exports include a full JSON analysis and a printable visual report. Files/results stay in the current browser session; C3 does not persist uploads or run history.

Validation: `python -m pytest tests/test_c3_api.py` exercises the real checkpoint, manual and C1/C2 handoff adapters, invalid masks and missing-model responses. Frontend: `npm run lint` and `npm run build` from `frontend/web`.
