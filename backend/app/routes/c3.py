"""Manual uploads and C1/C2 JSON handoff share the same inference service."""

import io
import json
import warnings

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from backend.app.services.c3 import surface_analysis

router = APIRouter(prefix="/c3", tags=["C3 surface geometry"])
MAX_BYTES = 15 * 1024 * 1024


@router.get("/research/{artifact}")
def research(artifact: str):
    from fastapi.responses import FileResponse
    from backend.app.services.c3 import BUNDLE

    allowed = {
        "stages": "research_only/C3-28_reference_NOT_DEPLOYABLE/reconstruction_six_stage.png",
        "comparison": "research_only/C3-28_reference_NOT_DEPLOYABLE/reference_vs_damaged_surface.png",
        "gate": "research_only/C3-28_reference_NOT_DEPLOYABLE/model_selection_correction_v1.json",
        "pothole-evaluation": "pothole_visual_severity/evaluation_metrics.json",
    }
    relative = allowed.get(artifact)
    if relative is None or not (BUNDLE / relative).is_file():
        raise HTTPException(404, "Research artifact unavailable")
    return FileResponse(BUNDLE / relative)


def decode(data, mask=False):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                w, h = image.size
                if min(w, h) < 8 or w * h > 12_000_000:
                    raise ValueError(
                        "Images must be at least 8x8 and at most 12 megapixels"
                    )
                image = ImageOps.exif_transpose(image)
                if mask:
                    color = np.array(image.convert("RGB"))
                    if not (
                        np.array_equal(color[..., 0], color[..., 1])
                        and np.array_equal(color[..., 1], color[..., 2])
                    ):
                        raise ValueError(
                            "Upload a binary damage mask, not a photograph or colored annotation"
                        )
                    a = color[..., 0]
                    if not np.isin(a, [0, 1, 255]).all():
                        raise ValueError(
                            "Damage masks must contain only black and white pixels (0/1 or 0/255)"
                        )
                    if not (a > 0).any() or (a > 0).all():
                        raise ValueError(
                            "Damage masks need both marked damage and an unmarked background"
                        )
                    return a > 0
                return np.array(image.convert("RGB"))
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        raise ValueError("Invalid or oversized image") from None


def align_mask(binary, shape, name="Damage mask"):
    if binary.shape != shape:
        raise ValueError(
            f"{name} is {binary.shape[1]} x {binary.shape[0]} pixels, but the road photo is "
            f"{shape[1]} x {shape[0]} pixels. Choose the mask made for this same photo "
            "at its original size. To analyze a different road photo, upload it separately with its own mask."
        )
    return binary


async def read_image(upload, mask=False):
    data = await upload.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Each upload must be at most 15 MB")
    return decode(data, mask)


@router.get("/model")
def model():
    return {
        "ready": surface_analysis.bridge is not None,
        "visual_models": {kind: kind in surface_analysis.visual_models for kind in ("pothole", "crack")},
        "physical_units_verified": False,
    }


@router.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    mask: UploadFile | None = File(None),
    crack_mask: UploadFile | None = File(None),
    detections: str | None = Form(None),
    field_measurements: str | None = Form(None),
):
    try:
        rgb = await read_image(image)
        input_warnings = []
        if detections is not None:
            if mask is not None or crack_mask is not None:
                raise ValueError("Send manual masks or detections, not both")
            if len(detections) > 16_000_000:
                raise HTTPException(413, "Detection payload too large")
            records = json.loads(detections)
            if not isinstance(records, list) or not 1 <= len(records) <= 20:
                raise ValueError("Provide between 1 and 20 detections")
        else:
            if mask is None and crack_mask is None:
                raise ValueError("Upload at least one pothole or crack damage mask")
            records = []
            for kind, upload in [("pothole", mask), ("crack", crack_mask)]:
                if upload is not None:
                    name = f"{kind.title()} mask"
                    try:
                        binary = align_mask(await read_image(upload, True), rgb.shape[:2], name)
                    except ValueError as exc:
                        input_warnings.append({"damage_type": kind, "message": f"{name} was not analyzed. {exc}"})
                        continue
                    records.append(
                        {
                            "defect_id": kind,
                            "damage_type": kind,
                            "mask_space": "frame",
                            "mask": binary,
                            "bbox_xyxy": [0, 0, rgb.shape[1], rgb.shape[0]],
                        }
                    )
            if not records:
                raise ValueError(" ".join(item["message"] for item in input_warnings))
        # Run CPU inference outside the event loop.
        from starlette.concurrency import run_in_threadpool

        fields = (
            json.loads(field_measurements) if field_measurements is not None else None
        )
        result = await run_in_threadpool(surface_analysis.analyze, rgb, records, fields)
        result["input_warnings"] = input_warnings
        if input_warnings:
            result["summary"] += " Some uploaded masks were not analyzed. See the input messages."
        return result
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from None
    except RuntimeError:
        raise HTTPException(
            503,
            "Pothole analysis is temporarily unavailable. Please try again later; crack-only analysis is still available.",
        ) from None


@router.post("/report")
async def report(
    image: UploadFile = File(...),
    mask: UploadFile | None = File(None),
    crack_mask: UploadFile | None = File(None),
    detections: str | None = Form(None),
    field_measurements: str | None = Form(None),
):
    from fastapi.responses import Response
    from starlette.concurrency import run_in_threadpool
    from backend.app.services.c3_report import generate_report

    result = await analyze(image, mask, crack_mask, detections, field_measurements)
    await image.seek(0)
    rgb = await read_image(image)
    pdf = await run_in_threadpool(generate_report, rgb, result)
    return Response(
        pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="roadlens-road-damage.pdf"'
        },
    )


async def independent_assessment(image, mask, crack_mask, crack_image, crack_matches_image=False):
    """Separate pothole RGB/mask and independent crack input coordinate systems."""
    from starlette.concurrency import run_in_threadpool

    results, sources, input_warnings = [], {}, []
    model_unavailable = False
    if mask is not None:
        if image is None:
            input_warnings.append({"damage_type": "pothole", "message": "Add the pothole road photo to analyze its mask."})
        else:
            try:
                rgb = await read_image(image)
                binary = align_mask(await read_image(mask, True), rgb.shape[:2], "Pothole mask")
                records = [{"defect_id": "pothole", "damage_type": "pothole", "bbox_xyxy": [0, 0, rgb.shape[1], rgb.shape[0]], "mask_space": "frame", "mask": binary}]
                result = await run_in_threadpool(surface_analysis.analyze, rgb, records)
                results.append(("pothole", result))
                sources["pothole"] = rgb
            except ValueError as exc:
                input_warnings.append({"damage_type": "pothole", "message": f"Pothole was not analyzed. {exc}"})
            except RuntimeError:
                model_unavailable = True
                input_warnings.append({"damage_type": "pothole", "message": "Pothole analysis is temporarily unavailable. The separate crack input can still be analyzed."})
    if crack_mask is not None:
        try:
            binary = await read_image(crack_mask, True)
            if crack_image is None and image is not None and (crack_matches_image or mask is None):
                await image.seek(0)
                crack_image = image
            shape_only = crack_image is None
            if shape_only:
                # A mask canvas is used only for topology/visualization; never RGB model inference.
                rgb = np.repeat((binary.astype(np.uint8) * 255)[..., None], 3, axis=2)
            else:
                rgb = await read_image(crack_image)
                binary = align_mask(binary, rgb.shape[:2], "Crack mask")
            records = [{"defect_id": "crack", "damage_type": "crack", "bbox_xyxy": [0, 0, rgb.shape[1], rgb.shape[0]], "mask_space": "frame", "mask": binary}]
            result = await run_in_threadpool(surface_analysis.analyze, rgb, records, shape_only=shape_only)
            results.append(("crack", result))
            sources["crack"] = rgb
        except ValueError as exc:
            input_warnings.append({"damage_type": "crack", "message": f"Crack was not analyzed. {exc}"})
    if not results:
        if model_unavailable:
            raise RuntimeError("Pothole analysis is temporarily unavailable")
        raise ValueError(" ".join(item["message"] for item in input_warnings) or "Upload a pothole photo with its mask, or a separate crack mask.")
    defects = []
    for group, result in results:
        for defect in result["defects"]:
            defect.update(source_group=group, source_width=result["frame_width"], source_height=result["frame_height"])
            defects.append(defect)
    if len(defects) > 20:
        raise ValueError("Analyze at most 20 separate damaged areas at a time")
    first = results[0][1]
    result = {
        "frame_width": first["frame_width"], "frame_height": first["frame_height"],
        "defects": defects, "defect_count": len(defects), "input_warnings": input_warnings,
        "summary": f"{len(defects)} damaged area(s) analyzed. Pothole and crack inputs are assessed separately.",
    }
    return result, sources


@router.post("/assess")
async def assess(
    image: UploadFile | None = File(None),
    mask: UploadFile | None = File(None),
    crack_mask: UploadFile | None = File(None),
    crack_image: UploadFile | None = File(None),
    crack_matches_image: bool = Form(False),
):
    try:
        result, _ = await independent_assessment(image, mask, crack_mask, crack_image, crack_matches_image)
        return result
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from None
    except RuntimeError:
        raise HTTPException(503, "Analysis is temporarily unavailable. Please try again later.") from None


@router.post("/assessment-report")
async def assessment_report(
    image: UploadFile | None = File(None),
    mask: UploadFile | None = File(None),
    crack_mask: UploadFile | None = File(None),
    crack_image: UploadFile | None = File(None),
    crack_matches_image: bool = Form(False),
):
    from fastapi.responses import Response
    from starlette.concurrency import run_in_threadpool
    from backend.app.services.c3_report import generate_report

    try:
        result, sources = await independent_assessment(image, mask, crack_mask, crack_image, crack_matches_image)
        pdf = await run_in_threadpool(generate_report, next(iter(sources.values())), result, sources)
        return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="roadlens-road-damage.pdf"'})
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from None
    except RuntimeError:
        raise HTTPException(503, "Analysis is temporarily unavailable. Please try again later.") from None
