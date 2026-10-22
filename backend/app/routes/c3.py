"""Manual uploads and C1/C2 JSON handoff share the same inference service."""
import io
import json
import warnings

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from backend.app.services.c3 import surface_analysis

router = APIRouter(prefix='/c3', tags=['C3 surface geometry'])
MAX_BYTES = 15 * 1024 * 1024


def decode(data, mask=False):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                w, h = image.size
                if min(w, h) < 8 or w * h > 12_000_000:
                    raise ValueError('Images must be at least 8x8 and at most 12 megapixels')
                image = ImageOps.exif_transpose(image)
                if mask:
                    a = np.array(image.convert('L'))
                    # Preserve 0/1 masks; threshold ordinary grayscale/color images.
                    return a > (0 if a.max() <= 1 else 127)
                return np.array(image.convert('RGB'))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ValueError('Invalid or oversized image') from None


def align_mask(binary, shape):
    if binary.shape != shape:
        binary = np.array(Image.fromarray(binary).resize(
            (shape[1], shape[0]), Image.Resampling.NEAREST
        ), dtype=bool)
    return binary


async def read_image(upload, mask=False):
    data = await upload.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, 'Each upload must be at most 15 MB')
    return decode(data, mask)


@router.get('/model')
def model():
    return {'ready': surface_analysis.bridge is not None, 'physical_units_verified': False}


@router.post('/analyze')
async def analyze(image: UploadFile = File(...), mask: UploadFile | None = File(None),
                  crack_mask: UploadFile | None = File(None), detections: str | None = Form(None)):
    try:
        rgb = await read_image(image)
        if detections is not None:
            if mask is not None or crack_mask is not None:
                raise ValueError('Send manual masks or detections, not both')
            if len(detections) > 16_000_000:
                raise HTTPException(413, 'Detection payload too large')
            records = json.loads(detections)
            if not isinstance(records, list) or not 1 <= len(records) <= 20:
                raise ValueError('Provide between 1 and 20 detections')
        else:
            if mask is None:
                raise ValueError('Upload an aligned pothole mask')
            binary = align_mask(await read_image(mask, True), rgb.shape[:2])
            if not binary.any():
                raise ValueError('Pothole mask must contain damage pixels (light foreground on a dark background)')
            y, x = np.nonzero(binary)
            record = {'defect_id': 'manual-1', 'bbox_xyxy': [int(x.min()), int(y.min()), int(x.max())+1, int(y.max())+1], 'mask_space': 'frame', 'mask': binary}
            if crack_mask is not None:
                record['crack_mask'] = align_mask(await read_image(crack_mask, True), rgb.shape[:2])
            records = [record]
        # Run CPU inference outside the event loop.
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(surface_analysis.analyze, rgb, records)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from None
    except RuntimeError:
        raise HTTPException(503, 'C3 inference unavailable. Check the model bundle and server logs.') from None
