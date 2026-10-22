"""App adapter for build_crdl_3d_visualization.py supplied by the user.

Retain its paired-mask, padded ROI, native statistics and negative depression
convention. Normalize display coordinates only; no physical geometry is inferred.
"""

import numpy as np
from PIL import Image


def prepare_relative_surface(raw_map, full_mask):
    raw = np.asarray(raw_map, dtype=np.float32)
    mask = np.asarray(full_mask, dtype=bool)
    if raw.ndim != 2 or min(raw.shape) < 2 or mask.ndim != 2:
        raise ValueError("Relative surface requires a 2D prediction and paired mask")
    if not np.isfinite(raw).all() or (raw < -1e-6).any():
        raise ValueError("Relative surface prediction must be finite and nonnegative")
    h, w = raw.shape
    paired = np.asarray(
        Image.fromarray(mask.astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST),
        dtype=bool,
    )
    if not paired.any():
        return None  # Tiny regions can disappear on the model's native grid.
    raw = np.maximum(raw, 0)
    masked = np.where(paired, raw, 0.0)
    yy, xx = np.nonzero(paired)
    xmin, xmax = max(0, int(xx.min()) - 24), min(w, int(xx.max()) + 25)
    ymin, ymax = max(0, int(yy.min()) - 24), min(h, int(yy.max()) + 25)
    maximum = float(masked.max())
    step = 2
    ys, xs = np.arange(ymin, ymax, step), np.arange(xmin, xmax, step)
    sample = masked[np.ix_(ys, xs)]
    # Never replace a narrow real depression with a flat decimated surface.
    if maximum > 0 and not sample.any():
        step = 1
        ys, xs = np.arange(ymin, ymax), np.arange(xmin, xmax)
        sample = masked[np.ix_(ys, xs)]
    relative = sample / maximum if maximum > 0 else np.zeros_like(sample)
    image_width = xmax - xmin
    image_height = ymax - ymin
    longest = max(image_width, image_height)
    return {
        "x": (xs / (w - 1) * 100).tolist(),
        "y": (ys / (h - 1) * 100).tolist(),
        "z": (-relative).tolist(),
        "depression": relative.tolist(),
        "aspect_ratio": {
            "x": 1.38 * image_width / longest,
            "y": 1.38 * image_height / longest,
            "z": 0.5,
        },
        "has_depression": maximum > 0,
        "physical_scale_verified": False,
        "native_shape": [h, w],
        "sampling_step": step,
        "native_statistics": {
            "max_raw": maximum,
            "mean_raw": float(raw[paired].mean()),
            "d95_raw": float(np.percentile(raw[paired], 95)),
        },
    }
