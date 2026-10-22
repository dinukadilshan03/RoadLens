
import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    binary_erosion,
    label,
    convolve
)
from skimage.morphology import skeletonize


def _mask(x, name):

    a = np.asarray(x)

    if (
        a.ndim != 2
        or (
            not np.issubdtype(a.dtype, np.number)
            and a.dtype != bool
        )
    ):
        raise ValueError(
            f"{name} must be a 2-D numeric/bool array"
        )

    if not np.isfinite(a).all():
        raise ValueError(
            f"{name} has nonfinite values"
        )

    return a > 0


def pixel_measurements(
    pothole_mask,
    crack_mask=None,
    crack_confidence=None
):

    m = _mask(pothole_mask, "pothole_mask")

    if m.size > 12_000_000:
        raise ValueError("Image too large")

    if not m.any():

        return {
            "status": "EMPTY_MASK",
            "footprint_area_px2": 0,
            "components_8": 0,
            "crack": None,
            "physical_units_available": False
        }

    st = np.ones((3, 3), dtype=np.uint8)

    components = int(
        label(m, structure=st)[1]
    )

    boundary = int(
        (
            m & ~binary_erosion(
                m,
                structure=st
            )
        ).sum()
    )

    yy, xx = np.nonzero(m)

    centered = np.stack(
        [
            xx.astype(float) - xx.mean(),
            yy.astype(float) - yy.mean()
        ],
        axis=1
    )

    covariance = (
        centered.T @ centered
        / max(len(centered), 1)
    )

    eigenvectors = np.linalg.eigh(
        covariance
    )[1][:, ::-1]

    projections = centered @ eigenvectors

    spans = np.ptp(
        projections,
        axis=0
    ) + 1.0

    result = {
        "status": "PIXEL_GEOMETRY_ONLY",
        "footprint_area_px2": int(m.sum()),
        "boundary_pixels_approx": boundary,
        "components_8": components,
        "projected_major_extent_px": float(
            max(spans)
        ),
        "projected_minor_extent_px": float(
            min(spans)
        ),
        "physical_units_available": False,
        "crack": None
    }

    if crack_mask is None:

        if crack_confidence is not None:
            raise ValueError(
                "Confidence supplied without crack mask"
            )

        return result

    c = _mask(crack_mask, "crack_mask")

    if c.shape != m.shape:
        raise ValueError(
            "Crack mask shape mismatch"
        )

    avg_conf = None

    if crack_confidence is not None:

        confidence = np.asarray(
            crack_confidence,
            dtype=float
        )

        if (
            confidence.shape != c.shape
            or not np.isfinite(confidence).all()
            or np.any(
                (confidence < 0)
                | (confidence > 1)
            )
        ):
            raise ValueError(
                "Invalid crack confidence map"
            )

        if c.any():

            avg_conf = float(
                np.mean(confidence[c])
            )

    # ------------------------------------------------
    # CRACK SKELETON AND TOPOLOGY
    # ------------------------------------------------

    skeleton = skeletonize(c)

    horizontal = int(
        np.sum(
            skeleton[:, :-1]
            & skeleton[:, 1:]
        )
    )

    vertical = int(
        np.sum(
            skeleton[:-1, :]
            & skeleton[1:, :]
        )
    )

    diagonal1 = int(
        np.sum(
            skeleton[:-1, :-1]
            & skeleton[1:, 1:]
        )
    )

    diagonal2 = int(
        np.sum(
            skeleton[1:, :-1]
            & skeleton[:-1, 1:]
        )
    )

    skeleton_length = float(
        horizontal
        + vertical
        + np.sqrt(2) * (
            diagonal1 + diagonal2
        )
    )

    neighbours = (
        convolve(
            skeleton.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            mode="constant",
            cval=0
        )
        - skeleton.astype(np.uint8)
    )

    endpoints = int(
        (
            skeleton
            & (neighbours == 1)
        ).sum()
    )

    junction_pixels = (
        skeleton
        & (neighbours >= 3)
    )

    junctions = int(
        label(
            junction_pixels,
            structure=st
        )[1]
    )

    # Approximate crack width from local
    # distance to the segmented crack boundary.

    distance = distance_transform_edt(c)

    widths = (
        2.0 * distance[skeleton]
    )

    result["crack"] = {

        "area_px2": int(c.sum()),

        "components_8": int(
            label(c, structure=st)[1]
        ),

        "skeleton_edge_length_px_approx":
            skeleton_length,

        "skeleton_pixels": int(
            skeleton.sum()
        ),

        "endpoint_pixels_approx":
            endpoints,

        "junction_clusters_approx":
            junctions,

        "median_width_px_approx":
            (
                float(np.median(widths))
                if widths.size else None
            ),

        "p95_width_px_approx":
            (
                float(
                    np.percentile(widths, 95)
                )
                if widths.size else None
            ),

        "mean_supplied_confidence":
            avg_conf,

        "confidence_calibrated": False,

        "length_mm": None,

        "width_mm": None
    }

    return result


# ================================================================
# CAMERA DEPTH -> XYZ
# ================================================================

def xyz_from_depth(
    depth_m,
    fx, fy, cx, cy
):

    z = np.asarray(
        depth_m,
        dtype=np.float64
    )

    if (
        z.ndim != 2
        or not np.isfinite(z).all()
        or (z <= 0).any()
    ):
        raise ValueError(
            "Requires complete positive aligned "
            "metric depth"
        )

    if (
        not all(
            np.isfinite(q)
            for q in (fx, fy, cx, cy)
        )
        or fx <= 0
        or fy <= 0
    ):
        raise ValueError(
            "Invalid camera intrinsics"
        )

    yy, xx = np.indices(z.shape)

    X = (xx - cx) * z / fx

    Y = (yy - cy) * z / fy

    return np.stack(
        [X, Y, z],
        axis=-1
    )


# ================================================================
# CONDITIONAL ROAD-NORMAL METRIC GEOMETRY
# ================================================================

def metric_xyz_measurements(
    xyz_m,
    pothole_mask,
    *,
    evidence
):

    # These evidence flags must come from genuine
    # calibration/provenance records.
    #
    # Passing booleans does not independently
    # certify a real-world dataset.

    required = (
        "xyz_units_m_verified",
        "rgb_xyz_registration_verified",
        "camera_frame_origin_verified",
        "provenance_id"
    )

    if (
        not isinstance(evidence, dict)
        or not all(
            evidence.get(k)
            for k in required
        )
    ):
        raise PermissionError(
            "Meter units, registration, "
            "camera-frame origin and provenance "
            "are required"
        )

    p = np.asarray(
        xyz_m,
        dtype=np.float64
    )

    m = _mask(
        pothole_mask,
        "pothole_mask"
    )

    if (
        p.shape != m.shape + (3,)
        or not m.any()
    ):
        raise ValueError(
            "XYZ must align with nonempty mask"
        )

    valid = (
        np.isfinite(p).all(axis=-1)
        & (p[..., 2] > 0)
    )

    coverage = float(
        (valid & m).sum() / m.sum()
    )

    if coverage < 0.90:
        raise ValueError(
            "Insufficient valid XYZ in pothole"
        )

    # ------------------------------------------------
    # HEALTHY ROAD REFERENCE
    # ------------------------------------------------

    dist = distance_transform_edt(~m)

    ring = (
        (dist >= 8)
        & (dist <= 35)
        & valid
    )

    if ring.sum() < 500:
        raise ValueError(
            "Insufficient healthy-road support"
        )

    support = p[ring]

    if len(support) > 18000:

        indices = np.random.default_rng(
            20261021
        ).choice(
            len(support),
            18000,
            replace=False
        )

        support = support[indices]

    # ------------------------------------------------
    # ROBUST 3D PLANE ESTIMATION
    # ------------------------------------------------

    weights = np.ones(len(support))

    for _ in range(6):

        center = np.average(
            support,
            axis=0,
            weights=weights
        )

        q = support - center

        covariance = (
            (q * weights[:, None]).T @ q
            / max(weights.sum(), 1)
        )

        _, vectors = np.linalg.eigh(
            covariance
        )

        normal = vectors[:, 0]

        # Orient normal toward camera origin.
        if np.dot(normal, -center) < 0:
            normal = -normal

        offset = -float(
            np.dot(normal, center)
        )

        residuals = np.abs(
            support @ normal + offset
        )

        sigma = max(
            1.4826 * np.median(
                np.abs(
                    residuals
                    - np.median(residuals)
                )
            ),
            0.00001
        )

        weights = np.minimum(
            1.0,
            1.345 * sigma
            / (residuals + 1e-12)
        )

    # ------------------------------------------------
    # RAY-PLANE INTERSECTION
    # ------------------------------------------------

    denominator = np.einsum(
        "ijk,k->ij",
        p,
        normal
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore"
    ):

        factor = -offset / denominator

        reference = (
            p * factor[..., None]
        )

    usable = (
        valid
        & np.isfinite(reference).all(axis=-1)
        & np.isfinite(factor)
        & (factor > 0)
    )

    if (
        (usable & m).sum() / m.sum()
        < 0.90
    ):
        raise ValueError(
            "Invalid ray-plane intersections"
        )

    # ------------------------------------------------
    # ROAD-NORMAL DEPTH
    # ------------------------------------------------

    signed_distance = np.einsum(
        "ijk,k->ij",
        p - reference,
        normal
    )

    depression = np.maximum(
        0.0,
        -signed_distance
    )

    # ------------------------------------------------
    # PROJECTED ROAD-PLANE AREA
    # ------------------------------------------------

    safe = usable.copy()

    safe[0, :] = False
    safe[-1, :] = False
    safe[:, 0] = False
    safe[:, -1] = False

    safe[1:-1, 1:-1] &= (
        usable[:-2, 1:-1]
        & usable[2:, 1:-1]
        & usable[1:-1, :-2]
        & usable[1:-1, 2:]
    )

    Q = np.where(
        usable[..., None],
        reference,
        0.0
    )

    dv, du = np.gradient(
        Q,
        axis=(0, 1)
    )

    area_per_pixel = np.linalg.norm(
        np.cross(du, dv),
        axis=-1
    )

    area_mask = (
        m
        & safe
        & np.isfinite(area_per_pixel)
        & (area_per_pixel > 0)
    )

    if (
        area_mask.sum() / m.sum()
        < 0.90
    ):
        raise ValueError(
            "Insufficient reliable footprint area"
        )

    values = depression[m & usable]

    # Volume approximated by integrating
    # road-normal depression over projected area.

    volume = float(
        np.sum(
            depression[area_mask]
            * area_per_pixel[area_mask]
        )
    )

    return {

        "status":
            "CONDITIONAL_CALIBRATED_INPUT_NOT_FIELD_VALIDATED",

        "source_provenance_id":
            str(evidence["provenance_id"]),

        "pothole_depth_d95_m":
            float(
                np.percentile(values, 95)
            ),

        "pothole_depth_max_m":
            float(values.max()),

        "pothole_depth_mean_m":
            float(values.mean()),

        "projected_plane_area_m2":
            float(
                area_per_pixel[area_mask].sum()
            ),

        "depression_volume_m3":
            volume,

        "mask_xyz_coverage":
            coverage,

        "area_support_fraction":
            float(
                area_mask.sum() / m.sum()
            ),

        "road_fit_healthy_median_abs_m":
            float(
                np.median(
                    np.abs(
                        support @ normal + offset
                    )
                )
            ),

        "physical_accuracy_independently_validated":
            False,

        "road_model":
            "Local 3D plane and "
            "road-normal ray-plane integration"
    }
