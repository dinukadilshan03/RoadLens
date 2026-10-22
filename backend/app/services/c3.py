from contextlib import asynccontextmanager
import logging
from threading import Lock
from pathlib import Path
import os
import io
import base64
import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_erosion, distance_transform_edt
from skimage.morphology import skeletonize
from ml.c3.geometry_measurements import pixel_measurements
from ml.c3.component3_handoff_service import _binary, _bbox
from ml.c3.severity_policy import evaluate_field_reference
from backend.app.services.c3_surface import prepare_relative_surface
from backend.app.services.c3_network import analyze_network


def png_url(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def damage_visuals(mask, kind):
    outline = mask & ~binary_erosion(mask)
    color = (221, 163, 40) if kind == "pothole" else (46, 133, 220)

    def layer(region, rgb):
        out = np.zeros((*mask.shape, 4), dtype=np.uint8)
        out[region] = (*rgb, 255)
        return png_url(out)

    visuals = {"outline": layer(outline, color)}
    if kind == "crack":
        center = skeletonize(mask)
        distance = distance_transform_edt(mask)
        width = np.zeros((*mask.shape, 4), dtype=np.uint8)
        t = distance[mask] / max(1, distance.max())
        width[mask, 0] = (50 + 205 * t).astype(np.uint8)
        width[mask, 1] = (170 - 80 * t).astype(np.uint8)
        width[mask, 2] = (230 - 190 * t).astype(np.uint8)
        width[mask, 3] = 255
        visuals.update(centerline=layer(center, (46, 133, 220)), width=png_url(width))
    return visuals


def normalize_records(rgb, records):
    h, w = rgb.shape[:2]
    normalized, used = [], set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each detection must be an object")
        identifier = record.get("defect_id")
        if (
            not isinstance(identifier, str)
            or not 1 <= len(identifier) <= 100
            or identifier in used
        ):
            raise ValueError(
                "Detection identifiers must be unique nonempty strings of at most 100 characters"
            )
        used.add(identifier)
        kind = record.get("damage_type", "pothole")
        if kind not in ("pothole", "crack"):
            raise ValueError("Damage type must be pothole or crack")
        x0, y0, x1, y1 = _bbox(record.get("bbox_xyxy"), w, h)
        space = record.get("mask_space")
        if space not in ("frame", "bbox"):
            raise ValueError("Mask coordinates must be frame or bbox")
        binary = _binary(
            record["mask"],
            (h, w) if space == "frame" else (y1 - y0, x1 - x0),
            "Damage mask",
        )
        full = np.zeros((h, w), dtype=bool)
        if space == "bbox":
            full[y0:y1, x0:x1] = binary
        else:
            full = binary
            outside = full.copy()
            outside[y0:y1, x0:x1] = False
            if outside.any():
                raise ValueError(
                    "Damage mask extends outside its supplied bounding box"
                )
        categories = [(kind, full)]
        if record.get("crack_mask") is not None:
            if kind == "crack":
                raise ValueError(
                    "A crack record cannot also contain an associated crack mask"
                )
            categories.append(
                ("crack", _binary(record["crack_mask"], (h, w), "Crack mask"))
            )
        for category, binary in categories:
            labels, count = label(binary, structure=np.ones((3, 3)))
            if count == 0:
                raise ValueError(
                    "Every supplied damage mask must contain marked damage"
                )
            if (1 if category == "crack" else count) + len(normalized) > 20:
                raise ValueError("Analyze at most 20 separate damaged areas at a time")
            for index in range(1, (1 if category == "crack" else count) + 1):
                region = binary if category == "crack" else labels == index
                yy, xx = np.nonzero(region)
                rid = (
                    identifier
                    if (count == 1 or category == "crack") and category == kind
                    else f"{identifier}-{category}-{index}"
                )
                normalized.append(
                    {
                        **record,
                        "defect_id": rid,
                        "damage_type": category,
                        "mask": region,
                        "mask_space": "frame",
                        "bbox_xyxy": [
                            int(xx.min()),
                            int(yy.min()),
                            int(xx.max()) + 1,
                            int(yy.max()) + 1,
                        ],
                    }
                )
    if len({r["defect_id"] for r in normalized}) != len(normalized):
        raise ValueError(
            "Generated defect identifiers conflict; use different detection identifiers"
        )
    return normalized


from ml.c3.component3_severity_bridge import Component3SeverityBridge

logger = logging.getLogger(__name__)
BUNDLE = Path(
    os.environ.get("C3_BUNDLE_PATH", Path(__file__).resolve().parents[3] / "ml" / "c3")
)


class SurfaceAnalysis:
    def __init__(self):
        self.bridge = None
        self.visual_models = {}
        self.lock = Lock()

    def load(self):
        try:
            self.bridge = Component3SeverityBridge(BUNDLE, device="cpu")
        except Exception:
            self.bridge = None
            logger.exception("C3 geometry model failed to initialize")
        self.visual_models = {}
        for kind, folder in (
            ("pothole", "pothole_visual_severity"),
            ("crack", "crack_visual_severity"),
        ):
            try:
                if kind == "pothole":
                    from ml.c3.pothole_visual_severity.visual_severity_inference import VisualSeverityClassifier
                    factory = VisualSeverityClassifier
                else:
                    from ml.c3.crack_visual_severity.crack_visual_severity_service import CrackVisualSeverityService
                    factory = CrackVisualSeverityService
                model = factory(BUNDLE / folder, device="cpu")
                if kind == "crack":
                    model.fusion = None  # RGB-only was selected; fusion remains research-only.
                self.visual_models[kind] = model
            except Exception:
                logger.exception("C3 %s visual severity model failed to initialize", kind)

    def analyze(self, rgb, detections, field_measurements=None, *, shape_only=False):
        records = normalize_records(rgb, detections)
        if shape_only and any(record["damage_type"] != "crack" for record in records):
            raise ValueError("Mask-only analysis supports cracks only")
        if field_measurements is not None and not isinstance(field_measurements, dict):
            raise ValueError(
                "Field measurements must map defect identifiers to inspection records"
            )
        fields = field_measurements or {}
        if not set(fields).issubset({r["defect_id"] for r in records}):
            raise ValueError("Field measurement contains an unknown defect identifier")
        outputs, counts = [], {"pothole": 0, "crack": 0}
        for record in records:
            kind, region = record["damage_type"], record["mask"]
            field = fields.get(record["defect_id"])
            if kind == "crack" and field is not None:
                raise ValueError(
                    "Crack field assessment is not supported by the current severity policy"
                )
            reference = evaluate_field_reference(field)
            if kind == "pothole":
                if self.bridge is None:
                    raise RuntimeError("C3 model unavailable")
                clean = {
                    k: v
                    for k, v in record.items()
                    if k not in ("crack_mask", "crack_confidence_map")
                }
                with self.lock:
                    defect = self.bridge.predict(rgb, [clean], include_maps=True)[
                        "defects"
                    ][0]
            else:
                defect = {
                    "defect_id": record["defect_id"],
                    "source_bbox_xyxy": record["bbox_xyxy"],
                    "geometry_raw": None,
                    "pixel_measurements": pixel_measurements(region, region),
                    "severity": {"physical_level": None},
                    "physical_measurements": {"crack_width_mm": None},
                }
            network = analyze_network(region, png_url) if kind == "crack" else None
            if network:
                defect["crack_network"] = network
            counts[kind] += 1
            review = (
                defect.get("reliability_status") == "REVIEW_REQUIRED_INPUT_QA"
                or region.sum() / region.size > 0.6
                or region[0].any()
                or region[-1].any()
                or region[:, 0].any()
                or region[:, -1].any()
            )
            model = None if shape_only else self.visual_models.get(kind)
            visual = None
            if model is not None and kind != "crack":
                try:
                    with self.lock:
                        visual = model.predict(rgb, record["bbox_xyxy"])
                except Exception:
                    logger.exception("C3 %s visual severity inference failed", kind)
            regional_findings = []
            if kind == "crack" and model is not None:
                for item in [r for r in network["regions"] if r["pixels"] > 1][:20]:
                    try:
                        with self.lock:
                            finding = model.predict(rgb, item["bbox_xyxy"])
                        regional_findings.append({"region_id": item["region_id"], "bbox_xyxy": item["bbox_xyxy"], "prediction": finding})
                    except Exception:
                        logger.exception("C3 regional crack inference failed")
                        regional_findings.append({"region_id": item["region_id"], "bbox_xyxy": item["bbox_xyxy"], "prediction": None})
                # Only a single unbranched region can retain a crop-level rating.
                if len(regional_findings) == 1 and network["connected_regions"] == 1 and not network["branched"]:
                    visual = regional_findings[0]["prediction"]
            if network:
                network["regional_visual_findings"] = regional_findings
                network["regions_evaluated"] = len(regional_findings)
                network["regional_evaluation_limit"] = 20
                network["aggregation_validated"] = False
            network_review = bool(network and (network["branched"] or network["connected_regions"] > 1))
            visual_label = (
                visual.get("visual_severity", visual.get("visual_crack_severity"))
                if visual else None
            )
            needs_review = review or network_review or visual is None or visual.get("review_required", True)
            visual_level = "Needs Review" if needs_review else visual_label.title()
            review_reasons = []
            if review:
                review_reasons.append("Check that the white mask follows the damage. Damage touching the edge of the photo may be cut off.")
            if network and not shape_only and len(regional_findings) < network["connected_regions"]:
                review_reasons.append(f"Visual findings cover {len(regional_findings)} of {network['connected_regions']} connected regions. Isolated pixels are excluded and inference is limited to 20 regions; all damage remains in morphology measurements.")
            if network_review:
                review_reasons.append("C3-27 was trained on individual crack crops. Regional findings do not establish whole-network severity; aggregation has not been validated.")
            if shape_only:
                review_reasons.append("Add the crack's own road photo if you want an AI severity rating.")
            elif visual is None and not regional_findings:
                review_reasons.append("The severity model is unavailable. Please try again later.")
            elif visual and visual.get("review_required"):
                review_reasons.append("The model could not choose a clear severity rating. Please review this area.")
            if kind == "crack":
                morphology = defect["pixel_measurements"]["crack"]
                shape = (
                    f"{network['connected_regions']} connected region(s); visible centerline extent {morphology['skeleton_edge_length_px_approx']:.1f} px; "
                    f"median / p95 width {morphology['median_width_px_approx']:.1f} / {morphology['p95_width_px_approx']:.1f} px. "
                    f"The marked crack has approximately {morphology['endpoint_pixels_approx']} ends and "
                    f"{morphology['junction_clusters_approx']} branching points. "
                    "The centerline shows the path of the crack."
                )
            else:
                shape = "The depression map and surface profile show the estimated shape of the damaged area."
            level = reference["reference_severity_uncertainty_robust"]
            if reference["status"] == "REFERENCE_NOMINAL_UNCERTAINTY_UNREPORTED":
                level = reference["reference_severity_nominal"]
            explanation = (
                "The severity result is unavailable. Please review this area."
                if visual is None else
                ("Please review the marked area and severity result." if needs_review else
                 f"The model rates the visible {kind} damage as {visual_level.lower()}.")
            )
            visuals = damage_visuals(region, kind)
            insufficient = kind == "crack" and defect["pixel_measurements"]["crack"]["skeleton_edge_length_px_approx"] == 0
            if defect.get("raw_axial_depression_map") is not None:
                depression = np.asarray(defect["raw_axial_depression_map"])
                defect["relative_surface"] = prepare_relative_surface(depression, region)
                insufficient = defect["relative_surface"] is None
                if defect["relative_surface"] is not None:
                    shape = (
                        "The map shows where the road surface is estimated to dip. "
                        "Warmer colors show deeper areas. Move the line on the image "
                        "to see the surface profile at that position."
                    )
                maximum = max(float(depression.max()), 1e-12)
                t = np.clip(depression / maximum, 0, 1)
                rgba = np.stack(
                    [
                        70 + 185 * t,
                        160 - 105 * t,
                        150 - 110 * t,
                        np.where(depression > 0, 255, 0),
                    ],
                    axis=-1,
                ).astype(np.uint8)
                visuals["depression"] = png_url(rgba)
                bins, _ = np.histogram(
                    depression[depression > 0], bins=16, range=(0, maximum)
                )
                defect["display_chart"] = {
                    "profiles": (depression / maximum).tolist(),
                    "distribution": bins.tolist(),
                }
            if insufficient:
                needs_review = True
                visual_level = "Needs Review"
                explanation = "The marked area is too small for a clear assessment. Please upload a closer photo and its matching mask."
                review_reasons.append("The marked area is too small to show its shape clearly. Try a closer, sharper photo and its matching mask.")
            if shape_only and not insufficient:
                visual_level = "Add crack photo"
                explanation = "The crack network shape has been analyzed. Supply its matching road photograph for regional visual findings."
            if insufficient:
                reliability = "Insufficient input"
            elif review:
                reliability = "Needs review"
            elif shape_only:
                reliability = "Shape analyzed"
            else:
                reliability = "Needs review" if needs_review else "Analysis completed"
            defect.update(
                damage_type=kind,
                source_kind="mask" if shape_only else "photo",
                analysis_mode="MASK_SHAPE_ONLY" if shape_only else "RGB_AND_MASK",
                visual_severity=visual,
                field_reference_severity=reference,
                visuals=visuals,
                presentation={
                    "label": (("Crack Network" if sum(r["damage_type"] == "crack" for r in records) == 1 else f"Crack Network {counts[kind]}") if kind == "crack" else f"Pothole {counts[kind]}"),
                    "severity": visual_level,
                    "assessment": reliability,
                    "explanation": explanation,
                    "shape": shape,
                    "complexity": network["complexity"]["label"] if network else None,
                    "complexity_explanation": network["complexity"]["explanation"] if network else None,
                    "review_reasons": review_reasons,
                    "physical_severity": level.title() if level else "Not established",
                    "measurement": (
                        f"Measured deepest point: {reference['max_depth_mm']:g} mm - entered from a field inspection."
                        if reference["max_depth_mm"] is not None
                        else ""
                    ),
                    "measurement_note": (
                        "Measurement uncertainty was not reported."
                        if level and reference["uncertainty_mm"] is None
                        else (
                            "The supplied measurements need review before a severity rating can be assigned."
                            if field and not level
                            else ""
                        )
                    ),
                    "reliability": reliability,
                    "recommendation": "Use these findings to plan a closer inspection of the damaged area.",
                },
            )
            outputs.append(defect)
        return {
            "frame_width": rgb.shape[1],
            "frame_height": rgb.shape[0],
            "defect_count": len(outputs),
            "defects": outputs,
            "summary": f"{counts['pothole']} pothole(s) and {counts['crack']} crack network(s) analyzed. Physical inspection is recommended.",
            "map_coordinates": {
                "space": "full_frame",
                "width": 256,
                "height": 192,
                "physical_scale_verified": False,
            },
        }


surface_analysis = SurfaceAnalysis()


@asynccontextmanager
async def lifespan(app):
    try:
        surface_analysis.load()
    except Exception:
        logger.exception("C3 model failed to initialize")
    yield
