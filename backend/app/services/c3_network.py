"""Whole-mask topology, conservative cleanup, and inspectable branches.

C3-26 cohort artifacts are required before assigning research-relative bands.
Never substitute arbitrary thresholds for the missing reference distribution.
"""
import numpy as np
from scipy.ndimage import label, convolve, binary_dilation
from skimage.morphology import skeletonize
from ml.c3.geometry_measurements import pixel_measurements


def analyze_network(mask, encode):
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, count = label(mask, structure)
    sizes = np.bincount(labels.ravel())
    # Retain all supplied damage in original measurements. Only isolated pixels
    # are excluded from the secondary topology view; thin real cracks survive.
    clean = mask & (sizes[labels] > 1)
    skeleton = skeletonize(clean)
    degree = convolve(skeleton.astype(np.uint8), structure, mode="constant") - skeleton
    junction = skeleton & (degree >= 3)
    # Adjacent junction pixels represent one intersection, not many branches.
    junction_zones = binary_dilation(junction, structure=structure) & skeleton
    branch_labels, branch_count = label(skeleton & ~junction_zones, structure)
    branch_sizes = np.bincount(branch_labels.ravel())
    branches = []
    def overlay(region):
        yy, xx = np.nonzero(region)
        x0, y0, x1, y1 = int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1
        rgba = np.zeros((y1-y0, x1-x0, 4), dtype=np.uint8)
        rgba[region[y0:y1, x0:x1]] = (230, 35, 45, 255)
        return [x0, y0, x1, y1], encode(rgba)
    removed_spur_pixels = 0
    for i in range(1, branch_count + 1):
        branch = branch_labels == i
        # Remove only a one-pixel terminal stub attached to a junction.
        if branch_sizes[i] == 1 and (branch & (degree == 1)).any() and (binary_dilation(branch) & junction_zones).any():
            skeleton[branch] = False
            removed_spur_pixels += 1
            continue
        bbox, image = overlay(branch)
        branches.append({"branch_id": len(branches)+1, "bbox_xyxy": bbox, "overlay": image, "pixels": int(branch.sum())})
    regions = []
    for i in range(1, count+1):
        bbox, region_overlay = overlay(labels == i)
        regions.append({"region_id": i, "bbox_xyxy": bbox, "pixels": int(sizes[i]), "overlay": region_overlay})
    incomplete = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    return {
        "connected_regions": int(count), "regions": regions, "branches": branches,
        "branched": bool(junction.any()), "visible_pattern_may_be_incomplete": incomplete,
        "original_measurements": pixel_measurements(mask, mask)["crack"],
        "cleaned_measurements": pixel_measurements(clean, clean)["crack"],
        "cleaning": {"removed_isolated_pixels": int(mask.sum()-clean.sum()), "removed_terminal_spur_pixels": removed_spur_pixels,
                     "policy": "Original measurements retained; only isolated pixels and one-pixel terminal skeleton stubs excluded from branch inspection."},
        "complexity": {"method": "C3-26", "label": "Needs Review", "physical_severity": None,
            "explanation": "Research-relative Lower / Moderate / Higher bands require the C3-26 reference cohort and scoring artifacts, which are not installed. No substitute severity thresholds are used. " + ("Damage touches the image boundary; the photographed pattern may be incomplete." if incomplete else "Only the visible photographed pattern is measured; damage may continue outside the photograph.")},
    }
