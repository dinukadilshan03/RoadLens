"""4-Part Precision Output Suite & Smart Routing Switch for Component C2.

Formats the segmentation outputs according to the RoadLens system contracts:
- Output A: High-Precision 2D Binary Mask (0/255 lossless PNG)
- Output B: 1-Pixel Skeleton Centerline (0/255 lossless PNG, continuous)
- Output C: Soft Confidence Map (float32 array [0,1] in .npz)
- Output D: High-Contrast Red Visual Overlay (RGB blended PNG for dashboard)

Implements the Smart Routing Switch to route pothole vs crack outputs to downstream
severity engines (Component C3) and municipal digital twin GIS (Component C4).

References:
- Proposal Section 2.2 Sub-Objective 5 (Smart Output Formatting Suite)
- Proposal Section 3.1 & Figure 3.3, Figure 4.1, Figure 4.2
- docs/interfaces.md Component Payload Requirements (C2)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from ml.c2.data.segmentation_dataset import morphological_thinning


class OutputSuiteGenerator:
    """Generates the 4-Part Precision Output Suite from network predictions."""

    @staticmethod
    def generate_suite(
        rgb_crop: np.ndarray,
        prob_map: np.ndarray,
        threshold: float = 0.5,
        damage_class: str = "crack",
        output_dir: Optional[Path] = None,
        prefix: str = "distress_001",
    ) -> Dict[str, Path | np.ndarray]:
        """Args:
            rgb_crop: (256, 256, 3) RGB uint8 image [0, 255]
            prob_map: (256, 256) Float probability map in [0.0, 1.0]
            threshold: Binarization cutoff (default: 0.5)
            damage_class: 'crack' (longitudinal, transverse, alligator) or 'pothole'
            output_dir: Optional directory to save output artifacts
            prefix: Filename prefix for artifacts
        Returns:
            Dictionary containing numpy arrays and saved artifact paths.
        """
        h, w = prob_map.shape[:2]

        # -------------------------------------------------------------
        # Output A: 2D High-Precision Binary Mask
        # Grayscale PNG, 0 background, 255 foreground (lossless)
        # -------------------------------------------------------------
        binary_mask = (prob_map >= threshold).astype(np.uint8) * 255

        # -------------------------------------------------------------
        # Output B: 1-Pixel Skeleton Centerline (via Smart Routing Switch)
        # Required for cracks; optional/zeroed for potholes
        # -------------------------------------------------------------
        is_crack = "crack" in damage_class.lower() or "d00" in damage_class.lower() or "d10" in damage_class.lower() or "d20" in damage_class.lower()
        if is_crack:
            skeleton_thin = morphological_thinning(binary_mask)
            skeleton_mask = (skeleton_thin > 0).astype(np.uint8) * 255
        else:
            skeleton_mask = np.zeros((h, w), dtype=np.uint8)

        # -------------------------------------------------------------
        # Output C: Soft Confidence Map
        # Numeric array [0.0, 1.0], float32 in .npz format (allow_pickle=False)
        # -------------------------------------------------------------
        confidence_map = prob_map.astype(np.float32)

        # -------------------------------------------------------------
        # Output D: High-Contrast Red Visual Overlay
        # Alpha-blended semi-transparent red distress contours on RGB crop
        # -------------------------------------------------------------
        overlay = rgb_crop.copy()
        red_tint = np.zeros_like(rgb_crop)
        red_tint[:, :, 0] = 255  # Red channel in RGB

        mask_bool = binary_mask > 0
        if np.any(mask_bool):
            # 60% original image + 40% red distress contour
            overlay[mask_bool] = (
                0.60 * rgb_crop[mask_bool] + 0.40 * red_tint[mask_bool]
            ).astype(np.uint8)

            # Draw crisp red outline along boundary contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 30, 30), thickness=1)

            # If crack, overlay skeleton in high-visibility bright yellow
            if is_crack:
                skel_bool = skeleton_mask > 0
                overlay[skel_bool] = [255, 255, 0]

        suite = {
            "mask_2d": binary_mask,
            "skeleton_1px": skeleton_mask,
            "confidence_npz": confidence_map,
            "visual_overlay": overlay,
            "is_crack": is_crack,
        }

        # Save artifacts to disk if output_dir is provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            mask_path = output_dir / f"{prefix}_mask_2d.png"
            skel_path = output_dir / f"{prefix}_skeleton_1px.png"
            conf_path = output_dir / f"{prefix}_confidence.npz"
            overlay_path = output_dir / f"{prefix}_visual_overlay.png"

            # Grayscale lossless PNGs
            cv2.imwrite(str(mask_path), binary_mask)
            cv2.imwrite(str(skel_path), skeleton_mask)

            # Compressed NPZ without pickle (interfaces.md requirement)
            np.savez_compressed(str(conf_path), confidence=confidence_map)

            # Visual overlay (RGB -> BGR for OpenCV write)
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            suite.update({
                "mask_path": mask_path,
                "skel_path": skel_path,
                "confidence_path": conf_path,
                "overlay_path": overlay_path,
            })

        return suite
