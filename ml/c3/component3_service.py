
from pathlib import Path
import json
import hashlib
import importlib.util

import numpy as np
import torch

from scipy.ndimage import label


def sha256(path):

    h = hashlib.sha256()

    with Path(path).open("rb") as f:

        for block in iter(
            lambda: f.read(1048576),
            b""
        ):
            h.update(block)

    return h.hexdigest()


class Component3Prototype:

    def __init__(self, bundle_path, device="cpu"):

        self.root = Path(bundle_path)

        meta = json.loads(
            (self.root / "bundle_config.json")
            .read_text(encoding="utf8")
        )

        self.meta = meta

        assert meta["physical_units_verified"] is False

        assert sha256(
            self.root / "selected_model.pt"
        ) == meta["selected_checkpoint_sha256"]

        assert sha256(
            self.root / "model_and_inference.py"
        ) == meta["architecture_sha256"]

        spec = importlib.util.spec_from_file_location(
            "crdl_bundle_model",
            self.root / "model_and_inference.py"
        )

        module = importlib.util.module_from_spec(spec)

        spec.loader.exec_module(module)

        self.module = module

        self.device = torch.device(device)

        ck = torch.load(
            self.root / "selected_model.pt",
            map_location="cpu",
            weights_only=True
        )

        assert ck["mode"] == meta["selected_mode"]

        self.scale_raw = float(ck["scale_raw"])

        self.prior_coef = ck["prior_coefficients"]

        self.model = module.RoadRelativeCRDL(
            base=ck["base"],
            mode=ck["mode"]
        )

        self.model.load_state_dict(
            ck["state_dict"],
            strict=True
        )

        self.model.to(self.device).eval()


    def predict(
        self,
        rgb,
        pothole_mask,
        crack_mask=None,
        include_map=False
    ):

        rgb = np.asarray(rgb)
        mask = np.asarray(pothole_mask)

        if (
            rgb.ndim != 3
            or rgb.shape[-1] != 3
            or rgb.dtype != np.uint8
        ):
            raise ValueError(
                "RGB must be HxWx3 uint8"
            )

        if mask.shape != rgb.shape[:2]:
            raise ValueError(
                "Pothole segmentation mask must align with RGB"
            )

        mask = (
            np.isfinite(mask)
            & (mask > 0)
        )

        # ----------------------------------------
        # Optional crack measurements (pixel only)
        # ----------------------------------------

        if crack_mask is not None:

            crack = np.asarray(crack_mask)

            if crack.shape != mask.shape:
                raise ValueError(
                    "Crack mask must align with RGB"
                )

            crack = (
                np.isfinite(crack)
                & (crack > 0)
            )

            comps = int(
                label(
                    crack,
                    np.ones(
                        (3, 3),
                        dtype=int
                    )
                )[1]
            )

            crack_result = {

                "crack_mask_area_px":
                    int(crack.sum()),

                "connected_components_8":
                    comps,

                "skeleton_length_px_approx":
                    None,

                "physical_length_mm":
                    None,

                "physical_width_mm":
                    None
            }

            try:

                from skimage.morphology import skeletonize

                sk = skeletonize(crack)

                edges = float(

                    (
                        sk[:, :-1]
                        & sk[:, 1:]
                    ).sum()

                    + (
                        sk[:-1, :]
                        & sk[1:, :]
                    ).sum()

                    + np.sqrt(2) * (
                        sk[:-1, :-1]
                        & sk[1:, 1:]
                    ).sum()

                    + np.sqrt(2) * (
                        sk[1:, :-1]
                        & sk[:-1, 1:]
                    ).sum()
                )

                crack_result[
                    "skeleton_length_px_approx"
                ] = edges

            except ImportError:

                pass

        else:

            crack_result = None

        # ----------------------------------------
        # Research reliability requirements
        # ----------------------------------------

        reasons = [

            "PHYSICAL_DEPTH_SCALE_UNVERIFIED",

            "SCENE_GENERALIZATION_UNVERIFIED",

            "REFERENCE_LABEL_COVERAGE_LIMITED",

            "UNCERTAINTY_NOT_CALIBRATED"
        ]

        result = {

            "schema_version":
                "crdl-c3/0.6-prototype",

            "model_version":
                self.meta["selected_mode"],

            "reliability_status":
                "REVIEW_ONLY",

            "reasons":
                reasons,

            "geometry_raw":
                None,

            "crack_geometry_px":
                crack_result,

            "physical_measurements": {

                "depth_d95_mm": None,
                "depth_max_mm": None,
                "area_m2": None,
                "volume_m3": None,
                "crack_length_mm": None
            },

            "severity": {

                "physical_level": None,
                "visual_level": None
            },

            "model_uncertainty":
                None
        }

        # ----------------------------------------
        # Empty mask safeguard
        # ----------------------------------------

        if not mask.any():

            result["reliability_status"] = (
                "UNSUPPORTED_EMPTY_MASK"
            )

            if include_map:

                result["raw_axial_depression_map"] = [

                    [0.0] * 256

                    for _ in range(192)
                ]

            return result

        # ----------------------------------------
        # Actual model inference
        # ----------------------------------------

        x, prior = self.module.prepare_inputs(

            rgb,
            mask,
            self.prior_coef,

            size_hw=(192, 256)
        )

        with torch.inference_mode():

            prediction = self.model(

                x[None].to(self.device),

                prior[None].to(self.device),

                self.scale_raw
            ).clamp(max=8)

            raw = (

                self.scale_raw

                * torch.expm1(prediction)

            )[0, 0].cpu().numpy().astype(
                np.float32
            )

        valid = (
            x[3].numpy() > 0
        )

        raw = np.where(
            valid,
            raw,
            0
        ).astype(np.float32)

        if (
            not np.isfinite(raw).all()
            or (raw < 0).any()
        ):
            raise RuntimeError(
                "Invalid model output"
            )

        values = raw[valid]

        result["geometry_raw"] = {

            "units":
                "UNVERIFIED_RAW_AXIAL_SENSOR_UNITS",

            "map_height":
                192,

            "map_width":
                256,

            "source_pothole_area_px":
                int(mask.sum()),

            "d95_raw":
                float(
                    np.percentile(
                        values,
                        95
                    )
                ),

            "mean_raw":
                float(values.mean()),

            "max_raw":
                float(values.max()),

            "trained_reference":
                "PothRGBD road-relative "
                "optical-axis residual"
        }

        if include_map:

            result["raw_axial_depression_map"] = (
                raw.tolist()
            )

        return result
