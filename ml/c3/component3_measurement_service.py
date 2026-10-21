
from pathlib import Path
import importlib.util


class Component3MeasurementService:

    def __init__(
        self,
        bundle_root,
        device="cpu"
    ):

        self.root = Path(bundle_root)

        spec = importlib.util.spec_from_file_location(
            "crdl_c317_base",
            self.root / "component3_service.py"
        )

        base_module = importlib.util.module_from_spec(
            spec
        )

        spec.loader.exec_module(
            base_module
        )

        spec2 = importlib.util.spec_from_file_location(
            "crdl_c317_geometry",
            self.root / "geometry_measurements.py"
        )

        geo_module = importlib.util.module_from_spec(
            spec2
        )

        spec2.loader.exec_module(
            geo_module
        )

        self.base = base_module.Component3Prototype(
            self.root,
            device=device
        )

        self.geometry = geo_module

    def predict(
        self,
        rgb,
        pothole_mask,
        crack_mask=None,
        crack_confidence=None,
        include_map=False
    ):

        pixels = self.geometry.pixel_measurements(
            pothole_mask,
            crack_mask,
            crack_confidence
        )

        # Actual trained CRDL inference.
        result = self.base.predict(
            rgb,
            pothole_mask,
            include_map=include_map
        )

        result["measurement_schema"] = (
            "crdl-c3/0.7-pixel-and-conditional-3d"
        )

        result["pixel_measurements"] = pixels

        result["physical_measurement_evidence"] = (
            "NOT_VERIFIED_FOR_POTHRGBD"
        )

        # Physical measurements and severity
        # remain evidence-gated by C3-14.
        return result
