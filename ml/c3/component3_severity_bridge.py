
from pathlib import Path
import importlib.util


class Component3SeverityBridge:

    def __init__(
        self,
        root,
        device="cpu"
    ):

        root = Path(root)

        spec = importlib.util.spec_from_file_location(
            "c320_underlying_bridge",
            root / "component3_bridge.py"
        )

        module = importlib.util.module_from_spec(
            spec
        )

        spec.loader.exec_module(module)

        policy_spec = (
            importlib.util.spec_from_file_location(
                "c320_field_policy",
                root / "severity_policy.py"
            )
        )

        policy = importlib.util.module_from_spec(
            policy_spec
        )

        policy_spec.loader.exec_module(
            policy
        )

        self.bridge = module.Component3Bridge(
            root,
            device=device
        )

        self.policy = policy

    def predict(
        self,
        rgb_uint8,
        detections,
        *,
        include_maps=False,
        field_measurements=None
    ):

        if (
            field_measurements is not None
            and not isinstance(
                field_measurements,
                dict
            )
        ):

            raise ValueError(
                "field_measurements must map "
                "defect_id to field record"
            )

        result = self.bridge.predict(
            rgb_uint8,
            detections,
            include_maps=include_maps
        )

        ids = {

            d["defect_id"]

            for d in result["defects"]
        }

        if (
            field_measurements
            and not set(
                field_measurements
            ).issubset(ids)
        ):

            raise ValueError(
                "Field measurement contains "
                "an unknown defect_id"
            )

        for defect in result["defects"]:

            field = (

                None

                if field_measurements is None

                else field_measurements.get(
                    defect["defect_id"]
                )
            )

            defect[
                "field_reference_severity"
            ] = (
                self.policy.evaluate_field_reference(
                    field
                )
            )

            # Model severity must still abstain.
            assert (
                defect["severity"]["physical_level"]
                is None
            )

            assert (
                defect["severity"]["visual_level"]
                is None
            )

        result[
            "field_reference_policy_version"
        ] = self.policy.POLICY_VERSION

        result[
            "field_reference_is_model_output"
        ] = False

        return result
