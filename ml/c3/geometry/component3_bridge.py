
from pathlib import Path
import importlib.util


class Component3Bridge:

    """Development-stage C1/C2-to-C3 schema adapter."""

    def __init__(self, root, device='cpu'):

        root = Path(root)

        spec = importlib.util.spec_from_file_location(
            'c319_saved_handoff',
            root / 'component3_handoff_service.py'
        )

        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        self.service = mod.Component3Handoff(
            root,
            device=device
        )

    def predict(
        self,
        rgb_uint8,
        detections,
        include_maps=False
    ):

        if not isinstance(detections, (list, tuple)):
            raise ValueError('detections must be a list')

        adapted = []

        for d in detections:

            if not isinstance(d, dict):
                raise ValueError('Invalid detection record')

            required = (
                'defect_id',
                'bbox_xyxy',
                'mask_space',
                'mask'
            )

            missing = [
                k for k in required
                if k not in d
            ]

            if missing:
                raise ValueError(
                    'Missing required keys: '
                    + ','.join(missing)
                )

            if any(
                k in d
                for k in (
                    'mask_probabilities',
                    'mask_logits',
                    'bbox_xywh',
                    'bbox_xywh_norm'
                )
            ):
                raise ValueError(
                    'Convert probabilities and bbox formats '
                    'explicitly before Component 3'
                )

            if (
                'score' in d
                and 'detection_confidence' in d
            ):
                raise ValueError(
                    'Ambiguous detection confidence'
                )

            if (
                'segmentation_score' in d
                and 'segmentation_confidence' in d
            ):
                raise ValueError(
                    'Ambiguous segmentation confidence'
                )

            p = {
                k: d[k]
                for k in required
            }

            p['detection_confidence'] = d.get(
                'detection_confidence',
                d.get('score')
            )

            p['segmentation_confidence'] = d.get(
                'segmentation_confidence',
                d.get('segmentation_score')
            )

            for k in (
                'crack_mask',
                'crack_confidence_map'
            ):
                if k in d:
                    p[k] = d[k]

            adapted.append(p)

        return self.service.predict_frame(
            rgb_uint8,
            adapted,
            include_maps=include_maps
        )
