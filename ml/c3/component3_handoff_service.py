
from pathlib import Path
import importlib.util
import numpy as np

POLICY_VERSION = 'C3-18.1.0'
SCHEMA_VERSION = 'crdl-c3/0.8-development'


def _binary(a, shape, field):

    a = np.asarray(a)

    if a.ndim != 2 or a.shape != tuple(shape):

        raise ValueError(
            f'{field} must be a 2D mask of shape {shape}'
        )

    if (
        a.dtype != bool
        and not np.issubdtype(a.dtype, np.number)
    ):

        raise ValueError(
            f'{field} must be numeric or bool'
        )

    if not np.isfinite(a).all():

        raise ValueError(
            f'{field} contains non-finite values'
        )

    # No silent thresholding of probability maps.
    if not np.all(
        (a == 0) | (a == 1) | (a == 255)
    ):

        raise ValueError(
            f'{field} requires binary 0/1 or 0/255; '
            'threshold C2 scores explicitly'
        )

    return a != 0


def _confidence(v, field):

    if v is None:
        return None

    if isinstance(v, (bool, np.bool_)):

        raise ValueError(
            f'{field} must be numeric'
        )

    if not np.isscalar(v):

        raise ValueError(
            f'{field} must be a scalar in [0,1]'
        )

    value = float(v)

    if not np.isfinite(value) or not 0 <= value <= 1:

        raise ValueError(
            f'{field} must be in [0,1]'
        )

    return value


def _rgb(a):

    a = np.asarray(a)

    if (
        a.ndim != 3
        or a.shape[2] != 3
        or a.dtype != np.uint8
        or a.shape[0] < 8
        or a.shape[1] < 8
        or a.shape[0] * a.shape[1] > 12_000_000
    ):

        raise ValueError(
            'RGB must be HxWx3 uint8 '
            'within supported size'
        )

    return a


def _bbox(box, w, h):

    if not isinstance(box, (list, tuple)) or len(box) != 4:

        raise ValueError(
            'bbox_xyxy requires four pixel integers'
        )

    if any(
        isinstance(v, (bool, np.bool_))
        or not isinstance(v, (int, np.integer))
        for v in box
    ):

        raise ValueError(
            'bbox_xyxy must contain integers'
        )

    x0, y0, x1, y1 = map(int, box)

    if not (
        0 <= x0 < x1 <= w
        and 0 <= y0 < y1 <= h
    ):

        raise ValueError(
            'bbox is outside image or empty'
        )

    return x0, y0, x1, y1


class Component3Handoff:

    def __init__(self, bundle_path, device='cpu'):

        self.root = Path(bundle_path)

        spec = importlib.util.spec_from_file_location(
            'crdl_c318_measurement',
            self.root / 'component3_measurement_service.py'
        )

        mod = importlib.util.module_from_spec(spec)

        spec.loader.exec_module(mod)

        self.backend = mod.Component3MeasurementService(
            self.root,
            device=device
        )

    def predict_frame(
        self,
        rgb,
        defects,
        *,
        include_maps=False
    ):

        frame = _rgb(rgb)

        h, w = frame.shape[:2]

        if (
            not isinstance(defects, (list, tuple))
            or len(defects) > 100
        ):

            raise ValueError(
                'defects must be a list with <=100 entries'
            )

        used_ids = set()
        outputs = []

        for defect in defects:

            if not isinstance(defect, dict):

                raise ValueError(
                    'Each defect must be an object'
                )

            defect_id = defect.get('defect_id')

            if (
                not isinstance(defect_id, str)
                or not defect_id
                or defect_id in used_ids
                or len(defect_id) > 128
            ):

                raise ValueError(
                    'defect_id must be unique and nonempty'
                )

            used_ids.add(defect_id)

            space = defect.get('mask_space')

            if space not in ('frame', 'bbox'):

                raise ValueError(
                    'mask_space must be frame or bbox'
                )

            bbox = _bbox(
                defect.get('bbox_xyxy'),
                w,
                h
            )

            x0, y0, x1, y1 = bbox

            # ----------------------------------------
            # Reconstruct full-frame mask
            # ----------------------------------------

            if space == 'bbox':

                roi = _binary(
                    defect['mask'],
                    (y1-y0, x1-x0),
                    'ROI pothole mask'
                )

                full_mask = np.zeros(
                    (h, w),
                    dtype=bool
                )

                full_mask[y0:y1, x0:x1] = roi

            else:

                full_mask = _binary(
                    defect['mask'],
                    (h, w),
                    'frame pothole mask'
                )

            # ----------------------------------------
            # Optional associated crack mask
            # ----------------------------------------

            crack = defect.get('crack_mask')

            confidence_map = defect.get(
                'crack_confidence_map'
            )

            if crack is not None:

                crack = _binary(
                    crack,
                    (h, w),
                    'associated crack mask'
                )

            if confidence_map is not None:

                confidence_map = np.asarray(
                    confidence_map,
                    dtype=float
                )

                if (
                    crack is None
                    or confidence_map.shape != (h, w)
                    or not np.isfinite(
                        confidence_map
                    ).all()
                    or (confidence_map < 0).any()
                    or (confidence_map > 1).any()
                ):

                    raise ValueError(
                        'Crack confidence map requires '
                        'aligned crack mask and values [0,1]'
                    )

            detection_confidence = _confidence(
                defect.get('detection_confidence'),
                'detection_confidence'
            )

            segmentation_confidence = _confidence(
                defect.get('segmentation_confidence'),
                'segmentation_confidence'
            )

            # ----------------------------------------
            # ACTUAL TRAINED CRDL INFERENCE
            # ----------------------------------------

            result = self.backend.predict(
                frame,
                full_mask,
                crack_mask=crack,
                crack_confidence=confidence_map,
                include_map=include_maps
            )

            # ----------------------------------------
            # INPUT QUALITY DIAGNOSTICS
            # ----------------------------------------

            mask_count = int(full_mask.sum())

            flags = []

            if mask_count == 0:

                flags.append(
                    'EMPTY_DEFECT_MASK'
                )

            else:

                if mask_count < 200:

                    flags.append(
                        'SMALL_MASK_HEURISTIC'
                    )

                if mask_count / (h * w) > 0.60:

                    flags.append(
                        'LARGE_MASK_HEURISTIC'
                    )

                if (
                    full_mask[0].any()
                    or full_mask[-1].any()
                    or full_mask[:, 0].any()
                    or full_mask[:, -1].any()
                ):

                    flags.append(
                        'MASK_TOUCHES_FRAME_BORDER'
                    )

            if detection_confidence is None:

                flags.append(
                    'DETECTION_CONFIDENCE_NOT_PROVIDED'
                )

            if segmentation_confidence is None:

                flags.append(
                    'SEGMENTATION_CONFIDENCE_NOT_PROVIDED'
                )

            if full_mask.any() and space == 'frame':

                exterior = full_mask.copy()

                exterior[y0:y1, x0:x1] = False

                if exterior.any():

                    flags.append(
                        'MASK_EXTENDS_OUTSIDE_SUPPLIED_BBOX'
                    )

            # ----------------------------------------
            # RELIABILITY DECISION
            # ----------------------------------------

            if mask_count == 0:

                reliability = 'UNSUPPORTED_EMPTY_MASK'

                action = 'CORRECT_INPUT'

            elif any(
                f in flags
                for f in (
                    'SMALL_MASK_HEURISTIC',
                    'LARGE_MASK_HEURISTIC',
                    'MASK_TOUCHES_FRAME_BORDER',
                    'MASK_EXTENDS_OUTSIDE_SUPPLIED_BBOX'
                )
            ):

                reliability = 'REVIEW_REQUIRED_INPUT_QA'

                action = 'REVIEW'

            else:

                reliability = 'REVIEW_ONLY_UNCALIBRATED'

                action = 'REVIEW'

            # These are input QA heuristics.
            # They are NOT calibrated risk rules.

            result['schema_version'] = SCHEMA_VERSION

            result['defect_id'] = defect_id

            result['source_bbox_xyxy'] = list(bbox)

            result['mask_space_received'] = space

            result['reliability_status'] = reliability

            result['reliability'] = {

                'policy_version': POLICY_VERSION,

                'action': action,

                'input_qa_flags': flags,

                'confidence_inputs_are_uncalibrated':
                    True,

                'detection_confidence_reported':
                    detection_confidence,

                'segmentation_confidence_reported':
                    segmentation_confidence,

                'independent_scene_validation':
                    False,

                'physical_scale_verified':
                    False,

                'metric_measurement_eligible':
                    False
            }

            # ----------------------------------------
            # SAFE SEVERITY ABSTENTION
            # ----------------------------------------

            result['severity'] = {

                'physical_level': None,

                'visual_level': None,

                'status':
                    'ABSTAIN_NO_VALIDATED_SEVERITY_THRESHOLDS',

                'risk_score': None
            }

            result['physical_measurements'] = {

                'depth_d95_mm': None,

                'depth_max_mm': None,

                'area_m2': None,

                'volume_m3': None,

                'crack_length_mm': None,

                'crack_width_mm': None
            }

            result['measurement_notes'] = (
                'Geometry RAW = unverified '
                'PothRGBD-style axial residual; '
                'pixel geometry is not physical area/length; '
                'no validated severity.'
            )

            outputs.append(result)

        return {

            'schema_version': SCHEMA_VERSION,

            'frame_height': h,

            'frame_width': w,

            'defect_count': len(outputs),

            'defects': outputs,

            'overall_status':
                'REVIEW_ONLY_NOT_FIELD_VALIDATED',

            'physical_calibration_verified':
                False,

            'teammate_model_handoff_verified':
                False
        }
