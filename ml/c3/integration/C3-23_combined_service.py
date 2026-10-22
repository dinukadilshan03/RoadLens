"""C3-23 experimental inference adapter: genuine C3-20 geometry + C3-22 visual-severity.

This file DOES NOT convert RAW geometry to mm and DOES NOT fuse model scores.
Keep C3-20 and C3-22 files side-by-side in their respective bundle directories.
"""
from __future__ import annotations
from pathlib import Path
import importlib.util
import numpy as np


def _load(path: Path, module_name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Required model module missing: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_rgb(image):
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError('RGB image must be uint8 HxWx3')
    return arr


def _validate_bbox(box, width, height):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError('Please supply four integer coordinates [x0,y0,x1,y1]')
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in box):
        raise ValueError('Bounding box coordinates must be integers')
    x0, y0, x1, y1 = [int(x) for x in box]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('Bounding box is invalid or outside original RGB image')
    return [x0, y0, x1, y1]


def _validate_mask(mask, shape):
    arr = np.asarray(mask)
    if arr.ndim != 2 or tuple(arr.shape) != tuple(shape):
        raise ValueError('Pothole mask must match original image height/width; no silent resizing')
    if not (arr.dtype == bool or np.issubdtype(arr.dtype, np.number)):
        raise ValueError('Pothole mask must be binary numeric image')
    if not np.isfinite(arr).all() or not np.isin(arr, [0, 1, 255]).all():
        raise ValueError('Pothole mask must contain only 0,1 or 255; not a road photograph')
    mask_bool = arr != 0
    if mask_bool.sum() == 0:
        raise ValueError('Pothole mask is empty')
    return mask_bool


class CombinedPotholeAssessment:
    def __init__(self, geometry_dir, visual_dir, device='cpu'):
        self.geometry_dir = Path(geometry_dir)
        self.visual_dir = Path(visual_dir)
        geo_module = _load(self.geometry_dir / 'component3_severity_bridge.py', 'c323_geometry_bridge')
        vis_module = _load(self.visual_dir / 'visual_severity_inference.py', 'c323_visual_severity')
        self.geometry = geo_module.Component3SeverityBridge(self.geometry_dir, device=device)
        self.visual = vis_module.VisualSeverityClassifier(self.visual_dir, device=device)

    def predict(self, rgb_uint8, *, defect_id, pothole_mask=None, bbox_xyxy=None, field_record=None, include_map=True):
        """Return separate branches and provenance, never a fictitious fused physical severity.

        If mask is absent but manually verified bbox exists, visual-severity + field policy run;
        CRDL geometry is properly marked unavailable. Field source must be independently measured.
        """
        rgb = _valid_rgb(rgb_uint8)
        H, W = rgb.shape[:2]
        if not isinstance(defect_id, str) or not defect_id.strip():
            raise ValueError('defect_id must be a nonempty identifier')
        mask = None
        qa = []
        if pothole_mask is not None:
            mask = _validate_mask(pothole_mask, (H, W))
            yy, xx = np.where(mask)
            derived = [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)]
            box = derived if bbox_xyxy is None else _validate_bbox(bbox_xyxy, W, H)
            x0, y0, x1, y1 = box
            outside = mask.copy()
            outside[y0:y1, x0:x1] = False
            if outside.any():
                raise ValueError('Pothole mask extends beyond supplied bbox')
            coverage = float(mask.mean())
            if coverage > 0.60: qa.append('MASK_COVERS_MOST_OF_IMAGE_REVIEW_INPUT')
            if coverage < 0.001: qa.append('VERY_SMALL_MASK_REVIEW_INPUT')
            if any((mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any())):
                qa.append('MASK_TOUCHES_IMAGE_BORDER_REVIEW_INPUT')
        elif bbox_xyxy is not None:
            box = _validate_bbox(bbox_xyxy, W, H)
            qa.append('NO_POTHOLE_MASK_CRDL_GEOMETRY_UNAVAILABLE')
        else:
            raise ValueError('Provide an aligned pothole_mask or an explicitly reviewed pothole bbox')

        visual = self.visual.predict(rgb, box)
        if field_record is not None and not isinstance(field_record, dict):
            raise ValueError('field_record must be a dictionary or None')

        geo_out = None
        if mask is not None:
            detections = [{'defect_id': defect_id, 'bbox_xyxy':box,
                           'mask_space':'frame', 'mask':mask.astype(np.uint8)}]
            optional_fields = {defect_id: field_record} if field_record is not None else None
            result = self.geometry.predict(rgb, detections, include_maps=include_map,
                                           field_measurements=optional_fields)
            geo_out = result['defects'][0]
            field = geo_out['field_reference_severity']
        else:
            field = self.geometry.policy.evaluate_field_reference(field_record)

        return {
            'defect_id':defect_id, 'bbox_xyxy':box, 'image_shape':[H,W],
            'has_geometry_map':bool(geo_out is not None), 'visual_severity':visual,
            'geometry':geo_out, 'field_reference_severity':field,
            'input_qa_notes':qa,
            'interpretation':{
                'visual_severity':'Experimental assessment from the photograph, NOT depth-certified severity',
                'crdl_geometry':'Road-relative RAW shape, NOT verified mm (available only when aligned mask exists)',
                'field_reference':'Based on separately supplied physical deepest-point measurement, not AI',
                'fused_physical_prediction':None,
                'recommended_action':'Human inspection required before physical maintenance decisions',
            },
        }
