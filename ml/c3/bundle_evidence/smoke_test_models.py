"""Verify real model checkpoints cold-load and produce structurally valid predictions.
Run from bundle root. Synthetic pixels are NOT a model accuracy test.
"""
import sys, importlib.util, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent / 'backend'/'component3'

def load(filename, name):
    spec=importlib.util.spec_from_file_location(name,filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

rgb=np.full((224,320,3),110,dtype=np.uint8)
mask=np.zeros((224,320),dtype=np.uint8)
mask[48:172,90:237]=1
adapter=load(ROOT/'integration'/'C3-23_combined_service.py','c323_local')
combined=adapter.CombinedPotholeAssessment(ROOT/'geometry',ROOT/'pothole_visual_severity',device='cpu')
res=combined.predict(rgb,defect_id='smoke-pothole',pothole_mask=mask,include_map=False)
assert res['visual_severity']['severity_kind']=='EXPERIMENTAL_IMAGE_VISUAL_SEVERITY'
assert res['geometry'] is not None
print('PASS | trained CRDL + trained pothole visual severity loaded and inferred')
crmod=load(ROOT/'crack_visual_severity'/'crack_visual_severity_service.py','crack_visual_local')
crack=crmod.CrackVisualSeverityService(ROOT/'crack_visual_severity',device='cpu')
res=crack.predict(rgb,[90,48,237,172])
assert res['analysis_mode']=='RGB_ONLY'
assert res['visual_crack_severity'] in {'LOW','HIGH','NEEDS_REVIEW'}
print('PASS | trained crack visual severity loaded and inferred (RGB_ONLY)')
print('NOTE | synthetic input smoke tests DO NOT establish model correctness on real roads')
