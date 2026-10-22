
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision import models, transforms

class VisualSeverityClassifier:
    # Experimental RGB-based pothole visual severity; NOT validated FHWA physical severity.
    def __init__(self, bundle_dir, device='cpu'):
        root=Path(bundle_dir)
        self.config=json.loads((root/'visual_severity_config.json').read_text())
        self.device=torch.device(device)
        net=models.mobilenet_v3_small(weights=None)
        net.classifier[-1]=nn.Linear(net.classifier[-1].in_features,3)
        ckpt=torch.load(root/'visual_severity_model.pt',map_location='cpu',weights_only=False)
        net.load_state_dict(ckpt['model_state_dict'],strict=True)
        self.net=net.to(self.device).eval()
        size=self.config['image_size']
        self.tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),
            transforms.Normalize(self.config['normalize_mean'],self.config['normalize_std'])])
    def predict(self, rgb_uint8, bbox_xyxy):
        arr=np.asarray(rgb_uint8)
        if arr.ndim!=3 or arr.shape[-1]!=3 or arr.dtype!=np.uint8:
            raise ValueError('Expected uint8 RGB image HxWx3')
        H,W=arr.shape[:2]
        x0,y0,x1,y1=[int(x) for x in bbox_xyxy]
        if not (0<=x0<x1<=W and 0<=y0<y1<=H):
            raise ValueError('Invalid half-open bbox')
        p=self.config['padding_fraction']; px=(x1-x0)*p;py=(y1-y0)*p
        b=(max(0,int(x0-px)),max(0,int(y0-py)),min(W,int(x1+px)),min(H,int(y1+py)))
        crop=Image.fromarray(arr).crop(b)
        x=self.tf(crop).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probs=self.net(x).softmax(-1)[0].cpu().numpy()
        idx=int(np.argmax(probs))
        thresh=self.config['review_threshold_provisional']
        review=bool(float(probs[idx])<thresh)
        return {'visual_severity':('NEEDS_REVIEW' if review else self.config['class_names'][idx]),
                'suggested_class':self.config['class_names'][idx],
                'review_required':review,
                'model_score_uncalibrated':float(probs[idx]),
                'severity_kind':'EXPERIMENTAL_IMAGE_VISUAL_SEVERITY',
                'physical_severity':None,
                'physical_dimensions':None,
                'units':'NONE',
                'model_version':self.config['model_version']}
