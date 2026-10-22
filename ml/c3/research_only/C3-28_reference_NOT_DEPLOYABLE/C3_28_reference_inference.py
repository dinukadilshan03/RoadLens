"""C3-28 experimental relative healthy-road reference prediction, NOT metric geometry.

Use alongside, not instead of, the frozen primary CRDL pothole depression model.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from PIL import Image

class HealthyRoadShapeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(4,16,5,stride=2,padding=2),nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(32,48,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(48,64,3,stride=2,padding=1),nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,5))
    def forward(self,x):return self.head(self.features(x))

class RelativeHealthyRoadReference:
    def __init__(self, bundle, device='cpu'):
        root=Path(bundle)
        ck=torch.load(root/'healthy_road_relative_shape_model.pt',map_location='cpu',weights_only=True)
        self.model=HealthyRoadShapeNet().to(device).eval()
        self.model.load_state_dict(ck['state_dict'],strict=True)
        self.mu=np.asarray(ck['mu'],dtype='float32')
        self.sd=np.asarray(ck['sd'],dtype='float32')
        self.device=device
    def predict(self,rgb_uint8,mask):
        rgb=np.asarray(rgb_uint8);m=np.asarray(mask)
        if rgb.ndim!=3 or rgb.shape[-1]!=3 or rgb.dtype!=np.uint8:
            raise ValueError('Expected uint8 RGB image HxWx3')
        if m.shape!=rgb.shape[:2] or not np.asarray(m>0).any():
            raise ValueError('Aligned nonempty binary pothole mask required')
        arr=np.asarray(Image.fromarray(rgb).resize((128,96),Image.Resampling.BILINEAR),dtype='float32').transpose(2,0,1)/255
        msmall=np.asarray(Image.fromarray((m>0).astype('uint8')*255).resize((128,96),Image.Resampling.NEAREST))>0
        x=torch.from_numpy(np.concatenate([arr,msmall[None].astype('float32')],axis=0)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            coeff=self.model(x)[0].cpu().numpy()*self.sd+self.mu
        yy,xx=np.mgrid[:96,:128]
        xx=(xx-(128-1)/2)/((128-1)/2)
        yy=(yy-(96-1)/2)/((96-1)/2)
        surface=coeff[0]*xx+coeff[1]*yy+coeff[2]*xx*xx+coeff[3]*yy*yy+coeff[4]*xx*yy
        return {'relative_healthy_surface_raw':surface.astype('float32'),
                'native_hw':[96,128],
                'units':'UNVERIFIED_RAW_RELATIVE_CAMERA_AXIS',
                'is_metric_3d':False,
                'model_kind':'EXPERIMENTAL_LEARNED_REFERENCE_SHAPE',
                'horizontal_coordinates':'image_normalized',
                'vertical_reference':'arbitrary image-centered zero'}
