"""C3-27 experimental visual crack severity. Requires torch, torchvision, numpy, Pillow,
cv2, scikit-image, scipy, scikit-learn, joblib. NOT physical/engineering severity.
The model only knows LOW / HIGH; uncertain outputs are NEEDS_REVIEW. No Moderate label exists.
"""
import json, math
from pathlib import Path
import numpy as np, torch, joblib
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from scipy import ndimage
from skimage.morphology import skeletonize
from sklearn.preprocessing import StandardScaler  # joblib dependency

def polygon_mask_features(mask):
    from PIL import Image
    mm=np.asarray(Image.fromarray((np.asarray(mask)>0).astype('uint8')*255).resize((128,128),Image.Resampling.NEAREST))>0
    if mm.sum()<18 or mm.mean()>.75:return None
    sk=skeletonize(mm);ls=int(sk.sum())
    if ls<4:return None
    n=ndimage.convolve(sk.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')-sk.astype(np.uint8)
    e=int(np.logical_and(sk,n==1).sum());j=int(np.logical_and(sk,n>=3).sum())
    w=2*ndimage.distance_transform_edt(mm)[sk]
    c=int(ndimage.label(mm)[1]);b=int(mm[0].any() or mm[-1].any() or mm[:,0].any() or mm[:,-1].any())
    holes=int((ndimage.binary_fill_holes(mm)&~mm).sum())
    return np.array([mm.mean(),ls/128.,np.median(w)/128.,np.percentile(w,95)/128.,
        e/max(1,ls),j/max(1,ls),min(c,30)/30.,b,holes/(128*128)],dtype=np.float64)

class CrackVisualSeverityService:
    def __init__(self,bundle,device='cpu'):
        self.path=Path(bundle)
        self.config=json.loads((self.path/'model_config.json').read_text())
        self.device=device
        net=mobilenet_v3_small(weights=None)
        net.classifier[-1]=torch.nn.Linear(net.classifier[-1].in_features,2)
        checkpoint=torch.load(self.path/'crack_visual_low_high_model.pt',map_location='cpu',weights_only=False)
        net.load_state_dict(checkpoint['state_dict']);self.net=net.to(device).eval()
        self.transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
          transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        self.fusion=joblib.load(self.path/'crack_polygon_fusion.joblib') if (self.path/'crack_polygon_fusion.joblib').exists() else None

    @torch.no_grad()
    def predict(self,rgb,bbox_xyxy,crack_mask=None):
        rgb=np.asarray(rgb)
        if rgb.ndim!=3 or rgb.shape[-1]!=3 or rgb.dtype!=np.uint8:raise ValueError('RGB uint8 HxWx3 required')
        h,w=rgb.shape[:2]
        x0,y0,x1,y1=map(int,bbox_xyxy)
        if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('Invalid half-open bbox')
        cx=(x0+x1)/2;cy=(y0+y1)/2;side=max(96,(x1-x0)*1.6,(y1-y0)*1.6)
        l=int(math.floor(cx-side/2));t=int(math.floor(cy-side/2));r=int(math.ceil(cx+side/2));b=int(math.ceil(cy+side/2))
        crop=Image.fromarray(rgb).crop((l,t,r,b));x=self.transform(crop).unsqueeze(0).to(self.device)
        logits=self.net(x).cpu().numpy()[0];T=self.config['temperature_from_validation']
        z=logits/T;z=z-z.max();p=np.exp(z);p=p/p.sum()
        mode='RGB_ONLY'
        if self.fusion is not None and crack_mask is not None:
            mask=np.asarray(crack_mask)
            if mask.shape!=(h,w):raise ValueError('Crack mask must align exactly with RGB image')
            # IMPORTANT: a supplied real segmentation mask is required; not a bbox approximation.
            mc=Image.fromarray(((mask>0)*255).astype('uint8')).crop((l,t,r,b))
            feats=polygon_mask_features(mc)
            if feats is not None:
                xy=np.r_[p[1],feats][None,:]
                p=self.fusion.predict_proba(xy)[0];mode='RGB_PLUS_MASK_MORPHOLOGY'
        k=int(np.argmax(p));score=float(np.max(p))
        review=score<float(self.config['review_threshold_precommitted'])
        return {'visual_crack_severity': 'NEEDS_REVIEW' if review else ['LOW','HIGH'][k],
            'suggested_visual_class':['LOW','HIGH'][k],'review_required':bool(review),
            'model_score_uncalibrated_for_real_deployment':score,'analysis_mode':mode,
            'severity_kind':'EXPERIMENTAL_EXPERT_LABELED_VISUAL_SEVERITY',
            'physical_severity':None,'physical_dimensions':None,'units':'NONE',
            'model_version':'C3-27.0.0-experimental'}
