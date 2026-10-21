
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt


class ConvBlock(nn.Module):

    def __init__(self, cin, cout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class RoadRelativeCRDL(nn.Module):

    def __init__(self, base=16, mode="DIRECT_RGB_MASK"):

        super().__init__()

        assert mode in (
            "DIRECT_RGB_MASK",
            "PRIOR_RGB_MASK",
            "PRIOR_MASK_ONLY"
        )

        self.mode = mode

        self.e1 = ConvBlock(4, base)
        self.e2 = ConvBlock(base, 2 * base)

        self.mid = ConvBlock(2 * base, 4 * base)

        self.d2 = ConvBlock(6 * base, 2 * base)
        self.d1 = ConvBlock(3 * base, base)

        self.head = nn.Conv2d(base, 1, 1)

        nn.init.zeros_(self.head.weight)

        nn.init.constant_(
            self.head.bias,
            -1.5 if mode == "DIRECT_RGB_MASK" else 0.0
        )

    def forward(self, x, prior_raw, scale_raw):

        mask = x[:, 3:4]

        if self.mode == "PRIOR_MASK_ONLY":

            x = torch.cat(
                (torch.zeros_like(x[:, :3]), mask),
                dim=1
            )

        a = self.e1(x)

        b = self.e2(F.avg_pool2d(a, 2))

        mid = self.mid(F.avg_pool2d(b, 2))

        c = self.d2(
            torch.cat(
                (
                    b,
                    F.interpolate(
                        mid,
                        size=b.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                ),
                dim=1
            )
        )

        decoded = self.d1(
            torch.cat(
                (
                    a,
                    F.interpolate(
                        c,
                        size=a.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                ),
                dim=1
            )
        )

        logits = self.head(decoded)

        if self.mode != "DIRECT_RGB_MASK":

            # At zero learned correction:
            # softplus(log(prior / scale))
            # = log1p(prior / scale)

            logits = logits + torch.log(
                (prior_raw / float(scale_raw)).clamp(
                    min=1e-6
                )
            )

        return F.softplus(logits) * mask


def mask_prior(mask, coef):

    mask = np.asarray(mask, dtype=bool)

    distance = distance_transform_edt(
        mask
    ).astype(np.float32)

    distance /= max(float(distance.max()), 1.0)

    prior = (
        float(coef[0])
        + float(coef[1]) * distance
    )

    return (prior * mask).astype(np.float32)


def prepare_inputs(rgb, mask, coef, size_hw=(192, 256)):

    rgb = np.asarray(rgb)
    mask = np.asarray(mask)

    if (
        rgb.ndim != 3
        or rgb.shape[2] != 3
        or mask.shape != rgb.shape[:2]
    ):
        raise ValueError(
            "Expected aligned RGB(H,W,3) and mask(H,W)"
        )

    h, w = size_hw

    image = np.asarray(
        Image.fromarray(
            rgb.astype(np.uint8)
        ).resize(
            (w, h),
            Image.Resampling.BILINEAR
        ),
        dtype=np.float32
    ) / 255.0

    binary = np.asarray(
        Image.fromarray(
            (mask > 0).astype(np.uint8) * 255
        ).resize(
            (w, h),
            Image.Resampling.NEAREST
        )
    ) > 0

    x = np.concatenate(
        (
            image.transpose(2, 0, 1),
            binary[None].astype(np.float32)
        ),
        axis=0
    )

    prior = mask_prior(binary, coef)

    return (
        torch.from_numpy(x).float(),
        torch.from_numpy(prior[None]).float()
    )


def infer_raw(model, rgb, mask, scale_raw, coef, device="cpu"):

    x, prior = prepare_inputs(
        rgb,
        mask,
        coef
    )

    model.eval()

    with torch.inference_mode():

        z = model(
            x[None].to(device),
            prior[None].to(device),
            scale_raw
        ).clamp(max=8)

        output = float(scale_raw) * torch.expm1(z)

    return output[0, 0].cpu().numpy().astype(
        np.float32
    )
