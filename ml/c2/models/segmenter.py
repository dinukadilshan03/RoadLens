"""Dual-Stream MobileNetV3 + LTCA + Alpha Gate Segmentation Engine for Component C2.

Combines parallel visual RGB and software-emulated pseudo-thermal encoders,
Linear Transposed Cross-Attention (LTCA) for low-memory channel fusion,
and a scale-aware dynamic Alpha attention gate for micro-crack preservation.

References:
- Proposal Section 2.1 & 2.2 Sub-Objectives 2 & 3
- Proposal Section 3.1 & Figure 3.1, Figure 3.2, Figure 4.2
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from ml.c2.models.alpha_gate import DynamicAlphaGate
from ml.c2.models.cgan import UNetGenerator256
from ml.c2.models.ltca import LinearTransposedCrossAttention


class DecoderBlock(nn.Module):
    """Upsampling decoder block with skip connections."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            # Match spatial dimensions if minor rounding differences occur
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class DualStreamLTCASegmenter(nn.Module):
    """Component C2 Core Segmentation Engine.
    
    Inputs:
        rgb: (B, 3, 256, 256) RGB optical crop
        thermal: Optional (B, 1, 256, 256) Pseudo-Thermal moisture map.
                 If None, the internal frozen cGAN software sensor generates it.
    Outputs:
        prob_map: (B, 1, 256, 256) Continuous prediction probability map in [0, 1]
        alpha:    (B, 1, 1, 1) or (B, 1, H, W) Dynamic scale-gating weight
        pseudo_thm: (B, 1, 256, 256) Synthesized pseudo-thermal map
    """

    def __init__(
        self,
        cgan_weights_path: str | Path | None = None,
        freeze_cgan: bool = True,
        feature_dim: int = 96,
    ):
        super().__init__()

        # 1. Stage 1: Software Thermal Sensor (Frozen Pix2Pix cGAN)
        self.cgan_sensor = UNetGenerator256(in_channels=3, out_channels=1)
        if cgan_weights_path is not None:
            weights_path = Path(cgan_weights_path)
            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location="cpu")
                # Handle full checkpoint vs state_dict
                if "generator_state_dict" in state_dict:
                    state_dict = state_dict["generator_state_dict"]
                self.cgan_sensor.load_state_dict(state_dict)
                print(f"[C2 Segmenter] Loaded frozen software thermal sensor from: {weights_path}")

        if freeze_cgan:
            for p in self.cgan_sensor.parameters():
                p.requires_grad = False
            self.cgan_sensor.eval()

        # 2. Stage 2: Dual MobileNetV3 Backbones
        # Visual RGB backbone (pretrained on ImageNet)
        mobilenet_rgb = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Thermal backbone (accepts 1 channel replicated or adapted)
        mobilenet_thm = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

        # Stage 1: Shallow spatial features (1/4 scale, 24 channels)
        self.rgb_low = nn.Sequential(*mobilenet_rgb.features[:4])
        self.thm_low = nn.Sequential(*mobilenet_thm.features[:4])

        # Stage 2: Deep contextual features (1/16 scale, 96 channels)
        self.rgb_high = nn.Sequential(*mobilenet_rgb.features[4:12])
        self.thm_high = nn.Sequential(*mobilenet_thm.features[4:12])

        # Adapter for 1-channel pseudo-thermal input to 3-channel backbone
        self.thm_stem = nn.Conv2d(1, 3, kernel_size=1, bias=False)

        # 3. Stage 3: Linear Transposed Cross-Attention (LTCA)
        # Matches channels O(C^2) to slash VRAM to ~67 MB
        self.ltca = LinearTransposedCrossAttention(dim=feature_dim, num_heads=4)

        # Dimension adaptation for spatial stream
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(24, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

        # 4. Stage 4: Scale-Aware Dynamic Alpha Gate
        self.alpha_gate = DynamicAlphaGate(channels=feature_dim, spatial_gate=True)

        # 5. Multi-Scale Progressive Decoder
        self.dec_high = DecoderBlock(in_channels=feature_dim, skip_channels=0, out_channels=48)
        self.dec_mid = DecoderBlock(in_channels=48, skip_channels=feature_dim, out_channels=24)
        self.dec_final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(24, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),  # Outputs continuous probability in [0, 1]
        )

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Synthesize Pseudo-Thermal map via frozen cGAN if not supplied
        if thermal is None:
            with torch.no_grad():
                # cGAN expects input in [-1, 1]
                pseudo_thm = self.cgan_sensor(rgb)
        else:
            pseudo_thm = thermal

        # Normalize RGB for MobileNetV3 (ImageNet statistics)
        # rgb is expected in [-1, 1], convert to [0, 1]
        rgb_norm = (rgb + 1.0) * 0.5
        thm_3ch = self.thm_stem((pseudo_thm + 1.0) * 0.5)

        # 2. Extract multi-scale visual and thermal features
        # Low-level spatial features (1/4 scale, 64x64, 24 channels)
        f_rgb_low = self.rgb_low(rgb_norm)
        f_thm_low = self.thm_low(thm_3ch)

        # High-level context features (1/16 scale, 16x16, 96 channels)
        f_rgb_high = self.rgb_high(f_rgb_low)
        f_thm_high = self.thm_high(f_thm_low)

        # 3. Transposed Cross-Attention Fusion (LTCA)
        # Fuses high-level RGB features with pseudo-thermal moisture context
        f_fused_high = self.ltca(f_rgb_high, f_thm_high)

        # Project low-level spatial features to feature_dim
        f_spatial = self.spatial_proj(f_rgb_low)

        # Upsample fused contextual features to match spatial feature scale (64x64)
        f_context_upsampled = F.interpolate(
            f_fused_high, size=f_spatial.shape[-2:], mode="bilinear", align_corners=False
        )

        # 4. Dynamic Alpha Gating: Balances global noise rejection vs micro-crack edges
        f_gated_spatial, alpha = self.alpha_gate(f_spatial, f_context_upsampled)

        # 5. Progressive Decoding
        d1 = self.dec_high(f_fused_high)  # 16x16 -> 32x32
        d2 = self.dec_mid(d1, f_gated_spatial)  # 32x32 + 64x64 skip -> 64x64
        prob_map = self.dec_final(d2)  # 64x64 -> 128x128 -> 256x256

        return prob_map, alpha, pseudo_thm
