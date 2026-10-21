"""Scale-Aware Dynamic Alpha Gating Module for Component C2.

Dynamically computes an attention gate alpha in [0.0, 1.0] to balance deep contextual
noise rejection (alpha -> 1.0: zoom out, rejects broad canopy shadows and glare)
against shallow spatial micro-crack detail preservation (alpha -> 0.0: zoom in,
preserves fine hairline cracks <3 mm).

References:
- Proposal Section 2.2 Sub-Objective 3 (Scale-Aware Feature Gating)
- Proposal Section 3.1 (a) & Figure 3.2 (Dynamic Alpha Attention Gate)
- Sun et al. (2022) DMA-Net (Multi-scale attention mechanism)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicAlphaGate(nn.Module):
    """Dynamic Alpha Attention Gate.
    
    Inputs:
        f_spatial: (B, C, H, W) Low-level visual features with fine boundary details
        f_context: (B, C, H, W) High-level fused contextual features (shadow/grate filtered)
    Outputs:
        f_gated:   (B, C, H, W) Scale-balanced feature representation
        alpha:     (B, 1, 1, 1) or (B, 1, H, W) Learned gating weight in [0, 1]
    """

    def __init__(self, channels: int, spatial_gate: bool = True):
        super().__init__()
        self.spatial_gate = spatial_gate

        # Global channel context descriptor
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels * 2, channels // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, 1, bias=True),
            nn.Sigmoid(),
        )

        if spatial_gate:
            # Local spatial attention map for pixel-level micro-edge preservation
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(channels * 2, channels // 4, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, 1, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )

        # Refinement convolution after fusion
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f_spatial: torch.Tensor, f_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = f_spatial.shape
        combined = torch.cat([f_spatial, f_context], dim=1)

        # 1. Global context alpha: determines global macro vs micro trade-off
        pooled = self.global_pool(combined).view(b, c * 2)
        global_alpha = self.channel_mlp(pooled).view(b, 1, 1, 1)

        if self.spatial_gate:
            # 2. Local spatial alpha: provides localized boundary weighting
            local_alpha = self.spatial_conv(combined)
            # Blended alpha: 0.5 * global + 0.5 * local
            alpha = 0.5 * global_alpha + 0.5 * local_alpha
        else:
            alpha = global_alpha

        # Gated combination:
        # alpha -> 1.0: context (shadow/glare rejection)
        # alpha -> 0.0: spatial (hairline crack preservation)
        f_gated = (alpha * f_context) + ((1.0 - alpha) * f_spatial)
        f_out = self.refine(f_gated)

        return f_out, alpha
