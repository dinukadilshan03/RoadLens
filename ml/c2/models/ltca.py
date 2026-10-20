"""Linear Transposed Cross-Attention (LTCA) Module for Component C2.

Performs cross-covariance channel-wise attention between RGB visual features and
software-emulated pseudo-thermal features. Bypasses quadratic spatial attention O(N^2)
to achieve linear O(N) spatial complexity with O(C^2) channel matching, reducing
VRAM footprint from 17 GB to ~67 MB.

References:
- Proposal Section 2.2 Sub-Objective 2 (Low-Memory Feature Fusion)
- Proposal Section 3.1 Stage 2 Pipeline (Transposed Cross-Attention Block)
- Proposal Table 3.1 & Table 11.1 (LTCA 67 MB VRAM Target)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFeedForwardNetwork(nn.Module):
    """Gated Depthwise Feed-Forward Network (GDFN) for feature refinement."""

    def __init__(self, dim: int, ffn_expansion_factor: float = 2.0, bias: bool = False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class LinearTransposedCrossAttention(nn.Module):
    """Linear Transposed Cross-Attention (LTCA) Block.
    
    Computes cross-attention across feature channels (C x C) rather than spatial
    dimensions (HW x HW).
    
    Inputs:
        x_rgb: (B, C, H, W) Visual stream features (queries)
        x_thm: (B, C, H, W) Emulated thermal stream features (keys & values)
    Outputs:
        fused: (B, C, H, W) Fused noise-resilient representation
    """

    def __init__(self, dim: int, num_heads: int = 4, bias: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Query projection from RGB visual features
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        # Key and Value projections from Pseudo-Thermal features
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        # Output projection
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # Normalization layers
        self.norm_rgb = nn.GroupNorm(1, dim)
        self.norm_thm = nn.GroupNorm(1, dim)
        self.norm_ffn = nn.GroupNorm(1, dim)

        # Gated feed-forward network
        self.gdfn = GatedFeedForwardNetwork(dim=dim)

    def forward(self, x_rgb: torch.Tensor, x_thm: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x_rgb.shape

        # Residual branch
        residual = x_rgb

        # Normalize inputs
        norm_rgb = self.norm_rgb(x_rgb)
        norm_thm = self.norm_thm(x_thm)

        # Project Q from RGB, K and V from Thermal
        q = self.q_dwconv(self.q_proj(norm_rgb))
        k = self.k_dwconv(self.k_proj(norm_thm))
        v = self.v_dwconv(self.v_proj(norm_thm))

        # Reshape to (B, num_heads, head_dim, HW) for channel-wise transposed attention
        q = q.reshape(b, self.num_heads, self.head_dim, h * w)
        k = k.reshape(b, self.num_heads, self.head_dim, h * w)
        v = v.reshape(b, self.num_heads, self.head_dim, h * w)

        # L2-normalize queries and keys along spatial dimension
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Transposed attention matrix: (B, heads, head_dim, head_dim)
        # O(C^2) complexity independent of spatial resolution HW
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)

        # Channel-weighted value aggregation: (B, heads, head_dim, HW)
        out = torch.matmul(attn, v)
        out = out.reshape(b, c, h, w)
        out = self.out_proj(out)

        # First residual connection
        x = residual + out

        # Feed-forward network with second residual
        x = x + self.gdfn(self.norm_ffn(x))
        return x
