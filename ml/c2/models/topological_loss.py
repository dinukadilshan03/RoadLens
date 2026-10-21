"""Skeletal Topological Loss & Composite Loss Functions for Component C2.

Implements differentiable morphological pooling to skeletonize predicted continuous
probability masks down to 1-pixel centerlines during backpropagation. Evaluates
topological precision and sensitivity to penalize line breaks and gap errors.

References:
- Proposal Section 2.2 Sub-Objective 4 (Topological Line Continuity)
- Proposal Section 3.1 (b) & Equation:
    L_Total = a * L_BCE + b * L_Dice + c * L_Topology
- Shit et al. (2021) "clDice - A Novel Topology-Preserving Loss Function"
- Xiao et al. (2024) IRFusionFormer (Topological Loss Benchmark)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Differentiable morphological min-pooling (erosion) using 3x3 structuring element."""
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """Differentiable morphological max-pooling (dilation) using 3x3 structuring element."""
    p1 = F.max_pool2d(img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = F.max_pool2d(img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.max(p1, p2)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    """Differentiable morphological opening: erosion followed by dilation."""
    return soft_dilate(soft_erode(img))


def soft_skeletonize(x: torch.Tensor, iterations: int = 4) -> torch.Tensor:
    """Extracts continuous differentiable 1-pixel centerline skeletons via morphological peeling.
    
    Inputs:
        x: (B, 1, H, W) Continuous probability map in [0, 1]
    Outputs:
        skel: (B, 1, H, W) Soft skeleton centerline map
    """
    img = x
    skel = F.relu(img - soft_open(img))

    for _ in range(iterations):
        img = soft_erode(img)
        skel = skel + F.relu(img - soft_open(img))

    return torch.clamp(skel, 0.0, 1.0)


class SoftclDiceLoss(nn.Module):
    """Topological Centerline Dice Loss (L_Topology).
    
    Evaluates Topological Precision (T-Precision) and Topological Sensitivity (T-Sensitivity)
    against ground truth skeletons. Penalizes single-pixel disconnects and bridges gaps.
    """

    def __init__(self, iter_: int = 4, smooth: float = 1.0):
        super().__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, y_skel: torch.Tensor | None = None) -> torch.Tensor:
        """Args:
            y_pred: (B, 1, H, W) Predicted soft probability map in [0, 1]
            y_true: (B, 1, H, W) Ground truth binary mask in {0, 1}
            y_skel: Optional (B, 1, H, W) Precomputed ground truth skeleton.
                    If None, it is derived via differentiable soft skeletonization.
        """
        # Extract predicted soft skeleton
        s_pred = soft_skeletonize(y_pred, iterations=self.iter)

        if y_skel is None:
            s_true = soft_skeletonize(y_true, iterations=self.iter)
        else:
            s_true = y_skel

        # Topological Precision: overlap of predicted skeleton with true mask
        t_prec = (torch.sum(s_pred * y_true, dim=(1, 2, 3)) + self.smooth) / (
            torch.sum(s_pred, dim=(1, 2, 3)) + self.smooth
        )

        # Topological Sensitivity: overlap of true skeleton with predicted mask
        t_sens = (torch.sum(s_true * y_pred, dim=(1, 2, 3)) + self.smooth) / (
            torch.sum(s_true, dim=(1, 2, 3)) + self.smooth
        )

        # Centerline Dice coefficient
        cl_dice = (2.0 * t_prec * t_sens) / (t_prec + t_sens + 1e-7)
        loss = 1.0 - torch.mean(cl_dice)
        return loss


class SoftDiceLoss(nn.Module):
    """Standard 2D spatial Dice Loss for region overlap."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        intersection = torch.sum(y_pred * y_true, dim=(1, 2, 3))
        cardinality = torch.sum(y_pred + y_true, dim=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - torch.mean(dice)


class CompositeTopologicalSegmentationLoss(nn.Module):
    """Joint Composite Loss for Component C2.
    
    L_Total = a * L_BCE + b * L_Dice + c * L_Topology
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 0.5):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.bce = nn.BCELoss()
        self.dice = SoftDiceLoss()
        self.topology = SoftclDiceLoss()

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        y_skel: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Numerical stability clamp
        y_pred_safe = torch.clamp(y_pred, 1e-6, 1.0 - 1e-6)

        loss_bce = self.bce(y_pred_safe, y_true)
        loss_dice = self.dice(y_pred, y_true)
        loss_topo = self.topology(y_pred, y_true, y_skel)

        total_loss = (self.a * loss_bce) + (self.b * loss_dice) + (self.c * loss_topo)

        components = {
            "loss_total": total_loss.item(),
            "loss_bce": loss_bce.item(),
            "loss_dice": loss_dice.item(),
            "loss_topology": loss_topo.item(),
        }
        return total_loss, components
