"""Pix2Pix cGAN Architecture for Component C2 (Stage 1: Pseudo-Thermal Synthesis).

Emulates pavement moisture thermodynamics by translating 256x256 RGB optical crops
into 256x256 single-channel pseudo-thermal moisture maps, eliminating expensive
physical infrared hardware ($20,000+).

References:
- Proposal Section 2.2 Sub-Objective 1
- Proposal Section 3.1 Two-Stage Pipeline
- Isola et al. (2017) "Image-to-Image Translation with Conditional Adversarial Networks"
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UNetDownBlock(nn.Module):
    """Downsampling convolutional block: Conv2d -> InstanceNorm -> LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not normalize,
            )
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUpBlock(nn.Module):
    """Upsampling transposed convolution block with UNet skip connections and dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class UNetGenerator256(nn.Module):
    """256x256 UNet Generator for Software Pseudo-Thermal Synthesis.
    
    Ingests:  (B, 3, 256, 256) RGB optical crop normalized in [-1, 1]
    Outputs:  (B, 1, 256, 256) Pseudo-Thermal moisture map in [-1, 1]
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        # Encoder (Downsampling: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1)
        self.down1 = UNetDownBlock(in_channels, 64, normalize=False)  # 128x128
        self.down2 = UNetDownBlock(64, 128)                           # 64x64
        self.down3 = UNetDownBlock(128, 256)                          # 32x32
        self.down4 = UNetDownBlock(256, 512)                          # 16x16
        self.down5 = UNetDownBlock(512, 512)                          # 8x8
        self.down6 = UNetDownBlock(512, 512)                          # 4x4
        self.down7 = UNetDownBlock(512, 512)                          # 2x2
        self.down8 = UNetDownBlock(512, 512, normalize=False)         # 1x1

        # Decoder (Upsampling with Skip Connections)
        self.up1 = UNetUpBlock(512, 512, dropout=0.5)                 # 2x2 (512+512)
        self.up2 = UNetUpBlock(1024, 512, dropout=0.5)                # 4x4 (512+512)
        self.up3 = UNetUpBlock(1024, 512, dropout=0.5)                # 8x8 (512+512)
        self.up4 = UNetUpBlock(1024, 512)                             # 16x16 (512+512)
        self.up5 = UNetUpBlock(1024, 256)                             # 32x32 (256+256)
        self.up6 = UNetUpBlock(512, 128)                              # 64x64 (128+128)
        self.up7 = UNetUpBlock(256, 64)                               # 128x128 (64+64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Scaled to [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder passes
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder passes with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN Discriminator conditioned on [RGB, Thermal].
    
    Ingests:  (B, 4, 256, 256) concatenation of [RGB (3), Thermal (1)]
    Outputs:  (B, 1, 30, 30) patch prediction matrix of authenticity
    """

    def __init__(self, in_channels: int = 4):
        super().__init__()

        def discriminator_block(in_f: int, out_f: int, normalize: bool = True):
            layers = [nn.Conv2d(in_f, out_f, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),  # 128x128
            *discriminator_block(64, 128),                           # 64x64
            *discriminator_block(128, 256),                          # 32x32
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1), # 31x31
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),    # 30x30 output patch
        )

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rgb, thermal], dim=1)
        return self.model(x)


def weights_init_normal(m: nn.Module) -> None:
    """Initializes weights using Gaussian N(0, 0.02) per Pix2Pix paper."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
