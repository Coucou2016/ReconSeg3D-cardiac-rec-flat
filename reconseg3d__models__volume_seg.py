"""3D UNet-style segmentation (paper-aligned compact variant).

Paper uses 3D nnU-Net on ~256×256×128. This module is a 2-level 3D UNet with
the same class interface: background / LV / RV / LVM (4-class) or +scar (5).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from reconseg3d.models.blocks import ConvBlock3D


class VolumeUNet3D(nn.Module):
    """3D UNet segmentation.

    Input: (B, C, D, H, W) or (B, C, T, D, H, W) (per-frame).
    Output logits: (B, K, D, H, W) or (B, K, T, D, H, W).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        c = base_channels
        self.enc1 = nn.Sequential(ConvBlock3D(in_channels, c), ConvBlock3D(c, c))
        self.down1 = ConvBlock3D(c, c * 2, stride=2)
        self.enc2 = nn.Sequential(ConvBlock3D(c * 2, c * 2), ConvBlock3D(c * 2, c * 2))
        self.down2 = ConvBlock3D(c * 2, c * 4, stride=2)
        self.bot = nn.Sequential(ConvBlock3D(c * 4, c * 4), ConvBlock3D(c * 4, c * 4))
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, 2, stride=2)
        self.dec2 = nn.Sequential(ConvBlock3D(c * 4, c * 2), ConvBlock3D(c * 2, c * 2))
        self.up1 = nn.ConvTranspose3d(c * 2, c, 2, stride=2)
        self.dec1 = nn.Sequential(ConvBlock3D(c * 2, c), ConvBlock3D(c, c))
        self.head = nn.Conv3d(c, num_classes, 1)

    def _forward_3d(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b = self.bot(self.down2(e2))
        u2 = self.up2(b)
        if u2.shape[-3:] != e2.shape[-3:]:
            u2 = F.interpolate(u2, size=e2.shape[-3:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        if u1.shape[-3:] != e1.shape[-3:]:
            u1 = F.interpolate(u1, size=e1.shape[-3:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        logits = self.head(d1)
        if logits.shape[-3:] != x.shape[-3:]:
            logits = F.interpolate(logits, size=x.shape[-3:], mode="trilinear", align_corners=False)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            return self._forward_3d(x)
        if x.ndim == 6:
            b, c, t, d, h, w = x.shape
            flat = x.permute(0, 2, 1, 3, 4, 5).reshape(b * t, c, d, h, w)
            logits = self._forward_3d(flat)
            k = logits.shape[1]
            return logits.view(b, t, k, d, h, w).permute(0, 2, 1, 3, 4, 5).contiguous()
        raise ValueError(f"Expected 5D or 6D volume, got shape {tuple(x.shape)}")


def build_volume_seg(cfg: dict[str, Any]) -> VolumeUNet3D:
    model_cfg = cfg.get("model", cfg)
    return VolumeUNet3D(
        in_channels=model_cfg.get("in_channels", 1),
        num_classes=model_cfg.get("num_seg_classes", 4),
        base_channels=model_cfg.get("base_channels", 16),
    )
