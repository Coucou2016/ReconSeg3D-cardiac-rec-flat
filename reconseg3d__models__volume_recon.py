"""Compact 3D reconstruction from a sparse volume (paper-aligned interface).

Original ReconSeg3D uses a 3D ViT encoder-decoder on ~256×256×128 grids.
This module implements the same I/O contract with a compact CNN UNet plus an
optional transformer bottleneck so tests can run on 16×32×32 (and similar)
without a full-scale ViT.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from reconseg3d.models.blocks import ConvBlock3D


class CompactVolumeRecon(nn.Module):
    """Sparse-to-dense 3D reconstruction (MSE training target).

    Accepts (B, C, D, H, W) or (B, C, T, D, H, W); 6D is decoded per-frame.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        embed_dim: int | None = None,
        num_heads: int = 4,
        num_layers: int = 2,
        use_transformer: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c = base_channels
        self.in_channels = in_channels
        self.use_transformer = use_transformer
        self.stem = ConvBlock3D(in_channels, c)
        self.down1 = nn.Sequential(ConvBlock3D(c, c * 2, stride=2), ConvBlock3D(c * 2, c * 2))
        self.down2 = nn.Sequential(ConvBlock3D(c * 2, c * 4, stride=2), ConvBlock3D(c * 4, c * 4))
        bottleneck_ch = c * 4
        embed_dim = embed_dim or bottleneck_ch
        self.proj_in = nn.Identity() if embed_dim == bottleneck_ch else nn.Conv3d(bottleneck_ch, embed_dim, 1)
        self.proj_out = nn.Identity() if embed_dim == bottleneck_ch else nn.Conv3d(embed_dim, bottleneck_ch, 1)
        if use_transformer:
            nhead = min(num_heads, embed_dim)
            while nhead > 1 and embed_dim % nhead != 0:
                nhead -= 1
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            self.transformer = None
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(bottleneck_ch, c * 2, 2, stride=2),
            nn.BatchNorm3d(c * 2),
            nn.ReLU(inplace=True),
        )
        self.dec1 = ConvBlock3D(c * 4, c * 2)
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(c * 2, c, 2, stride=2),
            nn.BatchNorm3d(c),
            nn.ReLU(inplace=True),
        )
        self.dec2 = ConvBlock3D(c * 2, c)
        self.out_conv = nn.Conv3d(c, in_channels, 1)

    def _forward_3d(self, x: torch.Tensor) -> torch.Tensor:
        skip0 = self.stem(x)
        skip1 = self.down1(skip0)
        z = self.down2(skip1)
        z = self.proj_in(z)
        if self.transformer is not None:
            b, ch, d, h, w = z.shape
            tokens = z.flatten(2).transpose(1, 2)
            tokens = self.transformer(tokens)
            z = tokens.transpose(1, 2).reshape(b, ch, d, h, w)
        z = self.proj_out(z)
        u1 = self.up1(z)
        if u1.shape[-3:] != skip1.shape[-3:]:
            u1 = F.interpolate(u1, size=skip1.shape[-3:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, skip1], dim=1))
        u2 = self.up2(d1)
        if u2.shape[-3:] != skip0.shape[-3:]:
            u2 = F.interpolate(u2, size=skip0.shape[-3:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, skip0], dim=1))
        out = self.out_conv(d2)
        if out.shape[-3:] != x.shape[-3:]:
            out = F.interpolate(out, size=x.shape[-3:], mode="trilinear", align_corners=False)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            return self._forward_3d(x)
        if x.ndim == 6:
            b, c, t, d, h, w = x.shape
            flat = x.permute(0, 2, 1, 3, 4, 5).reshape(b * t, c, d, h, w)
            rec = self._forward_3d(flat)
            return rec.view(b, t, c, d, h, w).permute(0, 2, 1, 3, 4, 5).contiguous()
        raise ValueError(f"Expected 5D or 6D volume, got shape {tuple(x.shape)}")


def build_volume_recon(cfg: dict[str, Any]) -> CompactVolumeRecon:
    model_cfg = cfg.get("model", cfg)
    return CompactVolumeRecon(
        in_channels=model_cfg.get("in_channels", 1),
        base_channels=model_cfg.get("base_channels", 16),
        embed_dim=model_cfg.get("recon_embed_dim"),
        num_heads=model_cfg.get("recon_num_heads", 4),
        num_layers=model_cfg.get("recon_num_layers", 2),
        use_transformer=model_cfg.get("recon_use_transformer", True),
        dropout=model_cfg.get("dropout", 0.0),
    )
