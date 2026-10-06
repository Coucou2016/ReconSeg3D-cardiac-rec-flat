"""HeartTTable-lite: spatial patches + temporal tokens + table tokens + CLS cross-attn."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeartTTableOutput:
    features: torch.Tensor
    risk_logits: torch.Tensor


class _CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query)
        k = self.norm_kv(kv)
        attn_out, _ = self.attn(q, k, k, need_weights=False)
        query = query + attn_out
        query = query + self.ff(query)
        return query


class HeartTTable(nn.Module):
    """
    Compact HeartTTable for 4D cine + tabular fusion.

    Designed to run on tiny sequences such as (B, 1, T=8, 16, 16, 16) in tests.
    Paper-scale would use larger embed/patch sizes on reconstructed 256³ grids.
    """

    def __init__(
        self,
        in_channels: int = 1,
        clinical_dim: int = 4,
        embed_dim: int = 32,
        patch_size: int | tuple[int, int, int] = 4,
        num_heads: int = 4,
        num_layers: int = 2,
        num_table_tokens: int = 4,
        max_frames: int = 32,
        max_patches: int = 512,
        dropout: float = 0.1,
        num_outputs: int = 1,
    ) -> None:
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.clinical_dim = clinical_dim
        self.num_table_tokens = num_table_tokens
        self.max_frames = max_frames
        heads = max(1, min(num_heads, embed_dim))
        while embed_dim % heads != 0 and heads > 1:
            heads -= 1
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.spatial_pos = nn.Parameter(torch.zeros(1, max_patches, embed_dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, max_frames, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if clinical_dim > 0:
            self.table_proj = nn.Sequential(
                nn.Linear(clinical_dim, embed_dim * num_table_tokens),
                nn.GELU(),
            )
        else:
            self.table_proj = None
        self.blocks = nn.ModuleList([_CrossAttnBlock(embed_dim, heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        hidden = max(embed_dim // 2, 8)
        self.risk_head = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_outputs),
        )
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _pad_to_patch(self, frames: torch.Tensor) -> torch.Tensor:
        pd, ph, pw = self.patch_size
        _bt, _c, d, h, w = frames.shape
        nd = max(pd, ((d + pd - 1) // pd) * pd)
        nh = max(ph, ((h + ph - 1) // ph) * ph)
        nw = max(pw, ((w + pw - 1) // pw) * pw)
        if (nd, nh, nw) == (d, h, w):
            return frames
        return F.interpolate(frames, size=(nd, nh, nw), mode="trilinear", align_corners=False)

    def forward(
        self,
        volume: torch.Tensor,
        clinical: torch.Tensor | None = None,
    ) -> HeartTTableOutput:
        """
        Args:
            volume: (B, C, T, D, H, W)
            clinical: optional (B, clinical_dim)
        """
        if volume.ndim != 6:
            raise ValueError(f"Expected (B,C,T,D,H,W), got {tuple(volume.shape)}")
        b, c, t, d, h, w = volume.shape
        frames = volume.permute(0, 2, 1, 3, 4, 5).reshape(b * t, c, d, h, w)
        frames = self._pad_to_patch(frames)
        tokens = self.patch_embed(frames)
        _bt, e, ds, hs, ws = tokens.shape
        p = ds * hs * ws
        spatial = tokens.flatten(2).transpose(1, 2)
        pos = self.spatial_pos
        if p != pos.shape[1]:
            pos = F.interpolate(pos.transpose(1, 2), size=p, mode="linear", align_corners=False).transpose(1, 2)
        spatial = spatial + pos[:, :p]
        spatial = spatial.view(b, t, p, e)
        t_use = min(t, self.max_frames)
        tpos = self.temporal_pos[:, :t_use]
        if t > self.max_frames:
            tpos = F.interpolate(
                self.temporal_pos.transpose(1, 2),
                size=t,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            t_use = t
        spatial = spatial[:, :t_use] + tpos.unsqueeze(2)
        spatial_tokens = spatial.reshape(b, t_use * p, e)
        temporal_tokens = spatial.mean(dim=2) + tpos

        parts = [spatial_tokens, temporal_tokens]
        if self.table_proj is not None:
            if clinical is None:
                clinical = torch.zeros(b, self.clinical_dim, device=volume.device, dtype=volume.dtype)
            table = self.table_proj(clinical.float()).view(b, self.num_table_tokens, e)
            parts.append(table)
        kv = torch.cat(parts, dim=1)
        cls = self.cls_token.expand(b, -1, -1)
        for block in self.blocks:
            cls = block(cls, kv)
        features = self.norm(cls.squeeze(1))
        risk = self.risk_head(features)
        if risk.shape[-1] == 1:
            risk = risk.squeeze(-1)
        return HeartTTableOutput(features=features, risk_logits=risk)
