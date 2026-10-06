SOURCE PACK CHUNK 2 — continue reading next chunks before proposing patches.

## FILE: reconseg3d/models/motion.py

```
"""Differentiable 3D motion (displacement) and volume warping."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from reconseg3d.models.blocks import ConvBlock3D


def identity_grid(d: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Normalized grid (D, H, W, 3) in grid_sample (x, y, z) order."""
    zz = torch.linspace(-1.0, 1.0, d, device=device, dtype=dtype)
    yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    grid_z, grid_y, grid_x = torch.meshgrid(zz, yy, xx, indexing="ij")
    return torch.stack((grid_x, grid_y, grid_z), dim=-1)


def flow_to_grid(flow: torch.Tensor) -> torch.Tensor:
    """
    Convert voxel displacement (B, 3, D, H, W) with channels (dz, dy, dx)
    into a sampling grid (B, D, H, W, 3) for ``grid_sample``.
    """
    b, _, d, h, w = flow.shape
    base = identity_grid(d, h, w, flow.device, flow.dtype).unsqueeze(0).expand(b, -1, -1, -1, -1)
    dz, dy, dx = flow[:, 0], flow[:, 1], flow[:, 2]
    scale_z = 2.0 / max(d - 1, 1)
    scale_y = 2.0 / max(h - 1, 1)
    scale_x = 2.0 / max(w - 1, 1)
    disp = torch.stack((dx * scale_x, dy * scale_y, dz * scale_z), dim=-1)
    return base + disp


def warp_volume(volume: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp (B, C, D, H, W) with displacement (B, 3, D, H, W)."""
    grid = flow_to_grid(flow)
    return F.grid_sample(volume, grid, mode="bilinear", padding_mode="border", align_corners=True)


class MotionNet(nn.Module):
    """Predict 3D displacement between consecutive reconstructed frames."""

    def __init__(self, in_channels: int = 1, base_channels: int = 8) -> None:
        super().__init__()
        c = max(base_channels, 4)
        self.encoder = nn.Sequential(
            ConvBlock3D(in_channels * 2, c),
            ConvBlock3D(c, c),
        )
        self.flow_head = nn.Conv3d(c, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    def forward_pair(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """src/tgt: (B, C, D, H, W) -> flow (B, 3, D, H, W) taking src toward tgt."""
        return self.flow_head(self.encoder(torch.cat([src, tgt], dim=1)))

    def forward(self, recon: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Args:
            recon: (B, C, T, D, H, W)
        Returns:
            flow_fwd, flow_bwd each (B, 3, T-1, D, H, W), or (None, None) if T < 2.
        """
        if recon.ndim != 6:
            raise ValueError(f"Expected (B,C,T,D,H,W), got {tuple(recon.shape)}")
        b, c, t, d, h, w = recon.shape
        if t < 2:
            return None, None
        src = recon[:, :, :-1].permute(0, 2, 1, 3, 4, 5).reshape(b * (t - 1), c, d, h, w)
        tgt = recon[:, :, 1:].permute(0, 2, 1, 3, 4, 5).reshape(b * (t - 1), c, d, h, w)
        fwd = self.forward_pair(src, tgt).view(b, t - 1, 3, d, h, w).permute(0, 2, 1, 3, 4, 5)
        bwd = self.forward_pair(tgt, src).view(b, t - 1, 3, d, h, w).permute(0, 2, 1, 3, 4, 5)
        return fwd.contiguous(), bwd.contiguous()


def warp_consistency_loss(recon: torch.Tensor, flow_fwd: torch.Tensor, flow_bwd: torch.Tensor | None = None) -> torch.Tensor:
    """||Vhat_{t+1} - warp(Vhat_t, u_t)||_1 (+ backward if provided)."""
    _b, _c, t, _d, _h, _w = recon.shape
    if t < 2:
        return recon.sum() * 0.0
    src = recon[:, :, :-1]
    tgt = recon[:, :, 1:]
    bt = src.shape[0] * src.shape[2]
    c = src.shape[1]
    d, h, w = src.shape[3:]
    src_f = src.permute(0, 2, 1, 3, 4, 5).reshape(bt, c, d, h, w)
    tgt_f = tgt.permute(0, 2, 1, 3, 4, 5).reshape(bt, c, d, h, w)
    flow_f = flow_fwd.permute(0, 2, 1, 3, 4, 5).reshape(bt, 3, d, h, w)
    warped = warp_volume(src_f, flow_f)
    loss = F.l1_loss(warped, tgt_f)
    if flow_bwd is not None:
        flow_b = flow_bwd.permute(0, 2, 1, 3, 4, 5).reshape(bt, 3, d, h, w)
        warped_b = warp_volume(tgt_f, flow_b)
        loss = loss + F.l1_loss(warped_b, src_f)
        loss = loss * 0.5
    return loss


def cycle_consistency_loss(recon: torch.Tensor, flow_fwd: torch.Tensor, flow_bwd: torch.Tensor) -> torch.Tensor:
    """Warp there and back: ||warp(warp(V_t, u_fwd), u_bwd) - V_t||_1."""
    _b, _c, t, _d, _h, _w = recon.shape
    if t < 2:
        return recon.sum() * 0.0
    src = recon[:, :, :-1]
    tgt = recon[:, :, 1:]
    bt = src.shape[0] * src.shape[2]
    c = src.shape[1]
    d, h, w = src.shape[3:]
    src_f = src.permute(0, 2, 1, 3, 4, 5).reshape(bt, c, d, h, w)
    tgt_f = tgt.permute(0, 2, 1, 3, 4, 5).reshape(bt, c, d, h, w)
    flow_f = flow_fwd.permute(0, 2, 1, 3, 4, 5).reshape(bt, 3, d, h, w)
    flow_b = flow_bwd.permute(0, 2, 1, 3, 4, 5).reshape(bt, 3, d, h, w)
    cyc_src = warp_volume(warp_volume(src_f, flow_f), flow_b)
    cyc_tgt = warp_volume(warp_volume(tgt_f, flow_b), flow_f)
    return 0.5 * (F.l1_loss(cyc_src, src_f) + F.l1_loss(cyc_tgt, tgt_f))


def volume_curve_loss(seg_logits_seq: torch.Tensor, lv_index: int = 1, rv_index: int = 2) -> torch.Tensor:
    """Soft second difference of LV/RV volume *fractions* over T (needs T >= 3).

    Counts are divided by D*H*W so the loss is O(1) and does not explode with
    spatial resolution (raw voxel counts made ``w_volsmooth`` dominate early smoke).
    """
    if seg_logits_seq.ndim != 6:
        raise ValueError(f"Expected (B,K,T,D,H,W), got {tuple(seg_logits_seq.shape)}")
    t = seg_logits_seq.shape[2]
    if t < 3:
        return seg_logits_seq.sum() * 0.0
    probs = F.softmax(seg_logits_seq, dim=1)
    k = probs.shape[1]
    spatial = float(probs.shape[3] * probs.shape[4] * probs.shape[5])
    terms = []
    for idx in (lv_index, rv_index):
        if idx >= k:
            continue
        # (B, T) fractional chamber volumes in [0, 1]
        frac = probs[:, idx].sum(dim=(2, 3, 4)) / max(spatial, 1.0)
        d2 = frac[:, 2:] - 2.0 * frac[:, 1:-1] + frac[:, :-2]
        terms.append((d2 ** 2).mean())
    if not terms:
        return seg_logits_seq.sum() * 0.0
    return sum(terms) / len(terms)


def temporal_seg_smoothness(seg_logits_seq: torch.Tensor) -> torch.Tensor:
    """L1 smoothness of softmax maps across adjacent frames."""
    if seg_logits_seq.shape[2] < 2:
        return seg_logits_seq.sum() * 0.0
    probs = F.softmax(seg_logits_seq, dim=1)
    return (probs[:, :, 1:] - probs[:, :, :-1]).abs().mean()

```

---
## FILE: reconseg3d/models/losses.py

```
"""Multi-task losses: recon / seg / MACE / Cox / motion / phenotype."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from reconseg3d.models.motion import (
    cycle_consistency_loss,
    temporal_seg_smoothness,
    volume_curve_loss,
    warp_consistency_loss,
)


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft Dice loss averaged over classes."""
    probs = F.softmax(logits, dim=1)
    targets = targets.long()
    valid = targets != ignore_index
    if not valid.any():
        return logits.sum() * 0.0

    loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
    count = 0
    for cls in range(num_classes):
        pred_c = probs[:, cls]
        target_c = (targets == cls).float()
        mask = valid.float()
        pred_c = pred_c * mask
        target_c = target_c * mask
        inter = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2 * inter + eps) / (union + eps)
        loss = loss + (1 - dice)
        count += 1
    return loss / max(count, 1)


def recon3d_mse(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Paper-style dense reconstruction MSE (α1)."""
    return F.mse_loss(reconstruction, target)


def seg3d_ce_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ce: nn.CrossEntropyLoss,
    use_dice: bool = True,
) -> torch.Tensor:
    """Paper-style 3D seg CE+Dice (α2)."""
    loss = ce(logits, targets.long())
    if use_dice:
        loss = loss + dice_loss(logits, targets, num_classes)
    return loss


def seg2d_from_slices(
    logits: torch.Tensor,
    targets: torch.Tensor,
    slice_mask: torch.Tensor,
    num_classes: int,
    ce: nn.CrossEntropyLoss,
    use_dice: bool = True,
) -> torch.Tensor:
    """
    Restrict CE+Dice to selected SA slices (paper α3 / seg2d).

    slice_mask: (B, D) or (B, D, H, W) with 1 on sampled slices.
    """
    target = targets.long().clone()
    mask = slice_mask
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    if mask.ndim == 2:
        mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(target)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1).expand_as(target) if mask.shape[1] != target.shape[1] else mask
    if mask.shape != target.shape:
        raise ValueError(f"slice_mask broadcast failed: {tuple(mask.shape)} vs {tuple(target.shape)}")
    target = target.masked_fill(~mask, -1)
    return seg3d_ce_dice(logits, target, num_classes, ce, use_dice=use_dice)


def cox_partial_likelihood(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Breslow Cox partial likelihood. NaN-safe for all-censored / single-event batches.

    risk: (B,) linear predictor (higher = higher hazard)
    time: (B,) follow-up
    event: (B,) 1 = event, 0 = censored
    """
    risk = risk.reshape(-1).float()
    time = time.reshape(-1).to(device=risk.device, dtype=risk.dtype)
    event = event.reshape(-1).to(device=risk.device, dtype=risk.dtype)
    n = risk.numel()
    if n == 0:
        return risk.sum() * 0.0
    n_events = event.sum()
    if n_events < 1:
        return (risk * 0.0).sum()
    # (N, N) at-risk: time_j >= time_i
    at_risk = time.unsqueeze(0) >= time.unsqueeze(1)
    risk_ij = risk.unsqueeze(0).expand(n, n)
    risk_ij = risk_ij.masked_fill(~at_risk, float("-inf"))
    lse = torch.logsumexp(risk_ij, dim=1)
    ll = event * (risk - lse)
    valid = torch.isfinite(ll) & (event > 0.5)
    if not valid.any():
        return (risk * 0.0).sum()
    return -(ll[valid].sum() / valid.float().sum().clamp_min(eps))


class FocalLoss(nn.Module):
    """Binary focal loss for MACE."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class MultiTaskLoss(nn.Module):
    """Weighted sum of seg + recon + MACE/Cox/phenotype + motion losses."""

    def __init__(
        self,
        num_seg_classes: int = 5,
        w_seg: float = 1.0,
        w_recon: float = 0.5,
        w_mace: float = 1.0,
        recon_loss: str = "l1",
        mace_loss: str = "bce",
        use_dice: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        alpha1: float = 0.0,
        alpha2: float = 0.0,
        alpha3: float = 0.0,
        w_warp: float = 0.0,
        w_cycle: float = 0.0,
        w_volsmooth: float = 0.0,
        w_segsmooth: float = 0.0,
        w_cox: float = 0.0,
        w_phenotype: float = 0.0,
        task: str = "mace",
        num_phenotype_classes: int = 5,
    ) -> None:
        super().__init__()
        self.num_seg_classes = num_seg_classes
        self.w_seg = w_seg
        self.w_recon = w_recon
        self.w_mace = w_mace
        self.use_dice = use_dice
        self.recon_loss = recon_loss
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.pheno_ce = nn.CrossEntropyLoss()
        self.mace_loss_type = mace_loss
        self.focal = FocalLoss(focal_alpha, focal_gamma)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.w_warp = w_warp
        self.w_cycle = w_cycle
        self.w_volsmooth = w_volsmooth
        self.w_segsmooth = w_segsmooth
        self.w_cox = w_cox
        self.w_phenotype = w_phenotype
        self.task = task
        self.num_phenotype_classes = num_phenotype_classes

    def forward(
        self,
        reconstruction: torch.Tensor,
        seg_logits: torch.Tensor,
        mace_logits: torch.Tensor,
        target_volume: torch.Tensor,
        target_seg: torch.Tensor,
        target_mace: torch.Tensor,
        flow: torch.Tensor | None = None,
        flow_bwd: torch.Tensor | None = None,
        seg_sequence: torch.Tensor | None = None,
        slice_mask: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        event: torch.Tensor | None = None,
        phenotype_logits: torch.Tensor | None = None,
        target_phenotype: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        zero = reconstruction.sum() * 0.0

        seg_ce = self.ce(seg_logits, target_seg.long())
        seg_dice = dice_loss(seg_logits, target_seg, self.num_seg_classes) if self.use_dice else zero
        seg_loss = seg_ce + seg_dice

        if self.recon_loss == "l2":
            recon_reg = F.mse_loss(reconstruction, target_volume)
        else:
            recon_reg = F.l1_loss(reconstruction, target_volume)
        recon_mse = recon3d_mse(reconstruction, target_volume)

        if self.task == "cox" or self.w_cox > 0:
            if time is None or event is None:
                cox_loss = zero
            else:
                cox_loss = cox_partial_likelihood(mace_logits, time, event)
        else:
            cox_loss = zero

        if self.task == "phenotype" and phenotype_logits is not None and target_phenotype is not None:
            pheno_loss = self.pheno_ce(phenotype_logits, target_phenotype.long())
        elif phenotype_logits is not None and target_phenotype is not None and self.w_phenotype > 0:
            pheno_loss = self.pheno_ce(phenotype_logits, target_phenotype.long())
        else:
            pheno_loss = zero

        if self.task == "cox":
            mace_loss = cox_loss
        elif self.mace_loss_type == "focal":
            mace_loss = self.focal(mace_logits, target_mace)
        else:
            mace_loss = F.binary_cross_entropy_with_logits(mace_logits, target_mace.float())

        w_seg = self.alpha2 if self.alpha2 > 0 else self.w_seg
        recon_term = self.w_recon * recon_reg + self.alpha1 * recon_mse
        total = w_seg * seg_loss + recon_term + self.w_mace * mace_loss

        seg2d_loss = zero
        if self.alpha3 > 0 and slice_mask is not None:
            seg2d_loss = seg2d_from_slices(
                seg_logits, target_seg, slice_mask, self.num_seg_classes, self.ce, self.use_dice
            )
            total = total + self.alpha3 * seg2d_loss

        warp_loss = zero
        cycle_loss = zero
        vol_loss = zero
        smooth_loss = zero
        if self.w_warp > 0 and flow is not None and reconstruction.shape[2] > 1:
            warp_loss = warp_consistency_loss(reconstruction, flow, flow_bwd)
            total = total + self.w_warp * warp_loss
        if self.w_cycle > 0 and flow is not None and flow_bwd is not None and reconstruction.shape[2] > 1:
            cycle_loss = cycle_consistency_loss(reconstruction, flow, flow_bwd)
            total = total + self.w_cycle * cycle_loss
        if seg_sequence is not None:
            if self.w_volsmooth > 0:
                vol_loss = volume_curve_loss(seg_sequence)
                total = total + self.w_volsmooth * vol_loss
            if self.w_segsmooth > 0:
                smooth_loss = temporal_seg_smoothness(seg_sequence)
                total = total + self.w_segsmooth * smooth_loss

        if self.task != "cox" and self.w_cox > 0:
            total = total + self.w_cox * cox_loss
        if self.w_phenotype > 0:
            total = total + self.w_phenotype * pheno_loss
        elif self.task == "phenotype":
            total = total + pheno_loss

        return {
            "total": total,
            "seg": seg_loss.detach(),
            "recon": recon_reg.detach(),
            "recon_mse": recon_mse.detach(),
            "mace": mace_loss.detach(),
            "seg2d": seg2d_loss.detach() if torch.is_tensor(seg2d_loss) else zero.detach(),
            "warp": warp_loss.detach() if torch.is_tensor(warp_loss) else zero.detach(),
            "cycle": cycle_loss.detach() if torch.is_tensor(cycle_loss) else zero.detach(),
            "volsmooth": vol_loss.detach() if torch.is_tensor(vol_loss) else zero.detach(),
            "segsmooth": smooth_loss.detach() if torch.is_tensor(smooth_loss) else zero.detach(),
            "cox": cox_loss.detach() if torch.is_tensor(cox_loss) else zero.detach(),
            "phenotype": pheno_loss.detach() if torch.is_tensor(pheno_loss) else zero.detach(),
        }

```

---
## FILE: reconseg3d/models/heart_ttable.py

```
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

```

---
