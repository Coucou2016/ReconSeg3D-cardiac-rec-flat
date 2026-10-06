"""Multi-task losses: recon / seg / motion geometry / optional MACE·Cox / phenotype.

Main publication path emphasizes reconstruction, segmentation, and geometry-aware
motion (inverse consistency, smoothness, Jacobian folding, loop). Image-cycle and
volume-curve are auxiliary / physiological regularizers. MACE and Cox are optional
proxy / extensibility heads (``w_mace`` / ``w_cox`` default 0 in publication configs).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from reconseg3d.models.motion import (
    cycle_consistency_loss,
    ed_reference_consistency_loss,
    folding_penalty,
    inverse_consistency_loss,
    loop_consistency_loss,
    smoothness_loss,
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


def labeled_phase_seg_loss(
    seg_sequence: torch.Tensor,
    segmentation_sequence: torch.Tensor,
    seg_valid_mask: torch.Tensor,
    num_classes: int,
    ce: nn.CrossEntropyLoss,
    use_dice: bool = True,
) -> torch.Tensor:
    """
    Supervise only labeled cardiac phases.

    Args:
        seg_sequence: (B, K, T, D, H, W) predicted logits
        segmentation_sequence: (B, T, D, H, W) GT labels (-1 = unlabeled)
        seg_valid_mask: (B, T) bool / 0-1 mask of labeled frames
    """
    if seg_sequence.ndim != 6:
        raise ValueError(f"Expected seg_sequence (B,K,T,D,H,W), got {tuple(seg_sequence.shape)}")
    b, k, t, d, h, w = seg_sequence.shape
    mask = seg_valid_mask
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    if mask.shape != (b, t):
        raise ValueError(f"seg_valid_mask must be (B,T)={b,t}, got {tuple(mask.shape)}")
    if not mask.any():
        return seg_sequence.sum() * 0.0
    logits = seg_sequence.permute(0, 2, 1, 3, 4, 5).reshape(b * t, k, d, h, w)
    targets = segmentation_sequence.reshape(b * t, d, h, w).long()
    flat_mask = mask.reshape(b * t)
    logits = logits[flat_mask]
    targets = targets[flat_mask]
    # Also honor ignore_index inside labeled frames
    return seg3d_ce_dice(logits, targets, num_classes, ce, use_dice=use_dice)


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

    Non-finite entries are dropped with a shared mask so risk/time/event stay aligned.
    Zero-event / empty batches return a graph-safe ``risk.sum() * 0.0``.
    """
    risk = risk.reshape(-1).float()
    time = time.reshape(-1).to(device=risk.device, dtype=risk.dtype)
    event = event.reshape(-1).to(device=risk.device, dtype=risk.dtype)
    if risk.numel() == 0:
        return risk.sum() * 0.0
    finite = torch.isfinite(risk) & torch.isfinite(time) & torch.isfinite(event)
    if not finite.any():
        return risk.sum() * 0.0
    risk = risk[finite]
    time = time[finite]
    event = event[finite]
    n = risk.numel()
    if float(event.sum().item()) < 1.0:
        return risk.sum() * 0.0
    # (N, N) at-risk: time_j >= time_i
    at_risk = time.unsqueeze(0) >= time.unsqueeze(1)
    risk_ij = risk.unsqueeze(0).expand(n, n)
    risk_ij = risk_ij.masked_fill(~at_risk, float("-inf"))
    lse = torch.logsumexp(risk_ij, dim=1)
    ll = event * (risk - lse)
    valid = torch.isfinite(ll) & (event > 0.5)
    if not valid.any():
        return risk.sum() * 0.0
    return -(ll[valid].sum() / valid.float().sum().clamp_min(eps))


class FocalLoss(nn.Module):
    """Binary focal loss for optional MACE proxy head.

    Uses standard α_t weighting: ``α_t = α·y + (1-α)·(1-y)``.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal = alpha_t * (1.0 - pt).clamp_min(0.0) ** self.gamma * bce
        return focal.mean()


class MultiTaskLoss(nn.Module):
    """Weighted sum of recon / seg / geometry-motion / optional risk heads.

    Motion geometry (preferred for publication):
        ``w_inv``, ``w_smooth``, ``w_jac``,
        ``w_loop`` (closed-cycle / ``L_periodic`` — T pairs incl. closing edge),
        ``w_ed_ref`` (ED-anchored composed-path inverse consistency)
    Auxiliary:
        ``w_warp`` (intensity), ``w_cycle`` (image-cycle, not L_inv),
        ``w_volsmooth`` (physiological volume-curve regularizer — demoted)
    Optional / supplementary:
        ``w_mace``, ``w_cox``, ``w_phenotype``
    """

    def __init__(
        self,
        num_seg_classes: int = 5,
        w_seg: float = 1.0,
        w_recon: float = 0.5,
        w_mace: float = 0.0,
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
        w_inv: float = 0.0,
        w_smooth: float = 0.0,
        w_jac: float = 0.0,
        jac_eps: float = 0.0,
        w_loop: float = 0.0,
        w_ed_ref: float = 0.0,
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
        self.w_inv = w_inv
        self.w_smooth = w_smooth
        self.w_jac = w_jac
        self.jac_eps = jac_eps
        self.w_loop = w_loop
        self.w_ed_ref = w_ed_ref
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
        segmentation_sequence: torch.Tensor | None = None,
        seg_valid_mask: torch.Tensor | None = None,
        slice_mask: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        event: torch.Tensor | None = None,
        phenotype_logits: torch.Tensor | None = None,
        target_phenotype: torch.Tensor | None = None,
        ed_index: torch.Tensor | int | None = None,
    ) -> dict[str, torch.Tensor]:
        zero = reconstruction.sum() * 0.0

        # Prefer multi-phase labeled supervision when sequence GT is provided.
        if (
            seg_sequence is not None
            and segmentation_sequence is not None
            and seg_valid_mask is not None
            and (self.alpha2 > 0 or self.w_seg > 0)
        ):
            seg_loss = labeled_phase_seg_loss(
                seg_sequence,
                segmentation_sequence,
                seg_valid_mask,
                self.num_seg_classes,
                self.ce,
                self.use_dice,
            )
            seg_ce = zero
            seg_dice = zero
        else:
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
        inv_loss = zero
        smooth_flow_loss = zero
        jac_loss = zero
        loop_loss = zero
        ed_ref_loss = zero
        vol_loss = zero
        smooth_loss = zero

        has_motion = flow is not None and reconstruction.shape[2] > 1
        if has_motion and self.w_warp > 0:
            warp_loss = warp_consistency_loss(reconstruction, flow, flow_bwd)
            total = total + self.w_warp * warp_loss
        # Image-cycle is auxiliary only (not L_inv).
        if has_motion and flow_bwd is not None and self.w_cycle > 0:
            cycle_loss = cycle_consistency_loss(reconstruction, flow, flow_bwd)
            total = total + self.w_cycle * cycle_loss
        if has_motion and flow_bwd is not None and self.w_inv > 0:
            inv_loss = inverse_consistency_loss(flow, flow_bwd)
            total = total + self.w_inv * inv_loss
        if has_motion and self.w_smooth > 0:
            smooth_flow_loss = smoothness_loss(flow)
            if flow_bwd is not None:
                smooth_flow_loss = 0.5 * (smooth_flow_loss + smoothness_loss(flow_bwd))
            total = total + self.w_smooth * smooth_flow_loss
        if has_motion and self.w_jac > 0:
            jac_loss = folding_penalty(flow, eps=self.jac_eps)
            if flow_bwd is not None:
                jac_loss = 0.5 * (jac_loss + folding_penalty(flow_bwd, eps=self.jac_eps))
            total = total + self.w_jac * jac_loss
        if has_motion and self.w_loop > 0:
            loop_loss = loop_consistency_loss(flow)
            total = total + self.w_loop * loop_loss
        # ED-anchored composed-path L_inv (defaults ed_index=0 when absent).
        if has_motion and self.w_ed_ref > 0:
            ed_ref_loss = ed_reference_consistency_loss(flow, flow_bwd, ed_index=ed_index)
            total = total + self.w_ed_ref * ed_ref_loss

        if seg_sequence is not None:
            # Physiological volume-curve regularizer (demoted vs geometry terms).
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

        def _d(t: torch.Tensor) -> torch.Tensor:
            return t.detach() if torch.is_tensor(t) else zero.detach()

        return {
            "total": total,
            "seg": seg_loss.detach(),
            "recon": recon_reg.detach(),
            "recon_mse": recon_mse.detach(),
            "mace": mace_loss.detach(),
            "seg2d": _d(seg2d_loss),
            "warp": _d(warp_loss),
            "cycle": _d(cycle_loss),
            "inv": _d(inv_loss),
            "smooth": _d(smooth_flow_loss),
            "jac": _d(jac_loss),
            "loop": _d(loop_loss),
            "ed_ref": _d(ed_ref_loss),
            "volsmooth": _d(vol_loss),
            "segsmooth": _d(smooth_loss),
            "cox": _d(cox_loss),
            "phenotype": _d(pheno_loss),
        }
