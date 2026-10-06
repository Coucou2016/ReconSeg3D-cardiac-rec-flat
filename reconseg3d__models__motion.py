"""Differentiable 3D motion (displacement) and volume warping.

Grid convention (shared by image and vector warps)
-------------------------------------------------
Displacement channels are ``(dz, dy, dx)`` in voxel units. ``flow_to_grid``
converts to ``grid_sample`` coordinates ``(x, y, z)`` with ``align_corners=True``
and scale ``2 / max(dim - 1, 1)``. Positive ``dx`` pulls samples from +x
(content appears to move toward -x).

Image-cycle vs inverse-consistency
----------------------------------
``cycle_consistency_loss`` (image-cycle) is an **auxiliary** intensity residual
after fwd/bwd warps. True inverse consistency is ``inverse_consistency_loss``:
``L_inv = ||u + W(v, u)|| + ||v + W(u, v)||`` with the same pull composition
as VoxelMorph-style registration.

Adjacent vs ED-anchored vs closed-cycle
---------------------------------------
MotionNet predicts **T** pull pairs for ``T`` frames: adjacent ``u_t``
(t→t+1 for ``t=0..T-2``) plus the **closing** edge ``u_{T-1}`` (T-1→0), and
matching backward pairs. Flow tensors are ``(B, 3, T, D, H, W)``.

Geometry losses ``L_inv`` / ``L_smooth`` / ``L_jac`` act on all ``T`` pairs
(including the closing edge). ``L_periodic`` / ``loop_consistency_loss``
composes **including the closing edge** so the cycle can close to ≈0.
Do **not** call adjacent-only (T-1) composition a "full cycle".

Linear ED↔frame paths use only the adjacent ``T-1`` slots (strip closing via
``adjacent_flow_stack``). The ED-anchored path then applies the same L_inv on
composed ``φ_{k→ED}`` / ``φ_{ED→k}`` (``w_ed_ref``).
"""

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
    into a sampling grid (B, D, H, W, 3) for ``grid_sample`` (align_corners=True).

    Scaling uses ``2 / max(dim - 1, 1)`` so singleton axes (D/H/W == 1) stay finite.
    Flow is interpreted as the inverse-sampling displacement expected by ``grid_sample``:
    a positive ``dx`` shifts the sampling location toward +x (source content appears
    to move toward -x in the warped volume).
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
    """Warp (B, C, D, H, W) with displacement (B, 3, D, H, W) via ``grid_sample``."""
    grid = flow_to_grid(flow)
    return F.grid_sample(volume, grid, mode="bilinear", padding_mode="border", align_corners=True)


def warp_vector(vec: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp a vector field with the **same** pull sampling as ``warp_volume``.

    Args:
        vec: (B, 3, D, H, W) field to be sampled
        flow: (B, 3, D, H, W) pull displacement (dz, dy, dx)
    """
    if vec.shape != flow.shape:
        raise ValueError(f"vec/flow shape mismatch: {tuple(vec.shape)} vs {tuple(flow.shape)}")
    return warp_volume(vec, flow)


def compose_pull(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Compose two pull displacements: ``u ∘ v ≈ v + W(u, v)``.

    Applies ``u`` after ``v`` under the same ``grid_sample`` convention as image warp.
    """
    return v + warp_vector(u, v)


def inverse_consistency_loss(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    True inverse-consistency: ``L_inv = ||u + W(v, u)|| + ||v + W(u, v)||``.

    For exact inverses, both residuals vanish. Uses the same warp as images.
    Accepts either a single pair (B, 3, D, H, W) or a temporal stack
    (B, 3, T-1, D, H, W).
    """
    u_f, v_f = _flatten_flow_pairs(u, v)
    res_u = u_f + warp_vector(v_f, u_f)
    res_v = v_f + warp_vector(u_f, v_f)
    return res_u.abs().mean() + res_v.abs().mean()


def _flatten_flow_pairs(
    flow_a: torch.Tensor, flow_b: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(B,3,D,H,W) or (B,3,T_pairs,D,H,W) -> flat (N,3,D,H,W)."""
    if flow_a.ndim == 5:
        return flow_a, flow_b
    if flow_a.ndim != 6:
        raise ValueError(f"Expected 5D or 6D flow, got {tuple(flow_a.shape)}")
    b, _, t_pairs, d, h, w = flow_a.shape
    a = flow_a.permute(0, 2, 1, 3, 4, 5).reshape(b * t_pairs, 3, d, h, w)
    if flow_b is None:
        return a, None
    if flow_b.shape != flow_a.shape:
        raise ValueError(f"flow pair shape mismatch: {tuple(flow_a.shape)} vs {tuple(flow_b.shape)}")
    bb = flow_b.permute(0, 2, 1, 3, 4, 5).reshape(b * t_pairs, 3, d, h, w)
    return a, bb


def infer_n_frames(flow_fwd: torch.Tensor, n_frames: int | None = None) -> tuple[int, bool]:
    """
    Infer frame count and whether ``flow_fwd`` includes a closing edge.

    Closed-cycle (MotionNet default): ``n_pairs == n_frames`` (last slot closes).
    Legacy open: ``n_pairs == n_frames - 1``.
    """
    if flow_fwd.ndim != 6:
        raise ValueError(f"Expected (B,3,T_pairs,D,H,W), got {tuple(flow_fwd.shape)}")
    n_pairs = int(flow_fwd.shape[2])
    if n_frames is not None:
        if n_pairs == n_frames:
            return int(n_frames), True
        if n_pairs == n_frames - 1:
            return int(n_frames), False
        raise ValueError(f"flow pairs {n_pairs} incompatible with n_frames={n_frames}")
    # Default: treat as closed cycle (T pairs for T frames).
    return n_pairs, True


def adjacent_flow_stack(flow: torch.Tensor, *, closed_cycle: bool | None = None) -> torch.Tensor:
    """Return adjacent-only stack ``(B,3,T-1,...)``; strip closing edge when present."""
    if flow.ndim != 6:
        raise ValueError(f"Expected 6D flow, got {tuple(flow.shape)}")
    if closed_cycle is None:
        # MotionNet emits closed stacks; legacy open stacks pass through.
        closed_cycle = True
    if closed_cycle and flow.shape[2] >= 1:
        return flow[:, :, :-1]
    return flow


def smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    """``L_smooth = ||∇u||²`` via finite differences on (dz, dy, dx) channels."""
    flow_f, _ = _flatten_flow_pairs(flow)
    dz = flow_f[:, :, 1:, :, :] - flow_f[:, :, :-1, :, :]
    dy = flow_f[:, :, :, 1:, :] - flow_f[:, :, :, :-1, :]
    dx = flow_f[:, :, :, :, 1:] - flow_f[:, :, :, :, :-1]
    return (dz.pow(2).mean() + dy.pow(2).mean() + dx.pow(2).mean()) / 3.0


def jacobian_determinant(flow: torch.Tensor) -> torch.Tensor:
    """
    Approximate ``det(I + ∇u)`` with central/forward finite differences.

    Returns (N, D-2, H-2, W-2) for interior voxels when D,H,W >= 3; otherwise
    a reduced interior matching available size.
    """
    flow_f, _ = _flatten_flow_pairs(flow)
    # partials of each component; channels: 0=dz, 1=dy, 2=dx
    # Use central differences on interior when possible.
    _, _, d, h, w = flow_f.shape
    if d < 2 or h < 2 or w < 2:
        # Degenerate grid: identity Jacobian
        return torch.ones(flow_f.shape[0], max(d - 2, 1), max(h - 2, 1), max(w - 2, 1), device=flow_f.device, dtype=flow_f.dtype)

    def _grad(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # field (N, D, H, W) -> grads along z,y,x on interior
        gz = 0.5 * (field[:, 2:, 1:-1, 1:-1] - field[:, :-2, 1:-1, 1:-1])
        gy = 0.5 * (field[:, 1:-1, 2:, 1:-1] - field[:, 1:-1, :-2, 1:-1])
        gx = 0.5 * (field[:, 1:-1, 1:-1, 2:] - field[:, 1:-1, 1:-1, :-2])
        return gz, gy, gx

    uz, uy, ux = flow_f[:, 0], flow_f[:, 1], flow_f[:, 2]
    duz_dz, duz_dy, duz_dx = _grad(uz)
    duy_dz, duy_dy, duy_dx = _grad(uy)
    dux_dz, dux_dy, dux_dx = _grad(ux)

    # J = I + ∇u with rows (z, y, x) matching channel order
    j00 = 1.0 + duz_dz
    j01 = duz_dy
    j02 = duz_dx
    j10 = duy_dz
    j11 = 1.0 + duy_dy
    j12 = duy_dx
    j20 = dux_dz
    j21 = dux_dy
    j22 = 1.0 + dux_dx
    det = (
        j00 * (j11 * j22 - j12 * j21)
        - j01 * (j10 * j22 - j12 * j20)
        + j02 * (j10 * j21 - j11 * j20)
    )
    return det


def folding_penalty(flow: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """``ReLU(ε - det J)`` mean; discourages local folding (det J < ε)."""
    det = jacobian_determinant(flow)
    return F.relu(eps - det).mean()


def jacobian_stats(flow: torch.Tensor, eps: float = 0.0) -> dict[str, float]:
    """Metric helpers: negative Jacobian ratio and detJ mean/min."""
    det = jacobian_determinant(flow).detach()
    if det.numel() == 0:
        return {"jac_neg_ratio": float("nan"), "jac_det_mean": float("nan"), "jac_det_min": float("nan")}
    neg = (det < eps).float().mean().item()
    return {
        "jac_neg_ratio": float(neg),
        "jac_det_mean": float(det.mean().item()),
        "jac_det_min": float(det.min().item()),
    }


def compose_flow_sequence(flows: list[torch.Tensor]) -> torch.Tensor:
    """Left-fold compose_pull over a list of (B,3,D,H,W) pull fields."""
    if not flows:
        raise ValueError("empty flow list")
    out = flows[0]
    for f in flows[1:]:
        out = compose_pull(f, out)
    return out


def loop_consistency_loss(flow_fwd: torch.Tensor, *, require_closing: bool = True) -> torch.Tensor:
    """
    Periodic / closed-cycle composition ≈ identity (``L_periodic``).

    ``flow_fwd`` should be ``(B, 3, T, D, H, W)`` including the closing edge
    ``u_{T-1}`` (T-1→0). Composes ``u_0 ∘ … ∘ u_{T-1}`` and penalizes
    ``||composed||_1``.

    If a legacy open stack ``(B,3,T-1,...)`` is passed and ``require_closing`` is
    False, adjacent-only composition is used — **not** a true full cycle.
    """
    if flow_fwd.ndim != 6:
        raise ValueError(f"Expected (B,3,T_pairs,D,H,W), got {tuple(flow_fwd.shape)}")
    n_pairs = int(flow_fwd.shape[2])
    if n_pairs < 1:
        return flow_fwd.sum() * 0.0
    if require_closing and n_pairs < 2:
        return flow_fwd.sum() * 0.0
    flows = [flow_fwd[:, :, i] for i in range(n_pairs)]
    composed = compose_flow_sequence(flows)
    return composed.abs().mean()


# Alias used in docs / PAPER_PLAN.
periodic_consistency_loss = loop_consistency_loss


def _as_ed_index(ed_index: torch.Tensor | int | None, batch: int, device: torch.device) -> torch.Tensor:
    """Normalize ED index to shape (B,) long on ``device``."""
    if ed_index is None:
        return torch.zeros(batch, dtype=torch.long, device=device)
    if isinstance(ed_index, int):
        return torch.full((batch,), int(ed_index), dtype=torch.long, device=device)
    t = ed_index.reshape(-1).long().to(device)
    if t.numel() == 1 and batch > 1:
        return t.expand(batch)
    if t.numel() != batch:
        raise ValueError(f"ed_index length {t.numel()} != batch {batch}")
    return t


def compose_flow_between_indices(
    flow_fwd: torch.Tensor,
    t_src: int,
    t_tgt: int,
    flow_bwd: torch.Tensor | None = None,
    *,
    n_frames: int | None = None,
) -> torch.Tensor:
    """
    Compose **adjacent** (non-wrapping) pull fields from frame ``t_src`` to ``t_tgt``.

    Accepts closed-cycle stacks ``(B,3,T,...)`` or legacy open ``(B,3,T-1,...)``.
    The closing edge is **not** used on linear paths.

    - ``t_tgt > t_src``: left-fold adjacent ``flow_fwd[:, :, t_src:t_tgt]``
    - ``t_tgt < t_src``: left-fold ``flow_bwd`` reverse adjacent steps
    - ``t_tgt == t_src``: zero field
    """
    if flow_fwd.ndim != 6:
        raise ValueError(f"Expected (B,3,T_pairs,D,H,W), got {tuple(flow_fwd.shape)}")
    t, closed = infer_n_frames(flow_fwd, n_frames=n_frames)
    adj_fwd = adjacent_flow_stack(flow_fwd, closed_cycle=closed)
    adj_bwd = adjacent_flow_stack(flow_bwd, closed_cycle=closed) if flow_bwd is not None else None
    b, _, t_adj, d, h, w = adj_fwd.shape
    if t_tgt == t_src:
        return torch.zeros(b, 3, d, h, w, device=flow_fwd.device, dtype=flow_fwd.dtype)
    if t_src < 0 or t_tgt < 0 or t_src >= t or t_tgt >= t:
        raise ValueError(f"t_src/t_tgt out of range for T={t}: {t_src}, {t_tgt}")
    if t_tgt > t_src:
        if t_tgt > t_adj:
            raise ValueError(f"t_tgt={t_tgt} exceeds available adjacent steps T-1={t_adj}")
        return compose_flow_sequence([adj_fwd[:, :, i] for i in range(t_src, t_tgt)])
    if adj_bwd is None:
        raise ValueError("flow_bwd required when composing toward earlier frames")
    if t_src > t_adj:
        raise ValueError(f"t_src={t_src} exceeds available backward steps T-1={t_adj}")
    # Adjacent bwd at index i takes frame i+1 toward frame i.
    return compose_flow_sequence([adj_bwd[:, :, i] for i in range(t_src - 1, t_tgt - 1, -1)])


def ed_anchored_paths(
    flow_fwd: torch.Tensor,
    ed_index: torch.Tensor | int | None = None,
    flow_bwd: torch.Tensor | None = None,
    *,
    n_frames: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build ED-anchored composed displacements for every frame.

    Composition (document clearly)
    ------------------------------
    MotionNet predicts closed-cycle ``u_t`` / ``v_t`` for ``t=0..T-1`` (last
    slot closes). Linear ED paths use only adjacent slots ``0..T-2``.

    For each frame ``k`` and ED index ``e``:

    - **to-ED** ``φ_{k→e}``: compose adjacent fields along the path from ``k``
      to ``e`` (forward folds of ``u`` if ``k < e``; backward folds of ``v`` if
      ``k > e``; zero if ``k == e``).
    - **from-ED** ``φ_{e→k}``: the opposite composition.

    Returns
    -------
    to_ed, from_ed : each ``(B, 3, T, D, H, W)``
    """
    if flow_fwd.ndim != 6:
        raise ValueError(f"Expected (B,3,T_pairs,D,H,W), got {tuple(flow_fwd.shape)}")
    b, _, _, d, h, w = flow_fwd.shape
    t, _closed = infer_n_frames(flow_fwd, n_frames=n_frames)
    ed = _as_ed_index(ed_index, b, flow_fwd.device)
    to_ed = torch.zeros(b, 3, t, d, h, w, device=flow_fwd.device, dtype=flow_fwd.dtype)
    from_ed = torch.zeros_like(to_ed)
    for e_val in ed.unique().tolist():
        e = int(e_val)
        if e < 0 or e >= t:
            raise ValueError(f"ed_index {e} out of range for T={t}")
        mask = ed == e
        for k in range(t):
            phi_to = compose_flow_between_indices(flow_fwd, k, e, flow_bwd=flow_bwd, n_frames=t)
            phi_from = compose_flow_between_indices(flow_fwd, e, k, flow_bwd=flow_bwd, n_frames=t)
            to_ed[mask, :, k] = phi_to[mask]
            from_ed[mask, :, k] = phi_from[mask]
    return to_ed, from_ed


def ed_reference_consistency_loss(
    flow_fwd: torch.Tensor,
    flow_bwd: torch.Tensor | None = None,
    ed_index: torch.Tensor | int | None = None,
    *,
    n_frames: int | None = None,
) -> torch.Tensor:
    """
    ED-anchored inverse consistency on composed paths.

    For each frame ``k ≠ e``, penalize
    ``||φ_{k→e} + W(φ_{e→k}, φ_{k→e})|| + ||φ_{e→k} + W(φ_{k→e}, φ_{e→k})||``
    using the same pull convention as adjacent ``inverse_consistency_loss``.
    Frames at ED contribute zero. Requires ``flow_bwd`` when any path goes
    backward in time relative to ED.
    """
    if flow_fwd.ndim != 6:
        raise ValueError(f"Expected (B,3,T_pairs,D,H,W), got {tuple(flow_fwd.shape)}")
    b, _, _, d, h, w = flow_fwd.shape
    t, _closed = infer_n_frames(flow_fwd, n_frames=n_frames)
    if t < 2:
        return flow_fwd.sum() * 0.0
    if flow_bwd is None:
        flow_bwd = torch.zeros_like(flow_fwd)
    to_ed, from_ed = ed_anchored_paths(flow_fwd, ed_index=ed_index, flow_bwd=flow_bwd, n_frames=t)
    ed = _as_ed_index(ed_index, b, flow_fwd.device)
    losses = []
    for k in range(t):
        active = ed != k
        if not bool(active.any()):
            continue
        u = to_ed[active, :, k]
        v = from_ed[active, :, k]
        losses.append(inverse_consistency_loss(u, v))
    if not losses:
        return flow_fwd.sum() * 0.0
    return sum(losses) / len(losses)


def scaling_and_squaring(v: torch.Tensor, steps: int = 7) -> torch.Tensor:
    """
    Integrate a stationary velocity field (SVF) via scaling-and-squaring.

    Enabled when ``MotionNet(use_svf=True)`` / ``model.use_svf: true`` (see
    ``configs/publication/publication_motion_svf.yaml``). Head outputs a
    stationary velocity; this converts it to a pull displacement.
    """
    if steps < 0:
        raise ValueError("steps must be >= 0")
    flow = v / float(2**steps)
    for _ in range(steps):
        flow = compose_pull(flow, flow)
    return flow


class MotionNet(nn.Module):
    """Predict 3D displacement (or SVF→displacement) between consecutive frames."""

    def __init__(self, in_channels: int = 1, base_channels: int = 8, use_svf: bool = False, svf_steps: int = 7) -> None:
        super().__init__()
        c = max(base_channels, 4)
        self.use_svf = use_svf
        self.svf_steps = int(svf_steps)
        self.encoder = nn.Sequential(
            ConvBlock3D(in_channels * 2, c),
            ConvBlock3D(c, c),
        )
        self.flow_head = nn.Conv3d(c, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    def forward_pair(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """src/tgt: (B, C, D, H, W) -> pull displacement (B, 3, D, H, W)."""
        raw = self.flow_head(self.encoder(torch.cat([src, tgt], dim=1)))
        if self.use_svf:
            return scaling_and_squaring(raw, steps=self.svf_steps)
        return raw

    def forward(self, recon: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Args:
            recon: (B, C, T, D, H, W)
        Returns:
            flow_fwd, flow_bwd each ``(B, 3, T, D, H, W)`` — adjacent pairs
            ``0..T-2`` plus closing ``T-1 → 0`` (and reverse). ``(None, None)``
            if ``T < 2``.
        """
        if recon.ndim != 6:
            raise ValueError(f"Expected (B,C,T,D,H,W), got {tuple(recon.shape)}")
        b, c, t, d, h, w = recon.shape
        if t < 2:
            return None, None
        # Adjacent t → t+1 plus closing T-1 → 0.
        src_adj = recon[:, :, :-1]
        tgt_adj = recon[:, :, 1:]
        src = torch.cat([src_adj, recon[:, :, -1:]], dim=2)
        tgt = torch.cat([tgt_adj, recon[:, :, :1]], dim=2)
        src_f = src.permute(0, 2, 1, 3, 4, 5).reshape(b * t, c, d, h, w)
        tgt_f = tgt.permute(0, 2, 1, 3, 4, 5).reshape(b * t, c, d, h, w)
        fwd = self.forward_pair(src_f, tgt_f).view(b, t, 3, d, h, w).permute(0, 2, 1, 3, 4, 5)
        bwd = self.forward_pair(tgt_f, src_f).view(b, t, 3, d, h, w).permute(0, 2, 1, 3, 4, 5)
        return fwd.contiguous(), bwd.contiguous()


def _pair_volumes_for_flow(
    recon: torch.Tensor, flow_fwd: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align src/tgt stacks to flow pairs (closed T or legacy T-1)."""
    b, c, t, d, h, w = recon.shape
    n_pairs = int(flow_fwd.shape[2])
    if n_pairs == t:
        src = torch.cat([recon[:, :, :-1], recon[:, :, -1:]], dim=2)
        tgt = torch.cat([recon[:, :, 1:], recon[:, :, :1]], dim=2)
    elif n_pairs == t - 1:
        src = recon[:, :, :-1]
        tgt = recon[:, :, 1:]
    else:
        raise ValueError(f"flow pairs {n_pairs} incompatible with recon T={t}")
    return src, tgt, flow_fwd


def warp_consistency_loss(recon: torch.Tensor, flow_fwd: torch.Tensor, flow_bwd: torch.Tensor | None = None) -> torch.Tensor:
    """||Vhat_{tgt} - warp(Vhat_src, u)||_1 over all predicted pairs (+ bwd)."""
    _b, _c, t, _d, _h, _w = recon.shape
    if t < 2:
        return recon.sum() * 0.0
    src, tgt, flow_fwd = _pair_volumes_for_flow(recon, flow_fwd)
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
    """
    **Auxiliary** image-cycle consistency: ||warp(warp(V_src, u_fwd), u_bwd) - V_src||_1.

    This is **not** coordinate-composed inverse-consistent flow regularization;
    prefer ``inverse_consistency_loss`` for L_inv. Kept as a light intensity
    auxiliary when ``w_cycle`` > 0. Operates on all MotionNet pairs (incl. close).
    """
    _b, _c, t, _d, _h, _w = recon.shape
    if t < 2:
        return recon.sum() * 0.0
    src, tgt, flow_fwd = _pair_volumes_for_flow(recon, flow_fwd)
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
    """
    Soft second difference of LV/RV volume *fractions* over T (needs T >= 3).

    **Physiological regularizer only** (demoted relative to inv/smooth/jac/loop).
    Counts are divided by D*H*W so the loss is O(1).
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
