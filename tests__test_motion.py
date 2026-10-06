"""Motion warp, inverse-consistency, smoothness, Jacobian, loop, ED-ref."""

import torch

from reconseg3d.models.losses import FocalLoss, MultiTaskLoss
from reconseg3d.models.motion import (
    MotionNet,
    compose_flow_between_indices,
    compose_pull,
    cycle_consistency_loss,
    ed_anchored_paths,
    ed_reference_consistency_loss,
    flow_to_grid,
    folding_penalty,
    inverse_consistency_loss,
    jacobian_determinant,
    jacobian_stats,
    loop_consistency_loss,
    smoothness_loss,
    warp_consistency_loss,
    warp_vector,
    warp_volume,
)
from reconseg3d.models.reconseg3d import ReconSeg3D


def test_warp_identity_near_zero():
    vol = torch.randn(2, 1, 8, 16, 16)
    flow = torch.zeros(2, 3, 8, 16, 16)
    warped = warp_volume(vol, flow)
    assert torch.allclose(warped, vol, atol=1e-5)


def test_flow_to_grid_channel_order_and_scale():
    """Channels are (dz, dy, dx); grid last-dim is (x, y, z) for grid_sample."""
    d, h, w = 4, 5, 6
    flow = torch.zeros(1, 3, d, h, w)
    flow[:, 2] = 1.0  # +1 voxel in x
    grid = flow_to_grid(flow)
    assert grid.shape == (1, d, h, w, 3)
    expected_dx = 2.0 / (w - 1)
    mid = (d // 2, h // 2, w // 2)
    base_x = torch.linspace(-1.0, 1.0, w)[mid[2]]
    assert torch.isclose(grid[0, mid[0], mid[1], mid[2], 0], base_x + expected_dx, atol=1e-5)


def test_warp_plus_one_voxel_x_shifts_content_left():
    """Positive dx samples from +x → content appears shifted toward -x."""
    vol = torch.zeros(1, 1, 4, 4, 8)
    vol[..., 4] = 1.0  # spike at x=4
    flow = torch.zeros(1, 3, 4, 4, 8)
    flow[:, 2] = 1.0
    warped = warp_volume(vol, flow)
    assert float(warped[..., 3].mean()) > float(vol[..., 3].mean())
    assert float(warped[..., 4].mean()) < float(vol[..., 4].mean()) + 1e-3


def test_warp_vector_matches_volume_convention():
    vec = torch.randn(1, 3, 8, 8, 8)
    flow = torch.zeros(1, 3, 8, 8, 8)
    flow[:, 2] = 1.0
    out = warp_vector(vec, flow)
    for c in range(3):
        assert torch.allclose(out[:, c : c + 1], warp_volume(vec[:, c : c + 1], flow), atol=1e-5)


def test_inverse_consistency_identity_flow():
    u = torch.zeros(2, 3, 8, 8, 8)
    v = torch.zeros(2, 3, 8, 8, 8)
    assert float(inverse_consistency_loss(u, v)) < 1e-6


def test_inverse_consistency_exact_inverse_translation():
    """+1 voxel u and −1 voxel v → L_inv ≈ 0 (interior / bilinear edge effects small)."""
    d, h, w = 16, 16, 16
    u = torch.zeros(1, 3, d, h, w)
    v = torch.zeros(1, 3, d, h, w)
    u[:, 2] = 1.0
    v[:, 2] = -1.0
    loss = inverse_consistency_loss(u, v)
    assert float(loss) < 0.15


def test_compose_pull_identity():
    u = torch.randn(1, 3, 8, 8, 8) * 0.01
    zero = torch.zeros_like(u)
    assert torch.allclose(compose_pull(u, zero), u, atol=1e-5)
    assert torch.allclose(compose_pull(zero, u), u, atol=1e-5)


def test_smoothness_and_jacobian_finite():
    flow = torch.randn(1, 3, 12, 12, 12) * 0.1
    s = smoothness_loss(flow)
    j = folding_penalty(flow, eps=0.0)
    det = jacobian_determinant(flow)
    stats = jacobian_stats(flow)
    assert torch.isfinite(s) and torch.isfinite(j)
    assert torch.isfinite(det).all()
    assert 0.0 <= stats["jac_neg_ratio"] <= 1.0


def test_loop_consistency_identity():
    # Closed-cycle stack: T pairs for T frames.
    flow = torch.zeros(1, 3, 4, 8, 8, 8)
    assert float(loop_consistency_loss(flow)) < 1e-6


def test_closed_cycle_periodic_only_with_closing_edge():
    """Synthetic +1+1+1-3=0 closes only when the closing edge is included."""
    d = h = w = 12
    # Three adjacent +1 dx and closing -3 → compose ≈ 0 (periodic).
    flow_closed = torch.zeros(1, 3, 4, d, h, w)
    flow_closed[:, 2, 0] = 1.0
    flow_closed[:, 2, 1] = 1.0
    flow_closed[:, 2, 2] = 1.0
    flow_closed[:, 2, 3] = -3.0
    closed = float(loop_consistency_loss(flow_closed))
    # Adjacent-only (no closing): +1+1+1 ≠ 0
    flow_open = flow_closed[:, :, :3]
    open_loss = float(loop_consistency_loss(flow_open, require_closing=False))
    assert closed < 0.25
    assert open_loss > closed + 0.5
    assert open_loss >= 1.0


def test_motionnet_emits_t_pairs_including_close():
    recon = torch.randn(1, 1, 4, 8, 16, 16)
    net = MotionNet(in_channels=1, base_channels=4)
    fwd, bwd = net(recon)
    assert fwd is not None and bwd is not None
    assert fwd.shape == (1, 3, 4, 8, 16, 16)
    assert bwd.shape == fwd.shape


def test_ed_anchored_paths_zero_when_identity_flows():
    """Zero adjacent+closing flows → zero ED-anchored compositions; L_ed_ref ≈ 0."""
    b, t, d, h, w = 2, 4, 8, 8, 8
    fwd = torch.zeros(b, 3, t, d, h, w)  # closed
    bwd = torch.zeros_like(fwd)
    to_ed, from_ed = ed_anchored_paths(fwd, ed_index=0, flow_bwd=bwd)
    assert to_ed.shape == (b, 3, t, d, h, w)
    assert torch.allclose(to_ed, torch.zeros_like(to_ed))
    assert torch.allclose(from_ed, torch.zeros_like(from_ed))
    assert float(ed_reference_consistency_loss(fwd, bwd, ed_index=0)) < 1e-6


def test_compose_flow_between_forward_and_backward():
    # Closed T=3 → 3 pairs; linear compose uses adjacent only.
    fwd = torch.zeros(1, 3, 3, 8, 8, 8)
    bwd = torch.zeros(1, 3, 3, 8, 8, 8)
    fwd[:, 2, 0] = 1.0
    fwd[:, 2, 1] = 1.0
    bwd[:, 2, 0] = -1.0
    bwd[:, 2, 1] = -1.0
    phi = compose_flow_between_indices(fwd, 0, 2, flow_bwd=bwd)
    assert float(phi[:, 2].abs().mean()) > 1.0
    phi_back = compose_flow_between_indices(fwd, 2, 0, flow_bwd=bwd)
    assert float(phi_back[:, 2].abs().mean()) > 1.0


def test_ed_reference_loss_backward():
    recon = torch.randn(1, 1, 4, 8, 16, 16, requires_grad=True)
    net = MotionNet(in_channels=1, base_channels=4)
    fwd, bwd = net(recon)
    loss = ed_reference_consistency_loss(fwd, bwd, ed_index=0)
    assert torch.isfinite(loss)
    loss.backward()
    assert recon.grad is not None


def test_flow_to_grid_singleton_axis_finite():
    flow = torch.zeros(1, 3, 1, 4, 4)
    flow[:, 0] = 0.5
    grid = flow_to_grid(flow)
    assert torch.isfinite(grid).all()


def test_warp_and_cycle_finite_backward():
    recon = torch.randn(2, 1, 4, 8, 16, 16, requires_grad=True)
    net = MotionNet(in_channels=1, base_channels=4)
    fwd, bwd = net(recon)
    assert fwd is not None and bwd is not None
    warp_l = warp_consistency_loss(recon, fwd, bwd)
    cyc_l = cycle_consistency_loss(recon, fwd, bwd)
    inv_l = inverse_consistency_loss(fwd, bwd)
    ed_l = ed_reference_consistency_loss(fwd, bwd, ed_index=1)
    total = (
        warp_l
        + cyc_l
        + inv_l
        + smoothness_loss(fwd)
        + folding_penalty(fwd)
        + loop_consistency_loss(fwd)
        + ed_l
    )
    assert torch.isfinite(total)
    total.backward()
    assert recon.grad is not None
    assert torch.isfinite(recon.grad).all()


def test_multitask_motion_geometry_terms():
    model = ReconSeg3D(clinical_dim=0, per_frame_recon=True, base_channels=8)
    criterion = MultiTaskLoss(
        num_seg_classes=5,
        w_warp=0.05,
        w_cycle=0.02,
        w_inv=0.1,
        w_smooth=0.05,
        w_jac=0.05,
        w_loop=0.05,
        w_ed_ref=0.05,
        w_volsmooth=0.01,
        alpha1=0.1,
        w_mace=0.0,
    )
    x = torch.randn(1, 1, 4, 8, 16, 16)
    seg = torch.randint(0, 5, (1, 8, 16, 16))
    mace = torch.tensor([1.0])
    out = model(x, seg_frame_indices=torch.tensor([0]))
    losses = criterion(
        out.reconstruction,
        out.segmentation,
        out.mace_logits,
        x,
        seg,
        mace,
        flow=out.flow,
        flow_bwd=out.flow_bwd,
        seg_sequence=out.seg_sequence,
        ed_index=torch.tensor([0]),
    )
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert torch.isfinite(losses["inv"])
    assert torch.isfinite(losses["smooth"])
    assert torch.isfinite(losses["jac"])
    assert torch.isfinite(losses["loop"])
    assert torch.isfinite(losses["ed_ref"])


def test_focal_loss_alpha_t_weighting():
    """α_t = α·y + (1-α)·(1-y); positive and negative classes weighted differently."""
    fl = FocalLoss(alpha=0.25, gamma=0.0)
    logits = torch.tensor([0.0, 0.0])
    y = torch.tensor([1.0, 0.0])
    loss = fl(logits, y)
    assert torch.isfinite(loss)
    import math

    assert abs(float(loss) - 0.5 * math.log(2)) < 1e-4


def test_volume_curve_loss_normalized_scale():
    """Fractional volumes keep volsmooth O(1), not O((DHW)^2)."""
    from reconseg3d.models.motion import volume_curve_loss

    logits = torch.zeros(2, 5, 8, 16, 32, 32)
    logits[:, 1] = 0.5
    logits[:, 2] = 0.5
    flat = volume_curve_loss(logits)
    assert float(flat) < 1e-6

    noisy = torch.randn(2, 5, 8, 16, 32, 32)
    v = volume_curve_loss(noisy)
    assert torch.isfinite(v)
    assert float(v) < 1.0
