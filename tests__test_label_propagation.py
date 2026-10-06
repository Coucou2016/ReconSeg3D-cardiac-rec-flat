"""Metrics: physical HD95 spacing, ED↔ES label propagation, SSIM proxy naming."""

import numpy as np
import torch

from reconseg3d.training.metrics import (
    compose_flows_between,
    ed_es_label_propagation_metrics,
    hd95_binary,
    propagate_label_with_flow,
)


def test_hd95_spacing_anisotropic():
    """1-voxel z-shift → ~1mm vs ~8mm for spacing (1,1,1) vs (8,1,1)."""
    pred = np.zeros((8, 8, 8), dtype=bool)
    tgt = np.zeros((8, 8, 8), dtype=bool)
    # Single-voxel blobs separated by 1 voxel along D (z).
    pred[3, 4, 4] = True
    tgt[4, 4, 4] = True
    vox = hd95_binary(pred, tgt, spacing=(1.0, 1.0, 1.0))
    mm = hd95_binary(pred, tgt, spacing=(8.0, 1.0, 1.0))
    assert abs(vox - 1.0) < 0.2
    assert abs(mm - 8.0) < 1.0


def test_label_propagation_known_translation():
    """Warp a blob by +1 voxel dx; Dice vs manually shifted GT is high."""
    label = torch.zeros(1, 8, 8, 12, dtype=torch.long)
    label[0, 3:6, 3:6, 5:8] = 1
    shifted = torch.zeros_like(label)
    shifted[0, 3:6, 3:6, 4:7] = 1  # content moved toward -x under +dx pull
    flow = torch.zeros(1, 3, 8, 8, 12)
    flow[:, 2] = 1.0
    prop = propagate_label_with_flow(label, flow, num_classes=2)
    # Interior overlap should be strong
    inter = ((prop == 1) & (shifted == 1)).sum().item()
    union = ((prop == 1) | (shifted == 1)).sum().item()
    dice = 2 * inter / max(union, 1)
    assert dice > 0.7


def test_ed_es_propagation_api_synthetic():
    """API smoke: identity closed-cycle flow → near-perfect Dice (same ED/ES masks)."""
    seg = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    seg[0, 2:6, 2:6, 2:6] = 1
    # T=3 closed → 3 pairs
    flow = torch.zeros(1, 3, 3, 8, 8, 8)
    flow_bwd = torch.zeros_like(flow)
    m = ed_es_label_propagation_metrics(
        seg,
        seg,
        flow,
        ed_index=0,
        es_index=2,
        flow_bwd=flow_bwd,
        num_classes=2,
        compute_hd95=False,
    )
    assert m["prop_ed2es_dice_mean"] > 0.99


def test_compose_flows_between_identity():
    # Closed T=4
    flow = torch.zeros(1, 3, 4, 4, 4, 4)
    c = compose_flows_between(flow, 0, 3)
    assert torch.allclose(c, torch.zeros_like(c))


def test_motion_metrics_batch_invariant():
    """P0-4: B=1,2,4 with the same patients → aggregate equal within 1e-6."""
    torch.manual_seed(0)
    # Four synthetic patients with distinct ED/ES.
    patients = []
    for i in range(4):
        seg_ed = torch.zeros(8, 8, 8, dtype=torch.long)
        seg_es = torch.zeros(8, 8, 8, dtype=torch.long)
        seg_ed[2:5, 2:5, 2:5] = 1
        seg_es[2:5, 2:5, 3:6] = 1
        ed_i, es_i = 0, 2 + (i % 2)  # 2 or 3
        t = 4
        flow = torch.zeros(1, 3, t, 8, 8, 8)
        flow_bwd = torch.zeros_like(flow)
        patients.append((seg_ed, seg_es, flow, flow_bwd, ed_i, es_i))

    def run_batch(indices: list[int]) -> dict[str, float]:
        seg_ed = torch.stack([patients[i][0] for i in indices], dim=0)
        seg_es = torch.stack([patients[i][1] for i in indices], dim=0)
        flow = torch.cat([patients[i][2] for i in indices], dim=0)
        flow_bwd = torch.cat([patients[i][3] for i in indices], dim=0)
        ed = torch.tensor([patients[i][4] for i in indices], dtype=torch.long)
        es = torch.tensor([patients[i][5] for i in indices], dtype=torch.long)
        return ed_es_label_propagation_metrics(
            seg_ed,
            seg_es,
            flow,
            ed,
            es,
            flow_bwd=flow_bwd,
            num_classes=2,
            compute_hd95=False,
        )

    # Same four patients aggregated as B=1 (mean of singles), B=2, B=4.
    singles = [run_batch([i])["prop_ed2es_dice_mean"] for i in range(4)]
    mean_singles = float(sum(singles) / len(singles))
    b2_a = run_batch([0, 1])["prop_ed2es_dice_mean"]
    b2_b = run_batch([2, 3])["prop_ed2es_dice_mean"]
    mean_b2 = 0.5 * (b2_a + b2_b)
    b4 = run_batch([0, 1, 2, 3])["prop_ed2es_dice_mean"]
    assert abs(mean_singles - mean_b2) < 1e-6
    assert abs(mean_singles - b4) < 1e-6
    assert abs(mean_b2 - b4) < 1e-6
