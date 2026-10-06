"""Stage-2 metrics: SSIM3D, physical EF, patient aggregation."""

import numpy as np
import torch
import yaml

from reconseg3d.training.metrics import (
    aggregate_case_metrics,
    chamber_volume_ml,
    compute_metrics,
    physical_edv_esv_ef,
    ssim_3d,
    ssim_global,
    weighted_mean_metrics,
)


def test_ssim_3d_identical_is_one():
    x = torch.rand(1, 1, 12, 12, 12)
    assert ssim_3d(x, x.clone()) > 0.99


def test_ssim_3d_noise_lower_than_identical():
    x = torch.rand(2, 1, 4, 16, 16, 16)
    noisy = x + 0.5 * torch.randn_like(x)
    assert ssim_3d(x, x) > ssim_3d(x, noisy)


def test_ssim_3d_falls_back_on_tiny_grid():
    x = torch.rand(1, 1, 4, 4, 4)
    # window 7 > spatial → global proxy path
    v = ssim_3d(x, x, window_size=7)
    assert v > 0.99


def test_compute_metrics_includes_ssim_3d_not_recon_ssim():
    batch = {
        "volume": torch.randn(1, 1, 4, 12, 16, 16),
        "segmentation": torch.randint(0, 4, (1, 12, 16, 16)),
        "mace": torch.tensor([0.0]),
    }
    outputs = {
        "segmentation": torch.randn(1, 4, 12, 16, 16),
        "reconstruction": batch["volume"],
        "mace_logits": torch.tensor([0.0]),
    }
    m = compute_metrics(outputs, batch, num_classes=4, compute_hd95=False)
    assert "ssim_3d" in m
    assert m["ssim_3d"] > 0.9
    assert "recon_ssim_proxy" in m
    assert "recon_ssim" not in m


def test_physical_edv_esv_ef_ml():
    # 1000 LV voxels * 1mm^3 = 1 mL at ED; 500 at ES → EF 50%
    seg_ed = torch.zeros(1, 10, 10, 10, dtype=torch.long)
    seg_es = torch.zeros(1, 10, 10, 10, dtype=torch.long)
    seg_ed[0, :10, :10, :10] = 0
    # Fill exactly 1000 and 500 voxels with LV=1
    seg_ed.view(-1)[:1000] = 1
    seg_es.view(-1)[:500] = 1
    spacing = (1.0, 1.0, 1.0)
    out = physical_edv_esv_ef(seg_ed, seg_es, spacing, lv_index=1)
    assert abs(out["edv_ml"] - 1.0) < 1e-5
    assert abs(out["esv_ml"] - 0.5) < 1e-5
    assert abs(out["ef_percent"] - 50.0) < 1e-3


def test_no_ml_keys_without_spacing():
    seg_ed = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    seg_es = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    seg_ed.view(-1)[:100] = 1
    seg_es.view(-1)[:40] = 1
    out = physical_edv_esv_ef(seg_ed, seg_es, None, lv_index=1)
    assert "edv_ml" not in out and "esv_ml" not in out and "ef_percent" not in out
    assert "edv_vox" in out and "esv_vox" in out and "ef_proxy" in out
    assert abs(out["edv_vox"] - 100.0) < 1e-5
    assert abs(out["esv_vox"] - 40.0) < 1e-5


def test_compute_metrics_no_ml_without_spacing():
    b, t, d, h, w = 1, 4, 8, 8, 8
    seg_seq = torch.full((b, t, d, h, w), 0, dtype=torch.long)
    ed = torch.zeros(d, h, w, dtype=torch.long)
    es = torch.zeros(d, h, w, dtype=torch.long)
    ed.view(-1)[:200] = 1
    es.view(-1)[:100] = 1
    seg_seq[0, 0] = ed
    seg_seq[0, 2] = es
    logits = torch.nn.functional.one_hot(seg_seq, num_classes=4).permute(0, 5, 1, 2, 3, 4).float() * 10.0
    batch = {
        "volume": torch.randn(b, 1, t, d, h, w),
        "segmentation": ed.unsqueeze(0),
        "segmentation_sequence": seg_seq,
        "ed_index": torch.tensor([0]),
        "es_index": torch.tensor([2]),
        "mace": torch.tensor([0.0]),
    }
    outputs = {
        "segmentation": logits[:, :, 0],
        "reconstruction": batch["volume"],
        "seg_sequence": logits,
        "mace_logits": torch.tensor([0.0]),
    }
    m = compute_metrics(outputs, batch, num_classes=4, compute_hd95=False, spacing=None)
    assert "edv_ml" not in m and "esv_ml" not in m
    assert "edv_vox" in m and "esv_vox" in m


def test_chamber_volume_scales_with_spacing():
    mask = torch.ones(1, 4, 4, 4, dtype=torch.long)  # all LV if class 1 — use class 1 fill
    mask = torch.full((1, 4, 4, 4), 1, dtype=torch.long)
    v_unit = chamber_volume_ml(mask, (1.0, 1.0, 1.0), class_index=1)
    v_2 = chamber_volume_ml(mask, (2.0, 1.0, 1.0), class_index=1)
    assert float(v_2.item()) == 2.0 * float(v_unit.item())


def test_aggregate_case_metrics_bootstrap():
    rows = [{"dice_mean": 0.8, "patient_id": "a"}, {"dice_mean": 0.9, "patient_id": "b"}]
    summary, boot = aggregate_case_metrics(rows, n_boot=50, seed=0)
    assert summary["n_cases"] == 2.0
    assert abs(summary["dice_mean_mean"] - 0.85) < 1e-6
    assert "dice_mean" in boot
    assert boot["dice_mean"]["lo"] <= boot["dice_mean"]["mean"] <= boot["dice_mean"]["hi"]


def test_weighted_mean_not_naive_batch_mean():
    # Batch of 1 with dice 1.0 and batch of 3 with dice 0.0 → weighted 0.25
    batches = [{"dice_mean": 1.0}, {"dice_mean": 0.0}]
    sizes = [1, 3]
    out = weighted_mean_metrics(batches, sizes)
    assert abs(out["dice_mean"] - 0.25) < 1e-6


def test_compute_metrics_physical_ef_from_ed_es():
    b, t, d, h, w = 1, 4, 8, 8, 8
    seg_seq = torch.full((b, t, d, h, w), 0, dtype=torch.long)
    ed = torch.zeros(d, h, w, dtype=torch.long)
    es = torch.zeros(d, h, w, dtype=torch.long)
    ed.view(-1)[:200] = 1
    es.view(-1)[:100] = 1
    seg_seq[0, 0] = ed
    seg_seq[0, 2] = es
    logits = torch.nn.functional.one_hot(seg_seq, num_classes=4).permute(0, 5, 1, 2, 3, 4).float() * 10.0

    batch = {
        "volume": torch.randn(b, 1, t, d, h, w),
        "segmentation": ed.unsqueeze(0),
        "segmentation_sequence": seg_seq,
        "ed_index": torch.tensor([0]),
        "es_index": torch.tensor([2]),
        "spacing": torch.tensor([1.0, 1.0, 1.0]),
        "mace": torch.tensor([0.0]),
    }
    outputs = {
        "segmentation": logits[:, :, 0],
        "reconstruction": batch["volume"],
        "seg_sequence": logits,
        "mace_logits": torch.tensor([0.0]),
    }
    m = compute_metrics(outputs, batch, num_classes=4, compute_hd95=False)
    assert "edv_ml" in m and "esv_ml" in m and "ef_percent" in m
    assert "ef_proxy" in m  # labeled proxy still present
    assert abs(m["ef_percent"] - 50.0) < 1.0


def test_smoke_and_default_selection_honest():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel, expect_task, expect_metric in [
        ("configs/smoke_motion.yaml", "motion", "prop_ed2es_dice_mean"),
        ("configs/default.yaml", "joint", "loss_total"),
        ("configs/paper_recon.yaml", "reconstruction", "recon_mae"),
    ]:
        cfg = yaml.safe_load((root / rel).read_text(encoding="utf-8"))
        assert cfg["model"]["task"] == expect_task
        assert cfg["selection"]["metric"] == expect_metric
        assert cfg["model"]["task"] != "mace" or cfg["loss"].get("w_mace", 0) > 0
