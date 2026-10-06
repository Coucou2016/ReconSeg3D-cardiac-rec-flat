"""Metrics edge-case tests."""

import numpy as np
import torch

from reconseg3d.training.metrics import _safe_auc, compute_metrics, mace_metrics


def test_auc_single_class_returns_nan():
    y = np.array([0, 0, 0])
    s = np.array([0.1, 0.2, 0.3])
    assert np.isnan(_safe_auc(y, s))


def test_auc_binary():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    auc = _safe_auc(y, s)
    assert auc == 1.0


def test_mace_metrics():
    logits = torch.tensor([2.0, -2.0])
    labels = torch.tensor([1.0, 0.0])
    m = mace_metrics(logits, labels)
    assert m["mace_accuracy"] == 1.0
    assert m["mace_auc"] == 1.0


def test_compute_metrics_smoke():
    batch = {
        "volume": torch.randn(2, 1, 4, 8, 16, 16),
        "segmentation": torch.randint(0, 5, (2, 8, 16, 16)),
        "mace": torch.tensor([0.0, 1.0]),
    }
    outputs = {
        "segmentation": torch.randn(2, 5, 8, 16, 16),
        "reconstruction": batch["volume"],
        "mace_logits": torch.tensor([0.5, -0.5]),
    }
    m = compute_metrics(outputs, batch, num_classes=5)
    assert "dice_mean" in m
    assert "mace_auc" in m
    # Perfect recon should score high on global SSIM proxy (data_range-aware).
    assert m["recon_ssim_proxy"] > 0.9
    assert "recon_ssim" not in m


def test_pool_ranking_metrics_epoch_level():
    """Mean-of-batch AUC is invalid; Trainer pools logits then scores once."""
    from reconseg3d.training.trainer import Trainer

    extras = [
        {"mace_logits": torch.tensor([2.0, -2.0]), "mace": torch.tensor([1.0, 0.0])},
        {"mace_logits": torch.tensor([-1.0, 1.5]), "mace": torch.tensor([0.0, 1.0])},
    ]
    pooled = Trainer._pool_ranking_metrics(extras)
    assert pooled["mace_auc"] == 1.0
    assert pooled["mace_accuracy"] == 1.0


def test_checkpoint_selection_motion_vs_segmentation():
    """P0-5: motion selects prop Dice (max); segmentation selects dice_mean (max)."""
    from reconseg3d.training.trainer import Trainer

    motion_metric, motion_mode = Trainer._resolve_selection({}, "motion")
    assert motion_metric == "prop_ed2es_dice_mean"
    assert motion_mode == "max"
    seg_metric, seg_mode = Trainer._resolve_selection({}, "segmentation")
    assert seg_metric == "dice_mean"
    assert seg_mode == "max"
    recon_metric, recon_mode = Trainer._resolve_selection({}, "reconstruction")
    assert recon_metric == "recon_mae"
    assert recon_mode == "min"

    # Override via config.
    m, mode = Trainer._resolve_selection({"metric": "loss_total", "mode": "min"}, "motion")
    assert m == "loss_total" and mode == "min"

    class _T:
        selection_metric = "prop_ed2es_dice_mean"
        selection_mode = "max"
        task = "motion"
        best_selection_score = float("-inf")
        _selection_score = Trainer._selection_score
        _is_better = Trainer._is_better

    t = _T()
    score, mode = Trainer._selection_score(t, {"prop_ed2es_dice_mean": 0.8, "loss_total": 1.0})
    assert score == 0.8 and mode == "max"
    assert Trainer._is_better(t, score, mode)
    t.best_selection_score = 0.8
    # Motion fallback when prop metric missing.
    t.selection_metric = "prop_ed2es_dice_mean"
    score2, mode2 = Trainer._selection_score(t, {"loss_total": 0.5})
    assert mode2 == "min" and score2 == 0.5
