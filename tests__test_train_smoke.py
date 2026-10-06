"""Short synthetic train with motion losses stays finite."""

import math
from pathlib import Path

from reconseg3d.training.trainer import Trainer
from reconseg3d.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_train_one_epoch_motion_no_nan():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg["train"]["epochs"] = 1
    cfg["data"]["train_samples"] = 4
    cfg["data"]["val_samples"] = 2
    cfg["data"]["num_frames"] = 4
    cfg["data"]["spatial_size"] = [8, 16, 16]
    cfg["data"]["batch_size"] = 2
    cfg["model"]["base_channels"] = 8
    cfg["metrics"] = {"hd95": False}
    trainer = Trainer(cfg)
    history = trainer.fit()
    assert "loss_total" in history
    assert math.isfinite(history["loss_total"])
    assert math.isfinite(history.get("loss_warp", 0.0))
    assert math.isfinite(history.get("loss_cycle", 0.0))
