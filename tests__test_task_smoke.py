"""Phenotype and Cox tasks stay finite for one synthetic epoch."""

import math
from pathlib import Path

from reconseg3d.training.trainer import Trainer
from reconseg3d.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _tiny(cfg: dict) -> dict:
    cfg["train"]["epochs"] = 1
    cfg["data"]["source"] = "synthetic"
    cfg["data"]["train_samples"] = 4
    cfg["data"]["val_samples"] = 2
    cfg["data"]["num_frames"] = 4
    cfg["data"]["spatial_size"] = [8, 16, 16]
    cfg["data"]["batch_size"] = 2
    cfg["model"]["base_channels"] = 8
    cfg["metrics"] = {"hd95": False}
    cfg["train"]["early_stop_patience"] = 0
    return cfg


def test_train_phenotype_no_nan(tmp_path: Path):
    cfg = _tiny(load_config(ROOT / "configs" / "default.yaml"))
    cfg["output_dir"] = str(tmp_path / "pheno")
    cfg["run_name"] = "pheno_smoke"
    cfg["model"]["task"] = "phenotype"
    cfg["model"]["num_phenotype_classes"] = 5
    cfg["loss"]["w_mace"] = 0.0
    cfg["loss"]["w_phenotype"] = 1.0
    history = Trainer(cfg).fit()
    assert math.isfinite(history["loss_total"])
    assert math.isfinite(history.get("loss_phenotype", 0.0))
    assert (tmp_path / "pheno" / "metrics.json").exists()
    assert (tmp_path / "pheno" / "config_snapshot.yaml").exists()


def test_train_cox_no_nan(tmp_path: Path):
    cfg = _tiny(load_config(ROOT / "configs" / "default.yaml"))
    cfg["output_dir"] = str(tmp_path / "cox")
    cfg["run_name"] = "cox_smoke"
    cfg["model"]["task"] = "cox"
    cfg["loss"]["w_mace"] = 1.0
    cfg["loss"]["w_cox"] = 0.0
    history = Trainer(cfg).fit()
    assert math.isfinite(history["loss_total"])
    assert math.isfinite(history.get("loss_cox", history.get("loss_mace", 0.0)))
