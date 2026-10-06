#!/usr/bin/env python
"""Train ReconSeg3D."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reconseg3d.training.trainer import Trainer
from reconseg3d.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ReconSeg3D")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    trainer = Trainer(cfg)
    history = trainer.fit()
    keys = ("loss_total", "mace_auc", "c_index", "phenotype_acc", "dice_mean", "recon_psnr")
    summary = {k: history.get(k) for k in keys if k in history}
    print(f"Training done. {summary}")


if __name__ == "__main__":
    main()
