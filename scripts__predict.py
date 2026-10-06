#!/usr/bin/env python
"""Run inference on synthetic or manifest data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reconseg3d.data.dataset import build_dataloader
from reconseg3d.inference.predictor import Predictor
from reconseg3d.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict MACE / phenotype and segmentation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--out-dir", type=str, default="predictions")
    parser.add_argument("--max-batches", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    predictor = Predictor(args.checkpoint)
    loader = build_dataloader(cfg, "val")
    task = predictor.cfg.get("model", {}).get("task", cfg.get("model", {}).get("task", "mace"))

    out_root = Path(args.out_dir)
    for i, batch in enumerate(loader):
        if i >= args.max_batches:
            break
        result = predictor.predict_batch(batch)
        case_dir = out_root / f"batch_{i:04d}"
        predictor.export_numpy(result, case_dir)
        mace = result["mace_prob"].cpu().tolist()
        msg = f"batch {i}: mace_prob={mace}"
        if "phenotype_prob" in result:
            probs = result["phenotype_prob"].cpu().tolist()
            preds = result["phenotype_pred"].cpu().tolist()
            msg += f" phenotype_pred={preds} phenotype_prob={probs}"
        elif task == "phenotype":
            msg += " (no phenotype head in checkpoint)"
        print(msg)


if __name__ == "__main__":
    main()
