#!/usr/bin/env python
"""Multi-seed training harness (3–5 seeds). Does not invent result numbers."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the same config across multiple seeds")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-root", type=str, default="outputs/multiseed")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        run_dir = out_root / f"seed_{seed}"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--config",
            args.config,
            "--output-dir",
            str(run_dir),
            "--seed",
            str(seed),
        ]
        if args.epochs is not None:
            cmd.extend(["--epochs", str(args.epochs)])
        logging.info("seed=%s cmd=%s", seed, " ".join(cmd))
        runs.append({"seed": seed, "output_dir": str(run_dir), "cmd": cmd})
        if args.dry_run:
            continue
        subprocess.check_call(cmd, cwd=str(ROOT))

    manifest = {
        "config": args.config,
        "seeds": list(args.seeds),
        "runs": runs,
        "note": (
            "Aggregate mean±SD across seeds after eval. "
            "Do not invent clinical metrics; mark 待补充 without licensed data."
        ),
        "how_to_eval": (
            "for each seed: python scripts/eval.py --config <cfg> "
            "--checkpoint outputs/multiseed/seed_<S>/best.pt --split test"
        ),
    }
    path = out_root / "multiseed_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info("wrote %s", path)


if __name__ == "__main__":
    main()
