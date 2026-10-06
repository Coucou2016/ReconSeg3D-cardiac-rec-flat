#!/usr/bin/env python
"""Write diagnosis-stratified ACDC 5-fold split JSON files under splits/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reconseg3d.data.splits import write_acdc_folds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None, help="ACDC root (default data/acdc)")
    p.add_argument("--out-dir", type=str, default=str(ROOT / "splits"))
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-fake", type=int, default=20)
    p.add_argument(
        "--allow-degenerate",
        action="store_true",
        help="Allow empty train/val/test (CI/smoke placeholders only)",
    )
    args = p.parse_args()
    paths = write_acdc_folds(
        args.root,
        out_dir=args.out_dir,
        n_folds=args.n_folds,
        seed=args.seed,
        n_fake=args.n_fake,
        allow_degenerate=bool(args.allow_degenerate),
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
