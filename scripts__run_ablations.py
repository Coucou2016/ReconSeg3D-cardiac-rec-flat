#!/usr/bin/env python
"""Run a small paper ablation matrix (smoke epochs by default)."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ABLATION_CONFIGS = [
    "configs/ablations/broadcast_baseline.yaml",
    "configs/ablations/per_frame_no_motion.yaml",
    "configs/ablations/per_frame_motion.yaml",
    "configs/ablations/fusion_concat.yaml",
    "configs/ablations/fusion_heart_ttable.yaml",
    "configs/ablations/task_phenotype.yaml",
]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ReconSeg3D ablation matrix")
    parser.add_argument("--epochs", type=int, default=1, help="Smoke default=1; use 10+ for longer")
    parser.add_argument("--output-root", type=str, default="outputs/ablations")
    parser.add_argument("--prepare-data", action="store_true", default=True)
    parser.add_argument("--no-prepare-data", action="store_false", dest="prepare_data")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--summarize", action="store_true", default=True)
    parser.add_argument("--no-summarize", action="store_false", dest="summarize")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Subset of config stems, e.g. broadcast_baseline per_frame_motion",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    py = sys.executable

    if args.prepare_data:
        _run([py, str(ROOT / "scripts" / "prepare_demo_data.py")])

    configs = ABLATION_CONFIGS
    if args.only:
        want = set(args.only)
        configs = [c for c in configs if Path(c).stem in want]
        if not configs:
            raise SystemExit(f"No configs matched --only {args.only}")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for rel in configs:
        cfg_path = ROOT / rel
        name = cfg_path.stem
        run_dir = out_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        _run(
            [
                py,
                str(ROOT / "scripts" / "train.py"),
                "--config",
                str(cfg_path),
                "--epochs",
                str(args.epochs),
                "--output-dir",
                str(run_dir),
            ]
        )
        ckpt = run_dir / "last.pt"
        train_metrics = run_dir / "metrics.json"
        train_metrics_bak = run_dir / "metrics_train.json"
        if train_metrics.is_file():
            train_metrics.replace(train_metrics_bak)
        if not args.skip_eval and ckpt.exists():
            _run(
                [
                    py,
                    str(ROOT / "scripts" / "eval.py"),
                    "--config",
                    str(cfg_path),
                    "--checkpoint",
                    str(ckpt),
                    "--metrics-out",
                    str(run_dir / "metrics.json"),
                ]
            )
            # Preserve train loss_* into final metrics.json (eval has no criterion).
            if train_metrics_bak.is_file() and train_metrics.is_file():
                try:
                    final = json.loads(train_metrics.read_text(encoding="utf-8"))
                    prior = json.loads(train_metrics_bak.read_text(encoding="utf-8"))
                    for k, v in prior.items():
                        if str(k).startswith("loss_") and k not in final:
                            final[k] = v
                    train_metrics.write_text(json.dumps(final, indent=2), encoding="utf-8")
                except (OSError, json.JSONDecodeError, TypeError):
                    pass
        elif train_metrics_bak.is_file() and not train_metrics.is_file():
            train_metrics_bak.replace(train_metrics)
        results.append({"name": name, "config": rel, "output_dir": str(run_dir)})

    manifest = out_root / "ablation_manifest.json"
    manifest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {manifest}")

    if args.summarize:
        _run(
            [
                py,
                str(ROOT / "scripts" / "summarize_runs.py"),
                "--runs-root",
                str(out_root),
                "--out-md",
                str(out_root / "table.md"),
                "--out-csv",
                str(out_root / "table.csv"),
            ]
        )


if __name__ == "__main__":
    main()
