#!/usr/bin/env python
"""Collect run metrics.json files into markdown/CSV tables (PAPER_PLAN T1–T4 columns)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Columns aligned with docs/PAPER_PLAN.md tables T1–T4 where available.
TABLE_COLUMNS = [
    "run",
    "recon_psnr",
    "ssim_3d",
    "recon_ssim_proxy",
    "recon_mae",
    "dice_lv",
    "dice_rv",
    "dice_myo",
    "dice_mean",
    "prop_ed2es_dice_mean",
    "edv_ml",
    "esv_ml",
    "ef_percent",
    "ef_proxy",
    "warp_error",
    "cycle_error",
    "inv_error",
    "phenotype_acc",
    "phenotype_auc",
    "mace_auc",
    "c_index",
    "loss_total",
]


def _load_metrics(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path}")
    return data


def collect_runs(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics_files = sorted(runs_root.glob("**/metrics.json"))
    for path in metrics_files:
        metrics = _load_metrics(path)
        run_dir = path.parent
        run_name = metrics.get("run_name") or run_dir.name
        row: dict[str, Any] = {"run": run_name, "metrics_path": str(path)}
        for col in TABLE_COLUMNS:
            if col == "run":
                continue
            if col in metrics:
                row[col] = metrics[col]
            elif col.startswith("loss_") and col[5:] in metrics:
                row[col] = metrics[col[5:]]
        rows.append(row)
    return rows


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:  # NaN
            return "nan"
        return f"{v:.4f}"
    return str(v)


def to_markdown(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    cols = columns or TABLE_COLUMNS
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [
        "# Ablation summary (demo / public proxies)",
        "",
        "Not private AMI MACE. Empty cells mean the metric was not logged for that run.",
        "Demo / smoke numbers are pipeline checks only — do not cite as clinical performance.",
        "",
        header,
        sep,
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c)) for c in cols) + " |")
    lines.append("")
    return "\n".join(lines)


def to_csv(rows: list[dict[str, Any]], path: Path, columns: list[str] | None = None) -> None:
    cols = columns or TABLE_COLUMNS
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ReconSeg3D run metrics")
    parser.add_argument("--runs-root", type=str, default="outputs")
    parser.add_argument("--out-md", type=str, default=None)
    parser.add_argument("--out-csv", type=str, default=None)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    rows = collect_runs(runs_root)
    if not rows:
        print(f"No metrics.json under {runs_root}")
        return

    md = to_markdown(rows)
    print(md)
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md, encoding="utf-8")
        print(f"wrote {out_md}")
    if args.out_csv:
        to_csv(rows, Path(args.out_csv))
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
