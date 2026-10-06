#!/usr/bin/env python
"""Evaluate a checkpoint with patient-level metrics + bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reconseg3d.data.dataset import build_dataloader
from reconseg3d.inference.predictor import Predictor
from reconseg3d.models.reconseg3d import output_to_metric_dict
from reconseg3d.training.metrics import (
    aggregate_case_metrics,
    compute_metrics,
    weighted_mean_metrics,
)
from reconseg3d.training.trainer import EPOCH_RANKING_KEYS, Trainer
from reconseg3d.utils.config import load_config


def _json_safe(obj: dict) -> dict:
    out = {}
    for k, v in obj.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
        elif isinstance(v, dict):
            out[k] = _json_safe(v)
        else:
            out[k] = v
    return out


def _merge_prior_losses(out_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Keep train-time loss_* fields when eval overwrites metrics.json."""
    if not out_path.is_file():
        return payload
    try:
        prior = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return payload
    if not isinstance(prior, dict):
        return payload
    merged = dict(payload)
    for k, v in prior.items():
        if k.startswith("loss_") and k not in merged:
            merged[k] = v
    return merged


def _per_sample_metrics(
    metrics_sample: dict[str, float],
    batch: dict[str, Any],
    batch_idx: int,
) -> dict[str, Any]:
    """Attach identity fields to a single-sample metrics dict."""
    row: dict[str, Any] = dict(metrics_sample)
    pids = batch.get("patient_id")
    if isinstance(pids, (list, tuple)) and batch_idx < len(pids):
        row["patient_id"] = pids[batch_idx]
    elif isinstance(pids, str) and batch_idx == 0:
        row["patient_id"] = pids
    if "case_id" in batch:
        cid = batch["case_id"]
        if torch.is_tensor(cid):
            flat = cid.reshape(-1)
            row["case_id"] = int(flat[batch_idx].item()) if flat.numel() > batch_idx else int(flat[0].item())
        elif isinstance(cid, (list, tuple)) and batch_idx < len(cid):
            row["case_id"] = cid[batch_idx]
        elif batch_idx == 0:
            row["case_id"] = cid
    return row


def _slice_tensor_or_list(value: Any, idx: int, batch_size: int) -> Any:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value
        if value.shape[0] == batch_size:
            return value[idx : idx + 1]
        return value
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        item = value[idx]
        return [item] if isinstance(value, list) else (item,)
    return value


def _slice_batch(batch: dict[str, Any], idx: int, batch_size: int) -> dict[str, Any]:
    return {k: _slice_tensor_or_list(v, idx, batch_size) for k, v in batch.items()}


def _slice_outputs(outputs: dict[str, Any], idx: int, batch_size: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in outputs.items():
        out[k] = _slice_tensor_or_list(v, idx, batch_size)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ReconSeg3D (patient-level)")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Loader split. With ACDC fold JSON (data.fold / fold_file), test is real held-out.",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default=None,
        help="Write averaged metrics JSON (default: <checkpoint_dir>/metrics.json)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Write case_metrics.csv / summary_metrics.json / bootstrap_ci.json (default: <ckpt_dir>/results)",
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=None, help="Override eval batch size (prefer 1 for per-case CSV)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    if args.batch_size is not None:
        cfg.setdefault("data", {})["batch_size"] = int(args.batch_size)
    else:
        # Default to 1 so case_metrics.csv is truly per-patient.
        cfg.setdefault("data", {})["batch_size"] = 1

    predictor = Predictor(args.checkpoint)
    split = args.split
    if split == "test":
        fold = cfg.get("data", {}).get("fold")
        fold_file = cfg.get("data", {}).get("fold_file")
        if fold is None and not fold_file and cfg.get("data", {}).get("source") == "acdc":
            logging.info(
                "split=test without fold_file: ACDCDataset uses held-out test patients "
                "(prefer data.fold=0..4 or fold_file=splits/acdc_foldK.json)."
            )
    loader = build_dataloader(cfg, split)

    batch_metrics: list[dict[str, float]] = []
    batch_sizes: list[int] = []
    case_rows: list[dict[str, Any]] = []
    extras_list: list[dict[str, Any]] = []
    n = 0
    for batch in loader:
        volume = batch["volume"].to(predictor.device)
        clinical = batch["clinical"].to(predictor.device) if predictor.clinical_dim > 0 else None
        seg_frame_indices = batch.get("seg_frame_indices")
        if seg_frame_indices is not None:
            seg_frame_indices = seg_frame_indices.to(predictor.device)
        ed_index = batch.get("ed_index")
        es_index = batch.get("es_index")
        # Ensure phase indices reach the model / metrics path.
        out = predictor.model(volume, clinical, seg_frame_indices=seg_frame_indices)
        spacing = batch.get("spacing")
        metrics = compute_metrics(
            output_to_metric_dict(out),
            batch,
            num_classes=cfg.get("model", {}).get("num_seg_classes", 5),
            compute_hd95=bool(cfg.get("metrics", {}).get("hd95", False)),
            spacing=spacing,
        )
        # Record that ED/ES were present for auditing.
        if ed_index is not None:
            metrics["ed_index_mean"] = float(ed_index.float().mean().item())
        if es_index is not None:
            metrics["es_index_mean"] = float(es_index.float().mean().item())

        bsz = int(volume.shape[0])
        filtered = {
            k: v
            for k, v in metrics.items()
            if k not in EPOCH_RANKING_KEYS and isinstance(v, float)
        }
        batch_metrics.append(filtered)
        batch_sizes.append(bsz)
        # True per-patient rows: recompute metrics on each sample (never duplicate batch-mean).
        metric_dict = output_to_metric_dict(out)
        for bi in range(bsz):
            if bsz == 1:
                sample_metrics = filtered
            else:
                sample_batch = _slice_batch(batch, bi, bsz)
                sample_out = _slice_outputs(metric_dict, bi, bsz)
                sample_spacing = sample_batch.get("spacing", spacing)
                sample_raw = compute_metrics(
                    sample_out,
                    sample_batch,
                    num_classes=cfg.get("model", {}).get("num_seg_classes", 5),
                    compute_hd95=bool(cfg.get("metrics", {}).get("hd95", False)),
                    spacing=sample_spacing,
                )
                sample_metrics = {
                    k: v
                    for k, v in sample_raw.items()
                    if k not in EPOCH_RANKING_KEYS and isinstance(v, float)
                }
            case_rows.append(_per_sample_metrics(sample_metrics, batch, bi))

        extras: dict[str, Any] = {
            "mace_logits": out.mace_logits.detach().float().cpu(),
            "mace": batch["mace"].detach().float().cpu(),
        }
        if "time" in batch and "event" in batch:
            extras["time"] = batch["time"].detach().float().cpu()
            extras["event"] = batch["event"].detach().float().cpu()
        if out.phenotype_logits is not None and "phenotype" in batch:
            extras["phenotype_logits"] = out.phenotype_logits.detach().float().cpu()
            extras["phenotype"] = batch["phenotype"].detach().long().cpu()
        extras_list.append(extras)
        n += 1

    avg = weighted_mean_metrics(batch_metrics, batch_sizes)
    avg.pop("_n_samples", None)
    avg.update(Trainer._pool_ranking_metrics(extras_list))
    summary, boot = aggregate_case_metrics(case_rows, n_boot=args.n_boot)

    ckpt_path = Path(args.checkpoint)
    run_name = predictor.cfg.get("run_name") or ckpt_path.parent.name
    results_dir = Path(args.results_dir) if args.results_dir else ckpt_path.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # case_metrics.csv
    csv_path = results_dir / "case_metrics.csv"
    if case_rows:
        keys = sorted({k for r in case_rows for k in r.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in case_rows:
                writer.writerow({k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()})

    summary_path = results_dir / "summary_metrics.json"
    summary_payload = {
        "run_name": run_name,
        "split": args.split,
        "n_batches": n,
        "n_cases": len(case_rows),
        "note": "Patient-level mean/SD. Real clinical tables 待补充 without licensed data.",
        **_json_safe(summary),
        **{f"epoch_{k}": v for k, v in _json_safe(avg).items()},
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    boot_path = results_dir / "bootstrap_ci.json"
    boot_path.write_text(json.dumps(_json_safe(boot), indent=2), encoding="utf-8")

    payload = {
        "run_name": run_name,
        "split": args.split,
        "n_batches": n,
        "n_cases": len(case_rows),
        "results_dir": str(results_dir),
        **_json_safe(avg),
    }
    print(json.dumps(payload, indent=2))

    out_path = Path(args.metrics_out) if args.metrics_out else ckpt_path.parent / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _merge_prior_losses(out_path, payload)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info("wrote %s", out_path)
    logging.info("wrote %s / %s / %s", csv_path, summary_path, boot_path)


if __name__ == "__main__":
    main()
