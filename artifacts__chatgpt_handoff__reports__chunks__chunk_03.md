SOURCE PACK CHUNK 3 — continue reading next chunks before proposing patches.

## FILE: reconseg3d/training/trainer.py

```
"""Training and validation loops."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from reconseg3d.data.dataset import build_dataloader
from reconseg3d.models.losses import MultiTaskLoss
from reconseg3d.models.reconseg3d import ReconSeg3D, build_model, output_to_metric_dict
from reconseg3d.training.metrics import (
    _safe_auc,
    compute_metrics,
    concordance_index,
    mace_metrics,
)
from reconseg3d.utils.config import resolve_device, save_config_snapshot
from reconseg3d.utils.seed import set_seed
from reconseg3d.utils.shapes import validate_batch

logger = logging.getLogger(__name__)

LOSS_KEYS = ("seg", "recon", "recon_mse", "mace", "warp", "cycle", "volsmooth", "segsmooth", "seg2d", "cox", "phenotype")
# Ranking metrics must be pooled over the epoch — mean-of-batch-AUC is invalid.
EPOCH_RANKING_KEYS = ("mace_auc", "mace_sensitivity", "mace_specificity", "mace_accuracy", "c_index", "phenotype_acc", "phenotype_auc")


class Trainer:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        seed = cfg.get("seed", 42)
        set_seed(seed, deterministic=cfg.get("deterministic", True))

        self.device = torch.device(resolve_device(cfg))
        data_cfg = cfg.get("data", {})
        model_cfg = cfg.get("model", {})
        self.clinical_dim = int(data_cfg.get("clinical_dim", model_cfg.get("clinical_dim", 0)))
        if self.clinical_dim and not model_cfg.get("clinical_dim"):
            model_cfg = {**model_cfg, "clinical_dim": self.clinical_dim}
            cfg = {**cfg, "model": model_cfg}
            self.cfg = cfg

        task = model_cfg.get("task", "mace")
        if task == "phenotype" and int(model_cfg.get("num_phenotype_classes", 0) or 0) <= 0:
            model_cfg = {**model_cfg, "num_phenotype_classes": 5}
            cfg = {**cfg, "model": model_cfg}
            self.cfg = cfg

        self.model = build_model(cfg).to(self.device)
        if not isinstance(self.model, ReconSeg3D):
            raise TypeError("Trainer currently supports ReconSeg3D (set model.arch: reconseg3d)")
        loss_cfg = cfg.get("loss", {})
        task = model_cfg.get("task", loss_cfg.get("task", "mace"))
        self.task = str(task)
        self.criterion = MultiTaskLoss(
            num_seg_classes=model_cfg.get("num_seg_classes", 5),
            w_seg=loss_cfg.get("w_seg", 1.0),
            w_recon=loss_cfg.get("w_recon", 0.5),
            w_mace=loss_cfg.get("w_mace", 1.0),
            recon_loss=loss_cfg.get("recon_loss", "l1"),
            mace_loss=loss_cfg.get("mace_loss", "bce"),
            use_dice=loss_cfg.get("use_dice", True),
            alpha1=loss_cfg.get("alpha1", 0.0),
            alpha2=loss_cfg.get("alpha2", 0.0),
            alpha3=loss_cfg.get("alpha3", 0.0),
            w_warp=loss_cfg.get("w_warp", 0.0),
            w_cycle=loss_cfg.get("w_cycle", 0.0),
            w_volsmooth=loss_cfg.get("w_volsmooth", 0.0),
            w_segsmooth=loss_cfg.get("w_segsmooth", 0.0),
            w_cox=loss_cfg.get("w_cox", 0.0),
            w_phenotype=loss_cfg.get("w_phenotype", 0.0),
            task=task,
            num_phenotype_classes=model_cfg.get("num_phenotype_classes", 5),
        )
        train_cfg = cfg.get("train", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.get("lr", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )
        self.epochs = int(train_cfg.get("epochs", 5))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.use_amp = bool(train_cfg.get("amp", False)) and self.device.type == "cuda"
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self.amp_device = device_type
        self.scaler = GradScaler(device_type, enabled=self.use_amp)
        self.num_classes = model_cfg.get("num_seg_classes", 5)
        self.in_channels = model_cfg.get("in_channels", 1)
        self.compute_hd95 = bool(cfg.get("metrics", {}).get("hd95", False))
        self.early_stop_patience = int(train_cfg.get("early_stop_patience", 0) or 0)
        self.run_name = str(cfg.get("run_name") or Path(cfg.get("output_dir", "outputs")).name)
        self.output_dir = Path(cfg.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_auc = float("-inf")
        self.best_selection_score = float("-inf")
        self.best_val_loss = float("inf")
        self._epochs_no_improve = 0
        save_config_snapshot(self.cfg, self.output_dir / "config_snapshot.yaml")

    def _clinical(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        if self.clinical_dim <= 0:
            return None
        return batch["clinical"].to(self.device)

    def _optional(self, batch: dict[str, torch.Tensor], key: str) -> torch.Tensor | None:
        if key not in batch:
            return None
        return batch[key].to(self.device)

    def _step(
        self, batch: dict[str, torch.Tensor], train: bool
    ) -> tuple[dict[str, float], dict[str, Any]]:
        validate_batch(
            batch,
            in_channels=self.in_channels,
            clinical_dim=self.clinical_dim,
        )
        volume = batch["volume"].to(self.device)
        seg = batch["segmentation"].to(self.device)
        mace = batch["mace"].to(self.device)
        clinical = self._clinical(batch)
        target_vol = batch.get("volume_target", batch["volume"]).to(self.device)

        if train:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.model.eval()

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            with autocast(self.amp_device, enabled=self.use_amp):
                out = self.model(volume, clinical)
                losses = self.criterion(
                    out.reconstruction,
                    out.segmentation,
                    out.mace_logits,
                    target_vol,
                    seg,
                    mace,
                    flow=out.flow,
                    flow_bwd=out.flow_bwd,
                    seg_sequence=out.seg_sequence,
                    slice_mask=self._optional(batch, "slice_mask"),
                    time=self._optional(batch, "time"),
                    event=self._optional(batch, "event"),
                    phenotype_logits=out.phenotype_logits,
                    target_phenotype=self._optional(batch, "phenotype"),
                )
            if train:
                self.scaler.scale(losses["total"]).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

        metrics = compute_metrics(
            output_to_metric_dict(out),
            batch,
            num_classes=self.num_classes,
            compute_hd95=self.compute_hd95 and not train,
        )
        metrics["loss_total"] = losses["total"].item()
        for key in LOSS_KEYS:
            if key in losses:
                metrics[f"loss_{key}"] = float(losses[key])

        extras: dict[str, Any] = {
            "mace_logits": out.mace_logits.detach().float().cpu(),
            "mace": mace.detach().float().cpu(),
        }
        if "time" in batch and "event" in batch:
            extras["time"] = batch["time"].detach().float().cpu()
            extras["event"] = batch["event"].detach().float().cpu()
        if out.phenotype_logits is not None and "phenotype" in batch:
            extras["phenotype_logits"] = out.phenotype_logits.detach().float().cpu()
            extras["phenotype"] = batch["phenotype"].detach().long().cpu()
        return metrics, extras

    @staticmethod
    def _pool_ranking_metrics(extras_list: list[dict[str, Any]]) -> dict[str, float]:
        """Recompute AUC / C-index / phenotype metrics on pooled epoch predictions."""
        if not extras_list:
            return {}
        out: dict[str, float] = {}
        mace_logits = torch.cat([e["mace_logits"] for e in extras_list], dim=0)
        mace = torch.cat([e["mace"] for e in extras_list], dim=0)
        out.update(mace_metrics(mace_logits, mace))

        if all("time" in e and "event" in e for e in extras_list):
            out["c_index"] = concordance_index(
                mace_logits.numpy(),
                torch.cat([e["time"] for e in extras_list], dim=0).numpy(),
                torch.cat([e["event"] for e in extras_list], dim=0).numpy(),
            )

        if all("phenotype_logits" in e and "phenotype" in e for e in extras_list):
            logits_p = torch.cat([e["phenotype_logits"] for e in extras_list], dim=0)
            y_p = torch.cat([e["phenotype"] for e in extras_list], dim=0).numpy()
            pred_p = logits_p.argmax(dim=1).numpy()
            out["phenotype_acc"] = float((pred_p == y_p).mean()) if y_p.size else float("nan")
            if logits_p.shape[1] > 1:
                probs = torch.softmax(logits_p, dim=-1)[:, 1].numpy()
                out["phenotype_auc"] = _safe_auc((y_p == 1).astype("int32"), probs)
            else:
                out["phenotype_auc"] = float("nan")
        return out

    def _run_loader(self, loader: DataLoader, train: bool) -> dict[str, float]:
        sums: dict[str, float] = {}
        n = 0
        extras_list: list[dict[str, Any]] = []
        desc = "train" if train else "val"
        for batch in tqdm(loader, desc=desc, leave=False):
            m, extras = self._step(batch, train=train)
            for k, v in m.items():
                if k in EPOCH_RANKING_KEYS:
                    continue  # replaced by pooled epoch metrics below
                if isinstance(v, float) and v == v:
                    sums[k] = sums.get(k, 0.0) + v
            extras_list.append(extras)
            n += 1
        averaged = {k: v / max(n, 1) for k, v in sums.items()}
        averaged.update(self._pool_ranking_metrics(extras_list))
        return averaged

    def _selection_score(self, val_m: dict[str, float]) -> float:
        """Task-aware score for best.pt (higher is better)."""
        if self.task == "phenotype":
            for key in ("phenotype_acc", "phenotype_auc"):
                v = val_m.get(key, float("nan"))
                if isinstance(v, float) and v == v:
                    return v
        if self.task == "cox":
            v = val_m.get("c_index", float("nan"))
            if isinstance(v, float) and v == v:
                return v
        v = val_m.get("mace_auc", float("nan"))
        if isinstance(v, float) and v == v:
            return v
        return float("nan")

    def _write_metrics(self, metrics: dict[str, float], path: Path | None = None) -> Path:
        path = path or (self.output_dir / "metrics.json")
        payload = {
            "run_name": self.run_name,
            "task": self.task,
            **{k: (None if isinstance(v, float) and v != v else v) for k, v in metrics.items()},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def fit(self) -> dict[str, float]:
        train_loader = build_dataloader(self.cfg, "train")
        val_loader = build_dataloader(self.cfg, "val")
        history: dict[str, float] = {}

        for epoch in range(1, self.epochs + 1):
            train_m = self._run_loader(train_loader, train=True)
            val_m = self._run_loader(val_loader, train=False)
            logger.info(
                "epoch %d/%d run=%s train_loss=%.4f val_loss=%.4f val_auc=%s pheno_acc=%s",
                epoch,
                self.epochs,
                self.run_name,
                train_m.get("loss_total", 0),
                val_m.get("loss_total", 0),
                val_m.get("mace_auc"),
                val_m.get("phenotype_acc"),
            )
            for key in LOSS_KEYS:
                lk = f"loss_{key}"
                if lk in train_m:
                    logger.info("  train_%s=%.4f val_%s=%s", lk, train_m[lk], lk, val_m.get(lk))
            history = val_m

            sel = self._selection_score(val_m)
            auc = val_m.get("mace_auc", float("nan"))
            if isinstance(auc, float) and auc == auc and auc > self.best_auc:
                self.best_auc = auc
            saved_best = False
            if isinstance(sel, float) and sel == sel and sel > self.best_selection_score:
                self.best_selection_score = sel
                self.save_checkpoint("best.pt")
                saved_best = True

            val_loss = val_m.get("loss_total", float("nan"))
            if isinstance(val_loss, float) and val_loss == val_loss:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._epochs_no_improve = 0
                    if not saved_best and not (isinstance(sel, float) and sel == sel):
                        self.save_checkpoint("best.pt")
                else:
                    self._epochs_no_improve += 1
                if self.early_stop_patience > 0 and self._epochs_no_improve >= self.early_stop_patience:
                    logger.info(
                        "early stop at epoch %d (patience=%d, best_val_loss=%.4f)",
                        epoch,
                        self.early_stop_patience,
                        self.best_val_loss,
                    )
                    break

        self.save_checkpoint("last.pt")
        self._write_metrics(history)
        return history

    def save_checkpoint(self, name: str) -> Path:
        path = self.output_dir / name
        torch.save(
            {
                "model": self.model.state_dict(),
                "cfg": self.cfg,
                "run_name": self.run_name,
                "task": self.task,
                "best_auc": self.best_auc,
                "best_selection_score": self.best_selection_score,
                "best_val_loss": self.best_val_loss,
            },
            path,
        )
        return path

```

---
## FILE: reconseg3d/training/metrics.py

```
"""Evaluation metrics with safe edge-case handling."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from reconseg3d.models.motion import cycle_consistency_loss, volume_curve_loss, warp_consistency_loss

SEG_NAME = {1: "lv", 2: "rv", 3: "myo", 4: "scar"}
HD95_MAX_SURFACE = 4000


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC with guards for single-class or tiny batches."""
    y_true = np.asarray(y_true).astype(np.int32).ravel()
    y_score = np.asarray(y_score).astype(np.float64).ravel()
    if y_true.size == 0:
        return float("nan")
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.ndim != 3 or not mask.any():
        return np.zeros((0, 3), dtype=np.int32)
    pad = np.pad(mask, 1, mode="constant")
    inner = (
        pad[1:-1, 1:-1, 1:-1]
        & pad[:-2, 1:-1, 1:-1]
        & pad[2:, 1:-1, 1:-1]
        & pad[1:-1, :-2, 1:-1]
        & pad[1:-1, 2:, 1:-1]
        & pad[1:-1, 1:-1, :-2]
        & pad[1:-1, 1:-1, 2:]
    )
    surface = mask & ~inner
    if not surface.any():
        surface = mask
    return np.argwhere(surface)


def hd95_binary(pred: np.ndarray, target: np.ndarray) -> float:
    """95th-percentile Hausdorff distance (voxels). NaN if either mask is empty.

    Uses a numpy surface-to-surface implementation (no scipy required). Surfaces
    larger than ``HD95_MAX_SURFACE`` are subsampled so paper-scale volumes do not
    explode memory; document this if reporting HD95 on 256³ grids.
    """
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan")
    ps = _surface_voxels(pred)
    gs = _surface_voxels(target)
    if len(ps) == 0 or len(gs) == 0:
        return float("nan")
    rng = np.random.default_rng(0)
    if len(ps) > HD95_MAX_SURFACE:
        ps = ps[rng.choice(len(ps), HD95_MAX_SURFACE, replace=False)]
    if len(gs) > HD95_MAX_SURFACE:
        gs = gs[rng.choice(len(gs), HD95_MAX_SURFACE, replace=False)]
    delta = ps[:, None, :] - gs[None, :, :]
    dist = np.sqrt((delta.astype(np.float64) ** 2).sum(axis=-1))
    d_pg = dist.min(axis=1)
    d_gp = dist.min(axis=0)
    return float(np.percentile(np.concatenate([d_pg, d_gp]), 95))


def dice_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Mean Dice per class; pred/target (B,D,H,W) int."""
    scores: dict[str, float] = {}
    for cls in range(num_classes):
        p = (pred == cls).float()
        t = (target == cls).float()
        inter = (p * t).sum().item()
        union = p.sum().item() + t.sum().item()
        dice = (2 * inter + eps) / (union + eps) if union > 0 else 1.0
        scores[f"dice_class_{cls}"] = dice
        name = SEG_NAME.get(cls)
        if name is not None:
            scores[f"dice_{name}"] = dice
    valid = [v for k, v in scores.items() if k.startswith("dice_class_") and (k != "dice_class_0" or num_classes == 1)]
    scores["dice_mean"] = float(np.mean(valid)) if valid else 0.0
    return scores


def hd95_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict[str, float]:
    """Mean HD95 over LV/RV/MYO (classes 1–3 present in ``num_classes``)."""
    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()
    if pred_np.ndim == 3:
        pred_np = pred_np[None]
        tgt_np = tgt_np[None]
    out: dict[str, float] = {}
    acc: dict[str, list[float]] = {}
    for b in range(pred_np.shape[0]):
        for cls in range(1, min(num_classes, 4)):
            val = hd95_binary(pred_np[b] == cls, tgt_np[b] == cls)
            name = SEG_NAME.get(cls, str(cls))
            acc.setdefault(name, []).append(val)
    all_vals: list[float] = []
    for name, vals in acc.items():
        finite = [v for v in vals if v == v]
        out[f"hd95_{name}"] = float(np.mean(finite)) if finite else float("nan")
        all_vals.extend(finite)
    out["hd95_mean"] = float(np.mean(all_vals)) if all_vals else float("nan")
    return out


def iou_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> float:
    ious = []
    for cls in range(1, num_classes):
        p = pred == cls
        t = target == cls
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union > 0:
            ious.append((inter + eps) / (union + eps))
    return float(np.mean(ious)) if ious else 0.0


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float | None = None) -> float:
    pred = pred.float()
    target = target.float()
    mse = F.mse_loss(pred, target).item()
    if mse <= 0:
        return 99.0
    if data_range is None:
        data_range = float((target.max() - target.min()).item()) or 1.0
    return float(10.0 * math.log10((data_range ** 2) / mse))


def ssim_global(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Global (window-free) SSIM proxy. Not a 3D sliding-window SSIM.

    Stabilizers scale with ``data_range`` (Wang et al.): C1=(K1 L)^2, C2=(K2 L)^2.
    Fixed tiny C1/C2 without L made smoke recon SSIM collapse near 0 even for correlated volumes.
    """
    x = pred.float().reshape(-1)
    y = target.float().reshape(-1)
    if data_range is None:
        data_range = float((y.max() - y.min()).item()) or 1.0
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    mu_x = x.mean()
    mu_y = y.mean()
    var_x = x.var(unbiased=False)
    var_y = y.var(unbiased=False)
    cov = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    return float((num / den.clamp_min(1e-12)).item())


def ef_proxy_from_seg_sequence(seg_seq: torch.Tensor, lv_index: int = 1) -> float:
    """EF ≈ (max LV voxels − min LV voxels) / max over T. ``seg_seq`` (B,K,T,D,H,W) logits or (B,T,D,H,W) labels."""
    if seg_seq.ndim == 6:
        labels = seg_seq.argmax(dim=1)
    else:
        labels = seg_seq
    counts = (labels == lv_index).float().sum(dim=(2, 3, 4))
    edv = counts.max(dim=1).values
    esv = counts.min(dim=1).values
    ef = (edv - esv) / edv.clamp_min(1.0)
    return float(ef.mean().item())


def concordance_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell C-index. NaN if no comparable pairs."""
    risk = np.asarray(risk, dtype=np.float64).ravel()
    time = np.asarray(time, dtype=np.float64).ravel()
    event = np.asarray(event, dtype=np.float64).ravel()
    n = risk.size
    conc = 0.0
    total = 0.0
    for i in range(n):
        if event[i] <= 0:
            continue
        for j in range(n):
            if time[i] >= time[j]:
                continue
            total += 1.0
            if risk[i] > risk[j]:
                conc += 1.0
            elif risk[i] == risk[j]:
                conc += 0.5
    if total <= 0:
        return float("nan")
    return float(conc / total)


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 200,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap mean and (1-alpha) CI. Returns (mean, lo, hi). Tiny-n safe."""
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng if rng is not None else np.random.default_rng(0)
    n = values.size
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        means[b] = rng.choice(values, size=n, replace=True).mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(values.mean()), lo, hi


def mace_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    probs = torch.sigmoid(logits.detach()).cpu().numpy()
    y = labels.detach().cpu().numpy()
    pred_bin = (probs >= threshold).astype(np.int32)
    tp = int(((pred_bin == 1) & (y == 1)).sum())
    tn = int(((pred_bin == 0) & (y == 0)).sum())
    fp = int(((pred_bin == 1) & (y == 0)).sum())
    fn = int(((pred_bin == 0) & (y == 1)).sum())
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    acc = (tp + tn) / max(len(y), 1)
    return {
        "mace_auc": _safe_auc(y, probs),
        "mace_sensitivity": float(sens),
        "mace_specificity": float(spec),
        "mace_accuracy": float(acc),
    }


def compute_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    num_classes: int = 5,
    compute_hd95: bool = True,
) -> dict[str, float]:
    seg_pred = outputs["segmentation"].argmax(dim=1)
    seg_tgt = batch["segmentation"].long()
    if seg_tgt.device != seg_pred.device:
        seg_tgt = seg_tgt.to(seg_pred.device)
    metrics: dict[str, Any] = dice_per_class(seg_pred, seg_tgt, num_classes)
    metrics["iou_foreground"] = iou_per_class(seg_pred, seg_tgt, num_classes)
    if compute_hd95:
        try:
            metrics.update(hd95_per_class(seg_pred, seg_tgt, num_classes))
        except Exception:
            metrics["hd95_mean"] = float("nan")

    recon = outputs.get("reconstruction")
    vol = batch.get("volume_target", batch["volume"])
    if recon is not None:
        vol = vol.to(device=recon.device, dtype=recon.dtype)
        metrics["recon_mae"] = F.l1_loss(recon, vol, reduction="mean").item()
        metrics["recon_psnr"] = psnr(recon, vol)
        metrics["recon_ssim"] = ssim_global(recon, vol)

    if "mace_logits" in outputs and "mace" in batch:
        metrics.update(mace_metrics(outputs["mace_logits"], batch["mace"].to(outputs["mace_logits"].device)))

    if "mace_logits" in outputs and "time" in batch and "event" in batch:
        risk = outputs["mace_logits"].detach().cpu().numpy()
        metrics["c_index"] = concordance_index(
            risk,
            batch["time"].detach().cpu().numpy(),
            batch["event"].detach().cpu().numpy(),
        )

    if "phenotype_logits" in outputs and "phenotype" in batch:
        logits_p = outputs["phenotype_logits"]
        pred_p = logits_p.argmax(dim=1).detach().cpu().numpy()
        y_p = batch["phenotype"].detach().cpu().numpy()
        metrics["phenotype_acc"] = float((pred_p == y_p).mean()) if y_p.size else float("nan")
        # MINF (class 1) one-vs-rest AUC as public infarct-phenotype proxy (not 5y MACE).
        if logits_p.shape[1] > 1:
            probs = torch.softmax(logits_p.detach(), dim=-1)[:, 1].cpu().numpy()
            y_minf = (y_p == 1).astype(np.int32)
            metrics["phenotype_auc"] = _safe_auc(y_minf, probs)
        else:
            metrics["phenotype_auc"] = float("nan")

    seg_seq = outputs.get("seg_sequence")
    if seg_seq is not None:
        metrics["ef_proxy"] = ef_proxy_from_seg_sequence(seg_seq)
        try:
            metrics["vol_curve"] = float(volume_curve_loss(seg_seq).item())
        except Exception:
            metrics["vol_curve"] = float("nan")

    flow = outputs.get("flow")
    flow_bwd = outputs.get("flow_bwd")
    if recon is not None and flow is not None and recon.shape[2] > 1:
        try:
            metrics["warp_error"] = float(warp_consistency_loss(recon, flow, flow_bwd).item())
        except Exception:
            metrics["warp_error"] = float("nan")
        if flow_bwd is not None:
            try:
                metrics["cycle_error"] = float(cycle_consistency_loss(recon, flow, flow_bwd).item())
            except Exception:
                metrics["cycle_error"] = float("nan")

    for k, v in list(metrics.items()):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            metrics[k] = float("nan")
    return metrics

```

---
