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
    weighted_mean_metrics,
)
from reconseg3d.utils.config import resolve_device, save_config_snapshot
from reconseg3d.utils.seed import set_seed
from reconseg3d.utils.shapes import validate_batch

logger = logging.getLogger(__name__)

LOSS_KEYS = (
    "seg",
    "recon",
    "recon_mse",
    "mace",
    "warp",
    "cycle",
    "inv",
    "smooth",
    "jac",
    "loop",
    "volsmooth",
    "segsmooth",
    "seg2d",
    "cox",
    "phenotype",
)
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
            w_mace=loss_cfg.get("w_mace", 0.0),
            recon_loss=loss_cfg.get("recon_loss", "l1"),
            mace_loss=loss_cfg.get("mace_loss", "bce"),
            use_dice=loss_cfg.get("use_dice", True),
            alpha1=loss_cfg.get("alpha1", 0.0),
            alpha2=loss_cfg.get("alpha2", 0.0),
            alpha3=loss_cfg.get("alpha3", 0.0),
            w_warp=loss_cfg.get("w_warp", 0.0),
            w_cycle=loss_cfg.get("w_cycle", 0.0),
            w_inv=loss_cfg.get("w_inv", 0.0),
            w_smooth=loss_cfg.get("w_smooth", 0.0),
            w_jac=loss_cfg.get("w_jac", 0.0),
            jac_eps=float(loss_cfg.get("jac_eps", 0.0)),
            w_loop=loss_cfg.get("w_loop", 0.0),
            w_ed_ref=loss_cfg.get("w_ed_ref", 0.0),
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
        sel_cfg = cfg.get("selection", {}) or {}
        self.selection_metric, self.selection_mode = self._resolve_selection(sel_cfg, self.task)
        self.best_auc = float("-inf")
        self.best_selection_score = float("-inf") if self.selection_mode == "max" else float("inf")
        self.best_val_loss = float("inf")
        self._epochs_no_improve = 0
        save_config_snapshot(self.cfg, self.output_dir / "config_snapshot.yaml")

    @staticmethod
    def _resolve_selection(sel_cfg: dict[str, Any], task: str) -> tuple[str, str]:
        """Task-specific checkpoint metric. Higher-is-better unless mode=min."""
        defaults = {
            "motion": ("prop_ed2es_dice_mean", "max"),
            "segmentation": ("dice_mean", "max"),
            "reconstruction": ("recon_mae", "min"),
            "joint": ("loss_total", "min"),
            "phenotype": ("phenotype_acc", "max"),
            "cox": ("c_index", "max"),
            "mace": ("mace_auc", "max"),
        }
        metric = sel_cfg.get("metric")
        mode = sel_cfg.get("mode")
        if not metric:
            metric, default_mode = defaults.get(str(task), ("loss_total", "min"))
            mode = mode or default_mode
        mode = str(mode or "max").lower()
        if mode not in ("max", "min"):
            raise ValueError(f"selection.mode must be max|min, got {mode}")
        return str(metric), mode

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
        seg_frame_indices = self._optional(batch, "seg_frame_indices")

        if train:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.model.eval()

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            with autocast(self.amp_device, enabled=self.use_amp):
                out = self.model(volume, clinical, seg_frame_indices=seg_frame_indices)
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
                    segmentation_sequence=self._optional(batch, "segmentation_sequence"),
                    seg_valid_mask=self._optional(batch, "seg_valid_mask"),
                    slice_mask=self._optional(batch, "slice_mask"),
                    time=self._optional(batch, "time"),
                    event=self._optional(batch, "event"),
                    phenotype_logits=out.phenotype_logits,
                    target_phenotype=self._optional(batch, "phenotype"),
                    ed_index=self._optional(batch, "ed_index"),
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
            spacing=batch.get("spacing"),
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
        batch_metrics: list[dict[str, float]] = []
        batch_sizes: list[int] = []
        extras_list: list[dict[str, Any]] = []
        desc = "train" if train else "val"
        for batch in tqdm(loader, desc=desc, leave=False):
            m, extras = self._step(batch, train=train)
            # Sample-weighted aggregation (not naive batch-mean).
            bsz = int(batch["volume"].shape[0]) if "volume" in batch else 1
            filtered = {k: v for k, v in m.items() if k not in EPOCH_RANKING_KEYS and isinstance(v, float)}
            batch_metrics.append(filtered)
            batch_sizes.append(bsz)
            extras_list.append(extras)
        averaged = weighted_mean_metrics(batch_metrics, batch_sizes)
        averaged.pop("_n_samples", None)
        averaged.update(self._pool_ranking_metrics(extras_list))
        return averaged

    def _selection_score(self, val_m: dict[str, float]) -> tuple[float, str]:
        """Return (score, mode) for best.pt. Falls back to val loss when metric missing."""
        v = val_m.get(self.selection_metric, float("nan"))
        if isinstance(v, float) and v == v:
            return v, self.selection_mode
        # Fallbacks when primary metric absent (e.g. motion prop Dice before flows).
        if self.task == "motion":
            loss = val_m.get("loss_total", float("nan"))
            if isinstance(loss, float) and loss == loss:
                return loss, "min"
        if self.task == "phenotype":
            for key in ("phenotype_acc", "phenotype_auc"):
                pv = val_m.get(key, float("nan"))
                if isinstance(pv, float) and pv == pv:
                    return pv, "max"
        if self.task == "cox":
            cv = val_m.get("c_index", float("nan"))
            if isinstance(cv, float) and cv == cv:
                return cv, "max"
        loss = val_m.get("loss_total", float("nan"))
        if isinstance(loss, float) and loss == loss:
            return loss, "min"
        return float("nan"), self.selection_mode

    def _is_better(self, score: float, mode: str) -> bool:
        if not (isinstance(score, float) and score == score):
            return False
        if mode == "min":
            return score < self.best_selection_score
        return score > self.best_selection_score

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

            sel, sel_mode = self._selection_score(val_m)
            auc = val_m.get("mace_auc", float("nan"))
            if isinstance(auc, float) and auc == auc and auc > self.best_auc:
                self.best_auc = auc
            saved_best = False
            if self._is_better(sel, sel_mode):
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
