"""Inference: MACE risk, phenotype, reconstruction, and segmentation export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from reconseg3d.models.reconseg3d import build_model
from reconseg3d.utils.config import resolve_device
from reconseg3d.utils.shapes import assert_volume_shape, validate_batch


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str = "auto") -> None:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = ckpt["cfg"]
        self.run_name = ckpt.get("run_name") or self.cfg.get("run_name")
        self.device = torch.device(
            resolve_device({"device": device}) if device != "auto" else resolve_device(self.cfg)
        )
        self.model = build_model(self.cfg).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        data_cfg = self.cfg.get("data", {})
        self.clinical_dim = int(data_cfg.get("clinical_dim", 0))
        self.in_channels = self.cfg.get("model", {}).get("in_channels", 1)
        self.task = self.cfg.get("model", {}).get("task", "mace")

    @torch.no_grad()
    def predict_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        validate_batch(
            batch,
            in_channels=self.in_channels,
            clinical_dim=self.clinical_dim,
        )
        volume = batch["volume"].to(self.device)
        clinical = None
        if self.clinical_dim > 0:
            clinical = batch["clinical"].to(self.device)
        seg_frame_indices = None
        if "seg_frame_indices" in batch:
            seg_frame_indices = batch["seg_frame_indices"].to(self.device)
        out = self.model(volume, clinical, seg_frame_indices=seg_frame_indices)
        result: dict[str, torch.Tensor] = {
            "mace_prob": torch.sigmoid(out.mace_logits),
            "mace_logits": out.mace_logits,
            "segmentation": out.segmentation.argmax(dim=1),
            "seg_logits": out.segmentation,
            "reconstruction": out.reconstruction,
        }
        if out.flow is not None:
            result["flow"] = out.flow
        if out.flow_bwd is not None:
            result["flow_bwd"] = out.flow_bwd
        if out.seg_sequence is not None:
            result["seg_sequence"] = out.seg_sequence
        if out.phenotype_logits is not None:
            result["phenotype_logits"] = out.phenotype_logits
            result["phenotype_prob"] = torch.softmax(out.phenotype_logits, dim=-1)
            result["phenotype_pred"] = out.phenotype_logits.argmax(dim=-1)
        return result

    @torch.no_grad()
    def predict_volume(
        self,
        volume: torch.Tensor,
        clinical: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if volume.ndim == 5:
            volume = volume.unsqueeze(0)
        assert_volume_shape(volume, in_channels=self.in_channels)
        batch: dict[str, torch.Tensor] = {
            "volume": volume,
            "segmentation": torch.zeros(1, *volume.shape[-3:]),
            "mace": torch.zeros(1),
        }
        if self.clinical_dim > 0:
            if clinical is None:
                raise ValueError(f"clinical_dim={self.clinical_dim} requires clinical features")
            batch["clinical"] = clinical.unsqueeze(0) if clinical.ndim == 1 else clinical
        return self.predict_batch(batch)

    def export_numpy(self, result: dict[str, torch.Tensor], out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "mace_prob.npy", result["mace_prob"].cpu().numpy())
        np.save(out_dir / "segmentation.npy", result["segmentation"].cpu().numpy())
        if "reconstruction" in result:
            np.save(out_dir / "reconstruction.npy", result["reconstruction"].cpu().numpy())
        if "flow" in result:
            np.save(out_dir / "flow.npy", result["flow"].cpu().numpy())
        if "flow_bwd" in result:
            np.save(out_dir / "flow_bwd.npy", result["flow_bwd"].cpu().numpy())
        if "phenotype_prob" in result:
            np.save(out_dir / "phenotype_prob.npy", result["phenotype_prob"].cpu().numpy())
        if "phenotype_pred" in result:
            np.save(out_dir / "phenotype_pred.npy", result["phenotype_pred"].cpu().numpy())
