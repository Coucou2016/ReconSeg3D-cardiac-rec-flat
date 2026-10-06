"""ReconSeg3D: joint reconstruction, segmentation, and risk prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from reconseg3d.models.blocks import Decoder3D, TemporalEncoder
from reconseg3d.models.heart_ttable import HeartTTable
from reconseg3d.models.motion import MotionNet
from reconseg3d.utils.shapes import assert_volume_shape


@dataclass
class ReconSeg3DOutput:
    reconstruction: torch.Tensor
    segmentation: torch.Tensor
    mace_logits: torch.Tensor
    features: torch.Tensor
    flow: torch.Tensor | None = None
    flow_bwd: torch.Tensor | None = None
    seg_sequence: torch.Tensor | None = None
    phenotype_logits: torch.Tensor | None = None


def output_to_metric_dict(out: ReconSeg3DOutput) -> dict[str, torch.Tensor]:
    """Pack model outputs for ``compute_metrics`` / eval."""
    packed: dict[str, torch.Tensor] = {
        "segmentation": out.segmentation,
        "reconstruction": out.reconstruction,
        "mace_logits": out.mace_logits,
    }
    if out.seg_sequence is not None:
        packed["seg_sequence"] = out.seg_sequence
    if out.flow is not None:
        packed["flow"] = out.flow
    if out.flow_bwd is not None:
        packed["flow_bwd"] = out.flow_bwd
    if out.phenotype_logits is not None:
        packed["phenotype_logits"] = out.phenotype_logits
    return packed


class ReconSeg3D(nn.Module):
    """
    3D spatiotemporal cardiac model for AMI.

    Input shape: (B, C, T, D, H, W) — batch, channels, time, depth, height, width.
    Outputs:
        - reconstruction: (B, C, T, D, H, W) — per-frame decode by default
        - segmentation: (B, num_classes, D, H, W) — per-voxel labels at reference
        - mace_logits: (B,) — binary MACE / Cox log-risk
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_seg_classes: int = 5,
        base_channels: int = 16,
        temporal_mode: str = "temporal_conv",
        clinical_dim: int = 0,
        dropout: float = 0.2,
        per_frame_recon: bool = True,
        predict_motion: bool = True,
        per_frame_seg: bool = True,
        fusion: str = "concat",
        task: str = "mace",
        num_phenotype_classes: int = 0,
        ttable_embed_dim: int = 32,
        ttable_patch_size: int = 4,
        ttable_on_recon: bool = False,
        use_svf: bool = False,
        svf_steps: int = 7,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_seg_classes = num_seg_classes
        self.clinical_dim = clinical_dim
        self.per_frame_recon = per_frame_recon
        self.predict_motion = predict_motion and per_frame_recon
        self.per_frame_seg = per_frame_seg and per_frame_recon
        self.fusion = fusion
        self.task = task
        self.num_phenotype_classes = num_phenotype_classes
        self.ttable_on_recon = ttable_on_recon
        self.use_svf = use_svf

        self.encoder = TemporalEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            temporal_mode=temporal_mode,
        )
        feat_ch = self.encoder.out_channels

        self.seg_head = nn.Sequential(
            nn.Conv3d(feat_ch, feat_ch // 2, 3, padding=1),
            nn.BatchNorm3d(feat_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(feat_ch // 2, num_seg_classes, 1),
        )

        self.recon_decoder = Decoder3D(feat_ch, out_channels=in_channels)
        self.motion_net = MotionNet(
            in_channels=in_channels,
            base_channels=max(base_channels // 2, 4),
            use_svf=use_svf,
            svf_steps=svf_steps,
        )

        if fusion == "heart_ttable":
            self.heart_ttable = HeartTTable(
                in_channels=in_channels,
                clinical_dim=clinical_dim,
                embed_dim=ttable_embed_dim,
                patch_size=ttable_patch_size,
                dropout=dropout,
            )
            cls_in = ttable_embed_dim
        else:
            self.heart_ttable = None
            cls_in = feat_ch + clinical_dim

        hidden = max(cls_in // 2, 8)
        self.mace_pool = nn.AdaptiveAvgPool3d(1)
        self.mace_fc = nn.Sequential(
            nn.Linear(cls_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        if num_phenotype_classes > 0:
            self.pheno_fc = nn.Sequential(
                nn.Linear(cls_in, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_phenotype_classes),
            )
        else:
            self.pheno_fc = None

    def _interp_3d(self, x: torch.Tensor, spatial: tuple[int, int, int]) -> torch.Tensor:
        if x.shape[-3:] != spatial:
            return F.interpolate(x, size=spatial, mode="trilinear", align_corners=False)
        return x

    def _decode_sequence(self, seq: torch.Tensor, spatial: tuple[int, int, int]) -> torch.Tensor:
        b, fc, t, d2, h2, w2 = seq.shape
        flat = seq.permute(0, 2, 1, 3, 4, 5).reshape(b * t, fc, d2, h2, w2)
        recon = self._interp_3d(self.recon_decoder(flat), spatial)
        c_out = recon.shape[1]
        d, h, w = spatial
        return recon.view(b, t, c_out, d, h, w).permute(0, 2, 1, 3, 4, 5).contiguous()

    def _seg_sequence(self, seq: torch.Tensor, spatial: tuple[int, int, int]) -> torch.Tensor:
        b, fc, t, d2, h2, w2 = seq.shape
        flat = seq.permute(0, 2, 1, 3, 4, 5).reshape(b * t, fc, d2, h2, w2)
        logits = self._interp_3d(self.seg_head(flat), spatial)
        k = logits.shape[1]
        d, h, w = spatial
        return logits.view(b, t, k, d, h, w).permute(0, 2, 1, 3, 4, 5).contiguous()

    def _risk_branch(
        self,
        x: torch.Tensor,
        features: torch.Tensor,
        reconstruction: torch.Tensor,
        clinical: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (feat_vec, mace_logits).

        When ``fusion=heart_ttable``, use HeartTTable's own risk head (avoids an
        unused ``risk_head`` plus a second MLP on the same features).
        """
        if self.heart_ttable is not None:
            vol = reconstruction if self.ttable_on_recon else x
            ht = self.heart_ttable(vol, clinical)
            return ht.features, ht.risk_logits
        pooled = self.mace_pool(features).flatten(1)
        if clinical is not None and self.clinical_dim > 0:
            pooled = torch.cat([pooled, clinical], dim=1)
        return pooled, self.mace_fc(pooled).squeeze(-1)

    @staticmethod
    def _resolve_seg_ref_indices(
        batch_size: int,
        num_frames: int,
        seg_frame_indices: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Per-sample reference frame for the primary seg head (ED when known).

        Does **not** silently use ``t // 2`` when indices are provided. If
        ``seg_frame_indices`` is None, falls back to frame 0 (ED-first convention)
        rather than mid-cycle, and logs the fallback via the returned indices.
        """
        if seg_frame_indices is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        idx = seg_frame_indices
        if idx.ndim == 0:
            idx = idx.view(1).expand(batch_size)
        elif idx.ndim == 2:
            # (B, K) labeled frame list → take first (typically ED)
            idx = idx[:, 0]
        idx = idx.long().to(device)
        if idx.shape[0] != batch_size:
            raise ValueError(f"seg_frame_indices batch {idx.shape[0]} != {batch_size}")
        return idx.clamp(0, max(num_frames - 1, 0))

    def forward(
        self,
        x: torch.Tensor,
        clinical: torch.Tensor | None = None,
        seg_frame_indices: torch.Tensor | None = None,
    ) -> ReconSeg3DOutput:
        """
        Args:
            x: (B, C, T, D, H, W)
            clinical: optional (B, clinical_dim) tabular features
            seg_frame_indices: optional (B,) or (B, K) labeled phase indices;
                primary segmentation logits are taken at the first index (ED).
        """
        assert_volume_shape(x, in_channels=self.in_channels)
        b, c, t, d, h, w = x.shape
        if self.clinical_dim > 0:
            if clinical is None:
                raise ValueError(f"clinical_dim={self.clinical_dim} but clinical is None")
            if clinical.shape != (b, self.clinical_dim):
                raise ValueError(f"clinical shape must be ({b}, {self.clinical_dim}), got {tuple(clinical.shape)}")

        spatial = (d, h, w)
        flow = None
        flow_bwd = None
        seg_sequence = None

        if self.per_frame_recon:
            seq, features = self.encoder(x, return_sequence=True)
            reconstruction = self._decode_sequence(seq, spatial)
            if self.per_frame_seg:
                seg_sequence = self._seg_sequence(seq, spatial)
                ref_idx = self._resolve_seg_ref_indices(b, t, seg_frame_indices, x.device)
                # Per-sample reference frame from (B, K, T, D, H, W) → (B, K, D, H, W)
                batch_ix = torch.arange(b, device=x.device)
                seg_logits = seg_sequence[batch_ix, :, ref_idx]
            else:
                seg_logits = self._interp_3d(self.seg_head(features), spatial)
            if self.predict_motion:
                flow, flow_bwd = self.motion_net(reconstruction)
        else:
            features = self.encoder(x)
            seg_logits = self._interp_3d(self.seg_head(features), spatial)
            recon_low = self._interp_3d(self.recon_decoder(features), spatial)
            reconstruction = recon_low.unsqueeze(2).expand(b, c, t, d, h, w).contiguous()

        feat_vec, mace_logits = self._risk_branch(x, features, reconstruction, clinical)
        if mace_logits.ndim > 1 and mace_logits.shape[-1] == 1:
            mace_logits = mace_logits.squeeze(-1)
        phenotype_logits = self.pheno_fc(feat_vec) if self.pheno_fc is not None else None

        return ReconSeg3DOutput(
            reconstruction=reconstruction,
            segmentation=seg_logits,
            mace_logits=mace_logits,
            features=features,
            flow=flow,
            flow_bwd=flow_bwd,
            seg_sequence=seg_sequence,
            phenotype_logits=phenotype_logits,
        )


def build_model(cfg: dict[str, Any]):
    """Factory from config dict."""
    model_cfg = cfg.get("model", cfg)
    arch = model_cfg.get("arch", "reconseg3d")
    if arch == "volume_recon":
        from reconseg3d.models.volume_recon import build_volume_recon

        return build_volume_recon(cfg)
    if arch == "volume_seg":
        from reconseg3d.models.volume_seg import build_volume_seg

        return build_volume_seg(cfg)
    return ReconSeg3D(
        in_channels=model_cfg.get("in_channels", 1),
        num_seg_classes=model_cfg.get("num_seg_classes", 5),
        base_channels=model_cfg.get("base_channels", 16),
        temporal_mode=model_cfg.get("temporal_mode", "temporal_conv"),
        clinical_dim=model_cfg.get("clinical_dim", 0),
        dropout=model_cfg.get("dropout", 0.2),
        per_frame_recon=model_cfg.get("per_frame_recon", True),
        predict_motion=model_cfg.get("predict_motion", True),
        per_frame_seg=model_cfg.get("per_frame_seg", True),
        fusion=model_cfg.get("fusion", "concat"),
        task=model_cfg.get("task", "mace"),
        num_phenotype_classes=model_cfg.get("num_phenotype_classes", 0),
        ttable_embed_dim=model_cfg.get("ttable_embed_dim", 32),
        ttable_patch_size=model_cfg.get("ttable_patch_size", 4),
        ttable_on_recon=model_cfg.get("ttable_on_recon", False),
        use_svf=bool(model_cfg.get("use_svf", False)),
        svf_steps=int(model_cfg.get("svf_steps", 7)),
    )
