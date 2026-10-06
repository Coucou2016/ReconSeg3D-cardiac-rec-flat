# Critical source pack for ChatGPT audit\nPaste this after ACK. Do not invent missing files.\n\n\n===== BEGIN FILE: docs/PAPER_PLAN.md =====\n# Paper plan (living)

This repo is a **research re-implementation + extension**, not a drop-in reproduction of the private AMI numbers in the original npj Digital Medicine paper.

## What the original paper did

1. **ReconSeg3D (public MM-WHS + ACDC):** 3D ViT reconstruction of dense volumes from sparse short-axis stacks, then 3D nnU-Net segmentation (LV / RV / LVM).
2. **HeartTTable (private AMI):** spatial + temporal + table transformers with class-token cross-attention, Cox PH, 5-year MACE.

We **do not claim 0.934 AUC**. That figure is tied to private AMI data and the original training recipe.

## What this codebase can verify publicly

| Table | Question | Data | Metrics | Claim status |
|-------|----------|------|---------|--------------|
| **T1** Reconstruction | Sparse SA → dense cine/3D | ACDC, MM-WHS (or synthetic smoke) | PSNR, global-SSIM proxy, MAE | Public, compact tensors by default |
| **T2** Segmentation | 4-class LV/RV/MYO | ACDC, MM-WHS | Dice per class, optional HD95 | Public; not full nnU-Net 256³ |
| **T3** Infarct phenotyping | Disease / scar proxies | ACDC 5-class + MINF; EMIDEC scar | Accuracy / AUC on phenotype or MINF | **Proxy only**, not 5-year MACE |
| **T4** Motion-consistent 4D | Per-frame recon + warp | Synthetic / ACDC 4D | Warp L1, cycle error, volume-curve, EF proxy | **Main novelty of this repo** |

Private AMI Cox / HeartTTable 5-year MACE remains **out of scope** until the user attaches a de-identified manifest (`data.source: nifti`) and reports C-index with nested validation. Do not copy the original AUC.

## Implementation vs paper-scale

| Component | Paper | This repo (default) |
|-----------|-------|---------------------|
| Grid | 256×256×128 | `(D,H,W)=(16,32,32)` smoke; config comments show paper size |
| Recon backbone | 3D ViT | Compact 3D CNN + optional transformer bottleneck (`volume_recon.py`) |
| Seg backbone | 3D nnU-Net | Compact 3D UNet (`volume_seg.py`) + joint ReconSeg3D heads |
| Time | Broadcast recon (limitation of the previous prototype) | **Per-frame decode + motion warp** (default) |
| Risk | Cox PH on private AMI | BCE MACE (synthetic) / phenotype (ACDC) / Cox loss (API ready) |

## Demo paper pipeline (no private AMI)

```powershell
cd E:\Projects\20260523-ReconSeg3D-cardiac-rec
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

# 1) Fake ACDC + MM-WHS + EMIDEC under data/
python scripts/prepare_demo_data.py

# 2) Paper configs (1 epoch smoke)
python scripts/train.py --config configs/paper_recon.yaml --epochs 1 --output-dir outputs/paper_recon_smoke
python scripts/eval.py --config configs/paper_recon.yaml --checkpoint outputs/paper_recon_smoke/last.pt
python scripts/train.py --config configs/paper_acdc.yaml --epochs 1 --output-dir outputs/paper_acdc_smoke
python scripts/eval.py --config configs/paper_acdc.yaml --checkpoint outputs/paper_acdc_smoke/last.pt

# 3) Ablation matrix (broadcast / per_frame±motion / fusion / phenotype)
python scripts/run_ablations.py --epochs 1 --output-root outputs/ablations

# 4) Table from metrics.json (T1–T4 columns when present)
python scripts/summarize_runs.py --runs-root outputs/ablations --out-md outputs/ablations/table.md --out-csv outputs/ablations/table.csv
```

Longer runs: raise `--epochs` (configs default to 2–20) or edit `train.epochs` / `train.early_stop_patience`.

### Ablation configs (`configs/ablations/`)

| Config | Intent |
|--------|--------|
| `broadcast_baseline.yaml` | `per_frame_recon: false` |
| `per_frame_no_motion.yaml` | per-frame, `w_warp`/`w_cycle` = 0 |
| `per_frame_motion.yaml` | full warp + cycle |
| `fusion_concat.yaml` | concat MLP fusion |
| `fusion_heart_ttable.yaml` | HeartTTable-lite fusion |
| `task_phenotype.yaml` | fake/real ACDC phenotype |

Subset example:

```powershell
python scripts/run_ablations.py --epochs 1 --only broadcast_baseline per_frame_motion task_phenotype
```

If `data.root` for ACDC/MM-WHS/EMIDEC is empty and `data.auto_fake: true` (default on public loaders), training auto-calls `make_fake_*`. Prefer `prepare_demo_data.py` once for a shared tree.

## Suggested run order

1. Smoke: `python scripts/train.py --config configs/default.yaml --epochs 2`
2. Sparse recon+seg+motion: `configs/paper_recon.yaml` (synthetic or swap `data.source`)
3. ACDC phenotype: `configs/paper_acdc.yaml` after `prepare_demo_data.py` or a real ACDC root
4. Ablations: `scripts/run_ablations.py`
5. EMIDEC scar stub: `data.source: emidec` (see `docs/DATA.md`)
6. Private AMI: `data.source: nifti`, `model.fusion: heart_ttable`, `model.task: cox`

## Honest gaps for a paper

- No original weights, no private AMI cohort, no nested CV / calibration plots yet.
- Compact backbones are **interface-compatible**, not compute-matched to 3D ViT / nnU-Net.
- HD95 uses a subsampled numpy surface distance (not MONAI/SimpleITK).
- SSIM is a **global** proxy, not a 3D windowed SSIM (stabilizers scale with data range).
- Volume-curve / `w_volsmooth` uses **fractional** LV/RV volumes (counts / DHW), not raw voxel counts.
- Trainer pools MACE/phenotype AUC and C-index over the full epoch (not mean-of-batch-AUC); tiny fake splits can still look extreme.
- `best.pt` selection is task-aware: phenotype → phenotype_acc/auc, cox → c_index, else mace_auc (fallback: val loss).
- Slice-sampling is a simulation of sparse SA, not vendor undersampled k-space.
- Clinical table tokens need a real feature dictionary (labs, meds, ECG) before HeartTTable is more than a fusion ablation.
- Demo AUC / Dice on fake NIfTIs are **pipeline checks only** — do not cite as clinical performance.
\n===== END FILE: docs/PAPER_PLAN.md =====\n\n\n===== BEGIN FILE: reconseg3d/models/reconseg3d.py =====\n"""ReconSeg3D: joint reconstruction, segmentation, and risk prediction."""

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
        self.motion_net = MotionNet(in_channels=in_channels, base_channels=max(base_channels // 2, 4))

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

    def _risk_features(
        self,
        x: torch.Tensor,
        features: torch.Tensor,
        reconstruction: torch.Tensor,
        clinical: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.heart_ttable is not None:
            vol = reconstruction if self.ttable_on_recon else x
            return self.heart_ttable(vol, clinical).features
        pooled = self.mace_pool(features).flatten(1)
        if clinical is not None and self.clinical_dim > 0:
            pooled = torch.cat([pooled, clinical], dim=1)
        return pooled

    def forward(
        self,
        x: torch.Tensor,
        clinical: torch.Tensor | None = None,
    ) -> ReconSeg3DOutput:
        """
        Args:
            x: (B, C, T, D, H, W)
            clinical: optional (B, clinical_dim) tabular features
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
                ref = t // 2
                seg_logits = seg_sequence[:, :, ref]
            else:
                seg_logits = self._interp_3d(self.seg_head(features), spatial)
            if self.predict_motion:
                flow, flow_bwd = self.motion_net(reconstruction)
        else:
            features = self.encoder(x)
            seg_logits = self._interp_3d(self.seg_head(features), spatial)
            recon_low = self._interp_3d(self.recon_decoder(features), spatial)
            reconstruction = recon_low.unsqueeze(2).expand(b, c, t, d, h, w).contiguous()

        feat_vec = self._risk_features(x, features, reconstruction, clinical)
        mace_logits = self.mace_fc(feat_vec).squeeze(-1)
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
    )
\n===== END FILE: reconseg3d/models/reconseg3d.py =====\n\n\n===== BEGIN FILE: reconseg3d/models/motion.py =====\n"""Differentiable 3D motion (displacement) and volume warping."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from reconseg3d.models.blocks import ConvBlock3D


def identity_grid(d: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Normalized grid (D, H, W, 3) in grid_sample (x, y, z) order."""
    zz = torch.linspace(-1.0, 1.0, d, device=device, dtype=dtype)
    yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    grid_z, grid_y, grid_x = torch.meshgrid(zz, yy, xx, indexing="ij")
    return torch.stack((grid_x, grid_y, grid_z), dim=-1)


def flow_to_grid(flow: torch.Tensor) -> torch.Tensor:
    """
    Convert voxel displacement (B, 3, D, H, W) with channels (dz, dy, dx)
    into a sampling grid (B, D, H, W, 3) for ``grid_sample``.
    """
    b, _, d, h, w = flow.shape
    base = identity_grid(d, h, w, flow.device, flow.dtype).unsqueeze(0).expand(b, -1, -1, -1, -1)
    dz, dy, dx = flow[:, 0], flow[:, 1], flow[:, 2]
    scale_z = 2.0 / max(d - 1, 1)
    scale_y = 2.0 / max(h - 1, 1)
    scale_x = 2.0 / max(w - 1, 1)
    disp = torch.stack((dx * scale_x, dy * scale_y, dz * scale_z), dim=-1)
    return base + disp


def warp_volume(volume: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp (B, C, D, H, W) with displacement (B, 3, D, H, W)."""
    grid = flow_to_grid(flow)
    return F.grid_sample(volume, grid, mode="bilinear", padding_mode="border", align_corners=True)


class MotionNet(nn.Module):
    """Predict 3D displacement between consecutive reconstructed frames."""

    def __init__(self, in_channels: int = 1, base_channels: int = 8) -> None:
        super().__init__()
        c = max(base_channels, 4)
        self.encoder = nn.Sequential(
            ConvBlock3D(in_channels * 2, c),
            ConvBlock3D(c, c),
        )
        self.flow_head = nn.Conv3d(c, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    def forward_pair(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """src/tgt: (B, C, D, H, W) -> flow (B, 3, D, H, W) taking src toward tgt."""
        return self.flow_head(self.encoder(torch.cat([src, tgt], dim=1)))

    def forward(self, recon: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Args:
            recon: (B, C, T, D, H, W)
        Returns:
            flow_fwd, flow_bwd each (B, 3, T-1, D, H, W), or (None, None) if T < 2.
        """
        if recon.ndim != 6:
            raise ValueError(f"Expected (B,C,T,D,H,W), got {tuple(recon.shape)}")
        b, c, t, d, h, w = recon.shape
        if t < 2:
            return None, None
        src = recon[:, :, :-1].permute(0, 2, 1, 3, 4, 5).reshape(b * (t - 1), c, d, h, w)
        tgt = recon[:, :, 1:].permute(0, 2, 1, 3, 4, 5).reshape(b * (t - 1), c, d, h, w)
        fwd = self.forward_pair(src, tgt).view(b, t - 1, 3, d, h, w).permute(0, 2, 1, 3, 4, 5)
        bwd = self.forward_pair(tgt, src).view(b, t - 1, 3, d, h, w).permute(0, 2, 1, 3, 4, 5)
        return fwd.contiguous(), bwd.contiguous()


def warp_consistency_loss(recon: torch.Tensor, flow_fwd: torch.Tensor, flow_bwd: torch.Tensor | None = None) -> torch.Tensor:
    """||Vhat_{t+1} - warp(Vhat_t, u_t)||_1 (+ backward if provided)."""
    _b, _c, t, _d, _h, _w = recon.shape
    if t < 2:
        return recon.sum() * 0.0
    src = recon[:, :, :-1]
    tgt = recon[:, :, 1:]
    bt = src.shape[0] * src.shape[2]
    c = src.shape[1]
    d, h, w = src.shape[3:]
    src_f = src.permute(0, 2, 1, 3, 4, 5).reshape(bt, c, d, h, w)
    tgt_f = tgt.permute(0, 2, 1, 3, 4, 5).reshape(bt, c, d, h, w)
    flow_f = flow_fwd.permute(0, 2, 1, 3, 4, 5).reshape(bt, 3, d, h, w)
    warped = warp_volume(src_f, flow_f)
    loss = F.l1_loss(warped, tgt_f)
    if flow_bwd is not None:
        flow_b = flow_bwd.permute(0, 2, 1, 3, 4, 5).reshape(bt, 3, d, h, w)
        warped_b = warp_volume(tgt_f, flow_b)
        loss = loss + F.l1_loss(warped_b, src_f)
        loss = loss * 0.5
    return loss


def cycle_consistency_loss(recon: torch.Tensor, flow_fwd: torch.Tensor, flow_bwd: torch.Tensor) -> torch.Tensor:
    """Warp there and back: ||warp(warp(V_t, u_fwd), u_bwd) - V_t||_1."""
    _b, _c, t, _d, _h, _w = recon.shape
    if t < 2:
        return recon.sum() * 0.0
    src = recon[:, :, :-1]
    tgt = recon[:, :, 1:]
    bt = src.shape[0] * src.shape[2]
    c = src.shape[1]
    d, h, w = src.shape[3:]
    src_f = src.permute(0, 2, 1, 3, 4, 5).reshape(bt, c, d, h, w)
    tgt_f = tgt.permute(0, 2, 1, 3, 4, 5).reshape(bt, c, d, h, w)
    flow_f = flow_fwd.permute(0, 2, 1, 3, 4, 5).reshape(bt, 3, d, h, w)
    flow_b = flow_bwd.permute(0, 2, 1, 3, 4, 5).reshape(bt, 3, d, h, w)
    cyc_src = warp_volume(warp_volume(src_f, flow_f), flow_b)
    cyc_tgt = warp_volume(warp_volume(tgt_f, flow_b), flow_f)
    return 0.5 * (F.l1_loss(cyc_src, src_f) + F.l1_loss(cyc_tgt, tgt_f))


def volume_curve_loss(seg_logits_seq: torch.Tensor, lv_index: int = 1, rv_index: int = 2) -> torch.Tensor:
    """Soft second difference of LV/RV volume *fractions* over T (needs T >= 3).

    Counts are divided by D*H*W so the loss is O(1) and does not explode with
    spatial resolution (raw voxel counts made ``w_volsmooth`` dominate early smoke).
    """
    if seg_logits_seq.ndim != 6:
        raise ValueError(f"Expected (B,K,T,D,H,W), got {tuple(seg_logits_seq.shape)}")
    t = seg_logits_seq.shape[2]
    if t < 3:
        return seg_logits_seq.sum() * 0.0
    probs = F.softmax(seg_logits_seq, dim=1)
    k = probs.shape[1]
    spatial = float(probs.shape[3] * probs.shape[4] * probs.shape[5])
    terms = []
    for idx in (lv_index, rv_index):
        if idx >= k:
            continue
        # (B, T) fractional chamber volumes in [0, 1]
        frac = probs[:, idx].sum(dim=(2, 3, 4)) / max(spatial, 1.0)
        d2 = frac[:, 2:] - 2.0 * frac[:, 1:-1] + frac[:, :-2]
        terms.append((d2 ** 2).mean())
    if not terms:
        return seg_logits_seq.sum() * 0.0
    return sum(terms) / len(terms)


def temporal_seg_smoothness(seg_logits_seq: torch.Tensor) -> torch.Tensor:
    """L1 smoothness of softmax maps across adjacent frames."""
    if seg_logits_seq.shape[2] < 2:
        return seg_logits_seq.sum() * 0.0
    probs = F.softmax(seg_logits_seq, dim=1)
    return (probs[:, :, 1:] - probs[:, :, :-1]).abs().mean()
\n===== END FILE: reconseg3d/models/motion.py =====\n\n\n===== BEGIN FILE: reconseg3d/models/losses.py =====\n"""Multi-task losses: recon / seg / MACE / Cox / motion / phenotype."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from reconseg3d.models.motion import (
    cycle_consistency_loss,
    temporal_seg_smoothness,
    volume_curve_loss,
    warp_consistency_loss,
)


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft Dice loss averaged over classes."""
    probs = F.softmax(logits, dim=1)
    targets = targets.long()
    valid = targets != ignore_index
    if not valid.any():
        return logits.sum() * 0.0

    loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
    count = 0
    for cls in range(num_classes):
        pred_c = probs[:, cls]
        target_c = (targets == cls).float()
        mask = valid.float()
        pred_c = pred_c * mask
        target_c = target_c * mask
        inter = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2 * inter + eps) / (union + eps)
        loss = loss + (1 - dice)
        count += 1
    return loss / max(count, 1)


def recon3d_mse(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Paper-style dense reconstruction MSE (α1)."""
    return F.mse_loss(reconstruction, target)


def seg3d_ce_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ce: nn.CrossEntropyLoss,
    use_dice: bool = True,
) -> torch.Tensor:
    """Paper-style 3D seg CE+Dice (α2)."""
    loss = ce(logits, targets.long())
    if use_dice:
        loss = loss + dice_loss(logits, targets, num_classes)
    return loss


def seg2d_from_slices(
    logits: torch.Tensor,
    targets: torch.Tensor,
    slice_mask: torch.Tensor,
    num_classes: int,
    ce: nn.CrossEntropyLoss,
    use_dice: bool = True,
) -> torch.Tensor:
    """
    Restrict CE+Dice to selected SA slices (paper α3 / seg2d).

    slice_mask: (B, D) or (B, D, H, W) with 1 on sampled slices.
    """
    target = targets.long().clone()
    mask = slice_mask
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    if mask.ndim == 2:
        mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(target)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1).expand_as(target) if mask.shape[1] != target.shape[1] else mask
    if mask.shape != target.shape:
        raise ValueError(f"slice_mask broadcast failed: {tuple(mask.shape)} vs {tuple(target.shape)}")
    target = target.masked_fill(~mask, -1)
    return seg3d_ce_dice(logits, target, num_classes, ce, use_dice=use_dice)


def cox_partial_likelihood(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Breslow Cox partial likelihood. NaN-safe for all-censored / single-event batches.

    risk: (B,) linear predictor (higher = higher hazard)
    time: (B,) follow-up
    event: (B,) 1 = event, 0 = censored
    """
    risk = risk.reshape(-1).float()
    time = time.reshape(-1).to(device=risk.device, dtype=risk.dtype)
    event = event.reshape(-1).to(device=risk.device, dtype=risk.dtype)
    n = risk.numel()
    if n == 0:
        return risk.sum() * 0.0
    n_events = event.sum()
    if n_events < 1:
        return (risk * 0.0).sum()
    # (N, N) at-risk: time_j >= time_i
    at_risk = time.unsqueeze(0) >= time.unsqueeze(1)
    risk_ij = risk.unsqueeze(0).expand(n, n)
    risk_ij = risk_ij.masked_fill(~at_risk, float("-inf"))
    lse = torch.logsumexp(risk_ij, dim=1)
    ll = event * (risk - lse)
    valid = torch.isfinite(ll) & (event > 0.5)
    if not valid.any():
        return (risk * 0.0).sum()
    return -(ll[valid].sum() / valid.float().sum().clamp_min(eps))


class FocalLoss(nn.Module):
    """Binary focal loss for MACE."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class MultiTaskLoss(nn.Module):
    """Weighted sum of seg + recon + MACE/Cox/phenotype + motion losses."""

    def __init__(
        self,
        num_seg_classes: int = 5,
        w_seg: float = 1.0,
        w_recon: float = 0.5,
        w_mace: float = 1.0,
        recon_loss: str = "l1",
        mace_loss: str = "bce",
        use_dice: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        alpha1: float = 0.0,
        alpha2: float = 0.0,
        alpha3: float = 0.0,
        w_warp: float = 0.0,
        w_cycle: float = 0.0,
        w_volsmooth: float = 0.0,
        w_segsmooth: float = 0.0,
        w_cox: float = 0.0,
        w_phenotype: float = 0.0,
        task: str = "mace",
        num_phenotype_classes: int = 5,
    ) -> None:
        super().__init__()
        self.num_seg_classes = num_seg_classes
        self.w_seg = w_seg
        self.w_recon = w_recon
        self.w_mace = w_mace
        self.use_dice = use_dice
        self.recon_loss = recon_loss
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.pheno_ce = nn.CrossEntropyLoss()
        self.mace_loss_type = mace_loss
        self.focal = FocalLoss(focal_alpha, focal_gamma)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.w_warp = w_warp
        self.w_cycle = w_cycle
        self.w_volsmooth = w_volsmooth
        self.w_segsmooth = w_segsmooth
        self.w_cox = w_cox
        self.w_phenotype = w_phenotype
        self.task = task
        self.num_phenotype_classes = num_phenotype_classes

    def forward(
        self,
        reconstruction: torch.Tensor,
        seg_logits: torch.Tensor,
        mace_logits: torch.Tensor,
        target_volume: torch.Tensor,
        target_seg: torch.Tensor,
        target_mace: torch.Tensor,
        flow: torch.Tensor | None = None,
        flow_bwd: torch.Tensor | None = None,
        seg_sequence: torch.Tensor | None = None,
        slice_mask: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        event: torch.Tensor | None = None,
        phenotype_logits: torch.Tensor | None = None,
        target_phenotype: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        zero = reconstruction.sum() * 0.0

        seg_ce = self.ce(seg_logits, target_seg.long())
        seg_dice = dice_loss(seg_logits, target_seg, self.num_seg_classes) if self.use_dice else zero
        seg_loss = seg_ce + seg_dice

        if self.recon_loss == "l2":
            recon_reg = F.mse_loss(reconstruction, target_volume)
        else:
            recon_reg = F.l1_loss(reconstruction, target_volume)
        recon_mse = recon3d_mse(reconstruction, target_volume)

        if self.task == "cox" or self.w_cox > 0:
            if time is None or event is None:
                cox_loss = zero
            else:
                cox_loss = cox_partial_likelihood(mace_logits, time, event)
        else:
            cox_loss = zero

        if self.task == "phenotype" and phenotype_logits is not None and target_phenotype is not None:
            pheno_loss = self.pheno_ce(phenotype_logits, target_phenotype.long())
        elif phenotype_logits is not None and target_phenotype is not None and self.w_phenotype > 0:
            pheno_loss = self.pheno_ce(phenotype_logits, target_phenotype.long())
        else:
            pheno_loss = zero

        if self.task == "cox":
            mace_loss = cox_loss
        elif self.mace_loss_type == "focal":
            mace_loss = self.focal(mace_logits, target_mace)
        else:
            mace_loss = F.binary_cross_entropy_with_logits(mace_logits, target_mace.float())

        w_seg = self.alpha2 if self.alpha2 > 0 else self.w_seg
        recon_term = self.w_recon * recon_reg + self.alpha1 * recon_mse
        total = w_seg * seg_loss + recon_term + self.w_mace * mace_loss

        seg2d_loss = zero
        if self.alpha3 > 0 and slice_mask is not None:
            seg2d_loss = seg2d_from_slices(
                seg_logits, target_seg, slice_mask, self.num_seg_classes, self.ce, self.use_dice
            )
            total = total + self.alpha3 * seg2d_loss

        warp_loss = zero
        cycle_loss = zero
        vol_loss = zero
        smooth_loss = zero
        if self.w_warp > 0 and flow is not None and reconstruction.shape[2] > 1:
            warp_loss = warp_consistency_loss(reconstruction, flow, flow_bwd)
            total = total + self.w_warp * warp_loss
        if self.w_cycle > 0 and flow is not None and flow_bwd is not None and reconstruction.shape[2] > 1:
            cycle_loss = cycle_consistency_loss(reconstruction, flow, flow_bwd)
            total = total + self.w_cycle * cycle_loss
        if seg_sequence is not None:
            if self.w_volsmooth > 0:
                vol_loss = volume_curve_loss(seg_sequence)
                total = total + self.w_volsmooth * vol_loss
            if self.w_segsmooth > 0:
                smooth_loss = temporal_seg_smoothness(seg_sequence)
                total = total + self.w_segsmooth * smooth_loss

        if self.task != "cox" and self.w_cox > 0:
            total = total + self.w_cox * cox_loss
        if self.w_phenotype > 0:
            total = total + self.w_phenotype * pheno_loss
        elif self.task == "phenotype":
            total = total + pheno_loss

        return {
            "total": total,
            "seg": seg_loss.detach(),
            "recon": recon_reg.detach(),
            "recon_mse": recon_mse.detach(),
            "mace": mace_loss.detach(),
            "seg2d": seg2d_loss.detach() if torch.is_tensor(seg2d_loss) else zero.detach(),
            "warp": warp_loss.detach() if torch.is_tensor(warp_loss) else zero.detach(),
            "cycle": cycle_loss.detach() if torch.is_tensor(cycle_loss) else zero.detach(),
            "volsmooth": vol_loss.detach() if torch.is_tensor(vol_loss) else zero.detach(),
            "segsmooth": smooth_loss.detach() if torch.is_tensor(smooth_loss) else zero.detach(),
            "cox": cox_loss.detach() if torch.is_tensor(cox_loss) else zero.detach(),
            "phenotype": pheno_loss.detach() if torch.is_tensor(pheno_loss) else zero.detach(),
        }
\n===== END FILE: reconseg3d/models/losses.py =====\n\n\n===== BEGIN FILE: reconseg3d/training/trainer.py =====\n"""Training and validation loops."""

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
\n===== END FILE: reconseg3d/training/trainer.py =====\n\n\n===== BEGIN FILE: reconseg3d/training/metrics.py =====\n"""Evaluation metrics with safe edge-case handling."""

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
\n===== END FILE: reconseg3d/training/metrics.py =====\n