SOURCE PACK CHUNK 1 — continue reading next chunks before proposing patches.
# Source pack for ChatGPT audit
Do not invent missing files. SHA ZIP baseline D7830799...

---
## FILE: docs/PAPER_PLAN.md

```
# Paper plan (living)

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

```

---
## FILE: docs/DATA.md

```
# Data layout / 数据格式

## Tensor layout (canonical)

| Field | Shape | Dtype | Notes |
|-------|-------|-------|-------|
| `volume` | `(B, C, T, D, H, W)` | float32 | Cine MRI / CT sequence (sparse SA if `slice_sampling`) |
| `volume_target` | same as volume | float32 | Dense recon target (present when slice sampling) |
| `slice_mask` | `(B, D)` | float32 | 1 on sampled short-axis slices |
| `segmentation` | `(B, D, H, W)` | int64 | Reference-frame voxel labels |
| `mace` | `(B,)` | float32 | 0/1 MACE (dummy 0 on public datasets) |
| `time` / `event` | `(B,)` | float32 | Survival time + event indicator (Cox) |
| `phenotype` | `(B,)` | int64 | ACDC 5-class or MINF proxy |
| `clinical` | `(B, F)` | float32 | Optional tabular features |

If your files use `(B, C, D, H, W, T)`, convert with:

```python
from reconseg3d.utils.shapes import permute_dhw_t_to_ctdhw
volume = permute_dhw_t_to_ctdhw(volume)
```

`data.source` may be `synthetic` | `acdc` | `mmwhs` | `emidec` | `nifti`.

Fake NIfTI writers work **without** `nibabel` (numpy+gzip fallback in `reconseg3d.data.io`). Install `nibabel` for real challenge files.

Paper-scale reconstruction grid is **256×256×128** (H×W×D). This repo stores spatial size as **`(D, H, W)`**; compact tests use `[16, 32, 32]` or smaller.

## Segmentation classes

| ID | Structure | Used when |
|----|-----------|-----------|
| 0 | Background | always |
| 1 | Left ventricle (LV) | always |
| 2 | Right ventricle (RV) | ACDC / MM-WHS / synthetic |
| 3 | Myocardium (LVM) | always |
| 4 | Scar / infarct | EMIDEC / 5-class synthetic; omit for 4-class paper recon+seg |

## Sparse SA slice sampling

`reconseg3d.data.slice_sampling.sample_sparse_sa_stack` samples **S ∈ [8, 16]** slices along D (clipped to depth), applies in-plane rotation 1–5° and translation 1–5 px, adds noise, and **zeros unselected slices**. Enable with `data.slice_sampling: true` (see `configs/paper_recon.yaml`).

## ACDC (`data.source: acdc`)

Official layout:

```
data/acdc/patient001/
  Info.cfg                 # ED, ES, Height, Weight, NbFrame, Group
  patient001_4d.nii.gz     # (X, Y, Z, T)
  patient001_frame01.nii.gz
  patient001_frame01_gt.nii.gz
```

Obtain from the [ACDC challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/). Point the config:

```yaml
data:
  source: acdc
  root: E:/data/ACDC/training   # folder that contains patientXXX/
  spatial_size: [16, 32, 32]    # raise toward [128, 256, 256] for paper-scale
```

GT mapping: ACDC `{1=RV, 2=MYO, 3=LV}` → repo `{2=RV, 3=MYO, 1=LV}`.
Phenotype from `Group`: NOR / MINF / DCM / HCM / RV.

**Fake data for CI / local demos** (do not download in tests):

```powershell
python scripts/prepare_demo_data.py
# or:
python -c "from reconseg3d.data.acdc import make_fake_acdc; make_fake_acdc('data/acdc')"
```

If `data.auto_fake: true` (default for public loaders) and `root` has no patients, training creates a fake tree automatically.
## MM-WHS MRI (`data.source: mmwhs`)

```
data/mmwhs/
  mr_train_1001_image.nii.gz   # (X, Y, Z)
  mr_train_1001_label.nii.gz
```

Label map: **500→LV(1), 600→RV(2), 205→LVM(3)**; other structures → background.
3D volumes are tiled along T when `num_frames > 1` (motion losses should be down-weighted).

```powershell
python -c "from reconseg3d.data.mmwhs import make_fake_mmwhs; make_fake_mmwhs('data/mmwhs')"
```

## EMIDEC LGE (`data.source: emidec`)

```
data/emidec/Case_1/
  Images/Case_1.nii.gz
  Contours/Case_1.nii.gz
```

Contours: 1 myocardium, 2 cavity, 3 infarct, 4 MVO → repo LV=1, MYO=3, scar=4.
Set `data.stack_scar_channel: true` and `model.in_channels: 2` for `[cine, scar_mask]` input.

```powershell
python -c "from reconseg3d.data.emidec import make_fake_emidec; make_fake_emidec('data/emidec')"
```

## Real AMI manifest (`data.source: nifti`)

Private AMI / HeartTTable+Cox experiments (not for public AUC claims):

```
data/ami/
  train_manifest.json
  val_manifest.json
  case_001/
    cine.nii.gz
    seg.nii.gz
    label.json           # {"mace": 0|1, "time": 4.2, "event": 1, "phenotype": 1}
    clinical.json
```

## Synthetic demo

Set `data.source: synthetic` in `configs/default.yaml`. Labels correlate scar burden, troponin, EF, and age with MACE probability (pipeline checks only, **not** clinical validation).

```

---
## FILE: configs/default.yaml

```
seed: 42
deterministic: true
device: auto
output_dir: outputs
run_name: default

data:
  source: synthetic
  in_channels: 1
  num_frames: 8
  spatial_size: [16, 32, 32]  # paper-scale: [128, 256, 256] as (D, H, W) ~ 256×256×128
  batch_size: 2
  num_workers: 0  # Windows
  train_samples: 16
  val_samples: 8
  clinical_dim: 4
  slice_sampling: false

model:
  arch: reconseg3d
  in_channels: 1
  num_seg_classes: 5
  base_channels: 16
  temporal_mode: temporal_conv
  clinical_dim: 4
  dropout: 0.2
  per_frame_recon: true
  predict_motion: true
  per_frame_seg: true
  fusion: concat  # ablation; set heart_ttable for WP2 path
  task: mace  # mace | cox | phenotype

loss:
  w_seg: 1.0
  w_recon: 0.5
  w_mace: 1.0
  recon_loss: l1
  mace_loss: bce
  use_dice: true
  alpha1: 0.1   # recon3d MSE
  alpha2: 1.0   # seg3d CE+Dice (used if >0, else w_seg)
  alpha3: 0.0   # seg2d (needs slice_mask)
  w_warp: 0.1
  w_cycle: 0.1
  w_volsmooth: 0.05
  w_segsmooth: 0.0
  w_cox: 0.0
  w_phenotype: 0.0

metrics:
  hd95: false  # enable for paper tables; skip in smoke (slow on large grids)

train:
  epochs: 2
  lr: 0.001
  weight_decay: 0.0001
  grad_clip: 1.0
  amp: false
  early_stop_patience: 0

```

---
## FILE: configs/paper_acdc.yaml

```
# ACDC phenotype / 4-class segmentation (public proxy — not private AMI MACE).
# Point data.root at the official ACDC tree once downloaded:
#   data.root: E:/data/ACDC/training
# Fake layout for CI: python -c "from reconseg3d.data.acdc import make_fake_acdc; make_fake_acdc('data/acdc')"

seed: 42
deterministic: true
device: auto
output_dir: outputs/paper_acdc
run_name: paper_acdc

data:
  source: acdc
  root: data/acdc
  in_channels: 1
  num_frames: 8
  spatial_size: [16, 32, 32]  # paper-scale: [128, 256, 256] (D,H,W)
  batch_size: 2
  num_workers: 0
  clinical_dim: 4
  slice_sampling: false
  auto_fake: true  # if root empty: make_fake_acdc; or run scripts/prepare_demo_data.py

model:
  arch: reconseg3d
  in_channels: 1
  num_seg_classes: 4  # bg / LV / RV / LVM
  base_channels: 16
  temporal_mode: temporal_conv
  clinical_dim: 4
  dropout: 0.2
  per_frame_recon: true
  predict_motion: true
  per_frame_seg: true
  fusion: heart_ttable
  task: phenotype
  num_phenotype_classes: 5  # NOR, MINF, DCM, HCM, RV
  ttable_embed_dim: 32
  ttable_patch_size: 4
  ttable_on_recon: false

loss:
  w_seg: 1.0
  w_recon: 0.25
  w_mace: 0.0
  recon_loss: l1
  mace_loss: bce
  use_dice: true
  alpha1: 0.1
  alpha2: 1.0
  alpha3: 0.0
  w_warp: 0.1
  w_cycle: 0.1
  w_volsmooth: 0.05
  w_phenotype: 1.0
  w_cox: 0.0

metrics:
  hd95: true

train:
  epochs: 20
  lr: 0.0005
  weight_decay: 0.0001
  grad_clip: 1.0
  amp: false
  early_stop_patience: 0

```

---
## FILE: configs/paper_recon.yaml

```
# Sparse-SA reconstruction + 3D seg + motion (paper recon setting, compact tensors).
# Use data.source: acdc or mmwhs with real roots when data is available.
# Paper-scale comment: 256×256×128 in-plane/through-plane; keep spatial_size small here.

seed: 42
deterministic: true
device: auto
output_dir: outputs/paper_recon
run_name: paper_recon

data:
  source: synthetic
  in_channels: 1
  num_frames: 8
  spatial_size: [16, 32, 32]  # paper-scale: [128, 256, 256] (D, H, W)
  batch_size: 2
  num_workers: 0
  train_samples: 16
  val_samples: 8
  clinical_dim: 4
  slice_sampling: true
  slice_s_range: [8, 16]  # clipped to D; for D=16 this zeros some slices
  slice_rot_deg: [1, 5]
  slice_trans_px: [1, 5]
  slice_noise_std: 0.02
  # Optional public roots (auto_fake creates layouts if missing):
  # source: acdc | mmwhs ; root: data/acdc ; auto_fake: true

model:
  arch: reconseg3d
  in_channels: 1
  num_seg_classes: 4
  base_channels: 16
  temporal_mode: temporal_conv
  clinical_dim: 4
  dropout: 0.2
  per_frame_recon: true
  predict_motion: true
  per_frame_seg: true
  fusion: concat
  task: mace

loss:
  w_seg: 0.0
  w_recon: 0.0
  w_mace: 0.25
  recon_loss: l2
  mace_loss: bce
  use_dice: true
  alpha1: 0.1   # recon3d MSE
  alpha2: 1.0   # seg3d CE+Dice
  alpha3: 0.2   # seg2d on sampled slices
  w_warp: 0.1
  w_cycle: 0.1
  w_volsmooth: 0.05
  w_segsmooth: 0.02

metrics:
  hd95: false

train:
  epochs: 10
  lr: 0.001
  weight_decay: 0.0001
  grad_clip: 1.0
  amp: false
  early_stop_patience: 0

```

---
## FILE: reconseg3d/models/reconseg3d.py

```
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

```

---
