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

`reconseg3d.data.slice_sampling.sample_sparse_sa_stack` samples slices along D,
applies in-plane rotation 1–5°, and translation either:

- **Physical (preferred):** `slice_trans_mm: [1, 5]` with spacing `(sz,sy,sx)` mm
  from the NIfTI (or `slice_spacing_dhw` in config), or
- **Legacy pixels:** `slice_trans_px: [1, 5]` on small smoke grids.

Optional `slice_sampling_ratio` (e.g. `0.5`) sets `S = round(ratio * D)` instead
of the paper `S ∈ [8, 16]` range. Unselected slices are zeroed. Enable with
`data.slice_sampling: true`.

## ACDC folds

Diagnosis-stratified 5-fold lists live under `splits/acdc_fold{0-4}.json`
(NOR/MINF/DCM/HCM/RV). Point a config at a fold:

```yaml
data:
  source: acdc
  root: E:/data/ACDC/training
  fold: 0
  splits_dir: splits
```

Regenerate after mounting real data:

```powershell
python scripts/make_acdc_folds.py --root E:/data/ACDC/training
```

## M&Ms (`data.source: mms`)

External multi-site generalization. Download from https://www.ub.edu/mnms/ .
Publication config `configs/publication/publication_mms.yaml` sets
`allow_fake_data: false` and **hard-fails** if the root is empty. Subject tables
remain **待补充** until licensed data are evaluated.

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
