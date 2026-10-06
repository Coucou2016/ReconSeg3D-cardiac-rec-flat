# ReconSeg3D-cardiac-rec

**Geometry- and motion-constrained 4D reconstruction/segmentation from sparse SA CMR** (public methods extension).  
稀疏短轴 CMR 的几何/运动约束 4D 重建与分割方法扩展。

This repository is a **public methods codebase**. Main line = recon + seg + inverse-consistent motion + function proxies.  
It is **not** a claim of the original paper’s private-AMI **0.934 AUC**. HeartTTable / Cox / MACE heads are **supplementary**.

## 项目简介 / Overview

1. **Per-frame 3D reconstruction** `(B, C, T, D, H, W)` from sparse SA (broadcast ablation retained)
2. **Geometry-aware motion**: true `L_inv`, smoothness, Jacobian folding, **closed-cycle** `L_periodic` (T pairs incl. `T-1→0`); image-cycle and volume-curve are auxiliary / physiological only
3. **ED/ES-aware segmentation** (ACDC labeled phases; anchor-preserving temporal subsample; unlabeled frames via motion self-supervision)
4. **Optional** phenotype / Cox / MACE / HeartTTable-lite (off in `configs/publication_*.yaml`; phenotype proxy under `configs/proxy/`)

See [docs/PAPER_PLAN.md](docs/PAPER_PLAN.md) for claim boundaries. Demo metrics are **DEMO-labeled**.

## 安装 / Install

```powershell
cd E:\Projects\20260523-ReconSeg3D-cardiac-rec
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## 快速训练 / Quick train (synthetic, per-frame + motion)

```powershell
python scripts/train.py --config configs/default.yaml --epochs 2 --output-dir outputs
```

Sparse-SA recon + seg + motion losses:

```powershell
python scripts/train.py --config configs/paper_recon.yaml --epochs 2 --output-dir outputs/paper_recon
```

ACDC phenotype proxy (fake or real root — see [docs/DATA.md](docs/DATA.md)):

```powershell
python scripts/prepare_demo_data.py
python scripts/train.py --config configs/proxy/proxy_acdc_phenotype.yaml --epochs 2 --output-dir outputs/proxy_acdc
# shim still works: configs/paper_acdc.yaml
```

## Paper experiments (demo)

Local Table-style ablations **without private AMI** (fake ACDC/MM-WHS/EMIDEC under `data/`):

```powershell
python scripts/prepare_demo_data.py
python scripts/train.py --config configs/paper_recon.yaml --epochs 1 --output-dir outputs/paper_recon_smoke
python scripts/train.py --config configs/paper_acdc.yaml --epochs 1 --output-dir outputs/paper_acdc_smoke
python scripts/run_ablations.py --epochs 1 --output-root outputs/ablations
python scripts/summarize_runs.py --runs-root outputs/ablations --out-md outputs/ablations/table.md --out-csv outputs/ablations/table.csv
```

Ablation YAMLs live in `configs/ablations/` (broadcast vs per-frame, warp/cycle on/off, concat vs HeartTTable, phenotype).  
Eval writes `outputs/<run>/metrics.json`. **Do not claim private-AMI 0.934 AUC** from these demo numbers. Details: [docs/PAPER_PLAN.md](docs/PAPER_PLAN.md).

## 评估 / Evaluate

```powershell
python scripts/eval.py --config configs/default.yaml --checkpoint outputs/last.pt
# writes <checkpoint_dir>/metrics.json by default
```

## 推理 / Predict

```powershell
python scripts/predict.py --checkpoint outputs/last.pt --config configs/default.yaml --out-dir predictions
```

Exports reconstruction (and flow when present). With `model.task: phenotype`, also prints/saves phenotype probabilities.
## 测试 / Tests

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest tests/ -v
```

（部分 Conda 环境需禁用自动插件，见 `pytest.ini`。）

## 数据格式

见 [docs/DATA.md](docs/DATA.md)。标准张量布局为 **`(B, C, T, D, H, W)`**。

Point a real ACDC download at `configs/paper_acdc.yaml`:

```yaml
data:
  source: acdc
  root: E:/data/ACDC/training
```

## 模型架构（简述）

```
Input (B,C,T,D,H,W)  [optional sparse SA; mm translation / sampling_ratio]
  → per-frame 3D CNN encoder + temporal mix
       (temporal_conv | conv_lstm | temporal_attention)
  → per-frame recon decoder → (B, C, T, D, H, W)
  → MotionNet pull fields (optional SVF + scaling-and-squaring)
       L_inv / L_smooth / L_jac / L_periodic(closed-cycle) / L_ed_ref
  → seg head → (B, K, D, H, W) and optional (B, K, T, D, H, W)
  → concat MLP  or  HeartTTable-lite (supplementary)
```

**Publication (formal):** `configs/publication/` (`publication_recon`, `_motion`, `_motion_svf`, `_seg`, `_mms`, `_recon_hires`).  
**Smoke / CI:** `configs/smoke/smoke_motion.yaml`. **Proxy:** `configs/proxy/`.  
**Baselines:** `configs/baselines/` + `docs/BASELINES.md`. Root `paper_*` = DEMO shims — **not** formal tables.

`model.task`: `reconstruction` | `motion` | `segmentation` | `joint` | `phenotype` | `cox` | `mace` (last three supplementary).  
Checkpointing uses `selection.metric` / `mode` (never blind `mace_auc` when `w_mace:0`).

Folds: diagnosis-stratified API via `data.fold` / `fold_file`. Committed `splits/acdc_fold*.json` are **CI/smoke placeholders** only (`synthetic_placeholder: true`; see `splits/README.md`). Publication configs default `fold: null` until real folds are regenerated. Multi-seed: `scripts/run_multiseed.py`.  
Eval writes **one row per patient** in `results/case_metrics.csv` (default `batch_size=1`), plus `summary_metrics.json`, `bootstrap_ci.json`.

Standalone compact recon/seg: `CompactVolumeRecon`, `VolumeUNet3D`. `publication_recon_hires` uses **`[64,128,128]` intermediate** — **not** paper 256³. Optional VRAM-gated stub: `configs/publication/publication_recon_256.yaml` (not Done / no wall-clock claim).

## 复现 / Reproducibility

- 全局种子：`seed` in config / `--seed` on `train.py`; multi-seed harness under `scripts/run_multiseed.py`
- Windows 建议 `data.num_workers: 0`
- Best checkpoint uses `selection.metric` / `mode`
- Checkpoints store `cfg`, `run_name`, `best_selection_score` / `best_val_loss`; runs also write `config_snapshot.yaml` + `metrics.json`
- Patient-level metrics: sample-weighted trainer aggregation; eval bootstrap CIs

## 限制与假设 / Limitations

| 项目 | 说明 |
|------|------|
| 合成数据 | 仅用于流水线验证，不代表真实 AMI 分布 |
| 重建 | 默认 **逐帧解码**；旧版时间维广播仍可作为消融 |
| 骨干 | Compact CNN/UNet+small transformer，不是原论文 3D ViT / nnU-Net 256³；hires 仅为 `[64,128,128]` 中间分辨率 |
| MACE AUC | **不报告 0.934**；公开验证是重建/分割与 ACDC MINF、EMIDEC scar 代理 |
| Cox / HeartTTable | 接口已就绪；5 年 MACE 需要私有 AMI 队列 |
| HD95 / SSIM | HD95 uses batch `spacing`; tables prefer **`ssim_3d`**; `recon_ssim_proxy` is global-only (no `recon_ssim` alias) |
| EF | Primary = physical **EDV/ESV/EF (mL/%)** from spacing + ED/ES; `ef_proxy` is voxel max/min only |
| NIfTI | 需安装 `nibabel`；affine/spacing 随 sample 传递 |
| Loop | True closed-cycle `L_periodic` (T pairs incl. `T-1→0`); inv/smooth/jac/ed_ref also on |
| External data | Licensed ACDC / M&Ms subject tables remain **待补充** until mounted |

## 目录结构

```
reconseg3d/     # 模型、数据、训练、推理、baselines
configs/        # publication/ smoke/ proxy/ baselines/ ablations/ + root shims
splits/         # acdc_fold{0-4}.json (diagnosis-stratified)
scripts/        # train / eval / run_multiseed / make_acdc_folds / ...
tests/          # pytest
docs/           # DATA.md, PAPER_PLAN.md, BASELINES.md
```
