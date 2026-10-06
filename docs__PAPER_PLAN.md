# Paper plan (living)

This repo is a **research re-implementation + extension**, not a drop-in reproduction of the private AMI numbers in the original npj Digital Medicine paper.

## What the original paper did

1. **ReconSeg3D (public MM-WHS + ACDC):** 3D ViT reconstruction of dense volumes from sparse short-axis stacks, then 3D nnU-Net segmentation (LV / RV / LVM).
2. **HeartTTable (private AMI):** spatial + temporal + table transformers with class-token cross-attention, Cox PH, 5-year MACE.

We **do not claim 0.934 AUC**. That figure is tied to private AMI data and the original training recipe.

## Main publication line (this revision)

**Geometry- and motion-constrained 4D reconstruction/segmentation from sparse SA cine**, with:

| Term | Role |
|------|------|
| `L_inv` | True inverse consistency `‖u+W(v,u)‖ + ‖v+W(u,v)‖` (same pull grid as image warp) |
| `L_smooth` | `‖∇u‖²` |
| `L_jac` | `ReLU(ε − det J)` folding penalty |
| `L_periodic` / `L_loop` | **Closed-cycle** composition including edge `T-1→0` ≈ Id |
| `L_ed_ref` | ED-anchored composed-path inverse consistency (`φ_{k→ED}` vs `φ_{ED→k}`) |
| Image-cycle `w_cycle` | **Auxiliary** intensity round-trip only |
| `w_volsmooth` | **Demoted** physiological volume-curve regularizer |
| MACE / Cox / HeartTTable | **Supplementary / extensibility** (`w_mace:0`, `w_cox:0` in publication configs) |

Configs (honest naming):
- **Publication:** `configs/publication/` (+ root shims `publication_*.yaml`) — `allow_fake_data: false`
- **Smoke / CI:** `configs/smoke/` (+ `configs/smoke_motion.yaml`)
- **Proxy:** `configs/proxy/proxy_acdc_phenotype.yaml` (shim `configs/paper_acdc.yaml`)
- **DEMO shims:** `configs/paper_recon.yaml` — **not** formal result tables
- **Baselines:** `configs/baselines/`; temporal ablations under `configs/ablations/`

All publication configs set honest `task:` + `selection.metric` / `mode`.

### Deformation convention

Displacement channels `(dz, dy, dx)` in voxels → `grid_sample` with `align_corners=True`, scale `2/max(dim-1,1)`. Pull composition: `compose_pull(u,v) = v + W(u,v)`.

**Closed-cycle MotionNet:** predicts **T** pairs for `T` frames (adjacent `0..T-2` plus closing `T-1→0`). Do **not** call adjacent-only (T-1) compose a "full cycle".

**ED-anchored path:** left-fold **adjacent** fields into `φ_{k→ED}` / `φ_{ED→k}`; apply the same inverse-consistency residual (`w_ed_ref`). ED/ES from ACDC `Info.cfg`; temporal subsample uses `select_time_indices_keep_anchors`.

**Clinical inputs:** ACDC `clinical_base` is height/weight/nb only — phenotype/Group is never a feature (proxy configs only).

### Checkpoint selection

| task | default metric | mode |
|------|----------------|------|
| motion | `prop_ed2es_dice_mean` | max (fallback `loss_total` min) |
| segmentation | `dice_mean` | max |
| reconstruction | `recon_mae` | min |
| joint | `loss_total` | min |

Override with `selection.metric` / `selection.mode` in YAML.

## What this codebase can verify publicly

| Table | Question | Data | Metrics | Claim status |
|-------|----------|------|---------|--------------|
| **T1** Reconstruction | Sparse SA → dense cine/3D | ACDC, MM-WHS (or synthetic smoke) | PSNR, **`ssim_3d`** (windowed), `recon_ssim_proxy` (global), MAE | Public API; real-table **待补充** without licensed data |
| **T2** Segmentation | 4-class LV/RV/MYO at ED/ES | ACDC | Dice; HD95 with batch `spacing` | Public; ED/ES supervised only |
| **T3** Motion / label prop | ED↔ES warp of GT masks | ACDC / synthetic | Prop Dice + HD95 (mm), **per-patient** then aggregate + bootstrap CI | API ready; real-table **待补充** |
| **T3b** Function | EDV/ESV/EF | ACDC | **Physical mL / %** from spacing + ED/ES; `ef_proxy` labeled voxel proxy only | API ready |
| **T4** Geometry regularizers | inv / smooth / jac / periodic | Synthetic / ACDC 4D | inv_error, jac_neg_ratio, loop | Implemented (true closed cycle) |
| **Supp.** Phenotype / Cox / MACE | Proxy risk | ACDC phenotype; private AMI | Acc / C-index | Not main line |

## Implementation vs paper-scale

| Component | Paper | This repo |
|-----------|-------|-----------|
| Grid | 256×256×128 | Smoke `(16,32,32)`; publication mid `(32,64,64)`; **hires = intermediate `(64,128,128)` — not paper 256³**; optional stub `publication_recon_256.yaml` `(128,256,256)` if VRAM allows (not Done) |
| Motion | — | Closed-cycle inverse-consistent pull + ED-ref; image-cycle auxiliary |
| SVF | — | **Done** — `model.use_svf: true` + scaling-and-squaring (`publication_motion_svf.yaml`) |
| Temporal | — | `temporal_conv` \| `conv_lstm` \| `temporal_attention` |
| Risk | Cox on private AMI | Off in publication configs |
| Fusion | HeartTTable full | HeartTTable-lite ablation only |

## Tracked TODOs

### Done (in-repo)

- [x] ED-reference motion path (`w_ed_ref`) — 2026-09-14
- [x] Closed-cycle `L_periodic` (T pairs incl. closing edge) — 2026-09-14
- [x] Anchor-preserving temporal subsample + phenotype leakage fix — 2026-09-14
- [x] Windowed 3D SSIM (`ssim_3d`) + tests — 2026-09-15
- [x] Physical EDV/ESV/EF (mL/%) + patient-level CSV/JSON + bootstrap CI — 2026-09-15
- [x] Trainer sample-weighted aggregation — 2026-09-15
- [x] 5-fold diagnosis-stratified ACDC split **API** (`write_acdc_folds` / `fold`/`fold_file`) — 2026-09-15; committed `splits/acdc_fold*.json` are **CI/smoke placeholders only** (`synthetic_placeholder: true`; regenerate on licensed ACDC)
- [x] Multi-seed harness (`scripts/run_multiseed.py`) — 2026-09-15
- [x] Publication / smoke / proxy config hygiene; deprecate `paper_*` as formal results — 2026-09-15
- [x] Physical sparse SA (`slice_trans_mm` / `sampling_ratio`) — 2026-09-15
- [x] Eval `--split test` + phase indices/spacing + results artifacts — 2026-09-15
- [x] ConvLSTM / temporal attention configs + SVF publication variant — 2026-09-15
- [x] Baseline adapters (recon-only, compact VoxelMorph, FlowReg interface) — 2026-09-15
- [x] M&Ms loader stub + hard-fail publication config — 2026-09-15

### Externally blocked (infrastructure ready; tables 待补充)

- [ ] Full VoxelMorph / TransMorph / MulViMotion / FlowReg **numbers** on licensed ACDC (adapters exist; need data + optional external install)
- [ ] M&Ms download + multi-site **tables** (loader + hard-fail ready; need credentials/download)
- [ ] Real ACDC subject-level publication tables (need CREATIS / challenge license mount)
- [ ] Full 256³ paper-scale training runs (needs GPU + data; `publication_recon_hires` is `[64,128,128]` intermediate only; optional `publication_recon_256.yaml` stub — **not** claimed Done)
- [ ] Nested CV / calibration for any future survival claims (not main line)

## Demo paper pipeline (no private AMI)

```powershell
cd E:\Projects\20260523-ReconSeg3D-cardiac-rec
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python scripts/prepare_demo_data.py
python scripts/train.py --config configs/smoke/smoke_motion.yaml --epochs 1 --output-dir outputs/smoke_motion
python scripts/run_multiseed.py --config configs/smoke/smoke_motion.yaml --seeds 42 43 44 --epochs 1 --dry-run
```

Publication configs with `allow_fake_data: false` **hard-error** if `source: synthetic` or if ACDC/MM-WHS/EMIDEC/M&Ms roots are empty. Use smoke configs for synthetic CI.

Eval writes patient-level artifacts:

```powershell
python scripts/eval.py --config configs/smoke_motion.yaml --checkpoint outputs/smoke_motion/best.pt --split test
# → results/case_metrics.csv, summary_metrics.json, bootstrap_ci.json
```

## Honest gaps

- No original weights, no private AMI cohort.
- Global SSIM remains a **proxy** (`recon_ssim_proxy` only — no `recon_ssim` alias); prefer `ssim_3d` for tables.
- HD95 uses batch `spacing` `(sz,sy,sx)` mm when present; demo NIfTIs often unit spacing.
- Fake/demo metrics remain **DEMO-labeled** — do not invent real-table numbers.
- Volume-curve is a light physiological regularizer, not a primary motion claim.
