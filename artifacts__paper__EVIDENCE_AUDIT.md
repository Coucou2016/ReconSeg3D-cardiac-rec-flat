# Evidence authenticity audit

**Purpose:** One-to-one mapping between manuscript / report numbers and local measured artefacts.  
**Date:** 2026-09-15  
**Git HEAD (at audit authoring):** 055bd3efeae3435b25b0e40a84e11de4584c2c93  
**User manuscript:** **Not found** — see `docs/paper/AWAITING_USER_MANUSCRIPT.md`. Interim spine = `docs/paper/MANUSCRIPT_DRAFT.md`.

---

## 1. Hard non-claims (never taken from original npj paper)

| Quantity | Original paper (Gao et al., *npj Digit. Med.*) | This repo |
|----------|-----------------------------------------------|-----------|
| 5-year time-dependent AUC **0.934** | Private AMI external test | **Not used** |
| C-index **0.897** | Private AMI | **Not used** |
| Cohort n≈4511 / external n≈1029 | Private | **Not used** |
| Full HeartTTable three-modality pairwise cross-attn | Clinical product | **Not claimed** (HeartTTable-lite ablation only) |
| ACDC/MM-WHS “challenge leaderboard” Dice | Training sets for ReconSeg3D in original | **No invented ACDC tables** |

Any appearance of 0.934 / 0.897 in prose is **boundary language only** (what we refuse to claim).

---

## 2. DEMO vs real labels

| Artefact | Label | How decided |
|----------|-------|-------------|
| `outputs/ablations_smoke_v2/table.csv` | **DEMO** | Short smoke on fake/demo NIfTI |
| `outputs/metrics.json` | **DEMO** | Joint eval on demo tensors (geometry keys present) |
| `outputs/paper_recon_smoke/metrics.json` | **DEMO** | Named paper smoke run |
| `outputs/paper_acdc_smoke/metrics.json` | **DEMO** | Local ACDC-**named** placeholder tree, not CREATIS |
| `outputs/post_fix_smoke/metrics.json` | **DEMO** | Pre-geometry-key logging era (legacy) |
| `data/acdc/`, `data/mmwhs/`, `data/emidec/` | **DEMO placeholders** | Tiny synthetic volumes |
| Licensed ACDC subject CV tables | **待补充** | No licensed mount |
| Private AMI MACE | **Out of scope** | No data |

---

## 3. Numbers in manuscript §5 → source files → code entry points

### 3.1 Ablation table (PSNR / MAE / Dice mean)

| MS cell | Source | Computing path |
|---------|--------|----------------|
| broadcast / PF±motion / fusion rows | `outputs/ablations_smoke_v2/table.csv` | `scripts/run_ablations.py` → `scripts/train.py` / eval loop → `reconseg3d` metrics |
| Fig. 2 bars | same CSV via `scripts/make_paper_figures.py` | `fig2_recon_ablation` |
| Fig. 4 heatmap | same CSV | `fig4_seg_dice` |

Exact CSV values (rounded in MS):

```
broadcast_baseline: PSNR=19.075, MAE=0.6926, dice_mean=0.1682
per_frame_no_motion: PSNR=19.151, MAE=0.6919, dice_mean=0.1696
per_frame_motion: PSNR=19.148, MAE=0.6953, dice_mean=0.1588
fusion_concat: PSNR=19.093, MAE=0.6922, dice_mean=0.1394
fusion_heart_ttable: PSNR=19.260, MAE=0.6895, dice_mean=0.1890
```

### 3.2 Geometry logs (`inv_error`, Jac, loop, SSIM)

| MS metric | Source key | File | Code |
|-----------|------------|------|------|
| \(L_\mathrm{inv}\) | `inv_error` / `loss_inv` | `outputs/metrics.json` | `reconseg3d/models/motion.py::inverse_consistency_loss` composed in `reconseg3d/models/losses.py` |
| \(L_\mathrm{loop}\) | `loss_loop` | same | `motion.py` loop helpers + `losses.py` (`w_loop`) |
| \(L_\mathrm{smooth}\) | `loss_smooth` | same | `motion.py::smoothness_loss` |
| \(L_\mathrm{jac}\) / `jac_neg_ratio` | `loss_jac`, `jac_neg_ratio`, `jac_det_*` | same | `motion.py::folding_penalty`, `jacobian_stats` |
| Image-cycle | `cycle_error` / `loss_cycle` | same | intensity cycle aux in `motion.py` / `losses.py` (**≠** \(L_\mathrm{inv}\)) |
| `ssim_3d` | `ssim_3d` | same | metrics module (windowed) |
| `recon_ssim_proxy` | `recon_ssim_proxy` | same | global proxy; **not** aliased as `recon_ssim` in publication path |
| Fig. 3b | same JSON | `make_paper_figures.py::fig3_motion_metrics` |

Recorded DEMO values (from `outputs/metrics.json` at audit time):

```
inv_error=0.0017687, loss_loop=0.0017683, loss_smooth≈6.13e-8,
loss_jac=0.0, jac_neg_ratio=0.0, jac_det_mean=0.99982,
cycle_error=2.34e-5, recon_psnr=18.795, ssim_3d=0.5149,
recon_ssim_proxy=0.00977
```

### 3.3 Pipeline checkpoints

| MS row | File | Notes |
|--------|------|-------|
| paper_recon_smoke | `outputs/paper_recon_smoke/metrics.json` | PSNR≈19.027, dice_mean≈0.0480 |
| paper_acdc_smoke | `outputs/paper_acdc_smoke/metrics.json` | PSNR≈17.390; **placeholder ACDC tree** |

Eval entry: `scripts/eval.py`.

### 3.4 Figures

| Figure | Generator | Data |
|--------|-----------|------|
| Fig. 1 schematic | `scripts/make_paper_figures.py` | Non-quantitative |
| Fig. 2–4 | same | `ablations_smoke_v2/table.csv` (+ Fig. 3b geometry JSON) |
| Fig. 5 claim boundary | same | Qualitative ledger |
| Outputs | `artifacts/paper/figures/*.{png,pdf,svg}` | SciencePlots `science`+`nature`+`no-latex`; Times New Roman |

Meta: `artifacts/paper/figures/figure_meta.json`.

---

## 4. Commands used / expected for regeneration

```text
# Figures (SciencePlots already installed: SciencePlots 2.2.2)
python scripts/make_paper_figures.py

# Self-contained report + manuscript HTML/PDF
python scripts/build_report_bundle.py

# Ablations (DEMO; long) — do not invent rows if skipped
# python scripts/run_ablations.py ...

# Unit tests (geometry)
python -m pytest tests/test_motion.py tests/test_metrics.py tests/test_label_propagation.py -q
```

**Test status (this session):** Key geometry suites green — `tests/test_motion.py` + `test_metrics.py` + `test_label_propagation.py` → **30 passed** (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `--assert=plain`). Full-suite collection can still hit host `numcodecs`/`zarr` noise; historical roadmap closure reported **96 passed** (`artifacts/chatgpt_handoff/reports/20260915_full_roadmap_closure.md`).

---

## 5. Paper vs report separation

| Deliverable | Paths | Rules |
|-------------|-------|-------|
| Academic paper | `docs/paper/MANUSCRIPT_DRAFT.md`, `docs/paper/manuscript.html`, `artifacts/paper/*` | No `E:\` paths; DEMO labels; no session/rebuttal tone |
| Process report | `artifacts/report/report.{html,md,pdf}` | Full 来龙去脉, workspace paths, dual-agent notes OK |
| This audit | `docs/paper/EVIDENCE_AUDIT.md` (+ copy under `artifacts/paper/` if synced) | Authenticity ledger |

---

## 6. Sync checklist

- [x] MS tables cite only DEMO sources listed above  
- [x] Figs regenerated from same CSV/JSON  
- [x] Original 0.934 / 0.897 excluded from results  
- [ ] User mainline manuscript path provided → re-align section order  
- [ ] Licensed ACDC subject tables  
- [ ] Full pytest green on clean env  
