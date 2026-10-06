# Full roadmap closure (2026-09-15)

Exhaustive checklist for round-1/round-2 leftovers + PAPER_PLAN Stage-2/3 items.
Honesty: no invented clinical/table numbers; licensed ACDC/M&Ms subject tables remain **待补充**.

## Audit follow-up (same day)

Independent audit found real gaps; closed in a subsequent commit on `main`:

| Gap | Fix |
|-----|-----|
| B5 overclaim on committed folds | Committed `splits/acdc_fold*.json` marked **CI/smoke placeholders** (`synthetic_placeholder: true`, `splits/README.md`). Publication configs use `fold: null`. Loader hard-fails empty `train` when `allow_fake_data: false`. `write_acdc_folds` refuses degenerate folds unless `allow_degenerate=True`. |
| `edv_ml`/`esv_ml` without spacing | Without spacing → `edv_vox` / `esv_vox` / `ef_proxy` only (never `*_ml`). |
| case CSV duplicated batch-mean | `scripts/eval.py` recomputes per-sample metrics (default `batch_size=1`). |
| hires ≠ 256³ | Documented: `publication_recon_hires` = `[64,128,128]` intermediate; optional `publication_recon_256.yaml` stub — **not** Done. |

## Tests

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v
============================= 105 passed =============================
```

Audit-gap fix commit: `678c27b` on `main`.

## A. Tiny leftovers

| ID | Item | Status |
|----|------|--------|
| A1 | `MultiTaskLoss` docstring: `w_loop` = closed-cycle / `L_periodic` | **Done** |
| A2 | Clean misleading `task: mace` where unused; honest task + selection | **Done** (default/smoke/paper_recon/ablations; HeartTTable keeps `mace` when `w_mace>0`) |
| A3 | Remove dangerous `recon_ssim` alias; only `recon_ssim_proxy` / `ssim_3d` | **Done** (metrics + figure_meta rename) |

## B. Stage-2 evaluation infrastructure

| ID | Item | Status |
|----|------|--------|
| B1 | 3D local/windowed SSIM → `ssim_3d` + tests | **Done** |
| B2 | Physical EDV/ESV/EF (mL/%) via spacing + ED/ES; proxy labeled | **Done** (no `*_ml` without spacing) |
| B3 | Patient-level pipeline → `case_metrics.csv` / `summary_metrics.json` / `bootstrap_ci.json` | **Done** (`scripts/eval.py`; true per-case rows) |
| B4 | Trainer sample-weighted aggregation (not naive batch-mean) | **Done** |
| B5 | 5-fold diagnosis-stratified ACDC split **API** + loader | **Done (API)** — committed JSON are **placeholders only**, not publication CV |
| B6 | Multi-seed harness (3–5 seeds) | **Done** (`scripts/run_multiseed.py`) |
| B7 | Publication vs smoke vs proxy config hygiene; hires configs; deprecate `paper_*` as formal | **Done** (hires honesty: `[64,128,128]` ≠ 256³) |
| B8 | Physical sparse SA (`slice_trans_mm`, `sampling_ratio`) | **Done** |
| B9 | Eval test split + phase indices/spacing | **Done** |

## C. Stage-3 methods / baselines

| ID | Item | Status |
|----|------|--------|
| C1 | Temporal backends: TemporalConv / ConvLSTM3D / temporal attention + configs | **Done** |
| C2 | SVF usable (`use_svf` + scaling-and-squaring; `publication_motion_svf.yaml`) | **Done** |
| C3a | Recon-only / no-motion baseline | **Done** |
| C3b | Compact VoxelMorph-style in-repo | **Done** |
| C3c | FlowReg adapter + README (not vendored; no fake numbers) | **Done** (`docs/BASELINES.md`) |
| C4 | Related-work positioning locked (Neural ODE, TetHeart, FlowReg, VoxelMorph, TransMorph) | **Done** |
| C5 | M&Ms loader stub + publication hard-fail + download docs | **Done** |

## D. Docs / hygiene

| ID | Item | Status |
|----|------|--------|
| D1 | README architecture sync | **Done** |
| D2 | PAPER_PLAN Done vs externally blocked | **Done** |
| D3 | LICENSE/CITATION | **OK** (unchanged; still accurate) |
| D4 | SciencePlots figure_meta metrics key rename | **Done** (historical DEMO values; `ssim_3d` null until re-run) |
| D5 | ChatGPT brief MD | **Done** (see github_briefs) |

## E. Tests & push

| ID | Item | Status |
|----|------|--------|
| E1 | Expanded tests (SSIM3D, EF mL, folds, aggregate, docstring, selection) | **Done** |
| E2 | Full pytest green | **Done** (105 passed) |
| E3 | Commit + push (code/docs; no huge data) | **Done** — audit-gap fix `678c27b` on `main` |

## Externally blocked (infra complete; fail loud / 待补充)

| Item | Reason |
|------|--------|
| Real ACDC subject-level publication tables | CREATIS / challenge license + multi-GB download not on disk |
| Real publication 5-fold subject lists | Need licensed ACDC + `make_acdc_folds.py` (committed lists are placeholders) |
| M&Ms multi-site tables | Download wall / credentials; loader hard-fails in publication mode |
| FlowReg / TransMorph / MulViMotion **numeric** comparisons | Need external install + licensed data; adapters/docs only |
| Full 256³ training wall-clock | GPU + licensed data; hires = `[64,128,128]` only; `publication_recon_256.yaml` is an optional stub — **not Done** |
| Private AMI / 0.934 | Out of scope — never claimed |

## Config map (honest)

| Role | Path |
|------|------|
| Smoke/CI | `configs/smoke/smoke_motion.yaml` |
| Publication | `configs/publication/*.yaml` (+ root shims); `fold: null` until real folds |
| Hires (intermediate) | `publication_recon_hires.yaml` → `[64,128,128]` |
| Paper-scale stub | `publication_recon_256.yaml` → `[128,256,256]` (VRAM-gated; not Done) |
| Proxy phenotype | `configs/proxy/proxy_acdc_phenotype.yaml` |
| DEMO shim | `configs/paper_recon.yaml`, `paper_acdc.yaml` — **not** formal results |
| Baselines | `configs/baselines/` |
| Temporal ablations | `configs/ablations/temporal_*.yaml` |

## Non-claims

- No 0.934 / private AMI.
- No invented Dice/HD95/EF on missing licensed data.
- No publication 5-fold tables from committed placeholder JSON.
- No 256³ wall-clock from hires config.
- Smoke/DEMO metrics stay labeled.
