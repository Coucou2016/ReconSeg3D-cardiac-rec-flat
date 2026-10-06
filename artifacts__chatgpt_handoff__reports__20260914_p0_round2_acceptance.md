# P0 Round-2 Acceptance (2026-09-14)

Major revision **P0-1 through P0-7** before official paper tables.  
No 0.934 / private AMI claims. No invented real ACDC tables.

**Commit SHA:** `c1fe909`  
**Tests:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v` → **75 passed**

| ID | Item | Status | Notes |
|----|------|--------|-------|
| P0-1 | Phenotype leakage | **PASS** | `clinical_base` = height/weight/nb only; proxy at `configs/proxy/proxy_acdc_phenotype.yaml` (`clinical_dim: 3`); shim `configs/paper_acdc.yaml`; leakage tests |
| P0-2 | Anchor-preserving temporal sampling | **PASS** | `select_time_indices_keep_anchors` / `subsample_time_keep_anchors`; ED/ES exact indices; `test_temporal_subsample_preserves_ed_es`; docs prefer full-T + mask |
| P0-3 | True closed-cycle | **PASS** | MotionNet emits **T** pairs incl. `T-1→0`; `L_periodic`/`loop_consistency_loss` includes closing edge; synthetic `+1+1+1−3` test; docs renamed (no false “full cycle”) |
| P0-4 | Per-patient propagation metrics | **PASS** | `ed_es_label_propagation_metrics` + `compute_metrics` loop per `b`; `test_motion_metrics_batch_invariant` (B=1,2,4 within 1e-6) |
| P0-5 | Task-specific checkpoint selection | **PASS** | `selection.metric`/`mode`; publication tasks honest (`motion`/`segmentation`/`reconstruction`); defaults + motion vs seg tests |
| P0-6 | Eval phase indices | **PASS** | `scripts/eval.py` + predictor pass `seg_frame_indices`; trainer already passed indices/ED; `--split test` → val-style + TODO 5-fold |
| P0-7 | Spacing/affine to metrics | **PASS** | `load_nifti(..., with_meta=True)` → `NiftiVolume`; sample `spacing`/`affine`; resize scales spacing; HD95 1 vs 8 mm test |

### Also done (non-blocking)

- Dropped `metrics["recon_ssim"] = recon_ssim_proxy` alias; summarizer uses `recon_ssim_proxy`.
- README + `docs/PAPER_PLAN.md` synced (geometry, configs, honest loop / closed-cycle).
- Config tidy: `configs/proxy/`, `configs/publication/`, `configs/smoke/` mirrors + root shims.

### Remaining TODOs (PAPER_PLAN only — deferred)

- Windowed 3D SSIM
- Official 5-fold CV file lists / test split
- M&Ms multi-site
- FlowReg / VoxelMorph / TransMorph training baselines
- 256³ resolution runs
- ConvLSTM experiments
- Real ACDC subject-level ED↔ES tables (**待补充**, data-blocked)

### Honesty

- No private AMI AUC claims.
- No fabricated ACDC paper numbers.
- Demo / fake metrics remain DEMO-labeled.
