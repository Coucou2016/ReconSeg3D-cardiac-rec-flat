# Round-2 P0 Fix Report (2026-09-15)

Major-revision **result-credibility P0s** before formal paper tables.  
No fabricated ACDC real-table numbers. No 0.934 / private AMI claims. Demo metrics stay DEMO-labeled.

**P0 implementation SHA:** `c1fe909`  
**Report / README honesty on `main`:** starts `dd65948` (tip after push: check `git rev-parse HEAD`)
**Tests:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v` → **75 passed**

| ID | Item | Status | Evidence |
|----|------|--------|----------|
| **P0-1** | Phenotype target leakage | **PASS** | `ACDC_CLINICAL_FEATURES = (height, weight, nb_frame)` only; phenotype only in targets. Publication configs `clinical_dim: 0`. Proxy: `configs/proxy/proxy_acdc_phenotype.yaml` (`clinical_dim: 3`); shim `configs/paper_acdc.yaml`. Tests: `test_acdc_clinical_no_phenotype_leakage` |
| **P0-2** | Anchor-preserving temporal subsample | **PASS** | `select_time_indices_keep_anchors` / `subsample_time_keep_anchors` in `reconseg3d/data/io.py`; ACDC remaps via `selected.index(ed_orig)`. Test: `test_temporal_subsample_preserves_ed_es` (ED/ES retained; GT phase matches image) |
| **P0-3** | True closed-cycle `L_loop` | **PASS** | MotionNet emits **T** pairs incl. closing `T-1→0` (+ matching bwd). `loop_consistency_loss` / `L_periodic` composes all T edges. Synthetic `+1+1+1−3=0` only with closing edge: `test_closed_cycle_periodic_only_with_closing_edge`. Docs forbid calling adjacent-only a “full cycle” |
| **P0-4** | Per-patient ED/ES propagation metrics | **PASS** | `ed_es_label_propagation_metrics` + `compute_metrics` loop per sample `b` (never `ed_index.reshape(-1)[0]` for whole batch). Test: `test_motion_metrics_batch_invariant` (B=1/2/4 aggregates &lt;1e-6) |
| **P0-5** | Task-specific checkpoint selection | **PASS** | `selection: {metric, mode}`; `Trainer._resolve_selection` defaults by task. Publication: motion→`prop_ed2es_dice_mean` max; seg→`dice_mean` max; recon→`recon_mae` min. Honest `task:` (`motion`/`segmentation`/`reconstruction`). Test: `test_checkpoint_selection_motion_vs_segmentation` |
| **P0-6** | Eval passes phase indices | **PASS** | `scripts/eval.py` + `Predictor.predict_batch` pass `seg_frame_indices`; trainer already did. `--split test` → val-style held-out + TODO 5-fold warning |
| **P0-7** | Spacing/affine to metrics | **PASS** | `load_nifti(..., with_meta=True)` → `NiftiVolume` (affine + spacing); sample `spacing`/`affine`; `scale_spacing_dhw` after resize; wired into HD95 / propagation. Test: `test_hd95_spacing_anisotropic` (1 vx → ~1 mm vs ~8 mm for spacing `(1,1,1)` vs `(8,1,1)`). **No invented mm paper tables** without real data |

### Also (post-P0 hygiene)

- Dropped `metrics["recon_ssim"]` alias; keep **`recon_ssim_proxy` only** (`test_metrics` asserts `"recon_ssim" not in m`).
- README + `docs/PAPER_PLAN.md`: list inv/smooth/jac/ed_ref/true closed cycle; point to publication configs; deprecate `paper_*` as formal results.
- Config layout: `configs/publication/`, `configs/proxy/`, `configs/smoke/` + root shims (paths not broken).

### Session note (2026-09-15)

Working-tree had a **partial revert** of P0-3 (`motion.py` / `test_motion.py` / `PAPER_PLAN.md` back to open T−1 “full-cycle” wording). Restored from HEAD before re-running tests. Do not reintroduce adjacent-only compose as a full cycle.

---

## Remaining TODOs (deferred — not blocking P0)

- [ ] Windowed **3D SSIM** (if claiming SSIM in main tables)
- [ ] Official **5-fold** CV file lists / true test split
- [ ] **FlowReg** / VoxelMorph / TransMorph / MulViMotion trained baselines on licensed ACDC
- [ ] **M&Ms** multi-site tables
- [ ] Paper-scale **resolution** (e.g. 256³ / 128×256×256) runs
- [ ] ConvLSTM temporal experiments
- [ ] Real ACDC subject-level ED↔ES tables (**待补充**, data-blocked)
- [ ] SVF enable/ablation (`use_svf: true`) beyond stub

## Honesty

- No private AMI AUC / **0.934** claims.
- No fabricated ACDC formal-table numbers.
- Demo / fake metrics remain **DEMO-labeled**.
- `paper_*` configs = demo/proxy shims; formal line = `configs/publication_*.yaml`.
