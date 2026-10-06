# §十九 Acceptance — P0 gap closure (2026-09-14)

## Verdict

P0 completeness gaps that were still open after the 2026-09-13 motion revision are **closed for implementable items**. Real ACDC subject-level tables remain **待补充** (no licensed data → no invented numbers). No private-AMI / 0.934 claims.

## Done vs 待补充

| ID | Item | Status |
|----|------|--------|
| G1 | ED-reference motion path (compose adjacent → ED; `L_ed_ref` / `w_ed_ref`) | **Done** — `ed_anchored_paths`, `ed_reference_consistency_loss`; wired in `MultiTaskLoss` + publication/smoke/default configs |
| G2 | Publication fake-data hard gate | **Done** — `source: synthetic` + `allow_fake_data: false` raises; empty ACDC/MM-WHS/EMIDEC roots raise; `publication_recon/motion/seg` all forbid fake; `configs/smoke_motion.yaml` allows synthetic |
| G3 | Stale docs (`WRITING_FRAMEWORK`, report lead story) | **Done** — geometry/motion + true \(L_\mathrm{inv}\); image-cycle auxiliary; artifacts mirror updated; report bundle strings refreshed |
| G4 | ACDC multi-phase aug (ED **and** ES co-transform) | **Done** — `apply_train_transforms(..., extra_masks=)` |
| G5 | Tests (ED-ref, fake gate, ACDC aug) | **Done** — `tests/test_motion.py` + `tests/test_publication_gates.py` |
| — | Real ACDC ED↔ES subject-level tables | **待补充** (API + synthetic tests only) |
| — | SVF full ablation | **待补充** (stub `use_svf` remains) |
| — | VoxelMorph / TransMorph / MulViMotion baselines | **待补充** |
| — | M&Ms + 5-seed CI | **待补充** |

## Prior P0 (2026-09-13) still holds

Adjacent \(L_\mathrm{inv}\) / smooth / jac / loop, ED/ES labeled-phase loss, label-propagation API — unchanged and still **Done**.

## Tests

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v
============================= 69 passed in ~16s =============================
```

New/expanded: ED-ref tests in `tests/test_motion.py`; `tests/test_publication_gates.py` (fake gate + ACDC multi-phase aug).

## Honesty on data

- Synthetic / fake ACDC only for unit tests and `smoke_motion` / `paper_recon` / ablations.
- Publication configs hard-error on synthetic or empty public roots.
- No invented real-table Dice/HD95/AUC.
- No private AMI redistribution.

## GitHub

- Repo: https://github.com/Coucou2016/ReconSeg3D-cardiac-rec
- Commit: [`392f538`](https://github.com/Coucou2016/ReconSeg3D-cardiac-rec/commit/392f538) on `main`
- Acceptance: https://github.com/Coucou2016/ReconSeg3D-cardiac-rec/blob/main/artifacts/chatgpt_handoff/reports/20260914_p0_gap_closure.md
