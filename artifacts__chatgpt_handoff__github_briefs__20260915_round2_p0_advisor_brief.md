# Advisor brief — Round-2 P0 credibility (2026-09-15)

**Ask:** Confirm these P0s are sufficient before drafting formal ACDC tables.

## Diff (blunt)

1. **Leakage:** Clinical inputs = height/weight/nb only (or off). Phenotype is target-only.
2. **Temporal:** Subsample always keeps original ED/ES frames (index remap via `.index`).
3. **Loop:** MotionNet predicts T pairs including `T-1→0`; `L_periodic` needs that closing edge (`+1+1+1−3=0` synthetic).
4. **Metrics:** ED↔ES prop Dice/HD95 aggregated **per patient**, batch-invariant.
5. **Selection:** Publication checkpoints use task metrics (prop Dice / Dice / recon MAE), not `mace_auc` with `w_mace:0`.
6. **Eval:** Phase indices + spacing flow into model/metrics.
7. **Honesty:** No 0.934; no invented real ACDC tables; `paper_*` ≠ formal results.

**Evidence:** `artifacts/chatgpt_handoff/reports/20260915_round2_p0_fix.md`  
**Tests:** 75 passed.

## Still deferred (do not claim)

Windowed 3D SSIM · 5-fold lists · FlowReg/VoxelMorph baselines · M&Ms · 256³ · real ED↔ES tables.
