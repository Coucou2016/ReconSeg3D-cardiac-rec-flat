# ChatGPT Pro/Plus audit response (text-only CONTEXT)

- Conversation: [ReconSeg3D Audit Preparation](https://chatgpt.com/c/6a808651-37cc-83ea-a63d-2d2539a48d07)
- Mode: text CONTEXT packs only (no ZIP/file upload)
- Captured: 2026-08-16 after END OF CONTEXT

## Advisor summary (verbatim gist)

No confirmed P0 blocker. Main risks: methods-claim fidelity, real-data contracts, numerical edge cases.

### Prioritized findings

| Priority | Component | Finding / action |
|----------|-----------|------------------|
| High | heart_ttable + docs | Keep HeartTTable-lite naming; do not claim faithful 3-CLS pairwise cross-attn |
| High | ACDC / motion | Make real ACDC the primary motion-consistency validation (cine + ED/ES) |
| High | MM-WHS / EMIDEC | Do not imply temporal motion validation; static seg / table fusion |
| High | motion.py | Invariant tests: flow axis, voxel→grid scale, identity, ±1 voxel, cycle meaning |
| High | Cox | Shared finite mask on risk/time/event; graph-safe zero-event return |
| Medium | metrics | Physical-space HD95 before paper tables |
| Medium | phenotype/task | Per-task regression tests (pattern of HeartTTable risk-path) |
| Medium | paper protocol | Subject-level splits, calibration, CI > more architecture |

### HeartTTable decision (advisor)

Do **not** implement three modality CLS now. Novelty is motion-consistent 4D; fusion fidelity has lower ROI unless claiming “we reproduce HeartTTable.”

### Top three next code tasks (advisor)

1. Real ACDC E2E + contract tests (preserve full cine T; ED/ES only where annotated).
2. Paper-grade motion/seg eval + ablations on subject-level ACDC folds; physical HD95; warp/folding metrics.
3. Real MM-WHS/EMIDEC with motion off / T=1; split manifests; seeded CV; calibration/CI. Keep Cox tested but not as public-paper story.

### Warp / Cox notes (advisor)

- `grid_x += 2 * dx / (W-1)` (etc.), channels `(dz,dy,dx)`; guard dim==1.
- Cycle as implemented is image-cycle, not inverse-consistent flow composition — name it accurately.
- Cox: shared `isfinite` mask; `return risk.sum() * 0.0` for zero events.

## Cursor independent decisions

| Advisor item | Decision |
|--------------|----------|
| Keep HeartTTable-lite | **Accepted** (already documented + risk_logits path) |
| Do not build 3-CLS now | **Accepted** |
| Cox shared mask + graph-safe zero | **Implemented** in `losses.py` |
| Motion invariant tests + docs | **Implemented** in `tests/test_motion.py` + docstrings / PAPER_PLAN |
| Real ACDC primary validation | **Accepted as plan**; **blocked** until real ACDC on disk |
| Physical HD95 | **Deferred** (medium; smoke HD95 remains approximate) |
| Nested CV / calibration | **Deferred** (protocol work after real data) |
