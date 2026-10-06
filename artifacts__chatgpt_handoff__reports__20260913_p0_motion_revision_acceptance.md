# §十九 Acceptance — P0 motion revision (2026-09-13)

## Verdict

P0 geometry/motion revision implemented and verified locally (`58 passed`). Main paper path reframed as **geometry- and motion-constrained 4D from sparse SA**; MACE/Cox decoupled. No private-AMI / 0.934 claims. Demo metrics remain DEMO-labeled.

## Implemented vs deferred

| ID | Item | Status |
|----|------|--------|
| P0-1 | `warp_vector`, `compose_pull`, true `L_inv` | **Done** |
| P0-2 | `L_smooth`, `ReLU(ε-detJ)`, jac stats in metrics | **Done**; SVF stub (`use_svf`) + PAPER_PLAN TODO |
| P0-3 | `L_loop` adjacent full-cycle | **Done**; ED-anchored path documented TODO |
| P0-4 | Demote `volume_curve` / docs / pub weights | **Done** |
| P0-5 | Kill silent `t//2`; ED/ES batch fields + labeled-phase loss | **Done** |
| P0-6 | ED↔ES label propagation Dice/HD95 API + synthetic tests | **Done**; real ACDC tables **待补充** |
| P0-6b | Physical-spacing HD95; SSIM named proxy | **Done** |
| P0-7 | `publication_*.yaml` with `w_mace/w_cox=0`; FocalLoss α_t fix | **Done** |
| Docs | PAPER_PLAN, MANUSCRIPT_DRAFT, README, LICENSE, CITATION.cff | **Done** |
| Eng | `/data/` gitignore fix (was hiding `reconseg3d/data/`); `allow_fake_data:false` hard error | **Done** |
| — | VoxelMorph/TransMorph/MulViMotion baselines | **Deferred** (PAPER_PLAN) |
| — | M&Ms download / 5-seed CI | **Deferred** |
| — | Full SVF ablation enabled | **Deferred** (stub only) |

## Tests

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v
============================= 58 passed in ~17s =============================
```

New: `tests/test_ed_es.py`, `tests/test_label_propagation.py`; expanded `tests/test_motion.py`.

## ChatGPT rounds

| Round | Brief (GitHub path after push) | Browser |
|-------|--------------------------------|---------|
| 1 | `artifacts/chatgpt_handoff/github_briefs/20260913_inv_consistency_review.md` | Browser MCP had no open tabs; **independent implementation** from review |
| 2 | `artifacts/chatgpt_handoff/github_briefs/20260913_ed_es_experiment_matrix.md` | Same — briefs pushed for advisor paste |

Advisor is not ground truth; code + tests are.

## Honesty on data

- Synthetic / fake ACDC used for unit tests and DEMO pipeline only.
- `configs/publication_seg.yaml` / `publication_recon.yaml` set `allow_fake_data: false` (hard error if public roots empty).
- No invented real-table Dice/HD95/AUC.
- No ZIP uploads; no private AMI redistribution.

## GitHub

- Repo: https://github.com/Coucou2016/ReconSeg3D-cardiac-rec
- Commit: [`6e2ed9a`](https://github.com/Coucou2016/ReconSeg3D-cardiac-rec/commit/6e2ed9a) on `main`
- Brief URLs for ChatGPT paste:
  - https://github.com/Coucou2016/ReconSeg3D-cardiac-rec/blob/main/artifacts/chatgpt_handoff/github_briefs/20260913_inv_consistency_review.md
  - https://github.com/Coucou2016/ReconSeg3D-cardiac-rec/blob/main/artifacts/chatgpt_handoff/github_briefs/20260913_ed_es_experiment_matrix.md
- Critical fix this turn: `.gitignore` `data/` → `/data/` so `reconseg3d/data/` is versioned.
