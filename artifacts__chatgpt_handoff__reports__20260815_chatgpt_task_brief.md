# ChatGPT task brief — ReconSeg3D motion-consistent 4D (public verification)

You cannot see the local filesystem. A source ZIP is attached (or will be attached). Reason only from the ZIP + this brief. Do **not** invent clinical results.

## Background / goal

Continue advancing **ReconSeg3D**: motion-consistent **4D** extension of sparse SA → dense cine recon + multi-structure seg + risk/phenotype heads.  
**Public verification only** (synthetic / fake or real ACDC / MM-WHS / EMIDEC).  
This is a research re-implementation + extension — **not** a claim of the original private-AMI paper numbers.

**Forbidden:** claim private MACE AUC **0.934**; invent clinical results; include secrets; suggest git push.

## Architecture boundaries (must preserve)

- Canonical tensor layout: **`(B, C, T, D, H, W)`**
- Default: `model.per_frame_recon: true` (+ motion when enabled)
- Broadcast ablation still allowed via `per_frame_recon: false`
- Windows: `data.num_workers: 0`
- Do **not** destroy existing pytest suite
- Trainer currently supports `model.arch: reconseg3d` only

## Baseline already verified by Cursor (independent)

- pytest: **33 passed** with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
- Recent local Cursor patches (already in ZIP if dated `_2205`):
  - Epoch-pooled AUC / C-index / phenotype metrics (no mean-of-batch-AUC)
  - Task-aware `best.pt` selection
  - Data-range-aware global SSIM stabilizers
- Smoke train 1 epoch: finite losses (no NaN totals); `loss_volsmooth` can still dominate early

## Scope

1. Review ZIP code vs `docs/PAPER_PLAN.md` honesty gaps.
2. Identify remaining **bugs / gaps** (trainer edge cases, metrics honesty, ablation holes, data loaders, motion losses).
3. Propose **minimal** patches (unified diffs or full file contents for changed files only).
4. Improve weak spots: tests, ablations, metrics honesty, trainer edge cases — without expanding scope into private AMI claims.

### Known open issues for your audit (Cursor-observed)

1. `w_volsmooth` / volume-curve loss can explode in early smoke (~100s) — normalize or retune?
2. Full ablation matrix (6 configs) vs partial smoke tables.
3. Real ACDC path untested here; fake ACDC only.
4. HeartTTable clinical features remain dummy.
5. Nested CV / calibration / paper-scale grids not implemented.
6. Confirm no regressions to epoch-pooled ranking metrics after any further edits.

## Deliverables (acceptance-ready)

1. **Written audit report** (bugs with file:line, gaps vs PAPER_PLAN, risks).
2. **Unified diffs** or full contents for every changed file.
3. **Verification commands** (Windows PowerShell preferred).
4. **What remains blocked** without real ACDC / private AMI.

### Required verification commands

```powershell
cd <unzipped_repo>
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest tests/ -v
python scripts/train.py --config configs/default.yaml --epochs 1 --output-dir outputs/smoke_chatgpt
```

Acceptance: patches apply cleanly; tests pass; smoke train has **no NaN** in total loss; docs updated if behavior changes.

## Out of scope

- Claiming private AMI MACE validation / 0.934 AUC
- Git commit/push/deploy
- Expanding secrets or credentials
- Large refactors that break `(B,C,T,D,H,W)` or Windows `num_workers=0`
