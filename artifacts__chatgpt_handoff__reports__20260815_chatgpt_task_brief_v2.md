# ChatGPT task brief — ReconSeg3D (v2, post local fixes)

You cannot see the local filesystem unless a ZIP is attached. Reason from this brief + any attached ZIP. Do **not** invent clinical results.

## ZIP baseline (if attached)

- File: `reconseg3d_src_nogit_20260815_2222.zip`
- Size: 77729 bytes
- SHA-256: `D7830799AC8C24E10320227C500D63F6ED3E60AD9C33B97EF8E9C3C218418C78`
- No git repo in this workspace (dirty/local only). Secret-scanned before packaging.

## Background / goal

Continue advancing **ReconSeg3D**: motion-consistent **4D** extension (sparse SA → dense cine recon + multi-structure seg + risk/phenotype heads).

**Public verification only** (synthetic / fake ACDC / MM-WHS / EMIDEC). Research re-implementation + extension — **not** a claim of the original private-AMI paper numbers.

**Forbidden:** claim private MACE AUC **0.934**; invent clinical results; include secrets; suggest git push/PR/deploy.

## Architecture boundaries (must preserve)

- Canonical tensor layout: **`(B, C, T, D, H, W)`**
- Default: `model.per_frame_recon: true` (+ motion when enabled)
- Broadcast ablation via `per_frame_recon: false`
- Windows: `data.num_workers: 0`
- Do **not** destroy existing pytest suite
- Trainer currently supports `model.arch: reconseg3d` only

## Already verified by Cursor (independent) — treat as baseline

- pytest: **34 passed** with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
- Full ablation smoke **6/6** under `outputs/ablations_smoke_v2/`
- Smoke train: finite losses; `loss_volsmooth` was fixed via **fractional LV/RV volumes (/DHW)** — total loss ~19→~3.7 on smoke
- Epoch-pooled AUC / C-index / phenotype (not mean-of-batch-AUC)
- Task-aware `best.pt`; eval preserves `loss_total`; data-range SSIM stabilizers

## Scope for this ChatGPT session

1. Deep audit of ZIP vs `docs/PAPER_PLAN.md` honesty gaps.
2. Find remaining **bugs / gaps** (trainer edge cases, metrics honesty, data loaders, motion losses, HeartTTable stubs, paper-scale configs).
3. Propose **minimal** patches as unified diffs or full file contents for changed files only.
4. Prioritize highest-ROI next steps for a publishable public-data paper path (ACDC/MM-WHS/EMIDEC), not private AMI.

### Still open (Cursor-known)

1. Real ACDC / MM-WHS / EMIDEC paths untested (fake layouts only).
2. HeartTTable clinical features remain dummy dictionary.
3. Nested CV / calibration / paper-scale grids (256×256×128) not implemented.
4. Compact CNN/UNet ≠ original 3D ViT / nnU-Net — document honestly.
5. HD95 / SSIM remain approximate proxies.
6. Any regressions you spot in motion/warp/cycle, Cox NaN safety, phenotype head wiring.

## Deliverables (acceptance-ready)

1. **Written audit report** (bugs with file:line, gaps vs PAPER_PLAN, risks).
2. **Unified diffs** or full contents for every changed file.
3. **Verification commands** (Windows PowerShell).
4. **What remains blocked** without real ACDC / private AMI.

### Required verification commands (Cursor will re-run)

```powershell
cd <unzipped_or_repo>
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest tests/ -v
python scripts/train.py --config configs/default.yaml --epochs 1 --output-dir outputs/smoke_chatgpt
```

Acceptance: patches apply cleanly; tests pass; smoke train **no NaN** in total loss; docs updated if behavior changes.

## Out of scope

- Claiming private AMI MACE validation / 0.934 AUC
- Git commit/push/deploy
- Large refactors that break `(B,C,T,D,H,W)` or Windows `num_workers=0`
