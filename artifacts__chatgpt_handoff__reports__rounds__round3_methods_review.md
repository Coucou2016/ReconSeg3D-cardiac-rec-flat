# Round 3 — Methods section review

**Status:** `ChatGPT live blocked` — review against prior Plus audit + local code paths.  
**Public GitHub:** https://github.com/Coucou2016/ReconSeg3D-cardiac-rec  
**Key paths for advisor:** `reconseg3d/models/{reconseg3d,motion,heart_ttable,losses}.py`, `docs/DATA.md`, `configs/ablations/*`

## Prompt summary (ready paste)

```
Web search optional. Repo: https://github.com/Coucou2016/ReconSeg3D-cardiac-rec
Round 3 — Methods review. Paste below is §3 of our draft (summarized):
- Tensor (B,C,T,D,H,W); sparse SA via slice mask
- Per-frame 3D recon (not broadcast)
- MotionNet (dz,dy,dx); Warp L1; Image-cycle = intensity cycle after F/B warp (NOT inverse-consistent flow composition)
- Seg UNet; phenotype/BCE/Cox heads
- HeartTTable-lite: single CLS over concat spatial/temporal/table KV
Critique for reviewer traps. Suggest precise wording fixes only—no new fake equations.
```

## Substitute reply gist

Prior Plus: name image-cycle accurately; guard singleton axes; Cox shared finite mask; do not claim 3-CLS. Local verify: manuscript already distinguishes image-cycle; tighten Methods with equation-level naming and config pointers; state smoke grid sizes explicitly.

## Accepted / rejected

| Advice | Decision |
|--------|----------|
| Image-cycle ≠ inverse-consistent flow wording | **Accepted** — reinforced |
| Point to GitHub module paths in Methods | **Accepted** |
| Add unrolled iterative recon–motion–seg like Qian | **Rejected** (scope creep; not implemented) |
| Claim full HeartTTable | **Rejected** |

## Files changed

- `docs/paper/MANUSCRIPT_DRAFT.md` Methods polish
- This note
