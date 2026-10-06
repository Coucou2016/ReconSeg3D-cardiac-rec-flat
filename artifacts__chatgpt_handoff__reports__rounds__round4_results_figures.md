# Round 4 — Results & figures honesty

**Status:** `ChatGPT live blocked` — local data inventory + measured smoke metrics.  
**Public GitHub:** https://github.com/Coucou2016/ReconSeg3D-cardiac-rec

## Data inventory (verified 2026-08-16)

| Path | Reality |
|------|---------|
| `data/acdc/` | **Demo/fake**: 8 patients; 4D NIfTI ~488 KB; shape `(32,32,16,8)` — not challenge ACDC |
| `data/mmwhs/` | **Demo/fake**: 4 tiny image/label pairs |
| `data/emidec/` | **Demo/fake**: 4 tiny case pairs |
| `data/ami/` | Manifest **example only** — no private AMI |
| `outputs/ablations_smoke_v2/table.csv` | **Real measured smoke** numbers (usable for demo tables) |
| Official ACDC download | **Blocked** without CREATIS registration credentials |

## Prompt summary (ready paste)

```
Repo: https://github.com/Coucou2016/ReconSeg3D-cardiac-rec
Round 4 — Results/figures:
On-disk ACDC is auto-fake 32×32×16×8 (8 pts). Smoke table has PSNR~19.1 dB.
Which tables/figures are honest now? What must be marked 待补充?
SciencePlots fig1–5 already from smoke CSV — any caption risks?
Never invent clinical Dice/AUC or 0.934.
```

## Substitute reply gist

Honest now: demo ablation bars/heatmap; claim-boundary fig; pipeline schematic. Mark clinical ACDC/MM-WHS/EMIDEC tables 待补充. Caption every quantitative panel “demo/smoke”. Phenotype AUC=1.0 on tiny fake split = instability exhibit, not result.

## Accepted / rejected

| Advice | Decision |
|--------|----------|
| Label all smoke panels DEMO | **Accepted** |
| Publish fake-ACDC as “ACDC results” | **Rejected** |
| Keep measured smoke PSNR/MAE/Dice/warp/cycle | **Accepted** (with DEMO tag) |
| Fabricate subject-level CI | **Rejected** |

## Files changed

- Manuscript Results tables with DEMO labels from CSV
- Regenerated SciencePlots + report embeds
- This note
