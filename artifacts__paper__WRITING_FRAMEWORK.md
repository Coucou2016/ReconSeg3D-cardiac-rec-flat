# Writing architecture & innovation-framed paper outline

**Axes (nature-writing):** `task=manuscript` · `paper_type=methods` · `language=en` · `journal=nature-family` (npj Digital Medicine / MedIA-style methods)

**One-sentence argument:** On public cardiac MRI proxies (and synthetic smoke only for CI), we show that a **geometry- and motion-constrained** per-frame 4D ReconSeg3D pipeline—with **true inverse consistency** on adjacent flows, ED-anchored composed-path consistency, smoothness, and Jacobian folding, plus image-cycle as an intensity auxiliary only—provides a reproducible reconstruction–segmentation–motion evaluation stack, with HeartTTable-lite retained only as a fusion ablation—without claiming private AMI 5-year MACE AUC 0.934 or full HeartTTable parity.

---

## 1. Recommended venue / genre

| Option | Fit | Notes |
|--------|-----|-------|
| **Methods / software+methods** (MedIA, IEEE TMI, Frontiers in CV Med, arXiv first) | **Best** | Novelty is **geometry-constrained 4D** on public data + honest ablations |
| npj Digit. Med.–style clinical outcome paper | Poor until private AMI | Would require nested CV, calibration, external AMI MACE — **out of scope now** |
| Short technical note / code companion | Good interim | Emphasize reproducibility + claim boundary vs Gao et al. 2026 |

**Do not** submit as a clinical MACE superiority paper using smoke AUC/Dice.

---

## 2. Innovation framing (honest)

### Core novelty (claim)
- **Geometry- and motion-constrained 4D from sparse SA:** per-frame 3D decode + pull-field warps sharing one `grid_sample` convention; **true** \(L_\mathrm{inv}=\|u+W(v,u)\|+\|v+W(u,v)\|\); spatial smoothness; Jacobian folding penalty; adjacent loop; **ED-anchored** composed-path inverse consistency (`w_ed_ref`).
- **Image-cycle** (`w_cycle`) is an **auxiliary** intensity round-trip only — not a substitute for coordinate inverse consistency.
- Public-facing tables T1–T4 (recon / seg / motion geometry / ED↔ES label prop) on ACDC / MM-WHS / EMIDEC **when real data are mounted**; smoke otherwise labeled demo.

### Explicit non-claims
- No private AMI cohort; **do not report 0.934** AUC or C-index 0.897 from the original paper as ours.
- HeartTTable-lite ≠ three-modality pairwise class-token cross-attention; fusion is an **ablation**, not the product story.
- Compact CNN/UNet ≠ paper-scale 3D ViT / nnU-Net 256³.
- Demo metrics = pipeline checks only.

### Positioning vs original npj paper (Gao et al., *npj Digit. Med.* 2026; DOI `10.1038/s41746-026-02449-0`)
- Original: ReconSeg3D → HeartTTable on **n=4511 AMI** → 5-year MACE.
- This work: **methods extension** focusing on geometry/motion constraints on **public** data; risk head API ready for Cox when private data arrive.

### Related literature anchors (independently verified DOIs / arXiv)
| Topic | Citation | Role in outline |
|-------|----------|-----------------|
| Original clinical multimodal MACE | Gao et al. 2026, DOI 10.1038/s41746-026-02449-0 | Boundary / related work |
| Joint recon–motion–seg | Qian et al. ISBI 2024, DOI 10.1109/ISBI56570.2024.10635390 | Method neighbors |
| 4D myocardium recon | Yuan et al. ICCV 2023 | Motion–shape decoupling neighbor |
| Whole-sequence 4D seg | CSTM arXiv:2410.23191 | Temporal continuity neighbor |
| MRI recon DL review | Magma 2024, DOI 10.1007/s10334-024-01173-8 | Background |
| Cox DNN survival | DeepSurv, DOI 10.1186/s12874-018-0482-1 | Risk-head background (future AMI) |

---

## 3. Section architecture (methods paper)

1. **Title / Abstract** — geometry/motion-constrained 4D; true \(L_\mathrm{inv}\); public data; non-claim of AMI AUC.
2. **Introduction** — sparse SA cine gaps → need dense 4D; gap = unconstrained warps fold tissue / break cycle topology; contribution bullets.
3. **Related work** — recon; registration / inverse consistency; 4D seg; multimodal survival (HeartTTable / DeepSurv); position as methods not clinical parity.
4. **Methods**
   - Data contracts (ACDC / MM-WHS / EMIDEC / synthetic smoke)
   - Per-frame recon encoder–decoder
   - MotionNet; adjacent vs ED-anchored composition; warp scaling
   - Seg heads; phenotype / BCE / Cox APIs (optional)
   - HeartTTable-lite (single CLS over concat KV) — ablation only
   - Losses & training protocol (`publication_*.yaml`; `allow_fake_data: false`)
5. **Experiments** — smoke protocol (`smoke_motion.yaml`); planned subject-level CV on real ACDC; ablation matrix.
6. **Results** — T1–T4; mark 待补充 where real data missing; smoke figures clearly labeled.
7. **Discussion** — what geometry losses buy; failure modes; why not claim MACE.
8. **Limitations & reproducibility**
9. **Data / Code availability**

---

## 4. Figure panel plan (SciencePlots)

| Fig | Claim | Panels | Data source |
|-----|-------|--------|-------------|
| **Fig. 1** | Pipeline roles | Schematic boxes | Architecture (non-quantitative) |
| **Fig. 2** | Recon ablation | PSNR, MAE bars | `outputs/ablations_smoke_v2` |
| **Fig. 3** | Motion geometry | inv / jac / loop (+ aux cycle) | same / extended |
| **Fig. 4** | Seg Dice | Heatmap LV/RV/MYO/mean | same |
| **Fig. 5** | Claim boundary | Supported vs out-of-scope | Qualitative ledger |
| **Fig. ED1** 待补充 | Real ACDC ED/ES + cine motion | Volume curves, Dice CI | Real ACDC when available |
| **Fig. ED2** 待补充 | Calibration / C-index | Cox on private AMI | Out of scope until data |

All rendered with **SciencePlots** (`science` + `nature` + Times New Roman for Latin).

---

## 5. Tables (planned)

| Table | Content | Status |
|-------|---------|--------|
| T1 Recon | PSNR / `ssim_3d` / `recon_ssim_proxy` / MAE | API Done; real ACDC 待补充 |
| T2 Seg | Dice / HD95 at ED/ES | API Done; real physical tables 待补充 |
| T3 Motion geometry | inv, smooth, jac_neg, loop, ed_ref | Implemented; smoke OK |
| T3b Function | EDV/ESV/EF mL/% | Physical API Done; real 待补充 |
| T4 ED↔ES label prop | Prop Dice / HD95 + bootstrap CI | API + synthetic tests; real ACDC **待补充** |
| T5 Fusion | Concat vs HeartTTable-lite | Ablation only |

---

## 6. Advisor consultation status

- **Prior ChatGPT audit** (2026-08-16): accept HeartTTable-lite naming; motion as novelty; defer 3-CLS; real ACDC primary — see `artifacts/chatgpt_handoff/reports/20260816_chatgpt_audit_response.md`.
- **P0 motion revision** (2026-09-13) + **P0 gap closure** (2026-09-14): true \(L_\mathrm{inv}\), ED-ref path, publication fake-data hard gate — see acceptance reports under `artifacts/chatgpt_handoff/reports/`.
- Public GitHub: https://github.com/Coucou2016/ReconSeg3D-cardiac-rec

### On-disk data honesty

| Tree | Verdict |
|------|---------|
| `data/acdc` (demo layout) | **Demo/fake** — publication configs hard-error unless real patients mounted |
| `data/mmwhs`, `data/emidec` | **Demo/fake** |
| Official ACDC | CREATIS registration required |
| Smoke CSV / metrics.json | **Measured** demo numbers (OK for DEMO tables) |

---

## 7. Acceptance criteria for draft manuscript

- [x] Methods paper architecture
- [x] Explicit non-claims of 0.934 / full HeartTTable
- [x] Lead story = geometry/motion + true inverse consistency (image-cycle auxiliary)
- [x] ED-anchored + adjacent motion paths implemented and weighted
- [x] Publication configs forbid synthetic/fake (`allow_fake_data: false`)
- [x] SciencePlots figures from real local smoke logs
- [x] Demo inventory + CREATIS download blocker documented
- [ ] Real ACDC subject-level ED↔ES tables (blocked: challenge data not licensed/mounted)
- [ ] Nested CV / calibration (deferred)
- [x] SVF path + baseline adapters + M&Ms stub + 5-seed harness + folds (infra Done; licensed-data tables 待补充)
