# Round 3 log — Methods academic rewrite (2026-08-17)

## GitHub brief

- Blob: https://github.com/Coucou2016/ReconSeg3D-cardiac-rec/blob/main/artifacts/chatgpt_handoff/github_briefs/round3_methods_review.md
- Raw: https://raw.githubusercontent.com/Coucou2016/ReconSeg3D-cardiac-rec/main/artifacts/chatgpt_handoff/github_briefs/round3_methods_review.md

## Live ChatGPT

| Item | Value |
|------|-------|
| Live reply | **NO** |
| Fallback | nature-polishing axes `paper_type=methods` · `section=methods` · `language=en` · `journal=generic` (Nature-family methods framing) |

## Applied rewrite decisions

1. Removed config-flag prose (`fusion: heart_ttable`, CLI) from paper voice.
2. Image-cycle defined as intensity residual after fwd/bwd warps; explicit non-equivalence to inverse-consistent flow.
3. HeartTTable-lite paragraph cannot be read as pairwise three-CLS parity.
4. Underspecified for full reproducibility (exact loss weights, nested CV) → deferred to code/supplement; paper states demo protocol honestly.
5. Methods subsections 3.1–3.6 kept academic.

## Local edits

- §3 Methods in `MANUSCRIPT_DRAFT.md`
