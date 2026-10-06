# Round 3 brief — Methods academic rewrite review

**Date:** 2026-08-17  
**Blob:** https://github.com/Coucou2016/ReconSeg3D-cardiac-rec/blob/main/artifacts/chatgpt_handoff/github_briefs/round3_methods_review.md  
**Raw:** https://raw.githubusercontent.com/Coucou2016/ReconSeg3D-cardiac-rec/main/artifacts/chatgpt_handoff/github_briefs/round3_methods_review.md  

## Please open

1. This brief.
2. Methods excerpt (same file §3): https://github.com/Coucou2016/ReconSeg3D-cardiac-rec/blob/main/docs/paper/MANUSCRIPT_DRAFT.md  
3. Implementation plan honesty: https://github.com/Coucou2016/ReconSeg3D-cardiac-rec/blob/main/docs/PAPER_PLAN.md  

Optional code (for consistency check only; do not paste back):  
https://github.com/Coucou2016/ReconSeg3D-cardiac-rec/tree/main/reconseg3d/models  

## Reply headings

### A. Academic tone rewrite
Provide a cleaned **§3 Methods** outline (subsection titles + 1–2 sentence intent each) in Nature-family methods voice. Remove config-flag / CLI / smoke-engineering language from the paper voice.

### B. Image-cycle definition
Give a precise, reviewer-safe definition distinguishing image-cycle from inverse-consistent optical flow / diffeomorphic registration.

### C. HeartTTable-lite wording
One paragraph that reviewers cannot misread as full HeartTTable parity.

### D. Missing method details
List what is underspecified for reproducibility in a methods paper (loss weights, sampling of sparse SA, train/val protocol) vs what belongs only in code/supplement.

### E. Must-fix sentences
Quote (paraphrase OK) up to 8 draft sentences that are non-academic or overclaiming; supply replacements.
