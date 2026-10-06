# Round 4 brief — Results, tables, figures honesty

**Date:** 2026-08-17  
**Blob:** https://github.com/Coucou2016/ReconSeg3D-cardiac-rec/blob/main/artifacts/chatgpt_handoff/github_briefs/round4_results_figures.md  
**Raw:** https://raw.githubusercontent.com/Coucou2016/ReconSeg3D-cardiac-rec/main/artifacts/chatgpt_handoff/github_briefs/round4_results_figures.md  

## Please open

1. This brief.
2. Manuscript §§4–5: https://github.com/Coucou2016/ReconSeg3D-cardiac-rec/blob/main/docs/paper/MANUSCRIPT_DRAFT.md  
3. Data docs: https://github.com/Coucou2016/ReconSeg3D-cardiac-rec/blob/main/docs/DATA.md  

**Hard facts (do not contradict without evidence):**

- On-disk ACDC / MM-WHS / EMIDEC in this public project are **demo-scale fake NIfTI**, not challenge downloads.
- Quantitative tables currently come from **measured demo/smoke** ablations — valid as pipeline verification, **not** clinical performance.
- Private AMI 5-year AUC **0.934** is original-paper-only; never ours.
- Phenotype AUC 1.0 / accuracy 0.0 on tiny fake splits = metric instability, not a result.

## Reply headings

### A. What to keep vs cut
For each of Tables 1–2 and Figs 1–5: `keep-as-DEMO` / `move-to-supplement` / `cut` / `replace-with-placeholder`, with one-line rationale.

### B. Results prose rewrite
Rewrite §5 opening (≤150 words) so DEMO status is unmistakable but still academic (not engineering diary).

### C. Overclaim traps
List phrases that reviewers could weaponize; supply safer alternatives.

### D. Figure legends
Draft 5 short legends (Fig. 1–5) suitable for submission with DEMO caveats where needed.

### E. Real-data contingency
If licensed ACDC were mounted tomorrow, which **exact** tables/panels should appear first? What stays DEMO?
