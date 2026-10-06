# Round 1 brief — Paper vs research report separation + novelty framing

**Date:** 2026-08-17  
**Chat role:** ChatGPT Plus advisor (web search ON)  
**Repo:** https://github.com/Coucou2016/ReconSeg3D-cardiac-rec  
**This file (blob):** https://github.com/Coucou2016/ReconSeg3D-cardiac-rec/blob/main/artifacts/chatgpt_handoff/github_briefs/round1_paper_vs_report_novelty.md  
**Raw:** https://raw.githubusercontent.com/Coucou2016/ReconSeg3D-cardiac-rec/main/artifacts/chatgpt_handoff/github_briefs/round1_paper_vs_report_novelty.md  

## Please open and read (in order)

1. This brief (you are here).
2. Paper plan: https://github.com/Coucou2016/ReconSeg3D-cardiac-rec/blob/main/docs/PAPER_PLAN.md  
3. Writing framework: https://github.com/Coucou2016/ReconSeg3D-cardiac-rec/blob/main/docs/paper/WRITING_FRAMEWORK.md  
4. Full manuscript draft: https://github.com/Coucou2016/ReconSeg3D-cardiac-rec/blob/main/docs/paper/MANUSCRIPT_DRAFT.md  
5. Data contract: https://github.com/Coucou2016/ReconSeg3D-cardiac-rec/blob/main/docs/DATA.md  

**Constraint:** Do **not** ask the user to paste large files. Use the GitHub / raw URLs above.

---

## Context (short)

This is a **public methods extension** of Gao et al. (*npj Digital Medicine*, 2026) ReconSeg3D / HeartTTable work. Local workspace has **demo/fake** ACDC/MM-WHS/EMIDEC trees only; private AMI and original 5-year MACE AUC **0.934** are **out of scope** and must never appear as our result.

Two deliverable genres exist:

| Artifact | Path | Allowed content |
|----------|------|-----------------|
| **Paper** | `docs/paper/MANUSCRIPT_DRAFT.md` (+ HTML/PDF) | Academic methods style only. No Windows paths, no Cursor/ChatGPT process, no engineering diary, no “smoke test” tone as primary voice. Honest DEMO labels. |
| **Research report** | `artifacts/report/` | Process, local paths, file inventory, dual-agent notes, figure 来龙去脉 OK. |

Lead novelty claim: **motion-consistent 4D** (per-frame decode + warp + **image-cycle** intensity consistency) + HeartTTable-**lite** as fusion **ablation only**.

---

## Questions for you (structured reply requested)

Please reply with these exact section headings:

### A. Paper vs report leakage
List every phrase/section in `MANUSCRIPT_DRAFT.md` that should move to the research report (paths like `outputs/`, `scripts/`, CI tone, internal inventory tables that read as lab notes, Chinese 待补充 overuse, etc.). Give **replace-with** academic wording where possible.

### B. Novelty framing (1 paragraph)
Rewrite a submission-ready novelty paragraph for the Introduction (≤120 words) that a npj Digit Med / MedIA methods reviewer would accept—without overclaiming vs Qian et al. ISBI 2024 or Ye et al. WACV 2025.

### C. Title / Abstract cuts
Propose a tighter title (optional) and 3 concrete Abstract edits (delete / rephrase / add).

### D. Non-negotiable honesty checklist
Confirm or correct: (1) no private AMI AUC 0.934; (2) DEMO/public-proxy labels; (3) HeartTTable-lite ≠ full HeartTTable; (4) image-cycle ≠ inverse-consistent flow; (5) ISBI joint recon–motion–seg = **Qian et al.** (not other authors).

### E. Priority edits (top 7)
Ordered list of the highest-impact manuscript edits for Round 1 only (paper vs report + novelty). Mark each as `must` or `nice`.

### F. What NOT to invent
Explicitly list claims/numbers we must not fabricate before real licensed ACDC/MM-WHS/EMIDEC mounts.

---

## What we will do with your answer

Cursor will independently verify each point against the repo, apply accepted edits to the paper (and move rejected process detail into `artifacts/report/`), commit, and push before Round 2.
