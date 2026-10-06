# Round 5 — Full manuscript polish + risk/limitations

**Status:** `ChatGPT live blocked` — nature-polishing axes applied locally: `paper_type=methods`, `language=en`, `journal=generic` (Nature-family methods).  
**Public GitHub:** https://github.com/Coucou2016/ReconSeg3D-cardiac-rec

## Prompt summary (ready paste)

```
Web search ON. Repo: https://github.com/Coucou2016/ReconSeg3D-cardiac-rec
Round 5 — Polish + limitations:
Review full methods manuscript for: overclaim, image-cycle naming, HeartTTable-lite, DEMO vs clinical tables, Data/Code availability (public GitHub), consistency of numbers with outputs/ablations_smoke_v2/table.csv.
Return: ranked risk list + concrete sentence edits (no new invented metrics).
```

## Substitute reply gist

Top risks: (1) readers mistaking demo ACDC for challenge data; (2) phenotype AUC=1.0; (3) MACE AUC columns in smoke tables; (4) code availability missing GitHub URL. Mitigations: inventory box in Experiments; strike clinical reading of task metrics; add limitations on registration-gated public datasets; sync HTML/PDF/report.

## Accepted / rejected

| Advice | Decision |
|--------|----------|
| Add explicit demo inventory + download blocker | **Accepted** |
| Soften Conclusions to reproducibility contribution | **Accepted** |
| Add private AMI external validation claim | **Rejected** |
| Keep 0.934 as boundary citation only | **Accepted** |

## Files changed

- Full `MANUSCRIPT_DRAFT.md` consistency pass
- `WRITING_FRAMEWORK.md`, report bundle, HTML/PDF
- Acceptance report `20260816_five_round_acceptance.md`
- This note
