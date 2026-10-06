# Round 2 — Related work / literature (web search)

**Status:** `ChatGPT live blocked` — independent WebSearch/WebFetch used instead (honest substitute).  
**Public GitHub:** https://github.com/Coucou2016/ReconSeg3D-cardiac-rec

## Prompt summary (ready paste)

```
Web search ON. Repo: https://github.com/Coucou2016/ReconSeg3D-cardiac-rec
Round 2 — Related work:
1) Verify Qian et al. ISBI 2024 DOI for joint recon–motion–seg.
2) Position vs CSTM (Ye et al.; arXiv:2410.23191 / WACV 2025) and Gao et al. npj Digit Med 2026 DOI 10.1038/s41746-026-02449-0.
3) Suggest Nature/npj Digit Med methods-style related-work paragraph structure (3–4 short paras).
4) Flag any DOI/year conflicts. Do NOT invent citations.
```

## Independent verification gist

| Citation | Verified | Note |
|----------|----------|------|
| Qian et al. ISBI 2024 | DOI `10.1109/ISBI56570.2024.10635390` | Correct in draft |
| Gao et al. npj Digit. Med. | DOI `10.1038/s41746-026-02449-0` | Volume 9 article 325 (2026); AUC 0.934 is **theirs**, not ours |
| CSTM | arXiv `2410.23191` → **Ye et al., WACV 2025**, pp. 9514–9524 | Update authors/venue in manuscript |
| Hammernik MAGMA review | DOI `10.1007/s10334-024-01173-8` | Keep |
| DeepSurv | DOI `10.1186/s12874-018-0482-1` | Keep as risk-head background |

## Accepted / rejected

| Advice | Decision |
|--------|----------|
| Keep Qian DOI as written | **Accepted** (verified) |
| Cite CSTM as Ye et al. WACV 2025 + arXiv | **Accepted** |
| Mimic clinical outcome paper structure | **Rejected** — stay methods |
| Borrow 0.934 into our results | **Rejected** |

## Files changed

- `docs/paper/MANUSCRIPT_DRAFT.md` related work + refs
- This note
