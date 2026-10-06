# ChatGPT Plus literature + GitHub handoff — 2026-08-16 (evening)

## Outcome

| Item | Status |
|------|--------|
| Live ChatGPT Plus consultation | **BLOCKED** (browser MCP tab vanish) |
| Conversation URL (prior, still valid) | https://chatgpt.com/c/6a808651-37cc-83ea-a63d-2d2539a48d07 |
| Public GitHub for advisor fetch | **https://github.com/Coucou2016/ReconSeg3D-cardiac-rec** (PUBLIC; code+docs only) |
| Auth / CAPTCHA | Not reached — MCP could not retain a browser tab |
| Fabricated Plus transcript | **Rejected** |

## Browser MCP evidence (this turn)

1. `browser_tabs` new → creates `viewId` then tab disappears from list
2. `browser_navigate` without viewId → `No browser tab available. Please navigate to a page first.`
3. Navigate with explicit `viewId` → `Browser view not found: … Use browser_navigate without a viewId to create a new tab.`
4. `open_resource` ChatGPT URL → `unknown agent: …`
5. Not an auth wall; user need not re-login for this specific failure mode

**User action:** Open the conversation URL in Cursor Simple Browser / system browser manually, paste the prompt below, and optionally share the public GitHub URL so Plus can fetch code/docs (text-only policy; no ZIP).

---

## Intended paste (ready when browser works)

```
CONTEXT (text only; no ZIP uploads). Public code+docs:
https://github.com/Coucou2016/ReconSeg3D-cardiac-rec

Project: lightweight ReconSeg3D + motion-consistent 4D on public ACDC/MM-WHS/EMIDEC proxies / synthetic smoke.
HeartTTable-lite is fusion ablation only.
We MUST NOT claim private AMI 5y MACE AUC 0.934 or full HeartTTable parity as our result.

Please ENABLE WEB SEARCH and advise on:
(1) Nature / npj Digital Medicine / MedIA-style METHODS-paper architecture for this novelty;
(2) outline with motion-consistent 4D recon/seg as the core claim;
(3) innovation framing that keeps HeartTTable-lite as ablation only;
(4) SciencePlots-ready figure panel plan (pipeline, recon ablation, motion metrics, seg Dice, claim boundary);
(5) 8–12 key citations with verified DOIs only.
End with a claim-boundary checklist (supported vs out-of-scope).
```

---

## Independent literature notes (Cursor web search; adopt only if honest)

| Neighbor | ID | Role | Adopt? |
|----------|-----|------|--------|
| Gao et al. npj Digit. Med. 2026 | DOI 10.1038/s41746-026-02449-0 | Boundary / related | Yes (prior) |
| Qian et al. ISBI 2024 joint recon/motion/seg | DOI 10.1109/ISBI56570.2024.10635390 | Method neighbor | Yes (prior; author=Qian) |
| CSTM whole-sequence 4D seg | arXiv:2410.23191 / WACV 2025 | Temporal continuity neighbor | Yes as non-DOI arXiv anchor |
| CineMesh4D sparse cine→4D mesh | arXiv:2605.13994 | Sparse→dense 4D neighbor | Optional related-work mention; do not invent DOI |
| MAGMA 2024 MRI recon review | DOI 10.1007/s10334-024-01173-8 | Background | Yes (prior) |
| DeepSurv | DOI 10.1186/s12874-018-0482-1 | Cox head background | Yes (prior) |

**Framing decision (unchanged):** methods paper; novelty = motion-consistent 4D; HeartTTable-lite = ablation; no 0.934 claim; real ACDC tables 待补充.

---

## Advice accepted / rejected this turn

| Item | Decision |
|------|----------|
| Live Plus web-search dialogue | Blocked — no retained browser tab |
| Point advisor at public GitHub | **Accepted** (authorized this turn) |
| Cite smoke metrics as clinical | **Rejected** |
| Change core novelty / claim boundary | **No change** |
