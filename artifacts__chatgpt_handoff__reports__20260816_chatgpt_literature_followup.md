# ChatGPT Plus literature follow-up — 2026-08-16 (retry)

## Outcome

| Item | Status |
|------|--------|
| Live ChatGPT Plus consultation | **BLOCKED** |
| Conversation URL | https://chatgpt.com/c/6a808651-37cc-83ea-a63d-2d2539a48d07 (not opened) |
| Auth / CAPTCHA | **Not reached** (no interactive browser session) |
| Text paste / ZIP upload | N/A (session never started); policy remains text-only |
| Framework / manuscript edits this turn | **None** (no new Plus structural advice received) |

## Why blocked

1. MCP catalog for this worker lists only `cursor-app-control` — **`cursor-ide-browser` is not registered** (`GetMcpTools` → server not found).
2. Fallback `open_resource` on the ChatGPT URL failed: `unknown agent: 18b2090d-fc69-4dd3-95bb-37c9f283b2d4`.
3. Therefore no login wall was observed; **user authentication is not the blocker** — missing browser MCP / agent binding is.

**User action needed:** Enable `cursor-ide-browser` for the agent that should drive ChatGPT (or re-run this follow-up from a parent session that already has that MCP), then authenticate to ChatGPT if prompted. Do not share passwords/API keys in chat.

---

## Intended paste (ready when browser works)

```
CONTEXT (text only; no files/uploads). Project: lightweight ReconSeg3D + motion-consistent 4D
on public ACDC/MM-WHS/EMIDEC proxies / synthetic smoke; HeartTTable-lite is fusion ablation only.
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

## Deliverable verification (same turn)

| Path | Size | Verdict |
|------|------|---------|
| `docs/paper/WRITING_FRAMEWORK.md` | 6,356 B | PASS |
| `docs/paper/MANUSCRIPT_DRAFT.md` | 13,017 B | PASS |
| `artifacts/paper/manuscript.html` | 533,357 B | PASS (DOCTYPE, inline CSS, 5× Base64 figs; DOI text links only; no HTML `<table>`) |
| `artifacts/paper/manuscript.pdf` | 495,725 B | PASS (`%PDF-` header) |
| `artifacts/report/report.html` | 530,935 B | PASS (DOCTYPE, inline CSS, 5× Base64 imgs, 2 tables; **no** external CDN/image URLs) |
| `artifacts/report/report.md` | 1,267 B | PASS |
| `artifacts/report/report.pdf` | 772,771 B | PASS (`%PDF-` header) |
| `artifacts/paper/figures/` (fig1–5 pdf/png/svg) | present | PASS (PNG magic OK) |
| `scripts/make_paper_figures.py` | 9,578 B | PASS |
| `scripts/build_report_bundle.py` | 20,154 B | PASS |

No broken/missing deliverables flagged.

---

## Independent citation sanity check (no invented DOIs)

| Local label | DOI / ID | Check |
|-------------|----------|-------|
| Gao et al. npj Digit. Med. 2026 | `10.1038/s41746-026-02449-0` | **OK** — Nature page resolves; AUC 0.934 is **theirs**, not ours |
| MAGMA 2024 DL MRI recon review | `10.1007/s10334-024-01173-8` | **OK** — real review (Heckel et al.); framework’s “Hammernik et al.” shorthand is imprecise (Hammernik is cited *inside* that review) |
| ISBI 2024 joint recon/motion/seg | `10.1109/ISBI56570.2024.10635390` | **DOI OK**; author corrected to **Qian et al.** (Qian, Zhou, Hu, Qi) |
| DeepSurv | `10.1186/s12874-018-0482-1` | Accepted prior anchor; not re-fetched this turn |
| Yuan ICCV 2023 / CSTM arXiv:2410.23191 | no DOI in framework | Leave as non-DOI anchors; do not invent DOIs |

**Decision:** Author label for DOI `10.1109/ISBI56570.2024.10635390` corrected to **Qian et al.** in manuscript/framework sources (and regenerated HTML).

---

## Advice accepted / rejected this turn

| Item | Decision |
|------|----------|
| Live Plus web-search dialogue | **Blocked** — no browser MCP |
| Fabricate Plus transcript | **Rejected** |
| Change novelty / claim boundary | **No change** (prior decisions stand) |
| Cite smoke metrics as clinical AMI | **Rejected** (unchanged policy) |

Prior Plus audit still authoritative: `20260816_chatgpt_audit_response.md`. Prior substitute notes: `20260816_paper_framework_literature.md`.
