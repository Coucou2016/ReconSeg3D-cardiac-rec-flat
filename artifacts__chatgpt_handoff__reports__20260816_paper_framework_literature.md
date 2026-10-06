# ChatGPT / literature consultation notes — paper framework (2026-08-16)

## Consultation channel status

| Item | Status |
|------|--------|
| Prior conversation | https://chatgpt.com/c/6a808651-37cc-83ea-a63d-2d2539a48d07 (Plus; audit already captured 2026-08-16) |
| Live browser MCP (`cursor-ide-browser`) | **Unavailable** in this agent session (catalog shows only `cursor-app-control`; `open_resource` to ChatGPT URL failed with unknown-agent error) |
| ZIP/file upload | Forbidden by policy — N/A |
| Independent literature | **Performed** via web search / publisher pages; DOIs verified when listed below |
| Auth/CAPTCHA | Not reached (no interactive ChatGPT session this turn) |

**Decision:** Proceed with (1) prior ChatGPT audit decisions already accepted locally, (2) independent literature for writing architecture, (3) Cursor-owned manuscript framing. Do **not** treat this file as a verbatim Plus transcript of a new chat.

---

## Prior Plus advisor decisions still in force (accepted)

From `20260816_chatgpt_audit_response.md`:

- Novelty = **motion-consistent 4D**; keep **HeartTTable-lite** naming; do not implement 3-CLS pairwise now.
- Real ACDC should become primary public motion validation (blocked until data on disk).
- Name cycle loss as **image-cycle**, not inverse-consistent flow.
- Cox shared finite mask — already implemented.

---

## Intended paste prompt (for user / next dual-agent turn when browser works)

```
CONTEXT (text only; no files). Project: lightweight ReconSeg3D + motion-consistent 4D
on public ACDC/MM-WHS/EMIDEC proxies; HeartTTable-lite fusion ablation only.
We MUST NOT claim private AMI 5y MACE AUC 0.934 or full HeartTTable parity.
Please use web search. Recommend: (1) Nature / npj Digit Med / MedIA writing
architectures for a METHODS paper; (2) outline with motion-consistent 4D as novelty;
(3) how to frame innovation without overclaim; (4) SciencePlots-ready figure panel plan;
(5) 8–12 key citations with DOIs. End with claim-boundary checklist.
```

---

## Independent literature (verified anchors)

| Paper | Venue | Identifier | Use |
|-------|-------|------------|-----|
| Gao et al. 3D spatiotemporal cardiac recon for MACE in AMI | npj Digit. Med. 2026 | DOI 10.1038/s41746-026-02449-0 | Original clinical boundary |
| Qian et al. Unified DL recon+motion+seg | IEEE ISBI 2024 | DOI 10.1109/ISBI56570.2024.10635390 | Joint-task neighbor |
| Yuan et al. 4D myocardium recon | ICCV 2023 | Open access CVF PDF | 4D motion–shape neighbor |
| CSTM 4D cine seg | arXiv 2024 | arXiv:2410.23191 | Temporal continuity neighbor |
| Hammernik et al. DL MRI recon review | MAGMA 2024 | DOI 10.1007/s10334-024-01173-8 | Background |
| Katzman et al. DeepSurv | BMC Med Res Methodol 2018 | DOI 10.1186/s12874-018-0482-1 | Cox DNN background |

---

## Recommended writing architecture (Cursor synthesis)

**Genre:** Methods paper (not clinical outcome paper).

**Argument:** public motion-consistent 4D ReconSeg3D stack → ablations on recon/seg/motion → HeartTTable-lite ablation only → hard stop before AMI AUC.

**Outline:** see `docs/paper/WRITING_FRAMEWORK.md`.

**Figure plan:** Figs 1–5 SciencePlots under `artifacts/paper/figures/` (smoke-labeled).

---

## Advice accepted vs rejected (this turn)

| Advice / option | Decision |
|-----------------|----------|
| Methods paper architecture | **Accepted** |
| Lead with motion-consistent 4D | **Accepted** |
| HeartTTable as main novelty | **Rejected** (ablation only) |
| Cite smoke AUC as clinical | **Rejected** |
| Claim 0.934 | **Rejected** |
| Live Plus web-search dialogue | **Blocked** (no browser MCP) — substituted with independent search |

---

## Deliverable links produced after this note

- Framework: `docs/paper/WRITING_FRAMEWORK.md`
- Manuscript: `docs/paper/MANUSCRIPT_DRAFT.md`
- Figures: `artifacts/paper/figures/`
- Research report: `artifacts/report/report.html` (+ MD/PDF if available)
