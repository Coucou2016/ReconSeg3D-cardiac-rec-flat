# Dual-agent session live notes (updated 2026-08-16, paper/report turn)

## Policy (current)
- ChatGPT Pro/Plus = external advisor only (text paste, **no ZIP/file upload**).
- Cursor = sole implementer and independent verifier.
- No git commit / push / PR / deploy without explicit user authorization.
- Repo has **no `.git`** (`NO_GIT_REPO`) → **Git status = local only**.

## Conversation
- Title: ReconSeg3D Audit Preparation
- URL: https://chatgpt.com/c/6a808651-37cc-83ea-a63d-2d2539a48d07
- This turn: **live Plus paste blocked** — `cursor-ide-browser` MCP not available; only `cursor-app-control` present. Literature framework from independent web search + prior audit.

## Completed this turn (paper / report deliverables)
1. Nature-writing axes: methods manuscript, EN, nature-family framing.
2. SciencePlots installed; figures regenerated from `outputs/ablations_smoke_v2` → `artifacts/paper/figures/`.
3. Framework: `docs/paper/WRITING_FRAMEWORK.md`
4. Manuscript: `docs/paper/MANUSCRIPT_DRAFT.md` + HTML/PDF under `artifacts/paper/`
5. Research report: `artifacts/report/report.html` (Base64 figs, inline CSS) + `report.md` + `report.pdf`
6. ChatGPT/literature notes: `artifacts/chatgpt_handoff/reports/20260816_paper_framework_literature.md`
7. `requirements.txt` adds SciencePlots + matplotlib
8. Scripts: `scripts/make_paper_figures.py`, `scripts/build_report_bundle.py`

## Prior accepted advisor items (still in force)
- HeartTTable-lite; no 3-CLS now; motion = novelty; Cox mask + motion tests done.

## Blockers
- No interactive ChatGPT this turn (browser MCP missing).
- Real ACDC/MM-WHS/EMIDEC not mounted → public main tables 待补充.
- NO_GIT_REPO (no remote share URL for advisor fetch).

## Verify
```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest tests/ -v
python scripts/make_paper_figures.py
python scripts/build_report_bundle.py
```
