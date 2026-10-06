# Round 1 — Architecture & novelty framing

**Status:** `ChatGPT live blocked` (cursor-ide-browser: tabs create then vanish before navigate; `Browser view not found` / chicken-egg with lock).  
**Chat links:** intended https://chatgpt.com/c/6a808651-37cc-83ea-a63d-2d2539a48d07 · fallback advisor notes below.  
**Public GitHub (for advisor every round):** https://github.com/Coucou2016/ReconSeg3D-cardiac-rec

## Prompt summary (ready for user paste)

```
You are advising a methods paper (Nature-family / MedIA style). Web search ON.
Public repo (fetch code/docs only): https://github.com/Coucou2016/ReconSeg3D-cardiac-rec

Task — Round 1 Architecture & novelty:
1) Critique our framing: motion-consistent 4D (per-frame recon + warp + image-cycle) as LEAD novelty.
2) HeartTTable-lite = single CLS over concat KV; ablation only — OK?
3) Outline methods-paper section order for docs/paper/MANUSCRIPT_DRAFT.md.
4) Explicit non-claims: no private AMI; never claim AUC 0.934 as ours.
Return: (a) accept/reject novelty lead, (b) 6-bullet outline, (c) top 3 claim risks.
```

## Advisor reply gist (local substitute; prior Plus audit + independent synthesis)

Prior live Plus audit (same chat, earlier turn) already endorsed: motion as novelty; HeartTTable-lite naming; defer 3-CLS; real ACDC primary when available. Local Round-1 substitute agrees: lead with motion-consistent 4D training contract; fusion is ablation; venue = methods not clinical MACE.

## Accepted / rejected

| Advice | Decision |
|--------|----------|
| Motion-consistent 4D as lead | **Accepted** — manuscript title/abstract/intro |
| HeartTTable-lite ablation-only | **Accepted** |
| Clinical MACE parity narrative | **Rejected** |
| Invent real-ACDC superiority | **Rejected** |

## Files changed this round

- `docs/paper/MANUSCRIPT_DRAFT.md` (architecture polish)
- `docs/paper/WRITING_FRAMEWORK.md` (advisor status)
- This note
