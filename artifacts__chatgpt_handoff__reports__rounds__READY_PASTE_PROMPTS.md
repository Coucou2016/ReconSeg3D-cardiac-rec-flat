# Paste prompts for live ChatGPT (user fallback)

If browser MCP is blocked, open https://chatgpt.com/ (new **paper maturation** chat or https://chatgpt.com/c/6a808651-37cc-83ea-a63d-2d2539a48d07), enable **web search**, and paste rounds in order. Always include:

**Public GitHub:** https://github.com/Coucou2016/ReconSeg3D-cardiac-rec

---

## Round 1

You are advising a methods paper (Nature-family / MedIA style). Web search ON.
Public repo: https://github.com/Coucou2016/ReconSeg3D-cardiac-rec
Task — Architecture & novelty: (1) motion-consistent 4D (per-frame recon + warp + image-cycle) as LEAD novelty — critique; (2) HeartTTable-lite = single CLS over concat KV, ablation only — OK?; (3) methods-paper section order; (4) non-claims: no private AMI, never claim AUC 0.934 as ours. Return accept/reject novelty lead, 6-bullet outline, top 3 claim risks.

## Round 2

Web search ON. Repo: https://github.com/Coucou2016/ReconSeg3D-cardiac-rec
Related work: verify Qian ISBI 2024 DOI; position vs CSTM (arXiv:2410.23191 / WACV 2025) and Gao npj Digit Med DOI 10.1038/s41746-026-02449-0; suggest 3–4 short related-work paragraphs; flag DOI conflicts; do NOT invent citations.

## Round 3

Repo: https://github.com/Coucou2016/ReconSeg3D-cardiac-rec — read methods modules under reconseg3d/models/.
Methods: (B,C,T,D,H,W); per-frame recon; MotionNet; Warp L1; Image-cycle = intensity cycle NOT inverse-consistent flow; HeartTTable-lite. Critique reviewer traps; wording fixes only.

## Round 4

Repo as above. On-disk ACDC is auto-fake 32×32×16×8 (8 pts). Smoke PSNR~19.1. Which tables/figures are honest? Mark 待补充. Never invent clinical Dice/AUC or 0.934.

## Round 5

Repo as above. Full polish + limitations: overclaim risks, DEMO labeling, Data/Code availability with GitHub URL, consistency with outputs/ablations_smoke_v2/table.csv. Ranked risks + sentence edits; no invented metrics.
