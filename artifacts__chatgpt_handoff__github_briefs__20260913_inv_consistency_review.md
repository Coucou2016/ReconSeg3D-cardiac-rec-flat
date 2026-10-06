# ChatGPT brief (1/2): Inverse-consistency design — Review of your Diff

**Repo:** https://github.com/Coucou2016/ReconSeg3D-cardiac-rec  
**Focus files:** `reconseg3d/models/motion.py`, `reconseg3d/models/losses.py`, `tests/test_motion.py`, `docs/PAPER_PLAN.md`

## Ask ChatGPT

Please review the **inverse-consistency** design against VoxelMorph-style pull composition. Do **not** invent metrics. Comment on:

1. `warp_vector` / `compose_pull(u,v)=v+W(u,v)` sharing `align_corners=True` and `(dz,dy,dx)` with image warp.
2. `L_inv = ||u+W(v,u)|| + ||v+W(u,v)||` vs legacy **image-cycle** (kept as auxiliary `w_cycle`).
3. Smoothness `||∇u||²`, `ReLU(ε-detJ)`, and loop composition for short T.
4. Whether naming in the manuscript Methods is clear enough for registration-literate reviewers.
5. Any correctness bugs in the Diff (finite differences for Jacobian; border effects on exact ±1 voxel inverses).

## What we already implemented

- Unit tests: identity flow; +1 voxel x-shift direction; near-zero L_inv for exact ±1 translation; MultiTaskLoss wires `w_inv/w_smooth/w_jac/w_loop`.
- SVF + scaling-and-squaring is **stubbed** (`use_svf: false`) — deferred TODO.

## Response log location

Paste advisor notes under `artifacts/chatgpt_handoff/reports/` (this round). Dual-agent rule: advisor is **not** ground truth; verify against code + tests.
