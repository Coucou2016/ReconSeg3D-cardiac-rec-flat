# ChatGPT brief (2/2): ED/ES indices + experiment matrix

**Repo:** https://github.com/Coucou2016/ReconSeg3D-cardiac-rec  
**Focus files:** `reconseg3d/data/acdc.py`, `reconseg3d/models/reconseg3d.py`, `reconseg3d/training/metrics.py`, `configs/publication_*.yaml`, `tests/test_ed_es.py`, `tests/test_label_propagation.py`

## Ask ChatGPT

Please critique the **ED/ES wiring** and proposed **experiment matrix**. No fake AUCs.

1. Killing silent `t//2`: primary seg gathers `seg_frame_indices` (ED); `segmentation_sequence` + `seg_valid_mask` supervise labeled phases only.
2. ED↔ES label-propagation eval (compose adjacent flows; Dice + HD95 with spacing). Real ACDC tables marked **待补充** under demo data.
3. Publication configs zero `w_mace`/`w_cox`; main line = recon+seg+motion geometry.
4. Suggested paper tables: T1 recon, T2 ED/ES Dice, T3 prop Dice/HD95, T4 jac_neg_ratio / inv_error — what is missing for a Major Revision?
5. Deferred: VoxelMorph baselines, M&Ms, 5-seed CI, ED-anchored motion path.

## Independent status

- Fake ACDC loader emits ED/ES fields; synthetic batches too.
- Tests for index wiring + known-warp propagation + HD95 `(1,1,1)` vs `(8,1,1)` — green locally.

## Response log location

`artifacts/chatgpt_handoff/reports/` for this round.
