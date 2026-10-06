# Geometry- and motion-constrained 4D reconstruction and segmentation from sparse short-axis CMR

**Working manuscript (methods)**  
**Spine note:** A separate user-revised mainline file was **not located** in this session (`docs/paper/AWAITING_USER_MANUSCRIPT.md`). This draft remains the interim academic spine.  
**Status:** DEMO-regime quantitative panels only. Licensed subject-level tables: **待补充**.  
**Target genre:** Nature-family / methods · English (npj Digit. Med.–adjacent structure, not a clinical MACE claim)  
**Code:** https://github.com/Coucou2016/ReconSeg3D-cardiac-rec  

---

## Abstract

Sparse short-axis (SA) cine cardiac magnetic resonance (CMR) undersamples three-dimensional anatomy across the cardiac cycle. Dense spatiotemporal reconstruction can restore anatomical continuity for segmentation and functional analysis, yet unconstrained deep warps may fold tissue and break cycle topology. Prior clinical work has shown that reconstructed bi-ventricular cine volumes can support long-horizon risk modelling on private acute myocardial infarction (AMI) cohorts; those discrimination figures are tied to private data and are **not claimed here**.

We present a public methods extension that treats **geometry- and motion-constrained 4D recovery from sparse SA** as the primary scientific line. The stack couples per-frame volumetric decoding with pull-field warps that share a single sampling convention (`grid_sample`, `align_corners=True`, displacement channels \((d_z,d_y,d_x)\)). Training emphasises true inverse consistency \(L_\mathrm{inv}=\|u+W(v,u)\|+\|v+W(u,v)\|\), spatial smoothness, a Jacobian folding penalty \(\mathrm{ReLU}(\varepsilon-\det J)\), a periodic adjacent-loop residual, and an end-diastolic (ED)–anchored composed-path constraint. Image-cycle intensity residuals and ventricular volume-curve smoothing are retained only as **auxiliary / physiological** regularizers. Segmentation is supervised at labeled ED and end-systolic (ES) phases when available; unlabeled frames rely on motion self-supervision. Multimodal fusion and survival heads are **decoupled** from the publication path.

On controlled demo-scale tensors we verify end-to-end logging of reconstruction (PSNR/MAE, windowed `ssim_3d`, global `recon_ssim_proxy`), ED/ES Dice, inverse-consistency / Jacobian statistics, and ED↔ES label-propagation APIs. Licensed ACDC / MM-WHS / EMIDEC subject tables remain to be completed. We do not claim private-AMI MACE parity.

**Keywords:** cardiac MRI; 4D reconstruction; inverse-consistent registration; Jacobian regularity; multi-task learning; public benchmarks

---

## 1. Introduction

Cardiac cine CMR is typically acquired as a stack of two-dimensional SA slices. Slice gaps, through-plane motion, and temporal undersampling leave a sparse observation of bi-ventricular anatomy. Dense three-dimensional reconstruction across the cardiac cycle—effectively a four-dimensional (4D) volume—can improve anatomical continuity for segmentation and downstream phenotyping.

Clinical multimodal pipelines that reconstruct temporally resolved bi-ventricular volumes and then fuse imaging with tabular variables for five-year major adverse cardiovascular event (MACE) prediction illustrate the downstream value of dense cine volumes[^1]. Headline discrimination metrics in that setting are bound to private cohorts and therefore out of scope for any public reimplementation that lacks those data.

**Gap.** Lightweight public reimplementations often broadcast a single reconstructed volume across time, use mid-cycle frames as silent segmentation references, or regularize only intensity cycles after warping. Registration literature instead emphasises inverse-consistent displacement composition, diffeomorphic (non-folding) maps, and periodic cardiac constraints[^9][^10][^11]. Concurrent cardiac lines couple reconstruction with motion or mesh dynamics[^3][^4][^7][^8][^12], but a compact, auditable recon–seg–motion contract with named geometric losses and an honest multimodal/survival boundary remains useful.

**Contributions.**

1. Geometry-constrained 4D training: inverse consistency, smoothness, Jacobian folding, and loop consistency under one pull-warp convention, plus an ED-anchored composed path.
2. ED/ES-aware multi-phase supervision; unlabeled phases via motion self-supervision (no silent mid-cycle reference).
3. ED↔ES label-propagation evaluation (Dice + physical-spacing HD95 API).
4. Publication configs that zero MACE/Cox weights; HeartTTable-style fusion retained as an ablation only.
5. Explicit claim boundary: demo-regime pipeline verification ≠ licensed subject tables ≠ private AMI survival.

---

## 2. Related work

**Learned registration.** VoxelMorph-style CNNs predict dense displacements with similarity and smoothness losses[^9]. TransMorph and related transformers improve capacity for large deformations[^10]. Inverse-consistency and diffeomorphic integration (scaling-and-squaring of stationary velocity fields) reduce folding[^9][^11].

**Cardiac motion and 4D recovery.** Groupwise / multi-view motion models target beat-to-beat consistency[^12]. Unified recon–motion–seg unrolling[^3], decoupled motion–shape myocardium recovery[^4], memory-based whole-sequence segmentation[^5], Neural ODE continuous dynamics, and mesh/tetrahedra recoveries[^7][^8] address complementary geometry. Our stack is intentionally lightweight and registration-literate rather than mesh-complete.

**Multimodal survival.** DeepSurv-style Cox models[^6] and HeartTTable-style fusion[^1] motivate optional risk heads. We expose APIs but keep them off the main publication path.

**Positioning.** Relative to Gao et al.[^1], this draft is a methods extension focused on geometric motion constraints and public-proxy evaluation. Baseline adapters (compact VoxelMorph-style, FlowReg interface, recon-only) are provided in code; we do not claim leaderboard numbers without completed licensed-data runs, nor private AMI discrimination.

---

## 3. Methods

### 3.1 Problem setup

Let \(x \in \mathbb{R}^{B \times C \times T \times D \times H \times W}\) denote a cine volume batch. Sparse SA sampling is simulated by zeroing unobserved slices (or by sparse stacking when physical spacing metadata are available). The model predicts a dense reconstruction \(\hat{x}\), adjacent pull displacements \(u_t,v_t\), multi-class segmentations at labeled phases, and optional task heads.

### 3.2 Per-frame reconstruction

The default path encodes each frame with a compact 3D convolutional backbone and decodes per-frame volumes (`per_frame_recon=true`). Compact demonstration grids (for example \(16{\times}32{\times}32\), \(T{=}8\)) verify the pipeline; paper-scale grids remain configuration options. A broadcast baseline that copies a single reconstructed volume across time is retained only for ablation.

### 3.3 Deformation convention

Displacements use channels \((d_z,d_y,d_x)\) in voxels, converted to `grid_sample` coordinates with `align_corners=True` and scale \(2/\max(\mathrm{dim}-1,1)\). Vector fields are warped with the **same** sampler as images (`warp_vector`). Pull composition is \(u\circ v = v + W(u,v)\). Closed-cycle MotionNet stacks emit \(T\) adjacent pairs for \(T\) frames (last slot closes the loop).

### 3.4 Geometric and auxiliary losses

We minimize the following geometric terms (primary novelty):

- **Inverse consistency** \(L_\mathrm{inv}=\|u+W(v,u)\|+\|v+W(u,v)\|\) (coordinate cycle; not an intensity residual).
- **Smoothness** \(L_\mathrm{smooth}=\|\nabla u\|^2\) via finite differences on each displacement channel.
- **Folding penalty** \(L_\mathrm{jac}=\mathrm{ReLU}(\varepsilon-\det J)\) with \(J=I+\nabla u\); we also log `jac_neg_ratio`, `jac_det_mean`, and `jac_det_min`.
- **Loop consistency** \(L_\mathrm{loop}\): composition of adjacent forward fields over the available \(T\), including the closing edge when present, toward the identity.
- **ED-anchored path** \(L_\mathrm{ed\_ref}\): compose adjacent fields into \(\phi_{k\to\mathrm{ED}}\) / \(\phi_{\mathrm{ED}\to k}\) and apply the same inverse-consistency residual (`w_ed_ref`).

Auxiliary / demoted terms:

- **Image-cycle**: intensity residual after forward/backward warps (explicitly *not* \(L_\mathrm{inv}\)).
- **Volume-curve**: second-difference of fractional LV/RV volumes (physiological soft prior).
- **Warp intensity L1**: optional appearance alignment after pull sampling.

Optional stationary velocity fields with scaling-and-squaring are enabled via `model.use_svf: true`.

### 3.5 ED/ES segmentation

When ACDC-style ED/ES indices are available, batch fields expose `seg_frame_indices`, `segmentation_sequence`, and `seg_valid_mask`. Cross-entropy/Dice are applied only on labeled frames. The primary logits head gathers the ED index (not mid-cycle). Unlabeled phases rely on motion self-supervision and geometric regularizers.

### 3.6 Evaluation protocol

Reconstruction: PSNR, MAE, windowed **`ssim_3d`**, and global **`recon_ssim_proxy`** (never aliased as clinical SSIM). Segmentation: Dice; HD95 with optional physical spacing \((s_z,s_y,s_x)\) mm. Function: physical **EDV/ESV/EF (mL/%)** from ED/ES + spacing when available; `ef_proxy` is a voxel max/min label only. Motion: inverse-consistency residual, negative Jacobian ratio, ED↔ES label-propagation Dice/HD95 with patient-level summaries when real subjects are present.

### 3.7 HeartTTable-lite and risk heads (supplementary)

A single class token attending to concatenated spatial/temporal/table streams is retained as a fusion ablation. Publication YAML sets `w_mace: 0` and `w_cox: 0`. Binary MACE AUC on synthetic labels is logged only as a pipeline check.

---

## 4. Experiments

### 4.1 Datasets and honesty inventory

| Dataset | Role | Status |
|---------|------|--------|
| Synthetic / auto-fake | Pipeline / geometry unit tests | **DEMO** |
| Local challenge-named folders | Smoke trees (`data/acdc/`, etc.) | **DEMO placeholders** (not CREATIS challenge volumes) |
| Licensed ACDC / MM-WHS / EMIDEC | Subject-level recon–seg–motion | **待补充** (mount + train) |
| Private AMI | Five-year MACE | Out of scope |

### 4.2 Ablations and configs

Compare broadcast vs per-frame; geometry / motion weights on/off; concat vs HeartTTable-lite (supplementary). Publication configs (`publication_recon.yaml`, `publication_motion.yaml`, `publication_seg.yaml`) set `allow_fake_data: false` and hard-error empty public roots. Smoke uses dedicated smoke YAML only.

### 4.3 Implementation notes (academic)

Training and evaluation entry points live in the public repository (`scripts/train.py`, `scripts/eval.py`, `scripts/run_ablations.py`). Geometric losses are implemented in `reconseg3d/models/motion.py` and composed in `reconseg3d/models/losses.py`. No absolute local filesystem paths appear in tables or figures.

---

## 5. Results

All numbers below are **DEMO / smoke** measurements from controlled local runs. They demonstrate logging continuity, not clinical performance. Licensed subject tables: **待补充**.

### 5.1 Reconstruction ablation (DEMO)

Source: ablation summary table (`outputs/ablations_smoke_v2/table.csv`).

| Run | PSNR (dB) | MAE | Dice mean |
|-----|----------:|----:|----------:|
| broadcast_baseline | 19.08 | 0.693 | 0.168 |
| per_frame_no_motion | 19.15 | 0.692 | 0.170 |
| per_frame_motion | 19.15 | 0.695 | 0.159 |
| fusion_concat | 19.09 | 0.692 | 0.139 |
| fusion_heart_ttable | 19.26 | 0.689 | 0.189 |

Near-identical PSNR across smoke configs is expected under tiny fake tensors and short training; the panel verifies that reconstruction metrics are emitted for every ablation arm (Fig. 2).

### 5.2 Geometry and motion logs (DEMO)

Coordinate-level geometry from a joint demo evaluation (`outputs/metrics.json`):

| Metric | Value | Role |
|--------|------:|------|
| `inv_error` / `loss_inv` | 0.00177 | \(L_\mathrm{inv}\) residual |
| `loss_loop` | 0.00177 | closed-cycle composition |
| `loss_smooth` | \(6.1\times10^{-8}\) | spatial smoothness |
| `loss_jac` | 0.0 | folding penalty |
| `jac_neg_ratio` | 0.0 | fraction \(\det J < \varepsilon\) |
| `jac_det_mean` | 0.9998 | mean Jacobian determinant |
| `cycle_error` | \(2.3\times10^{-5}\) | **auxiliary** image-cycle (not \(L_\mathrm{inv}\)) |
| `recon_psnr` | 18.80 | reconstruction |
| `ssim_3d` | 0.515 | windowed SSIM |
| `recon_ssim_proxy` | 0.0098 | global proxy (not clinical SSIM) |

Ablation intensity proxies (warp L1 / image-cycle) are shown in Fig. 3a; geometry logs in Fig. 3b.

### 5.3 Segmentation smoke (DEMO)

Dice heatmap values (Fig. 4) are taken from the same ablation CSV. Under demo labels, RV Dice often collapses near zero after short training; this is a data/regime artefact, not an algorithmic upper bound.

### 5.4 Pipeline checkpoints (DEMO)

| Run | PSNR | Dice mean | Warp L1 | Image-cycle |
|-----|-----:|----------:|--------:|------------:|
| paper_recon_smoke | 19.027 | 0.0480 | 0.000294 | 0.000464 |
| paper_acdc_smoke | 17.390 | 0.0175 | 0.000086 | 0.000083 |

`paper_acdc_smoke` uses local placeholder “ACDC-named” folders, **not** licensed CREATIS volumes.

### 5.5 Unit-test evidence (non-tabular)

Identity and exact-inverse flows, +1-voxel warp direction, ED/ES index wiring, label-propagation with known translation, and HD95 spacing \((1,1,1)\) vs \((8,1,1)\) are covered by repository tests under `tests/` (geometry focus: `test_motion.py`, `test_label_propagation.py`, `test_metrics.py`). Environment note: a site-packages `numcodecs`/`zarr` conflict may block full `pytest` collection on some hosts; see evidence audit.

---

## 6. Discussion

Naming image-cycle separately from \(L_\mathrm{inv}\) prevents registration-literate misreading. Demoting volume-curve smoothing and zeroing MACE/Cox on the main path keeps the novelty claim aligned with geometry-constrained 4D recovery rather than clinical risk prediction.

**Limitations.** Compact backbones; simulated sparse SA (not \(k\)-space); licensed ACDC/M&Ms subject tables **待补充**; full high-resolution runs need GPU and licensed data; external registration baselines require additional installs; no private AMI evaluation.

---

## 7. Conclusions

We describe a geometry- and motion-constrained 4D extension of a ReconSeg3D-style public pipeline for sparse SA CMR, with ED/ES-aware supervision and an explicit refusal to claim private AMI clinical discrimination. Completing licensed subject tables and external registration baselines remains future work.

---

## Data availability

Public benchmarks follow their licenses (repository `docs/DATA.md`). No private AMI data are redistributed. Challenge-named CI folders may contain synthetic placeholders only.

## Code availability

https://github.com/Coucou2016/ReconSeg3D-cardiac-rec  

## References

[^1]: Gao, Q. et al. 3D Spatiotemporal cardiac reconstruction for predicting MACE in acute myocardial infarction. *npj Digit. Med.* **9**, 325 (2026). https://doi.org/10.1038/s41746-026-02449-0

[^2]: Hammernik, K. et al. Deep learning for accelerated and robust MRI reconstruction. *MAGMA* (2024). https://doi.org/10.1007/s10334-024-01173-8

[^3]: Qian, P. et al. Unified Deep Learning for Simultaneous Cardiac Cine MRI Reconstruction, Motion Estimation and Segmentation. *IEEE ISBI* (2024). https://doi.org/10.1109/ISBI56570.2024.10635390

[^4]: Yuan, X. et al. 4D Myocardium Reconstruction with Decoupled Motion and Shape Model. *ICCV* (2023).

[^5]: Ye, M., Xin, B., Axel, L. & Metaxas, D. Continuous Spatio-Temporal Memory Networks for 4D Cardiac Cine MRI Segmentation. In *WACV* 9514–9524 (2025). Also arXiv:2410.23191.

[^6]: Katzman, J. L. et al. DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. *BMC Med. Res. Methodol.* **18**, 24 (2018). https://doi.org/10.1186/s12874-018-0482-1

[^7]: Chen, Y., Yang, J., Mercadier, D. S., Le, H. & Fua, P. MedTet: an online motion model for 4D heart reconstruction. arXiv:2412.02589 (2024).

[^8]: Chen, Y. et al. End-to-end 4D heart mesh recovery across full-stack and sparse cardiac MRI. *Trans. Mach. Learn. Res.* (2026). Also arXiv:2509.12090.

[^9]: Balakrishnan, G. et al. VoxelMorph: a learning framework for deformable medical image registration. *IEEE TMI* (2019).

[^10]: Chen, J. et al. TransMorph: Transformer for unsupervised medical image registration. *Med. Image Anal.* (2022).

[^11]: Dalca, A. V. et al. Unsupervised learning of probabilistic diffeomorphic registration for images and surfaces. *Med. Image Anal.* (2019).

[^12]: Related multi-view / MulViMotion-style cardiac cine registration literature (citation to be finalized against the accepted venue list).

---

## Notes (not for submission body)

| Claim | Evidence | Status |
|-------|----------|--------|
| Geometry losses + ED/ES wiring exist | Code + unit tests | Supported (env permitting) |
| DEMO tables in §5 | Local `outputs/*.json` / ablation CSV | Supported; DEMO-labeled |
| Public clinical superiority | Licensed ACDC folds | Incomplete (**待补充**) |
| Private AMI AUC 0.934 / C-index 0.897 | — | Rejected / out of scope |
