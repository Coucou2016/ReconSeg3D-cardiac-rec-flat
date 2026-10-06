# Baseline comparison notes (no invented numbers)

This repository ships **adapters** for fair small-resolution comparisons. It does
**not** claim published Dice/HD95/AUC against VoxelMorph / TransMorph / FlowReg
without a completed local run on licensed data.

## In-repo baselines

| Name | Module | Config hint |
|------|--------|-------------|
| Reconstruction-only / no-motion | `reconseg3d.models.baselines.ReconOnlyBaseline` | `configs/baselines/recon_only.yaml` |
| Compact VoxelMorph-style | `CompactVoxelMorph` | `configs/baselines/voxelmorph_compact.yaml` |
| TemporalConv (default) | `TemporalEncoder(temporal_mode=temporal_conv)` | publication configs |
| ConvLSTM3D | `temporal_mode: conv_lstm` | `configs/ablations/temporal_conv_lstm.yaml` |
| Temporal attention | `temporal_mode: temporal_attention` | `configs/ablations/temporal_attention.yaml` |
| SVF + scaling-and-squaring | `model.use_svf: true` | `configs/publication/publication_motion_svf.yaml` |

```powershell
python -c "from reconseg3d.models.baselines import build_baseline; m=build_baseline('voxelmorph'); print(m)"
```

## External: FlowReg

Upstream: https://github.com/mathpluscode/FlowReg

1. Clone / install FlowReg in a **separate** environment (not vendored here).
2. Implement `FlowRegAdapter.load` / `predict` against your local install.
3. Run pairwise registration on the same resampled grid as this repo.
4. Only then fill comparison tables — leave cells **待补充** until then.

## Related-work positioning (locked)

| Method | Role vs this repo |
|--------|-------------------|
| VoxelMorph | Classic CNN registration; our compact baseline for small-grid fairness |
| TransMorph | Transformer registration; higher capacity — tracked external compare |
| FlowReg | Flow-based cardiac registration; adapter-only until installed |
| Neural ODE / continuous flows | Related continuous-time motion; not our primary discrete closed-cycle claim |
| TetHeart | Mesh/tetrahedral cardiac dynamics; complementary representation |
| Gao et al. ReconSeg3D + HeartTTable | Original recon→seg→MACE line; we extend **geometry/motion** publicly and **do not claim 0.934** |

## Honesty

- Smoke / DEMO metrics ≠ publication tables.
- M&Ms / licensed ACDC subject tables remain **待补充** without data on disk.
