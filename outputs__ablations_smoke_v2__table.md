# Ablation summary (demo / public proxies)

Not private AMI MACE. Empty cells mean the metric was not logged for that run.
Demo / smoke numbers are pipeline checks only — do not cite as clinical performance.

| run | recon_psnr | recon_ssim | recon_mae | dice_lv | dice_rv | dice_myo | dice_mean | warp_error | cycle_error | ef_proxy | phenotype_acc | phenotype_auc | mace_auc | c_index | loss_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broadcast_baseline | 19.0751 | 0.0130 | 0.6926 | 0.2266 | 0.0000 | 0.2779 | 0.1682 |  |  |  |  |  | 0.6250 | 0.5556 |  |
| fusion_concat | 19.0927 | 0.0136 | 0.6922 | 0.1835 | 0.0000 | 0.2348 | 0.1394 | 0.0005 | 0.0001 | 0.1457 |  |  | 0.6250 | 0.5556 |  |
| fusion_heart_ttable | 19.2604 | 0.0184 | 0.6895 | 0.0000 | 0.0000 | 0.5671 | 0.1890 | 0.0013 | 0.0008 | 0.0676 |  |  | 0.0000 | 0.1667 |  |
| per_frame_motion | 19.1478 | 0.0149 | 0.6953 | 0.0375 | 0.0000 | 0.4390 | 0.1588 | 0.0011 | 0.0003 | 0.0806 |  |  | 0.6250 | 0.5556 |  |
| per_frame_no_motion | 19.1506 | 0.0153 | 0.6919 | 0.0137 | 0.0000 | 0.4951 | 0.1696 | 0.0010 | 0.0000 | 0.0883 |  |  | 0.6250 | 0.5556 |  |
| task_phenotype | 17.3900 | 0.0044 | 0.6946 | 0.0527 | 0.0000 | 0.0000 | 0.0176 | 0.0001 | 0.0001 | 0.0008 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |  |
