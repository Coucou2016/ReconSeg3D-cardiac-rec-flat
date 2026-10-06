# Ablation summary (demo / public proxies)

Not private AMI MACE. Empty cells mean the metric was not logged for that run.

| run | recon_psnr | recon_ssim | recon_mae | dice_lv | dice_rv | dice_myo | dice_mean | warp_error | cycle_error | ef_proxy | phenotype_acc | phenotype_auc | mace_auc | c_index | loss_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broadcast_baseline | 19.1099 | 0.0001 | 0.6909 | 0.2498 | 0.0000 | 0.1626 | 0.1375 |  |  |  |  |  | 0.5000 | 0.5000 | 2.3991 |
| per_frame_motion | 19.0326 | 0.0000 | 0.6964 | 0.0885 | 0.0000 | 0.1287 | 0.0724 | 0.0012 | 0.0022 | 0.3486 |  |  | 0.5000 | 0.5000 | 4.5474 |
| task_phenotype | 17.3896 | 0.0000 | 0.6946 | 0.0525 | 0.0000 | 0.0000 | 0.0175 | 0.0001 | 0.0001 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 4.3670 |
