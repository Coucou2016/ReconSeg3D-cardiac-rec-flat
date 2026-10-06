# 几何与运动约束的四维心脏磁共振重建与分割（ReconSeg3D）研究汇报

> **声明：** 冒烟/演示指标；不得声称私有 AMI AUC 0.934。插图：`artifacts/paper/figures/`（SciencePlots）。
> 自包含 HTML：`artifacts/report/report.html`。公开代码：https://github.com/Coucou2016/ReconSeg3D-cardiac-rec

## 1. 摘要

本汇报对应面向公开数据代理的轻量 **ReconSeg3D** 实现。研究问题：如何在公开、可审计设定下
建成几何与运动约束的 4D 重建–分割栈。**核心创新：** 逐帧三维解码、可微 pull-field warp、
**真逆一致性** L_inv、光滑/jac/loop、ED 锚定路径；image-cycle 仅为强度辅助。
**HeartTTable-lite 仅作融合消融。**
真实 ACDC/MM-WHS/EMIDEC 受试者级主表仍为**待补充**。

## 2. 研究背景与目的

### 2.1 成像背景

短轴电影 CMR 为二维切片栈；层间间隙与过平面运动使三维解剖不完整，需要稠密 4D 表示。

### 2.2 原论文边界

Gao 等（npj Digit. Med. 2026；DOI `10.1038/s41746-026-02449-0`）在私有 AMI 队列报告五年
MACE 时间依赖 AUC **0.934**。该数字绑定私有数据与完整 HeartTTable，本仓库**不复制、不宣称**。

### 2.3 方法学空隙与目的

轻量复现常广播单帧重建或缺少显式运动损失。邻近工作（Qian 等 ISBI 2024 联合重建–运动–分割；
CSTM arXiv:2410.23191 全序列 4D 分割；CineMesh4D 稀疏 cine→4D mesh）强调时间连续性。
本工作目的：(1) 默认可训练运动合同；(2) 公开/冒烟消融矩阵；(3) methods 论文架构与主张边界；
(4) 自包含 HTML 研究报告 + SciencePlots 插图。

## 3. 数据、术语与评价协议

| 术语 | 含义 | 为何引入 |
|---|---|---|
| CMR | 心脏磁共振 | 成像模态 |
| SA cine | 短轴电影序列 | 稀疏观测 |
| 4D | 3D+时间 | 心动周期 |
| PSNR / MAE | 重建保真度 | 表 T1 |
| Dice / HD95 | 分割重叠 / 边界 | HD95 物理毫米版待补充 |
| Warp / Image-cycle | 运动对齐 / 图像往返误差 | ≠ 流场逆一致 |
| EF proxy | 射血分数代理 | 功能曲线 |
| HeartTTable-lite | 单 CLS×拼接 KV | 融合消融 |
| ACDC / MM-WHS / EMIDEC | 公开基准 | 可验证代理 |

数据合同见 `docs/DATA.md`。未挂载真实数据时不得写“已在 ACDC 达到某某 Dice”。

### 3.1 本机数据盘点

| 路径 | 判定 | 证据 |
|---|---|---|
| `data/acdc/` | **Demo/假** | 8 例；4D≈0.49MB；(32,32,16,8) |
| `data/mmwhs/` / `emidec/` | **Demo/假** | 各 4 对极小 NIfTI |
| `data/ami/` | 示例清单 | 无私有 AMI |
| `outputs/ablations_smoke_v2/table.csv` | **实测冒烟** | DEMO 表可用 |
| 官方 ACDC | **受阻** | 需 CREATIS 注册 |

## 4. 思路与方法

- 张量 `(B,C,T,D,H,W)`；默认 `per_frame_recon=true`。
- MotionNet `(dz,dy,dx)` + **L_inv / smooth / Jac / loop / ED-ref**；image-cycle 仅为强度辅助。
- 分割 LV/RV/MYO；任务头 mace/phenotype/cox；融合 concat 或 heart_ttable（lite）。
- **图1**（`fig1_pipeline.png`）：流水线示意——稀疏 SA → 逐帧 3D recon → 几何运动 → ED/ES 分割；
  HeartTTable-lite 脚注为消融。物理意义：把缺层短轴栈补成可度量 4D 表示。

## 5. 研究与工程过程（来龙去脉）

1. 主张边界先于实现（`docs/PAPER_PLAN.md`）。
2. 加固：Cox 掩码、分数体积 volsmooth、risk_logits、epoch 池化 AUC。
3. 冒烟消融 `outputs/ablations_smoke_v2`（6 配置）。
4. nature-writing methods 轴 + SciencePlots 图1–5 + 本报告 bundle。
5. 顾问：既往 Plus 审计已采纳；2026-08-16 五轮 live ChatGPT **0/5**（MCP tab 消失）。
6. **2026-08-17 GitHub-MD 五轮：** 全部 brief 已推送至 `artifacts/chatgpt_handoff/github_briefs/`；
   索引 `ASK_CHATGPT.md`。Cursor IDE browser MCP 仍无法保持 tab（navigate 循环失败）；
   live 回复计数见 `reports/rounds2/` 与验收文档。独立 WebSearch + nature-skills 已落地论文改写。
7. 文献：Qian=ISBI 作者核实；Gao DOI；Ye et al. WACV 2025；增补 MedTet/TetHeart（Chen et al.）。
8. 论文/报告分离：工程路径、冒烟日记语气迁出 `MANUSCRIPT_DRAFT.md`，保留于本报告。

## 6. 结果展示与图表解读

### 6.1 冒烟消融表（demo only）

| run | PSNR | MAE | Dice mean | Warp | Cycle | EF | MACE AUC* |
|---|---:|---:|---:|---:|---:|---:|---:|
| broadcast_baseline | 19.08 | 0.6926 | 0.1682 | — | — | — | 0.6250 |
| fusion_concat | 19.09 | 0.6922 | 0.1394 | 0.0005 | 0.0001 | 0.1457 | 0.6250 |
| fusion_heart_ttable | 19.26 | 0.6895 | 0.1890 | 0.0013 | 0.0008 | 0.0676 | 0.0000 |
| per_frame_motion | 19.15 | 0.6953 | 0.1588 | 0.0011 | 0.0003 | 0.0806 | 0.6250 |
| per_frame_no_motion | 19.15 | 0.6919 | 0.1696 | 0.0010 | 0.0000 | 0.0883 | 0.6250 |
| task_phenotype | 17.39 | 0.6946 | 0.0176 | 0.0001 | 0.0001 | 0.0008 | 0.0000 |

paper_recon_smoke：PSNR=19.027，Dice mean=0.0480，warp=0.000294，cycle=0.000464。
paper_acdc_smoke：PSNR=17.390，phenotype_acc=0.0，phenotype_auc=1.0（小样本不稳定）。

### 6.2 各图来龙去脉

- **图1 `fig1_pipeline`**：方法角色示意图（非定量）。说明默认数据流与消融脚注。
- **图2 `fig2_recon_ablation`**：子图 a PSNR、b MAE。对比 broadcast / PF-no-mot / PF+mot。
  回答“运动损失是否进入重建日志”。冒烟数值接近 → **不得**解读为临床增益。真实 ACDC **待补充**。
- **图3 `fig3_motion_metrics`**：子图 a Warp L1 与 Image-cycle（强度代理）；子图 b 来自
  `outputs/metrics.json` 的 L_inv / L_loop / Jac− ratio。Image-cycle ≠ inverse-consistent flow。
- **图4 `fig4_seg_dice`**：LV/RV/MYO/Mean Dice 热图。冒烟下 RV≈0 反映欠训练/假标签，非算法上界。
  物理毫米 HD95 **待补充**。
- **图5 `fig5_claim_boundary`**：主张支持度条形图。公开 recon/seg/几何运动 4D 在范围内；
  HeartTTable-lite 为部分支持（消融）；私有 AMI 0.934 **out of scope**。
- **审查文档**：`docs/paper/EVIDENCE_AUDIT.md`（指标↔文件↔代码入口 1:1）。
- **用户主线稿**：本回合未找到 → `docs/paper/AWAITING_USER_MANUSCRIPT.md`。

## 7. 讨论

运动项使时间一致性可消融、可审计。贡献不在追赶最大 Dice，而在轻量 ReconSeg3D 合同上显式落地
warp + image-cycle 并公开边界。审稿忌：术语偷换、lite=完整 HeartTTable、冒烟当临床。

## 8. 主要结论

1. 运动一致 4D 是可辩护主创新。  
2. HeartTTable-lite 仅消融。  
3. 冒烟证明流水线贯通，不是公开基准上界。  
4. 私有 AMI 0.934 永远不是本仓库结果。

## 9. 局限与待补充

- 真实 ACDC/MM-WHS/EMIDEC 主表：**待补充**
- 物理毫米 HD95、嵌套 CV、校准曲线：**待补充**
- 紧凑 CNN ≠ 原论文 256³ ViT/nnU-Net
- 2026-08-17 live ChatGPT：见验收 `20260817_five_round_github_md_acceptance.md`；
  brief 索引：`artifacts/chatgpt_handoff/github_briefs/ASK_CHATGPT.md`

## 10. 顾问通道与公开仓库

- 对话：https://chatgpt.com/c/6a808651-37cc-83ea-a63d-2d2539a48d07
- 公开 GitHub（代码+文档，无大 data/outputs）：https://github.com/Coucou2016/ReconSeg3D-cardiac-rec
- 五轮记录：`artifacts/chatgpt_handoff/reports/rounds/`
- 验收：`artifacts/chatgpt_handoff/reports/20260816_five_round_acceptance.md`
