# 这个仓库为什么故意不建文件夹（Flat Snapshot）

**一句话：为了让 ChatGPT / 其他 Agent / 任何审查工具在“一次目录列表”里就能看到全部代码、文档、论文、报告和核心结果数据，本仓库刻意取消了所有文件夹层级，全部文件放在根目录下。**

本文件就是那份“说明是故意这样放置”的文档。如果你的唯一目标是阅读与交叉审查（而不是直接 `pip install` 或 `pytest`），这种摆放方式最省事；如果你需要还原成可运行的工程结构，见文末《如何还原文件夹结构》。

---

## 1. 为什么扁平化：为“机器可读 + 交叉审查”优化

常规仓库把内容分层塞进 `reconseg3d/`、`docs/`、`artifacts/` 等目录。对人和 IDE 都友好，但对“让另一个模型整仓审阅”并不友好：模型往往只能看到目录列表，需要逐层展开、逐个抓取，容易漏文件，也难以建立全局印象（尤其当目录名相似、文件同名时）。

把全部文件压到根目录，带来几个直接好处：

1. **一次 `get_tree` 就能拿到完整清单**：235 个文件全部在顶层，不存在“目录里还有目录”的递归盲区。
2. **便于并行抓取与对齐**：审查方可以按文件名前缀（`reconseg3d__`、`artifacts__`、`tests__`）一次性定位同类内容。
3. **同名文件不会互相覆盖**：原路径被编码进文件名，`docs/paper/README.md` 与 `reconseg3d/data/README.md` 变成两个不同文件而不是互相遮蔽。
4. **可追溯**：每个扁平文件名都能反推出它原本的位置，配合 `FLAT_FILEMAP.json` 可逐字节核对。
5. **适合“交叉审查”场景**：两三个模型各自读同一份扁平文件集合，结论可以直接按文件名对齐，减少“你说的是哪个文件”的歧义。

---

## 2. 命名规则：`原目录__子目录__文件名`

扁平化使用双下划线 `__` 作为路径分隔符：

| 扁平文件名 | 原本的仓库路径 |
|---|---|
| `reconseg3d__models__motion.py` | `reconseg3d/models/motion.py` |
| `reconseg3d__data__acdc.py` | `reconseg3d/data/acdc.py` |
| `docs__paper__MANUSCRIPT_DRAFT.md` | `docs/paper/MANUSCRIPT_DRAFT.md` |
| `artifacts__report__report.html` | `artifacts/report/report.html` |
| `tests__test_motion.py` | `tests/test_motion.py` |
| `configs__publication__motion.yaml` | `configs/publication/motion.yaml` |
| `README.md` | `README.md`（本来就在根目录，保持原样） |

选 `__` 而不是 `_` 或 `-`，是因为 Python 模块里 `_` 太常见（`model_forward`、`test_paper_pipeline`），用双下划线才能稳定反解、不产生歧义。

完整的一对一映射（文件名 → 原始路径 → 字节数）记录在 `FLAT_FILEMAP.json` 中，共 235 条，可程序化校验。

---

## 3. 包含什么

四类内容，覆盖“读懂这项工作”所需的全部材料：

1. **主要代码**：`reconseg3d__*.py`（模型、数据、损失、训练、指标、推理）、`scripts__*.py`（数据准备、训练、评测、消融、多随机种子、出图、报告打包）、`tests__*.py`。
2. **文档**：`README.md`、`docs__*.md`、`docs__paper__*.md`（写作框架、手稿、证据审查文档等）。
3. **论文与报告**：手稿与研究报告的 **Markdown / HTML / PDF** 三种格式，含自包含单文件 `artifacts__report__report.html`（CSS 内联、图片 Base64、表格为 HTML 表格、无外部 CDN）。
4. **核心结果数据**：`outputs__**__*.json / *.csv / *.yaml / *.md`，即各次实验的指标文件、消融汇总表、患者级 `case_metrics.csv`、`summary_metrics.json`、`bootstrap_ci.json` 等小型可解析文件；以及 `artifacts__paper__figures__*`（SciencePlots 生成的 PNG/PDF/SVG 图）。

---

## 4. 刻意不包含什么

为保持仓库轻量、可公开、且不误传敏感或占体积内容，以下内容**故意排除**：

| 排除项 | 原因 |
|---|---|
| `*.pt` / `*.pth` / `*.ckpt` 权重（约 36.6 MB） | 体积大且非“可读材料”；结果表中的每个数字都能由仓库内代码+指标文件复算 |
| `data/`（约 5.1 MB） | 其中 ACDC / MM-WHS / EMIDEC 目录为**演示用合成数据**，不是官方数据；避免被误当成真实数据 |
| `predictions/` | 推理产物，可由代码重生成 |
| `.git`、缓存、虚拟环境、ZIP 交接包 | 与阅读审查无关 |
| 任何凭据、Cookie、Token、`.env` | 安全边界：不离开本地环境 |

因此，本仓库是**阅读与审查快照**，不是“一键复现训练”的发行版。

---

## 5. 阅读顺序建议（给 Agent / 审稿人）

1. `README.md` — 项目定位与声明边界。
2. `FLAT_LAYOUT.md`（本文件）+ `FLAT_FILEMAP.json` — 布局说明与路径映射。
3. `docs__paper__WRITING_FRAMEWORK.md` — 论文框架与创新点定位。
4. `docs__paper__MANUSCRIPT_DRAFT.md`（或同内容 HTML/PDF）— 论文正文。
5. `docs__paper__EVIDENCE_AUDIT.md` — **真实性审查文档**：论文/报告中每个数字对应哪个指标文件、由哪段代码算出、哪些属于 DEMO。
6. `artifacts__report__report.html` — 单文件科研报告（自包含，可直接双击打开）。
7. `reconseg3d__models__motion.py`、`reconseg3d__models__losses.py`、`reconseg3d__training__metrics.py`、`reconseg3d__training__trainer.py` — 方法核心实现。
8. `tests__test_motion.py`、`tests__test_metrics.py`、`tests__test_publication_gates.py` — 科学正确性测试（几何一致性、单位、泄漏防护）。

---

## 6. 关于数据与结论的诚实声明

* 本仓库**不声称**复现原始 npj Digital Medicine 论文的私有 AMI 队列结果（5 年 time-dependent AUC 0.934、C-index 0.897）。该结论依赖未公开的多中心真实 AMI 生存队列，公开数据无法重建该证据链。
* 仓库内 ACDC / MM-WHS / EMIDEC 相关目录为**合成演示数据**；所有基于它们的指标均标注为 DEMO / smoke，仅用于验证管线可运行，不代表临床性能。
* 未获得许可的公开数据集（如需注册下载的官方 ACDC、M&Ms）对应表格在论文中标注为 **待补充**，绝不臆造数值。

---

## 7. 如何还原文件夹结构

`FLAT_FILEMAP.json` 里每个条目都记录了 `original_path`。还原方式：

```python
import json, pathlib, shutil
mapping = json.load(open("FLAT_FILEMAP.json", encoding="utf-8"))
for flat_name, meta in mapping["files"].items():
    dst = pathlib.Path("restored") / meta["original_path"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(flat_name, dst)
```

还原后即为与源仓库一致的目录结构：

* `reconseg3d/`
* `configs/`
* `scripts/`
* `tests/`
* `docs/`
* `artifacts/`（含 `paper/`、`report/`、`chatgpt_handoff/`）

源仓库（保留标准目录结构、可用于实际运行）：
<https://github.com/Coucou2016/ReconSeg3D-cardiac-rec>

---

## 8. 维护约定

* 该扁平快照由脚本生成，脚本逻辑：读取源仓库 `git ls-files` 全部受控文件，附加 `outputs/` 下的 `.json/.csv/.yaml/.md/.txt` 小型结果文件，按 `__` 规则压平到根目录，并输出 `FLAT_FILEMAP.json`。
* 若源仓库更新，重新生成快照即可；不需要手工搬文件。
* 本仓库不接收文件夹形式的贡献（保持扁平是有意为之）；结构性变更请提交到源仓库。
