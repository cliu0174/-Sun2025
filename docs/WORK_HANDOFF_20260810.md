# PI-MSCL 论文交接说明

**更新日期：** 2026-08-17  
**协作方式：** 由作者人工审阅并决定修改方向；助手负责定位内容、提出可选修改、落实批准的修改和编译核验。本文档只记录当前材料与关键边界，不替代作者的写作和排版判断。

## 当前材料

- 当前主稿源码（后续润色的唯一准本）：`overleaf_submission_20260813/main.tex`
- 旧版镜像：`latex/springer-sn-2024/manuscript.tex`（不要再与主稿混合修改）
- 可直接上传 Overleaf 的独立包：`overleaf_submission_20260813/`
  - 仅含一个正文文件 `main.tex`
  - 图片与 `main.tex` 同级，命名为 `Figure1.png`--`Figure6.png`
- 参考文献：`overleaf_submission_20260813/references.bib`
- 既有审稿报告：`C:/Users/46139/.codex/visualizations/2026/08/10/019feac9-7a33-78e0-8023-25d14d117b33/manuscript_review/PI_MSCL_manuscript_comprehensive_review_20260810.md`

## 当前研究口径

- 数据：HUST 公开数据集，77 个 LFP/graphite 电芯；输入为 16 个充电循环特征。
- 任务：跨电芯 SOH 估计；采用 Li 式“轨迹集中监督”设定：固定全局标签预算按随机电芯顺序尽量集中地顺序分配给少数训练电芯。先完整标注若干电芯，再对最后一个电芯保留连续早期/中期前缀；其余训练电芯及该电芯后续循环均无 SOH 标签，但全部充电特征和循环顺序仍保留。
- 正式主实验标签预算：30%、10%、5%；固定电芯划分种子 42，三个配对重复为训练种子 `[929, 2262, 7]` 与标签分配种子 `[1929, 3262, 1007]`。
- 论文中的物理约束当前只保留软单调项（$\lambda_{\mathrm{mono}}=0.3$，$\epsilon=0.005$，$c_{\min}=300$）；变化率连续性未进入本轮正式主实验。
- `overleaf_submission_20260813/main.tex` 已于 2026-08-13 替换为真实结果并通过本地编译。正文不能再沿用“PI-MSCL 在所有低标签条件下带来精度增益”的旧结论。

## 审阅与修改约定

1. 作者优先按自己的判断审阅正文、图表、逻辑和格式；助手根据具体意见协助修改，不额外强制统一文风或章节样式。
2. 修改涉及数值、结论、实验范围或模型配置时，先核对数据来源；未经作者明确同意，不启动新的训练或替换主结果。
3. 主稿中仍有 `\iffalse ... \fi` 包裹的历史内容，未进入当前 PDF。是否保留、迁移或删除，由作者后续决定。
4. 投稿前需要作者确认目标期刊、作者与声明信息，以及代码可用性表述。

## Overleaf 同步约定（2026-08-17 起执行）

1. 作者在 Overleaf 所作的最新修改为优先版本。作者发送新的 Overleaf 内容或导出包后，先与本地版本逐项比对再合并，不能以本地旧版本覆盖作者的新修改。
2. 任何助手实施的正文、参考文献、表格或图片修改，必须同步至本地对应源文件和可上传目录 `overleaf_submission_20260813/`；图形修改还须同步至正式输出目录 `output/figures/formal_main_v3/`。
3. 修改 `main.tex`、`references.bib` 或被正文引用的图后，需编译 `overleaf_submission_20260813/main.tex`，并核验不存在致命编译错误、未定义引文或未定义交叉引用。
4. 每次图片更新的交付信息必须包含：图片预览、源图与 Overleaf 图路径、已更新目录、正文 Figure 编号、图片信息摘要，以及可直接用于正文的英文描述建议。
5. 每轮对话结束前，按实际完成内容更新本交接文档，注明已同步文件、编译状态与未完成事项。

## 常用辅助路径

- 正式图表：`output/figures/formal_main_v3/`
- 架构图源文件：`docs/PI-MSCL_architecture_single_column.drawio`
- 历史实验计划：`docs/IMPROVEMENT_PLAN.md`
- 中文参考稿：`latex/els-cas-templates/soh_cn_version.docx`

## 2026-08-13：轨迹集中监督正式实验（必须优先阅读）

### 已完成范围与可复现位置

- 已完成 54 组正式运行：36 组 $2\times2$ 主消融（4 模型 $\times$ 3 标签预算 $\times$ 3 重复）、6 组 $\lambda_{\mathrm{mono}}$ 敏感性、12 组同协议基线。
- 主实验、逐窗口预测、标签分配和轨迹指标：`experiments/paper_main_results/fixed_split/v4_label_protocols/trajectory_prefix_concentrated/`
- 同协议 XGBoost、LSTM、GRU、Attention CNN--LSTM：`experiments/paper_baselines/fixed_split/v2_trajectory_prefix_concentrated/`
- 每次运行均包含 `result.json`、`predictions.npz`、`trajectory_metrics.json`、`run_manifest.json`；神经网络运行还包含 `battery_split.json`，其中 `supervision_metadata` 给出具体完整/部分标注电芯。
- 目录命名已修复：5%、10%、30% 分别为 `r0p05`、`r0p1`、`r0p3`。不要混用修复前因一位小数导致 5%/10% 冲突的旧归档。

### 真实主结果：测试 MAE（%，均值 $\pm$ 样本标准差，$n=3$）

| 模型 | 30% | 10% | 5% |
|---|---:|---:|---:|
| 单尺度 CNN--LSTM | 1.7166 $\pm$ 0.1734 | 2.5318 $\pm$ 0.4265 | **2.5802 $\pm$ 0.4754** |
| 单尺度 PI-CNN--LSTM，$\lambda=0.3$ | 2.2847 $\pm$ 0.4808 | 3.8212 $\pm$ 1.0291 | 5.6577 $\pm$ 2.3801 |
| 多尺度 CNN--LSTM | **1.6540 $\pm$ 0.1548** | **2.2701 $\pm$ 0.2092** | 2.8744 $\pm$ 0.7338 |
| PI-MSCL，$\lambda=0.3$ | 2.5715 $\pm$ 0.1536 | 3.4983 $\pm$ 0.7225 | 5.2853 $\pm$ 1.9769 |

### 同协议 10% 基线：测试 MAE（%，$n=3$）

| 模型 | MAE |
|---|---:|
| 多尺度 CNN--LSTM | **2.2701 $\pm$ 0.2092** |
| Attention CNN--LSTM | 2.3279 $\pm$ 0.2514 |
| LSTM | 2.5099 $\pm$ 0.3489 |
| GRU | 3.2342 $\pm$ 1.7331 |
| XGBoost | 3.2883 $\pm$ 0.4248 |
| PI-MSCL，$\lambda=0.3$ | 3.4983 $\pm$ 0.7225 |

### 已证实的结论与叙事边界

1. 新协议有效地构造了“少数完整/连续前缀参考轨迹 + 大量无标签轨迹”的标签覆盖偏移；这才是正文需要严格对齐 Li 论文的研究背景。
2. 多尺度表征在 30% 和 10% 标签预算下提供最好的点估计；但 5% 下单尺度无物理模型略优，不能写成“多尺度在所有预算下都优”。
3. 现有软单调先验并未在该协议下稳定改善 MAE。5% 的多尺度 $\lambda$ 扫描为：$\lambda=0$，2.8744 $\pm$ 0.7338；$\lambda=0.1$，3.7190 $\pm$ 0.9295；$\lambda=0.3$，5.2853 $\pm$ 1.9769；$\lambda=0.5$，5.1803 $\pm$ 0.9635。
4. 因此，不可把“物理约束提高精度”写为主结论。可如实写为：固定强度的软单调先验在标签集中覆盖偏移下会产生过度正则化，必须重新标定或设计自适应机制；这是当前方法的限制与后续工作，而不是已被解决的问题。
5. 5% 时，$\lambda=0.3$ 的 PI-MSCL 也未稳定降低轨迹指标：其平均单调违规率为 0.1324%，而无物理多尺度模型为 0.0575%。因此暂不能将其包装为“以精度换轨迹合理性”的正面证据。

### 新对话的推荐起点

- 先阅读 `overleaf_submission_20260813/main.tex` 的现有五章结构和真实结果表。旧的 Figure1、Figure3--Figure6 为未引用的历史文件，不能重新加入正文。
- 润色时应将主线收敛为：**在 Li 式有限参考容量测试场景下，研究集中连续标签覆盖偏移；多尺度充电表征是经证据支持的有效部分；物理先验的失配与参考轨迹选择敏感性是必须透明讨论的限制。**
- 如需继续保留 PI-MSCL 作为论文主模型，需要先获得作者同意后再开发并验证新的先验自适应/权重重标定方法；不能以当前 $\lambda=0.3$ 结果支撑其优越性。

## 2026-08-17：Figure 1 容量轨迹图更新

### 已完成修改

- 图形生成脚本：`scripts/plot_hust_all_discharge_capacity.py`。使用 HUST 的 77 条逐循环放电容量轨迹，以各电芯前 20 个可用循环容量的均值归一化；使用低饱和度多色轨迹且不设置图例，颜色仅用于区分重叠曲线，不代表电芯类别。
- 图形显示范围：横轴保留全部观测寿命，动态显示至最长电芯（当前约 2670 圈）；纵轴显示范围更新为 75%--102%，低于 75% 的尾部在图中裁去。此前错误实施的“横轴截断至 2000 圈”已撤销，当前版本不再截断横轴。
- 正式本地图：`output/figures/formal_main_v3/fig2_hust_all_discharge_capacity.{png,pdf,svg,tiff}`；对应元数据：`output/figures/formal_main_v3/fig2_hust_all_discharge_capacity_source.json`。
- Overleaf 图：`overleaf_submission_20260813/Figure2.png`，已同步更新。尽管文件名为 `Figure2.png`，该文件在 `main.tex` 第 75 行被引用，正文编号为 **Figure 1**。
- 正文 PDF：`overleaf_submission_20260813/main.pdf` 已于本轮修改后重新编译成功（16 页）；未报告致命编译错误。现有 underfull box 提示为模板排版警告，非编译失败。

### 当前图的推荐正文表述

> Figure 1 shows the normalised discharge-capacity trajectories of the 77 HUST cells. The trajectories exhibit substantial cell-to-cell heterogeneity in degradation rate and lifetime, motivating battery-level data partitioning and the trajectory-concentrated supervision setting used in this study.

### 后续注意事项

- 后续若作者在 Overleaf 修改了 `main.tex`、图题或图片文件，须以作者最新导出内容为准，再将同等改动同步回上述本地路径。
- `Figure1.png` 与 `Figure3.png`--`Figure6.png` 仍为未被当前正文引用的历史文件；不可仅因其存在于上传目录而恢复引用。

## 2026-08-18：方法实现核查与 SOH 标签边界待修复（尚未改代码或正文）

### 本轮已核实的实现事实

- 正式 v4 主实验使用 `ms_cnn_lstm_v2`：三路并行 CNN（kernel 3/7/15），每路两个 `Conv1d--BN--ReLU--MaxPool(2,2)` block，通道均为 64→64；fusion 为 `1×1 Conv(192→128)--BN--ReLU`，无 fusion dropout；两次池化使 40 个输入循环变为 10 个时间步，再送入两层 LSTM（hidden=64）。
- 网络末端为 FC 后 `Sigmoid`，没有 `clip`/`clamp`，故预测严格位于 `(0,1)`；正式 runner 的 `boundary_weight=0`、`smoothness_weight=0`。
- 软单调损失并非固定全轨迹配对：训练 loader 以 `shuffle=True` 生成 batch，代码在每个 batch 内枚举所有候选两两组合，仅保留同电芯、后者循环更晚、`1≤Δc≤K=40`，且较早点 `c_j≥c_min=300` 的对。无论样本标签是否被 mask，预测均可进入该约束。
- 衰减权重为 `exp(-0.2 Δc)`，其中 `Δc` 是窗口目标所对应的原始循环索引差，而不是窗口内部的局部位置；因此有效权重集中在很短的局部范围。
- `c_min=300` 的历史动机是避免早期容量恢复阶段被单调约束错误惩罚；但“300”本身尚无正式敏感性/数据驱动选定证据。`K=40` 与窗口长度相同，也没有现存的 K 敏感性证据。
- checkpoint 和 early stopping 在当前正式 v4 默认按 validation MAE 选择，patience=20；配置中虽有 `min_delta=1e-5`，但当前比较逻辑未使用它。
- 正式 v4 主结果与 λ 敏感性共 43 个归档 run 的 `smoothness_weight` 都为 0。因此应表述为 `L_rate` **已从正式优化目标移除**，而不是从代码路径完全删除（PI 路径仍会计算/记录该项，但其权重为 0、无梯度贡献）。

### SOH 定义与数据的已确认冲突

- 正式调用链为 `run_paper_main_results.py → train_cross_battery.py → load_all_batteries() → load_single_hust_battery(train_ratio=1.0, normalize_target=True)`；论文 runner 未启用 3-sigma cleaning（`apply_cleaning=False`）。
- 当前 loader 只执行 `capacity / capacity[0]`，没有对 target 进行 `clip`、`clamp` 或任何上界处理；窗口化与监督 mask 也不会改变 target 值。
- 对 77 个 CSV 的直接审计：45 个电芯至少出现一个归一化容量比大于 1；全数据最大值为 `1.01107137`。这不是 Figure 1 的显示基准造成的假象。
- Figure 1 的脚本使用“前 20 个可用循环容量的平均值”归一化，训练标签使用“第一个循环容量”归一化；两种参考基准不同，不能将 Figure 1 的数值直接等同于训练标签。
- 但正式 v4 保存的预测进一步证实矛盾：例如 `trajectory_prefix_concentrated/multi_pi/r0p3/split42_mask1007_train7/predictions.npz` 中，30,878 个测试目标的范围为 `0.71492481–1.00713336`，其中 286 个大于 1；对应 Sigmoid 预测最大值为 `0.98072332`。

### 已建议、但尚未执行的修复路线

- **推荐路线（待作者确认）：** 保留有界 SOH 的研究口径与 Sigmoid，在 `load_single_hust_battery()` 中以 `y = clip(capacity / capacity[0], 0, 1)` 明确构造 SOH 标签；将超出 1 的原始归一化容量视为早期容量恢复/测量波动，而非 SOH 大于 100%。
- 不建议以整段生命周期的最大容量作为归一化参考，因为会引入未来信息；也不应只改正文定义而保留未截断的标签和 Sigmoid。
- 修复后必须：对 train/val/test 同步处理；新增 target 范围断言和被截断样本计数；区分 Figure 1 的原始容量轨迹与训练 SOH 标签口径；重跑所有最终进入正文的正式表格与图，不能继续使用当前数值。
- 另一路（保留原始容量比并移除 Sigmoid）在技术上可行，但会把文章从“有界 SOH 估计”改为“归一化容量比回归”，不建议作为当前 Ionics 投稿主线。

### 当前禁止事项

- 作者本轮明确要求先核对问题，故截至交接时**没有**修改 `train_cross_battery.py`、loader、正文、Overleaf 包或实验结果；**没有**启动重跑。
- 后续新对话应先由作者确认是否实施“标签 clip + 全部正式实验重跑”，再改代码和同步正文/Overleaf 文件。

## 2026-08-18：审查意见驱动的架构图候选版（未插入主稿）

### 已完成

- 根据作者提供的绘图 prompt 与审查意见，基于原图另存可编辑候选源图：`docs/PI-MSCL_architecture_reviewed_20260818.drawio`。
- 预览输出：`docs/PI-MSCL_architecture_reviewed_20260818_preview.{svg,png}`；其中 Panel (a) 显式画出三条 `k=3/7/15` 分支的两次 `Conv1D(64)–BN–ReLU–MaxPool(2)`，以及 `40×14→20×64→10×64` 的尺寸变化、融合、LSTM 和回归头。Panel (b) 以模型预测为共同输入，分出受掩监督、局部有序预测单调约束与局部速率连续性三条路径。
- 上述 `.drawio`、`.svg` 与 `.png` 已逐字节同步至 `overleaf_submission_20260813/` 同名文件；SHA-256 已核对一致。
- 本轮没有修改 `main.tex`、`references.bib` 或任何正文引用图，故未触发 LaTeX 编译；候选图尚未被主稿引用。

### 作者指定的候选图口径及已知不一致

- 候选图按审查意见标注 `40 cycles × 14 features`，但当前正式研究/交接口径为 **16 个充电特征**。这张候选图不能直接替换正文 Figure，除非正文与实验输入维度一并确认/修正。
- 候选图按审查意见写入 `L = L_sup + λ_mono L_mono + λ_rate L_rate`，并将 `L_rate` 作为训练目标的一项；正式 v4 主实验中 `smoothness_weight=0`，`L_rate` 没有梯度贡献。
- 候选图的受监督目标采用 `y=min(Q_i,t/Q_i,0,1)`，以符合作者提供的有界 SOH 绘图 prompt；当前正式 loader 尚未实施 clip，已有已归档正式目标大于 1。因此该标注代表建议中的修复口径，尚非现有正式结果的实现事实。

## 2026-08-18：图形优先的架构图候选版（未插入主稿）

### 已完成

- 作者要求恢复审查文字化修改前的图形表达，并参考 CNN--LSTM / BiLSTM 同类论文的紧凑视觉语法；已据此另存可编辑源图：`docs/PI-MSCL_architecture_visual_20260818.drawio`。原始 `PI-MSCL_architecture_single_column.drawio` 和审查版均未覆盖。
- 新版以输入小波形、三组叠放特征图、简化 LSTM 时步节点、预测 SOH 小曲线和稀疏标签点承载信息；上半区仅描述模型流，下半区仅保留正式启用的 masked supervision 与 soft monotonicity 两条训练路径。模块文字已压缩为共享卷积说明、kernel、融合维度、`h=64` 和损失符号。
- 口径按当前正式实现绘制：`40 × 16` charging features、三路 kernel `3/7/15`、两层 LSTM（`h=64`）、`L = L_sup + λ_mono L_mono`；未将目前权重为 0 的 `L_rate` 画入训练目标，也未把标签掩码误画为网络输入。
- 预览：`docs/PI-MSCL_architecture_visual_20260818_preview.{svg,png}`。上述 `.drawio`、`.svg` 和 `.png` 已同步到 `overleaf_submission_20260813/` 同名文件，并通过 SHA-256 一致性核验。
- 本轮未修改 `main.tex`、`references.bib` 或正文引用图片，新候选图尚无 Figure 编号，故无需触发 LaTeX 编译。正式主稿图的选用与图题仍待作者确认。

## 2026-08-18：按新绘图 prompt 重绘的图形化架构图候选版（未插入主稿）

### 已完成

- 按作者最新提供的 draw.io 绘图 prompt 和参考图重新绘制，并另存原生可编辑源图：`docs/PI-MSCL_architecture_prompt_graphical_20260818.drawio`；未覆盖单栏原图、审查版或上一版图形优先候选图。
- Panel (a) 采用矩阵输入、三路尺寸递减的叠放特征图、融合张量、两层显式 LSTM 单元、回归节点和预测 SOH 曲线；Panel (b) 采用稀疏保留标签点、带局部上升违规的有序预测轨迹、带斜率突变的局部轨迹及三路损失汇合。主要模块及内部元素均由原生 draw.io 矢量形状、文本和连接器组成，并按 input、各 CNN branch、fusion、LSTM、regression、output、各 loss pathway 和 objective 分组。
- 本地质检预览：`docs/PI-MSCL_architecture_prompt_graphical_20260818_preview.{svg,png}`。draw.io 结构验证结果为 **0 errors**；验证器报告的 overlap 主要来自有意叠放的特征图、容器内元素和轨迹强调圈，不是扁平图片或结构损坏。
- 上述 `.drawio`、`.svg` 与 `.png` 已同步至 `overleaf_submission_20260813/` 同名文件，SHA-256 逐一核验一致。
- 本轮未修改 `main.tex`、`references.bib`、现有 Figure 引用或正文内容，因此未触发 LaTeX 编译；该图仍是供作者审阅的独立候选稿。

### prompt 口径与正式实现的已知差异

- 新 prompt 明确要求输入标为 `40 × 14`，本候选图据此绘制；当前正式研究和 `AGENTS.md` 口径仍是 **16 维充电特征**。
- 新 prompt 明确要求画出 `Rate continuity / L_rate` 并写入总目标，本候选图据此保留三条损失路径；当前正式 v4 归档实验的 `smoothness_weight=0`，故 `L_rate` 在正式训练中无梯度贡献。
- 输出按 prompt 标为 `ŷ_i,t ∈ (0,1)`；这与 Sigmoid 网络输出一致，但当前 loader 尚未对少量大于 1 的归一化 target 实施 clip。该图不能在上述口径确认前直接替换主稿正式 Figure。

## 2026-08-18：参考架构图的单页可编辑 PPTX 重建

### 已完成

- 使用 `img2pptx` 工作流按作者提供的 1672 × 941 PNG 参考图进行单页矢量重建，输出目录为 `outputs/img2pptx_pi_mscl_20260818/`；核心文件为 `final.pptx`、`full.svg`、`component_manifest.json` 以及 12 个 `modules/*.svg` 独立模块。
- `final.pptx` 仅含 1 页，并以与原图一致的 1672:941 画幅嵌入完整 SVG；SVG 仅使用矢量形状、路径、文本和语义分组，不含 raster `<image>`、`foreignObject` 或滤镜。PPTX 包审计已核对内嵌 SVG 与 `full.svg` 的 SHA-256 完全一致。
- 已完成结构、独立模块、包含关系、对齐、边框层级、语义约束、约束覆盖、PPTX 包和 PowerPoint 实际导出预览审计，硬性检查全部通过。独立视觉复核发现的输入矩阵行序、分支主干颜色与拓扑、三尺度汇聚、融合后 10 步序列语义以及输出箭头遮挡刻度等问题均已修正。
- 在硬约束全部通过后进行了 3 次有界视觉优化，整图 MAE 从 11.7625 降至 11.6803，改善约 0.70%；未达到 10% 的软目标，因此按 `max_try=3` 停止并交付无硬错误的最佳候选稿。审计证据位于 `qa/`，PowerPoint 实际预览为 `qa/pptx_slide_preview.png`。
- 可编辑性分开记录：内嵌 SVG 完整性 **通过**；转换为 PowerPoint 形状的结构准备度 **通过**（原生 SVG、嵌套语义组、无不支持特性）；未在 PowerPoint 中实际执行“转换为形状/取消组合”，因此不宣称转换后的视觉保真度已经实测。
- 完整输出目录已同步至 `overleaf_submission_20260813/img2pptx_pi_mscl_20260818/`。本轮未修改 `main.tex`、`references.bib` 或正文 Figure 引用，未触发 LaTeX 编译。

### 口径说明

- 本次任务是对指定参考图的忠实重建，因此保留源图中的 `40 × 14` 输入与 `L_rate` 三损失目标；这不改变当前正式项目采用 **16 维充电特征**、且归档主实验 `smoothness_weight=0` 的实现口径。该 PPTX 当前是独立可编辑设计稿，不能据此认定正文或实验口径已经更新。

## 2026-08-18：彩色斜向 Conv1D 特征图与紫色 Input 的 PPTX 版本

### 已完成

- 按作者追加的视觉要求，在上一版单页可编辑 PPTX 基础上另存新版本：`outputs/img2pptx_pi_mscl_colored_stacks_20260818/final.pptx`；上一版 `outputs/img2pptx_pi_mscl_20260818/` 未覆盖。
- Panel (a) 的三条 Conv1D 支路已分别改成蓝、绿、紫色系，每个 `40 → 20 → 10` 阶段均使用五层斜向错位的可编辑矢量特征图，前层由 6 × 3 个同色系深浅方格组成。完整 SVG 中共有 248 个可编辑 `<polygon>` 元素，不包含 raster `<image>`。
- 最左侧 Input 矩阵保持蓝绿主体，并在中下部加入紫色单元与蓝—紫—绿过渡；这一新增颜色要求已写入 `component_manifest.json` 的源观察、硬约束与可执行审计。
- 已重新生成 `full.svg`、12 个独立 `modules/*.svg`、单页 `final.pptx` 和全部 QA 文件；PowerPoint 实际导出预览为 `qa/pptx_slide_preview.png`。布局、独立模块、包含关系、对齐、边框层级、语义、约束覆盖、PPTX 包与预览审计均通过。
- 独立视觉复核确认：三条支路颜色区分明确，斜向透视和内部深浅方格在三个尺度上保持一致，Input 紫色加入自然；未发现裁切、遮挡、断线、文字碰撞或明显对齐问题。仅有最浅的蓝绿后层在白底上对比较低，但轮廓可辨，属于非阻断项。
- 由于上一版不满足本轮新增的硬性视觉要求，按语义修正优先原则接受本版；整图 raster MAE 从上一版 11.6803 变为 12.0205，但所有新增语义约束和硬性 QA 均通过。该取舍已记录在 `qa/visual_similarity_audit.json`。
- 完整输出目录已同步至 `overleaf_submission_20260813/img2pptx_pi_mscl_colored_stacks_20260818/`。本轮未修改 `main.tex`、`references.bib` 或正文 Figure 引用，未触发 LaTeX 编译。

### 口径说明

- 本版仍按作者指定的参考图保留 `40 × 14` 和三项损失中的 `L_rate`；这只是独立设计稿的视觉更新，不改变当前正式研究的 16 维输入与归档主实验 `smoothness_weight=0` 口径。

## 2026-08-18：Conv1D 行优先渐变与立体融合张量 PPTX 版本

### 已完成

- 按作者最新视觉要求，在彩色斜向特征图版本基础上再次另存：`outputs/img2pptx_pi_mscl_gradient_tensor_20260818/final.pptx`；此前两个 PPTX 输出目录均未覆盖。
- 三条 Conv1D 支路的 `40 / 20 / 10` 共 9 个前层矩阵均改为严格的 6 × 3 行优先渐变：左上角最深，沿每行向右连续变浅，下一行在上一行末端基础上继续递减，右下角最浅。蓝、绿、紫各自保持独立色相；每个单元仍是可编辑 SVG 多边形，并带 `data-gradient-rank/row/col` 供审计。
- Fuse 与 LSTM 之间的融合张量改为五层透视错位堆叠，增加显式顶面、右侧面、深浅不同的层板、前层 10 × 2 网格和强化前缘描边；两侧箭头已相应缩短并与张量保持清晰间距。
- `component_manifest.json` 新增“行优先单调渐变”和“五层张量+顶/侧面”的硬性语义约束；`audit.py` 会检查 9 组各 18 个渐变 rank 的完整性及亮度严格递增，并检查张量的 5 个 layer、2 个 facet 和 20 个 front cells。
- 放大 QA 时发现独立 `multiscale_cnn.svg` 的原 bbox 未覆盖左侧蓝色图例；已将组件范围从 `x=215,width=555` 修正为 `x=90,width=680`，随后重新生成所有模块、PPTX、PowerPoint 实际预览及全部 QA。修正后独立模块包含完整三项图例。
- 独立视觉复核确认：9 个 Conv1D 矩阵均能读出左上深到右下浅的顺序；融合张量正面、顶面、右侧面和层板纵深明确；箭头无覆盖，未发现裁切、文字碰撞、断线或间距失衡。极端缩小或低质量打印时，同一行末端的浅色相邻格差异可能较细微，属于非阻断项。
- 所有布局、独立模块、包含关系、对齐、边框、语义、约束覆盖、PPTX 包与 PowerPoint 预览硬审计通过。由于本轮是新增语义要求，整图 raster MAE 从上一版 12.0205 变为 12.1129；该语义优先取舍和完整修正循环已记录在 `qa/visual_similarity_audit.json`。
- 完整输出目录已同步至 `overleaf_submission_20260813/img2pptx_pi_mscl_gradient_tensor_20260818/`。本轮未修改 `main.tex`、`references.bib` 或正文 Figure 引用，未触发 LaTeX 编译。

### 口径说明

- 本版继续忠实保留参考图的 `40 × 14` 和 `L_rate` 表达，只更新可编辑设计稿的视觉编码；正式研究仍以 16 维输入及已归档实验配置为准。

## 2026-08-18：Conv1D 时间单元比例化 PPTX 版本

### 已完成

- 作者确认 `40 → 20 → 10` 不应仅通过外框和数字表达，而应同步减少前层网格的时间单元数；已在上一版基础上另存：`outputs/img2pptx_pi_mscl_proportional_cells_20260818/final.pptx`，未覆盖任何已有版本。
- 每条蓝、绿、紫支路的三个 Conv1D 前层矩阵分别改为 `8 × 3`、`4 × 3`、`2 × 3`，对应 24、12、6 个可编辑单元；横向时间格数严格保持 `4:2:1`，抽象表示 `40:20:10`。纵向三行和五层叠板在三个阶段保持不变，用于表明 `64 ch.` 通道深度没有随池化下降。
- 三阶段外框宽度同步调整为约 `100 / 76 / 52 px`，兼顾尺寸递减和末级可读性；`40 / 20 / 10` 标签已重新居中，两段虚线箭头按新边界重新连接。
- 每个阶段仍保持左上最深、右下最浅的行优先单调渐变。`component_manifest.json` 新增时间格数比例硬约束，`audit.py` 会逐一验证 9 个矩阵的列数、三行、五层叠板、完整 gradient rank 和亮度单调性。
- PowerPoint 实际预览及放大模块已由主代理和独立视觉 QA 复核：三条支路均为 `8×3 → 4×3 → 2×3`，格数、外框和标签关系清楚，Fuse 汇聚保持正确；未发现裁切、遮挡、文字碰撞、断线或布局失衡。常规整页缩放时同一行内部渐变比放大图细微，但不影响比例关系和多级色深辨识。
- 布局、独立模块、包含关系、对齐、边框、语义、约束覆盖、PPTX 包与实际预览硬审计全部通过。由于本轮是语义修正，整图 raster MAE 从 12.1129 变为 12.1807；该取舍已记录在 `qa/visual_similarity_audit.json`。
- 完整输出目录已同步至 `overleaf_submission_20260813/img2pptx_pi_mscl_proportional_cells_20260818/`。本轮未修改 `main.tex`、`references.bib` 或正文 Figure 引用，未触发 LaTeX 编译。

### 口径说明

- `8/4/2` 是为论文图可读性采用的比例化抽象，不代表仅存在 8、4、2 个真实时间步；精确张量长度仍由下方 `40/20/10` 标注给出。图中继续保留参考稿的 `40 × 14` 与 `L_rate`，不改变正式研究的 16 维输入及归档实验口径。

## 2026-08-18：架构图原生 PowerPoint 形状版本

### 已完成

- 继续使用 `img2pptx` 重建得到的 `full.svg` 作为视觉与语义基准，在不覆盖此前 SVG 嵌入版的前提下，新建 `outputs/img2pptx_pi_mscl_native_shapes_20260818/`。
- 生成推荐的语义分组版 `PI-MSCL_architecture_native_shapes_20260818.pptx`：PowerPoint 顶层 29 个对象，递归展开为 580 个唯一命名对象，其中包括 538 个原生形状、17 条原生连接线、87 个可编辑文本框、278 个自定义矢量几何和 25 个语义分组。
- 同时生成直接单击编辑版 `PI-MSCL_architecture_native_shapes_fully_ungrouped_20260818.pptx`：555 个顶层原生对象、0 个分组。两个版本均为 0 个图片对象、0 个内嵌媒体文件，不再把完整 SVG 当作单一图片放入幻灯片。
- 第一轮 PowerPoint 实际预览发现新建形状继承了主题阴影；已清除所有非源图主题效果并完成第二轮修正验证。最终分组版和完全取消分组版的 PowerPoint 导出逐像素一致，相对上一版 `full.svg` 渲染的 RGB MAE 为 3.9507。
- `qa/native_pptx_final_audit.json` 的全部硬检查通过；`qa/native_editability_inventory.json` 提供完整对象名称、父组、类型与文本清单。独立视觉 QA 未发现元素遗漏、裁切、文字碰撞、断线或拓扑错误。
- 完整输出目录已同步至 `overleaf_submission_20260813/img2pptx_pi_mscl_native_shapes_20260818/`。本轮未修改 `main.tex`、`references.bib` 或正文 Figure 引用，因此未触发 LaTeX 编译。

### 编辑方式与口径说明

- 分组版适合按 Input、CNN 支路、Fuse、LSTM、回归头及损失路径整体移动；双击组或执行一次“取消组合”即可编辑内部单元。完全取消分组版可直接单击任意方格、曲线点、箭头或文字。
- 本版仍保留参考设计稿中的 `40 × 14` 与 `L_rate` 表达，只改变 PowerPoint 编辑结构；正式研究仍以 16 维输入和归档实验 `smoothness_weight=0` 为准，该图尚未插入主稿。

## 2026-08-18：原生形状版后层边框淡化修订

### 已完成

- 作者指出完全取消分组版的立体叠板中，后方衬底层边框仍比最初 SVG 嵌入版偏重。核查确认原生转换此前只把 SVG 元素的 `opacity` 应用于填充，没有同步应用到描边。
- 未覆盖上一版，新建 `outputs/img2pptx_pi_mscl_native_shapes_light_edges_20260818/`，并生成完全取消分组版 `PI-MSCL_architecture_native_shapes_fully_ungrouped_light_backplanes_20260818.pptx` 与视觉一致的语义分组版 `PI-MSCL_architecture_native_shapes_light_backplanes_20260818.pptx`。
- Conv1D 的 9 组 `40/20/10` 叠板及 Fuse 后融合张量现在均将同一层 `opacity` 同时作用于填充和边框：最靠后的衬底层最淡，向前逐层增强；最前表面网格和最终外轮廓保持清晰。
- PowerPoint 实际导出确认两种编辑版本逐像素一致。相对上一版 `full.svg` 基准的 RGB MAE 从 3.9507 降至 3.7638；所有原生对象、可编辑文本、无图片对象、无嵌入媒体等硬检查继续通过。
- 独立视觉 QA 确认后层边框渐淡自然，未出现错误消边、断层、层次粘连、立体感塌陷或前后关系反转。
- 完整输出目录已同步至 `overleaf_submission_20260813/img2pptx_pi_mscl_native_shapes_light_edges_20260818/`。本轮未修改 `main.tex`、`references.bib` 或正文 Figure 引用，因此未触发 LaTeX 编译。

### 口径说明

- 本轮只修正立体图形的视觉透明度与描边层次，仍保留参考设计稿中的 `40 × 14` 和 `L_rate`；不改变正式研究的 16 维输入与归档实验 `smoothness_weight=0` 口径。

## 2026-08-18：原生形状版柔化边框与紧凑色块修订

### 已完成

- 作者要求在后层边框淡化版基础上，将立体叠板的所有层边框再淡一档，并略微缩小彩色小方格之间的白色留白。为保留版本历史，新建 `outputs/img2pptx_pi_mscl_native_shapes_soft_edges_tight_cells_20260818/`，未覆盖此前 PPTX。
- Conv1D 蓝、绿、紫三色叠板的边框颜色分别向白色混合 32%；Fuse 后融合张量的顶面、侧面、后层及前轮廓边框向白色混合 28%。各层原有透明度关系继续保留，因此后层最淡、前层最清楚。
- Input、9 个 Conv1D 前表面及融合张量彩色小格的白色分隔线宽度缩小至上一版的 72%，以减小色块间留白，同时保留完整可辨的网格。
- 生成完全取消分组版 `PI-MSCL_architecture_native_shapes_fully_ungrouped_soft_edges_tight_cells_20260818.pptx` 和视觉一致的语义分组版 `PI-MSCL_architecture_native_shapes_soft_edges_tight_cells_20260818.pptx`。前者仍为 555 个顶层原生对象；两者均为 0 个图片对象和 0 个内嵌媒体文件。
- PowerPoint 实际导出、原生对象审计和文本可编辑探针全部通过；两个版本的渲染逐像素一致。相对 SVG 基准的 RGB MAE 为 3.7144，较上一版 3.7638 进一步下降。
- 独立视觉 QA 确认：所有叠板后层仍可辨，前表面清楚；白色缝隙缩小但未粘连；面板、箭头、文字、公式、曲线及模块位置均未误改。
- 完整输出目录已同步至 `overleaf_submission_20260813/img2pptx_pi_mscl_native_shapes_soft_edges_tight_cells_20260818/`。本轮未修改 `main.tex`、`references.bib` 或正文 Figure 引用，因此未触发 LaTeX 编译。

### 口径说明

- 本轮仅调整立体特征图与张量的边框及单元间距；图中仍保留参考设计稿的 `40 × 14` 与 `L_rate`，不改变正式研究的 16 维输入和归档实验 `smoothness_weight=0` 口径。

## 2026-08-18：Input、Conv1D 与融合张量紧凑网格修订

### 已完成

- 作者进一步澄清，希望缩小的是 Input、Conv1D 和 Fuse 后融合张量内部彩色单元之间的真实间距，而不仅是略微减小白线宽度。为保留版本历史，新建 `outputs/img2pptx_pi_mscl_native_shapes_compact_cells_20260818/`，未覆盖此前 PPTX。
- Input 原来采用 19 px 节距和 17 px 色块，存在约 2 px 的几何空隙；本版将每个彩色色块围绕中心对称放大为 18.2 px，使实际空隙缩小至约 0.8 px，最外层色块仍未触碰 Input 外框。
- Conv1D 的 9 个前表面网格与 Fuse 后融合张量前表面的白色单元描边进一步缩小至原始宽度的 42%，使色块更连续紧凑，同时保持逐格可辨。
- 保留上一版柔化叠板边框的处理；面板边框、箭头、文字、公式、曲线和布局未调整。
- 生成完全取消分组版 `PI-MSCL_architecture_native_shapes_fully_ungrouped_compact_cells_20260818.pptx` 与视觉一致的语义分组版 `PI-MSCL_architecture_native_shapes_compact_cells_20260818.pptx`。前者仍有 555 个顶层原生对象；两者均无图片对象和内嵌媒体。
- PowerPoint 实际导出、原生对象审计及文本可编辑探针全部通过；两个版本渲染逐像素一致，相对 SVG 基准的 RGB MAE 为 3.7347。
- 独立视觉 QA 确认：Input 网格更紧凑且不碰外框；Conv1D 与融合张量的白缝更细但无粘连或网格消失；其他模块未误改。
- 完整输出目录已同步至 `overleaf_submission_20260813/img2pptx_pi_mscl_native_shapes_compact_cells_20260818/`。本轮未修改 `main.tex`、`references.bib` 或正文 Figure 引用，因此未触发 LaTeX 编译。

### 口径说明

- 本轮仅改变三处彩色网格的单元间距，仍保留参考设计稿中的 `40 × 14` 与 `L_rate`；不改变正式研究的 16 维输入和归档实验 `smoothness_weight=0` 口径。

## 2026-08-18：Conv1D 前表面色块进一步紧凑修订

### 已完成

- 作者纠正说明：需要进一步收紧的是 Conv1D 九组最前表面中已有小色块之间的白色留白，不是在后方衬底层新增网格。此前误生成的“每层补网格”临时草稿未同步、未写入交接，并已从本地输出目录清理。
- 以 `img2pptx_pi_mscl_native_shapes_compact_cells_20260818` 为基线，新建 `outputs/img2pptx_pi_mscl_native_shapes_conv_cells_tighter_20260818/`，未覆盖已交付版本。
- 仅调整带 `data-gradient-rank` 的 126 个 Conv1D 前表面色块：白色描边宽度缩小至原始宽度的 22%；考虑 PowerPoint 会把极细线仍显示为接近一个屏幕像素，分隔线透明度同时限制为 48%，从而进一步弱化可见留白。
- 后方四层衬底继续保持纯色叠板，不新增网格。Input 与 Fuse 后融合张量保持上一版紧凑单元设置，面板、箭头、文字、公式、曲线和布局均未修改。
- 生成完全取消分组版 `PI-MSCL_architecture_native_shapes_fully_ungrouped_conv_cells_tighter_20260818.pptx` 与视觉一致的语义分组版 `PI-MSCL_architecture_native_shapes_conv_cells_tighter_20260818.pptx`。前者仍有 555 个顶层原生对象；两者均无图片对象和内嵌媒体。
- PowerPoint 实际导出、对象结构审计及文本可编辑探针全部通过；两个版本逐像素一致，相对 SVG 基准的 RGB MAE 为 3.7195。
- 独立视觉 QA 确认：仅 Conv1D 前表面白缝变细、色块更紧凑且逐格可辨；后层没有新增网格；Input、Fuse张量及其他模块未误改。
- 完整输出目录已同步至 `overleaf_submission_20260813/img2pptx_pi_mscl_native_shapes_conv_cells_tighter_20260818/`。本轮未修改 `main.tex`、`references.bib` 或正文 Figure 引用，因此未触发 LaTeX 编译。

### 口径说明

- 本轮仅调整 Conv1D 前表面色块的视觉分隔，仍保留参考设计稿中的 `40 × 14` 与 `L_rate`；不改变正式研究的 16 维输入和归档实验 `smoothness_weight=0` 口径。
