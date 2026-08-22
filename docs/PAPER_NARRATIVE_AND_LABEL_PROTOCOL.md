# 论文叙事与稀疏标签协议备忘

> 用途：作为第四章和英文稿 `latex/els-cas-templates/cas-dc-template.tex` 的共同写作依据。
> 
> 本文档固定“问题—协议—方法—证据—边界”的逻辑，避免将 HUST 上的受控标签掩码误写成真实现场天然发生的连续标签缺失。

## 1. 一句话论证

SOH 标定代价高，训练阶段可用循环级标签有限；无标签循环仍保留完整运行特征与退化顺序。本文以多尺度 CNN-LSTM 提取退化表征，并以软单调物理一致性约束提供不依赖真实 SOH 的结构性弱监督；在未见 HUST 电池上检验标签预算下降下的跨电池泛化与轨迹稳定性。

英文版：

> Costly capacity calibration limits the availability of cycle-level SOH labels during training, whereas unlabeled cycles still retain complete operating features and degradation order. We combine multi-scale CNN-LSTM representation learning with a soft-monotonic physics-consistency constraint that provides structural weak supervision without requiring true SOH labels, and evaluate cross-cell generalization and trajectory stability on unseen HUST cells under reduced label budgets.

## 2. 全文主线

```text
工程动机：SOH 标定代价高，训练阶段可用循环级标签有限
        ↓
数据结构：无标签循环仍保留完整运行特征与退化顺序
        ↓
受控验证：HUST 有完整真值，因此仅在训练电池上掩蔽部分 SOH 标签
        ↓
方法：多尺度 CNN-LSTM 提取退化表征；软单调约束提供不依赖真实 SOH 的结构性弱监督
        ↓
证据：在未见 HUST 电池上检验标签预算下降下的跨电池泛化与轨迹稳定性
        ↓
结论：多尺度模型与物理约束降低跨电池 SOH 建模对稠密容量标定的依赖；并非替代容量标定
```

## 3. 当前实现：必须如实表述

### 3.1 数据与划分

- 数据来源：HUST 公开锂离子电池全生命周期退化数据。
- 数据事实：原始样本具有完整 SOH/容量真值；真值用于构造训练掩码和最终测试评价。
- 划分：先按电池个体划分训练、验证和测试集；测试电池在训练中完全不可见。
- 标签掩码仅施加于训练集；验证与测试标签保持完整，用于模型选择与公平评价。

### 3.2 标签协议

当前 `train_cross_battery.py::generate_supervision_mask` 的实现是：

1. 对每个训练电池分别处理；
2. 在该电池全生命周期的循环样本中，随机保留约 `r` 比例的 SOH 标签；
3. 未保留标签的循环仍保留其输入特征和时序位置；
4. 无标签循环不进入监督 MSE，但可进入物理一致性相关计算；
5. 使用固定随机种子保证可复现。

**规范名称**：`controlled battery-stratified random cycle-level label masking`。

**中文名称**：受控的电池内分层随机循环级标签掩码；或受控稀疏 SOH 监督协议。

### 3.3 该协议能与不能证明什么

| 可以支持的结论 | 不能直接支持的结论 |
|---|---|
| 标签预算下降时的跨电池 SOH 估计鲁棒性 | 真实 BMS 中天然发生的标签缺失已被直接复现 |
| 物理约束对随机无标签循环的结构性引导作用 | 仅有早期标签时的中后期时间外推能力 |
| 多尺度表征与物理约束在不同标签比例下的相对作用 | 周期性容量标定、维护窗口或现场工况的直接部署效果 |
| 未见电池上的泛化与轨迹稳定性 | 少数完成全寿命标定电池向大量无标定电池的迁移能力 |

## 4. 术语表（英文稿必须统一）

| 规范术语 | 首次出现时的定义/用法 | 避免的写法 |
|---|---|---|
| SOH-label budget | 可用于训练监督的 SOH 标签数量或比例 | 将其等同于现场自然缺标 |
| controlled sparse-label supervision | HUST 完整真值上构造的受控稀疏监督 | real incomplete-label dataset |
| battery-stratified random cycle-level label masking | 每块训练电池内随机保留比例为 `r` 的循环标签 | continuous missing interval（除非另做连续区间实验） |
| cross-cell generalization | 测试电池在训练期间完全不可见 | 同电池不同循环插值 |
| PI-MSCL | Physics-Informed Multi-Scale CNN-LSTM（英文稿模型名） | PI-CNNLSTM、PI-MS-CNNLSTM、PI-MSCL 混用而不说明 |
| soft-monotonic physics-consistency constraint | 允许小幅波动、惩罚明显非物理 SOH 上升的约束 | hard monotonicity（除非确为硬约束） |
| structural weak supervision | 无真实 SOH 标签时由轨迹结构/物理先验提供的训练引导 | pseudo-label（本模型主线不生成伪 SOH 标签） |

`partial lifecycle supervision` 可以保留为宽泛研究动机，但英文稿第一次出现时必须补充限定：其在本文中以**电池内分层随机循环级标签掩码**实现，而不是自然发生的连续时间缺标。

## 5. 各章节应承担的论证任务

### Introduction

1. SOH 对安全和寿命管理的重要性；容量标定昂贵。
2. 现有深度模型通常依赖稠密标签；无标签运行特征却可连续获得。
3. 问题不是宣称已经复现所有现场缺标机制，而是在可控 HUST 基准上检验标签预算下降。
4. 提出模型：多尺度表征 + 标签掩码 + 软单调物理一致性。
5. 给出贡献及证据边界。

### Methodology

1. 从 PI-MSCL 整体框架开始，定义输入、输出和损失符号。
2. 说明多尺度 CNN 的作用是覆盖局部、中期和长期退化模式。
3. 说明 LSTM 建模跨循环健康演化。
4. 定义掩码监督损失和软单调约束，并说明后者为何可在无标签循环上提供结构性弱监督。
5. 明确主模型使用的损失项及未启用模块，避免“模块堆叠”印象。

### Experimental Setup

1. 说明 HUST 完整标签被用于受控模拟和最终评价。
2. 定义电池级划分与受控稀疏标签协议：先电池级划分，再仅在训练电池上按 `r=1.0/0.7/0.5/0.3` 掩码。
3. 明确验证/测试标签不参与训练或掩码规则选择。
4. 所有模型使用相同的数据划分、特征、标签预算和调参规则。

### Results and Discussion

结果章节采用“主结果在前、机制解释在后”的证据顺序，避免在读者尚未看到标签预算收益前就进入消融。

| 节 | 建议标题 | 要回答的问题 | 必须呈现的证据 |
|---|---|---|---|
| 4.1 | 完整监督下的跨电池基准结果 | PI-MSCL 是否是合格的跨电池 SOH 估计器？ | 与代表性基线的 MAE、RMSE、R²；多次训练的均值±标准差 |
| 4.2 | 标签预算下降下的跨电池估计性能 | 标签减少时，PI-MSCL 是否退化得更慢？ | `r=1.0/0.7/0.5/0.3` 下各模型的均值±标准差，以及性能—标签比例曲线；这是全文主结果 |
| 4.3 | 稀疏标签下的轨迹级稳定性分析 | 物理约束是否改善低标签下的真实退化轨迹，而非仅压低点误差？ | 未见电池案例轨迹 + 全测试集的单调性违背率、累计异常上升幅度或波动度；必须与 MAE/RMSE 并列 |
| 4.4 | 架构与物理约束消融实验 | 多尺度表征和软单调约束分别贡献什么，且约束是否在低标签下更重要？ | CNN-LSTM、仅多尺度、仅物理约束、PI-MSCL；至少比较 `r=1.0/0.5/0.3` |

**4.2 的统计口径：**每个 `r` 固定电池级训练/验证/测试划分，改变训练初始化与训练电池内标签掩码种子，报告多次重复后的 `mean ± std`。`r=1.0` 没有掩码差异，但仍应使用多个训练种子以保持统计口径一致。因此，无需再设独立的“掩码稳定性”章节。

**4.3 的叙事：**标签预算下降时，MAE/RMSE 不能揭示局部 SOH 回升、漂移和振荡；而软单调约束的理论作用正是为未标注循环提供轨迹结构引导。因此，只有同时证明“误差不增大（最好降低）”与“非物理轨迹行为减少”，物理一致性主张才闭环。案例电池须按预先声明的规则选择（如测试误差中位数的电池），而不是事后挑选最佳曲线。

异常风险提示、特征缺失鲁棒性、连续前缀或周期性标定协议均不是当前主线的必要正文结果；若实验成熟，可放入补充材料或后续工作，不能替代上述四段主证据。

### Conclusion

应使用：

> The proposed method reduces the dependence of cross-cell SOH modeling on dense cycle-level calibration labels; it does not replace capacity calibration.

不应使用：

> The method has been validated for direct deployment under naturally occurring BMS label gaps.

## 6. 可参照的五类有限标签叙事

| 工作 | 标签稀缺的定义 | 方法如何利用无标签数据 | 可借鉴点 | 与本文的关键差异 |
|---|---|---|---|---|
| Li et al., IEEE TII (2026) | 少数电池具有完整退化轨迹，标签覆盖按电池顺序选取而非全体随机撒点 | 退化辅助标签 + 选择性伪标签 + 共享表征 | “标签稀缺—无标签退化信息—跨电池验证”的完整论证框架 | 本文在每块训练电池内随机稀疏标签，不是少数完整标定电池 |
| Lin et al., Energy Storage Materials (2023) | 少量实验室标注数据与大量无标签充电数据 | 异构回归器协同训练并生成伪标签 | 将无标签运行数据视为可用退化信息 | 本文主要依赖物理约束，不以伪标签作为主线 |
| Salucci et al., Journal of Power Sources (2023) | 真实运行中参考 SOH 测试稀少 | 围绕参考测试建立局部标签学习与累计监测 | 真实现场/周期性标定的叙事边界 | 本文不是现场周期标定协议 |
| Han et al., Journal of Energy Storage (2024) | 仅有早期阶段标注 | 早期数据驱动扩增，服务中后期预测 | 连续早期标签与时间外推的论证方式 | 本文随机标签并不检验时间外推 |
| Qian et al., Energy Transition (2026) | 极少真实标签 | 从无标签 dQ/dV 曲线构造物理相关弱标签后预训练、微调 | 用非 SOH 的退化先验支持少标签学习 | 本文使用单调轨迹先验，不构造额外弱标签 |

### 从同类工作提炼的实验闭环

同类有限标签工作通常不以单一低标签结果作为结论，而是形成以下闭环：

```text
标签比例扫描（问题在标签下降时是否更突出）
        ↓
与监督/半监督基线比较（方法是否真正更节省标签）
        ↓
低标签案例或轨迹分析（误差改善是否对应合理退化行为）
        ↓
模块消融（无标签信息或物理先验由哪一部分发挥作用）
        ↓
必要时的跨条件/跨数据验证（结论是否局限于单一条件）
```

Li et al. 采用 100% 至 5% 的标签比例扫描，并在不同标签比例下进行模块消融、监督与半监督基线比较、参数敏感性和额外数据集验证；其核心结论是辅助退化信号与伪标签机制在极低标签条件下更有价值。Lin et al. 用“一个有标注电池 + 多个无标注电池”的设计，并递增无标注样本量，证明无标注充电数据可带来增益。Qian et al. 则用极少真实标签、跨条件和实车数据说明物理相关弱信号的泛化潜力。

本文应借鉴这一**证据结构**，但不复制其标签分配：本研究的可控实验单位是每块训练电池内的随机循环级标签预算。因此，正文最强、最聚焦的闭环是 4.1–4.4；没有额外跨数据集或不同标签机制结果时，不要为了“看起来完整”另设 4.5。

### 推荐参考框架

最适合本文的是 Li et al. 的论证骨架，但只借鉴其“标签稀缺—无标签退化信息—结构性训练信号—跨电池验证”框架，不照搬其“少数完整标定电池”的标签分配方案。

可直接用于英文稿的方法定位句：

> Inspired by the need to reduce dependence on costly SOH calibration, we construct a controlled sparse-label protocol on HUST. Unlike studies that assume full trajectories for only a few calibrated cells, our protocol distributes a fixed label budget within each training cell, allowing the effect of cycle-level SOH label availability to be isolated.

本文自己的核心句是：

> 本文的关键不是“随机少了多少标签”，而是：当真实 SOH 标签不足时，如何利用完整运行轨迹中的退化结构与物理先验，降低跨电池 SOH 建模对稠密容量标定的依赖。

一句话定稿：

> 以有限标定为工程动机，以 HUST 随机掩码为可控验证，以物理约束利用无标签退化结构为方法创新，以跨电池标签预算鲁棒性为实验证据。

## 7. 后续可扩展但不应与当前结果混写的协议

| 协议 | 标签分布 | 回答的问题 |
|---|---|---|
| 当前协议：随机稀疏 | 每块训练电池内随机保留 `r` 比例循环 | 一般标签预算鲁棒性 |
| 少数完整标注电池 | 仅少数训练电池完整有标签 | 跨电池标定覆盖不足/迁移 |
| 连续早期前缀 | 每块电池仅前 `r%` 生命周期有标签 | 中后期时间外推 |
| 周期性标定 | 固定循环间隔或维护窗口有标签 | BMS 定期容量校准 |

只有实际运行某一协议后，才可将对应能力写入正文结论。

## 8. 写作前检查清单

- [ ] 是否在 Introduction 中区分了工程动机与 HUST 上的受控模拟？
- [ ] 是否明确了掩码只作用于训练电池的监督标签？
- [ ] 是否将当前协议称为随机循环级标签掩码，而非连续时间缺标？
- [ ] 是否先报告全监督基准，再解释标签预算下降下的变化？
- [ ] 是否把多尺度与物理约束的贡献分开解释？
- [ ] 是否避免“所有指标均最优”“可直接现场部署”等超出证据边界的表述？
- [ ] 是否把异常风险提示限制为 SOH 轨迹的工程扩展？
- [ ] 是否保证 PI-MSCL、HUST、SOH、标签比例和指标名称全篇一致？

## 9. 参考文献（叙事定位用）

1. Li, Y. et al. *Auxiliary-Label Enhanced Semi-Supervised Learning with Selective Pseudo-Labeling for Battery Capacity Estimation*. IEEE Transactions on Industrial Informatics (2026). https://doi.org/10.1109/TII.2025.3645950
2. Lin, C., Xu, J., & Mei, X. *Improving state-of-health estimation for lithium-ion batteries via unlabeled charging data*. Energy Storage Materials 54, 85–97 (2023). https://doi.org/10.1016/j.ensm.2022.10.030
3. Salucci, C. B. et al. *A novel semi-supervised learning approach for State of Health monitoring of maritime lithium-ion batteries*. Journal of Power Sources 556, 232429 (2023). https://doi.org/10.1016/j.jpowsour.2022.232429
4. Han, D., Zhang, Y., & Ruan, H. *Improving the state-of-health estimation of lithium-ion batteries based on limited labeled data*. Journal of Energy Storage 100, 113744 (2024). https://doi.org/10.1016/j.est.2024.113744
5. Qian, C. et al. *Explainable state of health estimation of lithium-ion batteries with extremely minimal labels via weakly supervised learning*. Energy Transition (2026). https://doi.org/10.1016/j.etran.2026.100623

## 10. 论文系统重构执行约定（2026-08-06 锁定）

### 10.1 内容与证据来源

1. 论文叙事主体和章节事实以 `latex/els-cas-templates/soh_cn_version.docx` 为第一内容来源；英文稿不是自由改写，而是在不改变中文原意、实验事实和结论边界的前提下按 SCI 逻辑重构。
2. 已完成且可追溯的数据和结果可以直接进入正文，但必须核对实验配置、数据划分、标签掩码、随机种子、统计口径和模型名称，不能仅凭旧图或摘要数字转录。
3. 缺少的主线实验和脚本应补建并实际运行。首次结果不理想时，允许在不改变研究问题和评价口径的前提下诊断并调整策略；若多轮后改善仍有限，最终结果无论是否达到预期均如实写入正文，并解释失败边界。
4. 任何作者意图、目标期刊格式、实验口径或结论强度无法从本地证据确定时，写入“待作者统一确认清单”，不得自行补造或替作者决定。

### 10.2 图表与架构图规范

1. 所有结果图重新由原始数据或结果文件生成，位图输出至少 600 dpi；优先同时保留 PDF/SVG 等矢量版本。
2. 组图必须由绘图脚本直接排版生成；每个子图也必须由同一数据源独立生成，禁止从组图截图或裁剪得到子图。
3. 图形的字体、线宽、配色、尺寸、图例、刻度和导出方法以 `E:/毕设代码/CALCE/algorithms/tools/convergence_analysis.py` 及同项目其他科研绘图脚本为风格参照，并形成当前项目统一绘图配置。
4. 每张图先定义结论和证据作用，再选择图型；不得为了增加篇幅制作重复图。
5. 图注、表注仅简要说明对象、条件和必要缩写；比较结果、趋势解释与结论在正文展开，保证图文一一对应。
6. 方法总览架构图使用可编辑 `.drawio` 文件绘制。先生成预览并自检，再生成最终 Draw.io 图；重点检查节点/线条重叠、边线穿越、标签裁切、模糊、信息重复和阅读路径。图形不机械复刻中文稿的传统框图，而以“稀疏标定证据—多尺度网络—双目标学习”三联证据链组织，并用退化曲线、标定点、张量/网络单元和局部单调违反示意直接解释方法动机。

### 10.3 写作、文献与版式规范

1. 写作采用算法型 SCI 论文证据链：任务与边界 → 方法与设计理由 → 公平主结果 → 轨迹机制证据 → 消融 → 失败模式与适用边界。
2. 每轮正文写作后必须进行一次审稿人式自检，检查原创性定位、技术可靠性、公平比较、可复现性、图文一致性、结论边界和非专业读者可读性；发现的问题在下一轮实质修改，不能只形成评语。
3. 引言与相关工作扩展到至少 40 篇真实、可检索、与具体论断相匹配的文献。每条文献需核验题名、作者、期刊/会议、年份和 DOI/稳定页面，维护统一 BibTeX，禁止仅凭题名相关就作为证据引用。
4. 最终稿在作者指定的 Springer 单栏模板下达到 20 页以上；页数通过完整方法、实验、结果和讨论获得，不通过过长图注、重复背景、异常字号或空白排版凑页。
5. 术语、符号、数据划分、监督比例、模型名和指标必须遵循本备忘的术语表及后续锁定的 Terminology Ledger。

### 10.4 当前备份与待确认项

- 重构前备份：`backups/manuscript_pre_restructure_20260806/`。备份包含当前中文 DOCX、英文 LaTeX 工程、根目录 `main.tex`、中文扩写稿、第四章 Markdown、实验结果摘要、改进计划和本备忘。
- 已解决 TBC-001：作者提供的模板位于 `D:/Downloads/Download+the+journal+article+template+package+(December+2024+version)/sn-article-template/`。当前权威英文主稿为 `latex/springer-sn-2024/manuscript.tex`，使用 Springer Nature 2024 年 12 月 v3.1 单栏 `sn-mathphys-num` 配置；原 Elsevier CAS 稿仅保留作内容对照与回溯。
- 2026-08-06 基线编译状态：43 条 BibTeX 中 42 条被正文实际引用，正式 PDF 共 18 页；引用和交叉引用均已解析，未发现越界行。20 页目标将在补入泄漏隔离后的正式多种子结果、独立 600 dpi 子图/组图及相应讨论后复核，不以空白或冗余文字补页。
- 已解决 TBC-002：主表固定 battery split seed 42，以训练种子 `[929, 2262, 7]` 和独立 mask seeds `[1929, 3262, 1007]` 做三次配对重复；多个独立 battery split 只作敏感性验证，单独报告且不与主表混合平均。
- 已解决 TBC-008：正式实现和正文统一报告当前 loader/CSV 的 16 维充电特征；旧中文稿中的 14 维描述被该实现口径取代。
- 其余待确认事项在后续审计中继续编号追加；不因等待确认而停止可独立完成的实验复现和素材准备。

### 10.5 正文结构冻结文件

逐节正文映射、统一术语、结果证据矩阵、计划图表和第一轮作者式自检已写入：

- `docs/PAPER_SECTION_MAPPING.md`

后续正文改写以该文件为结构基线。若实验复算改变具体结论，只更新证据强度和结果措辞，不随意改变已经确定的主问题、随机循环级标签协议以及“主结果先于消融”的章节顺序。

### 10.6 实验公平性更正

旧训练入口中，无物理模型采用合并电池后的跨边界窗口化，有物理模型采用逐电池窗口化，导致旧 A1 与 Exp09c 比较存在数据管线混杂。论文实验统一改用电池边界保持窗口化；旧汇总结果只作开发线索，主结果与二维消融须在统一管线上重跑。轨迹指标同时统一使用真实循环索引和预先声明的容忍阈值。

进一步审计发现，旧跨电池入口曾在电池划分前对每块电池分别拟合 `StandardScaler`，随后又进行训练集标准化。这会使用验证/测试电池自身完整生命周期的统计量，构成跨电池预处理泄漏。现已保留单电池调用的兼容路径，但论文跨电池管线改为先读取原始特征、完成电池级划分，再仅用训练电池拟合一次标准化器并应用于验证/测试电池。所有旧结果因此仍不得作为最终证据；修复后 1-epoch smoke 与 14 项单元测试均已通过。正式证据归档使用 `v3_leakage_free` revision，smoke 与正式运行物理隔离，分批运行汇总会扫描该 revision 下的全部结果。

完整重跑矩阵、两种重复口径、断点续跑命令、归档结构和失败结果处理纪律见：

- `docs/PAPER_EXPERIMENT_EXECUTION_PLAN.md`

### 10.9 本地结果资产审计与暂停训练

2026-08-06 已完成对中文版表格、`docs/` 全部结果快照、`experiments/`/`results/` 中现存机器结果和旧绘图脚本的同步审计。结论是：本地旧结果数量充足，但至少分属中文版旧论文、4--5 月开发汇总和 8 月无泄漏固定划分三种协议，不能直接混合。旧结果没有被删除；其可复用范围、禁止用途、当前 4/48 正式进度和剩余任务详见：

- `docs/LOCAL_RESULT_ASSET_AUDIT.md`

作者已明确要求本轮检查后不再运行训练。当前训练状态为暂停，剩余正式矩阵只有在后续明确指令下恢复。

中文版 Tables 1--5 已进一步结构化为 `data/paper_results/legacy_cn_tables.csv` 并生成 600 dpi 候选组图和独立子图。所有资产均保留 `legacy` 标识和统计边界，未写入活动正文，详见 `docs/LOCAL_RESULT_ASSET_AUDIT.md` 第 8 节。

### 10.10 当前完成度与图件预览口径（2026-08-07）

正文、实验、图表和投稿元数据的最新完成度，以及 Fig. 1--5 的统一面板结构和交付格式，见：

- `docs/CURRENT_COMPLETION_AND_FIGURE_PREVIEW.md`

新增待确认项：

- TBC-012（已解决）：v5 用两行表示两层 LSTM、三列表示 `t-1/t/t+1` 时间展开，并明确 `hidden size 64`；输入已统一为 `40 cycles × 16 features`，不会被误读为三层网络。
- TBC-013：Fig. 1 panel (a) 已包含稀疏标签协议示意；是否仍在 Experimental Setup 另设独立 Fig. 2。若另设，必须从 Draw.io 源独立重排，不得裁剪 Fig. 1。

### 10.7 方法架构图 v3（正文候选）

- 可编辑源文件：`docs/PI-MSCL_architecture_v3.drawio`
- 同坐标预览：`docs/PI-MSCL_architecture_v3_preview.png/.pdf/.svg`
- 生成脚本：`scripts/generate_architecture_drawio_v3.py`
- 设计逻辑：面板 (a) 用完整特征轨迹和稀疏 SOH 标定点定义任务；面板 (b) 展开 `k=3/7/15` 三尺度卷积分支、融合、跨循环 LSTM 和回归头；面板 (c) 用带局部回升的预测曲线解释 masked MSE 与软单调损失如何互补。
- 自检结果：54 个可编辑顶点、35 条可编辑连接线，全部元素含有效 `mxGeometry`；PNG 为 600 dpi，5860 × 3374 px；经多轮人工视觉检查后，已清除卷积分支文字碰撞、比例符号缺字、损失公式溢出和主要连线穿越。
- v2 保留为简化回退稿：`docs/PI-MSCL_architecture_v2.drawio`，不作为当前正文首选。
- TBC-009：本机无 Draw.io desktop/CLI，最终排版前需在原生 Draw.io 打开 v5 一次，确认字体替换与箭头渲染。

### 10.8 参考文献核验

- 逐条 DOI、元数据、叙事用途和协议边界见 `docs/REFERENCE_AUDIT.md`。
- 当前 BibTeX 共 43 条且均含 DOI；引言使用 40 个不同引用键。
- HUST 原始论文作者与数据集贡献者已经依据 RSC 与 Mendeley Data 正式页面纠正。
- 引用数量达标不等于证据充分；有限标签文献只用于建立问题背景和对照协议，不能据此宣称本文随机循环级掩码已模拟真实现场缺标。
### 10.11 正式 48-run 结果后的主线证据表（2026-08-07）

正文的一句话主张现锁定为：

> 在 HUST 的受控循环级标签预算下，完整运行轨迹可通过软物理一致性提供不依赖真实 SOH 标签的结构监督；这种监督稳定改善低预算轨迹规则性，但点误差收益取决于标签预算与架构交互，不能表述为全面或单调增强的优势。

| 主张 | 正式证据 | 结论强度 |
|---|---|---|
| PI-MSCL 在密集标签下具有竞争力 | $r=1.0$ MAE $1.556\pm0.044\%$，四个 factorial 变体中最低 | 支持，但最低 RMSE/最高 $R^2$ 属于单尺度 CNN--LSTM |
| 70% 标签下结构监督可同时改善误差和轨迹 | PI-MSCL 相对 multi-scale no-PI：MAE 改善 0.045 pp，三项轨迹指标平均均改善 | 当前最强正面单元，仍只有 3 个配对重复和固定 split |
| 30% 标签下物理约束稳定改善轨迹规则性 | 单调违例率、累计上升超差和二阶差分的三组配对均为改善方向 | 支持结构监督作用 |
| 30% 标签下 PI-MSCL 稳定提高点精度 | MAE 配对为 +0.1444、+0.1546、-0.1706 pp；平均略恶化且 SD 很大 | 不支持，必须作为负结果报告 |
| 多尺度结构全面提高准确率 | no-PI 多尺度在 3/4 预算降低平均 MAE，但 4/4 预算 RMSE 更高、$R^2$ 更低 | 不支持；解释为误差分布权衡 |
| 标签越少，物理约束越有用 | $r=0.5/0.3$ 的 PI-MSCL 点误差收益不稳定或不利 | 不支持单调增强叙事 |

该证据表优先于早期开发结论。任何后续摘要、标题、图注或答辩材料若与此表冲突，必须回到正式 summary 和逐配对结果核对。
