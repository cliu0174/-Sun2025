# 当前完成度与正文图件预览说明

> 核对日期：2026-08-07  
> 状态边界：固定划分正式主矩阵已按用户后续授权恢复；神经网络扩展基线仍不与主矩阵并发。

## 1. 当前完成度结论

论文的叙事、方法、实验协议、代码公平性修复、参考文献和架构图已经完成；真正阻止定稿的部分集中在正式多种子结果及其下游图表。当前 Springer 主稿已编译为 21 页，达到页数目标，但页数不能代替尚缺的正式证据。固定划分 `v3_leakage_free` 主矩阵当前归档 32/48 个正式组合，完整单尺度半矩阵以及多尺度/no-PI 在 $r=1.0$ 和 $r=0.7$ 的三重复单元已经冻结并审计，后台继续运行多尺度半矩阵；已有归档不覆盖，一个 `smoke=true` 目录不计入正式结果。

### 1.1 已完成

- 重构前备份和 Springer Nature 2024-12 v3.1 单栏工程；
- 以中文版为内容主体的 Introduction、Methodology、Experimental Setup 和 Results 结构；
- 随机循环级标签预算、固定 split seed 42 和独立 mask/train seed 口径；
- 16 维特征口径、训练电池拟合标准化器和电池边界保持窗口化；
- 四模型 2×2 公平消融入口、结果归档、汇总器和轨迹指标代码；
- 43 条带 DOI 的 BibTeX，其中活动正文使用 42 条；42 篇期刊论文已由 Crossref 核验，HUST 数据集已由 DataCite 核验，详见 `docs/REFERENCE_AUDIT_20260807.md`；
- 可编辑 PI-MSCL 架构图 v5，以及 PNG/PDF/SVG/TIFF 预览；
- 架构图采用三色特征图薄片、双层时间展开 LSTM 和紧凑双损失路径；XML 解析通过，连接检查为 0 crossings；
- 正式结果绘图脚本：主结果/消融组图和独立子图、轨迹/误差组图和独立子图；
- 中文版 Tables 1--5 的机器可读旧结果登记，以及 600 dpi legacy 候选组图和子图；
- 多轮审稿人式自检、结果资产审计和参考文献审计。

### 1.2 尚未完成

| 优先级 | 未完成项 | 当前证据 | 完成条件 |
|---|---|---|---|
| P0 | 固定划分正式主矩阵 | 32/48 个正式组合已归档，无 `error.json`；24-run 单尺度半矩阵及多尺度/no-PI 的两个三重复预算单元完整 | 断点续跑完成 48 次并通过完整性校验 |
| P0 | 四标签比例 `mean ± std` 主表 | 汇总器会拒绝不完整矩阵 | 48 次归档完整、三组重复成对齐全 |
| P0 | 正式 Fig. 3 标签预算曲线 | 脚本完成，无完整数据 | 从正式汇总生成 MAE/RMSE 组图和子图 |
| P0 | 正式 Fig. 4 代表电池轨迹 | 选择规则和脚本完成 | 完整预测归档后按跨模型中位规则生成 |
| P0 | 正式 Fig. 5 二因素消融 | 脚本完成，无完整配对统计 | 从四模型同 seed 配对结果计算主效应和交互 |
| P0 | Results 4.1--4.4 的最终数字 | 当前只有结果壳和单 seed 诊断 | 表、图和正文均由正式汇总填充 |
| P0 | 摘要与结论的定量结果 | 当前故意不写旧数字 | 正式结果冻结后写入 |
| P1 | Springer 单栏 20 页以上 | 当前 21 页 | 正式结果加入后复核，禁止通过空白或重复内容维持页数 |
| P1 | 独立稀疏标签协议图是否保留 | 当前任务说明已包含在架构图 panel (a) | 作者决定合并在 Fig. 1 或另设 Fig. 2 |
| P1 | 旧中文版结果的投稿定位 | 已有候选图和未激活 LaTeX 模块 | 作者决定正文、补充材料或仅内部追溯 |
| P2 | 架构图原生 Draw.io 检查 | XML、PNG/PDF/SVG/TIFF 与正文实页均通过本地检查 | 投稿前在 Draw.io desktop 打开一次核对字体回退 |
| P2 | 投稿元数据 | 单位、Funding、Acknowledgements、代码仓库为 TBC | 作者提供正式信息 |
| P2 | 目标期刊最终格式 | 当前为 Springer 通用数值引用模板 | 确认期刊后复核栏目、字数和图表规范 |
| 可选 | 独立 battery-split 敏感性、额外现代基线 | 未纳入固定 split 主表 | 仅在主表完成后决定，不与主表混合 |

异常检测、连续前缀标签和周期性标签不是当前主线必需项，不计入论文完成门槛。

## 2. 架构图当前样式（v5）

### 2.1 版式

- 类型：双层 schematic-led composite；
- 阅读路径：`(a) 完整轨迹与稀疏标签 → 三尺度 Conv1D → 双层 LSTM → SOH`，再进入 `(b) masked MSE + soft monotonicity`；
- panel (a)：将运行小曲线、`40×16` 输入薄片、`k=3/7/15` 三色卷积分支、128-channel 融合、两层时间展开 LSTM、回归头和输出 SOH 曲线合为一条视觉主路径；
- panel (b)：将保留标签的数据路径和符合 `cmin/K` 配对条件（可包含无标签循环）的结构路径汇合为总损失。

### 2.2 视觉语义

| 色系 | 含义 |
|---|---|
| 绿色 | 观测/预测数据、完整特征轨迹和 SOH 曲线 |
| 蓝/绿/紫 | 短、中、长三种卷积感受野及其特征图 |
| 黄色 | LSTM 跨循环时序建模 |
| 红色 | masked MSE、软单调约束和总损失 |
| 灰色 | 容器、坐标轴和主数据流 |

整体采用白底、低饱和填充、深灰箭头、圆角容器和少量曲线示意。它不是逐层堆叠的传统网络框图，而是把研究问题、模型和学习目标放在同一证据路径中。

### 2.3 当前预览资产

- 可编辑源：`docs/PI-MSCL_architecture_v5.drawio`
- 聊天/快速检查：`docs/PI-MSCL_architecture_v5_preview.png`
- LaTeX 正文：`docs/PI-MSCL_architecture_v5_preview.pdf`
- 二次编辑/矢量检查：`docs/PI-MSCL_architecture_v5_preview.svg`
- 600 dpi 位图归档：`docs/PI-MSCL_architecture_v5_preview.tiff`

## 3. 正文插图的预定版式

### Fig. 1 — PI-MSCL 架构图

- archetype：schematic-led composite；
- 正文宽度：单栏全宽；
- 当前状态：已插入活动主稿；
- 最终交付：`.drawio + .pdf + .svg + 600 dpi .png`。

### Fig. 2 — 稀疏标签协议示意（待决定是否独立）

- archetype：简洁示意图；
- 画面：同一训练电池的完整特征曲线、随机保留 SOH 标定点、未标注点和“features retained / labels masked”标记；
- 当前状态：内容已包含在 Fig. 1 panel (a)，尚未另画，避免重复；
- 推荐决定：若 Experimental Setup 需要独立引用，则从 Draw.io 源重排成独立页面，不从 Fig. 1 截图。

### Fig. 3 — 标签预算主结果

- archetype：1×2 quantitative grid；
- panel (a)：四模型 MAE 随 `r` 变化，报告三次配对重复的 `mean ± std`；
- panel (b)：同一模型顺序和色彩下的 RMSE；
- 组图尺寸：约 7.2×3.65 in；
- 独立子图：MAE、RMSE 各约 6.5×4.35 in；
- 脚本：`scripts/plot_paper_main_results.py`；
- 状态：版式和导出代码完成，等待正式矩阵。

### Fig. 4 — 代表电池轨迹与误差

- archetype：2×1 aligned quantitative panels；
- panel (a)：真实 SOH 与四个 factorial 模型的预测轨迹；
- panel (b)：同一循环轴下的预测误差（percentage points）；
- 组图尺寸：约 7.2×6.0 in；
- 独立子图：轨迹、误差各约 6.5×4.35 in；
- 重复选择：先取三组配对 seed 中、跨四模型平均 test MAE 最接近中位数的一组；
- 电池选择：再在所选重复内取跨四模型平均 MAE 最接近测试电池中位数者，不事后挑最好个例；
- 脚本：`scripts/plot_paper_trajectories.py`；
- 状态：版式、选择规则和数据审计已实现，等待完整正式预测。

### Fig. 5 — 架构与物理约束二因素效应

- archetype：1×2 paired-effect grid；
- panel (a)：多尺度架构相对单尺度的 MAE 变化；
- panel (b)：软单调约束相对无物理模型的 MAE 变化；
- 横轴：`r={1.0,0.7,0.5,0.3}`；纵轴：paired MAE reduction (p.p.)，正值表示右侧变体降低误差，并保留零效应参考线；
- 每个点由共享 split/mask/train seeds 的逐配对 MAE 差值汇总为 mean ± sample SD，误差棒不使用“两个组均值标准差相加”的非配对近似；
- 组图尺寸：约 7.2×3.25 in；独立子图各约 6.5×4.35 in；
- 脚本：`scripts/plot_paper_main_results.py`；
- 状态：等待同 seed 的完整 2×2 配对结果。

### Results 图表接入映射（防止重复证据）

| Results 小节 | 正式图表 | 证据任务 |
|---|---|---|
| 4.1 Performance across SOH-label budgets | `fig3_performance_vs_label_budget_group`、`table_label_budget.tex`、`table_label_budget_r2.tex` | MAE/RMSE 趋势与 MAE/R-squared 精确读数 |
| 4.2 Same-protocol complementary baselines | `table_all_models_full_supervision.tex`、`table_all_models_endpoint_budget.tex` | 8 模型同尺度准确率与端点标签预算敏感性 |
| 4.3 Trajectory stability | `fig4_trajectory_and_error_group`、`table_trajectory_r0p3.tex` | 中位重复/中位电池轨迹、误差和三个结构指标 |
| 4.4 Factorial ablation | `fig5_ablation_effects_group`、`table_factorial_effects.tex` | 多尺度、物理约束及其配对效应解释 |

`table_full_supervision.tex` 是主四模型的备用/审计表。若 8 模型同协议表完整，则不在正文重复插入该四行子集；若补充基线矩阵未完成，则只能使用四模型表并明确缩小比较范围，不能用 smoke 或异协议数值补齐。

### Legacy 候选图 — 不属于当前正式编号

- 类型：1×2 quantitative grid；
- panel (a)：中文版 Table 4 的 MAE 点估计；
- panel (b)：中文版 Table 3 的 2×2 RMSE 点估计；
- 用途：展示现有结果的预期视觉风格，不作为 `v3_leakage_free` 正式证据；
- 当前资产：组图和两个独立子图均已生成。

## 4. 统一预览与投稿格式

| 用途 | 格式 | 要求 |
|---|---|---|
| 对话中快速预览 | PNG | 600 dpi 原图；应用内直接显示 |
| LaTeX 正文插入 | PDF | 矢量线条和字体，使用 `\includegraphics` |
| 后期编辑 | SVG | 文本保持可编辑，不嵌入栅格图 |
| 位图投稿/归档 | PNG 或 TIFF | 600 dpi，白底，按目标期刊要求选择 |
| 架构图源文件 | Draw.io | 所有节点、文本和连接线可编辑 |
| 定量图源数据 | CSV/JSON + Python | 图件可从底层结果完整重建 |

所有组图和独立子图必须分别由同一脚本、同一数据源直接生成；独立子图不得从组图裁切。组图使用小写粗体 panel labels，图注只说明对象、条件、重复次数和误差棒定义，趋势解释放在正文。

## 5. 本轮视觉自检

- 架构图：两轮视觉修订后未发现标签裁切、断裂箭头或跨模块线条交叉；`40×16`、两层 LSTM、hidden size 64 和物理参数均已明确。
- 正文实页：Tectonic/BibTeX 编译为 20 页；Fig. 1 位于第 6 页，单栏缩放后主流程、分支颜色、层/时间步和双损失路径仍可辨认。
- Legacy 候选图：图例、坐标轴、线型和标记清晰；颜色不是唯一编码；PNG/TIFF 为 600 dpi，PDF/SVG 已生成。
- 正式 Fig. 3--5：目前只有版式和脚本，不能以 legacy 图或单 seed 结果替代最终预览。
## 6. 2026-08-07 23:19 状态覆盖更新

本节覆盖文档前部仍保留的阶段性计数，防止把历史快照误读为当前状态。

- 固定划分正式主矩阵已由 32/48 更新为 **48/48 完成**；16 个模型×预算单元均为 3 次配对重复，0 缺失预测，0 错误归档。
- 正式 Fig. 3--5 已全部生成，不再处于“等待矩阵”状态。每组图及其两个独立子图均有 PNG/TIFF/PDF/SVG；PNG 元数据为 599.9988 dpi。
- 正式表 `table_full_supervision.tex`、`table_label_budget.tex`、`table_label_budget_r2.tex`、`table_trajectory_r0p3.tex` 和 `table_factorial_effects.tex` 已从严格 summary 生成并插入活动正文。
- Abstract、Results 4.1/4.3/4.4、Discussion 和 Conclusion 已改为正式数值与过去时论述；30% 标签下的不利准确率—轨迹一致性权衡已显式报告。
- 最新活动 PDF 为 **24 页**。Results 实页已检查，图表无裁切或溢出；将表格浮动位置由 `[t]` 改为 `[htbp]` 后，Results 标题稳定出现在表 4--5 之前。
- 同协议端点基线已启动，目标为 24 次正式运行（XGBoost、LSTM、GRU、attention CNN--LSTM × 两端点 × 三配对重复）。在其完成前，4.2 仍是唯一未闭合的结果小节。
- 架构图 v5 维持当前定稿候选，不再增加注意力、伪标签或其他代码未实现模块。投稿前仍保留一次原生 Draw.io 字体回退检查作为 TBC。

## 7. 2026-08-08 最终证据覆盖更新

- 同协议补充基线已完成 24/24 个正式运行，烟雾测试由 `smoke=true` 显式排除；三张八模型正式表已经生成并写入 4.2/4.3。
- 4.2 已由计划时占位文改为过去时结果；4.3 使用八模型统一轨迹表，不再只展示 PI-MSCL 因子家族。
- 最新 Springer PDF 为 **25 页**。Fig. 1 及第 13--20 页的主结果、补充基线、轨迹、消融、Discussion 和 Conclusion 已完成实页视觉检查。
- 测试 26/26 通过；编译无 overfull、undefined citation/reference、float-too-large 或 LaTeX error。
- 当前正式图表映射中，4.3 应读取 `table_all_models_trajectory_r0p3.tex`；旧 `table_trajectory_r0p3.tex` 仅保留为四模型审计资产，不进入活动正文。
