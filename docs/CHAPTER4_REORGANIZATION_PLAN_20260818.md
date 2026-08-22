# 第四章重组修改计划（按评阅意见）

> 创建日期：2026-08-18  
> 当前状态：**重组设计、4.1 LaTeX 片段和 4.3 五模型单种子筛选实验已启动；正文由用户在另一对话修改。**  
> 本计划以用户提供的 13 张章节设计截图为唯一的章节结构依据；截图中的要求应当与后续补充的交接材料共同校核。

## 1. 本轮边界与占位规则

### 1.1 本轮已做与未做

- 已阅读项目交接说明 `AGENTS.md`、项目进展/实验计划 `docs/IMPROVEMENT_PLAN.md`、现有中文章节草稿 `第四章_修改版.md`，并核查到可编辑的 Springer 稿件 `latex/springer-sn-2024/manuscript.tex`、Overleaf 工程 `overleaf_submission_20260813/main.tex` 与根目录 `main.tex`。
- 已发现可复用旧图，包括 `figures/fig1_mae_vs_ratio.png`、`figures/fig2_ablation_heatmap.png`、`figures/figA_r03_all_models.png` 等；它们仅作为**临时版式占位图**候选，不能被表述为新实验结果。
- 已完成 4.3 的五模型单种子筛选实验设计并启动正式串行运行；结果只作筛选，绝不替代正式 `mean \pm std` 主表。
- 当前训练代码实际输入为 **16 个充电特征**。此前截图中的“14 features”和草稿中的其他特征数表述已被代码执行结果否定；后续正文、图表与实验说明均统一采用 16 features，除非用户另行修改数据管线并重跑全部比较。

### 1.2 强制占位标识

后续所有新生成的中文正文和 LaTeX 必须使用以下显式标识，直到获得正式实验归档数据后才可删除：

| 对象 | 正文/表格标识 | LaTeX 实现要求 |
|---|---|---|
| 数值结果 | `【模拟占位，须以正式实验结果替换】` | 单元格或表注中保留该句；不使用任何现有实验数值冒充新协议结果 |
| 统计形式 | `【模拟占位：mean ± std，待正式多 seed 结果替换】` | 不写成确定性结论 |
| 旧图 | `【旧图临时占位，须按本节实验协议重新生成后替换】` | 图注必须逐图包含该句 |
| 旧表 | `【表格结构占位，数据须按本节协议重新统计】` | 表注必须逐表包含该句 |
| 推断文字 | `【基于模拟占位结果的写作示例】` | 禁止将“优于”“显著”等结论写为已验证事实 |

模拟值仅用于检查表格、浮动体和段落排版；不应被提交、答辩或投稿版本保留。

## 2. 重组总原则

第四章从原先“数据准备—网络设计—分散实验”的叙事，改为一个连续的证据链：

`统一实验条件 → 标签预算下降时的主结果 → 同一稀疏场景的公平横向比较 → 因子消融与轨迹级机制解释 → 小结`。

因此，原稿中独立的“Validation of the trajectory-concentrated supervision setting”不再保留为完整小节；标签轨迹分配机制已在第 2.3 节说明，第四章只在 4.1 交代其作为统一实验条件的实现与复现实务。

方法章节中应保留模型结构、损失定义和标签分配的理论动机；第四章只回答“表现如何”“相较于谁如何”“哪个模块改善了哪一类性质”。

## 3. 目标目录与每节职责

### 4.1 Experimental setup（实验设置）

**唯一职责：一次性交代全章共享且可复现的实验条件，不报告实验优劣。**

建议包含以下二级内容（可在终稿中合并为连续段落，避免过碎）：

1. HUST 数据集范围、清洗/归一化流程；
2. battery-level train/validation/test split，且测试电池对所有训练过程完全不可见；
3. trajectory-concentrated label allocation 的执行规则（只概述并引用第 2.3 节定义）；
4. 标签预算及其独立随机掩码；
5. 重复试验与报告形式（所有结果统一报告 `mean ± std`）；
6. 点预测指标：MAE、RMSE、$R^2$；
7. 轨迹质量指标：monotonicity violation rate、cumulative upward excess、rate discontinuity；
8. 对比模型、超参数匹配和全部公平比较条件。

必须明确的公平性声明为：相同 battery split、相同特征数、相同 40-cycle 输入窗口、相同标签分配/掩码、相同测试电池；深度模型还应使用可比较的训练预算和早停规则。

**实现一致性要求：**所有当前可复现实验均使用 **16 features**。先前图片中“14 features”的表述不得直接写入论文；若要改为 14 features，必须先明确特征剔除规则，并从 4.2--4.5 的所有模型重新运行。

### 4.2 SOH estimation under different label budgets（不同标签预算下的 SOH 估计）

**这是主实验，只回答：随着可用 SOH 标签减少，完整 PI-MSCL 的跨电池 SOH 估计能力如何变化？**

- 标签预算固定为 `100%, 50%, 30%, 20%, 10%`；100% 作为 full-supervision upper reference，10% 是正式极限压力场景。
- 结果只聚焦完整 PI-MSCL，不在此处展开模块消融。
- **Table 2**：行是标签比例，列为 MAE、RMSE、$R^2$，每个值均以 `mean ± std` 写入；现阶段只填模拟占位数值并在表注标红/显式标识。
- **Figure 3**：横轴 label ratio，纵轴 MAE/RMSE；绘制 PI-MSCL 与一个代表性基线（建议 MS-CNN-LSTM）。旧图 `figures/fig1_mae_vs_ratio.png` 可作为临时占位候选，但其图注必须写明“旧图临时占位，须按 100/50/30/20/10% 协议重新生成后替换”。
- **Figure 4**：在 `10% labels` 下，从完全未见的测试集中选缓慢、中等、较快退化各一块电池，展示 True SOH 与 PI-MSCL prediction 三张子图。旧轨迹图只能作版式占位；最终必须来自严格 held-out cells。

该节段落顺序固定为：总体指标 → 标签预算退化趋势 → 极低标签下三条完整轨迹。不要把标签分配机制再次写成“验证实验”。

### 4.3 Comparison with alternative estimators（与替代估计器的比较）

**唯一职责：完整 PI-MSCL 与其它估计方法相比如何。**

- 固定代表性稀疏标签情景：`r = 20%`。该预算已用于 4.2 的标签预算曲线，因而可使两节的完整 PI-MSCL 结果严格对齐；10% 留给 4.4 消融与 4.5 敏感性等极限压力分析。
- 正式模型集合固定为五个：**XGBoost、GRU、Transformer-SOH、CNN-BiGRU-Attention、PI-MSCL**。LSTM、CNN-LSTM、Attention CNN-LSTM 和 MS-CNN-LSTM 不再进入新 Table 3，以避免重复堆叠项目内部消融模型。
- 新增横向模型均为文献核心网络在统一协议下的复现：Transformer-SOH 对应 Shu *et al.* (*Journal of Energy Storage*, 2025, DOI: 10.1016/j.est.2024.115200)；CNN-BiGRU-Attention 对应 Wu *et al.* (*World Electric Vehicle Journal*, 2025, DOI: 10.3390/wevj16090487)。两者均不迁入原论文的数据集特异特征工程或元启发式调参，以维持公平性。
- **Table 3**：模型 × MAE/RMSE/$R^2$。本轮单种子筛选结果必须明确标作“single-run screening”；正式主表补齐相同多种子重复后才写 `mean ± std`。该表仅比较总体点估计性能，不塞入单调性权重、rate loss 参数或 violation 结果。
- **Figure 5**：同一 20% 协议下的逐 held-out cell 绝对误差分布，采用上下排布的横向箱线图：(a) MAE；(b) RMSE。每个点对应一个测试电池，而不是窗口级伪重复。该图与 Table 3 共同说明模型差异的总体水平与跨电池离散性。
- 统一公平条件必须写清：**same 16 features + same 40-cycle input + same label allocation + same validation/test cells + same training budget/early stopping**。
- 当前筛选固定 `split_seed=42`、`training_seed=2262`、`mask_seed=3262`，并在 `r = 20%` 下运行五模型比较。它仅用于同一协议下的单次筛选；不应被包装为正式泛化结论。

#### 已启动的后续训练队列（2026-08-19）

五模型 `r = 20%` 单种子筛选已完成；该结果只可作为 Table 3 的内部筛选/版式验证数据。`experiments/run_chapter4_remaining_experiments.py` 的队列另用于 4.2 的 5 项 PI-MSCL 标签预算（100/50/30/20/10%）、4.4 的 8 项 10% 完整 $2^3$ 消融及 4.5 的 36 项 10% $\lambda_{\mathrm{mono}}\times\lambda_{\mathrm{rate}}$ 扫描。此前 5% 队列的已完成结果仅作内部压力诊断，不进入正文。上述结果都不替代最终多种子主表。
- 文字只解释完整模型的相对点预测表现；若某个指标未最优，应如实讨论 point accuracy 与 trajectory quality 的差别，不能声称全面最优。

### 4.4 Ablation and trajectory-level analysis（消融与轨迹级分析）

**这是方法解释实验：回答 Multi-scale、Mono、Rate 是否有效，以及分别改善什么。** 原稿中分散的消融、单调性、速率连续性和轨迹分析应并入本节，避免三张重复表。

#### 4.4.1 Complete factorial ablation

固定 `10% labels`，报告三因素完整 $2^3=8$ 组：

| Multi-scale | Mono | Rate | 模型含义 |
|---:|---:|---|
| × | × | × | CNN-LSTM baseline |
| ✓ | × | × | MS-CNN-LSTM |
| × | ✓ | × | PI CNN-LSTM |
| × | × | ✓ | Rate CNN-LSTM |
| ✓ | ✓ | × | MS + mono |
| ✓ | × | ✓ | MS + rate |
| × | ✓ | ✓ | mono + rate |
| ✓ | ✓ | ✓ | PI-MSCL |

如果未来正式实验无法提供完整因子组合，最低限度是 `baseline → +multi-scale → +mono → +rate`，但本计划优先完整 factorial，以识别模块间交互。

#### 4.4.2 一张合并性能表

**Table 4** 合并替代旧的、彼此重叠的三张表。列为：MS、Mono、Rate、MAE、RMSE、Violation、Upward excess、Rate variation。

- MAE/RMSE：整体数值准确度，对应 Multi-scale 的主要作用；
- Violation/Upward excess：轨迹回升与累计回升量，对应 $mathcal{L}_{mono}$；
- Rate variation：循环间隔归一化后的速率不连续度，对应 $mathcal{L}_{rate}$。

速率指标不再沿用未经循环间隔处理的 second-difference roughness。训练损失和测试指标均采用循环间隔归一化的相邻变化率；后续 LaTeX 应采用：

$$r_{i,k}=\frac{\hat{y}_{i,k+1}-\hat{y}_{i,k}}{c_{i,k+1}-c_{i,k}},$$

并报告

$$R_{\mathrm{rate}}=\frac{1}{N}\sum |r_{i,k+1}-r_{i,k}|,$$

或其 RMS 形式（最终只能二选一并全章统一）。单调性累计上升量可写为：

$$E_{\uparrow}=\sum_t\max(0,\hat{y}_{t+1}-\hat{y}_t-\epsilon).$$

#### 4.4.3 轨迹行为图

**Figure 6**：同一块 held-out battery、`10% labels` 下，绘制 True SOH、CNN-LSTM、MS-CNN-LSTM、MS+mono、PI-MSCL；局部放大容易出现回升、锯齿或 rate jump 的区间。图的解读分工必须清楚：multi-scale 改善数值精度，mono 抑制上行 excursion，rate continuity 改善局部形状。

现有 `figures/figA_r03_all_models.png` 或已有轨迹图仅可临时放入该位置，且不得因标签比例或模型集合不一致而被描述为 5% 的真实结果。

### 4.5 Sensitivity to trajectory-constraint weights（轨迹约束权重敏感性）

**本节只研究 $\lambda_{\mathrm{mono}}$ 与 $\lambda_{\mathrm{rate}}$；删除与“reference-trajectory selection”不相符的旧标题及内容。**

- 固定最严苛的正式标签预算：`r = 10%`，以观察物理先验在稀疏标签下的实际影响。
- 使用二维权重扫描：

  $$\lambda_{\mathrm{mono}}\in\{0, 0.05, 0.1, 0.2, 0.3, 0.5\},$$

  $$\lambda_{\mathrm{rate}}\in\{0, 0.01, 0.05, 0.1, 0.2, 0.3\}.$$

  具体范围可在取得训练稳定性资料后调整，但必须保留二维而非仅扫描 $\lambda_{\mathrm{mono}}$ 的设计。
- **Figure 7** 由两个二维热图组成： (a) 横轴 $\lambda_{\mathrm{mono}}$、纵轴 $\lambda_{\mathrm{rate}}$、颜色为 MAE；(b) 相同横纵轴、颜色为轨迹一致性指标，优先使用 $E_{\uparrow}$ 或 $R_{\mathrm{rate}}$，最终只能选定一个作为面板 (b) 主指标并全文一致。
- 要分析的不是“哪个测试集数值最低”，而是是否存在 accuracy--trajectory consistency 的合理平衡区间：增大权重是否改善轨迹质量，是否以牺牲点预测精度为代价。
- 严格的调参链为：`validation set → 选择 λmono/λrate → 固定参数 → test evaluation`。敏感性图可展示测试集行为，但最终超参数选择及其理由必须只来自 validation；不可基于 test MAE 反选最优权重。

### 4.6 本章小结

按三句话收束，不引入新数据：

1. 标签预算下降时完整模型的总体稳健性（引用 4.2）；
2. 20% 稀疏场景中相对于替代方法的点预测定位及跨电池离散性（引用 4.3）；
3. 三模块与相应指标之间的可解释对应关系（引用 4.4）；
4. 权重选择的验证集原则与 accuracy--trajectory consistency 权衡（引用 4.5）。

## 4. 现有材料的迁移、保留与删除清单

| 现有内容 | 目标位置 | 处理方式 |
|---|---|---|
| `第四章_修改版.md` 的数据集、预处理、电池级划分与 Label Masking | 4.1 | 精简、合并；方法定义只引用第 2.3 节 |
| 原“部分生命周期监督设定验证” | 删除独立小节 | 其必要信息并入 4.1，结果重组至 4.2 |
| 原 full supervision 结果表与散点图 | 4.2 的 100% 行或移除 | 不再形成独立主实验；旧数值不得沿用到新协议 |
| 原 unseen batteries 轨迹图 | 4.2 Figure 4 / 4.4 Figure 5 | 仅可临时复用版式，图注标注待替换 |
| 原 robustness 图表 | 暂不纳入目标主线 | 后续材料若要求，可作为附录或独立扩展实验 |
| 旧的多张消融/违规/速率表 | 4.4 Table 4 | 合并为一张对应模块—指标关系的表 |
| `latex/springer-sn-2024/generated_results/*.tex` | 待后续核验 | 不能直接引用，因其命名与新 5%/10% 协议可能不一致 |

### 4.1 旧 Table 2、旧 Table 3 的处置

- **旧 Table 2：`Realised protocol-diagnosis quantities`**——从主文删除。其“12--13 complete cells、32--33 unlabeled cells、prefix 682--1897”等内容是标签分配的具体实例，而非结果证据；第 2.3 节已经定义该机制。若担心读者不能理解 10% 的稀疏程度，只在新 4.1 加一句定性说明：*Under the 10\% allocation, supervision is concentrated within only a small subset of the training trajectories.* 不再报告每个 seed 的具体数量。
- **旧 Table 3：`Empirical check of the soft trajectory prior`**——从主表删除或压缩为一句辅助说明。原始的 pair 数及比例统计不能证明当前由 $\mathcal{L}_{\mathrm{mono}}+\mathcal{L}_{\mathrm{rate}}$ 共同构成的 PI 有效性；真正的证据应是新 Table 4 的消融结果和轨迹级指标。

### 4.2 旧章节和旧表的精确迁移

| 当前内容 | 新位置/处理 |
|---|---|
| 旧 4.1 标签协议验证 | 大部分删除；必要设置并入新 4.1 |
| 旧 Table 2 | 删除 |
| 旧 Table 3 | 删除或压缩为一句辅助说明 |
| 旧 4.2 不同标签比例 | 重构为新 4.2 主实验 |
| 旧 Table 4 | 拆分：最终模型结果进入新 Table 2；消融结果进入新 Table 4 |
| 旧 4.3 baseline comparison | 基本保留为新 4.3；更新为 16 features 与五模型正式集合 |
| 旧 Table 5 | 在结果可复现后重制为新 Table 3 |
| 旧 4.4 factorial ablation | 大幅扩展为新 4.4 |
| 旧 Table 6 | 由新 Table 4 的 3-component factorial ablation 替代 |
| 旧 4.5 trajectory analysis | 合并至新 4.4，并以 Figure 5 承担核心证据 |
| 旧 Table 7 | 合并至新 Table 4，不再单独保留 |
| 旧 4.6 sensitivity | 改写为新 4.5 双权重敏感性 |
| 旧 Table 8 | 替换为 Figure 6 的 mono + rate 双权重热图 |

### 4.3 最终图表体系（3 表 + 5 图）

| 编号 | 位置 | 内容 |
|---|---|---|
| Table 2 | 4.2 | PI-MSCL performance across label budgets |
| Figure 3 | 4.2 | MAE/RMSE versus label ratio |
| Figure 4 | 4.2 | Representative held-out SOH trajectories under severe label scarcity |
| Table 3 | 4.3 | Comparison with alternative estimators at 20% labels |
| Figure 5 | 4.3 | Per-cell MAE and RMSE distributions across alternative estimators at 20% labels |
| Table 4 | 4.4 | Factorial ablation with point-estimation and trajectory-level metrics |
| Figure 6 | 4.4 | Representative trajectory comparison among ablation variants |
| Figure 7 | 4.5 | Sensitivity heatmaps for $\lambda_{\mathrm{mono}}$ and $\lambda_{\mathrm{rate}}$ |

除上述 3 张表和 5 张图外，不在第四章主文保留重复的协议诊断、源数据单调性或单模块重复统计表；必要的扩展结果后续移至附录。

## 5. 后续分阶段交付顺序

在收到用户补齐材料后，按以下顺序执行，每一步均保留上述占位标识：

1. 建立 `chapter4_reorganized_cn.tex`：仅写 4.1 的统一实验设置与 4.2 的主实验正文/图表骨架；
2. 运行并归档 4.3 的五模型单种子筛选；之后新增统一对比正文及 Table 3 骨架，显式保留 `single-run screening` 标识；
3. 新增 4.4 完整因子消融、指标定义、Table 4 与 Figure 5 骨架；
4. 新增 4.5 双权重敏感性分析及 Figure 6 骨架，并写入 validation-only 的选参声明；
5. 生成完整第四章 LaTeX（4.1--4.6），并做交叉引用、表图编号和占位标识检查；
6. 将同一正文同步到 `overleaf_submission_20260813/main.tex`（并根据实际入口确认是否还需同步根目录 `main.tex` 与 Springer `manuscript.tex`）；
7. 在正式数据到位后，以真实 `mean ± std`、匹配标签比例的图和可追溯实验归档替换全部模拟/旧图占位项。

## 6. 生成正文前必须确认的交接项

1. 最终论文模型名称：PI-MSCL、PI-MS-CNN-LSTM 或 MS-PI-CNNLSTM；全文只能保留一个主名称。
2. 特征维度：当前执行管线已确认 16；若用户未来坚持 14，须先改动数据管线并重跑所有章节实验。
3. 标签预算：最终正文协议采用 `100/50/30/20/10%`；此前截图中的 5% 及其已完成内部压力测试不进入正式主结果，正式结果到位前不得混用。
4. 输入窗口：确认“40-cycle input”是否为所有比较模型的实际配置。
5. `10%` 的完整 8 组因子实验、`10%` 的基线比较，以及三块完全未见电池的轨迹预测是否已有可核验结果；若无，只保留模拟占位和待办，绝不虚构完成状态。
6. 需要同步的最终稿入口：Overleaf 文件夹、Springer 单栏稿、根目录 `main.tex` 中哪一个是提交主文件。

## 7. 验收清单

- [x] 本地已建立重组计划文档。
- [x] 已启动 4.3 五模型单种子筛选实验；其结果明确限定为筛选，不用于替代正式多种子统计。
- [x] 已规定模拟数据、旧图、旧表的强制替换标识。
- [x] 已把截图要求的 4.1–4.5 结构、3 表 4 图体系、10% 完整因子消融与双权重敏感性纳入计划。
- [ ] 收到其余材料后，逐节生成 LaTeX。
- [ ] 同步改写 Overleaf 可上传稿的正文 `.tex`。
- [ ] 正式实验与图表归档完成后，替换全部占位内容并进行编译核验。
