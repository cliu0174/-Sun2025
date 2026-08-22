# 英文稿章节密度与中英文一致性审计（2026-08-07）

## 1. 本轮写作硬约束

1. 英文稿以中文版的研究问题、方法组件、变量含义和已确认事实为内容依据；英文允许重组、澄清和补充边界，但不得反转中文版结论。
2. 后续用户已经确认的新口径优先于中文版中的旧口径。凡两者不能同时成立，先记录为待同步项，不通过含混措辞掩盖冲突。
3. 小节必须承担独立的论证任务。若小节没有至少两段实质论证，且没有独立的公式、表格或图作为证据，则优先与相邻小节合并。
4. 图表只能增加证据，不用于机械凑页数。图注和表注保持简短，结果含义在正文展开。
5. 所有正式结果均按预先确定的固定电池划分和配对多种子协议汇总；正、负结果均保留，不挑选随机种子。

## 2. 当前英文稿密度诊断

| 部分 | 当前证据密度 | 处理决定 |
|---|---:|---|
| Introduction | 7 个实质段落，约 1239 词 | 保留四步逻辑：工程动机、方法进展、标签预算缺口、本文贡献；压缩重复的模型罗列。 |
| Problem formulation and learning interface | 2 段、2 个公式 | 保留，补清循环级随机掩码的适用边界。 |
| Overview of PI-MSCL | 2 段、1 幅架构图 | 保留；架构图为本节核心视觉证据。 |
| Multi-scale convolution | 2 段、2 个公式 | 保留，避免仅按网络层逐项说明，强调三种感受野与退化尺度的对应关系。 |
| LSTM temporal modeling | 1 段、2 个公式 | 偏薄。与多尺度卷积合并为“Multi-scale representation and temporal state modeling”，或补入完整状态映射和张量尺寸表后再独立。当前优先合并。 |
| Physics-informed objective | 3 段、4 个公式 | 保留；必须明确容忍阈值、延迟激活和仅在同一电池内计算。 |
| Dataset and SOH definition | 3 段、1 个公式 | 保留。 |
| Feature construction and preprocessing | 2 段、1 个表 | 保留；当前 16 维正式口径需与中文旧表同步。 |
| Split and sparse-label protocol | 3 段、1 个公式 | 保留；是主叙事的实验定义。 |
| Compared models and ablation | 2 段、1 个表 | 保留并扩展为同尺度基线、2×2 因子消融及 published-method context 三层。 |
| Training configuration | 2 段 | 保留；增加软件/硬件、早停、种子和模型选择规则后具备可复现性。 |
| Evaluation metrics | 1 段、2 个公式 | 偏薄。与统计汇总、轨迹指标合并扩展，或并入训练与评估协议。当前计划改为“Training, statistical aggregation, and evaluation metrics”。 |
| Results: Full supervision | 1 段、无正式表图 | 当前为占位。正式矩阵完成后与同尺度基线表共同呈现。 |
| Results: Label budgets | 2 段、无正式表图 | 当前为占位。正式矩阵完成后加入均值±标准差表和标签预算曲线。 |
| Results: Trajectory stability | 2 段、无正式表图 | 当前为占位。正式结果后加入代表性电池轨迹、误差演化和单调违规率。 |
| Results: Factorial ablation | 2 段、1 个公式 | 当前为占位。正式结果后加入 2×2 交互效应表/图；应放在主结果之后。 |
| Discussion | 未独立形成 | 从 Results 中分离，讨论物理项的标签依赖收益、失败情形、受控掩码外推边界和单数据集限制。 |
| Anomaly-risk section | 已用 `\iffalse` 排除 | 不用来扩充主稿。其数据来源和主任务不一致，除非用户以后确认恢复并提供可审计的真实实验来源。 |

## 3. 已发现的中英文待同步项

| 编号 | 中文版旧内容 | 当前确认口径 | 处理 |
|---|---|---|---|
| C-01 | 中文 4.4 声称部分监督来自自建数据集，且只在部分循环标定 SOH。 | 当前主线是在完整标注 HUST 训练电池内人为随机保留固定比例的循环级标签；测试电池保持完整标签。 | 英文坚持用户后续确认的 HUST 受控标签预算口径；中文版 4.1、4.4、4.6 后续需同步改写，避免投稿版本互相冲突。 |
| C-02 | 中文框架说明和特征表为 14 维。 | 项目记忆卡、正式泄漏隔离管线和架构图统一为 16 维。 | 英文和正式实验统一用 16 维；需生成 16 维特征表并回写中文版。 |
| C-03 | 中文表 4-2 至 4-6 为旧版单次/旧管线结果。 | 正式主表采用 split seed 42，三个训练种子和独立 mask seeds 的配对重复。 | 旧值只作历史资产，不进入最终英文结论；正式结果完成后同步替换中文表。 |
| C-04 | 中文第 4.5 节使用自建/注入异常场景。 | 当前英文主任务仅评价 HUST 受控标签预算，异常章节已从活动证据链排除。 | 保持排除并记录；不把合成风险结果包装成主实验。 |

## 4. 术语账本

| 概念 | 英文唯一写法 | 禁止/仅历史写法 |
|---|---|---|
| 健康状态 | state of health (SOH) | health degree |
| 部分生命周期监督 | partial lifecycle supervision | partial-life labels（仅非正式说明） |
| 循环级标签保留率 | cycle-level label-retention ratio, $r$ | random rate |
| 标签掩码 | label masking | sample deletion |
| 物理一致性 | physical consistency | physical truth |
| 软单调性约束 | soft monotonicity constraint | strict monotonic degradation |
| 多尺度物理一致性 CNN-LSTM | PI-MSCL | MS-PI-CNNLSTM、PI-MS-CNNLSTM（历史名） |
| 测试电池 | unseen test cells | unknown batteries |
| 标签预算 | label budget | missing-data rate（会混淆特征缺失） |

## 5. 计划中的最终章节结构

1. Introduction
2. Methodology
   1. Problem formulation and learning interface
   2. PI-MSCL overview
   3. Multi-scale representation and temporal state modeling
   4. Masked data fitting and physical-consistency objective
3. Experimental setup
   1. HUST dataset, SOH definition, and 16-dimensional feature set
   2. Battery-level split and controlled label-budget protocol
   3. Compared methods and factorial ablation
   4. Training, statistical aggregation, and evaluation metrics
4. Results
   1. Full-supervision and same-protocol baseline comparison
   2. Robustness to decreasing label budgets
   3. Trajectory-level behavior on unseen cells
   4. Factorial ablation and architecture–physics interaction
   5. Failure cases and unfavorable outcomes
5. Discussion
   1. What the label-budget experiment establishes
   2. Why and when the physical term helps or hurts
   3. Relation to recent weak-label, transfer, and physics-informed studies
   4. Limitations and external-validity boundary
6. Conclusion

该结构不设置只有一段文字的独立小节。最终是否保留 Results 4.5 取决于正式多种子结果中是否出现具有独立解释价值的失败模式。

## 6. 本轮结构落实（2026-08-07）

- 已将多尺度卷积与 LSTM 的薄小节合并为 `Multi-scale representation and temporal state modeling`。
- 已将训练配置、统计汇总和评价指标合并为一个可复现性小节。
- 已把 `Results and Discussion` 拆为独立的 `Results` 与 `Discussion`；Discussion 仅保留两节、共 7 个实质段落，分别承担“受控标签预算的可解释范围”和“物理一致性的权衡、失败与使用边界”。
- 当前编译为 Springer A4 单栏 20 页；无 undefined reference/citation、overfull box 或 float-too-large。正式结果表图尚未插入，最终页数预计自然增加。
- 新增 Results 4.5 仍是条件性决定：只有正式多种子矩阵出现可重复且具有独立解释价值的不利模式时才保留；否则负面发现并入 4.4 和 Discussion，避免单段小节。
