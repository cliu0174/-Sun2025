# PI-MSCL 叙事迁移：轨迹集中式 SOH 标签协议

## 1. 叙事决策

论文的主应用背景调整为：高质量容量/SOH 标定成本高，因而在研发、台架验证或有限诊断资源下，通常只有少数参考电池能够获得连续的早期或早中期 SOH 标定轨迹；大量其他循环或电池仍可提供充电特征和循环顺序，但没有可用于回归训练的数值 SOH 标签。

这是一种**轨迹集中式标签预算**，而不是随机地丢失每一个循环的原始数据。本文将它称为 **trajectory-prefix-concentrated SOH-label protocol**。该协议是对有限标定资源的受控代理，不等同于已经验证的在线 BMS 部署、定期校准或自然现场缺标过程。

原有的每电池随机循环级标签掩码保留为 **distributed-random SOH-label protocol**。它不再承载主应用主张，而用于在不引入标签时间覆盖偏差时，单独分析标签数量、模型架构和结构先验的机制。

## 2. 一句话中心论点

> In cross-cell SOH estimation with a limited number of continuously calibrated reference trajectories, complete but numerically unlabelled charging windows can still provide ordered structural information; PI-MSCL tests whether multi-scale representation and a soft monotonicity prior convert that information into improved point estimation or, when numerical anchors are insufficient, into more trajectory-consistent predictions.

中文：在跨电池 SOH 估计中，当标定资源只能覆盖少数连续参考轨迹时，完整但未获得数值 SOH 标签的充电窗口仍保留有序结构信息；PI-MSCL 检验多尺度表征与软单调先验能否将该信息转化为更好的点预测，或在数值锚点不足时转化为更可信的退化轨迹。

## 3. 术语账本

| Canonical term | 首次定义 | 不再使用或需限定的表述 |
|---|---|---|
| trajectory-prefix-concentrated SOH-label protocol | 标签优先集中于尽可能少的训练电池，并在每块被选电池中保留时间上连续的早期 SOH 标签，直至耗尽全局标签预算 | partial lifecycle supervision（除非限定定义） |
| distributed-random SOH-label protocol | 每个训练电池内随机保留同等比例的循环级 SOH 标签 | real-world missing labels |
| SOH-label budget | 可进入数值监督损失的训练循环级 SOH 标签总量或比例 | missing cycle data |
| reference trajectory | 具有连续早期 SOH 标定标签的训练电池轨迹 | fully observed deployment battery |
| soft monotonicity prior | 对同电池有序预测对施加的结构正则项 | electrochemical physics model |

## 4. 方法与数据协议

### 4.1 共同控制条件

- HUST 的 77 块 LFP/graphite 电池先按电池级别划分为 46/15/16 个训练、验证和测试电池（主划分 seed 42）。
- 特征标准化只在训练电池上拟合；40 周期窗口绝不跨电池边界。
- 验证和测试电池始终完整标注，仅用于模型选择和最终未见电池评估。
- 两种协议下，所有训练样本都保留完整充电特征、循环顺序和电池标识；未保留 SOH 标签的样本只是不进入 masked MSE。

### 4.2 主协议：trajectory-prefix-concentrated

对每个重复的标签种子，随机确定训练电池的顺序；按该顺序从每块电池最早的可用循环开始连续保留 SOH 标签，直到达到全局标签预算。被完整耗尽的电池构成参考轨迹；最后一块被选电池可能只保留早期前缀；剩余训练电池全部无数值 SOH 标签。这样在固定标签总量下显式形成标签域与未标记域的时间/电池覆盖差异。

主预算为 30%、10% 和 5%。其中 5% 是核心的极低标定资源场景；30% 用于观察标签域扩大后结构先验的作用是否变化。所有正式结果报告三组配对训练/标签种子上的 mean ± sample SD。

### 4.3 机制对照：distributed-random

在每块训练电池内随机保留相同的标签比例。该对照保留每块电池的全寿命标签覆盖机会，因而可将结果变化主要归因于标签数量，而不是标签集中和时间前缀偏移。

## 5. 方法叙事职责

1. **多尺度 CNN**：从完整充电窗口同时提取短期局部响应、中期阶段变化和较慢退化相关变化，避免单一感受野的固定折衷。
2. **LSTM**：将多尺度窗口特征压缩为 SOH 回归所需的时序状态。
3. **Masked data fitting**：少量真实 SOH 标签提供数值锚点，且不把模型生成值伪装为真实标签。
4. **Soft monotonicity prior**：在同一电池内利用有序窗口，抑制后期不合理向上跃迁；所有特征窗口都可参与该结构项，即使其 SOH 标签未被保留。
5. **研究问题**：该结构项不被预设为必然提高 MAE，而是被检验为在轨迹集中、数值锚点稀缺时能否改善点估计或轨迹一致性。

## 6. 建议的正文架构

### Introduction

1. 容量标定的成本与 SOH 估计的重要性。
2. 现实的标签稀缺通常是参考电池轨迹不足，而非每个循环独立随机漏标；不同低标签文献的监督单位、时间覆盖和电池隔离不可直接互换。
3. 因此存在双重难题：跨电池差异，以及少数早期参考轨迹与大量未标记窗口之间的覆盖偏移。
4. 本文以 PI-MSCL 研究完整未标记充电窗口的有序结构信息是否可补充有限数值锚点；同时以随机标签对照分离标签数量效应。

### Methodology

1. Task formulation and two label-availability protocols.
2. PI-MSCL overview.
3. Multi-scale representation and temporal state modelling.
4. Masked regression and soft monotonicity prior.
5. Protocol boundary: no claim of online BMS calibration, prefix extrapolation at inference, or universal chemistry transfer.

### Experimental setup

1. Dataset, cell-level split, features, preprocessing.
2. **Label-coverage protocol figure**: identical cell split and feature availability; panel (a) trajectory-concentrated prefixes; panel (b) distributed random labels; validation/test fully labelled.
3. Shared hyperparameters, paired seeds and reporting unit.
4. Metrics: MAE/RMSE/R2 plus monotonicity violation, upward excess and roughness.

### Results and analysis

1. Main result: trajectory-concentrated label budgets (5%, 10%, 30%).
2. Same-architecture contrast: MS-CNN-LSTM vs PI-MSCL, separating the soft-prior effect.
3. Full 2x2 factorial extension: single/multi-scale × without/with PI, if the initial contrast indicates a meaningful budget-dependent effect.
4. Cross-protocol comparison: trajectory-concentrated versus distributed-random labels at matched budgets.
5. Representative reference/target trajectory analysis and failure modes.
6. Expanded baseline comparison only after the main protocol result is stable.

### Discussion and conclusion

The main interpretation is conditional: structural information from complete unlabelled trajectories may be useful when calibration labels are concentrated in a small number of reference trajectories, but its value must be separated into point-accuracy and trajectory-consistency outcomes. Claims remain limited to the HUST LFP dataset, one primary cell split, three paired repetitions and the two explicit label protocols.

## 7. Evidence ladder and decision rules

| Claim | Required evidence | Interpretation rule |
|---|---|---|
| PI helps under concentrated calibration | PI-MSCL vs same-architecture MS-CNN-LSTM, paired seeds at 5/10/30% | Claim point-accuracy benefit only if paired MAE is directionally stable and mean improvement is material |
| PI improves trajectory consistency | Three trajectory metrics under same comparison | Can be claimed even if MAE does not improve, but must be framed as a trade-off |
| Multi-scale representation contributes | 2x2 extension under the main protocol | Do not infer from cross-architecture ranking alone |
| Protocol changes method behaviour | Matched-budget cross-protocol comparison | Treat as evidence of protocol sensitivity, not broad deployment generalisation |
| Generalisation beyond HUST-LFP | Independent chemistry, operating condition or split evidence | Absent evidence, state as future work only |

## 8. Title direction

Do not finalize the title until the new primary protocol results are complete. The intended direction is:

`Cross-Cell SOH Estimation From Limited Reference Trajectories With a Physics-Informed Multi-Scale CNN-LSTM`

If the new experiments show only a trajectory-consistency gain, use a more cautious title that avoids an accuracy claim, for example:

`Structured SOH Estimation From Limited Reference Trajectories: A Multi-Scale CNN-LSTM Study`
