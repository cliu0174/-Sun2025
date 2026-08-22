# PI-MSCL 架构图设计备忘（v5）

## 图要证明的核心结论

在每块训练电池仅保留固定比例循环级 SOH 标签的条件下，PI-MSCL 仍利用完整的 40-cycle × 16-feature 运行轨迹；三尺度 Conv1D 提取不同时间感受野的退化表征，两层 LSTM 建模跨循环演化，未标注循环通过预测轨迹参与软单调物理约束。

## 视觉参考与取舍

- Lu et al., *Nature Communications* 14, 3797 (2023), DOI: 10.1038/s41467-023-38458-w：参考其彩色特征图薄片、信号小曲线和从输入到预测的紧凑视觉语法。
- Wang et al., *Nature Communications* 15, 4332 (2024), DOI: 10.1038/s41467-024-48779-z：参考其分区清晰、用小型曲线和网络图直接承载方法逻辑的版式。
- Zhao et al., *Scientific Reports* 14, 29026 (2024), DOI: 10.1038/s41598-024-80421-2：仅参考一维卷积核在时间信号上滑动的表达方式。
- Dhairya et al., *Scientific Reports* (2025), DOI: 10.1038/s41598-025-28091-6：仅参考堆叠 LSTM 的层/时间步区分方法。

图未复制任何一篇文献的完整布局或具体图形；仅组合了成熟的视觉语法，网络结构、参数和监督机制均来自当前正文与实现。

## v5 结构

1. Panel (a)：完整运行轨迹与稀疏标签 → 40 × 16 输入张量 → k=3/7/15 三条 Conv1D 分支（每支两层、64 channels）→ concat + 1 × 1 Conv（128 channels）→ 两层 LSTM（hidden size 64）→ FC 64 + ReLU + dropout 0.4 + sigmoid → SOH 曲线。
2. Panel (b)：保留标签进入 masked MSE；满足同电池、`c≥cmin` 与间隔 `≤K` 的预测对进入 soft monotonicity，其中可以包含无标签循环；二者以 `L = Ldata + 0.3 Lmono` 合并，并紧凑标注 `ε=.005, cmin=300, K=40, α=.2`。

## 刻意避免的表达

- 不再用大量文字框逐层罗列网络，避免“把正文搬进图里”。
- 不画完整 LSTM 门控公式，避免信息过载；两行表示层数，三列表示时间步，彩色门控点仅作为视觉提示。
- 不把三条 Conv1D 分支画成三套重复说明；参数采用一次共享注释，分支只突出不同 kernel size。
- 不把随机标签掩码画成真实 BMS 间歇标定；图中明确为 controlled/fixed cycle-level label budget。

## 导出与自检

- 可编辑源文件：`docs/PI-MSCL_architecture_v5.drawio`。
- 导出：SVG、PDF、PNG、TIFF；位图为 600 dpi。
- XML 解析通过；Draw.io 结构检查为 0 errors、0 edge crossings。检查器报告的主要 overlap 来自有意叠放的特征图薄片、容器内标签和 LSTM 门控点。
- 视觉自检已完成两轮：第二轮修正了标题碰撞、输入卡片越界、分支扇出重叠和过密的 LSTM 门控文字。
