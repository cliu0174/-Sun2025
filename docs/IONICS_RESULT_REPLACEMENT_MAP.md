# Ionics 主稿：正式结果替换清单

## Canonical manuscript

正式投稿正文以 `overleaf_submission_20260813/main.tex` 为准。该文件中所有 `\pval{...}` 红色内容均为草拟占位，严禁直接用于投稿。

## 替换顺序

| 正文位置 | 需要替换的正式输出 | 叙事判定 |
|---|---|---|
| Abstract | 10% 和 5% 主结果的相对 MAE 改善 | 仅在 PI-MSCL 对匹配 MS--CNN--LSTM 有稳定优势时保留“reduced MAE”句式 |
| Table 2 | 每个 ratio 的实际 labelled windows、完整/部分 reference cells、标签生命周期覆盖 | 从 trajectory-prefix mask metadata 汇总 |
| Figure 3 + Table 5 | HUST 真值轨迹的 post-activation 相邻差分分布 | 若大幅上升比例较高，降低单调先验的措辞强度 |
| Table 4 | 每个 ratio 的正式协议诊断统计 | 必须按 allocation seed 给出均值和离散度 |
| Table 6 + Figure 4 | 30%/10%/5% 主性能矩阵和预算曲线 | 主准确度证据；替换后同步改写 4.2 两段解释 |
| Table 7 | 10% 同协议基线对比 | 所有方法必须使用同一 cell split、window、feature 和 label allocation |
| Table 8 + Figure 6 | 2x2 因子消融 | 若模块交互不稳定，正文应报告而不是宣称可加性 |
| Table 9 + Figure 5 | 5% 轨迹指标和代表电池 | 代表电池应按预注册 median-repetition / median-cell 规则选取 |
| Section 4.6 | allocation-seed 与 $\lambda_{\mathrm{mono}}$ 敏感性 | 用于检验结论是否依赖特定参考轨迹或先验权重 |
| Conclusion | 全部红色数值删除；按真实 evidence ladder 收束主结论 | 若主 MAE 不稳定，主结论降为 trajectory-consistency benefit |

## 图替换要求

`Figure1.png` 必须增加顺序抽取的集中标签分配示意；`Figure3.png` 必须替换为软先验真值诊断；`Figure4.png` 必须替换为不同标签比例下的预测与性能；`Figure5.png` 和 `Figure6.png` 必须用正式 run 的预测与消融输出重绘。每张图的 caption 已在主稿中预置其应表达的唯一结论。

## 投稿前硬检查

1. `main.tex` 中不得残留 `\pval{`、`[TBD]`、`Draft status`、`placeholder` 或红色草稿说明。
2. Abstract、Table 6、Table 7 和 Conclusion 的数字与正式结果归档逐项一致。
3. 每个比较模型均遵循相同的 trajectory-concentrated allocation 与 battery-level split。
4. 使用真实图片后重新执行 `pdflatex -> bibtex -> pdflatex -> pdflatex` 并人工查看 PDF。
