# 2026-08-07 执行更新：架构图、插图规范与正式实验

## 1. 架构图视觉参考与复现边界

本轮以 Xu 等发表于 *Energy* 的 CNN--LSTM--Skip 论文为主要视觉参考：

- H. Xu et al., “An improved CNN-LSTM model-based state-of-health estimation approach for lithium-ion batteries,” *Energy* 276 (2023) 127585, DOI: `10.1016/j.energy.2023.127585`。
- 选择其 Fig. 3 的原因：输入张量、卷积特征图、时间展开 LSTM 和回归输出之间的结构对应最直接，适合表达本文模型。
- 仅借鉴“张量切片—特征图堆叠—显式时间展开—输出头”的视觉语法；不复制其跳连、三层 LSTM、标签、配色或实验内容。

同时参考两篇 *Nature Communications* 电池建模论文的留白、浅色分区和曲线嵌入方式：

- J. Lu et al., DOI: `10.1038/s41467-023-38458-w`。
- F. Wang et al., DOI: `10.1038/s41467-024-48779-z`。

官方参考图仅保存在 `output/reference_figures/` 用于内部视觉核对，不进入正文和投稿材料。

## 2. PI-MSCL 架构图 v4

已生成并嵌入 Springer 活动主稿：

- 可编辑源：`docs/PI-MSCL_architecture_v4.drawio`
- 预览：`docs/PI-MSCL_architecture_v4_preview.png`
- 正文矢量图：`docs/PI-MSCL_architecture_v4_preview.pdf`
- 二次编辑：`docs/PI-MSCL_architecture_v4_preview.svg`
- 位图归档：`docs/PI-MSCL_architecture_v4_preview.tiff`

结构修正：

- 输入由通用 `B × 40 × F` 明确为 `B × 40 × 16`。
- 三尺度 Conv1D 分支明确为 `k=3/7/15`、每支两层、64 通道。
- LSTM 改为两行显式时间展开，清楚区分“时间单元”和“网络层”；对应正文的两层 LSTM、隐藏维数 64。
- 稀疏监督与物理损失压缩为底部两个辅助面板，主模型成为视觉中心。
- XML 可解析；含 79 个可编辑顶点、36 条连接边。PNG 为 5860×3374 px、600 dpi。

当前机器未发现 Draw.io desktop CLI，因此已完成 XML 解析、Matplotlib 原尺寸预览和矢量/位图导出检查；最终投稿前仍建议用 Draw.io 桌面端打开一次，核对字体回退。

## 3. 正文插图统一规范（已落实到脚本）

- 坐标轴标签：13--13.5 pt、粗体。
- 刻度文字：11--11.5 pt、半粗体。
- 图例：10.5--11 pt；条目较少时使用竖向单列。
- 删除所有定量子图内部标题；只保留 `(a)`、`(b)` panel label。
- 每一组图与每一个子图都从同一底层数据独立绘制，不从组图裁剪。
- 默认同时导出 PNG、TIFF、PDF、SVG；PNG/TIFF 均为 600 dpi。
- 相关测试已更新并通过：`4 passed`。

适用脚本：

- `scripts/paper_plot_style.py`
- `scripts/plot_paper_main_results.py`
- `scripts/plot_paper_trajectories.py`
- `scripts/plot_legacy_cn_results.py`

## 4. Springer 编译状态

`latex/springer-sn-2024/manuscript.tex` 已指向 v4 PDF。MiKTeX 当前仍是未完成初始化的新安装，但 Tectonic 已自动补齐缺失缓存并完成 LaTeX--BibTeX 全流程。新 PDF 为 A4 单栏 18 页，42 个活动引用均可解析；未发现 undefined citation、undefined reference 或 overfull box。现有警告主要是 Springer 模板分页产生的 underfull box。

已把第 1--6 页渲染为 PNG 做视觉核对。Fig. 1 位于第 6 页：图形未被裁切，矢量文字清晰，三尺度分支、两层 LSTM、稀疏监督和双损失路径在单栏宽度下仍可区分。20 页目标仍须依靠正式结果表、Fig. 3--5 和相应讨论扩展实现，不能通过人为留白完成。

## 5. 正式实验恢复

用户已明确“开始执行未执行的任务”，因此固定划分正式主矩阵已从断点恢复：

```powershell
D:\Users\46139\anaconda3\python.exe experiments\run_paper_main_results.py `
  --protocol fixed_split --split-seed 42 `
  --training-seeds 929 2262 7 `
  --mask-seeds 1929 3262 1007 `
  --device cuda
```

运行器会跳过已经完成且元数据一致的 4 个目录；剩余 44 次逐目录归档。禁止事后挑选 seed 或删除不理想结果。完整矩阵结束前，Fig. 3--5 和 Results 4.1--4.4 仍不得写成最终统计结论。

## 6. 新增写作与证据约束（用户 2026-08-07 确认）

- 避免内容过少的独立小节。若缺少独立论证任务和证据，优先与相邻小节合并；图表只能用于承载证据，不用于机械扩页。
- 英文稿必须与中文版事实保持一致；后续明确确认的新口径若替代中文旧口径，需同步记录并最终回写中文版。
- 每幅新图及其插入正文后的页面都要进行视觉复检；出现字体过小、留白失衡、图例拥挤、裁切或辨识困难时立即重画。
- 不利实验结果不得删除。所有负结果和处理决定写入 `docs/NEGATIVE_RESULTS_AND_DECISIONS.md`，最终统一向用户报告。
- 可自主评估新增实验，但纳入条件是科学问题明确、同尺度公平、无测试集调参、完整归档并能够解释其对主线的贡献。
- 经典与先进算法的比较分为两层：可复现模型在相同 HUST 协议下重跑；不同协议的已发表工作只作研究背景，禁止直接数值排名。
- 英文表达、作者信息、浮动体间距和声明格式参考已人工润色的 Springer/Ionics 论文。已同步 Chang Liu、Congyan Chen、Southeast University、基金号及作者贡献；数据可用性仍按本研究的 HUST 公共数据事实单独撰写。

## 7. 结构审计与新增基线方向

- 当前活动 Results 仍是正式结果占位文本；旧单次表格和异常风险段落已处于 `\iffalse`，不会用于凑页数。
- 已建立 `docs/SECTION_DEPTH_AND_CONSISTENCY_AUDIT.md`，记录小节合并方案、术语账本和中英文冲突项。
- 扩展同尺度候选为 XGBoost、GRU/LSTM 和 attention CNN-LSTM，优先比较 $r=1.0$ 与 $r=0.3$ 的三次配对均值；最终是否运行取决于代码审计能否保证与主矩阵完全相同的数据接口。
- 已发表方法背景重点覆盖弱标签/自监督、无目标标签迁移、physics-informed SOH 和近期 HUST CNN-LSTM 工作；不同特征与划分的文献数值不进入本文主排名。

## 8. 同尺度基线管线检查

- 新增 `experiments/run_paper_baselines.py`，预注册 XGBoost、LSTM、GRU 和 attention CNN--LSTM 在 $r=1.0$ 与 $r=0.3$ 两个端点的三次配对重复。
- XGBoost 的单次 `smoke=true` 检查已通过：固定划分、16 维输入、40-cycle 窗口、循环级随机掩码和测试指标均与主实验一致；仅保留标签窗口参与拟合。
- 该 smoke 结果不会进入正式聚合；完整数值与不利结果处理统一见 `docs/NEGATIVE_RESULTS_AND_DECISIONS.md`。
- 主矩阵仍在 GPU 上串行运行。为避免改变其训练环境，神经网络扩展基线在主矩阵结束前不并发启动。
- 神经网络基线开跑前的配置审计发现 GRU 原配置默认关闭学习率调度，而 LSTM/attention 默认启用。现已在公共覆盖配置中显式统一为与主矩阵相同的 20-epoch warmup 与 cosine decay，避免把优化策略差异误当成架构差异；该修正发生在任何正式神经基线运行之前。

## 9. 架构图 v5 与正文实页检查

- 根据用户对 v4“文字框排列感过强”的反馈，新建 `scripts/generate_architecture_drawio_v5.py`，不覆盖旧源文件。
- v5 采用两面板结构：主面板以运行曲线、三色特征图薄片、双层时间展开 LSTM 和 SOH 曲线形成视觉主路径；底部面板只保留 masked MSE 与 soft monotonicity 两条证据路径。
- 可编辑源为 `docs/PI-MSCL_architecture_v5.drawio`；同步生成 PDF、SVG、600 dpi PNG 和 600 dpi TIFF。位图尺寸为 5860×3374 px，DPI 元数据已核验。
- XML 解析通过；Draw.io validator 为 0 errors、0 edge crossings。检查器的 overlap 警告来自有意叠放的特征图薄片、卡片内部标签和 LSTM 门控提示点。
- 活动正文已切换到 v5，并同步把三面板说明改为两面板证据链。Tectonic/BibTeX 编译成功，主稿为 20 页；Fig. 1 第 6 页实页渲染未发现裁切或关键标签不可辨认。
- 视觉参考、复现边界和图形取舍单独记录在 `docs/ARCHITECTURE_FIGURE_DESIGN_RATIONALE.md`。

## 10. 正式矩阵断点续跑状态

- 归档复核口径为 `fixed_split/v3_leakage_free` 下非 smoke 的 `result.json + predictions.npz` 成对目录。
- 当前计数为 14/48 个正式组合，无 `error.json`；此前报告的 10/48 只计入了本日新增结果，已更正为包含昨日四个同协议配对变体的总数。
- 断点续跑进程使用同一命令、split seed 42、三组 train/mask seeds 和 CUDA 设备；已完成目录按元数据校验后跳过，不执行覆盖。

## 11. 架构图终尺寸修正与表格版式预检

- Fig. 1 左下角两条说明在独立预览中出现视觉粘连，现已改为上下错层并重新导出全部格式；活动主稿重新编译仍为 20 页，第 6 页实页复核通过。
- 正式主矩阵当前非 smoke 计数为 16/48；另有一个 `smoke=true` 的多尺度 PI 目录，仅用于早期管线检查，严格排除于正式计数与后续聚合。
- 同协议 8 模型全监督表和标签预算端点表已用合成数据完成单栏版式预检。两表保留 mean ± SD，不缩放字号；合成数据只用于布局，不进入正文。
- 正式表格生成器已统一使用可在 Springer 8 pt 表格字体中稳定渲染的文本正负号，并以纯文本 `R-squared`/`Change (pp)` 表头避免不必要的小号数学字体和 Unicode 编译依赖；相关回归测试共 6 项通过。

## 12. Results 结构与报告完整性修正

- 正文删除未被正式 `result.json` 归档的 `mean_upward_excess` 报告承诺，只保留三个完整轨迹指标，并补写 $D_2$ 的明确公式；偏差登记为 D-007。
- 新增各标签预算下的 R-squared 表生成器。五张主结果表和两张补充基线表均已用合成值完成 Springer 单栏预排版；无缩放、无 overfull box，合成值不进入活动正文。
- 消融效应图已改为基于共享 split/mask/train seeds 的逐配对 MAE 差值，显示 mean ± sample SD；正值明确表示右侧变体降低 MAE。
- Results 合并过薄的全监督小节，并新增有两张表承载的同协议补充基线小节。最新主稿 21 页，第 12--16 页实页检查通过。

## 13. 参考文献真实性审计与流水线回归测试

- `references.bib` 共 43 条，全部带 DOI；活动主稿编译出的参考文献为 42 条，其中 Introduction 在 Methodology 之前使用 40 个唯一引用键，严格满足“引言达到 40 篇”的口径；不存在引用键缺失，唯一未引用条目为 `zhang2023deep`，不为凑数量强行插入。
- 42 篇期刊论文 DOI 均由 Crossref REST API 返回正式 `journal-article` 元数据，题名与本地 BibTeX 相符；HUST 数据集 DOI `10.17632/nsc7hnsg4s.2` 由 DataCite 验证为 2022 年 Mendeley Dataset。完整证据见 `docs/REFERENCE_AUDIT_20260807.md`。
- 结果聚合、同协议基线表、主结果绘图、轨迹绘图、架构图、无泄漏归档、电池边界窗口、标准化范围和轨迹指标相关测试共 **26 项通过**。
- 固定划分主矩阵当前正式归档为 **23/48**，23 个结果均有 `predictions.npz`，无 `error.json`；后台 CUDA 进程继续运行且未重复启动。

## 14. 单尺度半矩阵冻结与最终验收表

- 主矩阵达到 **26/48**；其中 single-scale CNN--LSTM 与 PI-CNN--LSTM 的 4 个标签比例 x 3 次配对重复（共 24 次）已经完整，另外两个预存的多尺度低预算正式运行计入总数。全部正式结果均有预测归档且无 `error.json`。
- 单尺度 PI 的平均 MAE 相对 no-PI 在 $r=0.7$、$r=0.5$ 分别改善 0.0581、0.0809 个百分点，在 $r=1.0$、$r=0.3$ 分别恶化 0.0304、0.0185 个百分点；不同种子和轨迹指标方向并不完全一致，已更新 `docs/NEGATIVE_RESULTS_AND_DECISIONS.md`，不提前外推为多尺度结论。
- 后台进程已自动转入 `multi_no_pi/r1p0`，未修改超参数、未覆盖归档、未并行启动神经基线。
- 新建 `docs/GOAL_COMPLETION_AUDIT.md`，逐项记录中文依据、备份、正式矩阵、基线、600 dpi 组图/子图、Draw.io、40 篇引言文献、SCI 写作、简洁图注、20 页和待确认信息的证据与剩余门槛。

## 15. LaTeX 占位符与 legacy anomaly 证据边界复核

- 源文件中检索到 3 个 `Placeholder`，但逐一核对条件编译边界后确认：第 339--466 行的旧 Results 和第 471--550 行的旧 anomaly-risk extension 分别完整包在 `\iffalse...\fi` 中，均不进入活动 PDF。当前编译稿没有可见占位框。
- 旧异常扩展的本地资产已复核。`docs/battery_anomaly_report.docx` 只有场景、信号、指标和流程说明性表格，明确使用 14 维特征，没有逐运行结果或 checkpoint 归档；现有 Markdown 也仅保留服务器汇总。因此继续保持该节不编译，不能恢复为当前 16 维 `v3_leakage_free` 的正式证据。
- 先前把这两个异常图框称为“活动正文占位符”的状态判断已纠正；D-008 和 `docs/GOAL_COMPLETION_AUDIT.md` 已同步改为“未启用 legacy extension”。

## 16. 多尺度无物理约束单元的阶段性配对审计

- 正式主矩阵已达到 **32/48**；32 个非 smoke 目录均同时包含 `result.json` 与 `predictions.npz`，无 `error.json`。后台运行器继续进入 `multi_no_pi/r0p5`，未重启、未覆盖、未并发启动扩展基线。
- `multi_no_pi/r1p0` 三重复已完整：相对配对单尺度 no-PI，MAE 平均改善 0.0664 个百分点，但 RMSE 平均恶化 0.0999 个百分点，$R^2$ 平均下降 0.0119。该混合结果登记为 N-004。
- `multi_no_pi/r0p7` 三重复已完整：MAE 平均恶化 0.0038 个百分点，RMSE 平均恶化 0.1986 个百分点，$R^2$ 平均下降 0.0223；三个 MAE 配对方向为两次改善、一次明显恶化，显示较强种子敏感性。该结果登记为 N-005。
- 以上数字只用于完整预算单元的内部审计；在 48-run 矩阵完成前，活动 Results 仍不填入总体模型排序或 PI-MSCL 总结性主张。最终汇总必须同时报告 MAE、RMSE、$R^2$、轨迹指标及样本标准差，禁止只选有利指标。

## 17. PI-MSCL 在全监督与 70% 标签预算下的配对效应

- 正式主矩阵达到 **43/48**；43 个非 smoke 结果均有预测归档且无错误。`multi_pi/r1p0` 与 `multi_pi/r0p7` 两个三重复预算单元已经冻结，运行器继续进入 `multi_pi/r0p5`。
- 全监督时，PI 相对 multi-scale no-PI 的平均 MAE 改善 0.0262 个百分点、RMSE 改善 0.0643 个百分点、$R^2$ 提升 0.00794；但 MAE 仅 1/3 配对改善，平均收益由训练种子 2262 主导，单调违背率均值略增 0.0137 个百分点。该不稳定性已登记为 N-008。
- 在 $r=0.7$ 时，PI 平均改善 MAE 0.0452 个百分点、RMSE 0.0663 个百分点、$R^2$ 0.00800。MAE 与 RMSE 均为 2/3 配对改善，但三项轨迹指标的三个配对全部改善：单调违背率平均下降 0.2661 个百分点，累计上升超差平均下降 0.01256，平均绝对二阶差分平均下降 0.000205。
- 当前证据支持把 $r=0.7$ 视为“软单调结构监督开始呈现可重复轨迹收益”的核心候选单元，但不能在 $r=0.5$ 与 $r=0.3$ 完成前声称物理约束随标签预算降低而系统增强。

## 18. 正式主矩阵完成、严格聚合与投稿图表生成

- `fixed_split/v3_leakage_free` 正式主矩阵已完成 **48/48** 次非 smoke 运行；16 个模型×标签预算单元均包含 3 组预注册配对种子，全部同时具有 `result.json` 与 `predictions.npz`，无 `error.json`。
- 严格校验结果为 `RUNS=48`、`GROUPS=16`、`ISSUES=0`、`MISSING_PRED=0`。正式聚合文件为 `experiments/paper_main_results/fixed_split/v3_leakage_free/summary.json`。
- 已生成 6 张正式 LaTeX 表格和 3 组正式主图。所有定量组图与独立子图均由 Python 从同一底层归档独立绘制，不使用截图裁切；PNG/TIFF 为 600 dpi，PDF/SVG 保留矢量文字。
- Fig. 4 的代表性轨迹按预注册式中位配对规则自动选择训练/mask 种子 2262/3262 和电池 `10-6`，选择记录保存在 `output/figures/formal_main_v3/fig4_source.json`，不进行人工挑图。
- $r=0.3$ 的最终 PI-MSCL 配对结果已登记为 N-010：轨迹一致性三项指标均改善，但 MAE/RMSE/$R^2$ 的平均收益不稳定，因此正文必须按权衡而非全面优势叙述。

## 19. 同协议补充基线闭环、终稿改写与实页验收（2026-08-08）

- 补充端点矩阵完成 **24/24** 个非 smoke 运行：XGBoost、LSTM、GRU、attention CNN--LSTM × $r=1.0/0.3$ × 3 组配对种子。每个模型/比例单元恰有 3 次正式重复，0 个正式 `error.json`，0 个缺失 `predictions.npz`。
- 汇总文件含 25 条归档，其中 1 条为 `smoke=true` 的 XGBoost 管线检查；正式校验和表格生成器均显式排除该记录。启动期 stdout 故障保留于 `_diagnostics`，不作为模型失败或正式结果。
- 已生成并插入三张八模型表：全监督 MAE/RMSE/$R^2$、$r=1.0/0.3$ 配对 MAE 端点变化、30% 标签下统一轨迹指标。原四模型全监督表和轨迹表从活动正文移除，避免重复与选择性证据。
- 主要不利发现已登记为 N-011：PI-MSCL 仅在全监督 MAE 上居首；GRU 的全监督 RMSE/$R^2$ 最佳，attention CNN--LSTM 的 30% 标签 MAE 最低，LSTM/GRU 分别领先不同轨迹指标。Abstract、Results、Discussion、Conclusion 已同步收紧主张。
- 回归测试 **26/26 通过**。Springer 主稿编译为 **25 页**；日志无 overfull、未定义引用、浮动体过大或 LaTeX error。第 6、13--20 页已按 150 dpi 实页渲染复核，架构图、八模型表、结果图和结论均无裁切或不可辨文字。
- Reviewer Round 15 已完成三审稿人交叉自检。当前剩余非技术待确认项仅包括：中文版是否同步改写早期自建数据/14 维/旧结果口径、legacy anomaly-risk 是否在 Discussion 简短提及、代码仓库公开地址。

## 20. 最终交付资产与中英文一致性审计（2026-08-08）

- 正式 Fig. 3--5 均包含组图和两个独立子图，每一图均有 PNG/TIFF/PDF/SVG。逐文件读取元数据确认：所有 PNG 为 599.9988 dpi，所有 TIFF 为 600.0000 dpi；独立子图具有各自画布和矢量文件，不是组图裁切。
- 备份目录、活动 TeX/PDF、可编辑 Draw.io 和架构图四种导出格式均存在。活动正文只引用 v5 架构 PDF、三张正式组图和六张由严格汇总生成的表格。
- 去除 `\iffalse...\fi` 非活动内容后扫描主稿：无 Placeholder、TODO、TBC、自建数据、14 维、旧四模型表或 legacy anomaly-risk 引用。唯一未来时占位为 Code Availability 的仓库声明，已列为作者决定。
- 对中文版 201 个段落和 7 张表完成最终口径审计。冲突不仅包括自建数据和 14 维，还包括旧结果支持的“PI-MSCL 在低标签全面领先”“物理收益随标签减少扩大”以及不可复核异常实验。逐段位置、正式替代口径和建议章节结构已写入 `docs/CHINESE_SYNC_PROPOSAL_20260808.md`，原 DOCX 未修改。

## 21. Round 16 终稿主张校准与 24 页视觉复核（2026-08-08）

- 删除正文中没有正式结果支持的“多个独立电池划分敏感性分析”陈述；活动稿现在明确所有效应对固定 split seed 42 有条件成立，独立划分只作为未来补充验证。
- 将“independent effects”统一收紧为配对边际效应与架构—物理交互；删除暗示统计显著性或强复现性的 `detectably` / `reproducible` 措辞，改为均值、逐配对方向和离散性的可核查描述。
- 补写四类同协议基线的关键网络/XGBoost 配置，并明确共同训练协议不等价于架构特定最优调参；训练成本、推理延迟和能耗未作比较。
- Introduction 仍使用 40 个唯一引用键；自动回归测试 26/26 通过。Tectonic 重编译无 overfull、未定义引用、float-too-large 或 LaTeX error。
- 润色后 Springer 单栏 PDF 为 **24 页**。第 1、6、13--20 页已重新以 150 dpi 渲染并逐页检查；标题摘要、Fig. 1--4、Tables 4--9、Discussion、Conclusion 和 Declarations 均无裁切、越界、孤立标题或不可辨文字。
- Reviewer Round 16 已按技术可靠性、原创性/重要性、跨领域可读性三种视角完成，并记录一份交叉综合和不支持主张清单。新增偏差记录 D-012。
## 22. Results-and-analysis restructuring, new overall figure, and Round 17 QA (2026-08-08)

- Section 3 now contains datasets/SOH definition, feature preprocessing, a combined protocol-and-hyperparameter subsection with a formal hyperparameter table, loss functions, and evaluation/statistical reporting. The former compared-model subsection was removed.
- Section 4 was renamed **Experimental Results and Analysis** and reorganized into five evidence-bearing subsections: overall cross-cell prediction/error characteristics; label-budget sensitivity plus representative-cell behavior; same-protocol endpoint comparison; trajectory/physical-effect analysis; and factorial ablation/failure modes/scope.
- A new formal Fig. 2 was generated from existing non-smoke archives: pooled full-supervision agreement plus per-cell MAE distribution. Group and child panels were rendered independently as PNG/TIFF/PDF/SVG; no crop-derived child panel was used.
- The standalone Discussion section was removed. All retained interpretations now appear adjacent to the corresponding evidence in Section 4. Conclusion is Section 5 and contains exactly four paragraphs; the fourth combines limitations with proportionate future work.
- A new dated backup was created at `backups/manuscript_before_structure_revision_20260808/`.
- The manuscript compiled to 23 Springer single-column pages. All pages were rendered at 150 dpi and inspected; no clipping, overlap, illegible text, isolated heading, or malformed float was found. The log contains no overfull box, undefined citation/reference, float-too-large, or LaTeX error.
- Structural audit passed and regression tests passed 26/26. Round 17 reviewer assessment and N-012/D-013 were recorded.
## 23. Unified table and figure presentation rules (2026-08-08)

- Active manuscript tables were converted to full available text width with `tabular*`, preserving readable type rather than scaling table text down.
- The shared plotting style now suppresses top and right spines, retains common bold axis typography, and uses consistent 600-dpi PNG/TIFF plus editable PDF/SVG exports.
- Figure legends were moved from external canvas positions into deliberately reserved, white-backed corners of their associated panels. The label-budget and factorial panels reserve data-free space for this purpose.
- A new formal data-overview figure was added to Section 3.1. It plots all 77 HUST per-cycle discharge-capacity trajectories after initial-capacity normalization. It does not imply access to raw voltage-time discharge traces and is explicitly separated from the label-mask experiment.
- The post-edit Springer PDF remains 23 pages and was visually inspected at 150 dpi across the changed figure/table pages.
