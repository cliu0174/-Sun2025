# 论文轮次自检记录

> 本文件记录每轮实质写作后的审稿人式检查、已执行修正和遗留风险。评语不能替代修改；凡标记为“已修正”的问题均须已落实到正文、图表、代码或备忘。

## Round 1 — 引言主线、参考文献与方法总览图（2026-08-06）

### 本轮改动范围

- 将引言中的“部分生命周期监督”收敛为 HUST 上的受控、电池内随机循环级标签掩码，并明确其不是自然现场缺标的复现。
- 将有限标签文献按少样本、早期标签增强、自监督弱标签、半监督/伪标签等协议区分，避免与本文标签分配方案等同。
- 新增并核验 5 篇有限标签 SOH/容量估计文献；BibTeX 记录由 38 增至 43，引言实际引用 40 个唯一条目。
- 将贡献列表改为“受控协议—PI-MSCL—二维公平比较—轨迹评价”，删除无直接主线支撑的异常风险贡献。
- 将旧的异常退化风险章节用 LaTeX 条件块从主文排除；原文仍保留在源文件及重构前备份中，未删除。
- 将方法总览图升级为可编辑 v3 三联图：稀疏标定任务、三尺度 CNN-LSTM 网络和双目标学习。

### Reviewer 1：问题重要性与叙事边界

**发现的问题**

1. 旧引言把工程上的间歇标定、连续生命周期缺标和本文随机掩码混为同一情形，容易被质疑任务定义与实验协议不一致。
2. 旧贡献包含异常风险指示，但主实验没有可追溯的真实异常数据证据，削弱核心研究问题。
3. “标签有限”相关工作虽然方向相近，但标签减少单位和辅助信息来源差异很大，不能直接作为本文协议的先例。

**已执行修正**

- 正文明确写为在完全标注 HUST 上进行的 controlled intervention，并限定结论为标签预算鲁棒性和跨电池泛化。
- 将异常风险章节排除出主文证据链，正文结构恢复为 4.1–4.4。
- 在相关工作中显式比较不同标签稀缺协议，本文定位为每块训练电池内的 cycle-level label budget。

### Reviewer 2：技术公平性与结论强度

**发现的问题**

1. 旧稿可能依据数据管线不一致的 A1/Exp09c 结果宣称物理约束优势；该比较不能进入最终证据。
2. 软单调约束可能改善轨迹合理性但牺牲部分标签比例下的点误差，不能预设“所有比例、所有指标均优”。
3. 当前正式多种子主结果尚未在修正后的电池边界保持管线上跑完。

**已执行修正**

- 引言贡献只声明将执行二维公平比较和轨迹评价，不提前宣称全面优势。
- 主实验入口已固定统一窗口化与可分离随机种子；结果数值须由正式重跑归档后写入。
- 轨迹合理性与点误差在正文中作为两个互补结果维度报告。

### Reviewer 3：可复现性、图表与引用

**发现的问题**

1. 引言引用数量不足 40，且有限标签方向的直接支撑较少。
2. 旧架构图偏传统流程框，未直观说明“mask labels, retain trajectories”和软单调约束的作用位置。
3. 本机缺少原生 Draw.io 与 LaTeX 编译环境，最终字体替换和版式仍需外部验证。

**已执行修正**

- 当前 BibTeX 共 43 条；引言含 40 个唯一引用键，全文无缺失引用键或重复 BibTeX 键。
- v3 架构图含 54 个可编辑节点和 35 条可编辑连接线，PNG 为 5860 × 3374、600 dpi，并同步输出 PDF/SVG。
- 图中用稀疏标定点、完整运行曲线、三尺度网络单元和局部回升惩罚形成可视证据链；v2 仅保留为简化回退稿。

### 交叉审稿结论与下一轮要求

- 本轮后的主问题、方法设计和实验协议已能一一对应，但不能在正式重跑前锁定摘要与结论中的结果数字。
- 现有 43 条 BibTeX 仍需逐条完成题名、作者、来源、年份和 DOI/稳定页核验；“数量达到 40”不等于引用审计完成。
- 下一轮应重写 Methodology 与 Experimental Setup，使公式、实现和 v3 架构图严格对应；随后检查 HUST 特征维数、数据划分口径和模型规范名三个 TBC。
- 最终排版前必须在原生 Draw.io 打开 v3，并在作者确认的 Springer 单栏模板中编译与检查页数。

## Round 2 — 方法、实验设置与结果结构（2026-08-06）

### 本轮改动范围

- 将旧 2.1 的监督场景构造移入第 3 节；第 2 节只保留问题接口、网络、回归头和损失定义。
- 用 v3 PDF 预览替换方法图占位框，并让正文按图的三个面板逐项解释。
- 依据代码校正网络细节、物理参数、HUST 划分数量、预处理和训练配置。
- 修复跨电池预处理泄漏，并增加窗口级实际标签覆盖率归档。
- 将旧结果数字整体排除出编译主文，重建 4.1 全监督、4.2 标签预算、4.3 轨迹证据、4.4 二因素消融顺序。

### Reviewer 1：方法与实验问题是否分离

**发现的问题**

1. 旧 2.1 同时定义任务、HUST 随机掩码和标签比例，导致方法贡献与实验干预混杂。
2. 架构图占位框无法说明无标签循环如何同时进入特征路径和物理路径。
3. 原 masked-MSE 公式用附加常数近似零标签批次，但实现实际采用分段规则。

**已执行修正**

- 第 2 节仅定义输入窗口、监督指示变量和两条损失路径；随机采样、比例与 seed 全部放入 3.3。
- 正文按 v3 的任务、网络和双目标三个面板说明信息流。
- masked MSE 改为与代码一致的分段表达：有标签时按标签数归一化，无标签批次的数据项为零。

### Reviewer 2：数据泄漏与公平比较

**发现的问题**

1. 旧跨电池加载流程在 split 前对每块电池完整生命周期单独拟合 scaler，使未见电池统计量进入输入。
2. 旧稿写 46/16/15，但整数划分代码实际得到 46/15/16。
3. 旧结果块既来自混杂管线，又把消融放在标签预算主结果之前。

**已执行修正**

- 跨电池入口改为读取原始特征、先按电池划分、再只用训练电池拟合一次 scaler；单电池旧接口保持兼容。
- 正文按实际 46/15/16 报告，并明确窗口不得跨电池。
- 四个主模型由同一 `ms_cnn_lstm_v2` 类实现，唯一开关为尺度结构和物理项。
- 旧数字保留在条件归档块中但不参与编译；新 4.1–4.4 结果壳不含未经重跑的数字。

### Reviewer 3：参数、统计口径与可复现性

**发现的问题**

1. 旧正文将最大物理配对间距写成 20，而正式 runner 为 40；总损失还列出了权重为零的边界和平滑项。
2. 标签比例在原始 cycle level 生成，但 many-to-one 窗口的最初 39 个循环不能成为目标，需同时保存实际窗口级覆盖率。
3. 旧正文只说“多次平均”，没有声明标准差、三类 seed 或代表电池选择规则。

**已执行修正**

- 方法改为 $K=40$，主损失只写 masked MSE 与 $\lambda_{\mathrm{mono}}\mathcal{L}_{\mathrm{mono}}$；禁用项仅在正文说明。
- 每次运行新增 cycle-level 与 window-target-level 标签数和实现比例字段；论文可以如实区分名义预算和实际可训练目标覆盖率。
- 第 3 节明确 split/mask/initialization seed 分离，结果报告 mean $\pm$ sample standard deviation；代表轨迹按测试电池 MAE 中位规则确定。
- 修复后 14 项单元测试、Python 编译检查和 1-epoch CPU smoke 均通过；smoke 元数据被绘图脚本自动拒绝。

### 交叉审稿结论与下一轮要求

- TBC-008 已由作者确认：正文按当前 loader/CSV 的 16 维实现报告，旧中文稿 14 维描述不再沿用。
- TBC-002 已由作者确认：固定 split seed 42、重复 mask/train seeds 为主表口径；独立 battery splits 只作单独敏感性验证，两类结果不得混合求均值。
- 正式 GPU 主矩阵必须在两项公平性修复后重新运行；在此之前摘要与结论不写性能数字。
- 作者已提供 Springer Nature 2024 年 12 月 v3.1 单栏模板，已迁移为 `latex/springer-sn-2024/manuscript.tex` 并设为权威主稿。含参考文献的基线编译为 18 页；正式结果未完成前仍不宣称达到 20 页目标。

## Round 3 — 参考文献元数据与引用边界（2026-08-06）

### Reviewer 1：文献是否真实、可检索

**发现的问题**

1. 43 条 BibTeX 虽已达到数量要求，但多数旧条目没有 DOI，无法在投稿前快速排查错配。
2. HUST 原始论文和数据集条目存在作者姓名错误；两个 IEEE 条目沿用了 early-access 年份而非正式卷期年份。

**已执行修正**

- 43 条记录均已补齐 DOI，并建立 `docs/REFERENCE_AUDIT.md` 逐条记录用途与核验状态。
- 依据 RSC 正式页面将 HUST 论文作者改为 Guijun Ma、Songpei Xu、Benben Jiang、Cheng Cheng、Xin Yang、Yue Shen、Tao Yang、Yunhui Huang、Han Ding、Ye Yuan。
- 依据 Mendeley Data 正式页面将数据集贡献者改为 Ye Yuan、Guijun Ma、Songpei Xu。
- 将循环同步迁移学习论文的正式年改为 2023，将 IEEE T-IV 物理信息 PHM 论文的正式年改为 2024；保留原 BibTeX 键以避免破坏引用。

### Reviewer 2：引用是否支撑对应论点

**发现的问题**

1. “有限标签”文献混合了少数完整电池、早期生命周期、单循环片段、自监督弱标签和半监督伪标签，不能共同充当随机掩码协议的直接先例。
2. 2026 年在线论文可能尚未形成最终卷期，正文不应依赖其搜索摘要作定量结论。

**已执行修正**

- 引言把不同监督单元和辅助信息来源显式拆开；文献只支撑其实际研究设置。
- 本文协议继续限定为“split first, then randomly retain cycle-level labels within each training cell”，并明确是受控标签预算干预。
- 2026 年条目仅在已有正式 DOI 时保留，并在核验表中标记投稿前复查卷期。

### Reviewer 3：下一轮风险

- 文献元数据审计已经完成到 DOI 层面，但最终目标期刊确定后仍需按其参考文献格式统一大小写、期刊缩写和 online-first 处理。
- 引言的 40 个引用键需要在最终编译日志中检查未定义引用；当前静态检查不能替代 BibTeX/Biber 编译。
- 下一轮优先进行 LaTeX 可编译性与版面审计，并继续保持旧结果数字不进入活动正文。

## Round 4 — Springer 模板迁移、编译与逐页版式审计（2026-08-06）

### Reviewer 1：模板与文件权威性

**发现的问题**

1. 先前英文稿仍使用 Elsevier CAS 双栏类，无法据此判断作者要求的 Springer 单栏页数与浮动体行为。
2. 多个英文入口容易导致后续修改分叉。

**已执行修正**

- 采用作者提供的 Springer Nature 2024 年 12 月 v3.1 模板，建立 `latex/springer-sn-2024/`，并将 `manuscript.tex` 设为唯一权威英文主稿。
- 使用模板自带的 `sn-jnl.cls` 和 `sn-mathphys-num.bst`；原 Elsevier 稿只保留作内容追踪，不再承担最终排版。
- 在目录 README 和论文备忘中写明权威入口、构建产物及未完成的作者信息项。

### Reviewer 2：可编译性与引用完整性

**检查结果**

- Tectonic 完成 LaTeX--BibTeX--LaTeX 全流程，生成 A4 单栏 18 页 PDF。
- 43 条 BibTeX 中 42 条在活动正文被引用；最终日志无未定义引用、未定义交叉引用、越界行或书签层级警告。
- 剩余警告为模板分页导致的 underfull box，不对应文字裁切或元素碰撞。

### Reviewer 3：视觉与叙事边界

**检查结果与修正**

- 逐页渲染检查 18 页，未发现裁切、重叠、模糊、破碎公式或不可读表格；v3 架构图在单栏整宽下保持清晰。
- 将声明中的 `\bmhead` 改为无层级跳跃的简短粗体标签，消除 PDF 书签警告并维持 Springer 示例的紧凑声明样式。
- 当前 18 页不通过空白或重复背景扩充。正式 4.1--4.4 结果表、三组独立生成的 600 dpi 图件及证据对应讨论写入后，再复核 20 页目标。
- 摘要、结果和结论继续不写旧实验数字；作者单位、目标期刊、重复口径、特征维数、资助和代码仓库仍作为 TBC 保留。

## Round 5 — 本地结果资产审计与证据分级（2026-08-06）

### Reviewer 1：是否遗漏了中文版已有结果

**发现的问题**

1. 中文版 DOCX 已有完整主表和消融表，若只说“正式结果待跑”会造成旧成果被忽略的误解。
2. 5 月 `experiment_results_summary.md` 还保存了 Exp-09/10/11 的 n=3 聚合，具有明显写作价值。

**已执行修正**

- 逐表提取并登记中文版 Tables 1--6 的关键数值，同时核对了 14 维旧特征表与已确认 16 维实现之间的差异。
- 检查 `docs/` 中全部实验结果快照、旧 `metrics.json`、两份 legacy PKL、现有正式 JSON/PKL/NPZ 和旧绘图脚本。
- 建立 `docs/LOCAL_RESULT_ASSET_AUDIT.md`，将资产分为可直接保留、限定复用、禁止进入定量核心和仍缺失四类。

### Reviewer 2：旧结果能否直接替代重跑

**审查结论**

- 不能直接替代。中文版、5 月汇总和当前 `v3_leakage_free` 的 `r=0.3` PI-MSCL MAE 分别为 1.134%、1.0336% 和 1.7431%，显示三代协议和代码状态不等价。
- 旧管线存在已记录的物理开关/监督掩码问题，且本轮代码审计进一步确认跨电池窗口与按测试电池拟合标准化器的问题。
- 多数 5 月聚合没有在本地保留对应的完整逐种子预测目录，无法复算 SD、配对检验、代表轨迹和正式图。
- 因此旧结果可作为历史、开发和敏感性证据，但不得重命名为当前固定划分无泄漏主结果，也不得与当前结果求均值。

### Reviewer 3：当前新结果是否足够下结论

**审查结论**

- 当前正式结果只完成 `r=0.3`、split 42、mask 1929、train 929 的一个 2×2 配对组，共 4/48 次。
- 单 seed 显示物理约束显著降低轨迹的非物理上升，但点估计 MAE 略有代价；该结论只能作为诊断，不得写成总体统计结论。
- 正式表生成器会拒绝不完整或未配对的矩阵，避免把 `n=1` 误写成 `mean ± std`。

### 交叉审稿结论与交接

- 旧成果已经得到保留和定位；不是弃用，而是防止协议混杂。
- 作者要求本轮审计后停止训练，当前没有训练进程在运行；剩余 44 次正式训练留待后续明确指令。
- 下一轮若继续固定 split 无泄漏路线，应从断点恢复；若改为直接使用中文版结果，必须由作者明确批准改变证据口径，并在正文中如实标注。

## Round 6 — 中文版旧结果结构化与 600 dpi 候选图（2026-08-06）

### Reviewer 1：旧数值是否真正可追溯

**发现的问题**

- 旧绘图脚本大量硬编码 5 月汇总，且个别图使用替代单调指标或假定标准差；即使提高导出 dpi，也不能成为可靠图源。
- 中文版 Table 3 与 Table 5 对相近模型给出了不同 RMSE，若只按模型名汇总会产生静默冲突。

**已执行修正**

- 新建带 `source_id/table_id/metric/method/architecture/physics/ratio` 的 CSV 登记表，保留表级身份，不跨表去重。
- 建立独立元数据文件，明示无逐 seed 数据、协议不等价和禁止推断误差棒/显著性的边界。
- 新脚本只读取 CSV，不使用硬编码替代值，不读取当前 `v3_leakage_free` 结果，也不混合两代协议。

### Reviewer 2：图是否支撑主线而没有过度声称

**检查结果**

- 面板 a 仅显示中文版 Table 4 的四模型 MAE 点估计；面板 b 仅显示 Table 3 的架构×物理 RMSE 点估计。
- 结论被限定为“旧表报告的趋势”，没有误差棒、p 值、置信区间或“统计显著”措辞。
- 组图以 MAE 标签预算曲线为主证据、2×2 RMSE 曲线为结构性补充，符合当前叙事，但因协议状态未决而暂不进入活动正文。

### Reviewer 3：导出和版面质量

**检查结果**

- 组图和两个独立子图均由同一 Python 脚本、同一 CSV 直接生成，未截图或裁剪。
- PNG/TIFF 均为 600 dpi；同时导出 PDF/SVG。SVG 文本可编辑，且未嵌入栅格图。
- 原尺寸视觉检查未发现标签裁切、图例遮挡、线条重叠或难以区分的颜色编码；颜色以线型和标记形状双重编码。

### 交叉审稿结论

- 该组资产解决了“中文版已有结果未被利用”的素材层问题，同时没有模糊新旧证据边界。
- 下一步不是继续美化旧图，而是由作者决定其最终定位；训练继续保持暂停。

### 写作候选模块

- 已生成 `latex/springer-sn-2024/legacy_results_candidate.tex`，但未在活动主稿中 `\input`。
- 段落只报告旧表观察值及其差值；没有把历史趋势写成当前协议结论。
- 图注保持简短，来源、协议限制和解释边界放在正文展开，符合既定图文分工。

## Round 7 — 完成度复核与图件预览审计（2026-08-07）

### Reviewer 1：是否把脚本完成误写成论文图完成

**发现的问题**

- Fig. 3--5 的绘图脚本、布局和输出命名已经完成，但正式多种子矩阵只有 4/48 次，不能据此宣称正式图已生成。
- Legacy 候选图虽然是 600 dpi，也不能替代固定划分无泄漏结果。

**已执行修正**

- 在 `docs/CURRENT_COMPLETION_AND_FIGURE_PREVIEW.md` 中把“版式/脚本完成”和“正式图完成”分列。
- 正式 Fig. 3--5 均继续标记为等待完整矩阵；Legacy 图不分配正式图号。

### Reviewer 2：架构图是否存在模型语义歧义

**发现的问题**

- 三个横向 LSTM 方块未标时间索引，可能被误解为三层堆叠，而当前代码口径为两层 LSTM。
- 输入张量使用通用 `F`，未直接体现正文已确认的 16 维特征。

**处理**

- 两处问题已记录为 TBC-012；在作者确认前不擅自修改已插入的架构图。
- 其余视觉检查未发现节点/线条重叠、标签裁切或关键箭头中断。

### Reviewer 3：页数与投稿完整性

**检查结果**

- 当前 Springer 基线仍为 18 页，43 条 BibTeX 中活动正文使用 42 条。
- 20 页目标、Results 4.1--4.4、摘要/结论定量数字、单位/资助/代码仓库和目标期刊格式仍未完成。
- 页数只能通过真实结果表、图和讨论扩展，不能以重复图、长图注或空白填充。

### 交叉审稿结论

- 当前可展示的正式样式资产只有 Fig. 1 架构图；Legacy 结果图可展示风格但不能作为最终证据。
- 训练保持暂停。待作者决定 TBC-012/013 和后续是否恢复正式矩阵。

## Round 8 — 章节密度、协议一致性与扩展基线预审（2026-08-07）

### Reviewer 1：章节是否通过机械扩页维持

**发现与修正**

- 原 Methodology 的 LSTM 小节和 Experimental setup 的 metrics 小节均偏薄，已分别与相邻的表示学习、训练与统计协议合并。
- 将 `Results and Discussion` 拆分为独立的 `Results` 和 `Discussion`。Discussion 只设两个承担明确论证任务的小节，共 7 个实质段落，不插入装饰性图表。
- 当前 Springer 单栏编译达到 20 页；正式结果图表仍未插入，因此最终页数增长将来自证据而非留白。

### Reviewer 2：中英文与正式代码口径是否一致

**核查结果**

- 中文旧稿“自建数据集间歇标定”与用户后续确认的 HUST 受控随机掩码不能同时成立，已列为必须回写中文的 C-01/D-001；活动英文坚持后续确认口径。
- 数据加载器、运行日志、模型输入和英文特征表均为 16 维。配置文件中的 `top_k=6` 是未被跨电池入口执行的惰性旧字段，已列为 D-006。
- 正式调度器确实执行 20-epoch warmup（$2\times10^{-3}$ 到 $5\times10^{-3}$）后 cosine decay；正文描述保留。
- `min_delta=1e-5` 被归档但未进入训练判据。为避免中途改变 48-run 矩阵，正文已按真实行为改为“验证 MAE 严格下降即更新、20 epoch 无下降早停”，并登记 D-005。

### Reviewer 3：扩展基线是否公平且避免结果导向

**核查结果与修正**

- 同尺度数值基线限定为 XGBoost、LSTM、GRU 和 attention CNN--LSTM；均使用固定 split 42、16 维输入、40-cycle 窗口、相同掩码和配对种子。
- 不临时开发 Transformer：本地没有已审计实现，此时设计与调参会引入额外自由度。近期强方法只作协议背景，不与异构文献数值直接排名。
- XGBoost `smoke=true` 已验证只拟合保留标签窗口，结果自动排除正式聚合。
- 在任何正式神经扩展基线开始前，发现 GRU 默认关闭 scheduler；现已在公共覆盖配置中统一 warmup--cosine 调度，避免优化策略混杂。

### Reviewer 4：代码、编译和视觉质量

**检查结果**

- 全部本地回归测试通过：`17 passed`；仅 pytest 缓存目录权限警告，不影响测试结论。
- Tectonic/BibTeX 编译无 undefined citation/reference、overfull box 或 float-too-large；43 条 BibTeX 均含唯一 DOI，无重复 DOI。
- 新增 Discussion 与声明/参考文献衔接页已按 180 dpi 原页渲染检查，未发现孤立标题、裁切、过密表格或不平衡留白。
- 主矩阵已按用户后续授权恢复，当前无 `error.json`；在矩阵完成前不把中间结果写成总体结论。

### 交叉审稿结论

- 当前稿件的结构、作者信息、声明、文献数量和受控标签预算定义已稳定；正式 Results、摘要数值与结论仍必须等待完整多种子矩阵及同尺度基线。
- 所有不利结果、实现偏差与待同步中文项继续集中记录在 `docs/NEGATIVE_RESULTS_AND_DECISIONS.md`，不得在后续写作中静默消失。

## Round 9 — 架构图视觉重构与正文实页复核（2026-08-07）

### Reviewer 1：新图是否只是“换颜色的文字框”

**检查结果与修正**

- v5 不再用三个大分支卡片逐行复述层名；CNN 以蓝、绿、紫三组特征图薄片和 `k=3/7/15` 卷积核表示不同感受野，参数只作一次共享注释。
- 输入端用电压、电流和稀疏 SOH 点的小曲线承载数据语义；输出端用实际 SOH 退化曲线承载回归语义，减少从正文复制文字。
- 底部仅保留 masked MSE 与 soft monotonicity 两条互补证据路径，删除不承担独立论证任务的第三面板。

### Reviewer 2：LSTM 图形是否与实现一致

**检查结果**

- 两行明确代表两层 LSTM，三列明确代表 `t-1/t/t+1` 时间展开，不再把三个时间单元误读成三层网络。
- 图中标注 hidden size 64；回归头标注 FC 64、ReLU、dropout 0.4 和 sigmoid，与正文一致。
- LSTM 单元只保留四个彩色门控提示点，不重复门控方程，兼顾辨识度和信息密度。

### Reviewer 3：监督与物理约束是否夸大工程含义

**检查结果**

- 图题与 panel (b) 使用 `sparse lifecycle supervision` 和 `fixed cycle-level label budget`，没有把受控随机掩码表述为真实 BMS 间歇标定。
- 明确 `mask targets only`；完整 40-cycle × 16-feature 轨迹仍进入模型。
- 软单调路径标注作用于全部预测循环，并保留 `lambda=0.3, epsilon=.005, cmin=300, K=40, alpha=.2` 的实现口径。

### Reviewer 4：导出、插入与版面质量

**检查结果**

- `PI-MSCL_architecture_v5.drawio` XML 解析通过；Draw.io validator 为 0 errors、0 edge crossings。重叠警告来自有意叠放的特征图薄片和容器内部元素。
- SVG、PDF、PNG、TIFF 已重新生成；PNG/TIFF 为 600 dpi，SVG/PDF 保持矢量文本与线条。
- Tectonic/BibTeX 重新编译为 20 页；Fig. 1 位于第 6 页。180 dpi 实页检查未发现裁切、箭头断裂、线条穿越关键模块或无法辨认的层/时间步。

### 交叉审稿结论

- v5 可以替代 v4 进入活动正文。其视觉语法借鉴顶刊常见的信号小曲线、特征图薄片和时间展开单元，但网络结构、参数与稀疏监督/物理约束逻辑均由当前实现决定。
- 正式结果图仍必须等待 48-run 主矩阵完整；架构图完成不能被误写为论文全部图件完成。

## Round 10 — 架构图终尺寸复核与结果表预排版（2026-08-07）

### Reviewer 1：图形是否在最终正文尺寸下仍然成立

- 独立 600 dpi 预览暴露出输入卡下方 `mask targets only` 与 `40 cycles × 16 features` 在缩放后视觉粘连；已改为上下错层，而不是依赖源坐标中的理论间距。
- 更新后的 PNG/TIFF/PDF/SVG 与可编辑 Draw.io 已同步重建。正文重新编译为 20 页，Fig. 1 第 6 页实页检查未发现裁切、跨模块线条、断裂箭头或关键语义不可辨认。
- 结论边界：Fig. 1 只建立问题—模型—目标函数的结构对应，不构成性能优势证据。

### Reviewer 2：图中科学语义是否超过代码实现

- `mask targets only`、40-cycle × 16-feature 完整输入、三分支 `k=3/7/15`、两层 LSTM、hidden size 64、dropout 0.4、sigmoid 和软单调损失均可在当前正文/实现中核对。
- 软单调路径使用 `eligible predicted pairs`，避免把带有电池、循环起点和间隔条件的配对误写成“所有预测循环”。
- 受控随机掩码仍被限定为固定循环级标签预算协议，不写成真实 BMS 连续缺标或现场标定过程。

### Reviewer 3：正式结果表是否可读且统计口径明确

- 预先生成了不含真实结论的合成版式数据，仅用于检查 8 模型同协议比较表；合成值不会进入正文或正式结果归档。
- 全监督表和 `r=1.0/0.3` 端点表均在 Springer 原生 8 pt 表格字体下单栏容纳，无 `resizebox`、无 `small`、无 overfull box。
- 单元格保留三次配对重复的 mean ± SD；为避免模板 8 pt 数学字体缺失，排版实现使用文本正负号命令，显示语义不变。

### 交叉审稿结论

- 架构图现可作为活动正文的稳定 Fig. 1；此次修改来自终尺寸视觉证据，而非仅凭脚本坐标判断。
- 结果表版式已就绪，但表内正式数值仍必须等待主矩阵与同协议基线完整、通过种子配对校验后才能生成和插入。

## Round 11 — 轨迹指标与结果报告字段一致性（2026-08-07）

### Reviewer 1：方法定义能否由正式结果字段支持

- 核查发现评估函数虽计算 `mean_upward_excess`，正式 `result.json` 未归档该字段；原正文仍承诺报告该量，存在实现—写作不一致。
- 已删除该承诺，并把正式轨迹证据限定为单调违规率、每电池累计上升超差均值和平均绝对二阶差分三项。
- 补充了 $D_2$ 的显式公式及 eligible adjacent triplets 的分母语义，不改变训练或已归档结果。

### Reviewer 2：主性能指标是否完整报告

- MAE/RMSE 主图承担标签预算趋势，MAE 表承担精确读数，全监督表报告 MAE、RMSE、R²。
- 为避免只在 $r=1.0$ 报告 R²，新增加各标签预算 R² 表生成器；它读取同一严格验证后的 48-run summary，不引入新的实验或选择自由度。
- 五张主结果表均已用合成布局数据在 Springer 单栏中预排版；无缩小字号、无 overfull box。合成值不进入活动正文。

### Reviewer 3：新增表格是否造成冗余或人为扩页

- R² 表回答“标签预算下降时解释方差如何变化”，与 MAE/RMSE 的尺度不同，承担独立验证任务，不是重复展示同一数值。
- 轨迹表仍集中在稀疏端点，避免把每个比例的全部轨迹指标扩展成大而稀疏的表格。
- 当前正文仍为 20 页；正式图表插入后的页数增长将来自真实证据，不依赖留白或重复图。

### 交叉审稿结论

- 写作口径现与正式 JSON 字段一致；遗漏的额外轨迹字段已作为实现偏差登记，而不是在部分结果中事后补算。
- 新增 R² 表增强主结果完整性，版式与统计格式已提前验证，待完整矩阵后由同一生成器一次性填充。

## Round 12 — Results 证据顺序与实页结构（2026-08-07）

### Reviewer 1：算法论文的证据顺序是否清楚

- 将原本过薄的“Full-supervision reference”合并到“Performance across SOH-label budgets”，使 $r=1.0$ 成为主预算曲线的内部参照，而不是暗示为必然的“accuracy ceiling”。
- 在主预算结果之后、轨迹分析和二因素消融之前加入同协议补充基线小节，形成“主结果—外部模型族比较—轨迹证据—配对消融”的顺序。
- 异协议已发表数值继续只作背景，不进入同尺度排名。

### Reviewer 2：新增基线小节是否有足够证据承载

- 该小节预定包含 8 模型全监督表和 $r=1.0/0.3$ 配对端点变化表，且设置部分已经定义四个补充基线与公平性条件，不是只有一段文字的薄小节。
- 表格已用合成布局数据通过单栏预检；正式值必须等待 24-run 基线端点矩阵，不使用 smoke 或旧协议结果填充。
- 主四模型的全预算 2×2 因子矩阵仍承担机制解释，补充基线不参与主效应估计。

### Reviewer 3：分页与可读性

- 最新 Springer PDF 为 21 页。Results 从第 12 页开始，Discussion 第 14 页开始，Conclusion 第 16 页开始。
- 第 12--16 页逐页渲染检查未发现孤立标题、裁切、公式越界、引用断裂或不平衡留白；4.4 标题分两行但语义完整且未与正文分离。
- 当前页数增长来自新增同协议证据结构与严谨的轨迹指标定义，不是空白或重复图表。

### 交叉审稿结论

- Results 的结构现符合算法论文审稿路径，也遵守用户关于避免内容过少小节的要求。
- 所有结果段仍保持证据占位状态；正式数值、过去时陈述和强弱结论要等完整主矩阵与补充基线后一次性改写。

## Round 13 — 架构图主张边界与全监督多尺度配对结果（2026-08-07）

### Reviewer 1：架构图是否把视觉复杂度误写成方法创新

- v5 的三色 Conv1D 特征图、滑动卷积核和两层时序展开仅用于表达已实现的网络数据流，不作为性能证据。
- 正文图注保持一句简短说明，具体参数和两条损失路径由正文展开；没有以“top-journal style”等审美措辞支撑科学贡献。
- 图中未加入代码不存在的注意力、门控改造、伪标签或不确定性模块，避免因视觉丰富而扩大方法范围。

### Reviewer 2：全监督多尺度结果是否存在选择性指标报告

- `multi_no_pi/r=1.0` 三个预注册配对重复已经完整，且 3/3 配对的 MAE 均优于单尺度；平均改善 0.0664 个百分点。
- 同一批结果的 RMSE 平均恶化 0.0999 个百分点，$R^2$ 平均下降 0.0119；训练种子 2262 的 RMSE 恶化 0.4127 个百分点，不能因 MAE 改善而省略。
- 单调违背率和累计上升超差平均改善，但绝对二阶差分的配对方向不一致。因此只能报告“部分轨迹一致性改善”，不能概括为整体平滑性提升。

### Reviewer 3：当前证据能否提前支持 PI-MSCL 总体优越性

- 不能。当前完整的新单元仅隔离了全监督条件下的架构主效应，尚未覆盖其他三个标签预算及多尺度结构中的 PI 交互。
- Results 继续保留证据壳，不在 48-run 完整前写入总体均值、最优模型排序或标签预算鲁棒性结论。
- 不利权衡已登记为 N-004；最终写作必须同时呈现 MAE、RMSE、$R^2$ 和轨迹指标，不能围绕单一有利指标组织叙事。

### 交叉审稿结论

- 架构图可以保持为当前 Fig. 1，不再增加与真实实现无关的装饰或模块。
- 全监督多尺度结构表现为“平均绝对误差改善、尾部误差风险上升”的混合效应；这是待完整矩阵检验的机制线索，不是已经成立的论文总论断。
## Round 14 — 48-run 正式主结果写入、实页图表与主张边界（2026-08-07）

### Review setup

- 输入范围：活动 Springer 主稿的 Abstract、Results 4.1/4.3/4.4、Discussion、Conclusion，以及 48-run 严格汇总、正式表 4--8 和 Fig. 2--4。
- 评估边界：同协议四类补充基线仍在运行，因此 4.2 仅检查设计合理性，不判断最终模型排名。
- 共同事实：固定 split seed 42；四个模型×四个标签比例×三个配对重复；48/48 非 smoke 运行完整；测试电池完全标注且训练预处理无测试统计泄漏。

### Reviewer 1：技术可靠性与选择性报告风险

- 优点：正文同时报告 MAE、RMSE、$R^2$ 和三项轨迹指标；明确呈现全监督最低 MAE 与最低 RMSE/最高 $R^2$ 属于不同模型，并保留 30% 标签下 PI-MSCL 的不利 MAE 均值。
- 主要风险：代表性轨迹仍可能被误解为人工挑选。正文已给出中位重复—中位电池规则，并报告 seed 2262、mask 3262 和电池 10-6；选择源记录已归档，因此该风险被合理控制。
- 尚未闭合：4.2 必须等 24-run 端点基线完整并通过相同种子校验，不能使用既有 smoke 或异协议文献数值补表。
- 评估：主矩阵证据链技术上已成立；最终结论仍取决于补充基线闭合，但不影响本轮 2×2×4 内部效应的有效性。

### Reviewer 2：原创性、重要性与结论强度

- 优点：论文的可辩护贡献被限定为“在每个训练电池内受控分配循环级标签预算，并利用完整特征轨迹提供结构监督”，而不是把随机掩码包装为真实 BMS 缺标。
- 主要风险：若把 70% 标签结果外推成“标签越少，物理约束越有用”，会与 50%/30% 的不稳定或不利点误差冲突。Abstract、Results、Discussion 和 Conclusion 已统一改为预算依赖的条件性结论。
- 科学意义：当前证据更接近严谨的标签预算鲁棒性与准确率—结构一致性权衡，而非普适部署突破；这一边界在摘要末句和 Discussion 外部有效性段已明确。
- 评估：原创性主张可读且有边界，重要性主要面向电池 SOH 标签效率和受约束时序回归，不宜上升为跨化学体系通用结论。

### Reviewer 3：跨领域可读性与实页呈现

- 优点：Results 采用“主预算结果—同协议基线—轨迹稳定性—因子消融”的审稿路径；图注简短，具体数值和含义在正文展开。
- 实页检查：最新 PDF 为 24 页。将结果表改为 `[htbp]` 后，Results 标题先于表 4--5 出现；表格无溢出，Fig. 2--4 的粗体坐标轴、误差棒、竖向图例和 panel 标签在单栏终尺寸下可辨认。
- 主要风险：4.2 当前仍为将来时设计文字，导致该页证据密度偏低；正式基线表和过去时结果写入后应再次检查分页，不应靠额外留白或重复图扩页。
- 评估：非专业读者可以理解标签预算、结构约束和权衡结论；模型缩写已在前文定义，图表未承担超出正文解释的隐含主张。

### Cross-review synthesis

- 共识优势：无泄漏主矩阵完整；负结果未删除；点误差与轨迹一致性没有混为单一“性能”；代表性个案有确定性选择规则。
- 共识技术风险：同协议补充基线尚未闭合；固定单一电池划分和单一 LFP 数据源限制外部有效性；若忽略大标准差，容易高估模块主效应稳定性。
- 最重要的下一步：完成 24-run 基线、生成两张同协议表、把 4.2 改为有数值的过去时结果，再进行全稿三审稿人终审。

### Risk / unsupported claims

- 不支持：PI-MSCL 在所有标签预算下最优。
- 不支持：物理约束的点误差收益随标签减少而单调增强。
- 不支持：随机循环级掩码等价于连续前缀缺标、周期标定或自然现场缺失。
- 待证据：PI-MSCL 相对 XGBoost、LSTM、GRU 和 attention CNN--LSTM 的同协议端点排名。

## Round 15 — 24-run 补充基线闭环与终稿实页复核（2026-08-08）

### Reviewer 1：实验公平性、统计完整性与可复现性

- 补充矩阵含 XGBoost、LSTM、GRU、attention CNN--LSTM，覆盖 $r=1.0/0.3$ 和三个预登记配对种子；24/24 个正式结果完整，0 个正式错误，0 个预测归档缺失。一个 `smoke=true` 的 XGBoost 检查被汇总器显式排除。
- 跨归档校验确认补充基线与四模型主矩阵使用相同 split seed 42、训练/mask 种子、16 维输入、40-cycle 窗口和测试电池。三张表均由 JSON 自动生成，不手工抄录或挑选种子。
- 主要风险：$n=3$ 只能描述优化与掩码波动，不能支持精确显著性判断；正文使用均值±样本标准差并对小于离散性的端点变化保持谨慎。

### Reviewer 2：原创性、结论强度与负结果披露

- PI-MSCL 在八模型全监督比较中仅 MAE 最低；GRU 的 RMSE/$R^2$ 最优，attention CNN--LSTM 的 30% 标签 MAE 最低，LSTM/GRU 分别领先不同轨迹指标。摘要、Results、Discussion 和 Conclusion 已同步披露。
- 物理项的可辩护证据来自同架构、同掩码的配对比较：在 $r=0.3$ 改善三项轨迹量但没有稳定 MAE 收益；不得替换为“全局最平滑”或“稀疏监督最高精度”。
- 主贡献因此收敛为受控标签预算协议、泄漏隔离的跨电池评估，以及物理结构监督的条件性精度—规则性权衡，而非模型全面统治。

### Reviewer 3：叙事、图表和排版可读性

- Results 顺序为标签预算主结果→八模型同协议比较→统一轨迹指标→因子消融，证据由宽到深，4.2 不再是未来时占位段。
- 25 页 Springer 单栏 PDF 已逐页重点检查第 6、13–20 页。Fig. 1 的曲线、三 CNN 分支、两层 LSTM、回归头和双损失路径在最终缩放下可辨；Tables 6–8 无裁切，图例和坐标字号可读。
- 编译日志无 overfull、未定义引用、浮动体过大或 LaTeX error。存在模板分页引起的 underfull 警告，但实页未见异常空洞、孤立标题或内容截断。

### Cross-review synthesis

- 论文已从“PI-MSCL 是否全面优越”转为更可靠的条件性故事：全监督平均绝对误差优势、低预算下结构规则性改善，以及对大误差和监督锚点数量的敏感性同时成立。
- 同协议基线提高了结论可信度，但也降低了可宣称的模型支配范围；这是应保留的科学结果，不进行事后调参、删种子或替换指标。
- 当前随机掩码证据只支持分布式循环级标签预算鲁棒性。连续前缀、周期标定、跨化学体系和现场部署继续列为未来预声明验证，不进入当前结论。

### Unsupported claims after final audit

- 不支持：PI-MSCL 在所有监督比例或所有误差指标上最佳。
- 不支持：PI-MSCL 在 30% 标签下优于 attention CNN--LSTM 或 GRU 的点估计精度。
- 不支持：显式物理约束必然比普通 LSTM/GRU 产生更规则的轨迹。
- 不支持：随机循环级掩码等价于连续前缀、周期标定或自然现场缺失。

## Round 16 — 终稿主张校准、可复现性补足与 24 页实页复核（2026-08-08）

### Review setup

- 输入范围：活动 Springer 主稿的 Title、Abstract、Introduction、Methodology、Experimental setup、Results、Discussion、Conclusion，以及正式主矩阵、补充基线汇总和重新编译的 24 页 PDF。
- 评估边界：本轮不新增训练、不改动已冻结结果；仅核查论断是否由现有实验支持、基线是否足够可复现、语言是否准确，以及修改后的最终版面是否稳定。
- 共同事实：主矩阵 48/48、补充基线 24/24；固定 battery split seed 42；每个正式单元三次配对重复；随机循环级目标掩码只作用于训练电池，测试电池保持完整标签。

### Reviewer 1：技术可靠性、实验边界与可复现性

- 发现原 Experimental setup 曾声称多个独立电池划分被单独用作敏感性分析，但当前正式归档只有固定 split 42，无法支持该表述。活动正文现明确写为单一固定划分，并把效应量限定为对该电池划分有条件成立；独立划分被保留为未来补充分析。
- 二因素设计原先使用“independent effects”，可能误导读者认为两个模块效应在统计上独立。现改为基于共享 split/mask/train seeds 的配对边际效应与交互，并明确说明两模块在本矩阵中不呈独立可加性。
- 补充基线的层数、隐藏维度、卷积通道、注意力头数、dropout、XGBoost 树参数和共同优化协议已写入设置部分，同时声明没有证明架构特定最优调参或计算效率优势。
- 评估：主要实验链条可复核，关键限制与实施口径一致；固定单一划分仍是最重要的统计外部有效性限制。

### Reviewer 2：原创性、重要性与结论强度

- 删除“did not deteriorate detectably”和“reproducible structural improvement”等可能暗含显著性检验或强复现性的措辞，改为直接陈述均值方向、三个配对的结构指标方向及点误差混合符号。
- Abstract、Results、Discussion 和 Conclusion 均保持条件性结论：70% 标签下物理项同时改善 PI-MSCL 点误差与三项轨迹量；30% 标签下轨迹量改善，但 MAE 没有稳定收益；八模型比较也不支持全局支配。
- 论文最强可辩护贡献不是新型现场缺标机制，而是泄漏隔离的跨电池标签预算基准、完整特征轨迹上的标签独立结构监督，以及精度—轨迹一致性权衡的诚实量化。
- 评估：重要性足以支撑标签高成本条件下的 SOH 建模研究，但不能外推为部署就绪的 BMS 标定策略、跨化学体系通用性或连续时间外推能力。

### Reviewer 3：语言、叙事连贯性与实页呈现

- Introduction 的长文献枚举被拆分为研究背景与有限监督方法两组论证；Methodology、Experimental setup 和 Discussion 的长句也被拆分，减少一段承载多个限定条件的问题。
- 重新编译后的 PDF 为 24 页，仍满足 20 页以上要求。重点复核第 1、6、13--20 页：标题摘要、架构图、预算曲线、八模型表、代表轨迹、消融图和声明均无裁切、越界、孤立标题或不可辨文字。
- Fig. 2--4 的粗体坐标轴、panel 标签、误差棒和图例在 Springer 单栏终尺寸下可读；表格未使用截图或过度缩放，图注保持简短，解释仍在正文展开。
- 评估：稿件叙事已形成“受控问题定义—可核查模型—泄漏隔离实验—正负结果—边界化讨论”的闭环，语言密度和版面均达到可送作者终审的状态。

### Cross-review synthesis

- 三位审稿人的共识是：论文的可信度来自协议清楚、负结果完整和主张边界严格，而不是 PI-MSCL 在所有指标上的排名优势。
- 本轮修正的最高风险是删除未实际完成的独立划分敏感性声明；其次是消除暗示统计显著性、模块独立性或稳定复现性的措辞。
- 当前无需追加训练即可完成英文主稿的技术闭环。仍需作者决定的事项只有中文 DOCX 是否同步、legacy anomaly-risk 是否保留一句受限讨论，以及 Code Availability 的最终仓库信息。

### Risk / unsupported claims

- 不支持：结果已跨多个独立电池划分复现。
- 不支持：配对二因素结果证明多尺度与物理模块相互独立或可加。
- 不支持：均值未增加等价于通过显著性检验，或三次重复足以证明普遍稳健性。
- 不支持：同协议训练意味着每个基线均已获得架构特定最优调参或计算优势比较。
- 不支持：当前随机循环级掩码验证了连续前缀、周期标定、自然缺失或现场部署。
## Round 17 — Results reorganization, integrated analysis, and 23-page final QA (2026-08-08)

### Review setup

- **Input scope:** active Springer manuscript, formal 48-run factorial matrix, formal 24-run complementary baselines, all five active figures, eight active tables, and the newly compiled 23-page PDF.
- **Assessment boundary:** no new training was performed. All numerical claims were checked against existing non-smoke archives. External validation, multiple independent battery partitions, continuous-prefix masking, and periodic masking are not available in the current evidence package.
- **Shared claim:** PI-MSCL is evaluated under a controlled within-training-cell SOH-label budget; complete feature trajectories and a soft monotonic prior provide conditional structural supervision, but do not guarantee the lowest point error at every budget.

### Reviewer 1

- **Overall assessment:** The revised evidence order is technically coherent: aggregate accuracy, cell-level error dispersion, label-budget response, endpoint baselines, trajectory metrics, and paired factorial effects are now presented before the claim boundaries.
- **Who would be interested, and why:** Battery-health researchers and constrained time-series learning researchers can use the work as a reproducible example of separating label density from feature availability and of separating point accuracy from trajectory regularity.
- **Major strengths:** battery-level leakage isolation; fixed split and paired seeds; explicit random-mask semantics; unfavorable results retained; representative run and cell selected by a deterministic median rule; loss and metric definitions now colocated with the experimental protocol.
- **Major concerns:** one fixed HUST partition and three repetitions limit uncertainty characterization; random cycle-level masking does not establish temporal extrapolation or field-calibration realism; the physical term is a structural monotonic prior rather than an electrochemical model.
- **Technical failings before a broader case is established:** independent split sensitivity, continuous-prefix/periodic-mask experiments, and external chemistry or operating-condition validation remain absent.
- **Nature-style criteria:** technically sound within the declared protocol; originality is clearest in the controlled evidence design and conditional accuracy–regularity analysis; outstanding and interdisciplinary importance is not established from one dataset and one masking family; readability is substantially improved by integrating analysis with each result.
- **Recommendation posture:** technically credible for a specialist SCI venue, while broad claims must remain bounded until external and protocol-diversity tests are added.

### Reviewer 2

- **Overall assessment:** The strongest contribution is not universal model dominance, but the controlled demonstration that structural priors can improve trajectory behavior without a stable point-error advantage when numerical anchors become sparse.
- **Who would be interested, and why:** Readers studying weak supervision, label-efficient regression, and physics-guided learning may value the observed non-additivity between multi-scale representation and the monotonic regularizer.
- **Major strengths:** the manuscript now reports GRU, LSTM, and attention CNN–LSTM advantages alongside PI-MSCL results; the full-supervision mean-MAE advantage is explicitly distinguished from per-cell and tail-error behavior; the factorial analysis prevents a simple additive-modules narrative.
- **Major concerns:** novelty relative to prior weakly/semi-supervised battery SOH work remains primarily methodological and evaluative rather than a new physical mechanism; the significance case should not exceed robustness to a distributed target-label budget.
- **Technical failings before a broader case is established:** no evidence yet supports continuous early-to-late forecasting, periodic BMS calibration, cross-chemistry transfer, or deployment-level computational superiority.
- **Nature-style criteria:** original as a bounded protocol-plus-analysis contribution; scientifically useful but presently field-local; interdisciplinary reach is plausible through constrained time-series learning but not demonstrated experimentally; technical claims are appropriately qualified; the revised structure is readable.
- **Recommendation posture:** promising specialist contribution; the current manuscript should avoid framing the random mask as a direct surrogate for real BMS missingness.

### Reviewer 3

- **Overall assessment:** Removing the standalone Discussion improves the narrative because interpretation now appears next to the evidence that supports it. The four-paragraph conclusion is concise and does not repeat the entire results section.
- **Who would be interested, and why:** Nonspecialist machine-learning readers can understand the general problem as learning from complete input trajectories with incomplete target annotations, while battery readers retain the SOH-specific context.
- **Major strengths:** Fig. 2 provides both pooled agreement and per-cell error spread; Fig. 4 prevents aggregate statistics from hiding late-life bias; captions remain short and explanations are carried by the prose; the 23-page single-column PDF has no visible clipping or illegible labels.
- **Major concerns:** the dense terminology in the architecture and trajectory-metric passages still demands careful reading; the broad-interest motivation would be stronger with an external demonstration beyond HUST.
- **Technical failings before a broader case is established:** not assessable as a deployment study because inference cost, energy use, online calibration, and field data are not reported; these are correctly not claimed as current results.
- **Nature-style criteria:** clear and professionally presented; originality and importance are understandable but remain more compelling to the immediate battery/ML community than to a broad interdisciplinary readership; technical caveats are visible rather than hidden.
- **Recommendation posture:** readable and submission-ready for a relevant specialist journal after author decisions on Chinese synchronization and code availability.

### Cross-review synthesis

- **Consensus strengths:** the protocol is explicit and leakage-aware; result interpretation is adjacent to evidence; negative and mixed results are preserved; the manuscript distinguishes mean MAE, tail errors, per-cell variability, and trajectory regularity.
- **Consensus technical risks:** fixed single split, one LFP dataset, three repetitions, and random masking only; no evidence for prefix forecasting, periodic calibration, external chemistry transfer, or computational superiority.
- **Where emphasis differs:** Reviewer 1 prioritizes missing validation protocols, Reviewer 2 prioritizes novelty/significance boundaries, and Reviewer 3 prioritizes accessibility and figure-page presentation.
- **Broad-interest readout:** the complete-feature/incomplete-target formulation has cross-domain relevance, but the present evidence establishes a bounded battery-SOH result rather than far-reaching generality.
- **Most important unresolved issues:** synchronize or formally supersede conflicting Chinese legacy claims, confirm the code repository statement, and treat protocol-diversity/external validation as future work rather than current evidence.

### Risk / unsupported claims

- Unsupported: PI-MSCL is best under all label budgets, for every cell, or on all error metrics.
- Unsupported: random cycle-level masking is equivalent to continuous-prefix missingness, periodic BMS calibration, or natural field missingness.
- Unsupported: the soft monotonic constraint is a mechanistic electrochemical law.
- Unsupported: the study establishes cross-chemistry deployment generality or computational superiority.
