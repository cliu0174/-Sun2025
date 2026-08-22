# 论文实验重跑与证据归档计划

> 建立日期：2026-08-06  
> 当前状态：主实验脚本、汇总器和测试均已完成；`r=0.3` 的一个正式 2×2 配对组已完成（4/48），作者要求本轮起暂停训练  
> 运行入口：`experiments/run_paper_main_results.py`

## 1. 为什么主结果必须重跑

旧实验的均值对模型开发仍有参考价值，但尚不满足最终论文的可追溯和唯一变量要求：

1. 服务器逐种子 `result.json` 和逐循环预测未完整同步到本地，无法可靠复算标准差和轨迹指标；
2. 旧代码中无物理模型跨电池边界滑窗，有物理模型逐电池滑窗，导致架构—物理消融混入数据管线差异；
3. 旧 Exp-09 后验单调指标使用 `tolerance=0.01`，训练使用 `0.005`，且未保存真实循环索引；
4. 三个旧 seed 同时改变电池划分、标签掩码和训练初始化，正文必须准确选择并声明重复实验回答的问题；
5. 旧跨电池加载流程在电池划分前对每块电池单独拟合标准化器，使验证/测试电池自身完整生命周期统计量进入预处理，构成跨电池评估泄漏。

因此，旧数值只作为预期范围和故障诊断基线；正文中的主表、误差棒、轨迹图和二维消融以统一管线重跑结果为准。

## 2. 已完成的代码更正

### 2.1 训练入口

`train_cross_battery.py` 已新增：

- `split_seed`：与训练初始化 seed 分离；
- `preserve_battery_boundaries=True`：有/无物理约束均逐电池窗口化；
- `results_dir_override`：每次运行使用独立目录，不再相互覆盖；
- `save_diagnostic_plots`：正式批量训练可关闭旧 300 dpi 诊断图；
- 保存逐预测 `battery_ids` 与真实窗口目标 `cycle_indices`；
- 同时归档 cycle-level 名义标签数与 many-to-one 窗口目标的实际有标签数/比例；
- 在结果中保存 train/val/test 电池列表和全部 seed 元数据。

旧跨电池窗口模式仍可通过 `preserve_battery_boundaries=False` 复现，但不得用于论文。

### 2.2 无泄漏特征标准化

`data_loaders/data_loader_hust.py` 已增加 `standardize_features` 开关。单电池旧任务默认保持原行为；论文跨电池入口显式设为 `False`，先读取每块电池的原始 16 列候选特征，再完成 battery-level train/validation/test 划分，最后只在训练电池上拟合一次 `StandardScaler` 并变换其余集合。任何在测试电池上单独拟合或使用其生命周期统计量的结果均不得进入论文。

### 2.3 轨迹评价

新增 `evaluation/trajectory_metrics.py`，统一计算：

- MAE、RMSE；
- 单调违反率（百分数）；
- 平均违反超量；
- 每块电池累计异常上升超量；
- 一阶变化幅度与二阶轨迹粗糙度；
- 每块电池的独立结果；
- 对每块测试电池先计算四个主模型的 MAE 均值，再选择最接近跨电池中位数的电池；并列时按 battery ID 决定，避免依赖任一模型或事后挑图。

评价和训练统一采用 `tolerance=0.005`、`min_cycle=300`，并使用保存的真实循环索引。

### 2.4 自动化测试

当前 14 项测试全部通过，包括：

- 单调序列无违反；
- tolerance 与 min-cycle 正确生效；
- 电池内按 cycle 排序；
- 中位误差电池选择可重复；
- 新管线窗口不跨电池；
- 旧兼容模式能够复现跨边界问题，从而证明回归测试有效；
- 跨电池读取返回未缩放特征，且单电池兼容模式仍保持原标准化行为；
- 600 dpi 组图/子图导出与 smoke/单次结果拒绝规则。
- smoke 与正式运行目录隔离，分批执行后的汇总不会覆盖此前完成的模型。
- 代表轨迹的跨模型中位电池选择、600 dpi 组图与独立子图导出。

运行命令：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_paper_runner_archival tests.test_paper_trajectory_plotting tests.test_feature_standardization_scope tests.test_battery_boundary_windowing tests.test_trajectory_metrics tests.test_paper_plotting -v
```

## 3. 冻结的主实验矩阵

所有组统一使用 `ms_cnn_lstm_v2` 实现、相同输入、窗口长度 40、训练配置、标签掩码和电池划分。唯一变化为尺度结构和软单调约束两个因子。

| ID | 卷积结构 | 软单调约束 | 论文作用 |
|---|---|---|---|
| `single_no_pi` | 单尺度 kernel 7 | 无 | 纯数据驱动参照 |
| `single_pi` | 单尺度 kernel 7 | 有 | 单尺度下的物理约束效应 |
| `multi_no_pi` | 并行 kernel 3/7/15 | 无 | 多尺度结构的独立效应 |
| `multi_pi` | 并行 kernel 3/7/15 | 有 | 主模型 PI-MSCL |

主标签预算：`r = 1.0, 0.7, 0.5, 0.3`。

这个 2×2 矩阵同时服务：

- 4.1 完整监督跨电池结果；
- 4.2 标签预算下降主结果；
- 4.3 稀疏监督轨迹行为；
- 4.4 架构—物理约束二维消融。

不再分别运行一套彼此配置不一致的“主结果”和“消融结果”。

## 4. 两种合法重复口径

`--protocol` 仍为必填项以防误运行。作者已确认以 fixed split 作为主表口径，并将 independent splits 作为单独的跨划分敏感性验证。

### A. independent_splits

每个 repeat seed 同时定义一个独立电池划分；同一 repeat 内四个模型和四个 `r` 共用相同划分和掩码，因此比较保持配对。

回答：结论能否跨不同随机电池划分保持稳定。  
限制：总方差同时包含划分、掩码和训练随机性，不能解释成只改变掩码。

```powershell
.\.venv\Scripts\python.exe experiments\run_paper_main_results.py --protocol independent_splits --device cuda
```

### B. fixed_split

所有重复共用一个固定电池划分，训练 seed 与 mask seed 按 repeat 变化。

回答：固定未见电池集合后，模型对训练初始化和随机标签位置是否稳定。  
限制：结论依赖该特定电池划分，需要在局限性中说明，或另加少量外部分割复核。

```powershell
.\.venv\Scripts\python.exe experiments\run_paper_main_results.py --protocol fixed_split --split-seed 42 --device cuda
```

TBC-002 已解决：B 为主表，固定 split seed 42；训练 seeds 为 `[929, 2262, 7]`，对应的独立 mask seeds 为 `[1929, 3262, 1007]`。A 仅作为跨划分敏感性补充，结果单独报告，不与 B 混合平均。

正式主矩阵命令：

```powershell
.\.venv\Scripts\python.exe experiments\run_paper_main_results.py --protocol fixed_split --split-seed 42 --training-seeds 929 2262 7 --mask-seeds 1929 3262 1007 --device cuda
```

## 5. 输出和断点续跑

每个运行目录独立保存：

- `run_manifest.json`：模型、协议、三个 seed、标签比例和完整 override；
- `battery_split.json`：train/val/test 电池列表；
- `model_checkpoint.pth`；
- `results.pkl`：训练历史和完整结果；
- `predictions.npz`：预测、真值、battery ID、cycle index；
- `trajectory_metrics.json`：总指标与逐电池指标；
- `result.json`：用于汇总和作图的扁平结果；
- 失败时保存 `error.json`。

完全匹配当前清单的已完成目录会自动跳过；元数据不匹配时脚本停止并要求人工检查，只有显式使用 `--force` 才会覆盖。泄漏修复后的正式证据使用独立 pipeline revision，smoke 目录使用 `_smoke` 后缀。分批运行时，汇总器扫描该 revision 下的全部已完成结果，而不是只汇总最后一批。正式运行结果存放于：

```text
experiments/paper_main_results/<protocol>/v3_leakage_free/
```

## 6. 已完成烟雾测试

烟雾命令：

```powershell
.\.venv\Scripts\python.exe experiments\run_paper_main_results.py \
  --protocol fixed_split --split-seed 42 \
  --models multi_pi --ratios 0.3 \
  --training-seeds 929 --mask-seeds 1929 \
  --device cpu --smoke
```

结果：成功完成；30,878 个测试窗口均保存 battery/cycle 元数据；完整归档链可读。最近一次单个 1-epoch CPU 运行的训练入口耗时约 63 s。该结果标记为 `smoke=true`，汇总器不会将其纳入正式统计，也不得进入正文。该 smoke 发生在 revision 目录引入前，仅用于实现检查，正式运行仍须在 `v3_leakage_free` 下生成。

项目 `.venv` 的 PyTorch 为 CPU-only；系统 Anaconda 环境已核验可使用 RTX 2050 CUDA，并完成 `r=0.3`、split 42、mask 1929、train 929 的四模型正式配对组。正式 48 次矩阵（4 模型 × 4 标签率 × 3 repeats）当前完成 4 次、剩余 44 次。按作者 2026-08-06 的指示，本轮审计后暂停所有训练；只有收到后续明确继续指令才恢复。

## 7. 结果不理想时的调整纪律

调整只允许围绕预先定义的问题进行，不通过更换测试电池、事后挑选 seed 或删除负结果制造优势。

1. 先检查实现、数据泄漏、seed 配对、训练收敛和指标单位；
2. 若主模型在 `r=0.3` 表现不佳，只在验证集上调整 `λ_mono`、`δ` 或延迟启用位置；测试集不用于调参；
3. 每轮策略、搜索空间、验证依据和全部测试结果保留在独立目录；
4. 多轮后改善微薄或为负，仍将最终真实结果写入正文，并把“架构是主要收益、物理约束具有条件性”作为讨论边界；
5. 不因为结果不符合预期临时改成连续前缀、周期性或其他标签协议。

## 8. 尚缺的实验/材料

| 缺口 | 是否主线必需 | 实现状态 | 下一步 |
|---|---|---|---|
| 公平 2×2 × 四标签率重跑 | 是 | 脚本完成 | GPU 正式运行 |
| 逐种子 mean ± std | 是 | 汇总器完成 | 等正式运行 |
| 轨迹级全测试指标 | 是 | 代码和测试完成 | 由正式预测自动生成 |
| 代表电池轨迹 | 是 | 跨模型中位电池规则和脚本完成 | 正式结果后执行 |
| 600 dpi 主结果/消融/轨迹图 | 是 | 组图/独立子图脚本和测试完成 | 等正式结果生成最终资产 |
| 传统/现代额外基线 | 视主表完整性 | 待盘点 | 统一管线后选择少量代表模型 |
| 不确定性、特征扰动 | 否，补充 | 旧汇总存在 | 原始结果可追溯后决定是否重跑 |
| 异常检测 | 否 | 旧结果为弱/负 | 暂不进入主文 |
| 连续前缀/周期性标签 | 否 | 未运行 | 不进入当前结论 |

本地旧结果、中文版表格、开发阶段汇总与当前无泄漏结果的完整分级见：

- `docs/LOCAL_RESULT_ASSET_AUDIT.md`

中文版 Tables 1--5 已整理为机器可读登记表，并在不训练的情况下生成 600 dpi legacy 候选组图和独立子图；它们尚未进入当前活动正文，不替代正式主矩阵。

## 9. 下一次运行前检查

- [x] 作者确认 TBC-002：fixed split 主表，independent splits 单独作敏感性验证；
- [x] 本机 CUDA 状态已核验：RTX 2050 4 GB、驱动 561.09、PyTorch 2.5.1+cu121，`torch.cuda.is_available()` 为 True；
- [x] 已在本机 CUDA 环境完成一个非 smoke 的正式 2×2 配对组，核对时间、显存、收敛和归档；
- [x] 已核验四个正式目录均包含 manifest、split、checkpoint、predictions、result 和 trajectory metrics；
- [ ] 训练暂停；后续经作者明确同意后再利用断点续跑完成剩余 44 次；
- [ ] 同步整个 `paper_main_results` 目录，而不是只抄均值；
- [ ] 作图前运行全部 14 项回归测试并重新聚合；
- [ ] 使用 `scripts/plot_paper_main_results.py` 生成 Fig. 3/5，使用 `scripts/plot_paper_trajectories.py` 生成 Fig. 4 及 `fig4_source.json`。
