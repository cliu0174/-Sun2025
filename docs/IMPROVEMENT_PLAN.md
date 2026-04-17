# PI-CNNLSTM 模块化改进提优计划

> **项目**: 基于物理一致性约束的电池 SOH 估计（PI-CNNLSTM）
> **数据集**: HUST 77 块电池，14 维特征
> **基线**: **baseline-v2.2**（CNN-LSTM + 软单调约束）
> **创建日期**: 2026-04-17
> **最后更新**: 2026-04-17（Stage 0 启动前对齐）
> **状态**: Stage 0 执行中
> **用途**: 硕士论文第四章 / 潜在期刊投稿

---

## 0. 背景摘要（给未来的自己/Claude）

本项目是一个电池健康状态（SOH）估计任务，核心场景是**部分生命周期监督**（Partial Lifecycle Supervision）——即每块电池只有一部分循环有 SOH 标签，模型需要在监督稀疏的条件下泛化到全生命周期。

### 🔴 关键现实确认（非常重要！）

1. **当前代码中没有部分监督实现**——`utils/data_augmentation` 目录不存在，`scenario3` 及其它 ImportError
2. **当前物理约束开关是关闭的**（`cnn_lstm_config.json` 中 `enabled: false`）——需要 Stage 0 中改回 true 并重跑
3. **当前"最好结果"实际是纯 CNN-LSTM**（无物理约束、无部分监督）
4. **需要实现的基础设施**：部分监督机制（label masking 方式）

### 最终定稿的核心叙事（论文故事线）

```
论文主故事：

在电池 SOH 估计的"部分生命周期监督"场景下，我们：
1. 通过注意力机制（M1）增强时序表征
2. 通过速率连续性约束（M4）精化物理先验
3. 通过自适应损失权重（M5）替代人工调参
4. 通过 MC Dropout（M2）+ 不确定性伪标签（M6）主动利用未标注区间信息
5. 通过置信区间评估（M7）为 BMS 决策提供可靠性量化

核心卖点图：
  "监督稀疏度 vs 模型误差"——我们的方法在稀疏监督下优势更大
```

### 推荐的最终模块组合

```
核心架构：CNN-LSTM + 循环级注意力（M1）
损失函数：自适应权重（M5）+ 软单调 + 边界 + 速率连续性约束（M4）
训练策略：不确定性引导伪标签扩展部分监督（M6）⭐核心
评估：MC Dropout 置信区间（M2+M7）+ 特征缺失鲁棒性（M8）
```

---

## 1. 基线协议（baseline-v2.2）

### 1.1 数据划分
- **比例**：60 / 20 / 20（训练 / 验证 / 测试）
- **划分单位**：按电池划分（不是按循环），确保测试集电池完全未见
- **种子**：42（固定）
- **数量**：77 块电池 → 46 / 15 / 15（整数截断）
- **实现位置**：`train_cross_battery.py::split_batteries()` 第 174 行

### 1.2 模型结构
- **主干**：CNN-LSTM（`models/cnn_lstm.py`）
- **CNN**：channels=[256, 128], kernel_size=7
- **LSTM**：hidden=64, num_layers=2
- **FC**：[64]
- **Dropout**：0.4
- **窗口大小**：40
- **特征选择**：Top-6 by correlation（from 14 维）

### 1.3 物理约束（baseline-v2.2 **必须启用**）
- `physics_constraints.enabled: true`
- `monotonic_weight: 0.1`（软单调）
- `boundary_weight: 0.05`（边界约束）—— **需要改！当前是 0.0**
- `monotonic_tolerance: 0.01`
- `temporal_decay`：启用，exp 衰减，alpha=0.2

### 1.4 部分监督协议
- **实现方式**：**Label Masking**（不是 Sample Dropping！）
  - 保留所有 (x, y) 样本
  - 为每个样本增加 `is_labeled` 布尔标志
  - 按电池随机选择 N% 的循环标记为 `is_labeled=True`
  - Loss 计算时：
    - **MSE (监督 loss)**：只对 `is_labeled=True` 的样本计算
    - **物理约束 (mono/bound/smooth)**：对**所有样本**计算（包括无标签样本）
- **监督比例矩阵**：`[1.0, 0.7, 0.5, 0.3]`
  - 100% = 全监督（作为上界参考）
  - 70%, 50%, 30% = 部分监督（论文核心场景）
- **随机性**：按 (battery_id, seed) 确定 mask 种子，保证可复现

### 1.5 训练配置
- **epochs**: 200
- **batch_size**: 256
- **optimizer**: Adam
- **lr_scheduler**: WarmupCosineDecay（warmup 20 epoch, base_lr=0.005）
- **early_stopping**: patience=20

### 1.6 基线评估指标
| 指标 | 含义 |
|------|------|
| MAE | 平均绝对误差 |
| RMSE | 均方根误差 |
| MAPE | 平均百分比误差 |
| R² | 决定系数 |

### 1.7 基线跑法（5 seeds × 4 ratios = 20 runs）
- 种子：`[42, 123, 456, 789, 2024]`
- 监督比例：`[1.0, 0.7, 0.5, 0.3]`
- 每个 (seed, ratio) 组合独立训练一次
- 最终指标：取 5 个种子的 `mean ± std`

---

## 2. 模块化设计原则

- **可插拔**：每个改进点独立封装为一个模块，通过配置开关（config flag）控制启用
- **可组合**：模块间通过清晰接口通信，支持任意组合消融
- **可复现**：每次实验固定 seed，保留完整配置快照
- **可比较**：统一训练脚本 + 统一评估接口，仅改配置即可切换实验

---

## 3. 模块清单

### 3.1 模块总览表

| ID | 模块 | 文件位置 | 类型 | 依赖 | 优先级 |
|----|------|---------|------|------|--------|
| M1 | 循环级注意力 | `models/modules/attention.py` | 架构 | 无 | P0 |
| M2 | MC Dropout 包装 | `models/modules/mc_dropout.py` | 架构 | 无 | P0 |
| M3 | 个体归一化 | `models/modules/instance_norm.py` | 架构 | 无 | P2 |
| M4 | 速率连续性约束 | `losses/physics/rate_smoothness.py` | 损失 | 无 | P0 |
| M5 | 自适应损失权重 | `losses/adaptive_weight.py` | 损失 | 无 | P0 |
| M6 | 不确定性伪标签 | `training/pseudo_labeling.py` | 训练 | M2 | P1 |
| M7 | 置信区间评估 | `evaluation/uncertainty_eval.py` | 评估 | M2 | P0 |
| M8 | 特征缺失鲁棒性 | `evaluation/robustness_eval.py` | 评估 | 无 | P1 |

> **优先级说明**：P0 必做，P1 强烈建议，P2 时间充裕再做

### 3.2 各模块详述

#### M1：循环级注意力（Cycle-level Attention）
- **目的**：让模型自适应聚焦于对当前 SOH 估计最重要的历史循环
- **位置**：LSTM 输出之后、FC 头之前
- **接口**：输入 `(B, T, H)`，输出 `(B, H)` 聚合向量 + 可选 `(B, T)` 注意力权重
- **实现要点**：轻量 Multi-Head Self-Attention（4 heads），输出 Global Average Pool 或 [CLS] token
- **论文说辞**：
  > "电池退化过程中存在信息不对称性——近期循环与历史特定退化节点对当前健康状态的贡献不尽相同。本文在 CNN-LSTM 的时序表征层引入循环级注意力机制，使模型能够自适应地聚焦退化敏感的历史区间。"

#### M2：MC Dropout 包装（Monte Carlo Dropout）
- **目的**：测试时保留 Dropout，多次前向传播得到预测分布
- **位置**：FC 头中的 Dropout 层
- **接口**：`predict_with_uncertainty(x, n_samples=50) -> (mean, std)`
- **实现要点**：训练时正常 Dropout，推理时强制 `model.train()` 或自定义 `MCDropout` 层
- **论文说辞**：
  > "在部分生命周期监督条件下，模型对未观测退化阶段的估计本质上包含不可约减的认知不确定性。本文通过测试时保留 Dropout 进行多次随机前向传播，构建 SOH 预测的置信区间。"

#### M3：个体归一化（Instance Normalization）
- **目的**：消除电池个体间的绝对幅值差异，让模型聚焦退化模式形状
- **位置**：CNN 输入前的第一层
- **实现**：`nn.InstanceNorm1d`

#### M4：速率连续性约束（Rate Smoothness）
- **目的**：约束相邻循环的 SOH 差分不能突变，即退化速率连续
- **公式**：`L_smooth = Σ (Δ²SOH)²`（二阶差分平方和）
- **实现**：
  ```python
  def rate_smoothness_loss(soh_pred):
      dsoh = soh_pred[:, 1:] - soh_pred[:, :-1]
      d2soh = dsoh[:, 1:] - dsoh[:, :-1]
      return (d2soh ** 2).mean()
  ```
- **超参扫描**：weight ∈ {0.01, 0.05, 0.1, 0.2}
- **论文说辞**：
  > "传统软单调性约束仅要求 SOH 序列全局非上升，但在局部窗口内允许预测值出现突变。本文进一步引入退化速率连续性约束：对相邻循环的 SOH 差分序列施加平滑正则。"

#### M5：自适应损失权重（Uncertainty-Weighted Multi-task）
- **目的**：消除人工调参，让模型自动平衡各损失项
- **公式**：`L = Σ (1/2σᵢ²)·Lᵢ + log σᵢ`，其中 `log σᵢ` 为可学习参数
- **实现**：
  ```python
  class UncertaintyWeightedLoss(nn.Module):
      def __init__(self, num_tasks):
          super().__init__()
          self.log_vars = nn.Parameter(torch.zeros(num_tasks))
      def forward(self, losses):
          weights = torch.exp(-self.log_vars)
          return (weights * losses).sum() + self.log_vars.sum()
  ```
- **注意**：启用 M5 时，其他固定权重必须全部关闭；可对 log σᵢ 加 L2 正则防止塌缩

#### M6：不确定性引导伪标签（Uncertainty-Guided Pseudo-Labeling）⭐核心创新
- **目的**：在部分监督场景下，利用模型对未标注区间的高置信度预测作为伪标签
- **依赖**：M2（MC Dropout）
- **流程**：
  1. Warmup (e.g. 30 epoch)：仅用真实标签训练
  2. 每 K epoch 重新生成伪标签：
     - MC Dropout 多次推理得到 `(μ, σ²)`
     - 筛选 `σ² < τ` 的循环作为伪标签
     - 伪标签权重 `= 1/(σ² + ε)`
  3. 总损失：`L = L_sup + λ_pseudo · L_pseudo + L_physics`
- **风险**：伪标签漂移 → 限制数量上限，或每次重置后重新筛选
- **论文说辞**：
  > "本文提出不确定性引导的自监督边界扩展策略：在训练过程中利用模型对未标注区间的预测置信度动态筛选高质量伪标签，仅将不确定性低于自适应阈值的预测纳入辅助监督。"

#### M7：置信区间评估
- **依赖**：M2
- **指标**：
  - PICP（Prediction Interval Coverage Probability）：95% CI 理论上应覆盖 95% 真值
  - MPIW（Mean Prediction Interval Width）
  - 不确定性-误差 Spearman 相关系数

#### M8：特征缺失鲁棒性评估
- **测试场景**：
  - 随机 mask 0/1/2/3 维特征
  - 添加高斯噪声（σ = 0.01, 0.05, 0.1）
  - 传感器线性漂移
- **对比**：Baseline vs Full 的性能衰减幅度

---

## 4. 分阶段实施路线图

### Stage 0：基线锁定（2 天）← **当前阶段**

#### Step 0.1: 代码审查（已完成 ✅）
- 读 `train_cross_battery.py`，理解训练流程
- 确认 60/20/20 + 种子 42 已在 `split_batteries()` 实现
- 确认 `PhysicsConstrainedLoss` 结构支持 label masking 扩展
- 发现 `scenario3` 等无法使用（`utils/data_augmentation` 缺失）

#### Step 0.2: 实现部分监督（label masking）
- 在 `prepare_cross_battery_data()` 中新增 `supervision_ratio` 参数
- 生成 `train_supervision_mask`（按电池随机 mask）
- 修改 Dataset 传递 mask 到 batch
- 修改 `PhysicsConstrainedLoss.forward()` 支持 masked MSE
- **交付**：`train_cross_battery.py` 支持 `supervision_ratio ∈ [0, 1]` 参数

#### Step 0.3: 启用物理约束
- `cnn_lstm_config.json` 中：
  - `physics_constraints.enabled: false → true`
  - `boundary_weight: 0.0 → 0.05`（需要调，当前是 0）
- 其他参数保持默认

#### Step 0.4: 烟雾测试
- 用小 epoch 数（e.g. 5）跑一次完整流程，验证：
  - 部分监督 mask 正确
  - 物理约束 loss 非零
  - 训练能正常收敛
- **交付**：日志截图 / 简短 report

#### Step 0.5: 正式基线（20 次运行）
- 种子：`[42, 123, 456, 789, 2024]`
- 监督比例：`[1.0, 0.7, 0.5, 0.3]`
- **交付**：`experiments/baseline_v2.2/metrics.json`（mean ± std 表）

#### Step 0.6: 固化与 commit
- 冻结 `cnn_lstm_config.json` 作为 baseline-v2.2 的最终配置
- 更新 `CLAUDE.md` 和本文档的进度
- Commit message: `stage-0: lock baseline-v2.2 with physics + partial supervision`

### Stage 1：架构增强（1.5 天）
- [ ] **Exp-01**：Baseline + M1（注意力）
- [ ] **Exp-02**：Baseline + M2 + M7（MC Dropout + 置信区间）

### Stage 2：损失函数精化（2 天）
- [ ] **Exp-03**：Baseline + M4（速率连续性）
- [ ] **Exp-04**：Baseline + M5（自适应权重）
- [ ] **Exp-05**：Baseline + M4 + M5（组合）

### Stage 3：训练策略升级（2 天）⭐核心
- [ ] **Exp-06**：Baseline + M2 + M6（不确定性伪标签）

### Stage 4：全模块集成（1 天）
- [ ] **Exp-07**：Full Stack (M1+M2+M4+M5+M6+M7)
  - Warmup 30 epoch → L_sup + L_physics
  - Epoch 30+：每 5 epoch 更新伪标签
  - 全程 M5 自适应权重

### Stage 5：鲁棒性验证（1 天）
- [ ] **Exp-08**：M8 特征缺失鲁棒性

### 分析与写作（2 天）
- [ ] 消融表、曲线图、注意力可视化
- [ ] 论文章节撰写

**总预计工期：10.5 天**（原 10 天 + Stage 0 延长 0.5 天）

---

## 5. 消融矩阵（最终论文表格）

每个实验在 4 个监督比例下都要跑（ratio ∈ [1.0, 0.7, 0.5, 0.3]）：

| 实验 | M1 | M2 | M4 | M5 | M6 | MAE@100% | MAE@70% | MAE@50% | MAE@30% | PICP |
|------|----|----|----|----|----|---------|---------|---------|---------|------|
| baseline-v2.2 |  |  |  |  |  | - | - | - | - | - |
| Exp-01 | ✓ |  |  |  |  | | | | | |
| Exp-02 |  | ✓ |  |  |  | | | | | ✓ |
| Exp-03 |  |  | ✓ |  |  | | | | | |
| Exp-04 |  |  |  | ✓ |  | | | | | |
| Exp-05 |  |  | ✓ | ✓ |  | | | | | |
| Exp-06 |  | ✓ |  |  | ✓ | | | | | ✓ |
| **Exp-07 Full** | ✓ | ✓ | ✓ | ✓ | ✓ | | | | | ✓ |

每格填入 5 次运行的 `mean ± std`。

**核心论文图**：监督比例 vs MAE 曲线（多条线对应不同实验，显示稀疏度越高我们的方法增益越大）

---

## 6. 工程约定

### 6.1 实验记录（每次实验必存）
1. `config_snapshot.json` — 配置快照
2. `model_best.pt` — 最佳模型权重
3. `metrics.json` — 全量指标
4. `predictions.csv` — 测试集逐循环预测
5. `training_log.csv` — 逐 epoch 损失曲线
6. `seed.txt` — 随机种子
7. `supervision_mask.npz` — 该次运行的监督 mask（用于复现）

### 6.2 避坑清单
- [ ] M5 启用时，其他固定权重必须全部关闭
- [ ] M6 启用时，必须先有 M2 的 Dropout 层
- [ ] 所有实验用同一份数据划分（在 split_batteries 使用相同 seed）
- [ ] 每次改动推模块前，先跑一次"空改动"回归测试
- [ ] 部分监督的 mask 一旦确定，5 次重复运行应使用**相同**的 mask（但模型初始化种子不同）

---

## 7. 风险预案

| 风险 | 表现 | 应对 |
|------|------|------|
| M6 伪标签漂移 | 训练后期误差反弹 | 限制伪标签数量上限；每次重置后重新筛选 |
| M5 权重塌缩 | 某个 σᵢ 趋近于 0 | 对 log σᵢ 加 L2 正则 |
| M1 注意力过拟合 | 训练稳定但测试退化 | Attention Dropout + 减少 num_heads |
| 多模块组合冲突 | Exp-07 不如 Exp-06 | 回退到两两组合消融找冲突源 |
| 物理约束在全监督下不提升 | Stage 0 发现 | 说明物理约束主要在部分监督下有价值——这也是论文的一个卖点 |

---

## 8. 备用创新点（未采纳，作为可选扩展）

- **C1：退化阶段自感知约束激活**——通过拐点检测动态激活单调约束，替代固定循环阈值
- **C2：跨电池群体一致性协同约束**——batch 内相似电池对的 SOH 差值加惩罚
- **B2：课程学习**——按监督覆盖率排序，由易到难训练
- **B3：物理引导数据增强**——单调保序回归修正后的扰动增强
- **B5：Transformer 替代 LSTM**——工作量较大，性价比低

---

## 9. 下一步行动（给未来自己的提示）

如果这份计划在新对话中被加载：

1. **先确认当前进度**：查看 `experiments/` 目录，看已完成到哪个 Stage
2. **下一个要做的事**：
   - 如果尚未开始 → **Stage 0 基线锁定**
   - 如果 Stage 0 已完成 → 开始 Stage 1（M1 注意力）
   - 如果在 Stage X → 查看对应 Exp 的配置和代码进度
3. **Stage 0 的子步骤**（见第 4 节）：
   - Step 0.2: 实现 label masking 部分监督
   - Step 0.3: 启用物理约束（含改 boundary_weight）
   - Step 0.4: 烟雾测试
   - Step 0.5: 正式跑 20 次（5 seeds × 4 ratios）
   - Step 0.6: Commit & 固化

---

## 10. 参考文献线索（9 篇 PDF 位于 `paper/` 目录，已 .gitignore）

- `1-s2.0-S0360544225028579-main.pdf` — 论文 5：TS-PINN 同方差不确定性权重（M5 来源）
- `1-s2.0-S0951832025006325-main.pdf` — 论文 4：Bayesian PINN / MC Dropout（M2 来源）
- `s41598-026-37850-y.pdf` — 论文 8：部分可观测场景（M8 来源）
- 其他：双网络 F+G 结构（M3 借鉴 / C1 类似思路）
- 详细笔记：`paper/paper1_text.txt`, `paper2_text.txt`, `paper3_text.txt`, `chapter4_content.txt`

---

## 11. 变更日志

### 2026-04-17（Stage 0 启动前对齐）
- 基线改名：V6 → **baseline-v2.2**
- 数据划分：80/20 → **60/20/20**
- 新增"1. 基线协议"章节（明确数据、模型、物理约束、部分监督细节）
- 确认部分监督实现方式：**Label Masking**（不是 Sample Dropping）
- 新增"🔴 关键现实确认"段落，明确当前代码缺失 `utils/data_augmentation`
- Stage 0 从 0.5 天拆分为 6 个子步骤、2 天
- 新增"监督比例维度"到消融矩阵（每个实验在 4 个比例下都跑）
- 新增"核心论文图"的描述

---

**文档维护者**：liuchang2262@gmail.com
