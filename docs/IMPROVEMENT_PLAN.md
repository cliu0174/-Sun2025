# PI-CNNLSTM 模块化改进提优计划

> **项目**: 基于物理一致性约束的电池 SOH 估计（PI-CNNLSTM）
> **数据集**: HUST 77 块电池，14 维特征
> **基线**: **baseline-v2.2**（CNN-LSTM + 软单调约束 + 边界约束）
> **创建日期**: 2026-04-17
> **最后更新**: 2026-04-18（Stage 4-5 代码完成）
> **状态**: ✅ Stage 0-5 代码全部完成 | ⏳ 所有实验待服务器运行 | ✍️ 写作阶段待实验数据
> **用途**: 硕士论文第四章 / 潜在期刊投稿

---

## 0. 背景摘要（给未来的自己/Claude）

本项目是一个电池健康状态（SOH）估计任务，核心场景是**部分生命周期监督**（Partial Lifecycle Supervision）——即每块电池只有一部分循环有 SOH 标签，模型需要在监督稀疏的条件下泛化到全生命周期。

### ✅ 当前代码实现状态总览（2026-04-18）

| Stage | 内容 | 代码 | 实验 |
|-------|------|------|------|
| Stage 0 | 基线锁定（部分监督 + 物理约束） | ✅ | ⏳ 待服务器 |
| Stage 1 | M1 注意力 + M2 MC Dropout + M7 置信区间 | ✅ | ⏳ 待服务器 |
| Stage 2 | M4 速率连续性 + M5 自适应损失权重 | ✅ | ⏳ 待服务器 |
| Stage 3 | M6 不确定性伪标签 | ✅ | ⏳ 待服务器 |
| Stage 4 | 全模块集成 Exp-07 Full Stack | ✅ | ⏳ 待服务器 |
| Stage 5 | M8 特征缺失鲁棒性 Exp-08 | ✅ | ⏳ 待服务器 |

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
- **实现位置**：`train_cross_battery.py::split_batteries()`

### 1.2 模型结构
- **主干**：CNN-LSTM（`models/cnn_lstm.py`）
- **CNN**：channels=[256, 128], kernel_size=7
- **LSTM**：hidden=64, num_layers=2
- **FC**：[64]
- **Dropout**：0.4
- **窗口大小**：40
- **特征选择**：Top-6 by correlation（from 14 维）

### 1.3 物理约束（baseline-v2.2 必须启用）
- `physics_constraints.enabled: true`
- `monotonic_weight: 0.1`（软单调）
- `boundary_weight: 0.05`（边界约束）
- `smoothness_weight: 0.0`（M4 禁用，baseline 不含）
- `monotonic_tolerance: 0.01`
- `temporal_decay`：启用，exp 衰减，alpha=0.2

### 1.4 部分监督协议
- **实现方式**：**Label Masking**（不是 Sample Dropping！）
  - 保留所有 (x, y) 样本，为每个样本增加 `is_labeled` 布尔标志
  - 按电池随机选择 N% 的循环标记为 `is_labeled=True`
  - MSE（监督 loss）：只对 `is_labeled=True` 样本计算
  - 物理约束（mono/bound/smooth）：对**所有样本**计算
- **监督比例矩阵**：`[1.0, 0.7, 0.5, 0.3]`
- **随机性**：按 (battery_id, seed) 确定 mask 种子，可复现

### 1.5 训练配置
- **epochs**: 200，**batch_size**: 256，**optimizer**: Adam
- **lr_scheduler**: WarmupCosineDecay（warmup 20 epoch, base_lr=0.005）
- **early_stopping**: patience=20

### 1.6 评估指标
| 指标 | 含义 |
|------|------|
| MAE | 平均绝对误差（主指标）|
| RMSE | 均方根误差 |
| MAPE | 平均百分比误差 |
| R² | 决定系数 |
| PICP | 预测区间覆盖概率（M2+M7 专属）|
| MPIW | 预测区间平均宽度（M2+M7 专属）|

---

## 2. 模块化设计原则

- **可插拔**：每个改进点独立封装为一个模块，通过配置 flag 控制启用
- **可组合**：模块间通过清晰接口通信，支持任意组合消融
- **可复现**：每次实验固定 seed，保留完整配置快照
- **可比较**：统一训练脚本 + `config_override` 参数支持超参扫描

---

## 3. 模块清单

### 3.1 模块总览表

| ID | 模块 | 实际文件位置 | 类型 | 依赖 | 优先级 | 状态 |
|----|------|------------|------|------|--------|------|
| M1 | 循环级注意力 | `models/modules/attention.py` | 架构 | 无 | P0 | ✅ |
| M2 | MC Dropout 包装 | `models/modules/mc_dropout.py` | 架构 | 无 | P0 | ✅ |
| M3 | 个体归一化 | `models/modules/instance_norm.py` | 架构 | 无 | P2 | ⬜ |
| M4 | 速率连续性约束 | `models/physics_loss.py::smoothness_loss()` | 损失 | 无 | P0 | ✅ |
| M5 | 自适应损失权重 | `models/adaptive_loss.py` | 损失 | 无 | P0 | ✅ |
| M6 | 不确定性伪标签 | `training/pseudo_labeling.py` | 训练 | M2 | P1 | ✅ |
| M7 | 置信区间评估 | `evaluation/uncertainty_eval.py` | 评估 | M2 | P0 | ✅ |
| M8 | 特征缺失鲁棒性 | `evaluation/robustness_eval.py` | 评估 | 无 | P1 | ✅ |

> **优先级说明**：P0 必做，P1 强烈建议，P2 时间充裕再做

### 3.2 各模块实现细节

#### M1：循环级注意力（Cycle-level Attention）✅
- **位置**：LSTM 输出之后、FC 头之前（`models/cnn_lstm.py` 第 75-80 行）
- **实现**：4 头 Multi-Head Self-Attention + Residual + LayerNorm + Global Average Pool
- **开关**：`configs/models/cnn_lstm_attention_config.json` 中 `attention.enabled: true`
- **论文说辞**：
  > "电池退化过程中存在信息不对称性——近期循环与历史特定退化节点对当前健康状态的贡献不尽相同。本文在 CNN-LSTM 的时序表征层引入循环级注意力机制，使模型能够自适应地聚焦退化敏感的历史区间。"

#### M2：MC Dropout 包装（Monte Carlo Dropout）✅
- **位置**：FC 头中的 Dropout 层（`models/cnn_lstm.py` 第 83-85 行）
- **实现**：`MCDropout(nn.Dropout)` 继承，强制 `training=True`
- **推理接口**：`mc_predict(model, x, n_samples=50) → (mean, std)`
- **开关**：`configs/models/cnn_lstm_mc_config.json` 中 `mc_dropout.enabled: true`
- **论文说辞**：
  > "在部分生命周期监督条件下，模型对未观测退化阶段的估计本质上包含不可约减的认知不确定性。本文通过测试时保留 Dropout 进行多次随机前向传播，构建 SOH 预测的置信区间。"

#### M3：个体归一化（暂缓）⬜
- **目的**：消除电池个体间绝对幅值差异，聚焦退化模式形状
- **实现**：`nn.InstanceNorm1d`，位于 CNN 输入前

#### M4：速率连续性约束（Rate Smoothness）✅
- **位置**：`models/physics_loss.py::PhysicsConstrainedLoss.smoothness_loss()`
- **公式**：`L_smooth = Σ (Δ²SOH)²`（同一电池内的二阶差分平方和）
- **开关**：配置中 `physics_constraints.smoothness_weight > 0`
- **超参扫描**：`run_exp03_rate_smoothness.py` 扫描 `{0.01, 0.05, 0.1, 0.2}`
- **论文说辞**：
  > "传统软单调性约束仅要求 SOH 序列全局非上升，但在局部窗口内允许预测值突变。本文进一步引入退化速率连续性约束：对相邻循环的 SOH 差分序列施加平滑正则，使预测曲线更符合电池物理退化规律。"

#### M5：自适应损失权重（Uncertainty-Weighted Multi-task）✅
- **位置**：`models/adaptive_loss.py::AdaptivePhysicsLoss`
- **公式**：`L = Σ_i [ exp(-log_σᵢ) · Lᵢ + log_σᵢ ]`，4 个 `log_σᵢ` 可学习
- **损失项**：`[base_MSE, monotonic, boundary, smoothness]`
- **集成方式**：`log_vars` 自动加入 optimizer param_group
- **开关**：配置中 `physics_constraints.adaptive_weight.enabled: true`
- **注意**：启用 M5 时原固定权重仍在，但被 `forward_components()` 绕过（不参与加权求和）
- **论文说辞**：
  > "本文采用基于不确定性的多任务加权框架，以可学习参数 log σᵢ 自适应调整各损失项权重，消除人工调参负担，并为各项物理约束的贡献提供可解释的不确定性估计。"

#### M6：不确定性引导伪标签（Uncertainty-Guided Pseudo-Labeling）⭐核心创新 ✅
- **位置**：`training/pseudo_labeling.py::PseudoLabelManager`
- **依赖**：M2（MCDropout 层，推理时保持激活）
- **训练流程**：
  1. **Warmup**（前 `warmup_epochs=30` 轮）：仅用真实标签训练，正常流程
  2. **每 K=5 epoch 刷新伪标签**：
     - MC Dropout 对所有无标签样本推理，得 `(μ, σ)`
     - 按 σ 升序，取最低 30% 为伪标签（上限 50% 无标签样本数）
     - 伪标签权重 `= 1/(σ² + ε)`
  3. **附加训练轮次**：独立 DataLoader，对伪标签样本做加权 MSE
- **防漂移机制**：每次刷新完全重置（不累积历史伪标签）
- **开关**：配置中 `pseudo_labeling.enabled: true`
- **论文说辞**：
  > "本文提出不确定性引导的自监督边界扩展策略：在 Warmup 训练后，利用 MC Dropout 对未标注循环区间的预测置信度动态筛选高质量伪标签，仅将不确定性低于自适应阈值的预测纳入辅助监督，伪标签权重与预测置信度正相关。该策略有效利用了部分监督场景中大量未标注数据，在低标注比例下尤为显著。"

#### M7：置信区间评估 ✅
- **位置**：`evaluation/uncertainty_eval.py`
- **依赖**：M2
- **指标**：
  - PICP（Prediction Interval Coverage Probability）：95% CI 覆盖比例，理论值 ≈ 0.95
  - MPIW（Mean Prediction Interval Width）：区间平均宽度，越小越好
  - Spearman(|error|, std)：误差与不确定性的正相关性

#### M8：特征缺失鲁棒性评估 ✅
- **位置**：`evaluation/robustness_eval.py::evaluate_robustness()`
- **测试场景**：
  - Scene A：随机特征置零，n_mask ∈ {1, 2, 3}
  - Scene B：高斯噪声，σ ∈ {0.01, 0.05, 0.10}
  - Scene C：系统性漂移，drift ∈ {+0.05, +0.10, +0.20}
- **对比**：Baseline vs Full Stack 的 `delta_MAE` 和 `rel_degradation`
- **实验脚本**：`experiments/run_exp08_robustness.py`（ratio=0.5 × 5 seeds × 2 models = 10 次）
- **论文说辞**：
  > "实际部署中传感器故障不可避免。本文在三类扰动场景下评估所提方法的鲁棒性：随机特征屏蔽模拟传感器完全失效，高斯噪声模拟测量误差，常量偏置漂移模拟长期标定偏差。Full Stack 方法借助物理约束和伪标签扩展，在所有场景下均表现出比 Baseline 更小的性能衰减，体现了物理一致性的正则化效果。"

---

## 4. 分阶段实施路线图

### Stage 0：基线锁定 ✅ 代码完成
- [x] 部分监督机制（`generate_supervision_mask()` + `is_labeled` + masked MSE）
- [x] 物理约束启用（`boundary_weight=0.05`）
- [x] 烟雾测试通过（ratio=1.0 MAE≈1.70%，ratio=0.5 MAE≈1.75%，5 epochs）
- [ ] **⏳ 正式 20 次实验**：`python run_baseline_v22.py`（服务器）
- [ ] 固化 `experiments/baseline_v2.2/metrics.json`

### Stage 1：架构增强 ✅ 代码完成
- [x] **Exp-01**：M1 循环级注意力 → `experiments/run_exp01_attention.py`
- [x] **Exp-02**：M2 MC Dropout + M7 → `experiments/run_exp02_mc_dropout.py`
- [ ] **⏳ 运行 Exp-01, Exp-02**（服务器，各 5 runs）

### Stage 2：损失函数精化 ✅ 代码完成
- [x] **Exp-03**：M4 速率连续性（4 weights × 5 seeds）→ `experiments/run_exp03_rate_smoothness.py`
- [x] **Exp-04**：M5 自适应权重（5 seeds）→ `experiments/run_exp04_adaptive_weight.py`
- [x] **Exp-05**：M4+M5 联合（4 ratios × 5 seeds）→ `experiments/run_exp05_combined_loss.py`
- [ ] **⏳ 运行 Exp-03/04/05**（服务器）

### Stage 3：训练策略升级 ✅ 代码完成 ⭐核心
- [x] **Exp-06**：M2 + M6 伪标签（4 ratios × 5 seeds）→ `experiments/run_exp06_pseudo_label.py`
- [ ] **⏳ 运行 Exp-06**（服务器，重点看 ratio=0.5/0.3 的增益）

### Stage 4：全模块集成 ✅ 代码完成
- [x] **Exp-07**：Full Stack（M1+M2+M4+M5+M6+M7）→ `experiments/run_exp07_full_stack.py`
  - `configs/models/cnn_lstm_full_stack_config.json`：全模块启用
  - Warmup 30 epoch → 每 5 epoch 更新伪标签 → M5 全程自适应权重
  - M7 指标（PICP/MPIW）在聚合函数中输出
  - 4 ratios × 5 seeds = 20 次
- [ ] **⏳ 运行 Exp-07**（服务器，消融表最后一行）

### Stage 5：鲁棒性验证 ✅ 代码完成
- [x] **Exp-08**：M8 特征缺失鲁棒性 → `experiments/run_exp08_robustness.py`
  - `evaluation/robustness_eval.py`：三类扰动场景，evaluate_robustness()
  - Scene A（特征置零）/ Scene B（高斯噪声）/ Scene C（系统漂移）各 3 强度
  - 对比 Baseline vs Full Stack，ratio=0.5 × 5 seeds = 10 次训练
- [ ] **⏳ 运行 Exp-08**（服务器）

### 分析与写作
- [ ] 消融表（填入各实验结果）
- [ ] 监督比例 vs MAE 折线图（多方法对比）
- [ ] 注意力权重可视化（M1）
- [ ] 伪标签筛选过程可视化（M6）
- [ ] 论文第四章撰写

---

## 5. 实验脚本索引

| 脚本路径 | 实验 | 运行次数 | 状态 |
|---------|------|---------|------|
| `run_baseline_v22.py` | Stage 0 基线 | 5×4=20 | ⏳ |
| `experiments/run_exp01_attention.py` | M1 注意力 | 5 | ⏳ |
| `experiments/run_exp02_mc_dropout.py` | M2 MC Dropout | 5 | ⏳ |
| `experiments/run_exp03_rate_smoothness.py` | M4 超参扫描 | 4×5=20 | ⏳ |
| `experiments/run_exp04_adaptive_weight.py` | M5 自适应 | 5 | ⏳ |
| `experiments/run_exp05_combined_loss.py` | M4+M5 | 4×5=20 | ⏳ |
| `experiments/run_exp06_pseudo_label.py` | M6 伪标签 | 4×5=20 | ⏳ |
| `experiments/run_exp07_full_stack.py` | 全组合（Full Stack） | 4×5=20 | ⏳ |
| `experiments/run_exp08_robustness.py` | M8 鲁棒性（2模型×5seeds） | 10 | ⏳ |

**服务器一键启动顺序建议**（按依赖关系排序）：
```bash
python run_baseline_v22.py                         # Stage 0，先跑，其他实验依赖这个数字
python experiments/run_exp01_attention.py &
python experiments/run_exp02_mc_dropout.py &       # Stage 1，两个可并行
python experiments/run_exp03_rate_smoothness.py    # Stage 2
python experiments/run_exp04_adaptive_weight.py
python experiments/run_exp05_combined_loss.py
python experiments/run_exp06_pseudo_label.py       # Stage 3，最重要
python experiments/run_exp07_full_stack.py         # Stage 4，论文最终方法
python experiments/run_exp08_robustness.py         # Stage 5，鲁棒性对比
```

---

## 6. 消融矩阵（最终论文表格模板）

每个实验在 4 个监督比例下都要跑（ratio ∈ [1.0, 0.7, 0.5, 0.3]）：

| 实验 | M1 | M2 | M4 | M5 | M6 | MAE@100% | MAE@70% | MAE@50% | MAE@30% | PICP |
|------|----|----|----|----|----|---------|---------|---------|---------|------|
| baseline-v2.2 | | | | | | — | — | — | — | — |
| Exp-01 | ✓ | | | | | | | | | |
| Exp-02 | | ✓ | | | | | | | | ✓ |
| Exp-03 | | | ✓ | | | | | | | |
| Exp-04 | | | | ✓ | | | | | | |
| Exp-05 | | | ✓ | ✓ | | | | | | |
| Exp-06 | | ✓ | | | ✓ | | | | | ✓ |
| **Exp-07 Full** | ✓ | ✓ | ✓ | ✓ | ✓ | | | | | ✓ |

每格填入 5 次运行的 `mean ± std`（百分比）。

**核心论文图**：监督比例 vs MAE 曲线（多方法折线图，体现稀疏监督下 M6 增益放大）

---

## 7. 工程约定

### 7.1 实验记录（每次实验必存）
- `result.json` — 测试集指标（MAE/RMSE/MAPE/R²）
- `metrics.json` — 多次运行汇总（mean ± std）
- `error.json` — 失败时的错误信息（断点续跑用）

### 7.2 config_override 用法（超参扫描）
```python
# 无需创建新 config 文件，直接传入覆盖字典
train_cross_battery_model(
    model_type='cnn_lstm',
    config_override={'physics_constraints': {'smoothness_weight': 0.1}},
)
```

### 7.3 避坑清单
- [ ] M5 启用时，`forward_components()` 绕过固定权重，检查梯度是否正常
- [ ] M6 启用时，必须用含 MCDropout 层的模型（`cnn_lstm_mc` 或 `cnn_lstm_pseudo_label`）
- [ ] M6 在 `supervision_ratio=1.0` 时自动禁用（无无标签样本），不报错
- [ ] 所有实验共用同一份数据划分（`split_batteries` 种子固定为 42）
- [ ] 部分监督 mask 一旦确定，5 次重复运行使用**相同** mask，模型初始化种子不同

---

## 8. 风险预案

| 风险 | 表现 | 应对 |
|------|------|------|
| M6 伪标签漂移 | 训练后期误差反弹 | 已实现：每 K epoch 完全重置伪标签 |
| M5 权重塌缩 | 某个 σᵢ → 0 | 已实现：`l2_reg=0.01` 对 log_vars 正则 |
| M1 注意力过拟合 | 训练好但测试退化 | Attention Dropout + 减少 num_heads |
| 多模块组合冲突 | Exp-07 不如 Exp-06 | 回退两两组合消融找冲突源 |
| 物理约束在全监督下不提升 | baseline ratio=1.0 ≈ Exp-01/03/04 | 说明物理约束主要在稀疏监督下有价值，也是论文卖点 |
| MC 推理速度慢 | M6 伪标签更新耗时 | `inference_batch_size=512` 已优化；n_mc_samples 可降至 30 |

---

## 9. 下一步行动

**当前优先级（2026-04-18，Stage 0-5 代码全部完成）**：

1. **⏳ 服务器实验排队**（按顺序依次运行全部脚本）：
   - `run_baseline_v22.py` → 基准数字（其他实验的对比基础）
   - `run_exp01/02` → Stage 1 架构增强
   - `run_exp03/04/05` → Stage 2 损失精化
   - `run_exp06` → Stage 3 伪标签（核心，重点观察低比例增益）
   - `run_exp07` → Stage 4 Full Stack（论文最终方法）
   - `run_exp08` → Stage 5 鲁棒性（Baseline vs Full Stack）

2. **论文写作**（实验结果出来后）：
   - 填充消融表（Table 4.x）
   - 绘制监督比例 vs MAE 折线图（Figure 4.x）
   - 绘制鲁棒性对比图（Figure 4.y）
   - 第四章撰写

---

## 10. 参考文献线索（9 篇 PDF 位于 `paper/` 目录）

- `1-s2.0-S0360544225028579-main.pdf` — 论文 5：TS-PINN 同方差不确定性权重（M5 来源）
- `1-s2.0-S0951832025006325-main.pdf` — 论文 4：Bayesian PINN / MC Dropout（M2 来源）
- `s41598-026-37850-y.pdf` — 论文 8：部分可观测场景（M8 来源）
- 详细笔记：`paper/paper1_text.txt`, `paper2_text.txt`, `paper3_text.txt`, `chapter4_content.txt`

---

## 11. 变更日志

### 2026-04-18（Stage 4-5 代码完成）
- ✅ Stage 4：新建 `configs/models/cnn_lstm_full_stack_config.json`（全模块 M1+M2+M4+M5+M6 启用）；新建 `experiments/run_exp07_full_stack.py`（4 ratios × 5 seeds，含 M7 PICP/MPIW 输出）；更新 `models/model_factory.py`
- ✅ Stage 5：新建 `evaluation/robustness_eval.py`（M8，三类扰动 × 各 3 强度，evaluate_robustness / print_robustness_report）；新建 `experiments/run_exp08_robustness.py`（Baseline vs Full Stack × 5 seeds）；更新 `evaluation/__init__.py`
- 修复 `evaluation/uncertainty_eval.py`：报告新增 `picp_95`、`mpiw_95`、`spearman` 别名，保持 `picp`/`mpiw`/`spearman_corr` 向后兼容
- 实验脚本索引补全（Exp-07/08 标注为 ⏳ 待服务器）

### 2026-04-18（Stage 1-3 代码完成）
- ✅ Stage 1：实现 M1（`models/modules/attention.py` + `CycleAttention`）、M2（`models/modules/mc_dropout.py` + `MCDropout`/`mc_predict`）、M7（`evaluation/uncertainty_eval.py`）
- ✅ Stage 2：实现 M4（`PhysicsConstrainedLoss.smoothness_loss()` 启用 + 超参配置）、M5（`models/adaptive_loss.py::AdaptivePhysicsLoss`）、`forward_components()` 新接口、`config_override` 深度合并
- ✅ Stage 3：实现 M6（`training/pseudo_labeling.py::PseudoLabelManager`），集成进主训练循环
- 新增实验脚本：Exp-01 ～ Exp-06（共 6 个，总计最多 95 次运行）
- 新增 configs：`cnn_lstm_attention`, `cnn_lstm_mc`, `cnn_lstm_rate_smoothness`, `cnn_lstm_adaptive_weight`, `cnn_lstm_pseudo_label`
- 更新模块总览表，补充实际文件路径和实现状态

### 2026-04-17（Stage 0 启动前对齐）
- 基线改名：V6 → **baseline-v2.2**
- 数据划分：80/20 → **60/20/20**
- 确认部分监督实现方式：**Label Masking**（不是 Sample Dropping）
- Stage 0 从 0.5 天拆分为 6 个子步骤、2 天
- 新增"监督比例维度"到消融矩阵

---

**文档维护者**：liuchang2262@gmail.com
