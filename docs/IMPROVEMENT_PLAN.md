# PI-CNNLSTM 模块化改进提优计划

> **项目**: 基于物理一致性约束的电池 SOH 估计（PI-CNNLSTM）
> **数据集**: HUST 77 块电池，14 维特征
> **基线**: V6（软单调 + 边界 + 部分生命周期监督）
> **创建日期**: 2026-04-17
> **状态**: 待执行
> **用途**: 硕士论文第四章 / 潜在期刊投稿

---

## 0. 背景摘要（给未来的自己/Claude）

本项目是一个电池健康状态（SOH）估计任务，核心场景是**部分生命周期监督**（Partial Lifecycle Supervision）——即只有电池生命周期的部分区段有 SOH 标签，模型需要在监督稀疏的条件下泛化到完整生命周期。

当前 V6 基线已实现：
- CNN-LSTM 主干
- 软单调性约束（SOH 不应上升）
- 边界约束
- 部分监督的掩码 loss

**本计划的目标**：在 V6 基础上增加一系列模块化改进，逐模块消融验证，最终形成完整的论文工作。

### 推荐的最终组合（核心叙事）
```
核心架构：CNN-LSTM + 循环级注意力（M1）
损失函数：自适应权重（M5）+ 软单调 + 边界 + 速率连续性约束（M4）
训练策略：不确定性引导伪标签扩展部分监督（M6）
评估：MC Dropout 置信区间（M2+M7）+ 特征缺失鲁棒性（M8）
```

论文故事线："**在部分监督场景下，不仅引入物理约束，还通过不确定性引导主动利用了未标注区间的信息，同时用自适应权重替代人工调参的黑箱超参数**"——这个组合在 2025/2026 年的论文中基本没有人完整讲过。

---

## 1. 模块化设计原则

- **可插拔**：每个改进点独立封装为一个模块，通过配置开关（config flag）控制启用
- **可组合**：模块间通过清晰接口通信，支持任意组合消融
- **可复现**：每次实验固定 seed，保留完整配置快照
- **可比较**：统一训练脚本 + 统一评估接口，仅改配置即可切换实验

---

## 2. 推荐代码组织结构

```
PI-CNNLSTM/
├── configs/                          # 实验配置中心
│   ├── base.yaml                     # 基线（当前V6）
│   ├── exp01_attention.yaml          # +注意力
│   ├── exp02_adaptive_loss.yaml      # +自适应权重
│   ├── exp03_rate_smooth.yaml        # +速率连续性
│   ├── exp04_mc_dropout.yaml         # +不确定性量化
│   ├── exp05_pseudo_label.yaml       # +伪标签扩展
│   └── exp06_full.yaml               # 全模块组合
│
├── models/
│   ├── backbone.py                   # 核心 CNN-LSTM
│   ├── modules/
│   │   ├── attention.py              # M1: 注意力模块
│   │   ├── mc_dropout.py             # M2: MC Dropout包装
│   │   └── instance_norm.py          # M3: 个体归一化（可选）
│   └── pi_cnnlstm.py                 # 顶层装配器
│
├── losses/
│   ├── base_loss.py                  # 基础MSE监督损失
│   ├── physics/
│   │   ├── monotonicity.py           # 现有：软单调约束
│   │   ├── boundary.py               # 现有：边界约束
│   │   └── rate_smoothness.py        # M4: 新增速率连续性
│   ├── adaptive_weight.py            # M5: 不确定性权重
│   └── composite_loss.py             # 损失装配器
│
├── training/
│   ├── trainer.py                    # 统一训练循环
│   ├── partial_supervision.py        # 部分监督掩码管理
│   ├── pseudo_labeling.py            # M6: 不确定性伪标签
│   └── callbacks.py                  # 日志/早停/保存
│
├── evaluation/
│   ├── metrics.py                    # MAE/RMSE/MAPE
│   ├── uncertainty_eval.py           # M7: 置信区间评估
│   ├── robustness_eval.py            # M8: 特征缺失鲁棒性
│   └── ablation_runner.py            # 自动消融运行器
│
├── experiments/                       # 实验产物
│   └── {exp_name}_{timestamp}/
│       ├── config.yaml
│       ├── model_best.pt
│       ├── metrics.json
│       └── predictions.csv
│
└── scripts/
    ├── run_single.py
    ├── run_ablation.py
    └── compare_results.py
```

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

### Stage 0：基线锁定（0.5 天）
- [ ] 冻结当前 V6 模型作为 **Baseline-V6**
- [ ] 定义统一评估协议：
  - 固定 80/20 电池划分（种子 42）
  - 部分监督比例矩阵：`[0.3, 0.5, 0.7, 1.0]`
  - 5 次独立运行取 mean ± std
- [ ] 输出基线指标表
- [ ] 冻结 `configs/base.yaml`
- **交付**：`experiments/baseline_v6/metrics.json`

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

**总预计工期：10 天**

---

## 5. 消融矩阵（最终论文表格）

| 实验 | M1 | M2 | M4 | M5 | M6 | MAE | RMSE | PICP |
|------|----|----|----|----|----|-----|------|------|
| Baseline-V6 |  |  |  |  |  | - | - | - |
| Exp-01 | ✓ |  |  |  |  | | | |
| Exp-02 |  | ✓ |  |  |  | | | ✓ |
| Exp-03 |  |  | ✓ |  |  | | | |
| Exp-04 |  |  |  | ✓ |  | | | |
| Exp-05 |  |  | ✓ | ✓ |  | | | |
| Exp-06 |  | ✓ |  |  | ✓ | | | ✓ |
| **Exp-07 Full** | ✓ | ✓ | ✓ | ✓ | ✓ | | | ✓ |

每格填入 5 次运行的 `mean ± std`。

---

## 6. 配置文件模板

### `configs/base.yaml`（基线）
```yaml
model:
  cnn_channels: [32, 64]
  lstm_hidden: 64
  dropout: 0.3
  use_attention: false        # M1 开关
  use_instance_norm: false    # M3 开关

loss:
  supervised:
    enabled: true
    weight: 1.0
  monotonicity:
    enabled: true
    weight: 0.1
  boundary:
    enabled: true
    weight: 0.05
  rate_smooth:                # M4
    enabled: false
    weight: 0.05
  adaptive_weight:            # M5
    enabled: false

training:
  epochs: 100
  batch_size: 32
  warmup_epochs: 30
  pseudo_label:               # M6
    enabled: false
    tau: 0.01
    update_every: 5

evaluation:
  mc_dropout:                 # M2 + M7
    enabled: false
    n_samples: 50
  robustness:                 # M8
    enabled: false

data:
  split_file: data/splits/split_v1.json
  seed: 42
```

---

## 7. 工程约定

### 7.1 接口规范
- **模型前向**：`forward(x) -> dict(pred=..., features=..., attn_weights=...)`
- **损失函数**：`loss_fn(pred_dict, target, mask) -> dict(total=..., sup=..., mono=..., ...)`
- **评估函数**：`evaluate(model, dataloader, cfg) -> dict(mae=..., rmse=..., picp=..., ...)`

### 7.2 实验记录（每次实验必存）
1. `config.yaml` — 配置快照
2. `model_best.pt` — 最佳模型权重
3. `metrics.json` — 全量指标
4. `predictions.csv` — 测试集逐循环预测
5. `training_log.csv` — 逐 epoch 损失曲线
6. `seed.txt` — 随机种子

### 7.3 避坑清单
- [ ] M5 启用时，其他固定权重必须全部关闭
- [ ] M6 启用时，必须先有 M2 的 Dropout 层
- [ ] 所有实验用同一份数据划分（`data/splits/split_v1.json`）
- [ ] 每次改动推模块前，先跑一次"空改动"回归测试

---

## 8. 风险预案

| 风险 | 表现 | 应对 |
|------|------|------|
| M6 伪标签漂移 | 训练后期误差反弹 | 限制伪标签数量上限；每次重置后重新筛选 |
| M5 权重塌缩 | 某个 σᵢ 趋近于 0 | 对 log σᵢ 加 L2 正则 |
| M1 注意力过拟合 | 训练稳定但测试退化 | Attention Dropout + 减少 num_heads |
| 多模块组合冲突 | Exp-07 不如 Exp-06 | 回退到两两组合消融找冲突源 |

---

## 9. 备用创新点（未采纳，作为可选扩展）

- **C1：退化阶段自感知约束激活**——通过拐点检测动态激活单调约束，替代固定循环阈值
- **C2：跨电池群体一致性协同约束**——batch 内相似电池对的 SOH 差值加惩罚
- **B2：课程学习**——按监督覆盖率排序，由易到难训练
- **B3：物理引导数据增强**——单调保序回归修正后的扰动增强
- **B5：Transformer 替代 LSTM**——工作量较大，性价比低

---

## 10. 下一步行动（给未来自己的提示）

如果这份计划在新对话中被加载：

1. **先确认当前进度**：查看 `experiments/` 目录，看已完成到哪个 Stage
2. **下一个要做的事**：
   - 如果尚未开始 → **Stage 0 基线锁定**：把当前 `train_cross_battery.py` 重构为模块化骨架
   - 如果在 Stage X → 查看对应 Exp 的配置和代码进度
3. **建议的第一个实现模块**：**M2（MC Dropout）+ M7（置信区间评估）**——零成本、立即出图、论文效果直观

---

## 11. 参考文献线索（9 篇 PDF 位于 `paper/` 目录）

- `1-s2.0-S0360544225028579-main.pdf` — 论文 5：TS-PINN 同方差不确定性权重（M5 来源）
- `1-s2.0-S0951832025006325-main.pdf` — 论文 4：Bayesian PINN / MC Dropout（M2 来源）
- `s41598-026-37850-y.pdf` — 论文 8：部分可观测场景（M8 来源）
- 其他：双网络 F+G 结构（M3 借鉴 / C1 类似思路）
- 详细笔记：`paper/paper1_text.txt`, `paper2_text.txt`, `paper3_text.txt`, `chapter4_content.txt`

---

**文档维护者**：liuchang2262@gmail.com
**最后更新**：2026-04-17
