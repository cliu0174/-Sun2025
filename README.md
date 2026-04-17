# PI-CNNLSTM — 基于物理一致性约束的锂电池 SOH 估计

> **论文对应实现**：《基于物理一致性约束的多模态 SOH 融合估计方法》（第四章）  
> **核心场景**：部分生命周期监督（Partial Lifecycle Supervision）——仅部分循环区段有 SOH 标签

---

## 项目概述

本项目在 **HUST 数据集**（77 块电池，14 维特征）上实现了 PI-CNNLSTM 模型，将物理先验知识（单调递减、边界约束）融入损失函数，在标签稀缺的部分监督场景下估计电池健康状态（SOH）。

**核心创新点：**
- **物理约束损失**：软单调性 + 边界约束，适用于无标签样本
- **Label Masking 部分监督**：保留全部样本，随机遮蔽部分标签；无标签样本仍参与物理约束
- **跨电池泛化**：60/20/20 电池级划分，评估跨电池泛化能力

---

## 环境准备

```bash
pip install -r requirements.txt
# 需要 Python 3.8+，PyTorch 2.0+，推荐 CUDA GPU
```

---

## 快速开始

### 1. 单次训练（baseline-v2.2）

```bash
python train_cross_battery.py
```

在 `__main__` 块中调整关键参数：

```python
MODEL_TYPE        = 'cnn_lstm'  # 模型类型
SEED              = 42          # 随机种子
SUPERVISION_RATIO = 1.0         # 监督比例：1.0=全监督，0.5=半监督，0.3=稀疏监督
```

### 2. Stage 0 基线扫描（20次实验）

```bash
python run_baseline_v22.py
```

自动运行 **5 seeds × 4 supervision ratios = 20 次**实验，支持断点续跑，结果汇总到 `experiments/baseline_v2.2/metrics.json`。

### 3. 模型推理

```bash
python inference.py
```

---

## 项目结构

```
1111-soh/
├── train_cross_battery.py      # 主训练脚本（核心入口）
├── run_baseline_v22.py         # Stage 0 基线扫描脚本
├── inference.py                # 模型推理
├── compare_models.py           # 多模型对比
├── feature_extraction.py       # 14 维特征提取逻辑
│
├── models/
│   ├── cnn_lstm.py             # CNN-LSTM 模型定义
│   ├── baseline_models.py      # FNN / CNN / LSTM / GRU 等基线模型
│   ├── model_factory.py        # 模型工厂 + 统一接口
│   └── physics_loss.py         # 物理约束损失函数 ⭐
│
├── data_loaders/
│   └── data_loader_hust.py     # HUST 数据加载 + 窗口化 + supervision mask
│
├── configs/models/             # 各模型 JSON 配置文件
│   └── cnn_lstm_config.json    # 当前基线配置（物理约束参数在此调整）
│
├── utils/
│   └── lr_schedulers.py        # 学习率调度器（WarmupCosineDecay 等）
│
├── data/HUST data/             # 原始数据（77 块电池 .csv，未上传）
├── experiments/                # 实验结果（metrics.json 等）
├── results/                    # 训练输出图表
└── docs/
    └── IMPROVEMENT_PLAN.md     # 完整改进计划（8 个模块，5 个 Stage）⭐
```

---

## 核心模块说明

### 物理约束损失（`models/physics_loss.py`）

```python
PhysicsConstrainedLoss(
    base_loss_weight  = 1.0,   # MSE 权重
    monotonic_weight  = 0.1,   # 软单调性约束
    boundary_weight   = 0.05,  # 边界约束 [0, 1]
    monotonic_tolerance = 0.01 # 允许的微小上升幅度
)
```

支持 `supervision_mask` 参数：**有标签样本**计算 MSE，**无标签样本**只计算物理约束。

### 部分监督机制（Label Masking）

```python
# 在 train_cross_battery.py 的 __main__ 中设置
SUPERVISION_RATIO = 0.5   # 训练集中 50% 的循环有 SOH 标签
SUPERVISION_SEED  = None  # None = 与主 SEED 一致，保证可复现
```

每块电池独立随机采样，确保每块电池都保留至少 1 个标签。验证集和测试集始终保持全监督。

### 模型配置

物理约束参数统一在 `configs/models/cnn_lstm_config.json` 中调整：

```json
"physics_constraints": {
    "enabled": true,
    "monotonic_weight": 0.1,
    "boundary_weight": 0.05
}
```

---

## 实验设计

当前阶段 **Stage 0**：锁定 baseline-v2.2 基准数值

| 实验条件 | 配置 |
|---------|------|
| 模型 | CNN-LSTM（物理约束开启） |
| 数据划分 | 60/20/20（电池级，seed=42） |
| 种子组 | [42, 123, 456, 789, 1024] |
| 监督比例 | [1.0, 0.7, 0.5, 0.3] |
| 总实验数 | 20 次 |

完整的 8 模块改进计划见 [`docs/IMPROVEMENT_PLAN.md`](docs/IMPROVEMENT_PLAN.md)。

---

## 数据集

**HUST 锂离子电池数据集**（不含在仓库中，请自行获取）

- 77 块电池，每块约 100–600 个充放电循环
- 14 维特征：充放电容量、能量、内阻、温度等
- 放置路径：`data/HUST data/*.csv`

---

## 参考文献

Sun, G., Liu, Y., & Liu, X. (2025). A method for estimating lithium-ion battery state of health based on physics-informed machine learning. *Journal of Power Sources*, 627, 235767.

---

## 联系

liuchang2262@gmail.com
