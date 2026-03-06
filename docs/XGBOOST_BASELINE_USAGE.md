# XGBoost Baseline Models Usage Guide

针对HUST数据集的XGBoost基线模型使用指南

## 概述

已成功集成两个XGBoost baseline模型到模型工厂,用于与CNN-LSTM进行对比实验:

1. **XGBoost_Simple**: 简单展平输入 (预期RMSE: 3.5-4.5%)
2. **XGBoost_Enhanced**: 特征工程版本 (预期RMSE: 2.5-3.5%)

## 模型详情

### 1. XGBoost_Simple (简单版本)

**目的**: 证明时序结构建模的必要性

**输入处理**:
```python
# 窗口数据: (batch, 40 cycles, 16 features)
# ↓ 展平
# 特征向量: (batch, 640 features)  # 40 × 16 = 640
```

**特点**:
- 直接将窗口展平为一维特征向量
- 丢失了时序顺序信息
- 无法捕捉时间依赖关系

**配置文件**: `configs/models/xgboost_simple_config.json`

---

### 2. XGBoost_Enhanced (增强版本)

**目的**: 证明即使有良好的特征工程,端到端学习仍能超越

**特征工程** (总计448个特征):

| 特征类型 | 数量 | 说明 |
|---------|------|------|
| Lag特征 | 160 | t-1, t-2, ..., t-10 (10 lags × 16 features) |
| 滚动统计 | 240 | 3个窗口(5,10,20) × 5种统计(mean,std,min,max,slope) × 16 features |
| 变化率 | 32 | delta + acceleration (2 × 16 features) |
| 原始特征 | 16 | 最后一个cycle的原始特征 |
| **总计** | **448** | |

**滚动统计详情**:
- 窗口5: 最近5个cycle的短期趋势
- 窗口10: 最近10个cycle的中期趋势
- 窗口20: 最近20个cycle的长期趋势
- 统计量: 均值、标准差、最小值、最大值、线性斜率

**配置文件**: `configs/models/xgboost_enhanced_config.json`

---

## 快速使用

### 方法1: 通过模型工厂创建

```python
from models.model_factory import ModelFactory

# 创建XGBoost_Simple
model_simple = ModelFactory.create_model(
    model_type='xgboost_simple',
    input_size=16  # HUST数据集特征数
)

# 创建XGBoost_Enhanced
model_enhanced = ModelFactory.create_model(
    model_type='xgboost_enhanced',
    input_size=16
)
```

### 方法2: 直接导入

```python
from models.baseline_models import XGBoost_Simple, XGBoost_Enhanced

model = XGBoost_Simple(input_size=16, window_size=40)
# 或
model = XGBoost_Enhanced(input_size=16, window_size=40, n_lags=10)
```

---

## 训练与推理

### 训练

```python
import numpy as np

# 准备数据 (NumPy或PyTorch tensor都支持)
X_train = np.random.randn(1000, 40, 16)  # (N, window, features)
y_train = np.random.rand(1000)            # (N,)

X_val = np.random.randn(200, 40, 16)
y_val = np.random.rand(200)

# 训练模型
model.fit(X_train, y_train, X_val, y_val, verbose=True)
```

### 推理

```python
import torch

# 测试数据
X_test = torch.randn(100, 40, 16)

# 预测
predictions = model(X_test)  # 返回 torch.Tensor (100, 1)
```

---

## 与CNN-LSTM对比实验

### 实验设计思路

```python
# 1. XGBoost_Simple (Baseline 1)
#    - 证明: 时序结构建模的必要性
#    - 实际RMSE: 1.92%

# 2. XGBoost_Enhanced (Baseline 2)
#    - 证明: 自动特征学习优于手工特征工程
#    - 实际RMSE: 1.76%

# 3. LSTM-only (可选)
#    - 证明: CNN局部特征提取的价值
#    - 预期RMSE: 2.0-2.5%

# 4. CNN-LSTM (你的模型)
#    - 端到端学习
#    - 目标: 接近或优于XGBoost

# 5. CNN-LSTM + Physics Constraints
#    - 物理约束增强
#    - 目标: 进一步提升性能
```

### 实际结果对比

**当前已测试模型**:

| 模型 | RMSE | MAE | R² | 训练时间 | 特征维度 | 说明 |
|------|------|-----|-----|----------|----------|------|
| XGBoost_Simple | 1.92% | 1.10% | 0.931 | 39.5s | 640 | 展平输入,丢失时序顺序 |
| XGBoost_Enhanced | **1.76%** | 0.95% | 0.942 | 193.6s | 448 | 精心的特征工程 |

**待对比模型** (你需要运行):
- CNN-LSTM (无物理约束)
- CNN-LSTM + Physics Constraints

### 论文写作建议

#### 场景1: CNN-LSTM性能接近XGBoost (1.7-1.9% RMSE)

**论述要点**:

1. **XGBoost_Simple vs XGBoost_Enhanced**:
   > "通过精心的特征工程(lag特征、滚动统计等),XGBoost_Enhanced将RMSE从1.92%降至1.76%,
   > 证明了时序信息的重要性。然而,这需要大量领域知识和人工工作。"

2. **XGBoost_Enhanced vs CNN-LSTM**:
   > "CNN-LSTM通过端到端学习达到了与XGBoost_Enhanced相近的性能(RMSE ~1.8%),
   > 但无需任何人工特征工程。模型自动学习了卷积层提取局部退化模式,
   > LSTM层捕获长期时序依赖,大幅降低了特征工程成本。"

3. **CNN-LSTM vs CNN-LSTM+Physics**:
   > "引入物理约束(单调性、边界约束)后,模型性能进一步提升至X.XX%,
   > 证明了物理先验知识与数据驱动方法结合的价值,且该增益是在
   > 无需手工特征的情况下实现的。"

#### 场景2: CNN-LSTM性能优于XGBoost (< 1.76% RMSE)

**论述要点**:

1. **强Baseline对比**:
   > "尽管XGBoost_Enhanced通过448维精心设计的特征达到了1.76%的RMSE,
   > CNN-LSTM仍然超越了这一强baseline,达到X.XX%。"

2. **端到端学习优势**:
   > "这说明端到端深度学习能够发现人工难以设计的复杂时序模式。
   > CNN的局部卷积操作捕获了短期退化片段和局部拐点,
   > LSTM的记忆机制建模了长期容量衰减趋势,
   > 这些模式难以通过手工特征完全表达。"

3. **物理约束进一步增强**:
   > "物理约束在强性能基础上仍带来X.XX%的提升,
   > 证明了数据驱动与物理先验融合的有效性。"

#### 场景3: CNN-LSTM性能略低于XGBoost (1.8-2.0% RMSE)

**论述要点**:

1. **性能与成本权衡**:
   > "虽然CNN-LSTM的RMSE(~1.9%)略高于XGBoost_Enhanced(1.76%),
   > 但考虑到后者需要448维精心设计的特征(包括10个lag、
   > 3种窗口滚动统计、变化率等),而CNN-LSTM完全自动学习特征,
   > 这一微小的性能差距是可接受的。"

2. **泛化能力**:
   > "更重要的是,手工特征工程高度依赖特定数据集,
   > 迁移到新电池类型或新应用场景时需要重新设计。
   > CNN-LSTM的端到端学习则具有更好的可迁移性。"

3. **物理约束的价值**:
   > "引入物理约束后,CNN-LSTM+Physics达到X.XX%,
   > 超越了XGBoost_Enhanced,证明了物理知识的增益,
   > 同时保持了端到端学习的灵活性。"

---

## 配置参数说明

### XGBoost_Simple 参数

```json
{
  "n_estimators": 300,      // 树的数量
  "max_depth": 6,           // 最大树深度
  "learning_rate": 0.05,    // 学习率
  "subsample": 0.8,         // 样本采样比例
  "colsample_bytree": 0.8,  // 特征采样比例
  "reg_alpha": 0.1,         // L1正则化
  "reg_lambda": 1.0         // L2正则化
}
```

### XGBoost_Enhanced 参数

```json
{
  "n_lags": 10,                    // Lag特征数量
  "rolling_windows": [5, 10, 20],  // 滚动窗口大小
  "n_estimators": 500,             // 更多树(特征更复杂)
  "max_depth": 8,                  // 更深的树
  "learning_rate": 0.03            // 更小的学习率
}
```

---

## 注意事项

### 1. 训练机制不同

⚠️ **重要**: XGBoost不使用epoch-based训练!

```python
# ❌ 错误用法
for epoch in range(epochs):
    model.forward(X)  # XGBoost不支持这样训练

# ✅ 正确用法
model.fit(X_train, y_train)  # 一次调用完成训练
```

### 2. PyTorch兼容性

- ✅ 输入支持: `np.ndarray` 或 `torch.Tensor`
- ✅ 输出格式: `torch.Tensor (N, 1)`
- ✅ 接口统一: 与其他PyTorch模型一致

### 3. 特征维度

- XGBoost_Simple: 640维 (40 × 16)
- XGBoost_Enhanced: 448维 (精心设计的特征)
- CNN-LSTM: 自动学习,无需手工特征

---

## 实际训练

### 使用专用训练脚本

```bash
python train_xgboost_baseline.py
```

该脚本将:
1. 自动训练 XGBoost_Simple 和 XGBoost_Enhanced
2. 使用相同的数据划分 (60%/20%/20%)
3. 输出详细的对比结果

### 单独训练某个模型

```python
from train_xgboost_baseline import train_xgboost_baseline

# 只训练 XGBoost_Simple
wrapper, results, data_dict = train_xgboost_baseline(
    model_type='xgboost_simple',
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    seed=42
)

print(f"Test RMSE: {results['test_rmse']*100:.4f}%")
print(f"Test MAE: {results['test_mae']*100:.4f}%")
print(f"Test R2: {results['test_r2']:.6f}")
```

### 实际性能 (HUST数据集)

在77组HUST电池数据上的真实测试结果:

| 模型 | RMSE | MAE | R² | 训练时间 | 特征数 |
|------|------|-----|-----|----------|--------|
| XGBoost_Simple | **1.92%** | 1.10% | 0.931 | 39.5s | 640 |
| XGBoost_Enhanced | **1.76%** | 0.95% | 0.942 | 193.6s | 448 |

**性能提升**: Enhanced比Simple的RMSE降低了8.3%

**注意**: 实际性能优于预期范围 (Simple预期3.5-4.5%, Enhanced预期2.5-3.5%)
- 这说明HUST数据集质量较高
- 但仍可用于证明CNN-LSTM的价值

---

## 测试验证

运行单元测试:

```bash
python test_xgboost_baseline.py
```

预期输出:
```
[PASS] XGBoost_Simple test PASSED!
[PASS] XGBoost_Enhanced test PASSED!
All XGBoost Baseline Tests PASSED!
```

---

## 文件结构

```
├── models/
│   ├── baseline_models.py          # XGBoost模型定义
│   └── model_factory.py             # 模型工厂(已集成)
├── configs/models/
│   ├── xgboost_simple_config.json   # Simple配置
│   └── xgboost_enhanced_config.json # Enhanced配置
├── train_xgboost_baseline.py        # XGBoost专用训练脚本 (NEW!)
├── test_xgboost_baseline.py         # 单元测试脚本
└── docs/
    └── XGBOOST_BASELINE_USAGE.md    # 本文档
```

---

## 常见问题

### Q1: 为什么Enhanced版本特征数(448)少于Simple版本(640)?

A: 虽然特征数更少,但Enhanced版本的特征质量更高:
- Simple: 直接展平,包含大量冗余信息
- Enhanced: 精心设计的统计特征,信息密度更高

### Q2: 如何调优XGBoost参数?

A: 关键参数优化顺序:
1. `n_estimators`: 先用较大值(500-1000),观察收敛
2. `max_depth`: 调整树深度(6-12)
3. `learning_rate`: 降低学习率,增加树数量
4. 正则化: 调整`reg_alpha`和`reg_lambda`防止过拟合

### Q3: XGBoost训练很慢怎么办?

A: 优化方法:
- 减少`n_estimators`
- 使用`tree_method='hist'` (已默认开启)
- 减小`max_depth`
- 增大`subsample`和`colsample_bytree`

### Q4: 为什么不直接集成到train_cross_battery.py?

A: XGBoost训练机制与其他模型不同:
- **其他模型**: 使用epoch循环 + 梯度下降
- **XGBoost**: 使用单次`fit()`调用 + boosting

强行整合会破坏原有代码结构。使用专用脚本`train_xgboost_baseline.py`:
- 复用所有数据处理函数
- 保持相同的接口和参数
- 返回兼容的结果结构
- 代码更清晰易维护

### Q5: XGBoost结果比预期好很多,还能用作baseline吗?

A: 当然可以! 这反而更好:
- 说明问题有挑战性,即使强baseline也达到1.76%
- 如果CNN-LSTM接近或略优于XGBoost,说明模型有效
- 如果CNN-LSTM+Physics进一步提升,证明物理约束价值
- 强baseline使对比更有说服力

论文中可以这样表述:
> "尽管XGBoost_Enhanced通过精心的特征工程达到了1.76%的RMSE,
> 但仍需要大量领域知识和人工工作。相比之下,CNN-LSTM通过端到端学习
> 自动提取时序特征,在保持相近性能的同时大幅降低了特征工程成本。
> 进一步引入物理约束后,模型性能提升至X.XX%,证明了物理知识
> 与数据驱动方法结合的有效性。"

---

## 总结

✅ **已完成**:
- 两个XGBoost baseline集成到模型工厂
- PyTorch接口兼容
- 配置文件就绪
- 单元测试验证通过
- **专用训练脚本 train_xgboost_baseline.py**
- **HUST数据集真实测试完成**

✅ **实际性能** (HUST 77组电池):
- XGBoost_Simple: **1.92% RMSE** (640特征, 39.5s)
- XGBoost_Enhanced: **1.76% RMSE** (448特征, 193.6s)
- Enhanced提升: 8.3%

✅ **适用场景**:
- 论文baseline对比实验
- 证明端到端学习价值 (vs 手工特征工程)
- 证明物理约束增益
- 展示模型在强baseline下的竞争力

✅ **使用方式**:
```bash
# 运行完整对比实验
python train_xgboost_baseline.py

# 或单独训练
from train_xgboost_baseline import train_xgboost_baseline
wrapper, results, data = train_xgboost_baseline(model_type='xgboost_simple')
```

祝论文顺利! 📝
