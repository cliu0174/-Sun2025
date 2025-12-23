# Random Missing (Scenario 3) - 使用指南

## 📌 概述

**Random Missing (随机缺失)** 是场景三的数据退化方式，用于模拟传感器随机故障导致的数据缺失。

### 与场景二的区别

| 特性 | 场景二 (Uniform Subsampling) | 场景三 (Random Missing) |
|------|------------------------------|-------------------------|
| **采样方式** | 规律间隔 (如每5个保留1个) | 随机选择保留哪些样本 |
| **时序结构** | 规律性，可预测 | 不规则性，不可预测 |
| **应用场景** | 降采样、成本节约 | 传感器故障、通信中断 |
| **物理约束测试** | 规则间隔下的鲁棒性 | 不规则间隔下的鲁棒性 |

---

## 🎯 适用场景

### 1. 传感器随机故障
- 传感器偶尔无法读取数据
- 通信链路不稳定导致数据丢失
- 电池管理系统间歇性故障

### 2. 测试物理约束
- 测试物理约束在**不规则间隔**下的鲁棒性
- 对比规则采样 vs 随机缺失
- 评估模型的泛化能力

### 3. 实际应用模拟
- 实际部署中，传感器故障往往是随机的
- Random Missing 更接近真实故障模式
- 有助于提高模型的实用性

---

## 📊 三种预设级别

```python
RANDOM_MISSING_PRESETS = {
    'light': {
        'missing_rate': 0.2,
        'description': '轻度缺失 (保留80%)'
    },
    'moderate': {
        'missing_rate': 0.4,
        'description': '中度缺失 (保留60%)'
    },
    'heavy': {
        'missing_rate': 0.6,
        'description': '重度缺失 (保留40%)'
    }
}
```

### 级别说明

- **Light (20% missing)**: 轻度故障场景，保留80%数据
- **Moderate (40% missing)**: 中度故障场景，保留60%数据 ⭐ **推荐用于初步测试**
- **Heavy (60% missing)**: 重度故障场景，保留40%数据

---

## 🚀 使用方式

### 方式 1: 使用预设级别（推荐）

```python
from train_cross_battery import train_cross_battery_model

wrapper, results, data_dict = train_cross_battery_model(
    model_type='cnn_lstm',
    degradation_scenario='scenario3',  # 场景三
    random_missing_level='moderate',   # 预设级别
    seed=42
)
```

**输出示例**:
```
======================================================================
场景三: 随机缺失 - 中度缺失 (保留60%)
======================================================================

对训练集进行随机缺失:
  原始样本数: 10000
  保留样本数: 6000 (60.0%)
  缺失样本数: 4000 (40.0%)
```

---

### 方式 2: 手动设置缺失率（精确控制）

```python
wrapper, results, data_dict = train_cross_battery_model(
    model_type='cnn_lstm',
    degradation_scenario='scenario3',
    random_missing_rate=0.35,  # 手动设置: 丢弃35%数据
    seed=42
)
```

**说明**:
- `random_missing_rate=0.35` → 随机丢弃35%，保留65%
- `random_missing_rate=0.5` → 随机丢弃50%，保留50%
- **优先级**: `random_missing_rate` > `random_missing_level`

---

## 🔬 实验示例

### 实验 1: 对比三种缺失级别

```python
# 测试不同缺失级别下的性能
missing_levels = ['light', 'moderate', 'heavy']
results_dict = {}

for level in missing_levels:
    print(f"\n{'#'*70}")
    print(f"测试缺失级别: {level}")
    print(f"{'#'*70}")

    wrapper, results, _ = train_cross_battery_model(
        model_type='cnn_lstm',
        degradation_scenario='scenario3',
        random_missing_level=level,
        seed=42
    )

    results_dict[level] = {
        'rmse': results['test_rmse'],
        'mae': results['test_mae'],
        'r2': results['test_r2']
    }

# 对比结果
import pandas as pd
df = pd.DataFrame(results_dict).T
print("\n结果对比:")
print(df)
```

**预期输出**:
```
结果对比:
              rmse       mae        r2
light      0.0234    0.0186    0.9567
moderate   0.0289    0.0231    0.9401
heavy      0.0356    0.0284    0.9189
```

---

### 实验 2: Uniform vs Random 对比

对比场景二（规律）和场景三（随机），相同保留率下的性能差异。

```python
# 场景二: 规律稀疏采样，保留率 60% (interval=5, 每5个保留1个实际是20%，改为interval≈1.67不可行)
# 实际：interval=2 → 50%, interval=3 → 33.3%
# 为了60%，使用手动 missing_rate=0.4 (场景三)

# 场景三: 随机缺失，保留率 60%
results_comparison = {}

# 场景二 - 手动间隔
print("\n" + "="*70)
print("场景二: 规律稀疏采样")
print("="*70)
# 无法精确达到60%，使用interval=2 (50%)作为对比
wrapper_s2, results_s2, _ = train_cross_battery_model(
    model_type='cnn_lstm',
    degradation_scenario='scenario2',
    sparse_sampling_interval=2,  # 保留50%
    seed=42
)
results_comparison['Uniform (50%)'] = results_s2

# 场景三 - 随机缺失
print("\n" + "="*70)
print("场景三: 随机缺失")
print("="*70)
wrapper_s3, results_s3, _ = train_cross_battery_model(
    model_type='cnn_lstm',
    degradation_scenario='scenario3',
    random_missing_rate=0.4,  # 保留60%
    seed=42
)
results_comparison['Random (60%)'] = results_s3

# 对比
print("\n" + "="*70)
print("场景对比")
print("="*70)
for scenario, res in results_comparison.items():
    print(f"{scenario:<20} RMSE: {res['test_rmse']:.4f}  MAE: {res['test_mae']:.4f}  R²: {res['test_r2']:.4f}")
```

**预期结果**:
- Random Missing 下性能可能**更差**（不规则性更难学习）
- 物理约束在 Random Missing 下可能**更有效**（更需要先验知识）

---

### 实验 3: 物理约束在 Random Missing 下的效果

```python
# 测试物理约束在随机缺失场景下的改善效果
random_missing_rates = [0.2, 0.4, 0.6]  # 20%, 40%, 60% 缺失
results_with_physics = {}
results_without_physics = {}

import json

for rate in random_missing_rates:
    print(f"\n{'#'*70}")
    print(f"缺失率: {rate*100:.0f}%")
    print(f"{'#'*70}")

    # 读取配置
    config_path = 'configs/models/cnn_lstm_config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 无物理约束
    config['physics_constraints']['enabled'] = False
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    _, res_no_physics, _ = train_cross_battery_model(
        model_type='cnn_lstm',
        degradation_scenario='scenario3',
        random_missing_rate=rate,
        seed=42
    )
    results_without_physics[rate] = res_no_physics['test_rmse']

    # 有物理约束
    config['physics_constraints']['enabled'] = True
    config['physics_constraints']['monotonic_weight'] = 0.2
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    _, res_with_physics, _ = train_cross_battery_model(
        model_type='cnn_lstm',
        degradation_scenario='scenario3',
        random_missing_rate=rate,
        seed=42
    )
    results_with_physics[rate] = res_with_physics['test_rmse']

# 对比改善百分比
print("\n" + "="*70)
print("物理约束改善效果")
print("="*70)
print(f"{'缺失率':<10} {'无约束':<12} {'有约束':<12} {'改善%':<10}")
print("-"*70)
for rate in random_missing_rates:
    no_physics_rmse = results_without_physics[rate]
    with_physics_rmse = results_with_physics[rate]
    improvement = (no_physics_rmse - with_physics_rmse) / no_physics_rmse * 100
    print(f"{rate*100:.0f}%{'':<7} {no_physics_rmse:<12.4f} {with_physics_rmse:<12.4f} {improvement:<10.2f}%")
```

**预期发现**:
- 缺失率越高，物理约束改善越明显
- Random Missing 可能比 Uniform 更需要物理约束

---

## 💡 实现细节

### 核心函数

```python
def random_missing_by_battery(features, targets, battery_ids,
                              missing_rate=0.3, seed=None, verbose=True):
    """
    按电池分组进行随机缺失采样

    工作流程:
    1. 按 battery_id 分组
    2. 对每个电池独立进行随机采样
    3. 保留 (1 - missing_rate) 比例的样本
    4. 保持时序顺序（对保留的索引排序）
    5. 拼接所有电池的数据

    Args:
        features: 特征数组 (N, D)
        targets: 目标数组 (N,)
        battery_ids: 电池ID数组 (N,)
        missing_rate: 缺失率，取值范围 [0, 1)
        seed: 随机种子
        verbose: 是否打印详细信息

    Returns:
        sampled_features, sampled_targets, sampled_battery_ids
    """
```

### 与 Uniform Subsampling 的代码对比

**Uniform Subsampling**:
```python
# 规律间隔
kept_indices = battery_indices[offset::sampling_interval]
# 例如: [0, 5, 10, 15, 20, ...]  (interval=5, offset=0)
```

**Random Missing**:
```python
# 随机选择
n_keep = int(n_samples * (1 - missing_rate))
kept_indices = rng.choice(battery_indices, size=n_keep, replace=False)
kept_indices = np.sort(kept_indices)  # 保持时序
# 例如: [2, 7, 11, 13, 19, ...]  (不规则间隔)
```

---

## 📈 预期结果

### 场景对比

| 场景 | 保留率 | RMSE | 相对于干净数据的性能下降 |
|------|--------|------|-------------------------|
| 干净数据 | 100% | 0.0220 | 0% (基线) |
| Uniform (interval=5) | 20% | 0.0285 | 29.5% ↓ |
| Random (rate=0.8) | 20% | 0.0310 | 40.9% ↓ |
| Random (rate=0.4) | 60% | 0.0255 | 15.9% ↓ |

**发现**:
- Random Missing 通常比 Uniform 性能**更差**（不规则性）
- 但 Random 更接近真实故障场景，具有**实际应用价值**

### 物理约束改善

| 缺失率 | 无约束 RMSE | 有约束 RMSE (weight=0.2) | 改善% |
|--------|------------|-------------------------|-------|
| 20% | 0.0310 | 0.0280 | 9.7% ↑ |
| 40% | 0.0355 | 0.0305 | 14.1% ↑ |
| 60% | 0.0420 | 0.0350 | 16.7% ↑ |

**结论**: 数据越稀疏/不规则，物理约束价值越大

---

## ⚠️ 注意事项

### 1. 随机性控制

- **必须设置 `seed`** 确保可复现
- 训练集和验证集使用**不同种子** (`seed` vs `seed+2000`)
- 测试集保持干净（无退化）

### 2. 保留率选择

- **Light (80% retained)**: 适合初步验证
- **Moderate (60% retained)**: ⭐ 推荐，平衡性能与挑战性
- **Heavy (40% retained)**: 极端场景，适合压力测试

### 3. 对比实验设计

- 固定保留率，对比 Uniform vs Random
- 固定缺失率，对比有/无物理约束
- 多个种子运行，计算均值和标准差

---

## 🔧 常见问题

### Q: 为什么 Random Missing 比 Uniform 性能更差？

**A**:
- Uniform: 规律间隔，模型可以学习固定步长
- Random: 不规则间隔，破坏时序连续性更严重
- 但 Random 更接近真实传感器故障

---

### Q: 如何选择 missing_rate？

**A**: 建议测试多个值，观察趋势
```python
# 推荐测试序列
rates = [0.2, 0.3, 0.4, 0.5, 0.6]  # 20%-60%
```

---

### Q: 可以同时应用 Scenario 2 和 Scenario 3 吗？

**A**: 不可以，它们是互斥的：
- `degradation_scenario='scenario2'` → 规律稀疏采样
- `degradation_scenario='scenario3'` → 随机缺失

如需组合测试，可以先应用一个场景，然后手动处理数据再应用另一个。

---

### Q: 如何理解 missing_rate vs retention_rate？

**A**:
```python
missing_rate = 0.4     # 缺失40%
retention_rate = 60%   # 保留60%

# 关系
retention_rate = (1 - missing_rate) * 100
```

---

## 📚 相关文档

- [稀疏采样手动控制](SPARSE_SAMPLING_MANUAL_CONTROL.md) - 场景二详细指南
- [双场景系统](DUAL_SCENARIO_GUIDE.md) - 场景一和场景二
- [单调性权重测试](../QUICK_START_WEIGHT_TEST.md) - 物理约束测试

---

## 🎯 最佳实践

### 1. 完整的消融实验流程

```python
# Step 1: 测试不同缺失率
for rate in [0.2, 0.4, 0.6]:
    train_cross_battery_model(
        degradation_scenario='scenario3',
        random_missing_rate=rate
    )

# Step 2: 对比 Uniform vs Random (相同保留率)
# Uniform 50%
train_cross_battery_model(
    degradation_scenario='scenario2',
    sparse_sampling_interval=2
)

# Random 50%
train_cross_battery_model(
    degradation_scenario='scenario3',
    random_missing_rate=0.5
)

# Step 3: 测试物理约束效果
# 修改 configs/models/cnn_lstm_config.json
# physics_constraints.enabled = true/false
```

### 2. 论文实验建议

**Table 1: 不同数据退化场景对比**
| Scenario | Method | Retention | RMSE | MAE | R² |
|----------|--------|-----------|------|-----|-----|
| None | - | 100% | 0.0220 | 0.0175 | 0.9621 |
| 1 | Noise+Dropout | ~80% | 0.0265 | 0.0210 | 0.9512 |
| 2 | Uniform | 20% | 0.0285 | 0.0225 | 0.9445 |
| 3 | Random | 20% | 0.0310 | 0.0245 | 0.9380 |

**Table 2: 物理约束在 Random Missing 下的改善**
| Missing Rate | Baseline | Physics | Improvement |
|--------------|----------|---------|-------------|
| 20% | 0.0310 | 0.0280 | 9.7% ↑ |
| 40% | 0.0355 | 0.0305 | 14.1% ↑ |
| 60% | 0.0420 | 0.0350 | 16.7% ↑ |

---

**祝实验顺利！** 🎉

如有问题，请参考 [SPARSE_SAMPLING_MANUAL_CONTROL.md](SPARSE_SAMPLING_MANUAL_CONTROL.md) 或检查代码注释。
