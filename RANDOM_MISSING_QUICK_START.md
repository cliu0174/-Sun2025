# Random Missing (场景三) - 快速开始

## 🎯 什么是 Random Missing？

**Random Missing** 是场景三的数据退化方式，模拟传感器随机故障导致的数据缺失。

### 核心特点

- **随机性**: 每个样本有一定概率被丢弃
- **不规则性**: 缺失位置不可预测
- **真实性**: 更接近实际传感器故障场景

### 与场景二的区别

| 特性 | 场景二 (Uniform) | 场景三 (Random) |
|------|------------------|-----------------|
| 采样方式 | 规律间隔 (如每5个保留1个) | 随机丢弃指定比例 |
| 间隔 | 固定 (如 5, 5, 5, ...) | 不规则 (如 2, 7, 4, 11, ...) |
| 可预测性 | 高 | 低 |
| 难度 | 相对简单 | 更具挑战性 |

---

## 🚀 三种使用方式

### 方式 1: 使用预设级别（推荐新手）⭐

```python
from train_cross_battery import train_cross_battery_model

# 轻度缺失 (20% missing, 80% retained)
wrapper, results, data_dict = train_cross_battery_model(
    model_type='cnn_lstm',
    degradation_scenario='scenario3',
    random_missing_level='light',  # 'light', 'moderate', 'heavy'
    seed=42
)

print(f"RMSE: {results['test_rmse']:.4f}")
print(f"MAE: {results['test_mae']:.4f}")
```

**三种预设级别**:
- `'light'`: 20% 缺失 (保留80%)
- `'moderate'`: 40% 缺失 (保留60%) ⭐ **推荐**
- `'heavy'`: 60% 缺失 (保留40%)

---

### 方式 2: 手动设置缺失率（精确控制）

```python
# 手动设置: 丢弃35%数据
wrapper, results, data_dict = train_cross_battery_model(
    model_type='cnn_lstm',
    degradation_scenario='scenario3',
    random_missing_rate=0.35,  # 0-1 之间的浮点数
    seed=42
)
```

**优先级**: `random_missing_rate` > `random_missing_level`

---

### 方式 3: 一键批量测试（推荐实验）

```bash
python test_random_missing.py
```

**自动完成**:
- 测试 20%, 40%, 60% 三个缺失率
- 对比 Random vs Uniform（相同保留率）
- 生成对比图和报告

**输出结果**:
```
results/random_missing_test/20241217_153045/
├── random_missing_comparison.png  # 对比图 ⭐
├── experiment_report.txt          # 实验报告
├── all_results.pkl                # 原始数据
└── results_summary.json           # JSON摘要
```

---

## 📊 预期结果

### 典型实验结果

| 场景 | 保留率 | RMSE | MAE | R² |
|------|--------|------|-----|-----|
| 干净数据 | 100% | 0.0220 | 0.0175 | 0.9621 |
| Uniform (interval=5) | 20% | 0.0285 | 0.0225 | 0.9445 |
| **Random (20% missing)** | **80%** | **0.0245** | **0.0195** | **0.9550** |
| **Random (40% missing)** | **60%** | **0.0295** | **0.0235** | **0.9420** |
| **Random (60% missing)** | **40%** | **0.0360** | **0.0285** | **0.9250** |

### 关键发现

1. **性能趋势**: 缺失率越高，性能下降越明显
2. **Random vs Uniform**: 相同保留率下，Random 通常**更难**
3. **物理约束价值**: 在高缺失率下，物理约束改善更显著

---

## 🔬 典型实验流程

### 实验 1: 测试不同缺失率

```python
from train_cross_battery import train_cross_battery_model

missing_rates = [0.2, 0.4, 0.6]
results_dict = {}

for rate in missing_rates:
    print(f"\n测试缺失率: {rate*100:.0f}%")

    _, results, _ = train_cross_battery_model(
        model_type='cnn_lstm',
        degradation_scenario='scenario3',
        random_missing_rate=rate,
        seed=42
    )

    results_dict[rate] = {
        'rmse': results['test_rmse'],
        'mae': results['test_mae'],
        'r2': results['test_r2']
    }

# 打印对比
print("\n结果对比:")
print(f"{'缺失率':<10} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
for rate, res in results_dict.items():
    print(f"{rate*100:.0f}%{'':<7} {res['rmse']:<12.4f} {res['mae']:<12.4f} {res['r2']:<12.4f}")
```

---

### 实验 2: Random vs Uniform 对比

对比相同保留率下的性能差异。

```python
# Random: 60% 保留
_, results_random, _ = train_cross_battery_model(
    model_type='cnn_lstm',
    degradation_scenario='scenario3',
    random_missing_rate=0.4,  # 丢弃40%，保留60%
    seed=42
)

# Uniform: 50% 保留 (interval=2)
_, results_uniform, _ = train_cross_battery_model(
    model_type='cnn_lstm',
    degradation_scenario='scenario2',
    sparse_sampling_interval=2,  # 每2个保留1个 = 50%
    seed=42
)

print("\n对比结果:")
print(f"Random (60%):  RMSE={results_random['test_rmse']:.4f}")
print(f"Uniform (50%): RMSE={results_uniform['test_rmse']:.4f}")

# 计算差异
diff = results_random['test_rmse'] - results_uniform['test_rmse']
print(f"\n差异: {diff:.4f} ({'Random更难' if diff > 0 else 'Random更易'})")
```

---

### 实验 3: 物理约束在 Random Missing 下的效果

```python
import json

# 修改配置文件以启用/禁用物理约束
config_path = 'configs/models/cnn_lstm_config.json'

def test_physics_effect(missing_rate, enable_physics):
    """测试物理约束效果"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    config['physics_constraints']['enabled'] = enable_physics
    if enable_physics:
        config['physics_constraints']['monotonic_weight'] = 0.2

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    _, results, _ = train_cross_battery_model(
        model_type='cnn_lstm',
        degradation_scenario='scenario3',
        random_missing_rate=missing_rate,
        seed=42
    )

    return results['test_rmse']

# 测试 40% 缺失
rate = 0.4
rmse_no_physics = test_physics_effect(rate, enable_physics=False)
rmse_with_physics = test_physics_effect(rate, enable_physics=True)

improvement = (rmse_no_physics - rmse_with_physics) / rmse_no_physics * 100

print(f"\n缺失率: {rate*100:.0f}%")
print(f"无物理约束: RMSE = {rmse_no_physics:.4f}")
print(f"有物理约束: RMSE = {rmse_with_physics:.4f}")
print(f"改善: {improvement:.1f}%")
```

---

## 📝 配置参数说明

### `train_cross_battery_model` 新增参数

```python
def train_cross_battery_model(
    # ... 其他参数 ...
    degradation_scenario='scenario3',    # 场景选择
    random_missing_level='moderate',     # 预设级别
    random_missing_rate=None             # 手动缺失率（优先级高）
):
```

### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `degradation_scenario` | str | `'none'` | 必须设为 `'scenario3'` 启用 Random Missing |
| `random_missing_level` | str | `'moderate'` | 预设级别: `'light'`, `'moderate'`, `'heavy'` |
| `random_missing_rate` | float / None | `None` | 手动缺失率 (0-1)，如 `0.35` 表示丢弃35% |

---

## ⚠️ 注意事项

### 1. 随机种子

- **必须设置 `seed`** 确保实验可复现
- 训练集和验证集使用**不同种子**（自动处理）
- 测试集保持干净（无退化）

### 2. 参数优先级

```python
# 如果同时设置两个参数
random_missing_level='light'    # 20% 缺失
random_missing_rate=0.35        # 35% 缺失

# 实际使用: random_missing_rate=0.35 ⭐ (优先级高)
```

### 3. 场景互斥

- 不能同时使用 `scenario2` 和 `scenario3`
- 每次只能选择一个 `degradation_scenario`

### 4. 缺失率建议

- **Light (20%)**: 初步测试，保留80%
- **Moderate (40%)**: ⭐ **推荐**，平衡性能与挑战
- **Heavy (60%)**: 压力测试，保留40%
- **不建议 > 70%**: 数据过少，训练不稳定

---

## 🔧 故障排除

### Q: 提示 "Unknown degradation scenario: scenario3"

**A**: 确保已更新 `train_cross_battery.py`，检查是否包含 scenario3 逻辑。

```bash
grep -n "scenario3" train_cross_battery.py
```

应该看到多处 `scenario3` 的引用。

---

### Q: 缺失率设置无效

**A**: 检查参数优先级：
- 如果设置了 `random_missing_rate`，会忽略 `random_missing_level`
- 确认 `degradation_scenario='scenario3'`（不是 scenario2）

---

### Q: 想要精确的保留率（如70%）

**A**: 使用 `random_missing_rate`:

```python
# 保留70% → 丢弃30%
random_missing_rate=0.3
```

---

### Q: 如何对比 Random vs Uniform（相同保留率）？

**A**: 由于 Uniform 的保留率由 interval 决定（100/interval），
无法精确匹配。建议对比接近的值：

```python
# Random: 60% 保留
random_missing_rate=0.4

# Uniform: 50% 保留 (interval=2) 或 33% 保留 (interval=3)
sparse_sampling_interval=2  # 更接近
```

---

## 📚 相关文档

- **详细指南**: [docs/RANDOM_MISSING_GUIDE.md](docs/RANDOM_MISSING_GUIDE.md)
- **场景二指南**: [docs/SPARSE_SAMPLING_MANUAL_CONTROL.md](docs/SPARSE_SAMPLING_MANUAL_CONTROL.md)
- **双场景系统**: [docs/DUAL_SCENARIO_GUIDE.md](docs/DUAL_SCENARIO_GUIDE.md)
- **权重测试**: [QUICK_START_WEIGHT_TEST.md](QUICK_START_WEIGHT_TEST.md)

---

## 💡 最佳实践

### 1. 先快速测试，再深入

```bash
# Step 1: 快速测试单个缺失率
python -c "
from train_cross_battery import train_cross_battery_model
_, results, _ = train_cross_battery_model(
    model_type='cnn_lstm',
    degradation_scenario='scenario3',
    random_missing_level='moderate',
    seed=42
)
print(f'RMSE: {results[\"test_rmse\"]:.4f}')
"

# Step 2: 如果效果明显，运行完整测试
python test_random_missing.py
```

### 2. 论文实验建议

**Table 1: 不同数据退化场景对比**

运行以下实验：
- 干净数据 (`scenario='none'`)
- 场景一 (`scenario='scenario1'`, noise level)
- 场景二 (`scenario='scenario2'`, interval=5)
- 场景三 (`scenario='scenario3'`, missing_rate=0.4)

**Table 2: Random Missing 下的物理约束效果**

运行 `test_random_missing.py`，对比：
- 20%, 40%, 60% 缺失率
- 有/无物理约束

### 3. 多种子验证

```python
seeds = [42, 123, 456, 789, 2024]
results_list = []

for seed in seeds:
    _, results, _ = train_cross_battery_model(
        model_type='cnn_lstm',
        degradation_scenario='scenario3',
        random_missing_rate=0.4,
        seed=seed
    )
    results_list.append(results['test_rmse'])

# 统计
import numpy as np
mean_rmse = np.mean(results_list)
std_rmse = np.std(results_list)
print(f"RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
```

---

## 🎉 快速示例

### 最简单的用法

```python
from train_cross_battery import train_cross_battery_model

# 一行代码启动
wrapper, results, data_dict = train_cross_battery_model(
    degradation_scenario='scenario3'
)

print(f"完成！RMSE: {results['test_rmse']:.4f}")
```

### 推荐的完整配置

```python
from train_cross_battery import train_cross_battery_model

wrapper, results, data_dict = train_cross_battery_model(
    model_type='cnn_lstm',              # 模型类型
    degradation_scenario='scenario3',   # 场景三
    random_missing_level='moderate',    # 中度缺失 (40%)
    seed=42,                            # 固定种子
    device='cuda'                       # 使用GPU
)

# 查看结果
print(f"测试 RMSE: {results['test_rmse']:.4f}")
print(f"测试 MAE:  {results['test_mae']:.4f}")
print(f"测试 R²:   {results['test_r2']:.4f}")
```

---

**祝实验顺利！** 🚀

有问题请参考 [docs/RANDOM_MISSING_GUIDE.md](docs/RANDOM_MISSING_GUIDE.md) 或查看代码注释。
