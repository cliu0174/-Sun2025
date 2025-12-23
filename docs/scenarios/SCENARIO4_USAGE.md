# Scenario 4: 连续循环缺失 (Consecutive Cycle Drop)

## 功能说明

Scenario 4 模拟真实场景中**传感器/采集系统故障**导致**连续一段时间**没有数据的情况。

### 与其他场景的区别

| 场景 | 缺失模式 | 应用场景 |
|-----|---------|---------|
| Scenario 1 | 随机噪声 + 随机丢弃点 | 测量噪声和偶发故障 |
| Scenario 2 | 规律间隔采样 | 定期HPPC/DST测试 |
| Scenario 3 | 随机分散丢弃点 | 随机传感器读数失败 |
| **Scenario 4** | **连续段缺失** | **传感器/采集系统故障** |

### 关键特点

1. **连续性**：丢弃的是**连续的一段循环**，而不是随机分散的点
2. **多段故障**：支持多个不重叠的连续缺失段（模拟多次故障）
3. **按电池独立**：每个电池独立进行缺失，保持时序结构

## 使用方法

### 1. 预设级别

在 `train_cross_battery.py` 中设置：

```python
DEGRADATION_SCENARIO = 'scenario4'
CYCLE_DROP_LEVEL = 'moderate'  # 'light', 'moderate', 'heavy'
```

预设配置：
- **light**: 20% 丢弃, 1段连续缺失 (保留80%循环)
- **moderate**: 30% 丢弃, 2段连续缺失 (保留70%循环)
- **heavy**: 50% 丢弃, 3段连续缺失 (保留50%循环)

### 2. 手动配置

```python
DEGRADATION_SCENARIO = 'scenario4'
CYCLE_DROP_RATE = 0.4           # 丢弃40%循环
CYCLE_DROP_NUM_GAPS = 2         # 分成2段连续缺失
```

### 3. 混合配置

```python
DEGRADATION_SCENARIO = 'scenario4'
CYCLE_DROP_LEVEL = 'moderate'   # 基础预设
CYCLE_DROP_RATE = 0.35          # 覆盖丢弃率（使用35%而不是预设的30%）
# CYCLE_DROP_NUM_GAPS = None     # 保持预设的2段
```

## 示例

### 示例1：使用 moderate 预设

```python
原始: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]  (20个循环)

应用 moderate 预设（30%丢弃，2段）:
缺失段1: cycle [2~5]    (连续4个)
缺失段2: cycle [13~15]  (连续3个)

保留: [0,1, 6,7,8,9,10,11,12, 16,17,18,19]  (14个，70%)
```

### 示例2：手动配置

```python
CYCLE_DROP_RATE = 0.5     # 丢弃50%
CYCLE_DROP_NUM_GAPS = 1   # 1段连续缺失

原始: 100个循环
结果: 1个长度约50的连续缺失段，保留约50个循环
```

## 完整训练示例

```python
# train_cross_battery.py

MODEL_TYPE = 'cnn_lstm'
DEGRADATION_SCENARIO = 'scenario4'

# 使用预设
CYCLE_DROP_LEVEL = 'heavy'  # 50% 丢弃, 3段

# 或手动配置
# CYCLE_DROP_RATE = 0.4
# CYCLE_DROP_NUM_GAPS = 2

wrapper, results, data_dict = train_cross_battery_model(
    model_type=MODEL_TYPE,
    degradation_scenario=DEGRADATION_SCENARIO,
    cycle_drop_level=CYCLE_DROP_LEVEL,
    cycle_drop_rate=CYCLE_DROP_RATE,
    cycle_drop_num_gaps=CYCLE_DROP_NUM_GAPS,
    # ... 其他参数
)
```

## 输出示例

### 默认输出 (show_battery_details=False)

训练时会显示简洁的统计摘要：

```
场景四: Consecutive Cycle Drop (连续循环缺失)
======================================================================
总丢弃率: 30.0% (保留 70.0%)
缺失段数量: 2
原始样本数: 15420
电池数量: 46
(详细电池信息已隐藏，设置 show_battery_details=True 查看)
----------------------------------------------------------------------
总保留样本数: 10794
实际保留率: 70.0%
======================================================================
```

### 详细输出 (show_battery_details=True)

当需要调试或详细分析时，可以启用详细输出：

```
场景四: Consecutive Cycle Drop (连续循环缺失)
======================================================================
总丢弃率: 30.0% (保留 70.0%)
缺失段数量: 2
原始样本数: 15420
电池数量: 46
----------------------------------------------------------------------
  电池 1-1: 200 cycles → 140 cycles (70.0%)
    缺失段1: cycle [45~78] (长度=34)
    缺失段2: cycle [156~181] (长度=26)

  电池 1-2: 180 cycles → 126 cycles (70.0%)
    缺失段1: cycle [23~48] (长度=26)
    缺失段2: cycle [99~126] (长度=28)

  ... (所有46个电池的详细信息)

总保留样本数: 10794
实际保留率: 70.0%
======================================================================
```

## 注意事项

1. **测试集保持干净**：测试集不应用任何退化，用于公平对比
2. **种子固定**：使用 `seed` 参数确保可复现
3. **验证集使用不同种子**：避免与训练集产生完全相同的缺失模式

## API 参考

### consecutive_cycle_drop_by_battery()

```python
from utils.data_augmentation import consecutive_cycle_drop_by_battery

sampled_features, sampled_targets, sampled_battery_ids = consecutive_cycle_drop_by_battery(
    features,                  # (N, feature_dim) 特征数组
    targets,                   # (N,) 目标数组
    battery_ids,               # (N,) 电池ID数组
    cycle_drop_rate=0.3,       # 总丢弃比例 (0~1)
    num_gaps=2,                # 缺失段数量
    seed=42,                   # 随机种子
    verbose=True,              # 是否打印统计摘要
    show_battery_details=False # 是否打印每个电池的详细缺失信息
)
```

**参数说明**:
- `verbose=True`: 打印总体统计摘要（丢弃率、样本数等）
- `show_battery_details=True`: 额外打印每个电池的详细缺失段信息（默认关闭以减少输出）

### get_consecutive_cycle_drop_preset()

```python
from utils.data_augmentation import get_consecutive_cycle_drop_preset

preset = get_consecutive_cycle_drop_preset('moderate')
# Returns: {'cycle_drop_rate': 0.3, 'num_gaps': 2, 'description': '...'}
```
