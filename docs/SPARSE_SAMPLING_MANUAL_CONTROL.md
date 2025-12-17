# 稀疏采样手动间隔控制使用指南

## 概述

现在支持两种方式控制稀疏采样间隔：
1. **预设级别**：使用 `SPARSE_SAMPLING_LEVEL` 参数（快速便捷）
2. **手动间隔**：使用 `SPARSE_SAMPLING_INTERVAL` 参数（精确控制）

**优先级**：`SPARSE_SAMPLING_INTERVAL` > `SPARSE_SAMPLING_LEVEL`

如果设置了 `SPARSE_SAMPLING_INTERVAL`，将忽略 `SPARSE_SAMPLING_LEVEL`。

---

## 使用方法

### 方法 1: 使用预设级别（原有方式）

```python
# 在 train_cross_battery.py 中设置
DEGRADATION_SCENARIO = 'scenario2'
SPARSE_SAMPLING_LEVEL = 'moderate'  # 'dense', 'moderate', 'sparse', 'very_sparse'
SPARSE_SAMPLING_INTERVAL = None     # 必须设为 None
```

**预设级别对照表**：
| 级别 | 间隔 | 保留率 | 说明 |
|------|------|--------|------|
| `dense` | 2 | 50% | 密集采样 |
| `moderate` | 5 | 20% | 中等采样 |
| `sparse` | 10 | 10% | 稀疏采样 |
| `very_sparse` | 20 | 5% | 极稀疏采样 |

---

### 方法 2: 手动设置间隔（新功能）⭐

```python
# 在 train_cross_battery.py 中设置
DEGRADATION_SCENARIO = 'scenario2'
SPARSE_SAMPLING_LEVEL = 'moderate'  # 将被忽略
SPARSE_SAMPLING_INTERVAL = 7        # 直接设置间隔为 7
```

**效果**：每 7 个循环保留 1 个，保留率 ≈ 14.3%

---

## 实验设计建议

### 标准化稀疏采样方案

#### Scheme 1: Uniform Subsampling（均匀子采样）

**轻度稀疏**（用于对比基线）：
```python
SPARSE_SAMPLING_INTERVAL = 2   # 保留 50%
SPARSE_SAMPLING_INTERVAL = 3   # 保留 33%
```

**中度稀疏**（主要测试区域）：
```python
SPARSE_SAMPLING_INTERVAL = 5   # 保留 20%
SPARSE_SAMPLING_INTERVAL = 7   # 保留 14%
SPARSE_SAMPLING_INTERVAL = 10  # 保留 10%
```

**重度稀疏**（极限测试）：
```python
SPARSE_SAMPLING_INTERVAL = 15  # 保留 6.7%
SPARSE_SAMPLING_INTERVAL = 20  # 保留 5%
SPARSE_SAMPLING_INTERVAL = 25  # 保留 4%
```

---

## 使用案例

### 案例 1: 快速测试（使用预设）

```python
# 场景：快速验证代码是否正常运行
DEGRADATION_SCENARIO = 'scenario2'
SPARSE_SAMPLING_LEVEL = 'moderate'
SPARSE_SAMPLING_INTERVAL = None
```

**输出**：
```
场景二: 稀疏采样 - 中等采样 (每5个循环保留1个, 20%)
```

---

### 案例 2: 精确实验（手动设置）

```python
# 场景：绘制 Robustness Curve，需要精确控制间隔
DEGRADATION_SCENARIO = 'scenario2'
SPARSE_SAMPLING_LEVEL = None  # 可以不设置
SPARSE_SAMPLING_INTERVAL = 7  # 间隔 7
```

**输出**：
```
场景二: 稀疏采样 - 手动间隔 (每7个循环保留1个, 保留率≈14.3%)
```

---

### 案例 3: 批量实验（循环测试）

```python
# 生成 Robustness Curve
intervals = [2, 3, 5, 7, 10, 15, 20]
results = []

for interval in intervals:
    # 修改配置
    SPARSE_SAMPLING_INTERVAL = interval

    # 运行训练
    wrapper, results_dict, data_dict = train_cross_battery_model(
        model_type='cnn_lstm',
        degradation_scenario='scenario2',
        sparse_sampling_interval=interval,
        # ... 其他参数
    )

    # 记录结果
    results.append({
        'interval': interval,
        'retention_rate': 100.0 / interval,
        'rmse': results_dict['test_rmse']
    })

# 绘制曲线
import matplotlib.pyplot as plt
plt.plot([r['retention_rate'] for r in results],
         [r['rmse'] for r in results])
plt.xlabel('Retention Rate (%)')
plt.ylabel('RMSE (%)')
plt.title('Robustness Curve: RMSE vs. Data Sparsity')
plt.show()
```

---

## 间隔选择建议

### 常用间隔及其含义

| 间隔 | 保留率 | 适用场景 | 说明 |
|------|--------|---------|------|
| 2 | 50% | 基线对比 | 轻度稀疏，接近密集数据 |
| 3 | 33% | 轻度测试 | 较为稀疏，但仍有充足数据 |
| 5 | 20% | 标准测试 | 中等稀疏，常用于验证 |
| 7 | 14.3% | 中度测试 | 较为稀疏，物理约束价值开始体现 |
| 10 | 10% | 重度测试 | 稀疏程度高，考验模型鲁棒性 |
| 15 | 6.7% | 极限测试 | 极度稀疏，接近实际 HPPC 场景 |
| 20 | 5% | 极端场景 | 极端稀疏，测试物理约束的极限价值 |

---

## 技术细节

### 实现原理

```python
# 在 prepare_cross_battery_data() 中
if sparse_sampling_interval is not None:
    # 使用手动间隔
    sampling_interval = sparse_sampling_interval
    retention_rate = 100.0 / sampling_interval
    description = f"手动间隔 (每{sampling_interval}个循环保留1个, 保留率≈{retention_rate:.1f}%)"
else:
    # 使用预设级别
    sampling_params = get_sparse_sampling_preset(sparse_sampling_level)
    sampling_interval = sampling_params['sampling_interval']
    description = sampling_params['description']
```

### 采样逻辑

```python
# 按电池分组进行稀疏采样
for battery_id in unique_battery_ids:
    battery_samples = get_samples_for_battery(battery_id)

    # 每隔 interval 个样本保留 1 个
    keep_indices = np.arange(0, len(battery_samples), interval)
    retained_samples = battery_samples[keep_indices]

    # 保留原始 cycle 索引
    retained_cycles = original_cycles[keep_indices]
```

**关键**：保留原始 cycle 索引，不是数组索引！

---

## 常见问题

### Q1: 设置了两个参数，哪个生效？

**A**: `SPARSE_SAMPLING_INTERVAL` 优先级更高。

```python
SPARSE_SAMPLING_LEVEL = 'moderate'      # interval=5（被忽略）
SPARSE_SAMPLING_INTERVAL = 7            # interval=7（实际生效）
```

**输出**：间隔 = 7

---

### Q2: 如何完全禁用稀疏采样？

**A**: 设置场景为 `'none'`：

```python
DEGRADATION_SCENARIO = 'none'  # 无退化，使用干净数据
```

---

### Q3: 间隔太大会有什么问题？

**A**:
- 样本数太少，模型可能无法收敛
- 建议最大间隔 ≤ 20（保留率 ≥ 5%）
- 如果电池只有 200 个 cycle，interval=20 只剩 10 个样本

---

### Q4: 能否对不同电池设置不同间隔？

**A**: 当前不支持。所有电池使用相同间隔。

未来可以扩展为：
```python
SPARSE_SAMPLING_INTERVAL = {
    '1-1': 5,
    '2-3': 10,
    # ...
}
```

---

## 完整示例

```python
# train_cross_battery.py 配置示例

if __name__ == "__main__":
    # 基本配置
    MODEL_TYPE = 'cnn_lstm'
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 42

    # ===== 稀疏采样配置 =====
    DEGRADATION_SCENARIO = 'scenario2'

    # 方式 1: 预设级别（快速测试）
    # SPARSE_SAMPLING_LEVEL = 'moderate'
    # SPARSE_SAMPLING_INTERVAL = None

    # 方式 2: 手动间隔（精确控制）⭐
    SPARSE_SAMPLING_LEVEL = None  # 可选，会被忽略
    SPARSE_SAMPLING_INTERVAL = 7  # 每 7 个循环保留 1 个

    # 训练
    wrapper, results, data_dict = train_cross_battery_model(
        model_type=MODEL_TYPE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        device=DEVICE,
        seed=SEED,
        degradation_scenario=DEGRADATION_SCENARIO,
        sparse_sampling_level=SPARSE_SAMPLING_LEVEL,
        sparse_sampling_interval=SPARSE_SAMPLING_INTERVAL
    )
```

---

## 测试验证

运行测试脚本：

```bash
python examples/test_manual_interval.py
```

**输出示例**：
```
======================================================================
不同间隔的采样结果
======================================================================
间隔         保留样本数           保留率             说明
----------------------------------------------------------------------
2          50                50.0%        密集采样
3          34                34.0%        中等采样
5          20                20.0%        中等采样
7          15                15.0%        稀疏采样
10         10                10.0%        稀疏采样
15         7                  7.0%        极稀疏采样
20         5                  5.0%        极稀疏采样
```

---

## 总结

### 新增功能

✅ 支持手动设置任意间隔
✅ 保持向后兼容（预设级别仍可用）
✅ 优先级清晰（手动 > 预设）
✅ 自动计算并显示保留率

### 适用场景

1. **论文实验**：绘制 Robustness Curve，需要精确控制间隔
2. **消融实验**：测试不同稀疏程度对模型的影响
3. **超参数搜索**：寻找最优稀疏采样间隔

### 推荐配置

**快速验证**：使用预设级别
**正式实验**：使用手动间隔，精确控制

---

**更新日期**：2024-12-17
**版本**：v1.0
