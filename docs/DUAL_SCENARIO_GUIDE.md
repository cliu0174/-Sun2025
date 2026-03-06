# 双场景系统使用指南

## 概述

本系统提供两个独立的数据退化场景，用于验证物理约束在不同场景下的鲁棒性和有效性。

### 场景一: 随机退化
- **特点**: 随机高斯噪声 + 随机样本丢弃
- **用途**: 模拟测量误差和数据采集不稳定
- **级别**: light (轻度), medium (中度), heavy (重度)

### 场景二: 稀疏采样
- **特点**: 规律间隔采样 (每N个循环保留1个)
- **用途**: 模拟定期容量测试 (如HPPC/DST测试)
- **级别**: dense (密集), moderate (中等), sparse (稀疏), very_sparse (极稀疏)

---

## 使用方法

### 1. 基本配置

在 `train_cross_battery.py` 的主函数中修改配置参数：

```python
# ===== 数据退化场景选择 =====
DEGRADATION_SCENARIO = 'none'  # 'none', 'scenario1', 'scenario2'

# 场景一参数 (仅当 DEGRADATION_SCENARIO='scenario1' 时生效)
NOISE_LEVEL = 'light'           # 'light', 'medium', 'heavy'

# 场景二参数 (仅当 DEGRADATION_SCENARIO='scenario2' 时生效)
SPARSE_SAMPLING_LEVEL = 'moderate'  # 'dense', 'moderate', 'sparse', 'very_sparse'
```

### 2. 场景选项详解

#### 场景选择 (DEGRADATION_SCENARIO)

- **`'none'`**: 无退化 (干净数据)
  - 使用完整的原始数据
  - 适用于基线实验

- **`'scenario1'`**: 随机退化
  - 对训练集和验证集添加高斯噪声
  - 随机丢弃部分样本
  - 测试集保持干净

- **`'scenario2'`**: 稀疏采样
  - 对训练集和验证集进行规律间隔采样
  - 保持时序结构
  - 测试集保持干净

#### 场景一噪声级别 (NOISE_LEVEL)

| 级别 | 特征噪声σ | 容量噪声σ | 丢弃率 | 描述 |
|------|-----------|-----------|--------|------|
| `light` | 0.01 | 0.005 | 5% | 轻度噪声，适合初步验证 |
| `medium` | 0.05 | 0.02 | 15% | 中度噪声，模拟常见测量误差 |
| `heavy` | 0.10 | 0.05 | 30% | 重度噪声，极端场景测试 |

#### 场景二采样级别 (SPARSE_SAMPLING_LEVEL)

| 级别 | 采样间隔 | 保留率 | 描述 |
|------|----------|--------|------|
| `dense` | 2 | 50% | 密集采样，每2个循环测试1次 |
| `moderate` | 5 | 20% | 中等采样，每5个循环测试1次 |
| `sparse` | 10 | 10% | 稀疏采样，每10个循环测试1次 |
| `very_sparse` | 20 | 5% | 极稀疏采样，每20个循环测试1次 |

---

## 实验示例

### 示例1: 干净数据基线

```python
DEGRADATION_SCENARIO = 'none'
```

运行后查看物理约束在干净数据上的表现。

### 示例2: 轻度随机退化

```python
DEGRADATION_SCENARIO = 'scenario1'
NOISE_LEVEL = 'light'
```

验证物理约束能否在轻度噪声下保持效果。

### 示例3: 极稀疏采样

```python
DEGRADATION_SCENARIO = 'scenario2'
SPARSE_SAMPLING_LEVEL = 'very_sparse'
```

测试物理约束在数据极度稀疏 (仅5%样本) 时的表现。

---

## 实验设计建议

### 对比实验设计

建议按以下顺序进行实验，系统对比物理约束的作用：

#### 第一阶段: 基线建立
1. **实验1**: `DEGRADATION_SCENARIO='none'`, 物理约束关闭
   - 记录基线性能 (MAE, RMSE, R²)
2. **实验2**: `DEGRADATION_SCENARIO='none'`, 物理约束开启
   - 对比物理约束在干净数据上的作用

#### 第二阶段: 场景一测试
3. **实验3**: `DEGRADATION_SCENARIO='scenario1'`, `NOISE_LEVEL='light'`, 物理约束关闭
4. **实验4**: `DEGRADATION_SCENARIO='scenario1'`, `NOISE_LEVEL='light'`, 物理约束开启
5. **实验5-6**: 重复3-4，使用 `NOISE_LEVEL='medium'`
6. **实验7-8**: 重复3-4，使用 `NOISE_LEVEL='heavy'`

#### 第三阶段: 场景二测试
7. **实验9**: `DEGRADATION_SCENARIO='scenario2'`, `SPARSE_SAMPLING_LEVEL='moderate'`, 物理约束关闭
8. **实验10**: `DEGRADATION_SCENARIO='scenario2'`, `SPARSE_SAMPLING_LEVEL='moderate'`, 物理约束开启
9. **实验11-12**: 重复9-10，使用 `SPARSE_SAMPLING_LEVEL='sparse'`
10. **实验13-14**: 重复9-10，使用 `SPARSE_SAMPLING_LEVEL='very_sparse'`

### 结果分析

对比以下指标的变化趋势：

1. **测试集误差** (MAE, RMSE)
   - 物理约束是否降低了误差？
   - 在哪种场景下效果最明显？

2. **单调性违反率**
   - 预测曲线是否符合单调递减？
   - 物理约束是否消除了非物理预测？

3. **预测平滑度**
   - 预测曲线是否有锯齿？
   - 物理约束是否使曲线更平滑？

---

## 注意事项

### 1. 种子设置
- 所有场景使用相同的随机种子 (`SEED=42`)
- 确保实验可复现

### 2. 测试集保持干净
- 无论选择哪个场景，测试集始终保持干净
- 这确保了公平对比 (都在相同的测试集上评估)

### 3. 场景独立性
- 场景一和场景二完全独立
- 不要同时启用两个场景
- 使用 `DEGRADATION_SCENARIO` 参数切换

### 4. 物理约束调参
- 不同场景可能需要不同的物理约束权重
- 场景一 (噪声): 可能需要降低 `curvature_weight`
- 场景二 (稀疏): 可能需要增加 `smoothness_weight`

---

## 测试脚本

运行测试脚本验证双场景系统：

```bash
python examples/test_dual_scenarios.py
```

测试内容：
- 场景一的3个噪声级别
- 场景二的4个采样级别
- 多电池稀疏采样
- 两个场景的独立性

---

## 常见问题

### Q1: 场景一和场景二可以同时使用吗？
A: 不可以。两个场景是独立的，用于不同的退化模拟。请使用 `DEGRADATION_SCENARIO` 参数选择其中一个。

### Q2: 测试集会被退化吗？
A: 不会。无论选择哪个场景，测试集始终保持干净，以确保公平对比。

### Q3: 如何选择合适的退化级别？
A:
- 场景一：从 `light` 开始，逐步增加到 `heavy`
- 场景二：从 `moderate` 开始，根据实际应用调整

### Q4: 物理约束在哪个场景下效果更好？
A: 这正是本实验要回答的问题！一般来说：
- 场景一 (噪声)：物理约束帮助过滤噪声，减少非物理预测
- 场景二 (稀疏)：物理约束帮助在稀疏点之间进行平滑插值

---

## 相关文档

- `docs/DATA_AUGMENTATION_GUIDE.md`: 数据增强详细指南
- `NOISE_AUGMENTATION_SUMMARY.md`: 噪声增强总结
- `utils/data_augmentation.py`: 数据增强实现代码

---

**更新时间**: 2025-12-15
**版本**: 1.0
