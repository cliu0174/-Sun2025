# 三元组采样与稀疏采样兼容性分析报告

## 执行摘要

本报告分析了**三元组采样（Triplet Sampling）**与**场景二稀疏采样（Sparse Sampling）**在跨电池训练中的兼容性问题。

**核心发现：两者存在严重的不兼容问题，稀疏采样会显著破坏三元组采样所需的连续性。**

---

## 1. 三元组采样工作原理

### 1.1 数据结构要求

三元组采样需要连续的三个样本：**(t, t+k, t+2k)**

- **t**: 当前时刻
- **t+k**: k步后的时刻
- **t+2k**: 2k步后的时刻

默认配置：`step_k=1`（相邻样本）

### 1.2 实现位置

**文件：** `d:\Projects\1111-soh\data_loaders\data_loader_hust.py`

**关键方法：** `HUSTBatteryDatasetWithMetadata._build_valid_triplets()` (第410-430行)

```python
def _build_valid_triplets(self):
    """
    Build list of valid indices for Triplet mode.
    Only keep samples that have TWO valid next samples from the SAME battery.
    Returns samples that can form (t, t+k, t+2k) triplets.
    """
    valid_indices = []

    for idx in range(len(self.X) - 2 * self.step_k):
        # Check if both next samples are from same battery
        if (self.battery_ids[idx] == self.battery_ids[idx + self.step_k] and
            self.battery_ids[idx] == self.battery_ids[idx + 2 * self.step_k]):

            # Check if cycles are exactly step_k apart for both transitions
            cycle_diff_1 = self.cycle_indices[idx + self.step_k] - self.cycle_indices[idx]
            cycle_diff_2 = self.cycle_indices[idx + 2 * self.step_k] - self.cycle_indices[idx + self.step_k]

            if cycle_diff_1 == self.step_k and cycle_diff_2 == self.step_k:
                valid_indices.append(idx)

    return valid_indices
```

### 1.3 严格性要求

三元组采样有两个严格的验证条件：

1. **电池ID一致性**：三个样本必须来自同一个电池
2. **周期间隔精确性**：`cycle_diff_1 == step_k` 且 `cycle_diff_2 == step_k`

**这意味着：当 step_k=1 时，必须是连续的相邻样本（cycle 100, 101, 102）**

---

## 2. 稀疏采样对三元组的破坏

### 2.1 场景二稀疏采样机制

**文件：** `d:\Projects\1111-soh\utils\data_augmentation.py`

**关键方法：** `apply_sparse_sampling_by_battery()` (第246-314行)

**采样规则：** 保留 `indices = [offset, offset+interval, offset+2*interval, ...]`

### 2.2 稀疏采样预设

当前配置使用 `SPARSE_SAMPLING_LEVEL = 'sparse'`：

```python
'sparse': {
    'sampling_interval': 10,
    'description': '稀疏采样 (每10个循环测试1次, 保留10%)'
}
```

**这意味着：每10个周期保留1个样本**

### 2.3 破坏示例

#### 原始数据（dense data）

```
Battery 1-1:
Cycles: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, ...]
```

#### 稀疏采样后（sampling_interval=10）

```
Battery 1-1:
Cycles: [0,                   10,                   20, ...]
          ↑                    ↑                     ↑
        保留                  保留                  保留
```

#### 三元组采样尝试（step_k=1）

- **索引0**: cycles [0, 10, 20]
  - `cycle_diff_1 = 10 - 0 = 10`
  - `cycle_diff_2 = 20 - 10 = 10`
  - **验证失败**：需要 `cycle_diff == 1`，实际为10

**结果：所有三元组都无法通过验证！**

---

## 3. 数据流程追踪

### 3.1 执行顺序

```
train_cross_battery.py (主脚本)
    ↓
prepare_cross_battery_data() (行136-298)
    ↓ 应用场景二稀疏采样 (行238-270)
    ↓ apply_sparse_sampling_by_battery()
    ↓ 每10个循环保留1个 → 样本减少90%
    ↓
create_dataloaders() (行301-533)
    ↓ 对每个电池调用 apply_windowing_with_metadata()
    ↓ 创建 HUSTBatteryDatasetWithMetadata (行389, 405, 422)
    ↓ triplet_mode=True, step_k=1
    ↓
HUSTBatteryDatasetWithMetadata.__init__() (行350-391)
    ↓ 调用 _build_valid_triplets()
    ↓ 检查 cycle_diff == step_k (必须等于1)
    ↓
    ✗ 全部验证失败 (实际cycle_diff=10)
    ↓
valid_indices = [] (空列表)
    ↓
len(dataset) = 0 (没有有效样本)
    ↓
训练失败或产生严重偏差
```

### 3.2 问题定位

**问题发生在 `HUSTBatteryDatasetWithMetadata.__getitem__()` 之前**

- 稀疏采样在**数据准备阶段**执行（行252-256）
- 三元组验证在**Dataset构造阶段**执行（行386-388）
- 稀疏采样破坏了cycle的连续性，导致三元组验证全部失败

---

## 4. 定量影响分析

### 4.1 三元组可用性

假设一个电池有1000个原始周期：

#### 场景A：无稀疏采样（dense data）

```
可用样本: 1000
可构建三元组数量: 1000 - 2*1 = 998 个
三元组可用率: 99.8%
```

#### 场景B：稀疏采样 (interval=10)

```
稀疏采样后样本: 100 (保留10%)
可构建三元组数量: 0 个 (因为cycle_diff=10, 不满足step_k=1的要求)
三元组可用率: 0%
```

### 4.2 训练数据损失

假设跨电池训练有46个电池，每个电池平均1000个周期：

#### 原始数据

```
总样本: 46,000
窗口化后 (window_size=40): ~45,960
三元组有效样本: ~45,880 (每个电池损失2个样本)
```

#### 稀疏采样后

```
稀疏采样后总样本: 4,600 (保留10%)
窗口化后: ~4,560
三元组有效样本: 0 (全部无效)
```

**结果：三元组采样完全失效！**

---

## 5. DataLoader批次采样的影响

### 5.1 DataLoader的角色

**文件：** `train_cross_battery.py` (行431-433)

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,  # ← 只打乱batch顺序
    collate_fn=custom_collate_fn
)
```

### 5.2 Shuffle不会影响三元组

**原因：**

1. **三元组在Dataset构造时已经确定**（`_build_valid_triplets()` 在 `__init__` 中执行）
2. **DataLoader的shuffle只是打乱batch的采样顺序**，不改变样本内部的三元组结构
3. **`__getitem__(idx)` 通过 `valid_indices[idx]` 访问**，三元组关系保持不变

**结论：DataLoader的shuffle对三元组采样无影响。**

---

## 6. 根本原因分析

### 6.1 设计冲突

| 维度         | 三元组采样要求          | 稀疏采样效果           | 冲突结果           |
| ------------ | ----------------------- | ---------------------- | ------------------ |
| **周期间隔** | cycle_diff == step_k    | cycle_diff == interval | ✗ 不匹配（1 vs 10） |
| **连续性**   | 需要连续的相邻样本      | 大幅跳跃，跳过中间样本 | ✗ 连续性被破坏      |
| **密度**     | 需要高密度数据          | 降低密度至10%          | ✗ 密度不足          |

### 6.2 验证逻辑过严

`_build_valid_triplets()` 要求：

```python
if cycle_diff_1 == self.step_k and cycle_diff_2 == self.step_k:
    valid_indices.append(idx)
```

**这个逻辑假设数据是完全连续的，没有考虑稀疏采样的情况。**

---

## 7. 解决方案

### 方案A：禁用三元组采样（推荐用于稀疏数据）

**修改配置文件：** `configs/models/cnn_lstm_config.json`

```json
"physics_constraints": {
  "triplet_sampling": {
    "enabled": false  // ← 关闭三元组采样
  }
}
```

**适用场景：**
- 场景二（稀疏采样）
- 数据本身已经不连续

---

### 方案B：调整step_k以匹配稀疏采样间隔（需修改代码）

**问题：** 当前代码不支持此功能，需要修改 `_build_valid_triplets()`

**修改思路：**

```python
def _build_valid_triplets(self):
    """修改后的版本：支持稀疏数据"""
    valid_indices = []

    for idx in range(len(self.X) - 2 * self.step_k):
        if (self.battery_ids[idx] == self.battery_ids[idx + self.step_k] and
            self.battery_ids[idx] == self.battery_ids[idx + 2 * self.step_k]):

            # 修改：允许 cycle_diff 是 step_k 的倍数（兼容稀疏采样）
            cycle_diff_1 = self.cycle_indices[idx + self.step_k] - self.cycle_indices[idx]
            cycle_diff_2 = self.cycle_indices[idx + 2 * self.step_k] - self.cycle_indices[idx + self.step_k]

            # 新逻辑：接受任意等间隔的三元组
            if cycle_diff_1 > 0 and cycle_diff_1 == cycle_diff_2:
                valid_indices.append(idx)

    return valid_indices
```

**效果：**
- 稀疏采样后 (interval=10): cycles [0, 10, 20] → `cycle_diff=[10, 10]` → ✓ 验证通过
- 原始密集数据 (step_k=1): cycles [0, 1, 2] → `cycle_diff=[1, 1]` → ✓ 验证通过

**优点：**
- 兼容密集数据和稀疏数据
- 三元组物理约束仍然有效（曲率约束不依赖绝对步长）

**缺点：**
- 需要修改核心代码
- 需要重新测试验证

---

### 方案C：使用孪生采样替代三元组采样

**修改配置文件：**

```json
"physics_constraints": {
  "siamese_sampling": {
    "enabled": true,   // ← 启用孪生采样
    "split_threshold": 300,
    "step_k": 1
  },
  "triplet_sampling": {
    "enabled": false   // ← 关闭三元组采样
  }
}
```

**原因：**
- 孪生采样只需要两个样本 (t, t+k)，对稀疏性的容忍度更高
- 孪生采样的验证逻辑与三元组相同，也会受稀疏采样影响
- **但是：** 孪生采样也需要修改才能适配稀疏数据（参考方案B）

---

### 方案D：先三元组采样，再稀疏采样（不推荐）

**思路：** 调整数据处理顺序

**问题：**
- 违反了物理实际（稀疏采样应该是数据采集阶段的限制）
- 会引入训练/测试不一致（测试集保持干净数据）

**结论：不推荐此方案**

---

## 8. 推荐执行方案

### 8.1 短期方案（立即可行）

**针对场景二（稀疏采样）：禁用三元组采样**

1. 修改 `train_cross_battery.py` 主脚本底部配置：
   ```python
   DEGRADATION_SCENARIO = 'scenario2'
   SPARSE_SAMPLING_LEVEL = 'sparse'  # 每10个保留1个
   ```

2. 修改 `configs/models/cnn_lstm_config.json`：
   ```json
   "triplet_sampling": {
     "enabled": false  // ← 关闭
   }
   ```

3. 使用标准物理约束（PhysicsConstrainedLoss）：
   - 软单调性约束（仍然有效）
   - 时间衰减权重（仍然有效）
   - 边界约束（仍然有效）

**效果：**
- 训练可以正常进行
- 保留大部分物理约束（除了二阶曲率约束）
- 适用于稀疏数据场景

---

### 8.2 长期方案（需要开发）

**修改三元组验证逻辑以支持稀疏数据**

1. 修改 `data_loaders/data_loader_hust.py` 的 `_build_valid_triplets()` 方法
2. 允许不等间隔的三元组（只要 `cycle_diff_1 == cycle_diff_2`）
3. 修改配置支持自适应 step_k（自动检测稀疏间隔）

**实现步骤：**

```python
# Step 1: 检测稀疏间隔
def _detect_sampling_interval(self):
    """自动检测数据的采样间隔"""
    for battery_id in np.unique(self.battery_ids):
        mask = self.battery_ids == battery_id
        cycles = self.cycle_indices[mask]
        if len(cycles) > 1:
            diffs = np.diff(cycles)
            # 返回最常见的间隔
            return np.bincount(diffs).argmax()
    return 1

# Step 2: 修改验证逻辑
def _build_valid_triplets(self):
    valid_indices = []
    detected_interval = self._detect_sampling_interval()

    for idx in range(len(self.X) - 2 * self.step_k):
        if (self.battery_ids[idx] == self.battery_ids[idx + self.step_k] and
            self.battery_ids[idx] == self.battery_ids[idx + 2 * self.step_k]):

            cycle_diff_1 = self.cycle_indices[idx + self.step_k] - self.cycle_indices[idx]
            cycle_diff_2 = self.cycle_indices[idx + 2 * self.step_k] - self.cycle_indices[idx + self.step_k]

            # 新逻辑：等间隔即可
            if cycle_diff_1 > 0 and cycle_diff_1 == cycle_diff_2:
                valid_indices.append(idx)

    return valid_indices
```

---

## 9. 实验建议

### 9.1 对比实验设计

| 实验组 | 数据场景 | 三元组采样 | 预期效果 |
| ------ | -------- | ---------- | -------- |
| **A**  | 干净数据（无退化） | ✓ 启用 | 基线性能 |
| **B**  | 稀疏采样（interval=10） | ✓ 启用 | ✗ 训练失败或性能极差 |
| **C**  | 稀疏采样（interval=10） | ✗ 禁用 | 性能下降（损失二阶约束） |
| **D**  | 稀疏采样（interval=10） | ✓ 启用（修改后） | 性能恢复（需要实现方案B） |

### 9.2 验证步骤

1. **运行实验B**（当前配置）
   ```bash
   python train_cross_battery.py
   ```
   **预期：** Dataset长度为0，或训练loss异常

2. **运行实验C**（禁用三元组）
   - 修改配置关闭 `triplet_sampling.enabled = false`
   - 重新训练
   - **预期：** 训练正常，但测试集性能下降

3. **对比实验A vs C**
   - 评估稀疏采样+无三元组的性能损失
   - 量化二阶约束的贡献

---

## 10. 总结

### 10.1 核心结论

1. **三元组采样与稀疏采样存在严重不兼容**
   - 三元组要求 `cycle_diff == step_k` (严格等于)
   - 稀疏采样导致 `cycle_diff == sampling_interval` (远大于step_k)
   - **结果：所有三元组验证失败，有效样本数=0**

2. **问题位置明确**
   - 稀疏采样：`utils/data_augmentation.py` 行246-314
   - 三元组验证：`data_loaders/data_loader_hust.py` 行410-430
   - 冲突发生在Dataset构造阶段

3. **DataLoader的shuffle不影响三元组**
   - 三元组关系在Dataset创建时固定
   - shuffle只改变batch采样顺序

### 10.2 立即行动

**针对当前实验（场景二 + 稀疏采样）：**

1. 禁用三元组采样（修改配置文件）
2. 使用标准物理约束（保留单调性+边界约束）
3. 记录性能下降，作为后续改进的baseline

### 10.3 后续工作

1. 实现方案B（修改验证逻辑）
2. 设计对比实验（A/B/C/D四组）
3. 评估二阶约束在稀疏数据下的实际贡献

---

## 附录：代码路径索引

| 功能 | 文件路径 | 行号 |
| ---- | -------- | ---- |
| 三元组验证逻辑 | `data_loaders/data_loader_hust.py` | 410-430 |
| 孪生验证逻辑 | `data_loaders/data_loader_hust.py` | 393-408 |
| Dataset构造 | `data_loaders/data_loader_hust.py` | 350-391 |
| 稀疏采样（单电池） | `utils/data_augmentation.py` | 197-243 |
| 稀疏采样（跨电池） | `utils/data_augmentation.py` | 246-314 |
| 数据准备（场景二） | `train_cross_battery.py` | 238-270 |
| 创建DataLoader | `train_cross_battery.py` | 301-533 |
| 三元组损失函数 | `models/physics_loss.py` | 531-698 |
| 孪生损失函数 | `models/physics_loss.py` | 382-528 |

---

**报告生成时间：** 2025-12-15
**分析人员：** Claude Sonnet 4.5
**代码版本：** HUST-clean branch
