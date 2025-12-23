# 三元组采样 + 稀疏采样兼容性 - 快速修复指南

## 问题概述

**症状：** 使用场景二稀疏采样（`interval=10`）+ 三元组采样时，训练失败或Dataset长度为0

**根本原因：** 三元组验证要求 `cycle_diff == step_k (1)`，但稀疏采样后 `cycle_diff = 10`，导致所有三元组验证失败

---

## 方案1：禁用三元组采样（推荐用于紧急情况）

### 修改配置文件

**文件：** `d:\Projects\1111-soh\configs\models\cnn_lstm_config.json`

**修改第72-77行：**

```json
"triplet_sampling": {
  "enabled": false,  // ← 改为 false
  "split_threshold": 200,
  "step_k": 1,
  "description": "三元组采样：二阶曲率约束，消除预测曲线锯齿"
}
```

### 重新运行训练

```bash
python train_cross_battery.py
```

### 预期效果

- ✓ 训练可以正常进行
- ✓ 保留单调性约束、边界约束
- ✗ 损失二阶曲率约束（可能出现预测曲线锯齿）

### 适用场景

- 场景二：稀疏采样（`DEGRADATION_SCENARIO = 'scenario2'`）
- 需要立即运行实验
- 可以接受损失二阶约束

---

## 方案2：修改验证逻辑（推荐用于长期使用）

### 修改Dataset代码

**文件：** `d:\Projects\1111-soh\data_loaders\data_loader_hust.py`

**修改第410-430行的 `_build_valid_triplets()` 方法：**

#### 原始代码（行424-427）

```python
# Check if cycles are exactly step_k apart for both transitions
cycle_diff_1 = self.cycle_indices[idx + self.step_k] - self.cycle_indices[idx]
cycle_diff_2 = self.cycle_indices[idx + 2 * self.step_k] - self.cycle_indices[idx + self.step_k]

if cycle_diff_1 == self.step_k and cycle_diff_2 == self.step_k:
    valid_indices.append(idx)
```

#### 修改后的代码

```python
# Check if cycles are equally spaced (supports both dense and sparse data)
cycle_diff_1 = self.cycle_indices[idx + self.step_k] - self.cycle_indices[idx]
cycle_diff_2 = self.cycle_indices[idx + 2 * self.step_k] - self.cycle_indices[idx + self.step_k]

# New logic: accept any equally-spaced triplets
# This works for both dense data (diff=1) and sparse data (diff=10)
if cycle_diff_1 > 0 and cycle_diff_1 == cycle_diff_2:
    valid_indices.append(idx)
```

### 效果对比

| 数据类型 | 示例Cycles | cycle_diff | 原逻辑 | 新逻辑 |
| -------- | ---------- | ---------- | ------ | ------ |
| 密集数据 | [0, 1, 2] | [1, 1] | ✓ 通过 | ✓ 通过 |
| 稀疏数据 | [0, 10, 20] | [10, 10] | ✗ 失败 | ✓ 通过 |

### 预期效果

- ✓ 兼容密集和稀疏数据
- ✓ 保留二阶曲率约束（物理意义不变）
- ✓ 训练正常进行

### 验证修改

运行测试（可选）：

```bash
python -m pytest data_loaders/test_data_loader_hust.py -v
```

或者直接运行训练验证：

```bash
python train_cross_battery.py
```

---

## 方案3：使用孪生采样（折中方案）

### 修改配置文件

**文件：** `d:\Projects\1111-soh\configs\models\cnn_lstm_config.json`

**修改第66-77行：**

```json
"siamese_sampling": {
  "enabled": true,   // ← 改为 true
  "split_threshold": 300,
  "step_k": 1,
  "description": "孪生采样：前split_threshold循环无约束，后期强制单调"
},
"triplet_sampling": {
  "enabled": false,  // ← 改为 false
  "split_threshold": 200,
  "step_k": 1,
  "description": "三元组采样：二阶曲率约束，消除预测曲线锯齿"
}
```

### 注意事项

**孪生采样也受稀疏采样影响！** 需要类似方案2的修改：

**文件：** `d:\Projects\1111-soh\data_loaders\data_loader_hust.py`

**修改第393-408行的 `_build_valid_pairs()` 方法：**

```python
def _build_valid_pairs(self):
    """修改后：支持稀疏数据"""
    valid_indices = []

    for idx in range(len(self.X) - self.step_k):
        # Check if next sample is from same battery
        if self.battery_ids[idx] == self.battery_ids[idx + self.step_k]:
            # 新逻辑：只要cycle递增即可（不要求 == step_k）
            cycle_diff = self.cycle_indices[idx + self.step_k] - self.cycle_indices[idx]
            if cycle_diff > 0:  # ← 改为 > 0（原来是 == self.step_k）
                valid_indices.append(idx)

    return valid_indices
```

---

## 验证修改是否生效

### 检查Dataset长度

在 `train_cross_battery.py` 的第431行后添加调试代码：

```python
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

# 添加调试输出
print(f"\n[DEBUG] Train dataset length: {len(train_dataset)}")
print(f"[DEBUG] Train loader batches: {len(train_loader)}")
if len(train_dataset) == 0:
    print("[ERROR] Dataset is empty! Triplet validation failed.")
```

### 预期输出

#### 修改前（失败）

```
[DEBUG] Train dataset length: 0
[DEBUG] Train loader batches: 0
[ERROR] Dataset is empty! Triplet validation failed.
```

#### 修改后（成功）

```
[DEBUG] Train dataset length: 4468
[DEBUG] Train loader batches: 18
```

---

## 对比实验建议

### 实验组设计

| 实验 | 场景 | 三元组采样 | 配置 |
| ---- | ---- | ---------- | ---- |
| **A** | 干净数据 | ✓ 启用 | `DEGRADATION_SCENARIO='none'` |
| **B** | 稀疏采样 | ✗ 禁用 | `DEGRADATION_SCENARIO='scenario2'`, `triplet_sampling.enabled=false` |
| **C** | 稀疏采样 | ✓ 启用（修改后） | `DEGRADATION_SCENARIO='scenario2'`, 应用方案2 |

### 评估指标

- 训练成功率（是否能正常训练）
- 测试集MAE、RMSE
- 预测曲线平滑度（是否有锯齿）

---

## 常见问题

### Q1: 为什么不直接调整 step_k？

**A:** `step_k` 是Dataset内部的索引步长，与数据的实际cycle间隔无关。即使改为 `step_k=10`，仍然无法匹配稀疏采样后的索引。

### Q2: 修改验证逻辑会影响物理约束吗？

**A:** 不会。二阶曲率约束 `curvature = pred_3 - 2*pred_2 + pred_1` 的物理意义是"消除锯齿"，不依赖于样本的绝对时间间隔，只要求等间隔即可。

### Q3: 如果同时需要随机噪声（场景一）和三元组采样？

**A:** 场景一（随机噪声+随机丢弃）与三元组采样兼容性更好，因为随机丢弃不会系统性地破坏连续性。但仍需注意：
- 如果 `drop_ratio` 太高（>30%），可能导致部分电池的三元组大量失效
- 建议使用较低的 `drop_ratio`（如10-15%）

---

## 推荐执行方案

### 短期（立即可行）

1. 禁用三元组采样（方案1）
2. 完成场景二实验
3. 记录性能下降作为baseline

### 中期（1-2天）

1. 实现方案2（修改验证逻辑）
2. 运行对比实验（A vs B vs C）
3. 评估二阶约束的实际贡献

### 长期（后续优化）

1. 添加自动检测稀疏间隔的功能
2. 支持配置文件指定 `adaptive_step=true`
3. 完善测试用例

---

## 附录：文件路径快速索引

| 文件 | 用途 | 关键行号 |
| ---- | ---- | -------- |
| `configs/models/cnn_lstm_config.json` | 三元组采样配置 | 72-77 |
| `data_loaders/data_loader_hust.py` | 三元组验证逻辑 | 410-430 |
| `data_loaders/data_loader_hust.py` | 孪生验证逻辑 | 393-408 |
| `train_cross_battery.py` | 主训练脚本 | 1464-1528 |
| `utils/data_augmentation.py` | 稀疏采样实现 | 246-314 |

---

**最后更新：** 2025-12-15
**建议优先级：** 方案2 > 方案1 > 方案3
