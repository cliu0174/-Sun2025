# 物理约束使用指南

## 🎉 实现完成！

物理约束损失函数已成功集成到项目中，适用于 Many-to-One 模型（LSTM, GRU, BiLSTM, BiGRU）。

---

## ✨ 核心特性

### 1. **软单调性约束**
- 允许 `tolerance` 范围内的轻微上升（默认 1%）
- 只惩罚明显违反物理规律的上升趋势
- 公式：`ReLU(SOH[t+k] - SOH[t] - tolerance)`

### 2. **时间衰减权重**
- 相邻 cycle 约束强（k=1, weight=0.82）
- 远距离 cycle 约束弱（k=20, weight=0.02）
- 处理 batch 内样本稀疏问题
- 支持三种衰减类型：指数、线性、倒数

### 3. **边界约束**
- 确保 SOH ∈ [0, 1]
- 惩罚越界预测

### 4. **平滑性约束**（可选）
- 惩罚剧烈的 SOH 变化
- 使用二阶差分

---

## 📁 新增文件

### 1. 核心实现
- `models/physics_loss.py` - 物理约束损失函数
- `train_with_physics.py` - 带物理约束的训练脚本

### 2. 数据处理
- `data_loaders/data_loader_hust.py` - 添加了：
  - `apply_windowing_with_metadata()` 函数
  - `HUSTBatteryDatasetWithMetadata` 类

### 3. 配置文件
- `configs/models/lstm_config.json` - 添加 `physics_constraints` 配置项
- `configs/models/gru_config.json`
- `configs/models/bilstm_config.json`
- `configs/models/bigru_config.json`

### 4. 测试和文档
- `test_physics_integration.py` - 集成测试脚本
- `PHYSICS_CONSTRAINTS_IMPLEMENTATION_PLAN.md` - 详细实现方案
- `PHYSICS_CONSTRAINTS_USAGE.md` - 本文档

---

## 🚀 快速开始

### 方法 1: 使用新的训练脚本（推荐）

```bash
# 使用物理约束训练 LSTM（默认启用）
python train_with_physics.py --model lstm

# 训练 GRU
python train_with_physics.py --model gru

# 不使用物理约束（对比实验）
python train_with_physics.py --model lstm --no_physics

# 使用数据清洗
python train_with_physics.py --model lstm --cleaning

# 限制电池数量（快速测试）
python train_with_physics.py --model lstm --max_batteries 10
```

### 方法 2: 在配置文件中启用

编辑 `configs/models/lstm_config.json`：

```json
{
  "physics_constraints": {
    "enabled": true,  // 改为 true
    "monotonic_weight": 0.1,
    "monotonic_tolerance": 0.01,
    "temporal_decay": {
      "enabled": true,
      "max_step": 20,
      "decay_alpha": 0.2
    }
  }
}
```

然后使用 `train_with_physics.py` 训练。

---

## ⚙️ 配置参数详解

### 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | false | 是否启用物理约束 |
| `base_loss_weight` | float | 1.0 | 基础 MSE 损失权重 |
| `monotonic_weight` | float | 0.1 | 单调性约束权重 |
| `boundary_weight` | float | 0.05 | 边界约束权重 |
| `smoothness_weight` | float | 0.0 | 平滑性约束权重 |

### 软约束参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `monotonic_tolerance` | float | 0.01 | 容忍度（允许的最大上升幅度）|

**示例**：
- `0.01` = 允许 1% 的上升
- `0.02` = 允许 2% 的上升（更宽松）
- `0.005` = 允许 0.5% 的上升（更严格）

### 时间衰减参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temporal_decay.enabled` | bool | true | 是否启用时间衰减 |
| `temporal_decay.max_step` | int | 20 | 最大考虑的时间步长 |
| `temporal_decay.decay_type` | str | "exp" | 衰减类型 |
| `temporal_decay.decay_alpha` | float | 0.2 | 衰减系数 |

**衰减类型**：
- `"exp"`: 指数衰减 `w(k) = exp(-alpha * k)`
- `"linear"`: 线性衰减 `w(k) = max(0, 1 - alpha * k)`
- `"inverse"`: 倒数衰减 `w(k) = 1 / k`

**权重示例**（alpha=0.2, type="exp"）：

| 步长 k | 权重 | 物理意义 |
|--------|------|----------|
| 1 | 0.82 | 相邻，最强约束 |
| 2 | 0.67 | 隔1个 |
| 5 | 0.37 | 隔4个 |
| 10 | 0.14 | 隔9个 |
| 20 | 0.02 | 隔19个，很弱 |

---

## 📊 配置推荐

### 保守配置（轻度约束）

适用于：初次尝试，数据质量好

```json
{
  "physics_constraints": {
    "enabled": true,
    "monotonic_weight": 0.05,
    "boundary_weight": 0.02,
    "smoothness_weight": 0.0,
    "monotonic_tolerance": 0.02,
    "temporal_decay": {
      "enabled": true,
      "max_step": 10,
      "decay_alpha": 0.15
    }
  }
}
```

### 标准配置（中等约束）

适用于：一般情况，推荐使用

```json
{
  "physics_constraints": {
    "enabled": true,
    "monotonic_weight": 0.1,
    "boundary_weight": 0.05,
    "smoothness_weight": 0.0,
    "monotonic_tolerance": 0.01,
    "temporal_decay": {
      "enabled": true,
      "max_step": 20,
      "decay_alpha": 0.2
    }
  }
}
```

### 激进配置（强约束）

适用于：数据噪声大，需要强约束

```json
{
  "physics_constraints": {
    "enabled": true,
    "monotonic_weight": 0.2,
    "boundary_weight": 0.1,
    "smoothness_weight": 0.05,
    "monotonic_tolerance": 0.005,
    "temporal_decay": {
      "enabled": true,
      "max_step": 30,
      "decay_alpha": 0.3
    }
  }
}
```

---

## 🧪 测试和验证

### 1. 运行集成测试

```bash
python test_physics_integration.py
```

**预期输出**：
```
======================================================================
测试 1: Windowing + Metadata
[OK] Windowing 测试通过

测试 2: Dataset 创建
[OK] Dataset 测试通过

测试 3: DataLoader
[OK] DataLoader 测试通过

测试 4: 物理损失计算
[OK] 物理损失测试通过

所有测试通过！
======================================================================
```

### 2. 对比实验

```bash
# 实验 1: 无物理约束（baseline）
python train_with_physics.py --model lstm --no_physics

# 实验 2: 有物理约束
python train_with_physics.py --model lstm

# 对比结果
# 结果保存在: results/physics_constraints/
```

---

## 📈 工作原理

### 数据流

```
原始数据 (1487 cycles, 16 features)
         ↓
  load_single_hust_battery()
         ↓
  (train: 1115, test: 372)
         ↓
  apply_windowing_with_metadata(window_size=40)
         ↓
  (N=1076 windows, each with battery_id & cycle_idx)
         ↓
  HUSTBatteryDatasetWithMetadata
         ↓
  DataLoader (batch_size=256, shuffle=True)
         ↓
  每个 batch 包含:
  - window: (256, 40, 16)
  - target_soh: (256, 1)
  - battery_id: ["1-1", "6-2", ...]  # 256 个
  - cycle_idx: [50, 40, 55, ...]     # 256 个
```

### 物理约束计算

```
Batch 内 (打乱后):
[
  {"battery_id": "1-1", "cycle": 50, "pred": 0.92},
  {"battery_id": "6-2", "cycle": 40, "pred": 0.88},
  {"battery_id": "1-1", "cycle": 55, "pred": 0.91},
  {"battery_id": "1-1", "cycle": 70, "pred": 0.90},
  ...
]
         ↓
  按 battery_id 分组
         ↓
电池 "1-1": [50, 55, 70] → 排序 → 配对
电池 "6-2": [40, 45] → 排序 → 配对
         ↓
  计算所有配对的约束
         ↓
  (50, 55): k=5, w=0.37, violation=ReLU(0.91-0.92-0.01)=0
  (50, 70): k=20, w=0.02, violation=ReLU(0.90-0.92-0.01)=0
  (55, 70): k=15, w=0.05, violation=ReLU(0.90-0.91-0.01)=0
         ↓
  总损失 = base_loss + 0.1 * mono_loss + 0.05 * bound_loss
```

---

## 🔧 调试技巧

### 1. 查看详细损失

在训练脚本中：

```python
if isinstance(criterion, PhysicsConstrainedLoss):
    details = criterion.get_loss_details()
    print(f"Base: {details['base']:.6f}")
    print(f"Monotonic: {details['monotonic']:.6f}")
    print(f"Boundary: {details['boundary']:.6f}")
    print(f"Smoothness: {details['smoothness']:.6f}")
```

### 2. 启用 verbose 模式

```python
criterion = PhysicsConstrainedLoss(
    ...,
    verbose=True  # 打印详细的配对信息
)
```

### 3. 检查 batch 内配对

```python
# 在训练循环中
battery_ids = batch['battery_id']
cycle_indices = batch['cycle_idx']

print(f"Batch 内电池: {set(battery_ids)}")
for bid in set(battery_ids):
    indices = [i for i, b in enumerate(battery_ids) if b == bid]
    cycles = cycle_indices[indices]
    print(f"  {bid}: {len(indices)} 个样本, cycles: {sorted(cycles.tolist())}")
```

---

## ⚠️ 注意事项

1. **Only for Many-to-One models**
   - 当前实现仅支持 LSTM, GRU, BiLSTM, BiGRU 等输出单点的模型
   - Many-to-Many (Seq2Seq) 模型需要不同的实现

2. **Batch 内样本可能稀疏**
   - 由于 `shuffle=True`，一个 batch 内同一电池的样本可能很少
   - 时间衰减权重正是为了处理这种情况

3. **权重调整**
   - 如果损失不收敛，降低 `monotonic_weight`
   - 如果物理约束不明显，增加 `monotonic_weight`
   - 建议从保守配置开始，逐步调整

4. **训练时间**
   - 物理约束会增加计算时间（~10-20%）
   - 主要来自 batch 内的配对计算

5. **数据清洗**
   - 物理约束和数据清洗可以同时使用
   - 建议先清洗数据，再应用物理约束

---

## 📚 参考资料

- **实现方案**：`PHYSICS_CONSTRAINTS_IMPLEMENTATION_PLAN.md`
- **核心代码**：`models/physics_loss.py`
- **训练脚本**：`train_with_physics.py`
- **测试代码**：`test_physics_integration.py`

---

## 🎯 下一步

1. **对比实验**
   ```bash
   python train_with_physics.py --model lstm --no_physics  # baseline
   python train_with_physics.py --model lstm                # with physics
   ```

2. **参数调优**
   - 调整 `monotonic_weight` (0.05 ~ 0.2)
   - 调整 `monotonic_tolerance` (0.005 ~ 0.02)
   - 尝试不同的衰减类型

3. **可视化**
   - 绘制物理损失曲线
   - 可视化约束违反情况
   - 分析不同配置的效果

---

**实现完成！祝训练顺利！** 🎉
