# 孪生采样 + 分段约束使用指南

## 概述

本实现为PINN训练添加了**孪生/配对采样（Siamese/Pairwise Sampling）**和**分段约束（Split Constraint）**功能，用于处理电池数据中早期循环容量再生现象。

### 核心特性

1. **Dataset返回相邻配对样本**：`(x_t, x_{t+1}, y_t, y_{t+1})`
2. **分段约束**：物理损失仅应用于循环数 >= 阈值的样本
3. **向后兼容**：标准模式通过 `siamese_mode=False` 保持原有功能
4. **简单切换**：通过单一标志在两种模式间切换

---

## 快速开始

### 方法1：通过配置文件启用（推荐）

编辑 `configs/models/lstm_config.json`，找到 `physics_constraints.siamese_sampling` 部分：

```json
"siamese_sampling": {
  "enabled": true,
  "split_threshold": 300,
  "step_k": 1,
  "description": "孪生采样：前split_threshold循环无约束，后期强制单调"
}
```

然后直接运行：

```bash
python train_with_physics.py --model lstm
```

### 方法2：通过命令行参数启用

```bash
# 启用孪生模式，分段阈值设为300循环
python train_with_physics.py --model lstm --siamese --split_threshold 300
```

**注意**：命令行参数会覆盖配置文件中的设置。

---

## 配置参数

### 配置文件位置

`configs/models/lstm_config.json` (或其他模型配置文件)

### 孪生采样配置项

```json
"physics_constraints": {
  "siamese_sampling": {
    "enabled": false,          // 是否启用孪生采样模式
    "split_threshold": 300,    // 物理损失掩码的循环阈值
    "step_k": 1,               // 配对步长 (1 = 相邻样本)
    "description": "孪生采样：前split_threshold循环无约束，后期强制单调"
  }
}
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `lstm` | 模型类型 (lstm, gru, bilstm, bigru等) |
| `--device` | str | `cuda` | 计算设备 (cuda/cpu) |
| `--max_batteries` | int | `None` | 最多加载电池数量 |
| `--no_physics` | 标志 | False | 禁用物理约束 |
| `--cleaning` | 标志 | False | 启用数据清洗 |
| **`--siamese`** | **标志** | **False** | **启用孪生采样模式** |
| **`--split_threshold`** | **int** | **300** | **物理损失掩码的循环阈值** |
| **`--step_k`** | **int** | **1** | **配对步长 (1 = 相邻样本)** |

---

## 使用示例

### 示例1：基础孪生模式

```bash
python train_with_physics.py \
    --model lstm \
    --siamese \
    --split_threshold 300
```

**效果说明**：
- Dataset返回配对：`(x_t, x_{t+1})`
- 对于循环 **< 300**：仅应用MSE损失（允许容量上升）
- 对于循环 **≥ 300**：应用MSE + 物理损失（强制单调下降）

### 示例2：不同分段阈值

根据我们的数据分析：

```bash
# 保守策略 (300循环)
python train_with_physics.py --model lstm --siamese --split_threshold 300

# 中等策略 (500循环)
python train_with_physics.py --model lstm --siamese --split_threshold 500

# 激进策略 (800循环)
python train_with_physics.py --model lstm --siamese --split_threshold 800
```

### 示例3：快速测试（3个电池）

```bash
python train_with_physics.py \
    --model lstm \
    --siamese \
    --split_threshold 300 \
    --max_batteries 3
```

### 示例4：对比实验

```bash
# 基线（无物理约束）
python train_with_physics.py --model lstm --no_physics

# 标准物理约束（原有方法）
python train_with_physics.py --model lstm

# 孪生物理约束（新方法）
python train_with_physics.py --model lstm --siamese --split_threshold 300
```

---

## 工作原理

### 数据流

#### 标准模式 (`siamese_mode=False`)

```
CSV → load_single_hust_battery() → apply_windowing_with_metadata()
  ↓
Dataset返回: {'window': X_t, 'target_soh': Y_t, 'cycle_idx': cycle_t}
  ↓
训练: pred = model(X_t)
     loss = PhysicsConstrainedLoss(pred, Y_t, battery_ids, cycle_indices)
```

#### 孪生模式 (`siamese_mode=True`)

```
CSV → load_single_hust_battery() → apply_windowing_with_metadata()
  ↓
Dataset返回: {
    'x_t': X_t,
    'x_next': X_{t+1},
    'y_t': Y_t,
    'y_next': Y_{t+1},
    'cycle_index': cycle_t
}
  ↓
训练: pred_t = model(X_t)
     pred_next = model(X_{t+1})
     loss = SiamesePhysicsLoss(pred_t, pred_next, Y_t, Y_{t+1}, cycle_t)
       ↓
     if cycle_t < 300:
         loss = MSE(pred_t, Y_t) + MSE(pred_next, Y_{t+1})
     else:
         loss = MSE + 物理约束
```

### 损失函数组成

#### SiamesePhysicsLoss

```python
总损失 = MSE部分 + 物理部分

MSE部分 = (MSE(pred_t, y_t) + MSE(pred_next, y_next)) / 2
  # 对所有样本都应用

物理部分 = mask × (单调性损失 + 平滑性损失)
  # 仅在 mask = 1 时应用（cycle >= threshold）

其中:
    mask = 1 if cycle_index >= split_threshold else 0

    单调性损失 = ReLU(pred_next - pred_t - tolerance)²
      # 惩罚容量上升

    平滑性损失 = (pred_next - pred_t)²
      # 惩罚大幅跳变
```

---

## 配置说明

### 模型配置 (configs/models/lstm_config.json)

孪生模式使用以下物理约束配置：

```json
{
  "physics_constraints": {
    "enabled": true,
    "base_loss_weight": 1.0,
    "monotonic_weight": 0.1,        // 单调性约束权重
    "smoothness_weight": 0.01,      // 平滑性约束权重
    "boundary_weight": 0.05,        // 边界约束权重
    "monotonic_tolerance": 0.0      // 容忍度 (0 = 严格下降)
  }
}
```

**注意**：在孪生模式下：
- `split_threshold` 通过命令行参数设置
- `temporal_decay` 参数**不使用**（孪生模式使用直接配对）

---

## 预期效果

### 数据统计（来自分析）

| 阈值 | 早期非单调率 | 后期非单调率 | 改善幅度 |
|------|-------------|-------------|---------|
| 300循环 | 42.4% | 34.8% | 7.5% |
| 500循环 | 43.0% | 33.3% | **9.7%** |
| 800循环 | 42.5% | 30.4% | **12.2%** |

### 训练输出

```
加载 77 个电池数据 (孪生采样模式)...
成功加载 77 个电池

数据集大小:
  训练集: 45000 配对
  验证集: 15000 配对
  测试集: 20000 配对

模型: LSTM
  参数数量: 50,000

损失函数: SiamesePhysicsLoss (孪生采样 + 分段约束)
  分段阈值: 300 cycles
  单调性权重: 0.1
  平滑性权重: 0.01
  容忍度: 0.0

Epoch 1/100: Train Loss = 0.003456, Val Loss = 0.002987
  [详细] Base: 0.0023, Mono: 0.0004, Smooth: 0.0001, Bound: 0.0000, Mask: 67.3%
```

**Mask激活比例** = 循环数 >= 阈值的样本百分比（物理约束生效）

---

## 实现细节

### 修改的文件

1. **data_loaders/data_loader_hust.py**
   - `HUSTBatteryDatasetWithMetadata`: 添加 `siamese_mode` 参数
   - 当 `siamese_mode=True` 时返回配对样本

2. **models/physics_loss.py**
   - 添加 `SiamesePhysicsLoss` 类
   - 实现基于mask的分段约束

3. **models/__init__.py**
   - 导出 `SiamesePhysicsLoss`

4. **train_with_physics.py**
   - 更新 `load_batteries_with_windowing()` 支持孪生模式
   - 更新 `custom_collate_fn()` 处理两种模式
   - 更新训练循环处理两种模式
   - 添加命令行参数：`--siamese`, `--split_threshold`, `--step_k`

### 向后兼容性

✅ **所有现有功能保留**：
- 标准模式与之前完全相同
- 对现有代码无破坏性更改
- 默认参数保持原有行为

---

## 故障排除

### 问题："KeyError: 'x_t'"

**原因**：Dataset处于标准模式但训练期望孪生模式。

**解决**：确保 `siamese_mode` 一致设置：

```python
# 在 load_batteries_with_windowing() 中
siamese_mode=True

# 在 train_with_physics_constraints() 中
siamese_mode=True
```

### 问题："Mask激活比例 = 0%"

**原因**：`split_threshold` 设置过高，所有样本都被掩码。

**解决**：降低阈值：

```bash
python train_with_physics.py --siamese --split_threshold 100
```

### 问题："数据集样本过少"

**原因**：孪生模式会过滤掉没有有效下一样本的数据。

**预期**：数据集大小略微减少（step_k=1时约1%）

---

## 性能对比

### 推荐实验

| 实验 | 命令 | 预期结果 |
|------|------|---------|
| 基线 | `--model lstm --no_physics` | 最佳MSE，无约束 |
| 标准PINN | `--model lstm` | 中等，惩罚早期上升 |
| 孪生(300) | `--model lstm --siamese --split_threshold 300` | 应优于标准 |
| 孪生(500) | `--model lstm --siamese --split_threshold 500` | 可能最佳平衡 |
| 孪生(800) | `--model lstm --siamese --split_threshold 800` | 最宽松 |

---

## 下一步操作

1. **快速测试**：用3个电池验证设置
   ```bash
   python train_with_physics.py --model lstm --siamese --max_batteries 3
   ```

2. **阈值调优**：尝试不同阈值
   ```bash
   # Windows PowerShell
   foreach ($threshold in 300,500,800) {
       python train_with_physics.py --model lstm --siamese --split_threshold $threshold
   }

   # Linux/Mac
   for threshold in 300 500 800; do
       python train_with_physics.py --model lstm --siamese --split_threshold $threshold
   done
   ```

3. **完整训练**：在全部77个电池上训练
   ```bash
   python train_with_physics.py --model lstm --siamese --split_threshold 500
   ```

4. **结果分析**：比较 `results/physics_constraints/` 中的结果

---

## 技术说明

### 为什么使用孪生采样？

**问题**：早期循环显示容量再生（35-43%非单调），但标准PINN惩罚所有违规。

**解决方案**：带分段约束的孪生采样允许：
- **早期循环**：从数据自由学习（仅MSE）
- **后期循环**：强制物理约束（MSE + 物理）

### 为什么不直接修改容忍度？

将 `monotonic_tolerance` 提高到0.05（5%）有帮助，但：
- 仍然惩罚30%+的后期循环数据
- 无法区分早期vs后期循环
- 孪生模式提供精确控制

---

## 核心优势

### 与现有方法对比

| 特性 | 标准PINN | 孪生PINN |
|------|---------|---------|
| **早期循环处理** | 惩罚容量上升 ❌ | 允许容量上升 ✅ |
| **后期循环处理** | 强制单调 ✅ | 强制单调 ✅ |
| **灵活性** | 固定容忍度 | 分阶段约束 |
| **计算效率** | 需batch内查找 | 直接配对 |
| **代码复杂度** | 380行 | ~150行 |

### 适用场景

✅ **适合使用孪生模式**：
- 数据显示明显的早期容量再生现象
- 需要区分早期/后期循环的不同物理行为
- 希望精确控制物理约束的应用时机

❌ **不需要孪生模式**：
- 数据本身就是单调下降
- 容量波动可通过提高容忍度处理
- 只需要简单的全局约束

---

## 参数调优建议

### split_threshold 选择指南

根据数据分析结果：

| 阈值 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| **300** | 保守策略 | 早期自由度高 | 后期仍有34.8%非单调 |
| **500** | 推荐策略 | 较好平衡 | 中期过渡 |
| **800** | 激进策略 | 最大改善(30.4%) | 可能过于宽松 |

### 权重调优

```json
{
  "monotonic_weight": 0.05-0.1,   // 单调性：中等重要
  "smoothness_weight": 0.01-0.02, // 平滑性：低重要
  "boundary_weight": 0.05,        // 边界：固定
  "monotonic_tolerance": 0.0      // 容忍度：严格（后期）
}
```

---

## 实验流程建议

### 第一阶段：验证功能

```bash
# 1. 快速测试（3个电池，10 epochs）
python train_with_physics.py --model lstm --siamese --max_batteries 3

# 2. 检查输出
# - 数据集大小是否正确
# - Mask激活比例是否合理
# - Loss是否下降
```

### 第二阶段：阈值对比

```bash
# 分别测试不同阈值（使用10个电池）
python train_with_physics.py --model lstm --siamese --split_threshold 300 --max_batteries 10
python train_with_physics.py --model lstm --siamese --split_threshold 500 --max_batteries 10
python train_with_physics.py --model lstm --siamese --split_threshold 800 --max_batteries 10
```

### 第三阶段：完整训练

```bash
# 使用最佳阈值在全部电池上训练
python train_with_physics.py --model lstm --siamese --split_threshold 500
```

### 第四阶段：模型对比

```bash
# 基线
python train_with_physics.py --model lstm --no_physics

# 标准PINN
python train_with_physics.py --model lstm

# 孪生PINN（最佳配置）
python train_with_physics.py --model lstm --siamese --split_threshold 500
```

---

## 常见问题

### Q1: 孪生模式会显著增加训练时间吗？

A: 是的，约增加**30-50%**，因为：
- 每个样本需要两次前向传播
- 但计算更高效（无需batch内查找配对）

### Q2: 可以在测试集上使用孪生模式吗？

A: 可以，但通常**不推荐**：
- 测试时只关心预测精度
- 物理约束仅用于训练引导
- 标准单样本预测更直观

### Q3: split_threshold如何影响结果？

A:
- **过低**（如100）：物理约束应用过早，仍惩罚早期波动
- **过高**（如1000）：大部分数据无约束，失去物理引导
- **适中**（500左右）：根据数据特性达到平衡

### Q4: 能否动态调整threshold？

A: 当前版本使用固定阈值。如需动态调整，可以：
1. 在 `SiamesePhysicsLoss` 中添加渐进mask
2. 使用 soft mask（如sigmoid函数）

---

## 参考资料

- 数据分析：参见 `PINN_Analysis_Report.md` 中的相关章节
- 原始PhysicsConstrainedLoss：`models/physics_loss.py:19-380`
- SiamesePhysicsLoss：`models/physics_loss.py:382-528`
- 英文版使用指南：`SIAMESE_USAGE_GUIDE.md`

---

## 版本信息

- **生成日期**：2025-12-08
- **作者**：Claude Code
- **版本**：1.0

---

## 快速命令参考卡

```bash
# 标准模式（保持兼容）
python train_with_physics.py --model lstm

# 孪生模式（推荐配置）
python train_with_physics.py --model lstm --siamese --split_threshold 500

# 快速测试
python train_with_physics.py --model lstm --siamese --max_batteries 3

# 无物理约束基线
python train_with_physics.py --model lstm --no_physics

# 激进配置（最大容忍度）
python train_with_physics.py --model lstm --siamese --split_threshold 800

# 保守配置（早期宽松）
python train_with_physics.py --model lstm --siamese --split_threshold 300
```

---

**提示**：首次使用建议先用 `--max_batteries 3` 快速测试，确认一切正常后再进行完整训练。
