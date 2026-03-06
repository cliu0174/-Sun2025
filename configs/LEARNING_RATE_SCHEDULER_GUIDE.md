# 学习率调度器使用指南

本项目支持多种学习率调度策略，可在配置文件的 `scheduler` 部分灵活选择。

## 快速开始

在你的模型配置文件（如 `configs/models/cnn_lstm_config.json`）中，找到 `training.scheduler` 部分：

```json
{
  "training": {
    "learning_rate": 0.004,  // 初始学习率
    "scheduler": {
      "enabled": true,       // 是否启用调度器
      "type": "CosineAnnealingLR",  // 调度器类型
      ...                    // 其他参数
    }
  }
}
```

---

## 支持的调度器类型

### 1. **CosineAnnealingLR** - 余弦退火 ⭐ 推荐

从初始学习率平滑下降到最小学习率，曲线呈余弦形状。

**适用场景**：大多数训练场景，特别是需要精细调优的情况

**配置示例**：
```json
{
  "scheduler": {
    "enabled": true,
    "type": "CosineAnnealingLR",
    "T_max": 200,           // 总训练轮数（通常等于num_epochs）
    "eta_min": 0.0001       // 最小学习率
  }
}
```

**学习率变化**：0.004 → 平滑下降 → 0.0001


---

### 2. **CosineAnnealingWarmRestarts** - 带重启的余弦退火

周期性重启学习率，有助于跳出局部最优。

**适用场景**：复杂优化问题，容易陷入局部最优

**配置示例**：
```json
{
  "scheduler": {
    "enabled": true,
    "type": "CosineAnnealingWarmRestarts",
    "T_0": 50,              // 第一次重启的周期（epoch数）
    "T_mult": 1,            // 每次重启后周期的倍增因子（1表示固定周期）
    "eta_min": 0.0001       // 最小学习率
  }
}
```

**学习率变化**：0.004 → 下降到0.0001 → 重启到0.004 → 下降 → 重启...


---

### 3. **StepLR** - 阶梯衰减

每隔固定轮数，学习率乘以衰减因子。

**适用场景**：需要阶段性调整学习率，简单稳定

**配置示例**：
```json
{
  "scheduler": {
    "enabled": true,
    "type": "StepLR",
    "step_size": 30,        // 每30个epoch衰减一次
    "gamma": 0.5            // 衰减因子（学习率乘以0.5）
  }
}
```

**学习率变化**：
- Epoch 0-29: 0.004
- Epoch 30-59: 0.002
- Epoch 60-89: 0.001
- Epoch 90-119: 0.0005
- ...


---

### 4. **MultiStepLR** - 多阶梯衰减

在指定的epoch处降低学习率。

**适用场景**：已知训练过程中的关键节点

**配置示例**：
```json
{
  "scheduler": {
    "enabled": true,
    "type": "MultiStepLR",
    "milestones": [60, 120, 160],  // 在这些epoch处降低学习率
    "gamma": 0.2                    // 衰减因子
  }
}
```

**学习率变化**：
- Epoch 0-59: 0.004
- Epoch 60-119: 0.0008 (0.004 × 0.2)
- Epoch 120-159: 0.00016
- Epoch 160+: 0.000032


---

### 5. **ExponentialLR** - 指数衰减

每个epoch学习率乘以固定衰减因子。

**适用场景**：需要快速降低学习率

**配置示例**：
```json
{
  "scheduler": {
    "enabled": true,
    "type": "ExponentialLR",
    "gamma": 0.95           // 每个epoch学习率乘以0.95
  }
}
```

**学习率变化**：连续平滑衰减
- Epoch 0: 0.004
- Epoch 1: 0.0038
- Epoch 10: 0.0024
- Epoch 50: 0.00031
- ...


---

### 6. **ReduceLROnPlateau** - 自适应衰减

当验证指标停止改善时自动降低学习率。

**适用场景**：希望根据性能自动调整，不想手动设置epoch

**配置示例**：
```json
{
  "scheduler": {
    "enabled": true,
    "type": "ReduceLROnPlateau",
    "mode": "min",          // 'min'表示指标越小越好（如loss），'max'表示越大越好
    "factor": 0.5,          // 衰减因子
    "patience": 10,         // 容忍轮数（10个epoch没改善就降低学习率）
    "min_lr": 0.0001        // 最小学习率
  }
}
```

**学习率变化**：根据验证集性能动态调整


---

### 7. **WarmupCosineDecay** - Warmup + 余弦衰减（原有实现）

Warmup阶段线性增长，然后余弦衰减。

**适用场景**：大模型训练，需要warmup阶段稳定初期训练

**配置示例**：
```json
{
  "scheduler": {
    "enabled": true,
    "type": "WarmupCosineDecay",
    "warmup_epochs": 30,    // Warmup阶段的epoch数
    "warmup_lr": 0.002,     // Warmup起始学习率
    "base_lr": 0.005,       // Warmup结束后的最大学习率
    "final_lr": 0.0001      // 最终学习率
  }
}
```

**学习率变化**：
- Epoch 0-30: 0.002 → 线性增长到 0.005 (Warmup)
- Epoch 31-200: 0.005 → 余弦下降到 0.0001


---

### 8. **LinearDecay** - 线性衰减

从初始学习率线性下降到最终学习率。

**适用场景**：快速实验，最简单的衰减策略

**配置示例**：
```json
{
  "scheduler": {
    "enabled": true,
    "type": "LinearDecay",
    "end_lr": 0.0001        // 最终学习率
  }
}
```

**学习率变化**：0.004 → 线性下降 → 0.0001


---

## 推荐配置

### 针对你的需求（0.004 → 0.0001）

#### 方案1：CosineAnnealingLR（最推荐）
```json
{
  "training": {
    "learning_rate": 0.004,
    "num_epochs": 200,
    "scheduler": {
      "enabled": true,
      "type": "CosineAnnealingLR",
      "T_max": 200,
      "eta_min": 0.0001
    }
  }
}
```
- 优点：平滑下降，后期学习率非常小，适合精细调优
- 缺点：无

#### 方案2：StepLR（简单直接）
```json
{
  "scheduler": {
    "enabled": true,
    "type": "StepLR",
    "step_size": 40,
    "gamma": 0.5
  }
}
```
- Epoch 0-39: 0.004
- Epoch 40-79: 0.002
- Epoch 80-119: 0.001
- Epoch 120-159: 0.0005
- Epoch 160+: 0.00025

#### 方案3：MultiStepLR（精确控制）
```json
{
  "scheduler": {
    "enabled": true,
    "type": "MultiStepLR",
    "milestones": [60, 120, 160],
    "gamma": 0.25
  }
}
```
- Epoch 0-59: 0.004
- Epoch 60-119: 0.001
- Epoch 120-159: 0.00025
- Epoch 160+: 0.0000625

---

## 如何禁用学习率调度器

如果想使用固定学习率，只需设置：
```json
{
  "scheduler": {
    "enabled": false
  }
}
```

---

## 注意事项

1. **初始学习率**：在 `training.learning_rate` 中设置，这是所有调度器的起点
2. **调度器类型**：通过 `scheduler.type` 选择
3. **参数验证**：错误的参数会导致训练失败，请参考上述示例
4. **ReduceLROnPlateau**：这是唯一需要传入验证损失的调度器，代码已自动处理

---

## 可视化对比

假设初始学习率=0.004，训练200个epoch：

```
CosineAnnealingLR:        ╲╲╲╲╲╲____
StepLR (step=40):         ████▀▀▀▀▀▀
ExponentialLR:            ╲╲╲╲______
WarmupCosineDecay:        ╱▔▔╲╲╲╲___
CosineAnnealingWarmRestarts: ╲╲▔╲╲▔╲╲▔╲
LinearDecay:              ╲╲╲╲╲╲╲╲╲╲
```

---

## 实验建议

1. **首次训练**：使用 CosineAnnealingLR
2. **快速迭代**：使用 StepLR
3. **已知最佳节点**：使用 MultiStepLR
4. **自适应调整**：使用 ReduceLROnPlateau
5. **大模型/不稳定**：使用 WarmupCosineDecay

---

## 常见问题

**Q: 如何知道选择哪种调度器？**
A: 从CosineAnnealingLR开始，如果训练不稳定，尝试WarmupCosineDecay；如果想精确控制，使用MultiStepLR。

**Q: 学习率下降太快怎么办？**
A: 增大 `eta_min`、减小 `gamma`、或增大 `step_size`。

**Q: 训练初期震荡怎么办？**
A: 使用WarmupCosineDecay，增加warmup_epochs。

**Q: 如何查看当前学习率？**
A: 训练过程中会每10个epoch打印一次当前学习率。
