# 物理约束损失函数使用指南

## 概述

本项目已集成物理约束损失函数，用于改进时序模型（LSTM/GRU）的SOH预测性能。物理约束可以帮助模型：
- ✅ 遵守电池退化的物理规律（单调递减）
- ✅ 防止非物理预测（如SOH增长或超出[0,1]范围）
- ✅ 提高泛化能力和预测稳定性

---

## 快速开始

### 1. 启用物理约束

编辑模型配置文件（如 `configs/models/lstm_config.json`），修改 `physics_constraints` 部分：

```json
{
  "physics_constraints": {
    "enabled": true,              // 设为 true 启用物理约束
    "monotonic_weight": 0.1,      // 单调性约束权重（推荐 0.05-0.2）
    "boundary_weight": 0.0,       // 边界约束权重（可选）
    "smoothness_weight": 0.0      // 平滑性约束权重（可选）
  }
}
```

### 2. 正常训练

启用后，训练脚本会自动使用物理约束：

```bash
# 跨电池训练
python train_cross_battery.py

# 单电池训练
python train_single_model.py
```

训练时会显示：
```
[物理约束] 启用物理约束损失函数
  - 单调性权重: 0.1
  - 边界权重: 0.0
  - 平滑性权重: 0.0
```

### 3. 禁用物理约束

如需禁用，设置 `"enabled": false` 即可恢复标准MSE损失。

---

## 物理约束类型

### 1. 单调性约束（Monotonicity）

**原理**：确保SOH随时间单调递减（电池容量不可逆衰减）

**损失计算**：
```python
# 惩罚SOH增长
diff = pred[t+1] - pred[t]
monotonic_loss = mean(ReLU(diff)^2)  # 只惩罚正增长
```

**推荐权重**：`0.05 - 0.2`

**适用场景**：
- 时序模型（LSTM/GRU/BiLSTM/BiGRU）
- 预测曲线出现非物理波动
- 数据噪声较大

---

### 2. 边界约束（Boundary）

**原理**：确保SOH在[0, 1]范围内

**损失计算**：
```python
# 惩罚超出边界
lower_violation = ReLU(-pred)        # pred < 0
upper_violation = ReLU(pred - 1)     # pred > 1
boundary_loss = mean(lower_violation^2 + upper_violation^2)
```

**推荐权重**：`0.01 - 0.1`

**适用场景**：
- 模型预测超出物理范围
- 作为辅助约束配合单调性使用

---

### 3. 平滑性约束（Smoothness）

**原理**：惩罚SOH曲线的剧烈变化

**损失计算**：
```python
# 惩罚二阶导数
second_order_diff = pred[t+1] - 2*pred[t] + pred[t-1]
smoothness_loss = mean(second_order_diff^2)
```

**推荐权重**：`0.01 - 0.05`

**适用场景**：
- 预测曲线过于曲折
- 数据噪声导致不稳定预测

---

## 配置示例

### 示例1：仅单调性约束（推荐）

```json
{
  "physics_constraints": {
    "enabled": true,
    "monotonic_weight": 0.1,
    "boundary_weight": 0.0,
    "smoothness_weight": 0.0
  }
}
```

**适合**：大多数情况，轻量级约束

---

### 示例2：单调性 + 边界约束

```json
{
  "physics_constraints": {
    "enabled": true,
    "monotonic_weight": 0.1,
    "boundary_weight": 0.05,
    "smoothness_weight": 0.0
  }
}
```

**适合**：模型预测不稳定，需要更强约束

---

### 示例3：组合约束

```json
{
  "physics_constraints": {
    "enabled": true,
    "monotonic_weight": 0.1,
    "boundary_weight": 0.05,
    "smoothness_weight": 0.02
  }
}
```

**适合**：数据噪声大，预测波动明显

---

## 权重调整建议

### 从小到大逐步调整

1. **初始值**：`monotonic_weight = 0.1`
2. **观察效果**：
   - 如果预测仍有增长 → 增大权重至 `0.2`
   - 如果预测过于平滑 → 减小权重至 `0.05`
3. **验证指标**：观察测试集MAE/RMSE是否改善

### 权重过大的风险

⚠️ **权重过大** 会导致：
- 模型过度平滑，丢失细节
- 优化困难，收敛慢
- 可能降低预测精度

**建议**：从小权重开始，逐步增加

---

## 对比实验

### 实验设计

```python
# 实验1：无物理约束
config['physics_constraints']['enabled'] = False
# 训练并记录结果

# 实验2：单调性约束
config['physics_constraints']['enabled'] = True
config['physics_constraints']['monotonic_weight'] = 0.1
# 训练并记录结果

# 对比：测试集MAE/RMSE
```

### 预期效果

**可能的改善**：
- MAE提升 `0.1-0.3%`
- 预测曲线更平滑
- 外推能力更好

**不一定有效的情况**：
- 数据量很大且质量高
- 模型已经学到物理规律
- 非时序模型（CNN/FNN）

---

## 已配置的模型

以下模型配置文件已添加物理约束选项（默认禁用）：

- `configs/models/lstm_config.json`
- `configs/models/gru_config.json`
- `configs/models/bilstm_config.json`
- `configs/models/bigru_config.json`

**非时序模型（不推荐启用）**：
- CNN, FNN, MLP, ResCNN 不需要物理约束

---

## 技术细节

### 总损失公式

```
Total Loss = Data Loss + λ_mono * Monotonic Loss
                      + λ_bound * Boundary Loss
                      + λ_smooth * Smoothness Loss
```

### 代码实现

```python
from models import PhysicsConstrainedLoss

# 创建物理约束损失函数
criterion = PhysicsConstrainedLoss(
    base_loss='mse',
    monotonic_weight=0.1,
    boundary_weight=0.0,
    smoothness_weight=0.0
)

# 训练时使用
loss = criterion(predictions, targets)
loss.backward()
```

### 获取详细损失信息

```python
loss, details = criterion(predictions, targets, return_details=True)

print(details)
# {
#   'data_loss': 0.001234,
#   'monotonic_loss': 0.000056,
#   'boundary_loss': 0.000000,
#   'total_loss': 0.001290
# }
```

---

## FAQ

### Q1: 物理约束会增加训练时间吗？

**A**: 几乎不会。物理约束只是在损失计算时增加了几个简单的数学操作，开销可忽略。

### Q2: 所有模型都需要物理约束吗？

**A**: 不需要。主要推荐用于时序模型（LSTM/GRU）。CNN/FNN 等非时序模型通常不需要。

### Q3: 如何判断物理约束是否有效？

**A**:
1. 观察测试集MAE/RMSE是否下降
2. 绘制预测曲线，检查是否更平滑合理
3. 检查是否消除了非物理预测（如SOH增长）

### Q4: 权重设多少合适？

**A**:
- 单调性：`0.1` 是一个好的起点
- 边界：`0.05` 或不启用
- 平滑性：`0.02` 或不启用
- 根据实验效果微调

### Q5: 训练时如何查看各项损失？

**A**: 修改训练代码，使用 `return_details=True` 获取详细信息：

```python
loss, details = criterion(predictions, targets, return_details=True)
print(f"Data: {details['data_loss']:.6f}, "
      f"Monotonic: {details['monotonic_loss']:.6f}")
```

---

## 参考文献

- Physics-Informed Neural Networks (PINNs)
- 电池容量衰减模型
- 单调性约束优化

---

## 更新日志

**2024-11-24**:
- 实现单调性、边界、平滑性约束
- 集成到统一模型工厂
- 添加配置文件支持
- 完成测试和文档

---

**需要帮助？** 查看 `test_physics_constraints.py` 获取完整示例。
