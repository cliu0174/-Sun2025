# Seq2Seq 模型修改总结

## 概述

成功将 LSTM、GRU、BiLSTM、BiGRU 四个基准模型从 **Many-to-One** 架构修改为 **Many-to-Many (Seq2Seq)** 架构，以支持在损失函数中引入物理约束。

## 修改动机

为了在损失函数中引入物理约束（单调性、边界、平滑性等），需要模型输出整个序列的预测值，而不是单个值。这样才能计算序列级别的物理约束损失。

- **修改前**: Many-to-One - 输入序列 → 输出单个SOH值 `(batch, 1)`
- **修改后**: Many-to-Many (Seq2Seq) - 输入序列 → 输出序列SOH值 `(batch, seq_len, 1)`

## 修改的文件

### 1. models/baseline_models.py

修改了四个RNN模型的 `forward()` 方法：

#### LSTM (第212-246行)
```python
# 修改前：只使用最后一个隐藏状态
out = h_n[-1]  # (batch, hidden_size)
out = self.fc_network(out)  # (batch, 1)

# 修改后：对每个时间步应用全连接层
lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, seq_len, hidden_size)
batch_size, seq_len, hidden_size = lstm_out.size()
lstm_out_flat = lstm_out.reshape(-1, hidden_size)  # (batch*seq_len, hidden_size)
out_flat = self.fc_network(lstm_out_flat)  # (batch*seq_len, 1)
out = out_flat.reshape(batch_size, seq_len, 1)  # (batch, seq_len, 1)
```

#### GRU (第292-320行)
- 与LSTM相同的修改逻辑
- 使用 `gru_out` 而不是 `h_n`

#### BiLSTM (第368-397行)
- 与LSTM相同的修改逻辑
- 注意：`lstm_out` 已经是双向拼接后的输出 `(batch, seq_len, hidden_size*2)`

#### BiGRU (第445-474行)
- 与GRU相同的修改逻辑
- 注意：`gru_out` 已经是双向拼接后的输出

### 2. train_cross_battery.py

更新了模型类型检测逻辑 (第332-345行)：

```python
# 修改前：
is_seq2seq = 'seq2seq' in config_model_type

# 修改后：
# 所有包含 'lstm' 或 'gru' 的模型都是Seq2Seq
is_seq2seq = any(model_name in config_model_type for model_name in ['lstm', 'gru'])
```

**重要**: 现在LSTM、GRU、BiLSTM、BiGRU都被识别为Seq2Seq模型，会自动：
- 设置 `window_size > 1`（从配置文件读取）
- 在数据加载时返回完整序列目标
- 在训练时不对目标进行 `unsqueeze(1)` 操作

### 3. test_seq2seq_integration.py (新文件)

创建了完整的测试套件，验证：
- ✓ 四个模型的输出形状正确 `(batch, seq_len, 1)`
- ✓ 输出范围正确 `[0, 1]`（Sigmoid激活）
- ✓ ModelFactory集成正常
- ✓ 损失函数计算正常

## 架构对比

| 特性 | Many-to-One (修改前) | Many-to-Many (修改后) |
|------|---------------------|----------------------|
| 输入形状 | `(batch, seq_len, features)` | `(batch, seq_len, features)` |
| 输出形状 | `(batch, 1)` | `(batch, seq_len, 1)` |
| FC层应用 | 只对最后隐藏状态 | 对每个时间步 |
| 目标形状 | `(batch,)` | `(batch, seq_len, 1)` |
| 物理约束 | ❌ 无法计算 | ✅ 可以计算 |

## 使用示例

### 训练Seq2Seq模型

```python
# 运行跨电池训练
python train_cross_battery.py

# 在脚本中设置模型类型
MODEL_TYPE = 'lstm'  # 或 'gru', 'bilstm', 'bigru'
```

脚本会自动：
1. 检测到LSTM/GRU模型
2. 设置 `is_seq2seq = True`
3. 使用配置文件中的 `window_size`（默认40）
4. 创建Seq2Seq数据加载器
5. 正确处理序列输出

### 配置文件示例

```json
{
  "model_type": "LSTM",
  "data": {
    "window_size": 40,
    "train_ratio": 0.75
  },
  "architecture": {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout_rate": 0.2
  }
}
```

## 物理约束支持

修改后的Seq2Seq架构为物理约束提供了基础。未来可以在损失函数中添加：

### 1. 单调性约束
```python
# 预测: (batch, seq_len, 1)
diff = predictions[:, 1:, 0] - predictions[:, :-1, 0]  # (batch, seq_len-1)
monotonic_violations = torch.relu(diff)  # 正差值表示违反单调递减
monotonic_loss = torch.mean(monotonic_violations ** 2)
```

### 2. 边界约束
```python
# SOH应在[0, 1]范围内（Sigmoid已保证）
boundary_loss_lower = torch.relu(-predictions).mean()
boundary_loss_upper = torch.relu(predictions - 1).mean()
```

### 3. 平滑性约束
```python
# 二阶差分
second_order_diff = predictions[:, 2:, 0] - 2*predictions[:, 1:-1, 0] + predictions[:, :-2, 0]
smoothness_loss = torch.mean(second_order_diff ** 2)
```

### 完整损失函数示例
```python
def physics_constrained_loss(predictions, targets, lambda_physics=0.1):
    # 基础MSE损失
    mse_loss = F.mse_loss(predictions, targets)

    # 物理约束损失
    diff = predictions[:, 1:, 0] - predictions[:, :-1, 0]
    monotonic_loss = torch.relu(diff).pow(2).mean()

    # 总损失
    total_loss = mse_loss + lambda_physics * monotonic_loss
    return total_loss
```

## 测试结果

运行 `python test_seq2seq_integration.py` 的结果：

```
✓ LSTM: 输出 (16, 40, 1), 参数量 56,897
✓ GRU: 输出 (16, 40, 1), 参数量 43,329
✓ BiLSTM: 输出 (16, 40, 1), 参数量 145,985
✓ BiGRU: 输出 (16, 40, 1), 参数量 110,657
✓ ModelFactory 集成正常
✓ 损失函数计算正常
```

## 与原有Seq2Seq模型的关系

- `models/seq2seq_models.py` 中的独立Seq2Seq模型仍然保留
- 基准模型现在也是Seq2Seq架构，功能等价
- 可以考虑未来删除 `seq2seq_models.py`，统一使用基准模型

## 注意事项

1. **数据加载**: `train_cross_battery.py` 会根据模型类型自动切换数据加载模式
2. **训练时间**: Seq2Seq模型需要预测整个序列，训练时间略长于Many-to-One
3. **内存占用**: 序列输出需要更多GPU内存
4. **推理**: 测试时会展平序列输出 `(batch, seq_len, 1) → (batch*seq_len,)` 用于评估
5. **物理约束**: 目前已移除，但Seq2Seq架构为未来重新引入提供了基础

## 下一步

1. ✅ 完成Seq2Seq架构修改
2. ✅ 集成到跨电池训练
3. ✅ 验证模型输出正确性
4. 🔄 在Seq2Seq基础上重新设计物理约束（可选）
5. 🔄 训练并对比Seq2Seq vs Many-to-One性能

## 总结

成功将LSTM、GRU、BiLSTM、BiGRU模型转换为Seq2Seq架构，为未来引入物理约束奠定了基础。所有修改已通过测试，可以正常用于跨电池训练。
