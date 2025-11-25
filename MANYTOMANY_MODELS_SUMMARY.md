# Many-to-Many 模型创建总结

## 概述

成功创建了**独立的 Many-to-Many 版本模型**，与原有的 Many-to-One 模型**共存**，而不是替换它们。现在你可以根据需求选择使用 Many-to-One 或 Many-to-Many 架构。

## 模型架构对比

| 类型 | 模型名称 | 输入形状 | 输出形状 | 用途 |
|------|---------|---------|---------|------|
| Many-to-One | LSTM, GRU, BiLSTM, BiGRU | (batch, seq_len, features) | (batch, 1) | 标准SOH预测 |
| Many-to-Many | LSTMManyToMany, GRUManyToMany, BiLSTMManyToMany, BiGRUManyToMany | (batch, seq_len, features) | (batch, seq_len, 1) | 物理约束SOH预测 |

## 创建的新模型

### 1. LSTMManyToMany
- 位置: [models/baseline_models.py](d:\Projects\1111-soh\models\baseline_models.py) (第644-708行)
- 配置文件: [configs/models/lstm_manytomany_config.json](d:\Projects\1111-soh\configs\models\lstm_manytomany_config.json)
- 输出: 整个序列的 SOH 预测 `(batch, seq_len, 1)`

### 2. GRUManyToMany
- 位置: [models/baseline_models.py](d:\Projects\1111-soh\models\baseline_models.py) (第711-775行)
- 配置文件: [configs/models/gru_manytomany_config.json](d:\Projects\1111-soh\configs\models\gru_manytomany_config.json)
- 输出: 整个序列的 SOH 预测 `(batch, seq_len, 1)`

### 3. BiLSTMManyToMany
- 位置: [models/baseline_models.py](d:\Projects\1111-soh\models\baseline_models.py) (第778-843行)
- 配置文件: [configs/models/bilstm_manytomany_config.json](d:\Projects\1111-soh\configs\models\bilstm_manytomany_config.json)
- 输出: 整个序列的 SOH 预测 `(batch, seq_len, 1)`

### 4. BiGRUManyToMany
- 位置: [models/baseline_models.py](d:\Projects\1111-soh\models\baseline_models.py) (第846-911行)
- 配置文件: [configs/models/bigru_manytomany_config.json](d:\Projects\1111-soh\configs\models\bigru_manytomany_config.json)
- 输出: 整个序列的 SOH 预测 `(batch, seq_len, 1)`

## 修改的文件

### 1. models/baseline_models.py
在文件末尾（`if __name__ == "__main__"` 之前）添加了 4 个新的 Many-to-Many 模型类。

**核心区别**:
```python
# Many-to-One (原有模型)
out = h_n[-1]  # 只用最后一个隐藏状态
out = self.fc_network(out)  # (batch, 1)

# Many-to-Many (新模型)
lstm_out_flat = lstm_out.reshape(-1, hidden_size)  # 展平所有时间步
out_flat = self.fc_network(lstm_out_flat)  # 对每个时间步预测
out = out_flat.reshape(batch_size, seq_len, 1)  # (batch, seq_len, 1)
```

### 2. models/__init__.py
添加了新模型的导出：
```python
# Many-to-Many models for SOH estimation with physics constraints
from .baseline_models import LSTMManyToMany, GRUManyToMany, BiLSTMManyToMany, BiGRUManyToMany
```

更新了 `__all__` 列表以包含新模型。

### 3. models/model_factory.py
- 添加了导入: `LSTMManyToMany, GRUManyToMany, BiLSTMManyToMany, BiGRUManyToMany`
- 更新 `SUPPORTED_MODELS` 列表，添加: `lstm_manytomany`, `gru_manytomany`, `bilstm_manytomany`, `bigru_manytomany`
- 添加了 4 个新的创建方法:
  - `_create_lstm_manytomany()`
  - `_create_gru_manytomany()`
  - `_create_bilstm_manytomany()`
  - `_create_bigru_manytomany()`
- 在 `create_model()` 中添加了相应的 elif 分支

### 4. train_cross_battery.py (第333-351行)
更新了模型检测逻辑，支持三种类型：
```python
# 检测是否为 Many-to-Many 模型
is_seq2seq = any(keyword in config_model_type for keyword in ['manytomany', 'seq2seq'])

# 检测是否需要窗口化（包括 Many-to-One 的 LSTM/GRU）
needs_window = any(model_name in config_model_type for model_name in ['lstm', 'gru'])

if is_seq2seq:
    print(f"模型使用 Many-to-Many，窗口大小: {window_size}")
elif needs_window:
    print(f"模型使用 Many-to-One，窗口大小: {window_size}")
else:
    print(f"模型使用平坦特征（window_size=1）")
```

### 5. 新增配置文件
- `configs/models/lstm_manytomany_config.json`
- `configs/models/gru_manytomany_config.json`
- `configs/models/bilstm_manytomany_config.json`
- `configs/models/bigru_manytomany_config.json`

**配置特点**:
- `shuffle: false` - Seq2Seq 需要保持时序
- `window_size: 40` - 序列长度
- 其他参数与 Many-to-One 版本一致

### 6. 新增测试文件
- [test_manytomany_coexistence.py](d:\Projects\1111-soh\test_manytomany_coexistence.py) - 验证两种模型共存

## 使用方法

### 使用 Many-to-One 模型（原有功能）

```python
# 方式1: 直接创建
from models import LSTM, GRU, BiLSTM, BiGRU

model = LSTM(input_size=16, hidden_size=64, num_layers=2)
# 输入: (batch, seq_len, features)
# 输出: (batch, 1)

# 方式2: 通过 ModelFactory
from models import ModelFactory

model = ModelFactory.create_model('lstm', input_size=16)
# 输出: (batch, 1)

# 方式3: 跨电池训练
python train_cross_battery.py
# 在脚本中设置: MODEL_TYPE = 'lstm'
```

### 使用 Many-to-Many 模型（新功能）

```python
# 方式1: 直接创建
from models import LSTMManyToMany, GRUManyToMany

model = LSTMManyToMany(input_size=16, hidden_size=64, num_layers=2)
# 输入: (batch, seq_len, features)
# 输出: (batch, seq_len, 1)

# 方式2: 通过 ModelFactory
from models import ModelFactory

model = ModelFactory.create_model('lstm_manytomany', input_size=16)
# 输出: (batch, seq_len, 1)

# 方式3: 跨电池训练
python train_cross_battery.py
# 在脚本中设置: MODEL_TYPE = 'lstm_manytomany'
```

## 物理约束支持

Many-to-Many 模型输出整个序列，可以在损失函数中添加物理约束：

### 单调性约束
```python
def monotonic_loss(predictions):
    """
    SOH 应单调递减
    predictions: (batch, seq_len, 1)
    """
    diff = predictions[:, 1:, 0] - predictions[:, :-1, 0]  # (batch, seq_len-1)
    violations = torch.relu(diff)  # 正差值表示违反单调递减
    return torch.mean(violations ** 2)
```

### 边界约束
```python
def boundary_loss(predictions):
    """
    SOH 应在 [0, 1] 范围内（Sigmoid 已保证）
    """
    lower = torch.relu(-predictions)  # 小于0的部分
    upper = torch.relu(predictions - 1)  # 大于1的部分
    return lower.mean() + upper.mean()
```

### 平滑性约束
```python
def smoothness_loss(predictions):
    """
    SOH 变化应平滑
    """
    # 二阶差分
    second_diff = (predictions[:, 2:, 0] -
                   2 * predictions[:, 1:-1, 0] +
                   predictions[:, :-2, 0])
    return torch.mean(second_diff ** 2)
```

### 完整损失函数
```python
def physics_constrained_loss(predictions, targets,
                             lambda_mono=0.1,
                             lambda_smooth=0.05):
    # 基础 MSE 损失
    mse = F.mse_loss(predictions, targets)

    # 物理约束损失
    mono = monotonic_loss(predictions)
    smooth = smoothness_loss(predictions)

    # 总损失
    total = mse + lambda_mono * mono + lambda_smooth * smooth
    return total, {'mse': mse, 'mono': mono, 'smooth': smooth}
```

## 测试结果

运行 `python test_manytomany_coexistence.py` 的结果：

```
✓ Many-to-One 模型测试:
  - LSTM: 输出 (16, 1) ✓
  - GRU: 输出 (16, 1) ✓
  - BiLSTM: 输出 (16, 1) ✓
  - BiGRU: 输出 (16, 1) ✓

✓ Many-to-Many 模型测试:
  - LSTMManyToMany: 输出 (16, 40, 1) ✓
  - GRUManyToMany: 输出 (16, 40, 1) ✓
  - BiLSTMManyToMany: 输出 (16, 40, 1) ✓
  - BiGRUManyToMany: 输出 (16, 40, 1) ✓

✓ ModelFactory 测试: 所有模型创建成功 ✓
✓ 损失函数测试: 两种模型损失计算正常 ✓
```

## 模型选择建议

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 标准SOH预测 | LSTM, GRU, BiLSTM, BiGRU (Many-to-One) | 更快，参数更少，适合单点预测 |
| 需要物理约束 | LSTMManyToMany, GRUManyToMany 等 | 输出序列，可计算物理约束 |
| 探索性研究 | Many-to-Many | 提供更丰富的序列信息 |
| 生产部署 | Many-to-One | 推理速度快，内存占用小 |

## 与 seq2seq_models.py 的关系

- **seq2seq_models.py**: 之前创建的独立 Seq2Seq 模型文件，保留以兼容性
- **baseline_models.py 中的 ManyToMany**: 新创建的模型，与 baseline 模型在同一文件中，架构相同
- 两者功能等价，可以考虑在未来统一使用 baseline_models.py 中的版本

## 注意事项

1. **数据加载**: Many-to-Many 模型需要完整的序列目标，`train_cross_battery.py` 会自动处理
2. **训练时间**: Many-to-Many 模型预测整个序列，训练时间略长于 Many-to-One
3. **内存占用**: 序列输出需要更多 GPU 内存
4. **Shuffle**: Many-to-Many 训练时建议 `shuffle=False` 以保持时序
5. **物理约束**: 目前配置文件中未启用，可以在训练脚本中自行添加

## 下一步

1. ✅ 创建独立的 Many-to-Many 模型
2. ✅ 保留原有 Many-to-One 模型
3. ✅ 集成到 ModelFactory
4. ✅ 更新 train_cross_battery.py
5. ✅ 创建配置文件
6. ✅ 测试两种模型共存
7. 🔄 在 Many-to-Many 基础上添加物理约束损失函数（可选）
8. 🔄 训练并对比两种架构的性能

## 总结

成功实现了 Many-to-One 和 Many-to-Many 模型的**共存**方案：

- ✅ 保留原有 Many-to-One 模型（LSTM, GRU, BiLSTM, BiGRU）
- ✅ 创建新的 Many-to-Many 模型（LSTMManyToMany, GRUManyToMany, BiLSTMManyToMany, BiGRUManyToMany）
- ✅ ModelFactory 支持创建两种类型
- ✅ train_cross_battery.py 智能识别模型类型
- ✅ 所有测试通过

现在你可以灵活选择使用 Many-to-One（快速、简洁）或 Many-to-Many（支持物理约束）架构进行训练！
