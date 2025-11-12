# 跨电池训练指南

## 📊 什么是跨电池训练？

**单电池训练** (`train_single_model.py`):
- 使用单个电池的数据（如1-1）
- 划分：同一电池的前75%作为训练，后25%作为测试
- 适用于：测试模型在单个电池上的性能

**跨电池训练** (`train_cross_battery.py`):
- 使用**所有77组**HUST电池数据
- 划分：不同电池分配到train/val/test
- 适用于：测试模型的**泛化能力**（在新电池上的表现）

---

## 🎯 数据划分方式

### 你的需求：Train/Val/Test = 6:2:2

```
77组电池 → 随机打乱 → 划分

训练集: 46个电池 (60%) - 用于训练模型
验证集: 16个电池 (20%) - 用于调参和选择最佳模型
测试集: 15个电池 (20%) - 用于最终评估（完全未见过的电池）
```

### 为什么要这样划分？

- ✅ **训练集**: 学习不同电池的共同规律
- ✅ **验证集**: 在训练过程中评估泛化能力，防止过拟合
- ✅ **测试集**: 评估模型在**完全未见过的电池**上的性能

---

## 🚀 使用方法

### 基础用法

```bash
python train_cross_battery.py
```

**默认配置**:
- 模型: CNN
- 划分: 6:2:2
- 设备: CUDA (如果可用)
- 随机种子: 42 (确保可复现)

### 自定义配置

编辑 `train_cross_battery.py` 底部（第548-553行）:

```python
MODEL_TYPE = 'cnn'          # 改成: 'fnn', 'lstm', 'bpinn'
TRAIN_RATIO = 0.6           # 训练集比例
VAL_RATIO = 0.2             # 验证集比例
TEST_RATIO = 0.2            # 测试集比例
DEVICE = 'cuda'             # 或 'cpu'
SEED = 42                   # 改变种子会得到不同的数据划分
```

---

## 📂 输出结果

```
results/cross_battery/cnn/
├── model_checkpoint.pth       # 训练好的模型
├── battery_split.json         # 数据划分详情（哪些电池在哪个集合）
├── results.pkl                # 详细结果
├── training_history.png       # 训练/验证曲线
└── predictions.png            # 测试集预测对比
```

### 查看数据划分

```python
import json

with open('results/cross_battery/cnn/battery_split.json', 'r') as f:
    split = json.load(f)

print(f"训练集电池: {split['train_batteries']}")
print(f"验证集电池: {split['val_batteries']}")
print(f"测试集电池: {split['test_batteries']}")
```

---

## 🔄 与单电池训练的对比

| 特性 | 单电池训练 | 跨电池训练 |
|------|-----------|-----------|
| 数据来源 | 1个电池 | 77个电池 |
| 样本数 | ~1500 | ~115,000 |
| 训练时间 | 短 (几分钟) | 长 (10-30分钟) |
| 泛化能力 | 测试同一电池 | 测试**新电池** |
| 应用场景 | 特定电池监控 | 通用SOH估计模型 |

---

## 💡 使用建议

### 场景1: 测试模型泛化能力

如果你想知道模型在**新电池**上的表现：

```bash
python train_cross_battery.py
```

查看**测试集MAE** → 这代表模型在完全未见过的电池上的性能

### 场景2: 对比不同划分比例

**8:1:1 划分**（更多训练数据）:
```python
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
```

**7:1.5:1.5 划分**（平衡训练和验证）:
```python
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
```

### 场景3: K折交叉验证

修改 `SEED` 运行多次，得到不同的数据划分：

```python
# 运行5次，每次不同的seed
for seed in [42, 123, 456, 789, 2024]:
    SEED = seed
    # 训练...
```

然后平均5次的结果。

---

## 📊 预期性能

### 单电池 vs 跨电池性能对比

**单电池训练** (train_single_model.py):
- 测试MAE: ~2-3% (在同一电池的后续循环上)
- 原因: 同一电池的数据相似性高

**跨电池训练** (train_cross_battery.py):
- 测试MAE: ~3-5% (在完全新的电池上)
- 原因: 不同电池存在个体差异

**跨电池性能通常会略低，但这才是真实场景的性能！**

---

## ⚙️ 配置调整

### 快速测试（缩短训练时间）

编辑 `configs/models/cnn_config.json`:

```json
{
  "training": {
    "num_epochs": 500,      // 从2500改为500
    "batch_size": 128       // 从64改为128
  }
}
```

### 提高性能

```json
{
  "architecture": {
    "num_filters": 128,          // 增加模型容量
    "fc_hidden_sizes": [64, 32]  // 更大的全连接层
  },
  "training": {
    "num_epochs": 3000,     // 训练更久
    "learning_rate": 0.0005 // 降低学习率
  }
}
```

---

## 🔧 高级功能

### 1. 查看训练/验证曲线

训练完成后，打开：
```
results/cross_battery/cnn/training_history.png
```

**如何判断**:
- ✅ Train loss下降，Val loss下降 → 正常训练
- ⚠️ Train loss下降，Val loss上升 → 过拟合
- ⚠️ 两者都不下降 → 欠拟合

### 2. 分析预测误差

```python
import pickle
import numpy as np

with open('results/cross_battery/cnn/results.pkl', 'rb') as f:
    results = pickle.load(f)

predictions = np.array(results['predictions'])
targets = np.array(results['targets'])
errors = predictions - targets

print(f"平均误差: {np.mean(errors)*100:.4f}%")
print(f"误差标准差: {np.std(errors)*100:.4f}%")
print(f"最大正误差: {np.max(errors)*100:.4f}%")
print(f"最大负误差: {np.min(errors)*100:.4f}%")
```

### 3. 指定具体的电池划分

如果你想手动指定哪些电池在训练/验证/测试集：

修改 `train_cross_battery.py` 的 `split_batteries()` 函数：

```python
def split_batteries(battery_names, ...):
    # 手动指定
    train_batteries = ['1-1', '1-2', '1-3', ...]
    val_batteries = ['2-1', '2-2', ...]
    test_batteries = ['3-1', '3-2', ...]

    return train_batteries, val_batteries, test_batteries
```

---

## 🎯 实际工作流程

### 第一次使用

1. **快速测试**（确保能跑通）
   ```bash
   # 先改配置: num_epochs = 100
   python train_cross_battery.py
   ```

2. **查看结果**
   ```
   results/cross_battery/cnn/
   ```

3. **如果满意，完整训练**
   ```bash
   # 改回: num_epochs = 2500
   python train_cross_battery.py
   ```

### 对比不同模型

```bash
# 训练FNN
# 修改: MODEL_TYPE = 'fnn'
python train_cross_battery.py

# 训练CNN
# 修改: MODEL_TYPE = 'cnn'
python train_cross_battery.py

# 训练LSTM
# 修改: MODEL_TYPE = 'lstm'
python train_cross_battery.py

# 对比三个模型的 test_mae
```

---

## 📈 性能优化建议

### 防止过拟合

如果验证集误差上升：

```json
{
  "architecture": {
    "dropout_rate": 0.3     // 增加dropout
  }
}
```

或者减少模型容量：
```json
{
  "architecture": {
    "fc_hidden_sizes": [16]  // 更小的网络
  }
}
```

### 提高泛化能力

1. **数据增强** - 已自动进行特征标准化
2. **正则化** - 调整 `dropout_rate`
3. **早停** - 脚本已自动实现（保存验证集最佳模型）

---

## 🆚 何时使用哪个脚本？

### 使用 `train_single_model.py` 当：
- ✅ 测试算法在单个电池上的效果
- ✅ 快速原型和调试
- ✅ 需要快速得到结果（几分钟）

### 使用 `train_cross_battery.py` 当：
- ✅ 评估模型的**真实泛化能力**
- ✅ 训练通用的SOH估计模型
- ✅ 用于论文实验（更有说服力）
- ✅ 实际部署前的性能评估

---

## 📚 相关文档

- [单电池训练指南](TRAINING_GUIDE.md)
- [快速测试指南](QUICK_TEST.md)
- [模型工厂文档](notes/MODEL_FACTORY_GUIDE.md)

---

## ✅ 总结

**跨电池训练脚本已就绪！**

```bash
# 直接运行（使用77组电池，6:2:2划分）
python train_cross_battery.py
```

**数据划分**:
- 46个电池训练
- 16个电池验证
- 15个电池测试

**评估指标**:
- 测试集MAE：模型在新电池上的真实性能
- 这比单电池训练更能反映实际应用效果！

---

*开始跨电池训练吧！* 🚀
