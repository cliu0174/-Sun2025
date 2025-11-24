# ResCNN 模型详细说明

## 概述

ResCNN（Residual Convolutional Neural Network）是基于残差块的1D卷积神经网络，专为电池SOH估计设计。

**核心优势**:
- 参数量比普通CNN少50%（8,465 vs 17,153）
- 残差连接缓解梯度消失
- BatchNorm加速收敛
- 灵活的架构配置

---

## 快速使用

### 方式1: 直接创建
```python
from models import ResCNN
model = ResCNN(input_size=17)
```

### 方式2: 使用工厂
```python
from models import ModelFactory
model = ModelFactory.create_model('rescnn', input_size=17)
```

### 方式3: 自定义参数
```python
model = ModelFactory.create_model(
    'rescnn',
    input_size=17,
    channel_config=[16, 32, 48, 32, 16],  # 更大通道
    stride_config=[1, 2, 2, 1, 1],
    dropout_rate=0.2
)
```

---

## 模型架构

### ResBlock结构
```
输入 → [Conv(3x3)+BN+ReLU → Conv(3x3)+BN] + Skip Connection → ReLU → 输出
```

### 完整网络（默认配置）

| 层 | 输入形状 | 输出形状 | 通道 | Stride |
|---|---------|---------|------|--------|
| Input | (N, 17) | (N, 1, 17) | - | - |
| ResBlock1 | (N, 1, 17) | (N, 8, 17) | 1→8 | 1 |
| ResBlock2 | (N, 8, 17) | (N, 16, 9) | 8→16 | 2 |
| ResBlock3 | (N, 16, 9) | (N, 24, 5) | 16→24 | 2 |
| ResBlock4 | (N, 24, 5) | (N, 16, 5) | 24→16 | 1 |
| ResBlock5 | (N, 16, 5) | (N, 8, 5) | 16→8 | 1 |
| Flatten | (N, 8, 5) | (N, 40) | - | - |
| Linear+Sigmoid | (N, 40) | (N, 1) | - | - |

---

## 配置文件

位置: `configs/models/rescnn_config.json`

```json
{
  "architecture": {
    "input_size": -1,
    "channel_config": [8, 16, 24, 16, 8],
    "stride_config": [1, 2, 2, 1, 1],
    "dropout_rate": 0.0
  },
  "training": {
    "num_epochs": 600,
    "batch_size": 256,
    "learning_rate": 0.0001
  }
}
```

---

## 推荐配置

### 小数据集 (< 1000样本)
```python
channel_config=[4, 8, 12, 8, 4]   # 减小通道
stride_config=[1, 2, 1, 1, 1]     # 减少下采样
dropout_rate=0.3                   # 增加正则化
```

### 大数据集 (> 5000样本)
```python
channel_config=[16, 32, 48, 32, 16]  # 增大通道
stride_config=[1, 2, 2, 1, 1]
dropout_rate=0.1
```

### 高维输入 (> 30特征)
```python
channel_config=[8, 16, 32, 24, 16, 8]  # 增加层数
stride_config=[1, 2, 2, 2, 1, 1]        # 更多下采样
```

---

## 训练示例

### 单电池训练
```bash
# 编辑 train_single_model.py
MODEL_TYPE = 'rescnn'

# 运行
python train_single_model.py
```

### 跨电池训练
```bash
# 编辑 train_cross_battery.py
MODEL_TYPE = 'rescnn'

# 运行
python train_cross_battery.py
```

---

## 完整代码示例

```python
import torch
from models import UnifiedModelWrapper
from data_loaders import load_single_hust_battery

# 1. 加载数据
data = load_single_hust_battery('data/HUST data/Cell_1.csv')

# 2. 创建模型
wrapper = UnifiedModelWrapper(
    'rescnn',
    input_size=data['train_features'].shape[1],
    device='cuda'
)

model = wrapper.model
optimizer = wrapper.get_optimizer()
criterion = wrapper.criterion

# 3. 训练
for epoch in range(100):
    # 训练代码...
    pass

# 4. 保存
wrapper.save_checkpoint('rescnn_model.pth', epoch=100)
```

---

## 与其他模型对比

| 模型 | 参数量 | 训练稳定性 | 适用场景 |
|------|--------|-----------|---------|
| FNN | 中 | ⭐⭐ | 简单特征 |
| CNN | 17,153 | ⭐⭐ | 信号处理 |
| LSTM | 多 | ⭐⭐ | 时序数据 |
| GRU | 中 | ⭐⭐ | 时序轻量 |
| **ResCNN** | **8,465** | **⭐⭐⭐** | **深层特征(推荐)** |

---

## 调试技巧

### 查看模型结构
```python
from models import ResCNN, ModelFactory
model = ResCNN(input_size=17)
print(model)
ModelFactory.print_model_info(model)
```

### 检查输出形状
```python
import torch
x = torch.randn(1, 17)
y = model(x)
print(f"Input: {x.shape}, Output: {y.shape}")
```

---

## 常见问题

**Q: 输出不在[0,1]范围？**
A: 检查模型最后是否有Sigmoid激活函数

**Q: 训练loss不下降？**
A: 降低学习率 (0.0001 → 0.00001)，增加dropout

**Q: 过拟合怎么办？**
A: `dropout_rate: 0.0 → 0.3`，减少通道数

**Q: 如何处理不同维度输入？**
A: 只需设置 `input_size` 参数，模型自动适配

---

## 项目集成状态

✅ 已添加到 `models/baseline_models.py`
✅ 已集成到模型工厂
✅ 已添加配置文件
✅ 已支持 train_cross_battery.py
✅ 已创建使用示例

---

**代码位置**:
- 模型定义: `models/baseline_models.py` (第296-364行)
- 工厂方法: `models/model_factory.py` (第219-227行)
- 配置文件: `configs/models/rescnn_config.json`

**创建日期**: 2025-11-14 | **版本**: 1.0
