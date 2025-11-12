# 统一模型工厂系统使用指南

## 📋 概述

统一模型工厂系统提供了一个标准化的接口来创建、配置和管理所有深度学习模型（FNN, CNN, LSTM, BPINN）。该系统的核心优势：

- ✅ **统一接口**: 所有模型使用相同的创建和配置方式
- ✅ **配置驱动**: 通过JSON配置文件管理超参数
- ✅ **灵活覆盖**: 支持运行时动态覆盖配置参数
- ✅ **向后兼容**: 保留原有代码，新旧系统可并存
- ✅ **易于扩展**: 添加新模型类型只需最小改动

---

## 🗂️ 项目结构

```
1111-soh/
├── configs/
│   └── models/           # 模型配置文件
│       ├── fnn_config.json
│       ├── cnn_config.json
│       ├── lstm_config.json
│       └── bpinn_config.json
│
├── models/
│   ├── __init__.py       # 统一导出接口
│   ├── model.py          # BPINN模型定义
│   ├── baseline_models.py # FNN/CNN/LSTM定义
│   ├── model_factory.py  # 🆕 统一模型工厂（推荐）
│   └── model_trainer.py  # 旧版训练器（向后兼容）
│
└── example_model_factory_usage.py  # 使用示例
```

---

## 🚀 快速开始

### 1. 基础用法：创建模型

```python
from models import ModelFactory

# 创建FNN模型（自动加载configs/models/fnn_config.json）
model = ModelFactory.create_model(
    model_type='fnn',
    input_size=6
)

# 创建BPINN模型
model = ModelFactory.create_model(
    model_type='bpinn',
    input_size=6
)

# 前向传播
import torch
x = torch.randn(32, 6)  # batch_size=32, input_size=6
output = model(x)        # shape: (32, 1)
```

### 2. 覆盖配置参数

```python
# 方法1: 直接传入覆盖参数
model = ModelFactory.create_model(
    model_type='fnn',
    input_size=6,
    hidden_sizes=[128, 64, 32],  # 覆盖默认的[64, 32, 16]
    dropout_rate=0.3             # 覆盖默认的0.2
)

# 方法2: 先加载配置再修改
from models import ConfigLoader

config = ConfigLoader.load_model_config('fnn')
config['architecture']['hidden_sizes'] = [256, 128, 64]

model = ModelFactory.create_model(
    model_type='fnn',
    input_size=6,
    config=config
)
```

### 3. 使用UnifiedModelWrapper

```python
from models import UnifiedModelWrapper

# 创建模型包装器（包含模型、优化器、损失函数）
wrapper = UnifiedModelWrapper(
    model_type='fnn',
    input_size=6,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 访问组件
model = wrapper.model
optimizer = wrapper.get_optimizer()
criterion = wrapper.criterion

# 打印模型信息
ModelFactory.print_model_info(model)
```

---

## 📖 配置文件格式

所有模型配置文件遵循统一格式：

```json
{
  "model_type": "FNN",
  "model_name": "Feedforward Neural Network",

  "architecture": {
    "input_size": -1,
    "hidden_sizes": [64, 32, 16],
    "dropout_rate": 0.2
  },

  "training": {
    "num_epochs": 2500,
    "batch_size": 64,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "scheduler": {
      "enabled": false,
      "type": "StepLR"
    }
  },

  "data": {
    "train_ratio": 0.75,
    "shuffle": true
  },

  "feature_selection": {
    "enabled": true,
    "correlation_threshold": 0.5,
    "top_k": 6
  }
}
```

### 配置字段说明

| 字段 | 说明 |
|------|------|
| `model_type` | 模型类型标识（FNN/CNN/LSTM/BPINN） |
| `architecture` | 模型架构参数（层数、神经元数等） |
| `training` | 训练超参数（学习率、批次大小等） |
| `data` | 数据处理参数 |
| `feature_selection` | 特征选择配置 |

---

## 🔧 核心API

### ModelFactory

**主要方法：**

```python
# 创建模型
model = ModelFactory.create_model(
    model_type: str,           # 'fnn', 'cnn', 'lstm', 'bpinn'
    input_size: int,           # 输入特征数量
    config: Optional[Dict] = None,      # 配置字典
    config_path: Optional[str] = None,  # 配置文件路径
    **kwargs                   # 覆盖参数
)

# 创建损失函数
criterion = ModelFactory.create_loss_function(
    model_type: str,
    config: Optional[Dict] = None,
    **kwargs
)

# 获取模型信息
info = ModelFactory.get_model_info(model)
# 返回: {'model_type', 'total_parameters', 'trainable_parameters', 'model_size_mb'}

# 打印模型信息
ModelFactory.print_model_info(model)
```

### ConfigLoader

```python
# 加载配置
config = ConfigLoader.load_model_config('fnn')

# 保存配置
ConfigLoader.save_config(config, 'custom_config.json')

# 打印配置
ConfigLoader.print_config(config)
```

### UnifiedModelWrapper

```python
# 初始化
wrapper = UnifiedModelWrapper(
    model_type='fnn',
    input_size=6,
    config=None,        # 可选
    config_path=None,   # 可选
    device='cpu',
    **kwargs
)

# 获取优化器
optimizer = wrapper.get_optimizer(learning_rate=0.001)

# 保存检查点
wrapper.save_checkpoint(
    save_path='checkpoints/model.pth',
    epoch=100,
    optimizer_state=optimizer.state_dict()
)

# 加载检查点
wrapper = UnifiedModelWrapper.load_checkpoint('checkpoints/model.pth')
```

---

## 🎯 实际应用示例

### 示例1: 对比不同模型

```python
from models import ModelFactory
import torch

input_size = 6
batch_size = 32
x = torch.randn(batch_size, input_size)

# 创建并测试所有模型
for model_type in ['fnn', 'cnn', 'lstm', 'bpinn']:
    model = ModelFactory.create_model(model_type, input_size)
    output = model(x)

    info = ModelFactory.get_model_info(model)
    print(f"{model_type.upper()}: {info['total_parameters']:,} params")
```

### 示例2: 为不同数据集创建模型

```python
# NASA数据集（6个特征）
nasa_model = ModelFactory.create_model('fnn', input_size=6)

# HUST数据集（假设8个特征）
hust_model = ModelFactory.create_model('fnn', input_size=8)

# 配置相同，但适应不同输入维度
```

### 示例3: 公平对比实验

```python
# 统一配置进行公平对比
common_config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 2000
}

results = {}
for model_type in ['fnn', 'cnn', 'lstm', 'bpinn']:
    wrapper = UnifiedModelWrapper(
        model_type=model_type,
        input_size=6
    )

    # 使用统一的训练参数
    wrapper.config['training'].update(common_config)

    # 训练模型...
    # results[model_type] = train_and_evaluate(wrapper)
```

### 示例4: 超参数调优

```python
# 网格搜索示例
learning_rates = [0.001, 0.0001]
hidden_sizes_list = [[64, 32], [128, 64, 32], [256, 128, 64]]

best_mae = float('inf')
best_config = None

for lr in learning_rates:
    for hidden_sizes in hidden_sizes_list:
        model = ModelFactory.create_model(
            model_type='fnn',
            input_size=6,
            hidden_sizes=hidden_sizes
        )

        # 训练并评估
        # mae = train_and_evaluate(model, lr)
        # if mae < best_mae:
        #     best_mae = mae
        #     best_config = {'lr': lr, 'hidden_sizes': hidden_sizes}
```

---

## 🔄 迁移指南

### 从旧代码迁移到新系统

**旧代码：**
```python
from models import FNN

model = FNN(input_size=6, hidden_sizes=[64, 32, 16], dropout_rate=0.2)
```

**新代码：**
```python
from models import ModelFactory

model = ModelFactory.create_model('fnn', input_size=6)
# 配置已从fnn_config.json自动加载
```

**优势：**
- 配置集中管理，易于复现实验
- 支持动态调整参数
- 更容易切换不同模型类型

---

## 📊 支持的模型类型

| 模型 | 类型标识 | 配置文件 | 主要用途 |
|------|---------|---------|---------|
| FNN | `'fnn'` | `fnn_config.json` | 基础前馈网络 |
| CNN | `'cnn'` | `cnn_config.json` | 1D卷积特征提取 |
| LSTM | `'lstm'` | `lstm_config.json` | 时序依赖建模 |
| BPINN | `'bpinn'` | `bpinn_config.json` | 物理约束网络 |

---

## ⚙️ 配置最佳实践

### 1. 针对不同数据集调整参数

**小数据集（<1000样本）：**
```json
{
  "architecture": {
    "hidden_sizes": [32, 16],
    "dropout_rate": 0.3
  },
  "training": {
    "batch_size": 16,
    "learning_rate": 0.0001
  }
}
```

**大数据集（>5000样本）：**
```json
{
  "architecture": {
    "hidden_sizes": [128, 64, 32],
    "dropout_rate": 0.2
  },
  "training": {
    "batch_size": 64,
    "learning_rate": 0.001
  }
}
```

### 2. 针对不同模型类型

**快速实验（调试）：**
- FNN: hidden_sizes=[32, 16], epochs=500
- BPINN: hidden_sizes=[10, 10], epochs=500

**最终性能：**
- FNN: hidden_sizes=[128, 64, 32], epochs=2500
- BPINN: hidden_sizes=[10, 10, 10], epochs=2000 + secondary=400

---

## 🐛 常见问题

### Q1: 如何保持现有代码不变？

A: 新系统完全向后兼容。旧代码继续使用 `FNN`, `CNN` 等类；新代码使用 `ModelFactory`。

### Q2: 配置文件在哪里？

A: 所有配置在 `configs/models/` 目录下。修改配置文件会影响所有使用默认配置的代码。

### Q3: 如何临时覆盖某个参数？

A: 使用kwargs传入：
```python
model = ModelFactory.create_model('fnn', input_size=6, learning_rate=0.0001)
```

### Q4: 如何添加新模型？

1. 在 `models/` 中定义模型类
2. 在 `configs/models/` 中创建配置文件
3. 在 `ModelFactory._create_xxx()` 中添加创建方法
4. 更新 `SUPPORTED_MODELS` 列表

### Q5: 如何确保实验可复现？

```python
# 保存完整配置
wrapper = UnifiedModelWrapper('fnn', input_size=6)
ConfigLoader.save_config(wrapper.config, 'experiment1_config.json')

# 加载时使用相同配置
wrapper = UnifiedModelWrapper('fnn', input_size=6, config_path='experiment1_config.json')
```

---

## 📚 更多资源

- **示例脚本**: [`example_model_factory_usage.py`](../example_model_factory_usage.py)
- **模型定义**: [`models/model_factory.py`](../models/model_factory.py)
- **配置示例**: [`configs/models/`](../configs/models/)

---

## 🎓 总结

统一模型工厂系统提供了：

1. **统一接口**：所有模型使用相同方式创建和配置
2. **配置管理**：集中式配置文件，易于版本控制
3. **灵活性**：支持运行时参数覆盖
4. **可维护性**：代码结构清晰，易于扩展
5. **可复现性**：配置文件确保实验可重复

**推荐使用新系统进行所有新开发！**

---

*最后更新: 2025-11-12*
