# 项目使用指南

## 📁 项目结构速览

```
1111-soh/
├── models/                        # 核心模型代码
│   ├── baseline_models.py         # 5个模型: FNN/CNN/LSTM/GRU/ResCNN
│   └── model_factory.py           # 模型工厂: 统一创建和管理
│
├── configs/models/                # 配置文件
│   ├── fnn_config.json
│   ├── cnn_config.json
│   ├── lstm_config.json
│   ├── gru_config.json
│   ├── mlp_config.json            # MLP配置
│   └── rescnn_config.json         # ResCNN配置
│
├── data_loaders/                  # 数据加载
├── data/HUST data/                # 77组电池CSV
├── results/                       # 训练结果保存
│
├── train_single_model.py          # ⭐ 单电池训练
├── train_cross_battery.py         # 🌐 跨电池训练（77组）
└── plot_hust_capacity_curves.py   # 绘制电池容量曲线
```

---

## 🚀 快速开始

### 1. 训练单个模型（单电池）
```bash
# 编辑 train_single_model.py，修改第19行:
MODEL_TYPE = 'rescnn'  # 可选: 'fnn', 'cnn', 'lstm', 'gru', 'mlp', 'rescnn'

# 运行
python train_single_model.py
```

### 2. 跨电池训练（所有77组）
```bash
# 编辑 train_cross_battery.py，修改第514行:
MODEL_TYPE = 'rescnn'

# 运行
python train_cross_battery.py
```

### 3. 绘制容量曲线
```bash
python plot_hust_capacity_curves.py
```

---

## 📚 models/ 文件夹

### `baseline_models.py` - 6个模型定义

| 模型 | 类名 | 特点 | 参数量 |
|------|------|------|--------|
| 全连接网络 | `FNN` | 简单快速 | 中 |
| 卷积网络 | `CNN` | 信号处理 | 多 |
| 长短期记忆 | `LSTM` | 时序建模 | 多 |
| 门控循环 | `GRU` | 轻量时序 | 中 |
| 多层感知机 | `MLP` | 编码器-预测器 | 中 |
| **残差CNN** | **`ResCNN`** | **推荐** | **少** |

**ResCNN优势**: 参数少50%、训练稳定、收敛快

### `model_factory.py` - 统一模型管理

**3个核心类**:
- `ConfigLoader`: 加载配置文件
- `ModelFactory`: 创建模型
- `UnifiedModelWrapper`: 封装模型+训练状态

---

## 💻 常用代码

### 方式1: 直接创建模型
```python
from models import ResCNN, MLP
model = ResCNN(input_size=17)
# 或
model = MLP(input_size=17)
```

### 方式2: 使用工厂创建
```python
from models import ModelFactory

# 使用默认配置
model = ModelFactory.create_model('rescnn', input_size=17)
# 或
model = ModelFactory.create_model('mlp', input_size=17)

# 自定义参数
model = ModelFactory.create_model(
    'rescnn',
    input_size=17,
    channel_config=[16, 32, 48, 32, 16],  # 更大通道
    dropout_rate=0.2                       # 添加dropout
)
```

### 方式3: 使用Wrapper（推荐）
```python
from models import UnifiedModelWrapper

wrapper = UnifiedModelWrapper('rescnn', input_size=17, device='cuda')
# 或
wrapper = UnifiedModelWrapper('mlp', input_size=17, device='cuda')
model = wrapper.model
optimizer = wrapper.get_optimizer()
criterion = wrapper.criterion

# 训练...

# 保存
wrapper.save_checkpoint('model.pth', epoch=100)

# 加载
wrapper = UnifiedModelWrapper.load_checkpoint('model.pth')
```

---

## ⚙️ 训练脚本详解

### `train_single_model.py` ⭐
- **用途**: 在单个电池上训练和测试模型
- **修改**: 第19行 `MODEL_TYPE = 'rescnn'`
- **输出**: `results/single_battery/rescnn/`
- **推荐**: 初步测试新模型时使用

### `train_cross_battery.py` 🌐
- **用途**: 使用所有77组电池训练通用模型
- **数据划分**: Train/Val/Test = 46/16/15 个电池
- **修改**: 第514行 `MODEL_TYPE = 'rescnn'`
- **输出**: `results/cross_battery/rescnn/`
- **推荐**: 评估模型泛化能力时使用

### `plot_hust_capacity_curves.py` 📈
- **用途**: 可视化所有电池的容量衰减曲线
- **输出**: 容量曲线图
- **推荐**: 分析数据特性时使用

---

## 🎛️ 自定义配置

### 方式1: 修改配置文件
编辑 `configs/models/rescnn_config.json`:
```json
{
  "architecture": {
    "channel_config": [8, 16, 24, 16, 8],
    "stride_config": [1, 2, 2, 1, 1],
    "dropout_rate": 0.0
  },
  "training": {
    "num_epochs": 600,
    "learning_rate": 0.0001,
    "batch_size": 256
  }
}
```

### 方式2: 代码中覆盖
```python
model = ModelFactory.create_model(
    'rescnn',
    input_size=17,
    channel_config=[4, 8, 12, 8, 4],  # 小模型
    dropout_rate=0.3                   # 高dropout
)
```

---

## 📊 结果文件

训练后会生成：
```
results/{single_battery|cross_battery}/rescnn/
├── model_checkpoint.pth      # 模型权重
├── training_history.png      # 训练曲线
├── predictions.png           # 预测对比图
├── results.pkl               # 详细结果
└── battery_split.json        # 数据划分（仅cross_battery）
```

---

## 🔧 models/ 核心代码位置

| 内容 | 文件 | 行数 |
|------|------|------|
| ResCNN模型定义 | `baseline_models.py` | 296-364 |
| ResBlock定义 | `baseline_models.py` | 259-293 |
| ResCNN工厂方法 | `model_factory.py` | 219-227 |
| 支持的模型列表 | `model_factory.py` | 113 |

---

## 🆘 常见问题

### Q: 如何在train_cross_battery中使用ResCNN?
A: 修改第514行: `MODEL_TYPE = 'rescnn'`

### Q: 训练很慢怎么办?
A: 确保使用GPU: `DEVICE = 'cuda'`

### Q: 如何调整学习率?
A: 修改配置文件 `learning_rate: 0.0001` 或代码中覆盖

### Q: 模型过拟合怎么办?
A: 增加dropout: `dropout_rate: 0.0 → 0.3`

---

## 📖 文档导航

- **README.md** - 项目总览
- **USAGE_GUIDE.md** (本文档) - 使用指南
- **RESCNN_README.md** - ResCNN详细说明

---

**最后更新**: 2025-11-14 | **版本**: 2.0 (添加ResCNN)
