# HUST电池SOH预测 - 基准模型训练系统

这是一个灵活的神经网络训练系统，用于在HUST锂离子电池数据集上训练和对比基准模型（FNN、CNN、LSTM）。

## 🎯 核心特点

### 1. **模型库架构**
- **FNN** (Feedforward Neural Network) - 简单高效的前馈网络
- **CNN** (Convolutional Neural Network) - 特征提取能力强
- **LSTM** (Long Short-Term Memory) - 序列建模能力

### 2. **灵活的配置系统**
- 每个模型有独立的JSON配置文件
- 支持命令行参数覆盖，快速调参
- 无需修改代码即可改变训练参数

### 3. **自动特征筛选**
- 基于皮尔逊相关系数自动筛选相关特征
- 可配置相关性阈值（默认0.5）
- 支持使用全部16个特征或筛选后的特征

### 4. **统一的训练流程**
- 统一的`ModelTrainer`类处理所有模型训练
- 自动选择最佳模型权重
- 详细的训练日志和结果保存

## 📦 项目结构

```
project/
├── main_hust_baseline.py              # 主训练脚本 ⭐
│
├── configs/
│   └── models/
│       ├── fnn_config.json            # FNN配置
│       ├── cnn_config.json            # CNN配置
│       └── lstm_config.json           # LSTM配置
│
├── src/
│   ├── baseline_models.py             # 模型类: FNN, CNN, LSTM
│   ├── model_trainer.py               # 统一训练器 + 配置加载器
│   ├── feature_selector.py            # 特征筛选模块
│   ├── data_loader_hust.py            # 数据加载
│   └── ...
│
├── data/
│   └── HUST data/                     # 原始数据 (77个电池, 16个特征)
│
├── results/
│   └── results_hust/
│       ├── 1-1/
│       │   ├── fnn/
│       │   │   ├── fnn_model.pth
│       │   │   └── results.pkl
│       │   ├── cnn/
│       │   └── lstm/
│       ├── 1-2/
│       └── ...
│
└── notes/
    ├── QUICK_START.md                 # 快速开始指南 ⭐
    ├── BASELINE_TRAINING_GUIDE.md     # 详细使用文档
    └── ...
```

## 🚀 快速开始

### 基础训练

```bash
# 训练FNN
python main_hust_baseline.py --model FNN --battery 1-1

# 训练CNN
python main_hust_baseline.py --model CNN --battery 1-1

# 训练LSTM
python main_hust_baseline.py --model LSTM --battery 1-1
```

### 修改参数（无需编辑配置文件）

```bash
# 修改学习率
python main_hust_baseline.py --model FNN --battery 1-1 --lr 0.0005

# 修改批量大小
python main_hust_baseline.py --model FNN --battery 1-1 --batch-size 32

# 修改训练轮数
python main_hust_baseline.py --model FNN --battery 1-1 --epochs 5000

# 禁用特征筛选（使用全部16个特征）
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.0

# 多个参数组合
python main_hust_baseline.py --model CNN --battery 1-1 \
  --lr 0.0005 --batch-size 64 --epochs 3000
```

### 批量训练

```bash
# 训练多个电池
python main_hust_baseline.py --model FNN --batteries 1-1 1-2 1-3 1-4 1-5

# 训练整个第一组（8个电池）
python main_hust_baseline.py --model FNN --batteries 1-1 1-2 1-3 1-4 1-5 1-6 1-7 1-8
```

## ⚙️ 配置文件系统

### 模型配置结构

每个配置文件包含以下部分：

```json
{
  "model_type": "FNN",
  "model_name": "Feedforward Neural Network",
  
  "architecture": {
    "input_size": 6,
    "hidden_sizes": [64, 32, 16],
    "dropout_rate": 0.2
  },
  
  "training": {
    "num_epochs": 2500,
    "batch_size": 64,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "loss_function": "MSELoss"
  },
  
  "data": {
    "train_ratio": 0.75,
    "normalize_target": true
  },
  
  "feature_selection": {
    "enabled": true,
    "correlation_threshold": 0.5
  }
}
```

### 修改配置文件

编辑 `configs/models/<model>_config.json` 可以永久改变模型配置：

#### 修改学习率
```json
"training": {
  "learning_rate": 0.0005  // 改为 0.0005
}
```

#### 修改隐藏层大小（FNN）
```json
"architecture": {
  "hidden_sizes": [128, 64, 32]  // 改为更大的网络
}
```

#### 修改卷积核数量（CNN）
```json
"architecture": {
  "num_filters": 128  // 改为 128 个卷积核
}
```

#### 禁用特征筛选
```json
"feature_selection": {
  "enabled": false  // 使用全部16个特征
}
```

或通过命令行临时修改：
```bash
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.0
```

## 📊 训练流程

```
1. 加载配置 JSON
   ↓
2. 应用命令行参数覆盖
   ↓
3. 加载电池数据
   ↓
4. 特征筛选（基于相关系数）
   ↓
5. 创建模型
   ↓
6. 训练循环
   - 前向传播
   - 计算损失
   - 反向传播
   - 更新权重
   - 保存最佳权重
   ↓
7. 最终评估
   ↓
8. 保存模型和结果
```

## 🔧 高级用法

### 特征筛选实验

```bash
# 测试不同的相关性阈值
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.0  # 全部特征
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.3
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.5  # 默认
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.7
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.9
```

### 参数调优

```bash
# 测试不同的学习率
for lr in 0.0001 0.0005 0.001 0.005; do
  python main_hust_baseline.py --model FNN --battery 1-1 --lr $lr
done

# 测试不同的批量大小
for bs in 16 32 64 128; do
  python main_hust_baseline.py --model FNN --battery 1-1 --batch-size $bs
done
```

### 多模型对比

```bash
# 在同一个电池上训练所有模型
for model in FNN CNN LSTM; do
  python main_hust_baseline.py --model $model --battery 1-1
done

# 查看结果对比
# ls results/results_hust/1-1/*/results.pkl
```

### 批量训练和统计

```bash
# 训练多个电池进行统计分析
python main_hust_baseline.py --model FNN \
  --batteries 1-1 1-2 1-3 1-4 1-5 1-6 1-7 1-8

# 显示每个电池的结果和平均值
```

## 📈 输出结果

### 目录结构

```
results/results_hust/
├── 1-1/
│   ├── fnn/
│   │   ├── fnn_model.pth        # 训练好的FNN模型
│   │   └── results.pkl          # 结果字典：配置、历史、指标
│   ├── cnn/
│   │   ├── cnn_model.pth
│   │   └── results.pkl
│   └── lstm/
│       ├── lstm_model.pth
│       └── results.pkl
├── 1-2/
│   └── ...
└── ...
```

### 结果文件内容

`results.pkl` 包含以下信息：

```python
{
    'config': {...},          # 训练使用的配置
    'history': {              # 训练历史
        'train_loss': [...],
        'test_mae': [...],
        'test_rmse': [...]
    },
    'results': {              # 最终结果
        'mae': float,
        'rmse': float,
        'best_epoch': int,
        'best_mae': float
    },
    'battery_name': str,
    'model_name': str,
    'data_info': {...}
}
```

### 读取结果

```python
import pickle

with open('results/results_hust/1-1/fnn/results.pkl', 'rb') as f:
    result = pickle.load(f)

print(f"MAE: {result['results']['mae']*100:.4f}%")
print(f"RMSE: {result['results']['rmse']*100:.4f}%")
print(f"Best Epoch: {result['results']['best_epoch']}")
```

## 💡 工作流建议

### 第一阶段：快速测试
```bash
# 测试所有模型是否正常工作
python main_hust_baseline.py --model FNN --battery 1-1 --epochs 10 --no-cuda
python main_hust_baseline.py --model CNN --battery 1-1 --epochs 10 --no-cuda
python main_hust_baseline.py --model LSTM --battery 1-1 --epochs 10 --no-cuda
```

### 第二阶段：模型选择
```bash
# 完整训练，选择最佳模型
python main_hust_baseline.py --model FNN --battery 1-1
python main_hust_baseline.py --model CNN --battery 1-1
python main_hust_baseline.py --model LSTM --battery 1-1

# 比较MAE和RMSE，选择最佳模型
```

### 第三阶段：参数调优
```bash
# 对最佳模型进行参数调优
python main_hust_baseline.py --model <best_model> --battery 1-1 --lr 0.0005
python main_hust_baseline.py --model <best_model> --battery 1-1 --lr 0.0008
python main_hust_baseline.py --model <best_model> --battery 1-1 --lr 0.001

# 测试特征筛选阈值
python main_hust_baseline.py --model <best_model> --battery 1-1 --corr-threshold 0.3
python main_hust_baseline.py --model <best_model> --battery 1-1 --corr-threshold 0.5
python main_hust_baseline.py --model <best_model> --battery 1-1 --corr-threshold 0.7
```

### 第四阶段：批量验证
```bash
# 用最优参数在多个电池上训练
python main_hust_baseline.py --model <best_model> \
  --batteries 1-1 1-2 1-3 1-4 1-5 1-6 1-7 1-8

# 验证泛化性能
```

## 🎓 理解系统架构

### ConfigLoader
```python
from src.model_trainer import ConfigLoader

# 加载配置
config = ConfigLoader.load_model_config_by_name('FNN')

# 打印配置
ConfigLoader.print_config(config)
```

### ModelFactory
```python
from src.model_trainer import ModelFactory

# 创建模型
model = ModelFactory.create_model(config, input_size=14)
```

### ModelTrainer
```python
from src.model_trainer import ModelTrainer
from src.data_loader_hust import load_single_hust_battery

# 加载数据
data = load_single_hust_battery('data/HUST data/1-1.csv')

# 创建训练器
trainer = ModelTrainer(config, device='cuda')

# 训练
history, results = trainer.train(data, verbose=True)

# 保存模型
trainer.save_model('results/model.pth')
```

## 📚 文档

- **快速开始**: `notes/QUICK_START.md` ← 从这里开始！
- **详细指南**: `notes/BASELINE_TRAINING_GUIDE.md`
- **特征分析**: `notes/HUST_BPINN_ANALYSIS.md`
- **数据信息**: `notes/DATASET_COMPARISON.md`

## 🔗 相关文件

| 文件 | 说明 |
|------|------|
| `main_hust_baseline.py` | 主训练脚本 |
| `src/baseline_models.py` | FNN, CNN, LSTM模型类 |
| `src/model_trainer.py` | 统一训练器和配置加载器 |
| `src/feature_selector.py` | 特征筛选模块 |
| `src/data_loader_hust.py` | 数据加载模块 |
| `configs/models/*.json` | 模型配置文件 |

## 🎯 常见问题

### Q: 如何修改学习率？
A: 两种方法：
1. 编辑 `configs/models/<model>_config.json` 中的 `learning_rate`
2. 使用命令行参数: `--lr 0.0005`

### Q: 如何使用全部16个特征？
A: 设置相关性阈值为0:
```bash
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.0
```

### Q: 如何固定随机种子保证可重现？
A: 使用 `--seed` 参数：
```bash
python main_hust_baseline.py --model FNN --battery 1-1 --seed 42
```

### Q: 训练太慢了怎么办？
A: 
1. 减少 `--epochs` 数量进行快速测试
2. 使用 `--batch-size 32` 或更小的批量大小
3. 使用 `--no-cuda` 用CPU训练

### Q: 如何对比不同的参数？
A: 运行多个命令，然后查看 `results/` 目录下的结果：
```bash
python main_hust_baseline.py --model FNN --battery 1-1 --lr 0.001
python main_hust_baseline.py --model FNN --battery 1-1 --lr 0.0005
python main_hust_baseline.py --model FNN --battery 1-1 --lr 0.002

# 查看 results/results_hust/1-1/fnn/results.pkl
```

## ✨ 更新日志

### v2.0 (2025-11-12)
- ✨ 完全重构为模型库 + 配置系统
- ✨ 统一的 `ModelTrainer` 类
- ✨ 支持命令行参数覆盖配置
- ✨ JSON配置文件系统
- ✨ 自动特征筛选功能
- 📚 详细的文档和快速开始指南

### v1.0 (之前)
- FNN/CNN/LSTM在同一脚本中训练

---

**项目主页**: [GitHub仓库](https://github.com/cliu0174/-Sun2025)  
**最后更新**: 2025-11-12
