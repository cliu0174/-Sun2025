# HUST数据集基准模型训练指南

## 🏗️ 项目结构

```
project/
├── configs/
│   └── models/
│       ├── fnn_config.json      # FNN配置
│       ├── cnn_config.json      # CNN配置
│       └── lstm_config.json     # LSTM配置
├── src/
│   ├── baseline_models.py       # 模型库
│   ├── model_trainer.py         # 统一训练器
│   ├── feature_selector.py      # 特征筛选
│   ├── data_loader_hust.py      # 数据加载
│   └── ...
├── data/
│   └── HUST data/               # 数据文件
├── results/
│   └── results_hust/            # 训练结果
└── main_hust_baseline.py        # 主训练脚本
```

## 📋 配置文件说明

每个模型都有一个JSON配置文件，包含以下部分：

### 1. 基本信息
```json
{
  "model_type": "FNN",                    // 模型类型
  "model_name": "Feedforward Neural Network",
  "description": "简单的前馈神经网络...",
```

### 2. 架构配置 (architecture)
```json
"architecture": {
  "input_size": 6,              // 输入特征数 (自动覆盖)
  "hidden_sizes": [64, 32, 16], // 隐藏层大小
  "dropout_rate": 0.2           // Dropout比率
}
```

**FNN特定参数:**
- `hidden_sizes`: 隐藏层大小列表

**CNN特定参数:**
- `num_filters`: 卷积核数量
- `kernel_size`: 卷积核大小
- `fc_hidden_sizes`: 全连接层大小

**LSTM特定参数:**
- `hidden_size`: LSTM隐藏层大小
- `num_layers`: LSTM层数
- `fc_hidden_sizes`: 全连接层大小

### 3. 训练参数 (training)
```json
"training": {
  "num_epochs": 2500,           // 训练轮数
  "batch_size": 64,             // 批量大小
  "learning_rate": 0.001,       // 学习率
  "optimizer": "Adam",          // 优化器
  "loss_function": "MSELoss"    // 损失函数
}
```

### 4. 特征筛选 (feature_selection)
```json
"feature_selection": {
  "enabled": true,              // 是否启用特征筛选
  "correlation_threshold": 0.5  // 相关系数阈值
}
```

当启用时，只有|相关系数| ≥ 阈值的特征才会被使用。

### 5. 数据设置 (data)
```json
"data": {
  "train_ratio": 0.75,          // 训练集比例
  "shuffle": true,              // 是否打乱数据
  "normalize_target": true      // 是否归一化目标值
}
```

## 🚀 使用方法

### 基础训练

#### 训练单个电池

```bash
# 训练FNN (使用默认配置)
python main_hust_baseline.py --model FNN --battery 1-1

# 训练CNN
python main_hust_baseline.py --model CNN --battery 1-2

# 训练LSTM
python main_hust_baseline.py --model LSTM --battery 1-3
```

#### 训练多个电池

```bash
# 同时训练3个电池
python main_hust_baseline.py --model FNN --batteries 1-1 1-2 1-3

# 训练第一组的所有电池
python main_hust_baseline.py --model CNN --batteries 1-1 1-2 1-3 1-4 1-5 1-6 1-7 1-8
```

### 参数覆盖

可以通过命令行参数临时覆盖配置文件中的值，无需修改JSON文件：

```bash
# 修改学习率
python main_hust_baseline.py --model FNN --battery 1-1 --lr 0.0005

# 修改批量大小
python main_hust_baseline.py --model FNN --battery 1-1 --batch-size 32

# 修改训练轮数
python main_hust_baseline.py --model FNN --battery 1-1 --epochs 5000

# 修改特征相关性阈值 (不筛选特征, 全部16个特征)
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.0

# 多个参数组合
python main_hust_baseline.py --model CNN --battery 1-1 \
  --lr 0.0005 --batch-size 64 --epochs 3000 --corr-threshold 0.6
```

### 高级选项

```bash
# 使用CPU训练 (不用GPU)
python main_hust_baseline.py --model LSTM --battery 1-1 --no-cuda

# 设置随机种子 (复现结果)
python main_hust_baseline.py --model FNN --battery 1-1 --seed 42
```

## 🔧 修改配置文件

### 永久修改学习率

编辑 `configs/models/fnn_config.json`:

```json
"training": {
  "num_epochs": 2500,
  "batch_size": 64,
  "learning_rate": 0.0005,    // ← 修改这里
  "optimizer": "Adam",
  "loss_function": "MSELoss"
}
```

### 修改隐藏层大小

编辑 `configs/models/fnn_config.json`:

```json
"architecture": {
  "input_size": 6,
  "hidden_sizes": [128, 64, 32],  // ← 从 [64, 32, 16] 改为 [128, 64, 32]
  "dropout_rate": 0.2
}
```

### 启用/禁用特征筛选

编辑 `configs/models/cnn_config.json`:

```json
"feature_selection": {
  "enabled": false,            // ← 改为 false 使用全部16个特征
  "correlation_threshold": 0.5
}
```

或通过命令行:

```bash
# 禁用特征筛选 (相关性阈值设为0)
python main_hust_baseline.py --model CNN --battery 1-1 --corr-threshold 0.0
```

## 📊 输出结果

训练结果保存在 `results/results_hust/` 目录：

```
results/results_hust/
├── 1-1/
│   ├── fnn/
│   │   ├── fnn_model.pth      # 训练好的FNN模型
│   │   └── results.pkl        # 训练结果和历史
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

## 📈 训练过程

1. **加载配置** - 从JSON文件读取模型参数
2. **加载数据** - 从CSV文件加载电池数据
3. **特征筛选** - (可选) 根据相关性阈值筛选特征
4. **创建模型** - 根据配置创建神经网络
5. **训练** - 迭代训练模型，保存最佳权重
6. **评估** - 在测试集上计算MAE和RMSE
7. **保存** - 保存模型和结果

## 🎯 工作流示例

### 场景1: 快速测试FNN

```bash
# 1. 使用默认配置训练单个电池
python main_hust_baseline.py --model FNN --battery 1-1

# 2. 查看结果
# results/results_hust/1-1/fnn/results.pkl
```

### 场景2: 调参优化

```bash
# 测试不同学习率
python main_hust_baseline.py --model FNN --battery 1-1 --lr 0.001   # 默认
python main_hust_baseline.py --model FNN --battery 1-1 --lr 0.0005  # 更小
python main_hust_baseline.py --model FNN --battery 1-1 --lr 0.002   # 更大

# 测试不同的特征筛选阈值
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.3
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.5
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.7
```

### 场景3: 对比多个模型

```bash
# 在同一个电池上训练所有模型
python main_hust_baseline.py --model FNN --battery 1-1
python main_hust_baseline.py --model CNN --battery 1-1
python main_hust_baseline.py --model LSTM --battery 1-1

# 查看 results/results_hust/1-1/ 下的结果
```

### 场景4: 批量训练

```bash
# 训练第一组的所有电池，用于统计分析
python main_hust_baseline.py --model FNN --batteries 1-1 1-2 1-3 1-4 1-5 1-6 1-7 1-8

# 训练多组电池进行对比
python main_hust_baseline.py --model CNN --batteries 1-1 2-1 3-1 4-1 5-1
```

## ⚙️ 创建新模型配置

如果要添加新模型（例如 GRU），按以下步骤：

1. 在 `src/baseline_models.py` 中实现模型类
2. 在 `src/model_trainer.py` 的 `ModelFactory` 中添加模型创建逻辑
3. 创建 `configs/models/gru_config.json` 配置文件
4. 更新 `main_hust_baseline.py` 的 `--model` 选项

## 💡 最佳实践

1. **配置文件管理**
   - 为不同的实验创建配置文件变体
   - 在JSON中添加注释记录实验目的

2. **参数调整**
   - 先用默认配置快速测试
   - 使用命令行参数进行细微调整
   - 保存最佳参数组合到配置文件

3. **结果记录**
   - 定期查看 `results/` 目录下的结果
   - 使用git版本控制配置文件
   - 记录关键指标变化

4. **可重现性**
   - 始终使用 `--seed` 参数固定随机种子
   - 记录使用的命令行参数
   - 保存训练结果到版本控制

## 🔍 故障排除

### 特征数不匹配
确保启用特征筛选后，特征数正确。检查 `correlation_threshold` 值。

### 内存不足
减小 `batch_size` 或使用 `--no-cuda` 选项使用CPU训练。

### 结果不可重现
确保使用相同的 `--seed` 值，且 `shuffle` 设置一致。

---

**更新于**: 2025-11-12
