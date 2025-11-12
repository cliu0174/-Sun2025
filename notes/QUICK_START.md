# 快速开始指南 - HUST数据集基准模型训练

## 🚀 5分钟快速上手

### 1. 基础训练

训练单个模型很简单：

```bash
# 训练FNN
python main_hust_baseline.py --model FNN --battery 1-1

# 训练CNN
python main_hust_baseline.py --model CNN --battery 1-1

# 训练LSTM
python main_hust_baseline.py --model LSTM --battery 1-1
```

### 2. 修改参数（无需编辑配置文件）

通过命令行修改参数，非常快速：

```bash
# 修改学习率
python main_hust_baseline.py --model FNN --battery 1-1 --lr 0.0005

# 修改批量大小
python main_hust_baseline.py --model FNN --battery 1-1 --batch-size 32

# 修改训练轮数
python main_hust_baseline.py --model FNN --battery 1-1 --epochs 5000

# 关闭特征筛选 (使用全部16个特征)
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.0

# 多个参数组合
python main_hust_baseline.py --model CNN --battery 1-1 \
  --lr 0.0005 --batch-size 64 --epochs 3000
```

### 3. 训练多个电池

```bash
# 同时训练多个电池
python main_hust_baseline.py --model FNN --batteries 1-1 1-2 1-3 1-4 1-5

# 训练整个第一组
python main_hust_baseline.py --model FNN --batteries 1-1 1-2 1-3 1-4 1-5 1-6 1-7 1-8
```

## 📁 项目结构

```
project/
├── configs/models/         # 模型配置文件
│   ├── fnn_config.json    # FNN配置
│   ├── cnn_config.json    # CNN配置
│   └── lstm_config.json   # LSTM配置
├── src/                   # 源代码
│   ├── baseline_models.py        # 模型库
│   ├── model_trainer.py          # 训练器
│   ├── feature_selector.py       # 特征筛选
│   └── data_loader_hust.py       # 数据加载
├── data/
│   └── HUST data/         # 原始数据
├── results/
│   └── results_hust/      # 训练结果
└── main_hust_baseline.py  # 主脚本
```

## ⚙️ 配置文件说明

模型配置存储在 `configs/models/` 目录中。主要参数：

### FNN 配置 (`fnn_config.json`)
```json
{
  "training": {
    "num_epochs": 2500,
    "batch_size": 64,
    "learning_rate": 0.001
  },
  "architecture": {
    "hidden_sizes": [64, 32, 16],
    "dropout_rate": 0.2
  },
  "feature_selection": {
    "enabled": true,
    "correlation_threshold": 0.5
  }
}
```

### CNN 配置 (`cnn_config.json`)
```json
{
  "architecture": {
    "num_filters": 64,
    "kernel_size": 3,
    "fc_hidden_sizes": [32, 16]
  }
}
```

### LSTM 配置 (`lstm_config.json`)
```json
{
  "architecture": {
    "hidden_size": 64,
    "num_layers": 2,
    "fc_hidden_sizes": [32, 16]
  }
}
```

## 📊 输出结果

每次训练的结果保存在：

```
results/results_hust/<电池名>/<模型名>/
├── <模型名>_model.pth    # 训练好的模型
└── results.pkl           # 训练结果和历史
```

例如：
```
results/results_hust/1-1/fnn/
├── fnn_model.pth
└── results.pkl
```

## 💡 常见用法

### 场景1: 快速测试新模型
```bash
# 用5个epoch快速测试FNN
python main_hust_baseline.py --model FNN --battery 1-1 --epochs 5
```

### 场景2: 对比三个模型
```bash
# 在同一个电池上运行三个模型
python main_hust_baseline.py --model FNN --battery 1-1
python main_hust_baseline.py --model CNN --battery 1-1
python main_hust_baseline.py --model LSTM --battery 1-1

# 查看 results/results_hust/1-1/ 下的结果对比
```

### 场景3: 调参优化
```bash
# 测试不同学习率
python main_hust_baseline.py --model FNN --battery 1-1 --lr 0.001
python main_hust_baseline.py --model FNN --battery 1-1 --lr 0.0005
python main_hust_baseline.py --model FNN --battery 1-1 --lr 0.002

# 对比结果，找到最佳学习率
```

### 场景4: 特征筛选实验
```bash
# 使用全部16个特征
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.0

# 使用|相关系数| >= 0.3的特征
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.3

# 使用|相关系数| >= 0.5的特征（默认）
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.5

# 使用|相关系数| >= 0.7的特征
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.7

# 比较不同阈值下的结果
```

### 场景5: 批量训练和统计
```bash
# 训练第一组所有电池，进行统计分析
python main_hust_baseline.py --model FNN --batteries 1-1 1-2 1-3 1-4 1-5 1-6 1-7 1-8

# 显示平均MAE和RMSE
```

## 🔧 修改配置文件

如果想永久修改某个参数，编辑相应的JSON文件：

### 修改FNN学习率

编辑 `configs/models/fnn_config.json`:

```json
"training": {
  "num_epochs": 2500,
  "batch_size": 64,
  "learning_rate": 0.0005,    // ← 从 0.001 改为 0.0005
  "optimizer": "Adam"
}
```

### 修改FNN隐藏层

编辑 `configs/models/fnn_config.json`:

```json
"architecture": {
  "input_size": 6,
  "hidden_sizes": [128, 64, 32],  // ← 改为更大的网络
  "dropout_rate": 0.2
}
```

### 禁用特征筛选

编辑 `configs/models/cnn_config.json`:

```json
"feature_selection": {
  "enabled": false,           // ← 改为 false
  "correlation_threshold": 0.5
}
```

或通过命令行（临时）：

```bash
python main_hust_baseline.py --model CNN --battery 1-1 --corr-threshold 0.0
```

## ✨ 工作流建议

### 第一步：快速测试
```bash
# 用少量epoch测试所有模型是否正常工作
python main_hust_baseline.py --model FNN --battery 1-1 --epochs 10
python main_hust_baseline.py --model CNN --battery 1-1 --epochs 10
python main_hust_baseline.py --model LSTM --battery 1-1 --epochs 10
```

### 第二步：选择最佳模型
```bash
# 在完整的2500个epoch上训练，选择最好的模型
python main_hust_baseline.py --model FNN --battery 1-1
python main_hust_baseline.py --model CNN --battery 1-1
python main_hust_baseline.py --model LSTM --battery 1-1

# 比较结果，记录MAE和RMSE
```

### 第三步：参数调优
```bash
# 对最佳模型进行参数调优
python main_hust_baseline.py --model FNN --battery 1-1 --lr 0.0005
python main_hust_baseline.py --model FNN --battery 1-1 --lr 0.0008
python main_hust_baseline.py --model FNN --battery 1-1 --lr 0.001

# 测试不同的特征筛选阈值
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.3
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.5
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.7
```

### 第四步：批量验证
```bash
# 用最佳参数在多个电池上训练，验证泛化性
python main_hust_baseline.py --model FNN --batteries 1-1 1-2 1-3 1-4 1-5
```

## 🎯 关键命令速查

| 任务 | 命令 |
|------|------|
| 训练FNN | `python main_hust_baseline.py --model FNN --battery 1-1` |
| 训练CNN | `python main_hust_baseline.py --model CNN --battery 1-1` |
| 训练LSTM | `python main_hust_baseline.py --model LSTM --battery 1-1` |
| 改学习率 | `--lr 0.0005` |
| 改批量大小 | `--batch-size 32` |
| 改训练轮数 | `--epochs 5000` |
| 改特征阈值 | `--corr-threshold 0.3` |
| 多个电池 | `--batteries 1-1 1-2 1-3` |
| 使用CPU | `--no-cuda` |
| 设置随机种子 | `--seed 42` |

## 📝 示例命令集

```bash
# 快速测试 (5个epoch)
python main_hust_baseline.py --model FNN --battery 1-1 --epochs 5 --no-cuda

# 完整训练 (2500个epoch，默认配置)
python main_hust_baseline.py --model FNN --battery 1-1

# 调参实验
python main_hust_baseline.py --model CNN --battery 1-1 --lr 0.0003 --batch-size 32

# 多电池批量训练
python main_hust_baseline.py --model LSTM --batteries 1-1 1-2 1-3 1-4 1-5 1-6 1-7 1-8

# 特征筛选对比
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.0
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.5
python main_hust_baseline.py --model FNN --battery 1-1 --corr-threshold 0.8

# 重现结果 (固定随机种子)
python main_hust_baseline.py --model FNN --battery 1-1 --seed 42
```

## 🔗 更多信息

- 详细指南见：`notes/BASELINE_TRAINING_GUIDE.md`
- 特征分析见：`notes/HUST_BPINN_ANALYSIS.md`
- 数据说明见：`notes/DATASET_COMPARISON.md`

---

**最后更新**: 2025-11-12
