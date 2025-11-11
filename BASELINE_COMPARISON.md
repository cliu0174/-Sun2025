# 基准模型对比说明

## 📚 概述

对比三个基准模型在电池SOH预测任务上的性能（Per-Battery模式）：
- **FNN** (Feedforward Neural Network) - 前馈神经网络
- **CNN** (Convolutional Neural Network) - 卷积神经网络
- **LSTM** (Long Short-Term Memory) - 长短期记忆网络

## 🚀 快速开始

### 训练并对比模型

```bash
# 训练B05电池
python main_comparison.py --battery B05

# 训练B06电池
python main_comparison.py --battery B06

# 训练B07电池
python main_comparison.py --battery B07
```

### 可视化结果

```bash
python plot_comparison_results.py --battery B05
```

## 📊 输出结果

### 控制台输出

```
======================================================================
COMPARISON SUMMARY - B05
======================================================================

Model        MAE (%)      RMSE (%)     Best Epoch/Iter
----------------------------------------------------------------------
FNN          0.9942       1.1209       2461
CNN          0.9906       1.2621       442
LSTM         1.2000       1.5000       2000
----------------------------------------------------------------------

Best Model: CNN (MAE=0.9906%)

All results saved to: results_comparison/B05/
```

### 生成的文件

```
results_comparison/
├── B05/
│   ├── comparison_results.pkl       # 所有结果数据
│   ├── comparison_bar_chart.png     # MAE/RMSE对比条形图
│   ├── comparison_table.png         # 对比表格
│   ├── fnn_model.pth               # FNN模型权重
│   ├── cnn_model.pth               # CNN模型权重
│   └── lstm_model.pth              # LSTM模型权重
├── B06/
│   └── ...
└── B07/
    └── ...
```

## 🏗️ 模型架构

### 1. FNN
```
Input(6) → FC(64) → ReLU → Dropout(0.2) → FC(32) → ReLU → Dropout(0.2) → FC(16) → ReLU → FC(1) → Sigmoid
```
- **参数量**: 3,073
- **特点**: 简单直接，训练快速

### 2. CNN
```
Input(6) → Conv1D(64, k=3) → ReLU → GlobalMaxPool → FC(32) → ReLU → FC(16) → ReLU → FC(1) → Sigmoid
```
- **参数量**: 2,881
- **特点**: 提取局部特征

### 3. LSTM
```
Input(6) → LSTM(64, 2 layers) → FC(32) → ReLU → FC(16) → ReLU → FC(1) → Sigmoid
```
- **参数量**: 53,057
- **特点**: 捕捉时序依赖

## ⚙️ 训练配置

```python
config = {
    'num_epochs': 2500,        # 训练轮数
    'learning_rate': 0.001,    # 学习率
    'batch_size': 64,          # Batch大小
    'train_ratio': 0.75,       # 75%训练，25%测试
}
```

## 📈 预期性能

| 模型 | MAE目标 | RMSE目标 | 训练时间(GPU) |
|------|---------|----------|--------------|
| FNN  | ~1.0%   | ~1.2%    | ~5分钟       |
| CNN  | ~1.0%   | ~1.3%    | ~5分钟       |
| LSTM | ~1.2%   | ~1.5%    | ~15分钟      |

## 🔍 模型对比

### FNN
- ✅ 简单高效，训练快
- ✅ 参数少，不易过拟合
- ❌ 无法捕捉特征间关系

### CNN
- ✅ 提取局部模式
- ✅ 参数量适中
- ❌ IC特征不是真正的空间数据

### LSTM
- ✅ 理论上能捕捉时序关系
- ❌ 参数量大（易过拟合）
- ❌ 训练慢
- ❌ 数据量小时优势不明显

## 💡 使用建议

### 快速实验
```bash
python main_comparison.py --battery B05
python plot_comparison_results.py --battery B05
```

### 训练所有电池
```bash
for battery in B05 B06 B07; do
    python main_comparison.py --battery $battery
    python plot_comparison_results.py --battery $battery
done
```

### 调参建议

**如果性能不佳**：
1. 增加训练轮数: `num_epochs = 3000-5000`
2. 调整学习率: `learning_rate = 0.0005` 或 `0.002`
3. 调整batch size: `batch_size = 32` 或 `128`

**如果LSTM过拟合**：
1. 增加Dropout: `dropout_rate = 0.3-0.5`
2. 减少层数: `num_layers = 1`
3. 减少隐藏单元: `hidden_size = 32`

## 📝 常见问题

### Q1: 训练需要多长时间？

**A**:
- **使用GPU**: FNN/CNN约5-10分钟，LSTM约15-20分钟
- **使用CPU**: 时间增加3-5倍

### Q2: 如何只训练一个模型？

**A**: 修改`main_comparison.py`，注释掉不需要的模型部分。

### Q3: 如何添加新的基准模型？

**A**:
1. 在`src/baseline_models.py`中添加新模型
2. 在`main_comparison.py`中添加训练代码
3. 参考现有FNN/CNN/LSTM的实现方式

### Q4: 结果文件在哪里？

**A**: 所有结果保存在`results_comparison/{battery}/`目录下。

## 🎯 下一步

1. ✅ 运行 `python main_comparison.py --battery B05`
2. ✅ 等待训练完成（约20-30分钟）
3. ✅ 运行 `python plot_comparison_results.py --battery B05`
4. ✅ 查看生成的对比图表
5. 🔧 根据需要调整参数
6. 🔁 重新训练并对比

Good luck! 🚀
