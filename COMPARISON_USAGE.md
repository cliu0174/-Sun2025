# 模型对比使用说明

## 📚 概述

对比四个模型在电池SOH预测任务上的性能：
- **FNN** (Feedforward Neural Network) - 前馈神经网络
- **CNN** (Convolutional Neural Network) - 卷积神经网络
- **LSTM** (Long Short-Term Memory) - 长短期记忆网络
- **BPINN** (Battery Physics-Informed Neural Network) - 电池物理信息神经网络

所有模型均采用**per-battery模式**训练。

---

## 🚀 快速开始

### 1. 训练并对比所有模型

**训练B05电池**（默认）：
```bash
python main_comparison.py
```

**训练B06电池**：
```bash
python main_comparison.py --battery B06
```

**训练B07电池**：
```bash
python main_comparison.py --battery B07
```

### 2. 可视化对比结果

```bash
python plot_comparison_results.py --battery B05
```

---

## 📊 输出结果

### 控制台输出示例

```
======================================================================
COMPARISON SUMMARY - B05
======================================================================

Model        MAE (%)      RMSE (%)     Best Epoch/Iter
----------------------------------------------------------------------
FNN          0.8234       1.0123       1245
CNN          0.7654       0.9876       1567
LSTM         0.8901       1.1234       1890
BPINN-1      0.5100       0.6700       306
BPINN-2      0.4850       0.6400       25
----------------------------------------------------------------------

Best Model: BPINN-2 (MAE=0.4850%)
```

### 生成的文件

```
results_comparison/
├── B05/
│   ├── comparison_results.pkl          # 所有模型结果
│   ├── comparison_bar_chart.png        # MAE/RMSE对比条形图
│   ├── comparison_table.png            # 对比表格
│   ├── fnn_model.pth                   # FNN模型
│   ├── cnn_model.pth                   # CNN模型
│   ├── lstm_model.pth                  # LSTM模型
│   └── bpinn_model.pth                 # BPINN模型
├── B06/
│   └── ...
└── B07/
    └── ...
```

---

## 🏗️ 模型架构

### 1. FNN (前馈神经网络)

```
Input (6) → FC(64) → ReLU → Dropout → FC(32) → ReLU → Dropout → FC(16) → ReLU → FC(1) → Sigmoid
```

- **参数量**: ~3,073
- **特点**: 简单直接，全连接层堆叠

### 2. CNN (卷积神经网络)

```
Input (6) → Reshape → Conv1D(64) → ReLU → GlobalMaxPool → FC(32) → ReLU → FC(16) → ReLU → FC(1) → Sigmoid
```

- **参数量**: ~2,881
- **特点**: 提取局部特征模式

### 3. LSTM (长短期记忆网络)

```
Input (6) → Reshape → LSTM(64, 2 layers) → FC(32) → ReLU → FC(16) → ReLU → FC(1) → Sigmoid
```

- **参数量**: ~53,057
- **特点**: 捕捉时序依赖关系

### 4. BPINN (物理信息神经网络)

```
Input (6) → FC(10) → Tanh → FC(10) → Tanh → FC(10) → Tanh → FC(1) → Sigmoid
```

- **参数量**: ~301
- **特点**:
  - 嵌入物理约束（单调性：∂SOH/∂P-IC > 0）
  - 两阶段训练（Phase 1 + Phase 2）
  - 参数量最小但性能最优

---

## ⚙️ 训练配置

所有模型使用相同的训练配置（公平对比）：

```python
config = {
    'num_epochs': 2500,           # 训练轮数
    'learning_rate': 0.001,       # 学习率
    'batch_size': 64,             # Batch大小
    'train_ratio': 0.75,          # 75%训练，25%测试
}
```

### BPINN额外配置

```python
bpinn_config = {
    'lambda_physics': 0.01,                # Phase 1物理约束权重
    'num_secondary_iterations': 800,       # Phase 2迭代次数
    'secondary_learning_rate': 0.001,      # Phase 2学习率
    'lambda_test_physics': 0.01,           # Phase 2测试物理约束权重
}
```

---

## 📈 预期性能

根据论文和实验，预期性能排序（从优到劣）：

1. **BPINN-2** (二次训练后) - MAE < 0.5%
2. **BPINN-1** (初次训练) - MAE < 0.8%
3. **CNN** - MAE ~ 0.8-1.2%
4. **FNN** - MAE ~ 1.0-1.5%
5. **LSTM** - MAE ~ 1.2-2.0%

**注意**：实际结果可能因数据集、随机种子等因素有所差异。

---

## 🔍 模型对比分析

### FNN
- ✅ **优点**: 简单高效，训练快速
- ❌ **缺点**: 无法捕捉特征间的空间/时序关系

### CNN
- ✅ **优点**: 提取局部特征模式，参数少
- ❌ **缺点**: IC特征不是真正的图像数据，卷积优势有限

### LSTM
- ✅ **优点**: 理论上能捕捉时序依赖
- ❌ **缺点**:
  - 参数量大（易过拟合）
  - 训练慢
  - 数据量较小时优势不明显

### BPINN
- ✅ **优点**:
  - 嵌入物理知识（单调性约束）
  - 参数量最小（仅301个参数）
  - 泛化能力强
  - 性能最优
- ❌ **缺点**: 需要领域知识设计物理约束

---

## 🎯 使用建议

### 场景1：复现论文结果（推荐）

使用BPINN进行per-battery训练：
```bash
python main_per_battery.py --battery B05
```

### 场景2：模型对比研究

训练所有模型并对比：
```bash
python main_comparison.py --battery B05
python plot_comparison_results.py --battery B05
```

### 场景3：快速基准测试

只训练基准模型（FNN/CNN/LSTM）：
- 修改`main_comparison.py`，注释掉BPINN部分
- 训练速度更快

---

## 💡 调参建议

### 如果所有模型性能都不好

1. **增加训练轮数**: `num_epochs = 3000-5000`
2. **调整学习率**:
   - 学习率太大 → 降低到 `0.0005`
   - 学习率太小 → 增加到 `0.002`
3. **检查数据质量**: 确保IC特征提取正确

### 如果LSTM过拟合

1. **增加Dropout**: `dropout_rate = 0.3-0.5`
2. **减少层数**: `num_layers = 1`
3. **减少隐藏单元**: `hidden_size = 32`

### 如果BPINN性能不佳

1. **调整物理约束权重**:
   - `lambda_physics = 0.001-0.1`
   - `lambda_test_physics = 0.001-0.1`
2. **Phase 2迭代次数**: `num_secondary_iterations = 200-1000`
3. **检查单调性约束**: 确保P-IC特征正确

---

## 📝 常见问题

### Q1: 训练需要多长时间？

**A**: 取决于硬件和配置
- **使用GPU**:
  - FNN/CNN: ~5-10分钟
  - LSTM: ~15-20分钟
  - BPINN: ~15-20分钟（两阶段）
- **使用CPU**: 时间增加3-5倍

### Q2: 为什么LSTM参数最多但性能不一定最好？

**A**:
1. 数据量较小（~100训练样本）
2. IC特征不是真正的时间序列
3. 容易过拟合

### Q3: BPINN为何参数少但性能好？

**A**:
1. 嵌入了物理先验知识（单调性）
2. 物理约束相当于正则化
3. 提高泛化能力

### Q4: 可以同时训练三个电池吗？

**A**: 可以创建脚本循环调用：
```bash
for battery in B05 B06 B07; do
    python main_comparison.py --battery $battery
done
```

---

## 📚 扩展阅读

- 原始论文: [论文链接]
- PINN综述: [Physics-Informed Neural Networks]
- 电池SOH估计: [Battery State-of-Health Estimation]

---

## 🔧 故障排除

### 错误: "CUDA out of memory"

**解决方案**:
1. 减小batch size: `batch_size = 32` 或 `16`
2. 使用CPU: 会自动回退

### 错误: "Results file not found"

**解决方案**:
先运行训练再可视化：
```bash
python main_comparison.py --battery B05
python plot_comparison_results.py --battery B05
```

### 模型性能异常（MAE > 10%）

**可能原因**:
1. 数据未正确标准化
2. 学习率过大导致发散
3. 模型初始化不当

**解决方案**:
1. 检查数据加载器
2. 降低学习率
3. 重新运行（换随机种子）

---

Good luck! 🚀
