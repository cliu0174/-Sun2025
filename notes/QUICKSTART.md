# 快速入门指南

## 概述

本项目实现了Sun等人(2025)论文中描述的BPINN(电池物理信息神经网络)用于SOH估计。实现包括:

1. **物理约束神经网络**: 强制执行P-IC到SOH的单调关系
2. **两阶段训练**:
   - 阶段1: 带物理约束的主训练
   - 阶段2: 二次训练(在测试数据上的在线优化)
3. **在NASA数据集上验证**: B05, B06, B07

## 安装

1. 创建虚拟环境(推荐):
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

## 数据结构

项目期望IC特征数据为CSV格式,包含以下列:
- `y_h`: 峰值IC (P-IC)
- `V_h`: 峰值电压
- `k_l`: 左侧斜率
- `k_r`: 右侧斜率
- `t_G`: 时间特征
- `Q_G`: 充电量特征
- `SOH`: 健康状态(目标变量)

您的数据文件已经在`data/`文件夹中以正确格式存储。

## 训练

### 基础训练

运行完整的训练流程:
```bash
python main.py
```

这将:
1. 从B05、B06、B07数据集加载数据
2. 训练BPINN-1(主训练,2000个epoch)
3. 执行二次训练得到BPINN-2(400次迭代)
4. 将结果和图表保存到`results/`文件夹
5. 保存训练好的模型

### 训练配置

您可以在[main.py](main.py:42-71)中修改配置:

```python
config = {
    'train_ratio': 0.6,          # 60%训练,40%测试
    'num_epochs': 2000,          # 主训练epoch数
    'learning_rate': 0.001,      # 学习率
    'lambda_physics': 0.01,      # 物理约束权重(ω_1)
    'num_secondary_iterations': 400,  # 二次训练迭代次数
    'lambda_test_physics': 0.01, # 测试物理损失权重(ω_2)
    # ... 其他参数
}
```

## 评估

评估训练好的模型:
```bash
python evaluate.py
```

这将加载保存的模型,并在训练集和测试集上提供详细的评估指标。

## 核心组件

### 1. 数据加载器 ([src/data_loader.py](src/data_loader.py))
- 从CSV文件加载IC特征
- 将数据分为训练/测试集(按论文要求60/40分割)
- 使用StandardScaler标准化特征
- 创建PyTorch DataLoaders

### 2. BPINN模型 ([src/model.py](src/model.py))
- 带物理约束的前馈神经网络
- 架构: 输入层(6) → 隐藏层(10) → 隐藏层(10) → 输出层(1)
- **BPINNLoss**: 结合数据损失(MSE) + 物理损失(∂SOH/∂P-IC > 0)
- **SecondaryTrainingLoss**: 用于测试期间的在线优化

### 3. 训练函数 ([src/train.py](src/train.py))
- `train_model()`: 主训练循环
- `secondary_training()`: 在测试数据上的在线优化
- `evaluate()`: 使用MAE和RMSE指标进行模型评估

### 4. 工具函数 ([src/utils.py](src/utils.py))
- 训练历史和预测的绘图函数
- 模型比较可视化
- 保存/加载结果的辅助函数

## 预期结果

根据论文,BPINN应该达到:
- **MAE < 0.4%** 在NASA数据集上
- **RMSE < 0.4%** 在NASA数据集上

二次训练(BPINN-2)应该比BPINN-1表现更好。

## 输出文件

训练后,您将在`results/`文件夹中找到:

- `bpinn_model_phase1.pth`: 主训练后的模型
- `bpinn_model.pth`: 二次训练后的最终模型
- `training_history_phase1.png`: 阶段1的训练曲线
- `predictions_phase1.png`: BPINN-1的预测图
- `secondary_training_history.png`: 二次训练曲线
- `predictions_phase2.png`: BPINN-2的预测图
- `model_comparison.png`: BPINN-1和BPINN-2的比较

## 理解物理约束

BPINN的关键创新是物理信息约束:

**∂SOH/∂P-IC > 0**

这强制当峰值IC(P-IC)增加时,SOH也应该增加,这反映了锂离子电池的物理退化机制。损失函数惩罚违反此约束的情况:

```
L_total = L_data + λ * L_physics
```

其中 `L_physics = mean(max(0, -∂SOH/∂P-IC)²)`

## 自定义

### 使用您自己的数据

1. 准备CSV格式的数据,包含所需列
2. 更新[main.py](main.py:45-49)中的文件路径:
```python
'data_files': [
    'data/your_battery1.csv',
    'data/your_battery2.csv',
]
```

### 更改模型架构

修改[main.py](main.py:54)中的`hidden_sizes`参数:
```python
'hidden_sizes': [20, 20],  # 两个隐藏层,每层20个神经元
```

### 调整物理约束权重

`lambda_physics`参数(论文中的ω_1)控制物理约束的强度:
```python
'lambda_physics': 0.01,  # 增加以获得更强的物理约束
```

## 故障排除

### 内存不足错误
- 减小配置中的`batch_size`
- 使用更小的模型(更少/更小的隐藏层)

### 性能不佳
- 增加`num_epochs`以进行更多训练
- 调整`lambda_physics`权重
- 检查数据质量和标准化

### 物理损失不下降
- 尝试初始时使用较小的`lambda_physics`
- 确保数据已正确标准化
- 检查数据质量问题

## 参考文献

Sun, G., Liu, Y., & Liu, X. (2025). A method for estimating lithium-ion battery state of health based on physics-informed machine learning. Journal of Power Sources, 627, 235767.

## 许可

本实现仅供研究和教育目的使用。
