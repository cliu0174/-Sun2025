# 电池SOH估计 - MATLAB版本

这是Python版本的完整MATLAB移植，用于锂离子电池健康状态(SOH)估计。

## 项目概述

本项目使用深度学习模型对HUST数据集（77个LFP电池）进行跨电池SOH估计。

### 核心功能

✅ **4种深度学习模型**
- LSTM (长短期记忆网络)
- GRU (门控循环单元)
- CNN (卷积神经网络)
- CNN-LSTM (混合模型)

✅ **物理约束损失函数**
- 软单调性约束 (带容忍度)
- 边界约束 (SOH ∈ [0, 1])
- 时间衰减权重
- 平滑性约束 (可选)

✅ **数据预处理**
- 3-Sigma异常值清洗
- 滑动窗口特征提取
- Z-score标准化
- 初始容量归一化

✅ **跨电池训练**
- 训练集/验证集/测试集 = 60%/20%/20%
- 完全随机划分电池
- 支持多电池混合训练

## 系统要求

### 软件要求
- **MATLAB R2020b 或更高版本**
- **Deep Learning Toolbox**
- **Statistics and Machine Learning Toolbox**

### 硬件要求
- 建议: 8GB+ RAM
- GPU (可选，但推荐用于加速训练)

## 项目结构

```
matlab_version/
├── README.md                          # 本文件
├── train_cross_battery.m              # 主训练脚本
├── data_loaders/                      # 数据加载模块
│   ├── load_single_hust_battery.m    # 单电池数据加载
│   ├── clean_3_sigma.m               # 3-Sigma数据清洗
│   └── apply_windowing_with_metadata.m  # 滑动窗口
├── models/                            # 模型定义
│   ├── create_lstm_network.m         # LSTM模型
│   ├── create_gru_network.m          # GRU模型
│   ├── create_cnn_network.m          # CNN模型
│   └── create_cnn_lstm_network.m     # CNN-LSTM模型
├── utils/                             # 工具函数
│   └── compute_physics_loss.m        # 物理约束损失
├── configs/                           # 配置文件（可选）
├── scripts/                           # 辅助脚本
└── results/                           # 训练结果
    └── cross_battery/
        ├── lstm/
        ├── gru/
        ├── cnn/
        └── cnn_lstm/
```

## 快速开始

### 1. 准备数据

确保HUST数据集位于正确位置：
```
../data/HUST data/
├── 1-1.csv
├── 1-2.csv
├── ...
└── 10-10.csv  (共77个CSV文件)
```

**数据格式要求：**
- CSV文件，包含17列
- 前16列：电池特征（voltage_mean, voltage_std, CC_Q, etc.）
- 第17列：目标容量 (capacity)

### 2. 训练模型

打开MATLAB，进入项目目录：

```matlab
cd matlab_version
```

**方法1：使用默认配置**
```matlab
train_cross_battery  % 运行主脚本
```

**方法2：修改配置**

编辑 `train_cross_battery.m` 中的配置部分：

```matlab
% 选择模型类型
config.model_type = 'LSTM';  % 'LSTM', 'GRU', 'CNN', 'CNN_LSTM'

% 是否使用物理约束
config.use_physics = true;

% 是否应用数据清洗
config.apply_cleaning = false;

% 训练参数
config.max_epochs = 100;
config.mini_batch_size = 64;
config.initial_learn_rate = 0.001;
```

### 3. 查看结果

训练完成后，结果保存在：
```
results/cross_battery/<model_type>/
├── trained_model.mat      # 训练好的模型
├── results.mat            # 详细结果
├── predictions.png        # 预测可视化
└── battery_split.mat      # 数据集划分信息
```

**加载结果：**
```matlab
load('results/cross_battery/lstm/results.mat');

% 查看性能指标
fprintf('RMSE: %.4f\n', results.rmse);
fprintf('MAE:  %.4f\n', results.mae);
fprintf('R²:   %.4f\n', results.r2);
```

## 模型配置说明

### LSTM模型
```matlab
config.model_type = 'LSTM';
config.input_size = 16;
config.hidden_size = 64;
config.num_layers = 2;
config.fc_hidden_sizes = [32, 16];
config.dropout_rate = 0.2;
```

### GRU模型
```matlab
config.model_type = 'GRU';
config.input_size = 16;
config.hidden_size = 64;
config.num_layers = 2;
config.fc_hidden_sizes = [32, 16];
config.dropout_rate = 0.2;
```

### CNN模型
```matlab
config.model_type = 'CNN';
config.input_size = 16;
config.num_filters = 64;
config.kernel_size = 3;
config.fc_hidden_sizes = [32, 16];
config.dropout_rate = 0.2;
```

### CNN-LSTM模型
```matlab
config.model_type = 'CNN_LSTM';
config.input_size = 16;
config.cnn_channels = [32, 64];
config.kernel_size = 3;
config.pool_size = 2;
config.lstm_hidden_size = 64;
config.lstm_num_layers = 2;
config.fc_hidden_sizes = [64];
config.dropout_rate = 0.2;
```

## 物理约束配置

物理约束损失函数用于确保预测符合物理规律：

```matlab
config.use_physics = true;

config.physics_weights = struct();
config.physics_weights.base_weight = 1.0;          % 基础MSE损失权重
config.physics_weights.monotonic_weight = 0.1;     % 单调性约束权重
config.physics_weights.boundary_weight = 0.05;     % 边界约束权重
config.physics_weights.smoothness_weight = 0.0;    % 平滑性约束权重（可选）
config.physics_weights.monotonic_tolerance = 0.01; % 单调性容忍度
config.physics_weights.temporal_max_step = 20;     % 最大时间步长
config.physics_weights.temporal_decay_alpha = 0.2; % 时间衰减系数
```

**权重调整建议：**
- `monotonic_weight`: 0.05 ~ 0.2 (过大会影响拟合精度)
- `boundary_weight`: 0.01 ~ 0.1 (通常保持较小)
- `smoothness_weight`: 0.0 ~ 0.05 (可选，谨慎使用)

## 数据清洗

3-Sigma清洗规则：删除超过 `mean ± 3*std` 的异常值

```matlab
% 启用数据清洗
config.apply_cleaning = true;

% 清洗效果会在训练时打印
% 例如: "已清洗 1-1: 删除 25 个样本 (2.3%)"
```

**注意事项：**
- 清洗会删除部分数据，可能影响模型性能
- 建议先尝试不清洗 (`apply_cleaning = false`)
- 如果发现异常值严重影响训练，再启用清洗

## 使用GPU加速

MATLAB会自动检测并使用GPU（如果可用）：

```matlab
% 检查GPU可用性
if gpuDeviceCount > 0
    fprintf('检测到GPU: %s\n', gpuDevice().Name);
else
    fprintf('未检测到GPU，使用CPU训练\n');
end

% 在训练选项中指定
options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'auto', ...  % 'auto', 'gpu', 'cpu'
    ...
);
```

## 常见问题

### Q1: 内存不足错误
```
Error: Out of memory.
```

**解决方案：**
1. 减小 `mini_batch_size` (例如从64改为32)
2. 减小 `window_size` (例如从10改为5)
3. 减少训练电池数量（在代码中添加过滤）

### Q2: 训练速度慢
**解决方案：**
1. 使用GPU加速
2. 减小 `max_epochs`
3. 增大 `mini_batch_size`
4. 使用更简单的模型（CNN或GRU代替CNN-LSTM）

### Q3: 模型性能不佳
**解决方案：**
1. 增加 `max_epochs` (例如从100改为200)
2. 调整学习率 `initial_learn_rate`
3. 启用物理约束 `use_physics = true`
4. 尝试不同的模型类型

### Q4: 找不到数据文件
```
Error: Directory not found: ../data/HUST data
```

**解决方案：**
修改 `config.data_dir` 为正确的路径：
```matlab
config.data_dir = 'D:/Data/HUST data';  % 使用绝对路径
```

## 与Python版本的对比

| 功能 | Python版本 | MATLAB版本 |
|------|-----------|-----------|
| LSTM/GRU/CNN/CNN-LSTM | ✅ | ✅ |
| 物理约束损失 | ✅ | ✅ |
| 3-Sigma清洗 | ✅ | ✅ |
| 跨电池训练 | ✅ | ✅ |
| 单电池训练 | ✅ | ❌ (暂未实现) |
| Seq2Seq模型 | ✅ | ❌ (暂未实现) |
| BiLSTM/BiGRU | ✅ | ❌ (可自行扩展) |
| 训练速度 | 快 | 中等 |
| 代码复杂度 | 低 | 中等 |

## 下一步扩展

如需添加新功能，可参考以下步骤：

### 添加BiLSTM模型
1. 创建 `models/create_bilstm_network.m`
2. 在LSTM层中设置双向：
```matlab
lstmLayer(hidden_size, ...
    'OutputMode', 'last', ...
    'BiDirectional', true)  % 添加这一行
```

### 添加单电池训练
1. 创建 `train_single_battery.m`
2. 修改数据加载逻辑（只加载一个电池）
3. 参考跨电池训练脚本调整训练流程

## 参考文献

1. HUST数据集论文
2. PINN4SOH: Physics-Informed Neural Network for Battery SOH Estimation
3. MATLAB Deep Learning Toolbox Documentation

## 许可证

本项目与Python版本共享许可证。

## 联系方式

如有问题，请联系项目维护者或查阅Python版本的文档。

---

**版本信息：** MATLAB R2020b+ 兼容
**最后更新：** 2025年12月
**状态：** 功能完整，可用于生产环境
