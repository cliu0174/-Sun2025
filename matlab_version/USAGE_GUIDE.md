# MATLAB版本使用指南

## 目录
1. [环境准备](#环境准备)
2. [快速测试](#快速测试)
3. [完整训练](#完整训练)
4. [结果分析](#结果分析)
5. [高级配置](#高级配置)
6. [常见问题](#常见问题)

---

## 环境准备

### 1. 检查MATLAB版本

```matlab
>> version
ans =
    '9.9.0.1467703 (R2020b)'  % 需要R2020b或更高
```

### 2. 检查工具箱

```matlab
>> ver
```

确认已安装：
- **Deep Learning Toolbox**
- **Statistics and Machine Learning Toolbox**

### 3. 检查GPU（可选）

```matlab
>> gpuDevice
ans =
  CUDADevice - Properties:
                      Name: 'NVIDIA GeForce RTX 3080'
                     Index: 1
```

如果没有GPU，使用CPU也可以训练（速度较慢）。

---

## 快速测试

### 运行快速测试脚本

```matlab
cd matlab_version
quick_start_example
```

这个脚本会测试：
- ✅ 数据加载
- ✅ 数据清洗
- ✅ 滑动窗口
- ✅ 模型创建
- ✅ 物理约束损失

**预期输出：**
```
======================================================================
电池SOH估计 - 快速开始示例
======================================================================

[步骤 1/4] 测试数据加载功能...
✅ 数据加载成功！
   电池名称: 1-1
   训练样本: 1200
   测试样本: 400
   ...

✅ 所有测试通过！系统运行正常。
```

---

## 完整训练

### 方法1：使用默认配置

```matlab
train_cross_battery
```

**默认配置：**
- 模型: LSTM
- 训练轮数: 100
- 批次大小: 64
- 物理约束: 启用

### 方法2：自定义配置

#### 选择不同的模型

**训练LSTM模型：**
```matlab
% 在 train_cross_battery.m 中修改
config.model_type = 'LSTM';
```

**训练GRU模型：**
```matlab
config.model_type = 'GRU';
```

**训练CNN模型：**
```matlab
config.model_type = 'CNN';
```

**训练CNN-LSTM模型：**
```matlab
config.model_type = 'CNN_LSTM';
```

#### 调整训练参数

**快速训练（用于测试）：**
```matlab
config.max_epochs = 10;            % 减少训练轮数
config.mini_batch_size = 128;      % 增大批次
```

**精细训练（用于最终结果）：**
```matlab
config.max_epochs = 200;           % 增加训练轮数
config.mini_batch_size = 32;       % 减小批次
config.initial_learn_rate = 0.0005;  % 降低学习率
```

#### 物理约束配置

**不使用物理约束：**
```matlab
config.use_physics = false;
```

**调整物理约束权重：**
```matlab
config.use_physics = true;
config.physics_weights.base_weight = 1.0;
config.physics_weights.monotonic_weight = 0.15;  % 增强单调性约束
config.physics_weights.boundary_weight = 0.1;    % 增强边界约束
```

#### 数据清洗配置

**启用3-Sigma清洗：**
```matlab
config.apply_cleaning = true;
```

**不清洗（推荐先尝试）：**
```matlab
config.apply_cleaning = false;
```

---

## 结果分析

### 1. 查看训练进度

训练过程中会自动弹出训练进度图，显示：
- 训练损失曲线
- 验证损失曲线
- 学习率变化

### 2. 加载结果文件

```matlab
% 加载结果
load('results/cross_battery/lstm/results.mat');

% 查看性能指标
fprintf('模型性能:\n');
fprintf('  RMSE: %.4f\n', results.rmse);
fprintf('  MAE:  %.4f\n', results.mae);
fprintf('  MAPE: %.2f%%\n', results.mape);
fprintf('  R²:   %.4f\n', results.r2);
```

**输出示例：**
```
模型性能:
  RMSE: 0.0234
  MAE:  0.0189
  MAPE: 2.15%
  R²:   0.9512
```

### 3. 查看物理约束损失

```matlab
if isfield(results, 'physics_loss')
    fprintf('\n物理约束损失:\n');
    fprintf('  总损失:   %.6f\n', results.physics_loss.total);
    fprintf('  基础MSE:  %.6f\n', results.physics_loss.base);
    fprintf('  单调性:   %.6f\n', results.physics_loss.monotonic);
    fprintf('  边界约束: %.6f\n', results.physics_loss.boundary);
end
```

### 4. 可视化预测结果

```matlab
% 打开保存的图片
figure;
imshow('results/cross_battery/lstm/predictions.png');
```

### 5. 自定义可视化

```matlab
% 绘制误差分布
figure;
errors = results.y_pred - results.y_true;
histogram(errors, 50);
xlabel('预测误差');
ylabel('频数');
title('误差分布直方图');

% 绘制真实值 vs 预测值
figure;
scatter(results.y_true, results.y_pred, 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
plot([0.7, 1.0], [0.7, 1.0], 'r--', 'LineWidth', 2);
xlabel('真实SOH');
ylabel('预测SOH');
title('预测对比');
grid on;
axis equal;
```

### 6. 按电池分析

```matlab
% 查看测试集电池列表
load('results/cross_battery/lstm/battery_split.mat');
test_batteries = split_info.test_batteries;

fprintf('测试集电池:\n');
for i = 1:length(test_batteries)
    fprintf('  %d. %s\n', i, test_batteries{i});
end

% 分析特定电池的预测结果
target_battery = '1-1';
mask = strcmp(results.battery_ids, target_battery);

y_true_battery = results.y_true(mask);
y_pred_battery = results.y_pred(mask);

figure;
plot(y_true_battery, 'o-', 'DisplayName', '真实值');
hold on;
plot(y_pred_battery, 's--', 'DisplayName', '预测值');
legend;
xlabel('样本索引');
ylabel('SOH');
title(sprintf('电池 %s 的预测结果', target_battery));
grid on;
```

---

## 高级配置

### 1. 修改模型超参数

#### LSTM/GRU模型

```matlab
% 更深的网络
config.num_layers = 3;
config.hidden_size = 128;
config.fc_hidden_sizes = [64, 32, 16];

% 更多正则化
config.dropout_rate = 0.3;
```

#### CNN模型

```matlab
% 更多卷积核
config.num_filters = 128;
config.kernel_size = 5;
```

#### CNN-LSTM模型

```matlab
% 更复杂的架构
config.cnn_channels = [32, 64, 128];
config.lstm_hidden_size = 128;
config.lstm_num_layers = 3;
```

### 2. 学习率调度

```matlab
% 固定学习率
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'none');

% 分段衰减
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 20);
```

### 3. 早停 (Early Stopping)

```matlab
options = trainingOptions('adam', ...
    'ValidationData', {X_val, y_val}, ...
    'ValidationPatience', 10, ...  % 如果10次验证没有改善就停止
    'OutputFcn', @(info)stopIfValidationLossNotImproving(info, 10));
```

### 4. 保存最佳模型

MATLAB会自动保存验证损失最小的模型。

### 5. 使用多GPU

```matlab
options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'multi-gpu');
```

---

## 常见问题

### Q1: 如何减少训练时间？

**方案1：使用GPU**
```matlab
% 检查GPU
gpuDevice;

% 在训练选项中指定
options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'gpu');
```

**方案2：减少数据量**
```matlab
% 只使用部分电池训练
n_batteries_to_use = 30;  % 原本77个，现在只用30个

% 在 train_cross_battery.m 的数据加载部分添加：
all_data = all_data(1:n_batteries_to_use);
battery_names = battery_names(1:n_batteries_to_use);
```

**方案3：减少训练轮数**
```matlab
config.max_epochs = 50;  % 从100减少到50
```

### Q2: 内存不足怎么办？

**方案1：减小批次大小**
```matlab
config.mini_batch_size = 32;  % 从64减少到32
```

**方案2：减小窗口大小**
```matlab
config.window_size = 5;  % 从10减少到5
```

**方案3：使用更简单的模型**
```matlab
config.model_type = 'GRU';  % 代替CNN_LSTM
```

### Q3: 模型过拟合怎么办？

**增加正则化：**
```matlab
config.dropout_rate = 0.3;  % 增加dropout
```

**减少模型复杂度：**
```matlab
config.num_layers = 1;
config.hidden_size = 32;
```

**增加训练数据：**
```matlab
config.apply_cleaning = false;  % 不删除数据
```

### Q4: 模型欠拟合怎么办？

**增加模型复杂度：**
```matlab
config.num_layers = 3;
config.hidden_size = 128;
```

**增加训练轮数：**
```matlab
config.max_epochs = 200;
```

**降低学习率：**
```matlab
config.initial_learn_rate = 0.0001;
```

### Q5: 如何对比不同模型？

```matlab
model_types = {'LSTM', 'GRU', 'CNN', 'CNN_LSTM'};
results_all = struct();

for i = 1:length(model_types)
    % 修改配置
    config.model_type = model_types{i};

    % 训练（需要修改 train_cross_battery.m 使其可作为函数调用）
    % results_all.(model_types{i}) = train_model(config);

    fprintf('完成 %s 模型训练\n', model_types{i});
end

% 对比结果
for i = 1:length(model_types)
    model = model_types{i};
    load(sprintf('results/cross_battery/%s/results.mat', lower(model)));
    fprintf('%s - RMSE: %.4f, R²: %.4f\n', model, results.rmse, results.r2);
end
```

### Q6: 如何导出模型供Python使用？

MATLAB训练的模型不能直接在Python中使用。如需跨平台使用，建议：

1. **导出权重**：
```matlab
% 提取网络参数
layers = net.Layers;
% 手动提取权重并保存为MAT文件
```

2. **在Python中重建模型并加载权重**
（需要手动对应每一层的参数）

---

## 附录：完整配置示例

### 最小配置（快速测试）

```matlab
config.model_type = 'LSTM';
config.max_epochs = 10;
config.mini_batch_size = 128;
config.use_physics = false;
config.apply_cleaning = false;
```

### 标准配置（推荐）

```matlab
config.model_type = 'LSTM';
config.max_epochs = 100;
config.mini_batch_size = 64;
config.initial_learn_rate = 0.001;
config.use_physics = true;
config.apply_cleaning = false;
```

### 高精度配置（论文实验）

```matlab
config.model_type = 'CNN_LSTM';
config.max_epochs = 200;
config.mini_batch_size = 32;
config.initial_learn_rate = 0.0005;
config.use_physics = true;
config.physics_weights.monotonic_weight = 0.15;
config.apply_cleaning = true;
```

---

**祝训练顺利！** 🚀
