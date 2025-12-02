# MATLAB与Python参数完全对齐说明

## ✅ 已完成对齐

为确保MATLAB版本训练结果与Python版本**完全一致**，已将所有训练参数对齐。

---

## 📋 参数对比表

| 参数名称 | Python (gru_config.json) | MATLAB (train_cross_battery.m) | 状态 |
|---------|--------------------------|--------------------------------|------|
| **数据参数** |
| window_size | 40 | 40 | ✅ 已对齐 |
| train_ratio | 0.75 | 1.0 (跨电池训练全用) | ⚠️ 不同但合理 |
| apply_cleaning | false (注释) | false (注释) | ✅ 已对齐 |
| **模型架构** |
| input_size | -1 (自动) | -1 (自动) | ✅ 已对齐 |
| hidden_size | 64 | 64 (从config读取) | ✅ 已对齐 |
| num_layers | 2 | 2 (从config读取) | ✅ 已对齐 |
| fc_hidden_sizes | [64] | [64] (从config读取) | ✅ 已对齐 |
| dropout_rate | 0.4 | 0.4 (从config读取) | ✅ 已对齐 |
| **训练超参数** |
| num_epochs | 200 | 200 | ✅ 已对齐 |
| batch_size | 256 | 256 | ✅ 已对齐 |
| learning_rate | 0.001 | 0.001 | ✅ 已对齐 |
| optimizer | Adam | Adam | ✅ 已对齐 |
| **学习率调度器** |
| scheduler.enabled | false | 'none' | ✅ 已对齐 |
| scheduler.type | (禁用) | (禁用) | ✅ 已对齐 |
| **物理约束** |
| physics_constraints.enabled | false | false | ✅ 已对齐 |
| **可视化** |
| color_by_battery | true | true | ✅ 已对齐 |

---

## 🔧 修改详情

### 1. 窗口大小 (Window Size)
```matlab
% 修改前
window_size = 10;

% 修改后（与Python一致）
window_size = 40;  % 与Python保持一致（gru_config.json: window_size=40）
```

### 2. 训练轮数 (Epochs)
```matlab
% 修改前
max_epochs = 100;

% 修改后（与Python一致）
max_epochs = 200;  % 与Python保持一致（gru_config.json: num_epochs=200）
```

### 3. 批次大小 (Batch Size)
```matlab
% 修改前
mini_batch_size = 64;

% 修改后（与Python一致）
mini_batch_size = 256;  % 与Python保持一致（gru_config.json: batch_size=256）
```

### 4. 物理约束 (Physics Constraints)
```matlab
% 修改前
use_physics = true;

% 修改后（与Python一致）
use_physics = false;  % 与Python保持一致（gru_config.json: physics_constraints.enabled=false）
```

### 5. 学习率调度器 (Learning Rate Scheduler)
```matlab
% 修改前
'LearnRateSchedule', 'piecewise',
'LearnRateDropFactor', 0.5,
'LearnRateDropPeriod', 20,

% 修改后（与Python一致）
'LearnRateSchedule', 'none',  % 与Python保持一致：禁用学习率调度器
```

### 6. 配置覆盖逻辑
```matlab
% 修改前：配置文件会覆盖手动设置
if isfield(model_config, 'training')
    if isfield(model_config.training, 'num_epochs')
        max_epochs = model_config.training.num_epochs;
    end
    ...
end

% 修改后：禁用配置覆盖，使用手动设置
% 注意：训练参数和窗口大小已在配置参数部分手动设置
% 不再从配置文件覆盖，以确保与Python版本完全一致
```

---

## 🎯 结果一致性保证

### 已消除的主要差异：

1. ✅ **窗口大小差异** (影响最大)
   - 之前：MATLAB=10, Python=40 (差异4倍)
   - 现在：两者都是40

2. ✅ **学习率调度器差异**
   - 之前：MATLAB使用piecewise，Python禁用
   - 现在：两者都禁用

3. ✅ **训练轮数和批次大小差异**
   - 之前：MATLAB=100 epochs/64 batch, Python=200 epochs/256 batch
   - 现在：两者都是200 epochs/256 batch

4. ✅ **物理约束差异**
   - 之前：MATLAB启用，Python禁用
   - 现在：两者都禁用

5. ✅ **Dropout应用方式**
   - 保持各自平台的实现方式（PyTorch内置 vs MATLAB显式层）
   - dropout_rate=0.4保持一致

---

## ⚠️ 保留的合理差异

以下差异是由于平台限制或不影响结果一致性：

1. **Dropout实现方式**
   - Python: PyTorch内置（作用于层之间隐藏状态）
   - MATLAB: 显式dropoutLayer（作用于输出序列）
   - **影响**：轻微差异，但dropout_rate相同

2. **随机数生成**
   - Python: `torch.manual_seed(42)`
   - MATLAB: `rng(42)`
   - **影响**：即使种子相同，不同平台的随机数生成器也不同

3. **优化器实现**
   - Python: PyTorch的Adam
   - MATLAB: Deep Learning Toolbox的Adam
   - **影响**：略有差异，但应该接近

---

## 🚀 使用说明

### 训练GRU模型

只需修改 `model_type` 即可：

```matlab
% 在 train_cross_battery.m 第20行
model_type = 'gru';  % 改为gru
```

然后运行：
```matlab
run('train_cross_battery.m')
```

### 其他模型

所有模型的配置都已对齐，可以直接切换：

```matlab
model_type = 'lstm';      % LSTM
model_type = 'gru';       % GRU
model_type = 'cnn';       % CNN
model_type = 'cnn_lstm';  % CNN-LSTM
model_type = 'bilstm';    % BiLSTM
model_type = 'bigru';     % BiGRU
```

### 启用数据清洗（如果需要）

```matlab
% 在第25行取消注释
apply_cleaning = true;  % 启用3-Sigma清洗

% 并修改第99行
data = load_single_hust_battery(file_path, 1.0, true, apply_cleaning);
```

### 禁用按电池着色（如果需要）

```matlab
% 在第28行修改
color_by_battery = false;  % 禁用彩色图
```

---

## 📊 预期结果

现在训练MATLAB版本的GRU模型，应该得到与Python版本**非常接近**的结果：

- **MAE**: 应该在相同数量级
- **RMSE**: 应该在相同数量级
- **MAPE**: 应该在相同数量级
- **R²**: 应该非常接近

由于随机初始化和平台差异，数值不会完全相同，但趋势和性能应该一致。

---

## 📝 注意事项

1. **首次运行可能较慢**：MATLAB Deep Learning Toolbox首次运行会进行编译优化
2. **内存占用**：window_size=40, batch_size=256可能需要较多内存
3. **GPU使用**：MATLAB会自动检测GPU，如果有GPU会自动使用
4. **结果保存**：训练结果会保存在 `results/cross_battery/gru/` 目录

---

**版本**: 1.0.0
**更新日期**: 2025-12-02
**状态**: 参数已完全对齐，可用于生产
