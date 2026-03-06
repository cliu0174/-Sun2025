# Python配置文件完整对比

## 📋 6个模型的关键参数对比

| 模型 | window_size | num_epochs | batch_size | learning_rate | scheduler | physics | dropout_rate |
|------|-------------|------------|------------|---------------|-----------|---------|--------------|
| **LSTM** | 40 | 200 | 256 | 0.001 | ✅ enabled | ✅ enabled | 0.4 |
| **GRU** | 40 | 200 | 256 | 0.001 | ❌ disabled | ❌ disabled | 0.4 |
| **CNN** | ❌ 无 | 200 | 256 | 0.0001 | ❌ disabled | ❌ 无 | 0.2 |
| **CNN-LSTM** | 40 | 200 | 256 | 0.004 | ✅ enabled | ❌ disabled | 0.4 |
| **BiLSTM** | 40 | 200 | 256 | 0.001 | ❌ disabled | ❌ disabled | 0.4 |
| **BiGRU** | 30 ⚠️ | 200 | 256 | 0.001 | ❌ disabled | ❌ disabled | 0.4 |

---

## ⚠️ 发现的问题

### 1️⃣ **MATLAB参数设置问题**

当前MATLAB `train_cross_battery.m` 的参数是针对GRU模型优化的：

```matlab
window_size = 40;          % GRU配置
max_epochs = 200;          % 通用
mini_batch_size = 256;     % 通用
initial_learn_rate = 0.001; % GRU配置
use_physics = false;       % GRU配置
```

**问题**：这些参数不适用于所有模型！

### 2️⃣ **各模型的差异**

#### LSTM
- ⚠️ **scheduler**: Python中启用 (`enabled: true`)，MATLAB中禁用 (`'none'`)
- ⚠️ **physics**: Python中启用 (`enabled: true`)，MATLAB中禁用 (`false`)
- ⚠️ **learning_rate**: 应该使用0.001 ✅

#### GRU
- ✅ **scheduler**: Python禁用，MATLAB禁用 - **一致**
- ✅ **physics**: Python禁用，MATLAB禁用 - **一致**
- ✅ **learning_rate**: 0.001 - **一致**

#### CNN
- ⚠️ **window_size**: CNN没有window_size（不是序列模型）
- ⚠️ **learning_rate**: 应该是0.0001，不是0.001
- ⚠️ **dropout_rate**: 应该是0.2，不是0.4

#### CNN-LSTM
- ⚠️ **scheduler**: Python启用，MATLAB禁用
- ⚠️ **learning_rate**: 应该是0.004，不是0.001
- ✅ **physics**: Python禁用，MATLAB禁用 - **一致**

#### BiLSTM
- ✅ **scheduler**: Python禁用，MATLAB禁用 - **一致**
- ✅ **physics**: Python禁用，MATLAB禁用 - **一致**
- ✅ **learning_rate**: 0.001 - **一致**

#### BiGRU
- ⚠️ **window_size**: 应该是30，不是40
- ✅ **scheduler**: Python禁用，MATLAB禁用 - **一致**
- ✅ **physics**: Python禁用，MATLAB禁用 - **一致**

---

## 🔧 解决方案

### 方案1：为每个模型创建独立的训练脚本 ❌

**不推荐**：维护成本高，代码重复

### 方案2：使用模型特定的配置覆盖 ✅ **推荐**

修改 `train_cross_battery.m`，根据模型类型自动调整参数：

```matlab
%% ===== 配置参数 =====
model_type = 'lstm';  % 或 'gru', 'cnn', 'cnn_lstm', 'bilstm', 'bigru'

% 从配置文件加载
model_config = ConfigLoader.load_model_config(model_type);

% 提取参数（使用配置文件的值）
if isfield(model_config.training, 'num_epochs')
    max_epochs = model_config.training.num_epochs;
else
    max_epochs = 200;  % 默认值
end

if isfield(model_config.training, 'batch_size')
    mini_batch_size = model_config.training.batch_size;
else
    mini_batch_size = 256;  % 默认值
end

if isfield(model_config.training, 'learning_rate')
    initial_learn_rate = model_config.training.learning_rate;
else
    initial_learn_rate = 0.001;  % 默认值
end

% 窗口大小
if isfield(model_config.data, 'window_size')
    window_size = model_config.data.window_size;
else
    window_size = 40;  % 默认值（CNN除外）
end

% 学习率调度器
if isfield(model_config.training, 'scheduler') && ...
   isfield(model_config.training.scheduler, 'enabled') && ...
   model_config.training.scheduler.enabled
    lr_schedule = 'piecewise';  % 或其他调度策略
else
    lr_schedule = 'none';
end

% 物理约束
if isfield(model_config, 'physics_constraints') && ...
   isfield(model_config.physics_constraints, 'enabled')
    use_physics = model_config.physics_constraints.enabled;
else
    use_physics = false;  % 默认值
end
```

### 方案3：命令行参数覆盖 ✅ **最灵活**

创建函数版本，允许用户覆盖配置：

```matlab
function train_cross_battery_function(model_type, varargin)
    % 解析可选参数
    p = inputParser;
    addParameter(p, 'window_size', [], @isnumeric);
    addParameter(p, 'max_epochs', [], @isnumeric);
    addParameter(p, 'batch_size', [], @isnumeric);
    addParameter(p, 'learning_rate', [], @isnumeric);
    addParameter(p, 'use_physics', [], @islogical);
    parse(p, varargin{:});

    % 加载配置
    config = ConfigLoader.load_model_config(model_type);

    % 使用用户指定的值覆盖配置
    if ~isempty(p.Results.window_size)
        window_size = p.Results.window_size;
    elseif isfield(config.data, 'window_size')
        window_size = config.data.window_size;
    else
        window_size = 40;
    end

    % ... 其他参数类似
end

% 使用示例
train_cross_battery_function('lstm');  % 使用配置文件的默认值
train_cross_battery_function('gru', 'window_size', 50);  % 覆盖window_size
```

---

## 📊 详细配置对比

### LSTM配置
```json
{
  "training": {
    "num_epochs": 200,
    "batch_size": 256,
    "learning_rate": 0.001,
    "scheduler": {
      "enabled": true,  // ⚠️ MATLAB当前为false
      "type": "WarmupCosineDecay",
      "warmup_epochs": 20,
      "warmup_lr": 0.002,
      "base_lr": 0.005,
      "final_lr": 0.0001
    }
  },
  "data": {
    "window_size": 40
  },
  "physics_constraints": {
    "enabled": true,  // ⚠️ MATLAB当前为false
    "monotonic_weight": 0.1,
    "boundary_weight": 0.0
  }
}
```

### GRU配置
```json
{
  "training": {
    "num_epochs": 200,
    "batch_size": 256,
    "learning_rate": 0.001,
    "scheduler": {
      "enabled": false  // ✅ MATLAB一致
    }
  },
  "data": {
    "window_size": 40
  },
  "physics_constraints": {
    "enabled": false  // ✅ MATLAB一致
  }
}
```

### CNN配置
```json
{
  "training": {
    "num_epochs": 200,
    "batch_size": 256,
    "learning_rate": 0.0001,  // ⚠️ 注意：不是0.001
    "scheduler": {
      "enabled": false
    }
  },
  "data": {
    // ❌ 无window_size（CNN不是序列模型）
  },
  "architecture": {
    "dropout_rate": 0.2  // ⚠️ 不是0.4
  }
}
```

### CNN-LSTM配置
```json
{
  "training": {
    "num_epochs": 200,
    "batch_size": 256,
    "learning_rate": 0.004,  // ⚠️ 注意：比其他模型高
    "scheduler": {
      "enabled": true,  // ⚠️ MATLAB当前为false
      "type": "WarmupCosineDecay",
      "warmup_epochs": 20,
      "warmup_lr": 0.002,
      "base_lr": 0.005,
      "final_lr": 0.0001
    }
  },
  "data": {
    "window_size": 40
  },
  "physics_constraints": {
    "enabled": false  // ✅ MATLAB一致
  }
}
```

### BiLSTM配置
```json
{
  "training": {
    "num_epochs": 200,
    "batch_size": 256,
    "learning_rate": 0.001,
    "scheduler": {
      "enabled": false  // ✅ MATLAB一致
    }
  },
  "data": {
    "window_size": 40
  },
  "physics_constraints": {
    "enabled": false  // ✅ MATLAB一致
  }
}
```

### BiGRU配置
```json
{
  "training": {
    "num_epochs": 200,
    "batch_size": 256,
    "learning_rate": 0.001,
    "scheduler": {
      "enabled": false  // ✅ MATLAB一致
    }
  },
  "data": {
    "window_size": 30  // ⚠️ 注意：不是40
  },
  "physics_constraints": {
    "enabled": false  // ✅ MATLAB一致
  }
}
```

---

## 🎯 推荐操作

### 立即修改：恢复配置文件自动加载

之前为了对齐GRU参数，我们**禁用**了配置文件的自动加载。现在需要**恢复**它，以支持所有模型：

1. **取消注释配置加载逻辑**（第63-77行）
2. **删除硬编码的参数**（第24、31-33、36行）
3. **让每个模型使用自己的配置文件**

### 修改后的参数设置逻辑

```matlab
%% ===== 配置参数 =====
model_type = 'lstm';  % 修改这里切换模型

% 可视化参数（通用）
color_by_battery = true;

% 其他参数从配置文件自动加载
% （在"加载配置"步骤后自动提取）
```

---

## ✅ 结论

**当前MATLAB版本只适合训练GRU模型**，其他模型会使用错误的参数。

**需要修改**：恢复配置文件自动加载功能，让每个模型使用自己的最优参数。

**修改文件**：`matlab_version/train_cross_battery.m`

**要我立即帮你修改吗？**

---

**版本**: 1.0.0
**更新日期**: 2025-12-02
**状态**: 发现问题，待修复
