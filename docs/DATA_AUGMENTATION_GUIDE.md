# 数据增强使用指南

## 📋 概述

为了验证物理约束在噪声数据上的鲁棒性,我们实现了数据增强功能,可以对训练/验证集添加噪声(模拟实测数据质量问题)。

**重要**: 测试集始终保持干净,用于公平对比实验结果。

## 🎯 实验目的

验证物理约束是否能帮助模型在噪声数据上学到更鲁棒的表示:

| 实验 | 训练数据 | 物理约束 | 测试集 | 用途 |
|------|---------|---------|--------|------|
| Exp-1 | 干净 | 无 | 干净 | 基线 |
| Exp-2 | 干净 | 有 | 干净 | 物理约束在干净数据上的提升 |
| **Exp-3** | **噪声** | **无** | **干净** | **噪声数据基线** |
| **Exp-4** | **噪声** | **有** | **干净** | **物理约束在噪声数据上的作用** |

**关键对比**: 如果 `(Exp-4的MAE - Exp-3的MAE)` 的改善幅度 > `(Exp-2的MAE - Exp-1的MAE)`, 说明物理约束在噪声环境下更有价值!

## 🔧 使用方法

### 方法1: 修改主函数开关 (推荐)

编辑 `train_cross_battery.py` 的主函数:

```python
# ===== 数据增强开关 =====
ADD_NOISE = True            # 改为 True 启用噪声
NOISE_LEVEL = 'medium'      # 选择噪声级别: 'light', 'medium', 'heavy'
```

然后正常运行:
```bash
python train_cross_battery.py
```

### 方法2: 函数调用时传参

```python
wrapper, results, data_dict = train_cross_battery_model(
    model_type='cnn_lstm',
    add_noise=True,          # 启用噪声
    noise_level='medium',    # 噪声级别
    noise_seed=42,           # 随机种子
    # ... 其他参数
)
```

## 📊 噪声级别说明

我们提供了3个预设的噪声级别:

### 1. Light (轻度噪声)
```python
NOISE_LEVEL = 'light'
```
- 特征噪声: σ = 0.01
- 容量噪声: σ = 0.005
- 丢弃比例: 5%
- **用途**: 测试轻微退化场景

### 2. Medium (中度噪声) - 推荐
```python
NOISE_LEVEL = 'medium'
```
- 特征噪声: σ = 0.05
- 容量噪声: σ = 0.02
- 丢弃比例: 15%
- **用途**: 模拟典型实测数据质量

### 3. Heavy (重度噪声)
```python
NOISE_LEVEL = 'heavy'
```
- 特征噪声: σ = 0.10
- 容量噪声: σ = 0.05
- 丢弃比例: 30%
- **用途**: 压力测试,极端场景

## 🎬 完整实验流程

### 步骤1: 干净数据 + 无物理约束 (基线)

```python
# train_cross_battery.py
ADD_NOISE = False
# configs/models/cnn_lstm_config.json
"physics_constraints": {"enabled": false}
```

运行并记录测试集MAE (例如: 2.5%)

### 步骤2: 干净数据 + 有物理约束

```python
ADD_NOISE = False
# 启用物理约束
"physics_constraints": {"enabled": true}
```

运行并记录测试集MAE (例如: 2.1%)
→ 改善: 2.5% - 2.1% = 0.4%

### 步骤3: 噪声数据 + 无物理约束

```python
ADD_NOISE = True
NOISE_LEVEL = 'medium'
# 关闭物理约束
"physics_constraints": {"enabled": false}
```

运行并记录测试集MAE (例如: 3.2%)

### 步骤4: 噪声数据 + 有物理约束

```python
ADD_NOISE = True
NOISE_LEVEL = 'medium'
# 启用物理约束
"physics_constraints": {"enabled": true}
```

运行并记录测试集MAE (例如: 2.6%)
→ 改善: 3.2% - 2.6% = 0.6%

### 结论

- 干净数据: 物理约束改善 0.4%
- 噪声数据: 物理约束改善 0.6%
- **物理约束在噪声环境下更有价值! (0.6% > 0.4%)**

## 📁 工作原理

数据增强在 `prepare_cross_battery_data()` 函数中进行:

```python
# 1. 合并数据
train_features, train_targets, train_battery_ids = merge_batteries(train_batteries)

# 2. 标准化
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)

# 3. 添加噪声 (仅训练集和验证集)
if add_noise:
    train_features_scaled, train_targets, train_battery_ids = add_degradation_noise(
        train_features_scaled, train_targets, train_battery_ids,
        feature_noise_std=0.05,  # 特征噪声
        target_noise_std=0.02,   # 容量噪声
        drop_ratio=0.15,         # 随机丢弃15%
        seed=42
    )

# 4. 测试集保持干净!
# test_features_scaled 不加噪声
```

## ⚠️ 注意事项

### 1. 测试集不应加噪声
- ✅ 训练集: 加噪声
- ✅ 验证集: 加噪声
- ❌ 测试集: 保持干净

### 2. 随机种子的重要性
```python
noise_seed=42  # 确保实验可复现
```

### 3. 容量值裁剪
噪声后的容量值会自动裁剪到 `[0, 1.2]` 范围,避免不合理的值。

### 4. 物理约束兼容性
- 随机丢弃样本可能影响时序连续性
- 孪生/三元组采样会自动跳过无效配对
- 单调性约束仍然有效(允许小幅波动)

## 📊 输出示例

启用噪声时,训练日志会显示:

```
======================================================================
数据增强: 中度噪声 (15% drop, σ_feat=0.05, σ_targ=0.02)
======================================================================

对训练集添加噪声:
======================================================================
数据退化增强 (模拟实测数据质量问题)
======================================================================
原始样本数: 50000
特征噪声: σ=0.0500, 实际噪声水平=0.0502
容量噪声: σ=0.0200, 实际噪声水平=0.0199
随机丢弃: 7500 个样本 (15.0%)
剩余样本数: 42500
======================================================================

对验证集添加噪声:
...

======================================================================
⚠️  测试集保持干净 (用于公平对比)
======================================================================
```

## 🔄 快速切换

只需修改一个变量即可切换:

```python
# 关闭噪声 (恢复之前的设定)
ADD_NOISE = False

# 启用噪声
ADD_NOISE = True
```

所有其他代码保持不变!

## 📚 相关文件

- 工具函数: [utils/data_augmentation.py](../utils/data_augmentation.py)
- 训练脚本: [train_cross_battery.py](../train_cross_battery.py)
- 模型配置: [configs/models/](../configs/models/)

## 💡 自定义噪声参数

如需自定义噪声参数,可以直接编辑 `utils/data_augmentation.py`:

```python
NOISE_PRESETS = {
    'custom': {
        'feature_noise_std': 0.03,
        'target_noise_std': 0.01,
        'drop_ratio': 0.10,
        'description': '自定义噪声'
    }
}
```

然后使用:
```python
NOISE_LEVEL = 'custom'
```
