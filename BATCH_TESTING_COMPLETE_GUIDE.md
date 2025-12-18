# 批量测试完整指南

本文档汇总了批量测试一致性问题的诊断、修复和验证方法。

---

## 📋 目录

1. [问题背景](#问题背景)
2. [诊断工具](#诊断工具)
3. [修复说明](#修复说明)
4. [验证方法](#验证方法)
5. [使用示例](#使用示例)
6. [故障排除](#故障排除)

---

## 问题背景

### 问题现象

使用 `test_monotonic_weights.py` 批量测试不同单调性权重时，发现：

**批量测试结果** vs **单独训练结果** 不一致，即使使用相同的配置参数。

**示例**:
```
批量测试 (weight=0.3):  RMSE = 0.0275
单独训练 (weight=0.3):  RMSE = 0.0285  ❌ 差异 > 0.001
```

### 根本原因

`test_monotonic_weights.py` 中的 `modify_config_monotonic_weight()` 函数在修改配置文件时：

✅ **设置了**:
- `monotonic_weight` (单调性权重)
- `enabled` (是否启用物理约束)

❌ **未设置**:
- `monotonic_tolerance` (软约束容差)
- `boundary_weight` (边界约束权重)
- `smoothness_weight` (平滑性约束权重)
- `curvature_weight` (曲率约束权重)

导致配置文件中**残留之前实验的参数值**，影响批量测试结果。

### 影响范围

所有使用 `test_monotonic_weights.py` 的批量实验：
- Scenario 2 (稀疏采样) 权重测试
- Scenario 3 (随机缺失) 权重测试
- `test_adaptive_weights_scenario3.py` 自适应权重测试

---

## 诊断工具

### 1. `diagnose_batch_vs_single.py`

**功能**: 全面诊断批量训练与单独训练的差异

**运行方式**:
```bash
python diagnose_batch_vs_single.py
```

**诊断内容**:
1. ✅ 对比两种训练方式的配置
2. ✅ 检查配置文件修改机制
3. ✅ 测试随机数生成器状态
4. ✅ 检查数据加载一致性
5. ✅ 测试相同权重两次训练是否结果一致 (可选，需20分钟)

**输出示例**:
```
======================================================================
配置对比诊断
======================================================================

配置1 (批量训练):
  model_type: cnn_lstm
  seed: 999
  degradation_scenario: scenario2
  sparse_sampling_interval: 2

配置2 (单独训练):
  model_type: cnn_lstm
  seed: 999
  degradation_scenario: scenario2
  sparse_sampling_interval: 2

差异:
  ✓ 训练参数完全一致

======================================================================
检查配置文件修改机制
======================================================================

修改前:
  monotonic_weight: 0.5
  enabled: True

修改后:
  monotonic_weight: 0.3
  enabled: True

✓ 配置文件修改成功
```

### 2. `verify_batch_fix.py`

**功能**: 快速验证批量测试一致性修复是否生效

**运行方式**:
```bash
python verify_batch_fix.py
```

**测试流程**:
1. 设置 `monotonic_weight=0.3` 并显式设置所有其他参数
2. 验证配置文件是否正确修改
3. 使用相同配置运行两次训练 (固定 seed=999)
4. 对比两次结果，判断是否一致

**预期输出** (修复成功):
```
======================================================================
第3步: 对比两次训练结果
======================================================================

  第1次 RMSE: 0.027534
  第2次 RMSE: 0.027534
  差异:       0.00000000

  第1次 MAE:  0.021823
  第2次 MAE:  0.021823
  差异:       0.00000000

======================================================================
结论
======================================================================

✅ 批量测试一致性修复成功！
   两次训练结果完全一致（差异 < 1e-6）
   说明配置文件参数设置正确，无残留值影响
```

---

## 修复说明

### 修复位置

文件: `test_monotonic_weights.py`

函数: `modify_config_monotonic_weight()` (第52-82行)

### 修复内容

**修复前** (仅设置2个参数):
```python
def modify_config_monotonic_weight(model_type, monotonic_weight):
    config_path = f'configs/models/{model_type}_config.json'

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    if 'physics_constraints' in config:
        config['physics_constraints']['monotonic_weight'] = monotonic_weight  # 只设置这个
        config['physics_constraints']['enabled'] = True                       # 和这个

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
```

**修复后** (显式设置7个参数):
```python
def modify_config_monotonic_weight(model_type, monotonic_weight):
    config_path = f'configs/models/{model_type}_config.json'

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    if 'physics_constraints' in config:
        config['physics_constraints']['monotonic_weight'] = monotonic_weight
        config['physics_constraints']['enabled'] = True

        # ⭐ 显式设置其他参数，确保批量测试一致性
        config['physics_constraints']['monotonic_tolerance'] = 0.01  # 软约束容差
        config['physics_constraints']['boundary_weight'] = 0.0       # 边界约束（不使用）
        config['physics_constraints']['smoothness_weight'] = 0.0     # 平滑性约束（不使用）
        config['physics_constraints']['curvature_weight'] = 0.0      # 曲率约束（不使用）

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  [CONFIG] monotonic_weight = {monotonic_weight}")
    print(f"  [CONFIG] monotonic_tolerance = 0.01 (固定)")
    print(f"  [CONFIG] 其他约束权重已重置为 0.0")
```

### 增强验证

在 `run_single_experiment()` 函数中（第118-158行），增加了配置验证：

```python
# 验证配置是否修改成功（扩展验证所有关键参数）
physics = verify_config.get('physics_constraints', {})

print(f"  [VERIFY] 完整物理约束配置:")
print(f"    enabled: {physics.get('enabled')}")
print(f"    monotonic_tolerance: {physics.get('monotonic_tolerance')}")
print(f"    boundary_weight: {physics.get('boundary_weight')}")
print(f"    smoothness_weight: {physics.get('smoothness_weight')}")
print(f"    curvature_weight: {physics.get('curvature_weight')}")

# 检查其他权重是否为 0（确保批量测试一致性）
unexpected_params = []
if physics.get('boundary_weight', 0.0) != 0.0:
    unexpected_params.append(f"boundary_weight={physics.get('boundary_weight')}")
if physics.get('smoothness_weight', 0.0) != 0.0:
    unexpected_params.append(f"smoothness_weight={physics.get('smoothness_weight')}")
if physics.get('curvature_weight', 0.0) != 0.0:
    unexpected_params.append(f"curvature_weight={physics.get('curvature_weight')}")

if unexpected_params:
    print(f"  [WARNING] 检测到非零参数: {', '.join(unexpected_params)}")
    print(f"  [WARNING] 这可能影响批量测试一致性！")
else:
    print(f"  [OK] 所有约束参数已正确设置")
```

---

## 验证方法

### 方法 1: 快速验证（推荐）

使用 `verify_batch_fix.py`:

```bash
python verify_batch_fix.py
```

**优点**:
- 快速（约20分钟，两次训练）
- 自动对比结果
- 清晰的成功/失败判断

**判断标准**:
- ✅ 差异 < 1e-6: 完全一致，修复成功
- ⚠️ 差异 < 1e-4: 基本一致，可能有轻微数值误差
- ❌ 差异 > 1e-4: 仍有问题，需进一步诊断

### 方法 2: 完整诊断

使用 `diagnose_batch_vs_single.py`:

```bash
python diagnose_batch_vs_single.py
```

选择 `y` 进行实际训练测试（约20分钟）。

### 方法 3: 手动对比

**步骤 1**: 批量测试

```python
# test_monotonic_weights.py
CONFIG = {
    'model_type': 'cnn_lstm',
    'seed': 999,
    'degradation_scenario': 'scenario2',
    'sparse_sampling_interval': 2,
    'monotonic_weights': [0.3],  # 只测试一个权重
}
```

运行: `python test_monotonic_weights.py`

记录 RMSE (例如: 0.027534)

**步骤 2**: 单独训练

修改 `train_cross_battery.py` 主函数配置:

```python
# 数据退化场景
DEGRADATION_SCENARIO = 'scenario2'
SPARSE_SAMPLING_INTERVAL = 2

# 随机种子
SEED = 999
```

修改 `configs/models/cnn_lstm_config.json`:

```json
"physics_constraints": {
  "enabled": true,
  "monotonic_weight": 0.3,
  "monotonic_tolerance": 0.01,
  "boundary_weight": 0.0,
  "smoothness_weight": 0.0,
  "curvature_weight": 0.0
}
```

运行: `python train_cross_battery.py`

**步骤 3**: 对比

如果两次 RMSE 差异 < 1e-4，说明修复成功。

---

## 使用示例

### 示例 1: Scenario 2 批量测试（修复后）

**配置** (`test_monotonic_weights.py`):

```python
CONFIG = {
    'model_type': 'cnn_lstm',
    'seed': 517,
    'degradation_scenario': 'scenario2',
    'sparse_sampling_interval': 2,
    'monotonic_weights': [0.0, 0.1, 0.2, 0.3, 0.5],
}
```

**运行**:
```bash
python test_monotonic_weights.py
```

**输出** (每个权重):
```
======================================================================
运行实验: monotonic_weight = 0.3
场景: Scenario 2 (Sparse Sampling, interval=2, 保留率≈50.0%)
======================================================================
  [CONFIG] monotonic_weight = 0.3
  [CONFIG] monotonic_tolerance = 0.01 (固定)
  [CONFIG] 其他约束权重已重置为 0.0
  [VERIFY] 配置文件中的权重: 0.3
  [VERIFY] 完整物理约束配置:
    enabled: True
    monotonic_tolerance: 0.01
    boundary_weight: 0.0
    smoothness_weight: 0.0
    curvature_weight: 0.0
  [OK] 所有约束参数已正确设置  ✅ 修复生效标志

[结果] Test RMSE: 0.0275, MAE: 0.0220, R²: 0.9465
```

### 示例 2: Scenario 3 批量测试（修复后）

**配置**:

```python
CONFIG = {
    'model_type': 'cnn_lstm',
    'seed': 517,
    'degradation_scenario': 'scenario3',
    'random_missing_rate': 0.4,  # 40% 缺失
    'monotonic_weights': [0.0, 0.1, 0.2, 0.3, 0.5],
}
```

**运行**:
```bash
python test_monotonic_weights.py
```

**预期**: 每个权重都会显示 `[OK] 所有约束参数已正确设置`

### 示例 3: 自适应权重测试（修复后）

`test_adaptive_weights_scenario3.py` 已自动应用修复（使用相同的 `modify_config_monotonic_weight()` 函数）。

**运行**:
```bash
python test_adaptive_weights_scenario3.py
```

**说明**: 该脚本会为不同缺失率测试不同权重范围，所有实验都使用一致的配置。

---

## 故障排除

### Q1: 仍然提示 "检测到非零参数"

**可能原因**: 配置文件中还有其他参数未重置。

**解决方案**:

检查 `configs/models/cnn_lstm_config.json`:

```json
"physics_constraints": {
  "enabled": true,
  "monotonic_weight": 0.3,
  "monotonic_tolerance": 0.01,
  "boundary_weight": 0.0,     // 必须为 0.0
  "smoothness_weight": 0.0,   // 必须为 0.0
  "curvature_weight": 0.0     // 必须为 0.0
}
```

如果有非零值，手动改为 0.0，或重新运行批量测试脚本。

### Q2: 两次训练结果仍然不一致

**可能原因 1**: GPU 非确定性操作

**解决方案**: 使用 CPU 测试

```python
CONFIG = {
    'device': 'cpu',  # 改为 CPU
}
```

**可能原因 2**: 多个脚本同时修改配置文件

**解决方案**: 确保没有其他脚本在运行，或使用独立的配置文件副本。

**可能原因 3**: 数据加载顺序随机性

**解决方案**: 检查 `configs/models/cnn_lstm_config.json` 中:

```json
"data": {
  "shuffle": true  // 如果为 true，可能导致轻微差异
}
```

即使 shuffle=true，固定 seed 后结果应该一致。如果不一致，可能是随机数生成器问题。

### Q3: 想测试其他物理约束参数（如 boundary_weight）

**当前设计**: `test_monotonic_weights.py` 专门用于测试 `monotonic_weight`，其他参数固定为 0。

**如果需要测试其他参数**: 修改 `modify_config_monotonic_weight()` 函数：

```python
def modify_config_monotonic_weight(model_type, monotonic_weight, boundary_weight=0.0):
    # ...
    config['physics_constraints']['monotonic_weight'] = monotonic_weight
    config['physics_constraints']['boundary_weight'] = boundary_weight  # 允许变化
    # ...
```

或创建新的批量测试脚本 `test_boundary_weights.py`。

---

## 总结

### 问题根源

配置文件中其他物理约束参数未显式重置 → 残留之前实验的值 → 批量测试与单独训练结果不一致

### 解决方案

在 `modify_config_monotonic_weight()` 中**显式设置所有相关参数**:
- `monotonic_tolerance = 0.01`
- `boundary_weight = 0.0`
- `smoothness_weight = 0.0`
- `curvature_weight = 0.0`

### 验证方法

**快速验证**: `python verify_batch_fix.py` (约20分钟)

**完整诊断**: `python diagnose_batch_vs_single.py` (约30分钟)

### 预期结果

修复后，批量测试与单独训练在**相同配置下应产生完全一致的结果** (RMSE 差异 < 1e-4)。

---

## 相关文档

- [BATCH_TESTING_FIX.md](BATCH_TESTING_FIX.md) - 修复说明详细版
- [TEST_MONOTONIC_WEIGHTS_USAGE.md](TEST_MONOTONIC_WEIGHTS_USAGE.md) - 批量测试使用指南
- [RANDOM_MISSING_QUICK_START.md](RANDOM_MISSING_QUICK_START.md) - Scenario 3 快速指南

---

**修复时间**: 2024-12-17
**影响范围**: 所有使用 `test_monotonic_weights.py` 和 `test_adaptive_weights_scenario3.py` 的批量实验
**修复状态**: ✅ 已完成
