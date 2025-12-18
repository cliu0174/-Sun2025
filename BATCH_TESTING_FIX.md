# 批量测试一致性问题修复说明

## 问题描述

在使用 `test_monotonic_weights.py` 进行批量测试时，发现结果与 `train_cross_battery.py` 单独训练的结果不一致。

## 根本原因

`modify_config_monotonic_weight()` 函数在修改配置文件时，**只设置了** `monotonic_weight` 和 `enabled` 参数，**没有显式设置**其他物理约束参数：

- `monotonic_tolerance` (软约束容差)
- `boundary_weight` (边界约束权重)
- `smoothness_weight` (平滑性约束权重)
- `curvature_weight` (曲率约束权重)

这导致配置文件中可能残留之前实验的参数值，造成批量测试结果不一致。

## 修复方案

### 1. 显式设置所有物理约束参数

修改 `test_monotonic_weights.py` 中的 `modify_config_monotonic_weight()` 函数（第52-82行）：

```python
def modify_config_monotonic_weight(model_type, monotonic_weight):
    """修改配置文件中的单调性权重（并显式设置其他参数）"""
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

### 2. 增强配置验证

在每次实验前，验证所有关键参数是否正确设置（第118-158行）：

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

## 验证修复是否生效

### 方法 1: 运行诊断脚本

使用 `diagnose_batch_vs_single.py` 诊断一致性：

```bash
python diagnose_batch_vs_single.py
```

该脚本会：
1. 对比两种训练方式的配置
2. 检查配置文件修改机制
3. 测试相同权重两次训练是否结果一致

### 方法 2: 查看批量测试输出

运行 `test_monotonic_weights.py` 时，现在会输出详细验证信息：

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
  [OK] 所有约束参数已正确设置
```

### 方法 3: 手动对比单独训练

**步骤 1**: 使用批量测试运行 weight=0.3

```bash
python test_monotonic_weights.py
# 配置: 'monotonic_weights': [0.3]
```

记录结果（例如 RMSE=0.0275）

**步骤 2**: 手动修改 `configs/models/cnn_lstm_config.json`

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

**步骤 3**: 运行单独训练

```bash
python train_cross_battery.py
# 使用相同的 seed、degradation_scenario、sparse_sampling_interval
```

**步骤 4**: 对比结果

如果 RMSE 完全一致（或差异 < 1e-4），说明修复成功。

## 预期输出示例

### 修复前（不一致）

```
批量测试 (weight=0.3):  RMSE = 0.0275
单独训练 (weight=0.3):  RMSE = 0.0285  ❌ 差异较大
```

可能原因：配置文件中 `boundary_weight=0.1` 残留

### 修复后（一致）

```
批量测试 (weight=0.3):  RMSE = 0.0275
单独训练 (weight=0.3):  RMSE = 0.0275  ✅ 完全一致
```

所有参数都显式设置，无残留值。

## 其他注意事项

### 1. 确保 seed 一致

批量测试和单独训练必须使用**相同的随机种子**：

```python
# test_monotonic_weights.py
CONFIG = {
    'seed': 517,  # 固定种子
}

# train_cross_battery.py (主函数配置)
SEED = 517  # 必须相同
```

### 2. 确保场景参数一致

**Scenario 2**:
```python
# 批量测试
'degradation_scenario': 'scenario2'
'sparse_sampling_interval': 2

# 单独训练
DEGRADATION_SCENARIO = 'scenario2'
SPARSE_SAMPLING_INTERVAL = 2
```

**Scenario 3**:
```python
# 批量测试
'degradation_scenario': 'scenario3'
'random_missing_rate': 0.4

# 单独训练
DEGRADATION_SCENARIO = 'scenario3'
RANDOM_MISSING_RATE = 0.4
```

### 3. 避免并行修改配置文件

**不要同时运行多个实验脚本**，否则它们会互相覆盖配置文件。

如果需要并行实验，建议：
- 为每个实验使用不同的配置文件副本
- 或使用配置作为参数传递，而不是修改文件

## 总结

**问题**: 批量测试与单独训练结果不一致
**原因**: 配置文件中其他物理约束参数未显式重置，残留之前实验的值
**解决**: 在 `modify_config_monotonic_weight()` 中显式设置所有相关参数
**验证**: 增强配置验证，输出所有关键参数供检查

修复后，批量测试与单独训练在**相同配置下应产生完全一致的结果**（RMSE 差异 < 1e-4）。

---

**修复时间**: 2024-12-17
**相关文件**:
- `test_monotonic_weights.py` (修改)
- `diagnose_batch_vs_single.py` (诊断工具)
- `configs/models/cnn_lstm_config.json` (配置文件)
