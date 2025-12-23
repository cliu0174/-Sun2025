# 单调性权重批量测试指南

## 概述

`test_monotonic_weights.py` 是一个一键测试脚本，用于批量测试不同单调性权重下物理约束的效果。

### 功能特点

✅ 自动测试多个单调性权重值
✅ 对比有/无物理约束的效果
✅ 生成可视化对比图表
✅ 自动保存实验数据和报告
✅ 支持自定义测试场景

---

## 快速开始

### 基础用法

```bash
python test_monotonic_weights.py
```

**默认配置**：
- 模型：CNN-LSTM
- 场景：Scenario 2（稀疏采样，间隔=5）
- 测试权重：[0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

---

## 配置选项

### 修改测试参数

在 `test_monotonic_weights.py` 顶部的 `CONFIG` 字典中修改：

```python
CONFIG = {
    # 基础配置
    'model_type': 'cnn_lstm',           # 模型类型
    'train_ratio': 0.6,                 # 训练集比例
    'val_ratio': 0.2,                   # 验证集比例
    'test_ratio': 0.2,                  # 测试集比例
    'device': 'cuda',                   # 计算设备
    'seed': 42,                         # 随机种子

    # 数据退化场景
    'degradation_scenario': 'scenario2', # 'none', 'scenario1', 'scenario2'
    'sparse_sampling_interval': 5,       # 稀疏采样间隔

    # 单调性权重范围 ⭐ 重点配置
    'monotonic_weights': [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],

    # 输出目录
    'output_dir': 'results/monotonic_weight_test',
}
```

---

## 使用案例

### 案例 1: 测试干净数据（无退化场景）

```python
CONFIG = {
    'model_type': 'cnn_lstm',
    'degradation_scenario': 'none',     # 无退化
    'monotonic_weights': [0.0, 0.1, 0.2, 0.5, 1.0],
    # ... 其他参数
}
```

**预期结果**：
- 物理约束改善有限
- 最优权重可能接近 0（无约束）

---

### 案例 2: 测试稀疏采样场景（主推）⭐

```python
CONFIG = {
    'model_type': 'cnn_lstm',
    'degradation_scenario': 'scenario2',
    'sparse_sampling_interval': 10,     # 重度稀疏（10%保留）
    'monotonic_weights': [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
    # ... 其他参数
}
```

**预期结果**：
- 物理约束显著改善
- 最优权重可能在 0.1-0.5 之间

---

### 案例 3: 精细搜索最优权重

```python
CONFIG = {
    'model_type': 'cnn_lstm',
    'degradation_scenario': 'scenario2',
    'sparse_sampling_interval': 5,
    # 在 0.05-0.2 之间精细搜索
    'monotonic_weights': [0.0, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2],
    # ... 其他参数
}
```

**预期结果**：
- 找到更精确的最优权重
- 用于论文实验

---

### 案例 4: 对比不同稀疏程度

**第一次运行**（中度稀疏）：
```python
CONFIG = {
    'sparse_sampling_interval': 5,  # 保留 20%
    'monotonic_weights': [0.0, 0.1, 0.2, 0.5],
}
```

**第二次运行**（重度稀疏）：
```python
CONFIG = {
    'sparse_sampling_interval': 10,  # 保留 10%
    'monotonic_weights': [0.0, 0.1, 0.2, 0.5],
}
```

**对比结果**：观察不同稀疏程度下最优权重的变化

---

## 输出结果

### 文件结构

```
results/monotonic_weight_test/
└── 20241217_143025/                    # 时间戳文件夹
    ├── monotonic_weight_comparison.png # 对比图（4个子图）
    ├── all_results.pkl                 # 原始数据（Python pickle）
    ├── results_summary.json            # 结果摘要（JSON）
    └── experiment_report.txt           # 文本报告
```

### 对比图说明

**图1: RMSE vs. Monotonic Weight**
- 展示不同权重下的测试 RMSE
- 红色星号标注最优权重
- 红色虚线标注基线（weight=0）

**图2: MAE vs. Monotonic Weight**
- 展示不同权重下的测试 MAE
- 类似于 RMSE，用于交叉验证

**图3: R² vs. Monotonic Weight**
- 展示不同权重下的 R² 分数
- R² 越高越好

**图4: Improvement over Baseline**
- 柱状图，显示相对基线（weight=0）的改善百分比
- 绿色=改善，红色=退化

---

## 实验报告示例

```
======================================================================
单调性权重实验报告
======================================================================

实验配置:
----------------------------------------------------------------------
模型类型: cnn_lstm
数据退化场景: scenario2
稀疏采样间隔: 5
随机种子: 42
测试权重: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

实验结果:
----------------------------------------------------------------------
权重         RMSE(%)      MAE(%)       R²           状态
----------------------------------------------------------------------
0.00         3.5234       2.8123       0.9234       成功
0.01         3.4512       2.7856       0.9256       成功
0.05         3.2145       2.6234       0.9378       成功
0.10         2.9876       2.4512       0.9456       成功      ⭐ 最优
0.20         3.0123       2.4678       0.9445       成功
0.50         3.1456       2.5789       0.9389       成功
1.00         3.3567       2.7234       0.9301       成功

最优配置:
----------------------------------------------------------------------
最低 RMSE: weight=0.10, RMSE=2.9876%
最高 R²:   weight=0.10, R²=0.9456

相对基线改善: +15.22%
```

---

## 工作原理

### 执行流程

1. **修改配置**
   - 自动修改 `configs/models/cnn_lstm_config.json`
   - 设置 `physics_constraints.monotonic_weight`

2. **运行训练**
   - 调用 `train_cross_battery_model()`
   - 使用相同的数据划分（固定种子）

3. **收集结果**
   - 提取 test_rmse, test_mae, test_r2
   - 记录训练时间和状态

4. **生成报告**
   - 绘制对比图
   - 保存实验数据
   - 生成文本报告

---

## 注意事项

### ⚠️ 重要提示

1. **配置文件修改**
   - 脚本会自动修改 `configs/models/cnn_lstm_config.json`
   - 建议备份原始配置文件

2. **实验时间**
   - 每个权重需要完整训练一次模型
   - 7个权重 × 每次约10分钟 ≈ 70分钟
   - 建议在后台运行或使用较少权重测试

3. **随机种子**
   - 固定种子确保数据划分一致
   - 不同权重间的对比才有意义

4. **设备要求**
   - 建议使用 GPU 加速
   - CPU 训练会非常慢

---

## 高级用法

### 并行运行（不推荐）

由于配置文件共享，不建议并行运行多个实验。

如需并行，请：
1. 为每个实验创建独立的配置文件副本
2. 修改脚本读取不同的配置文件

### 自定义输出目录

```python
CONFIG = {
    'output_dir': 'results/my_experiment',
    'timestamp': '20241217_experiment1'  # 固定时间戳
}
```

### 扩展测试其他超参数

参考脚本结构，可以创建类似脚本测试：
- `test_curvature_weights.py`（曲率权重）
- `test_boundary_weights.py`（边界权重）
- `test_temporal_decay.py`（时间衰减参数）

---

## 实验建议

### 推荐测试流程

**阶段 1: 粗粒度搜索**
```python
'monotonic_weights': [0.0, 0.1, 0.5, 1.0]  # 4个点，快速探索
```

**阶段 2: 细粒度搜索**
```python
# 假设阶段1发现0.1最优，则在0.05-0.2范围细化
'monotonic_weights': [0.0, 0.05, 0.1, 0.15, 0.2]
```

**阶段 3: 验证稳定性**
```python
# 使用不同随机种子验证最优权重
'seed': [42, 66, 88]
```

---

## 常见问题

### Q1: 实验中途失败怎么办？

**A**: 脚本会捕获异常并标记为失败，继续运行后续实验。

失败的实验结果：
```python
{
    'monotonic_weight': 0.5,
    'success': False,
    'error': '错误信息'
}
```

---

### Q2: 如何恢复被修改的配置文件？

**A**:
1. 从 Git 恢复：`git checkout configs/models/cnn_lstm_config.json`
2. 或手动设置回默认值：`monotonic_weight: 0.1`

---

### Q3: 能否测试其他模型？

**A**: 可以，修改 `model_type`:
```python
CONFIG = {
    'model_type': 'lstm',  # 或 'fnn', 'gru', 'bilstm'
}
```

---

### Q4: 如何对比两次实验结果？

**A**: 使用时间戳文件夹区分：
```
results/monotonic_weight_test/
├── 20241217_143025/  # 实验1: scenario2, interval=5
└── 20241217_150130/  # 实验2: scenario2, interval=10
```

分别加载 `all_results.pkl` 进行对比。

---

## 论文实验建议

### 实验 1: 基线对比（无退化）

```python
CONFIG = {
    'degradation_scenario': 'none',
    'monotonic_weights': [0.0, 0.05, 0.1, 0.2, 0.5],
}
```

**目的**：证明在干净数据下，物理约束改善有限。

---

### 实验 2: 稀疏场景对比（核心）⭐

```python
# 测试3种稀疏程度
intervals = [5, 10, 20]

for interval in intervals:
    CONFIG = {
        'degradation_scenario': 'scenario2',
        'sparse_sampling_interval': interval,
        'monotonic_weights': [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
    }
    # 运行实验
```

**目的**：
- 证明稀疏程度越高，物理约束价值越大
- 找到不同稀疏程度下的最优权重

**预期发现**：
- interval=5 (20%): 最优权重 ≈ 0.1-0.15
- interval=10 (10%): 最优权重 ≈ 0.2-0.3
- interval=20 (5%): 最优权重 ≈ 0.3-0.5

---

### 实验 3: 鲁棒性验证

```python
# 使用不同随机种子
seeds = [42, 66, 88, 100, 123]

for seed in seeds:
    CONFIG = {
        'seed': seed,
        'sparse_sampling_interval': 10,
        'monotonic_weights': [0.0, 0.1, 0.2],
    }
    # 运行实验
```

**目的**：验证最优权重的稳定性。

---

## 总结

### 核心价值

✅ **自动化**：一键运行，无需手动修改配置
✅ **系统化**：统一的实验流程和结果格式
✅ **可视化**：自动生成对比图表
✅ **可复现**：固定种子，详细记录

### 适用场景

1. **超参数搜索**：找到最优单调性权重
2. **消融实验**：验证物理约束的作用
3. **鲁棒性测试**：在不同场景下测试稳定性
4. **论文实验**：生成标准化的对比结果

---

**更新日期**：2024-12-17
**版本**：v1.0
