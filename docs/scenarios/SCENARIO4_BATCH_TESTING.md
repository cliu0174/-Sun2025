# Scenario 4 批量参数测试使用指南

## 功能概述

批量测试脚本现已支持 **Scenario 4 (连续循环缺失)** 的多参数组合测试，可以同时测试：
- 多个丢弃率 (cycle_drop_rates)
- 多个缺失段数 (cycle_drop_num_gaps_list)
- 多个单调性权重 (monotonic_weights)

## 主要特点

✅ **自动参数组合**: 自动生成所有 (drop_rate × num_gaps × weight) 组合
✅ **批量执行**: 逐一运行所有参数组合的训练实验
✅ **详细报告**: 生成包含所有参数的对比报告
✅ **可视化分析**: 自动生成 6 个子图的综合分析图表
✅ **最优配置**: 自动识别性能最佳的参数组合

## 快速开始

### 1. 修改配置文件

编辑 `test_monotonic_weights.py` 中的 CONFIG:

```python
CONFIG = {
    # 基础配置
    'model_type': 'cnn_lstm',
    'degradation_scenario': 'scenario4',  # 使用 Scenario 4

    # Scenario 4 批量参数 - 支持多个值
    'cycle_drop_rates': [0.2, 0.3, 0.5],      # 测试 3 种丢弃率
    'cycle_drop_num_gaps_list': [1, 2, 3],    # 测试 3 种缺失段数

    # 单调性权重 - 支持多个值
    'monotonic_weights': [0.0, 0.3, 0.5],     # 测试 3 种权重

    # 其他配置
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,
    'seed': 42,
}
```

### 2. 运行批量测试

```bash
python test_monotonic_weights.py
```

### 3. 查看结果

实验完成后，结果保存在 `results/monotonic_weight_test/{timestamp}/`:
- `experiment_report.txt` - 文本报告
- `results_summary.json` - JSON 格式结果
- `scenario4_parameter_analysis.png` - 可视化分析图表

## 实验规模估算

总实验数 = len(cycle_drop_rates) × len(cycle_drop_num_gaps_list) × len(monotonic_weights)

### 示例场景

#### 小规模测试 (推荐新手)
```python
'cycle_drop_rates': [0.3],           # 1个
'cycle_drop_num_gaps_list': [2],     # 1个
'monotonic_weights': [0.0, 0.3, 0.5] # 3个
# 总实验数: 1 × 1 × 3 = 3 个实验
# 预计时间: ~10 分钟
```

#### 中等规模测试
```python
'cycle_drop_rates': [0.2, 0.3, 0.5],     # 3个
'cycle_drop_num_gaps_list': [1, 2],      # 2个
'monotonic_weights': [0.0, 0.3, 0.5]     # 3个
# 总实验数: 3 × 2 × 3 = 18 个实验
# 预计时间: ~1 小时
```

#### 完整测试 (全面评估)
```python
'cycle_drop_rates': [0.2, 0.3, 0.5],           # 3个
'cycle_drop_num_gaps_list': [1, 2, 3],         # 3个
'monotonic_weights': [0.0, 0.1, 0.3, 0.5, 0.7] # 5个
# 总实验数: 3 × 3 × 5 = 45 个实验
# 预计时间: ~2.5 小时
```

## 输出说明

### 1. 控制台输出

运行时会显示：
- 总实验数和参数组合信息
- 每个实验的进度 (X/Total)
- 每个实验的场景参数 (drop_rate, num_gaps, weight)
- 实时训练结果 (RMSE, MAE, R²)
- 实验摘要表格
- 最优配置信息

示例输出：
```
总实验数: 27 (3 drop_rates × 3 num_gaps × 3 weights)

======================================================================
进度: 1/27
======================================================================
运行实验: monotonic_weight = 0.0
场景: Scenario 4 (Consecutive Cycle Drop, 丢弃20%, 1段连续缺失, 保留率≈80.0%)
...
[结果] Test RMSE: 0.0234, MAE: 0.0182, R²: 0.9456
```

### 2. 实验摘要表格

```
丢弃率     缺失段     权重       RMSE(%)      MAE(%)       R²
----------------------------------------------------------------------
0.20       1          0.00       2.3400       1.8200       0.9456
0.20       1          0.30       2.2800       1.7900       0.9478
0.20       1          0.50       2.3100       1.8000       0.9465
...
```

### 3. 最优配置

```
最优配置:
----------------------------------------------------------------------
最低 RMSE: drop_rate=0.2, num_gaps=1, weight=0.3, RMSE=2.2800%
最高 R²:   drop_rate=0.2, num_gaps=1, weight=0.3, R²=0.9478
```

### 4. 可视化图表 (scenario4_parameter_analysis.png)

6个子图全面分析：

1. **热力图**: Drop Rate vs Num Gaps
   - 显示每个 (drop_rate, num_gaps) 组合的最优 RMSE
   - 颜色越绿表示性能越好

2. **折线图 1**: Weight vs RMSE (固定 num_gaps)
   - 对比不同 drop_rate 在相同 num_gaps 下的表现
   - 帮助理解丢弃率对性能的影响

3. **折线图 2**: Weight vs RMSE (固定 drop_rate)
   - 对比不同 num_gaps 在相同 drop_rate 下的表现
   - 帮助理解缺失段数对性能的影响

4. **3D 散点图**: Drop Rate × Num Gaps × RMSE
   - 颜色表示权重值
   - 直观展示三维参数空间

5. **柱状图**: 每个参数组合的最优权重
   - 显示不同 (drop_rate, num_gaps) 组合下的最佳权重
   - 发现权重选择的规律

6. **Top 5 表格**: 性能最优的前5个配置
   - 快速识别最佳参数组合

## 常见问题

### Q1: 如何只测试单个参数？

A: 将参数列表设为单个值即可：
```python
'cycle_drop_rates': [0.3],        # 只测试一个丢弃率
'cycle_drop_num_gaps_list': [2],  # 只测试一个缺失段数
```

### Q2: 实验中断了怎么办？

A: 每个实验的结果都会实时显示，但完整报告需要等所有实验完成。建议：
- 使用较小的参数集先测试
- 确保有足够的运行时间
- 可以分批次测试不同参数组合

### Q3: 如何对比 Scenario 4 和其他场景？

A: 分别运行不同场景的批量测试，然后对比生成的报告：
```python
# 第一次运行
'degradation_scenario': 'scenario2',
'sparse_sampling_interval': 5,

# 第二次运行
'degradation_scenario': 'scenario4',
'cycle_drop_rates': [0.3],
'cycle_drop_num_gaps_list': [2],
```

### Q4: 内存不足怎么办？

A: 减少参数组合数量，或者增加 GPU 内存清理：
- 减少并行测试的参数数量
- 使用较小的批次大小
- 在每个实验后清理 GPU 缓存

## 最佳实践

1. **循序渐进**: 先用小参数集 (3-5个实验) 验证配置正确
2. **记录结果**: 每次批量测试后备份 `results/` 目录
3. **合理规划**: 估算总时间，选择合适的测试规模
4. **分析优先**: 重点关注可视化图表中的趋势和最优配置

## 参数选择建议

### 丢弃率 (cycle_drop_rates)
- **轻度**: 0.1, 0.2 (保留 80-90% 数据)
- **中度**: 0.3, 0.4 (保留 60-70% 数据)
- **重度**: 0.5, 0.6 (保留 40-50% 数据)

### 缺失段数 (cycle_drop_num_gaps_list)
- **单段**: 1 (一次长时间故障)
- **少量**: 2-3 (多次短时间故障)
- **频繁**: 4-5 (频繁间歇故障)

### 单调性权重 (monotonic_weights)
- **基线**: 0.0 (无物理约束)
- **轻度**: 0.1, 0.3 (少量物理约束)
- **中度**: 0.5, 0.7 (适度物理约束)
- **强度**: 1.0 (完全物理约束)

## 示例工作流

```bash
# 1. 快速验证 (3个实验, ~10分钟)
# 修改 CONFIG: drop_rates=[0.3], gaps=[2], weights=[0.0,0.3,0.5]
python test_monotonic_weights.py

# 2. 查看初步结果
cat results/monotonic_weight_test/*/experiment_report.txt

# 3. 中等规模测试 (18个实验, ~1小时)
# 修改 CONFIG: drop_rates=[0.2,0.3,0.5], gaps=[1,2], weights=[0.0,0.3,0.5]
python test_monotonic_weights.py

# 4. 分析结果
# 打开 results/monotonic_weight_test/*/scenario4_parameter_analysis.png

# 5. 根据结果调整参数，进行精细测试
```

---

**祝测试顺利！** 🎉
