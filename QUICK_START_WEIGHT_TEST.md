# 单调性权重测试 - 快速开始

## 🚀 三种使用方式

### 方式 1: 快速测试（推荐新手）⭐

**特点**：只测试3个关键权重，约30分钟完成

```bash
python quick_test_weights.py
```

**测试权重**：[0.0, 0.1, 0.2]

**适用场景**：
- 快速验证代码是否正常工作
- 初步了解物理约束的效果
- 时间有限的情况

---

### 方式 2: 完整测试（推荐正式实验）

**特点**：测试7个权重，全面探索，约70分钟完成

```bash
python test_monotonic_weights.py
```

**测试权重**：[0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

**适用场景**：
- 论文实验
- 寻找最优权重
- 完整的消融实验

---

### 方式 3: 自定义测试（推荐高级用户）

**编辑 `test_monotonic_weights.py` 顶部配置**：

```python
CONFIG = {
    # 修改测试权重
    'monotonic_weights': [0.0, 0.05, 0.1, 0.15, 0.2],

    # 修改稀疏程度
    'sparse_sampling_interval': 10,  # 改为10（保留10%）

    # 修改场景
    'degradation_scenario': 'scenario2',

    # ... 其他配置
}
```

然后运行：
```bash
python test_monotonic_weights.py
```

---

## 📊 输出结果

所有方式都会生成：

```
results/
├── monotonic_weight_test/（完整测试）
│   └── 20241217_143025/
│       ├── monotonic_weight_comparison.png  # 对比图 ⭐
│       ├── all_results.pkl                  # 原始数据
│       ├── results_summary.json             # 结果摘要
│       └── experiment_report.txt            # 文本报告
│
└── quick_test/（快速测试）
    └── 20241217_145612/
        └── ... (同上)
```

### 关键文件说明

1. **`monotonic_weight_comparison.png`** - 最重要！
   - 4个子图对比不同权重的效果
   - 红色星号标注最优权重
   - 直接用于论文

2. **`experiment_report.txt`** - 文本报告
   - 实验配置
   - 结果表格
   - 最优权重
   - 改善百分比

3. **`results_summary.json`** - 机器可读
   - 用于后续分析
   - 可导入Excel/MATLAB

---

## 🎯 典型使用流程

### 第一次使用（探索）

**步骤1**：快速测试
```bash
python quick_test_weights.py
```

**步骤2**：查看结果
- 打开 `results/quick_test/*/monotonic_weight_comparison.png`
- 观察哪个权重效果最好

**步骤3**：决定是否需要完整测试
- 如果效果显著 → 进行完整测试
- 如果效果不明显 → 换个场景测试

---

### 论文实验（系统）

**实验1**：无退化场景（基线）
```python
# 修改 test_monotonic_weights.py
CONFIG = {
    'degradation_scenario': 'none',
    'monotonic_weights': [0.0, 0.1, 0.2, 0.5],
}
```
```bash
python test_monotonic_weights.py
```

**实验2**：中度稀疏（interval=5）
```python
CONFIG = {
    'degradation_scenario': 'scenario2',
    'sparse_sampling_interval': 5,
    'monotonic_weights': [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
}
```
```bash
python test_monotonic_weights.py
```

**实验3**：重度稀疏（interval=10）
```python
CONFIG = {
    'degradation_scenario': 'scenario2',
    'sparse_sampling_interval': 10,
    'monotonic_weights': [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
}
```
```bash
python test_monotonic_weights.py
```

**实验4**：极度稀疏（interval=20）
```python
CONFIG = {
    'degradation_scenario': 'scenario2',
    'sparse_sampling_interval': 20,
    'monotonic_weights': [0.0, 0.1, 0.2, 0.3, 0.5, 1.0],
}
```
```bash
python test_monotonic_weights.py
```

**对比分析**：
- 收集4个实验的 `monotonic_weight_comparison.png`
- 观察不同稀疏程度下的最优权重变化
- 制作综合对比图

---

## 📈 预期结果

### 场景1: 无退化（干净数据）

**预期**：
- 最优权重接近 0（无约束）或很小（0.01-0.05）
- 物理约束改善 < 5%

**解读**：干净数据下，模型已经能很好地学习，物理约束价值有限

---

### 场景2: 中度稀疏（interval=5, 保留20%）

**预期**：
- 最优权重 ≈ 0.1-0.2
- 物理约束改善 10-20%

**解读**：稀疏采样破坏了时序连续性，物理约束帮助模型"填补空白"

---

### 场景3: 重度稀疏（interval=10, 保留10%）

**预期**：
- 最优权重 ≈ 0.2-0.3
- 物理约束改善 20-30%

**解读**：数据越稀疏，物理约束的正则化价值越大

---

### 场景4: 极度稀疏（interval=20, 保留5%）

**预期**：
- 最优权重 ≈ 0.3-0.5
- 物理约束改善 30-40%

**解读**：极端稀疏场景下，物理约束成为关键

---

## ⚠️ 注意事项

### 实验前

1. **备份配置文件**
   ```bash
   cp configs/models/cnn_lstm_config.json configs/models/cnn_lstm_config.json.bak
   ```

2. **检查GPU可用**
   ```bash
   nvidia-smi
   ```

3. **预估时间**
   - 快速测试：3个权重 × 10分钟 = 30分钟
   - 完整测试：7个权重 × 10分钟 = 70分钟

### 实验中

1. **不要中断**
   - 每个权重需要完整训练
   - 中断会导致该权重结果丢失

2. **监控进度**
   - 终端会显示进度：`进度: 3/7`
   - 每个权重完成后显示 RMSE

### 实验后

1. **恢复配置**
   ```bash
   cp configs/models/cnn_lstm_config.json.bak configs/models/cnn_lstm_config.json
   ```

2. **检查结果**
   - 确保所有权重都 `success=True`
   - 如有失败，查看 `experiment_report.txt` 中的错误信息

---

## 🔧 故障排除

### Q: 提示 CUDA out of memory

**A**: 减小 batch size
```python
# 在 configs/models/cnn_lstm_config.json 中
"training": {
    "batch_size": 32  # 改为 16 或 8
}
```

---

### Q: 某个权重训练失败

**A**:
- 脚本会继续运行后续权重
- 失败的权重在报告中标记为 `失败`
- 不影响其他权重的结果

---

### Q: 想要测试其他超参数

**A**: 参考脚本结构，创建类似脚本：
- `test_curvature_weights.py`（曲率权重）
- `test_boundary_weights.py`（边界权重）

只需修改：
```python
# 改为修改其他参数
config['physics_constraints']['curvature_weight'] = weight
```

---

## 📚 相关文档

- **详细指南**: [MONOTONIC_WEIGHT_TEST_GUIDE.md](docs/MONOTONIC_WEIGHT_TEST_GUIDE.md)
- **稀疏采样**: [SPARSE_SAMPLING_MANUAL_CONTROL.md](docs/SPARSE_SAMPLING_MANUAL_CONTROL.md)
- **双场景系统**: [DUAL_SCENARIO_GUIDE.md](docs/DUAL_SCENARIO_GUIDE.md)

---

## 💡 小贴士

### 提高效率

1. **先用快速测试探索**
   - 确定大致范围（例如：0.1附近最优）

2. **再用完整测试细化**
   - 在0.05-0.2范围内细化搜索

3. **最后验证稳定性**
   - 使用不同随机种子验证最优权重

### 批量实验

创建一个循环脚本：
```python
# batch_test.py
intervals = [5, 10, 20]

for interval in intervals:
    CONFIG['sparse_sampling_interval'] = interval
    main()  # 运行实验
```

---

**祝实验顺利！** 🎉
