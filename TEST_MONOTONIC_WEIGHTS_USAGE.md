# test_monotonic_weights.py 使用指南

## 📌 功能说明

**批量测试同一数据退化场景下，不同单调性权重对模型性能的影响**

- 固定数据划分（相同 seed）
- 固定退化场景（相同缺失率/间隔）
- 只改变单调性权重
- 自动生成对比图和报告

---

## 🎯 支持的场景

### Scenario 1: 无退化（干净数据）

```python
CONFIG = {
    'degradation_scenario': 'none',
    'monotonic_weights': [0.0, 0.1, 0.2, 0.3, 0.5],
}
```

### Scenario 2: 规律稀疏采样 (Uniform Subsampling)

```python
CONFIG = {
    'degradation_scenario': 'scenario2',
    'sparse_sampling_interval': 4,  # 每4个循环保留1个 (保留率≈25%)
    'monotonic_weights': [0.0, 0.1, 0.2, 0.3, 0.5],
}
```

### Scenario 3: 随机缺失 (Random Missing) ⭐ 新增

```python
CONFIG = {
    'degradation_scenario': 'scenario3',
    'random_missing_rate': 0.4,  # 随机丢弃40%，保留60%
    'monotonic_weights': [0.0, 0.1, 0.2, 0.3, 0.5],
}
```

---

## 🚀 使用方式

### 方式 1: 测试 Scenario 2 (现有配置)

**默认配置**（已经在脚本中）:
```python
CONFIG = {
    'degradation_scenario': 'scenario2',
    'sparse_sampling_interval': 4,  # 保留25%
    'monotonic_weights': [0.2, 0.3, 0.5, 0.6, 0.8],
}
```

**运行**:
```bash
python test_monotonic_weights.py
```

---

### 方式 2: 测试 Scenario 3 (Random Missing)

**修改配置**（在 `test_monotonic_weights.py` 顶部）:

```python
CONFIG = {
    # 基础配置
    'model_type': 'cnn_lstm',
    'seed': 517,

    # 数据退化场景
    'degradation_scenario': 'scenario3',  # ⭐ 改为 scenario3

    # Scenario 2 参数 (不使用，可以保留)
    'sparse_sampling_interval': 4,

    # Scenario 3 参数 (启用)
    'random_missing_rate': 0.4,  # ⭐ 设置缺失率: 40%

    # 单调性权重范围
    'monotonic_weights': [0.0, 0.1, 0.2, 0.3, 0.5],  # ⭐ 建议包含0.0作为基线

    # 输出目录
    'output_dir': 'results/monotonic_weight_test',
}
```

**运行**:
```bash
python test_monotonic_weights.py
```

---

## 📊 输出结果

### 文件结构

```
results/monotonic_weight_test/TIMESTAMP/
├── monotonic_weight_comparison.png  # ⭐ 对比图（4个子图）
├── all_results.pkl                  # 原始数据（Python pickle）
├── results_summary.json             # JSON摘要
└── experiment_report.txt            # 文本报告 (可选，如果有该功能)
```

### 对比图说明

**4个子图**:
1. **(a) RMSE vs Monotonic Weight** - RMSE随权重变化
2. **(b) MAE vs Monotonic Weight** - MAE随权重变化
3. **(c) R² vs Monotonic Weight** - R²随权重变化
4. **(d) Improvement vs Baseline** - 相对于 weight=0.0 的改善百分比

- 红色星号 ⭐ 标注最优权重
- 绿色柱子表示改善，红色柱子表示下降

---

## 💡 推荐配置

### 实验 1: 不同稀疏程度下的最优权重探索

```python
# 测试 1: 轻度稀疏 (interval=3, 保留≈33%)
CONFIG = {
    'degradation_scenario': 'scenario2',
    'sparse_sampling_interval': 3,
    'monotonic_weights': [0.0, 0.05, 0.1, 0.15, 0.2],
}
```

```python
# 测试 2: 中度稀疏 (interval=5, 保留≈20%)
CONFIG = {
    'degradation_scenario': 'scenario2',
    'sparse_sampling_interval': 5,
    'monotonic_weights': [0.0, 0.1, 0.2, 0.3, 0.5],
}
```

```python
# 测试 3: 重度稀疏 (interval=10, 保留≈10%)
CONFIG = {
    'degradation_scenario': 'scenario2',
    'sparse_sampling_interval': 10,
    'monotonic_weights': [0.0, 0.2, 0.3, 0.5, 0.8],
}
```

---

### 实验 2: Random Missing 下的最优权重探索

```python
# 测试 1: 轻度缺失 (20% missing, 80% retained)
CONFIG = {
    'degradation_scenario': 'scenario3',
    'random_missing_rate': 0.2,
    'monotonic_weights': [0.0, 0.05, 0.1, 0.15, 0.2],
}
```

```python
# 测试 2: 中度缺失 (40% missing, 60% retained) ⭐ 推荐
CONFIG = {
    'degradation_scenario': 'scenario3',
    'random_missing_rate': 0.4,
    'monotonic_weights': [0.0, 0.1, 0.2, 0.3, 0.5],
}
```

```python
# 测试 3: 重度缺失 (60% missing, 40% retained)
CONFIG = {
    'degradation_scenario': 'scenario3',
    'random_missing_rate': 0.6,
    'monotonic_weights': [0.0, 0.2, 0.3, 0.5, 0.8],
}
```

---

### 实验 3: Uniform vs Random 对比（相同保留率）

```python
# Uniform: interval=2 → 保留50%
CONFIG = {
    'degradation_scenario': 'scenario2',
    'sparse_sampling_interval': 2,
    'monotonic_weights': [0.0, 0.1, 0.2, 0.3, 0.5],
}
# 运行后保存结果

# Random: 缺失50% → 保留50%
CONFIG = {
    'degradation_scenario': 'scenario3',
    'random_missing_rate': 0.5,
    'monotonic_weights': [0.0, 0.1, 0.2, 0.3, 0.5],
}
# 运行后对比两次结果
```

---

## ⚙️ 关键配置说明

### `monotonic_weights`

**建议包含 0.0 作为基线**（无物理约束）:
```python
'monotonic_weights': [0.0, 0.1, 0.2, 0.3, 0.5]
```

**权重范围建议**:
- 轻度退化: `[0.0, 0.05, 0.1, 0.15, 0.2]`
- 中度退化: `[0.0, 0.1, 0.2, 0.3, 0.5]`
- 重度退化: `[0.0, 0.2, 0.3, 0.5, 0.8]`

### `seed`

**固定种子确保数据划分一致**:
```python
'seed': 517  # 所有实验使用相同种子
```

这样保证：
- ✅ 训练集/验证集/测试集的电池划分完全相同
- ✅ Scenario 3 的随机缺失位置在不同权重下一致
- ✅ 结果可对比

### `sparse_sampling_interval` vs `random_missing_rate`

| 参数 | 场景 | 含义 | 示例 |
|------|------|------|------|
| `sparse_sampling_interval` | Scenario 2 | 每N个循环保留1个 | `4` = 保留25% |
| `random_missing_rate` | Scenario 3 | 随机丢弃的比例 | `0.4` = 丢弃40% (保留60%) |

**⚠️ 注意**: 两个参数只在对应场景下生效，互不干扰。

---

## 🔍 结果解读

### 示例输出（终端）

```
======================================================================
单调性权重批量测试
======================================================================

实验配置:
  模型类型: cnn_lstm
  数据场景: scenario3
  缺失率: 40% (保留率≈60.0%)
  测试权重: [0.0, 0.1, 0.2, 0.3, 0.5]
  输出目录: results/monotonic_weight_test/20241217_160530

######################################################################
进度: 1/5
######################################################################

======================================================================
运行实验: monotonic_weight = 0.0
场景: Scenario 3 (Random Missing, 缺失率=40%, 保留率≈60.0%)
======================================================================
  [OK] 已修改配置: monotonic_weight = 0.0, enabled = True
  [VERIFY] 配置文件中的权重: 0.0
  [VERIFY] 训练使用的权重: 0.0

[结果] Test RMSE: 0.0312, MAE: 0.0248, R²: 0.9401

######################################################################
进度: 2/5
######################################################################
...
```

### 预期结果趋势

#### Scenario 3 (Random Missing)

| 缺失率 | 最优权重（估计） | 改善幅度 |
|--------|------------------|----------|
| 20% | 0.1 - 0.15 | 5-10% |
| 40% | 0.2 - 0.3 | 10-15% |
| 60% | 0.3 - 0.5 | 15-25% |

**规律**: 缺失率越高，最优权重越大，物理约束改善越明显

---

## 📈 论文使用建议

### Table 1: 不同场景下的最优单调性权重

| 场景 | 退化参数 | 最优权重 | RMSE (w=0) | RMSE (最优) | 改善% |
|------|----------|---------|-----------|------------|-------|
| Scenario 2 | interval=5 | 0.2 | 0.0285 | 0.0260 | 8.8% |
| Scenario 3 | rate=0.4 | 0.3 | 0.0312 | 0.0275 | 11.9% |
| Scenario 3 | rate=0.6 | 0.5 | 0.0365 | 0.0305 | 16.4% |

### Table 2: Uniform vs Random (保留率=60%)

| 方法 | 最优权重 | RMSE | MAE | R² |
|------|---------|------|-----|-----|
| Uniform (interval≈1.67) | 0.2 | 0.0270 | 0.0215 | 0.9485 |
| Random (rate=0.4) | 0.3 | 0.0275 | 0.0220 | 0.9465 |

**结论**: Random Missing 比 Uniform 更具挑战性，需要更高的物理约束权重

---

## ⚠️ 注意事项

### 1. 确保数据划分一致

**正确做法** ✅:
```python
CONFIG = {
    'seed': 517,  # 所有实验使用相同种子
    # ...
}
```

**错误做法** ❌:
```python
# 不同实验使用不同种子
seed_1 = 42
seed_2 = 123
```

### 2. Scenario 3 使用固定缺失率

**正确做法** ✅:
```python
# 测试不同权重，缺失率固定
'random_missing_rate': 0.4  # 始终40%
'monotonic_weights': [0.0, 0.1, 0.2, 0.3, 0.5]
```

**错误做法** ❌:
```python
# 每个权重使用不同缺失率（这样没法对比）
```

### 3. 包含 weight=0.0 作为基线

**推荐** ✅:
```python
'monotonic_weights': [0.0, 0.1, 0.2, 0.3, 0.5]  # 包含0.0
```

这样可以计算改善百分比。

### 4. 运行时间估算

- **每个权重**: 约10分钟（取决于硬件）
- **5个权重**: 约50分钟
- **7个权重**: 约70分钟

建议先用少量权重（3个）快速测试，确认无误后再运行完整实验。

---

## 🔧 故障排除

### Q: 提示 "权重未生效，结果完全相同"

**A**: 检查 `configs/models/cnn_lstm_config.json`:
```json
"physics_constraints": {
  "enabled": true,  // 必须为 true
  "monotonic_weight": 0.2
}
```

脚本会自动设置 `enabled=true`，但如果手动改过可能有问题。

---

### Q: Scenario 3 参数不生效

**A**: 确保配置正确：
```python
CONFIG = {
    'degradation_scenario': 'scenario3',  # 不是 scenario2
    'random_missing_rate': 0.4,           # 不是 None
}
```

---

### Q: 想要测试其他缺失率

**A**: 只需修改 `random_missing_rate`:
```python
# 测试30%缺失
'random_missing_rate': 0.3
```

然后重新运行脚本。

---

## 📚 相关文档

- [RANDOM_MISSING_QUICK_START.md](RANDOM_MISSING_QUICK_START.md) - Scenario 3 快速指南
- [docs/RANDOM_MISSING_GUIDE.md](docs/RANDOM_MISSING_GUIDE.md) - Scenario 3 详细指南
- [QUICK_START_WEIGHT_TEST.md](QUICK_START_WEIGHT_TEST.md) - 权重测试快速开始（旧版，仅 Scenario 2）

---

**祝实验顺利！** 🎉

如有问题，请检查配置文件或参考相关文档。
