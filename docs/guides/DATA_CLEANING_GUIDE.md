# HUST 数据清洗方案

## 概述

基于 PINN4SOH 论文的数据预处理方法，使用 **3-Sigma 规则**清洗 HUST 数据集的异常值。

## 3-Sigma 规则原理

### 统计学基础

对于正态分布的数据：
- 约 68% 的数据在 μ ± σ 范围内
- 约 95% 的数据在 μ ± 2σ 范围内
- 约 99.7% 的数据在 μ ± 3σ 范围内

**3-Sigma 规则**：删除超过 `mean ± 3*std` 的数据点，认为它们是异常值。

### 清洗步骤

```python
for each column in dataframe:
    mean = column.mean()
    std = column.std()
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    # 删除超出界限的行
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
```

## 使用方法

### 方法 1: 分析数据清洗效果（推荐先做这个）

```bash
python data_cleaning.py
```

**输出**：
- 每个电池的清洗统计
- 总体清洗率
- `data_cleaning_report.txt` 详细报告

**示例输出**：
```
======================================================================
3-Sigma 数据清洗结果
======================================================================
原始样本数: 1523
删除 NaN: 0 (0.00%)
删除异常值: 45 (2.95%)
最终样本数: 1478
总删除率: 2.95%

按列统计删除数量:
  voltage_mean: 12
  current_std: 18
  temperature: 15
======================================================================
```

### 方法 2: 对比有无清洗的效果

修改 `data_cleaning.py` 第 283 行，取消注释：

```python
if __name__ == "__main__":
    # 方案2: 对比有无清洗的效果
    print("\n方案2: 对比有无清洗的效果")
    data_original, data_cleaned, stats = compare_with_without_cleaning('data/HUST data')
```

### 方法 3: 集成到训练流程中

需要修改 `data_loaders/data_loader_hust.py`，在数据加载时应用清洗：

```python
from data_cleaning import clean_3_sigma

def load_hust_dataset(data_dir, apply_cleaning=True):
    # ... 读取 CSV ...

    if apply_cleaning:
        df, stats = clean_3_sigma(df, verbose=True)

    # ... 后续处理 ...
```

## PINN4SOH 的数据处理流程对比

### PINN4SOH 的完整流程

```python
# 1. 读取 CSV
df = pd.read_csv(file_path)

# 2. 异常值清洗（3-Sigma）
df = delete_3_sigma(df)

# 3. 容量归一化
df['capacity'] = df['capacity'] / 1.1  # 额定容量

# 4. 特征归一化（Min-Max 到 [-1, 1]）
features = 2 * (features - features.min()) / (features.max() - features.min()) - 1
```

### 当前项目的流程

```python
# 1. 读取 CSV
df = pd.read_csv(file_path)

# 2. 容量归一化（除以初始容量）
capacity = capacity / capacity[0]

# 3. 特征归一化（StandardScaler）
scaler = StandardScaler()
features = scaler.fit_transform(features)
```

### 关键差异

| 步骤 | PINN4SOH | 当前项目 | 建议 |
|------|---------|---------|------|
| **异常值清洗** | ✅ 3-Sigma | ❌ 无 | **建议添加** |
| **特征归一化** | Min-Max [-1,1] | StandardScaler | 各有优势 |
| **容量归一化** | ÷额定容量(1.1) | ÷初始容量 | 当前更准确 |

## 预期效果

根据 PINN4SOH 论文和数据质量分析：

### 可能的改进

1. **鲁棒性提升**
   - 删除异常测量值（传感器故障、数据采集错误）
   - 减少噪声对模型的干扰

2. **训练稳定性**
   - 减少梯度爆炸/消失的风险
   - 更平滑的损失曲线

3. **泛化能力**
   - 避免模型过拟合异常点
   - 在测试集上表现更稳定

### 预期删除率

根据 PINN4SOH 的经验：
- **正常删除率**: 2-5%
- **如果超过 10%**: 可能数据质量有问题，需要检查数据采集过程

### 性能提升预期

保守估计：
- MAE 改进: 0.1-0.3%
- RMSE 改进: 0.1-0.5%
- 训练稳定性: 明显提升

## 实验验证方案

### 完整对比实验

```bash
# 1. 先分析清洗效果
python data_cleaning.py

# 2. 训练原始数据模型（基线）
python train_cross_battery.py  # 不清洗

# 3. 修改数据加载器启用清洗
# 编辑 data_loaders/data_loader_hust.py 添加清洗

# 4. 训练清洗数据模型
python train_cross_battery.py  # 启用清洗

# 5. 对比结果
python compare_data_cleaning.py
```

### 快速验证（推荐）

```bash
# 1. 查看清洗效果
python data_cleaning.py

# 2. 手动检查报告
cat data_cleaning_report.txt

# 3. 如果删除率合理（2-5%），再集成到训练流程
```

## 注意事项

### 1. 不要过度清洗

- **删除率 < 5%**: 正常
- **删除率 5-10%**: 需要注意，检查是否有系统性问题
- **删除率 > 10%**: 可能参数设置不当或数据质量问题

### 2. 按列分析

某些列删除过多可能表明：
- 该列本身测量不稳定
- 该列对预测贡献小，可以考虑移除

### 3. Per-Battery 清洗 vs 全局清洗

当前实现是 **Per-Battery 清洗**（每个电池单独计算 mean/std）：
- ✅ 优点：考虑电池间差异
- ❌ 缺点：样本数少的电池统计不稳定

也可以考虑 **全局清洗**（所有电池一起计算）：
- ✅ 优点：统计更稳定
- ❌ 缺点：忽略电池间差异

### 4. 与物理约束的关系

数据清洗与物理约束是**互补的**：
- **数据清洗**: 移除测量异常
- **物理约束**: 引导模型学习物理规律

建议顺序：
1. 先应用数据清洗（提高数据质量）
2. 再加入物理约束（提高模型泛化）

## 参考

- **论文**: PINN4SOH - Physics-Informed Neural Networks for State of Health Estimation
- **代码**: `PINN4SOH/dataloader/dataloader.py`
- **方法**: 3-Sigma 规则（统计学经典方法）

## 文件列表

- `data_cleaning.py` - 数据清洗工具脚本
- `compare_data_cleaning.py` - 对比实验脚本
- `DATA_CLEANING_GUIDE.md` - 本使用指南
- `data_cleaning_report.txt` - 清洗报告（运行后生成）

---

**建议先运行 `python data_cleaning.py` 查看清洗效果，再决定是否集成到训练流程中。**
