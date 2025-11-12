# HUST电池特征分析报告

**电池**: 3-1

**样本数**: 1923

**SOH范围**: 0.7364 - 1.0040

---

## 📊 皮尔逊相关系数排名 (Top-10)

| 排名 | 特征 | 相关系数 | 绝对值 | 关系 | 物理约束潜力 |
|------|------|----------|--------|------|-------------|
| 1 | **current entropy** | 0.974940 | 0.974940 | 正相关 ↗ | *** Strong |
| 2 | **voltage entropy** | 0.963586 | 0.963586 | 正相关 ↗ | *** Strong |
| 3 | **current kurtosis** | 0.959987 | 0.959987 | 正相关 ↗ | *** Strong |
| 4 | **current skewness** | 0.959525 | 0.959525 | 正相关 ↗ | *** Strong |
| 5 | **CV charge time** | 0.957959 | 0.957959 | 正相关 ↗ | *** Strong |
| 6 | **CC Q** | 0.952486 | 0.952486 | 正相关 ↗ | *** Strong |
| 7 | **CC charge time** | 0.952460 | 0.952460 | 正相关 ↗ | *** Strong |
| 8 | **CV Q** | 0.948216 | 0.948216 | 正相关 ↗ | *** Strong |
| 9 | **voltage mean** | -0.926927 | 0.926927 | 负相关 ↘ | *** Strong |
| 10 | **current slope** | 0.914010 | 0.914010 | 正相关 ↗ | *** Strong |

---

## 🎯 BPINN物理约束建议

基于相关性分析，以下特征适合作为物理约束:

1. **current entropy** (相关性: 0.9749)
   - 约束: `∂SOH/∂current entropy > 0` (单调性)

2. **voltage entropy** (相关性: 0.9636)
   - 约束: `∂SOH/∂voltage entropy > 0` (单调性)

3. **current kurtosis** (相关性: 0.9600)
   - 约束: `∂SOH/∂current kurtosis > 0` (单调性)

4. **current skewness** (相关性: 0.9595)
   - 约束: `∂SOH/∂current skewness > 0` (单调性)

5. **CV charge time** (相关性: 0.9580)
   - 约束: `∂SOH/∂CV charge time > 0` (单调性)


---

## 📈 可视化图表

1. [相关系数对比图](3-1_correlation_comparison.png)
2. [Top-6散点图](3-1_top6_scatter.png)
3. [相关性矩阵](3-1_correlation_matrix.png)
4. [时间序列图](3-1_time_series.png)
