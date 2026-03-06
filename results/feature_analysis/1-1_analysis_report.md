# HUST电池特征分析报告

**电池**: 1-1

**样本数**: 1487

**SOH范围**: 0.7526 - 1.0000

---

## 📊 皮尔逊相关系数排名 (Top-10)

| 排名 | 特征 | 相关系数 | 绝对值 | 关系 | 物理约束潜力 |
|------|------|----------|--------|------|-------------|
| 1 | **current skewness** | 0.935916 | 0.935916 | 正相关 ↗ | *** Strong |
| 2 | **current kurtosis** | 0.904063 | 0.904063 | 正相关 ↗ | *** Strong |
| 3 | **current mean** | -0.715091 | 0.715091 | 负相关 ↘ | ** Medium |
| 4 | **voltage mean** | -0.683081 | 0.683081 | 负相关 ↘ | ** Medium |
| 5 | **voltage skewness** | 0.680519 | 0.680519 | 正相关 ↗ | ** Medium |
| 6 | **voltage slope** | 0.668394 | 0.668394 | 正相关 ↗ | ** Medium |
| 7 | **voltage kurtosis** | 0.662555 | 0.662555 | 正相关 ↗ | ** Medium |
| 8 | **CC Q** | 0.659200 | 0.659200 | 正相关 ↗ | ** Medium |
| 9 | **CC charge time** | 0.659175 | 0.659175 | 正相关 ↗ | ** Medium |
| 10 | **current slope** | -0.647323 | 0.647323 | 负相关 ↘ | ** Medium |

---

## 🎯 BPINN物理约束建议

基于相关性分析，以下特征适合作为物理约束:

1. **current skewness** (相关性: 0.9359)
   - 约束: `∂SOH/∂current skewness > 0` (单调性)

2. **current kurtosis** (相关性: 0.9041)
   - 约束: `∂SOH/∂current kurtosis > 0` (单调性)

3. **current mean** (相关性: -0.7151)
   - 约束: `∂SOH/∂current mean < 0` (单调性)


---

## 📈 可视化图表

1. [相关系数对比图](1-1_correlation_comparison.png)
2. [Top-6散点图](1-1_top6_scatter.png)
3. [相关性矩阵](1-1_correlation_matrix.png)
4. [时间序列图](1-1_time_series.png)
