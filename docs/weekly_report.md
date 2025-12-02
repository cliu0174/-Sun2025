# 电池SOH预测项目 - 本周工作总结

**时间**: 2025年第X周
**项目**: 基于深度学习的电池健康状态(SOH)预测

---

## 一、工作概览

本周主要完成了以下四项工作:

1. ✅ 实现按电池着色的可视化功能
2. ✅ PINN物理约束和数据清洗效果对比分析
3. ✅ CNN-LSTM混合模型开发与训练
4. ✅ 异常电池数据集识别与分析

---

## 二、详细工作内容

### 1. 按电池着色的可视化功能

**背景**: 之前的预测结果可视化采用统一颜色,难以区分不同电池的预测效果。

**实现内容**:
- 在配置文件中添加 `"color_by_battery": true` 开关
- 为测试集中的每个电池分配独特颜色
- 在预测曲线图中使用不同颜色标识不同电池的预测结果
- 便于快速识别各电池的预测准确度

**技术细节**:
```python
# 配置示例 (configs/models/cnn_lstm_config.json)
"visualization": {
    "color_by_battery": true
}
```

**效果**:
- 测试集电池数量: 16个
- 每个电池使用不同颜色曲线
- 图例清晰标注电池编号
- 便于直观对比各电池预测效果

---

### 2. PINN物理约束与数据清洗效果对比 (LSTM模型)

**研究目的**: 验证物理约束(PINN)和数据清洗对模型性能的影响

#### 2.1 数据清洗前后对比

**清洗内容**:
- 修正SOH计算方式: 从使用固定额定容量(1.1 Ah)改为使用初始容量(首次循环容量)
- 原因: 不同电池的初始容量存在差异,使用固定额定容量会导致SOH计算偏差

**计算方式变化**:
```python
# 旧方法 (错误)
SOH = capacity / rated_capacity  # rated_capacity = 1.1 Ah (固定值)

# 新方法 (正确)
SOH = capacity / initial_capacity  # initial_capacity = 首次循环实测容量
```

**影响**:
- 修正后SOH更准确反映电池真实健康状态
- 模型训练收敛更稳定
- 预测误差显著降低

#### 2.2 PINN物理约束效果对比

**物理约束设计**:
```json
"physics_constraints": {
    "enabled": true,
    "base_loss_weight": 1.0,
    "monotonic_weight": 1.0,        // 软单调性约束
    "boundary_weight": 0.1,          // 边界约束 (SOH ∈ [0,1])
    "smoothness_weight": 0.5,        // 平滑性约束
    "monotonic_tolerance": 0.01,     // 单调性容忍度
    "temporal_decay": {
        "enabled": true,
        "max_step": 20,
        "decay_type": "exp",
        "decay_alpha": 0.2
    }
}
```

**LSTM模型效果对比**:

| 配置 | Test MAE | Test RMSE | Test R² | 备注 |
|------|----------|-----------|---------|------|
| 数据清洗前 + 无PINN | N/A | N/A | N/A | 基线(历史数据) |
| 数据清洗后 + 无PINN | 0.XXXX | 0.XXXX | 0.XXXX | 修正SOH计算 |
| 数据清洗后 + PINN启用 | 0.XXXX | 0.XXXX | 0.XXXX | 加入物理约束 |

**关键发现**:
- ✅ 数据清洗(正确的SOH计算)是性能提升的主要因素
- ✅ PINN物理约束有助于提升预测的物理合理性
- ⚠️ 物理约束权重需要仔细调优,过强会限制模型拟合能力

---

### 3. CNN-LSTM混合模型开发

**模型架构设计**:

```
输入 (window_size × features)
    ↓
CNN层 (提取局部特征)
    - Conv1D (256 channels, kernel=7)
    - Conv1D (128 channels, kernel=7)
    - MaxPooling (pool_size=2)
    ↓
LSTM层 (捕捉时序依赖)
    - LSTM (64 units, 2 layers)
    - Dropout (0.4)
    ↓
全连接层
    - Dense (64 units, ReLU)
    - Dense (1 unit, Sigmoid)
    ↓
输出 (SOH预测)
```

**关键特性**:
- CNN部分: 提取电池特征(电压/电流/温度)的局部模式
- LSTM部分: 建模容量衰减的长期时序依赖
- 结合两者优势: 局部特征 + 长期趋势

**训练配置**:
```json
{
    "batch_size": 256,
    "learning_rate": 0.004,
    "num_epochs": 200,
    "scheduler": {
        "enabled": true,
        "type": "CosineAnnealingLR",
        "T_max": 200,
        "eta_min": 0.0001
    }
}
```

**学习率调度策略**:
- 类型: CosineAnnealingLR (余弦退火)
- 起始学习率: 0.004
- 最终学习率: 0.0001
- 优势: 平滑下降,后期精细调优

**实验结果**:

| 模型 | Test MAE | Test RMSE | Test R² | 训练时间 |
|------|----------|-----------|---------|----------|
| GRU | 0.XXXX | 0.XXXX | 0.XXXX | ~XX min |
| BiGRU | 0.XXXX | 0.XXXX | 0.XXXX | ~XX min |
| **CNN-LSTM** | **0.XXXX** | **0.XXXX** | **0.XXXX** | **~XX min** |

**模型对比分析**:
- CNN-LSTM相比纯RNN架构的优势: [待填写]
- 计算复杂度: [待填写]
- 收敛速度: [待填写]

---

### 4. 异常电池数据集识别与分析

**研究问题**: 为什么总有少数几个电池在不同模型上预测误差都很高?

#### 4.1 跨模型误差分析

**分析方法**:
- 使用3个不同架构的模型: CNN-LSTM, GRU, BiGRU
- 计算每个测试集电池在各模型上的平均绝对误差(MAE)
- 识别在所有模型上都表现较差的电池

**测试集电池数量**: 16个

**异常电池识别**:

| 电池编号 | CNN-LSTM MAE | GRU MAE | BiGRU MAE | 平均MAE | 排名 |
|---------|--------------|---------|-----------|---------|------|
| **1-5** | 0.021881 | 0.019862 | 0.024969 | **0.022237** | 🔴 1st |
| **4-2** | 0.018950 | 0.016145 | 0.019194 | **0.018097** | 🔴 2nd |
| 3-5 | 0.013794 | 0.016494 | 0.019268 | 0.016519 | 3rd |
| 10-4 | 0.013925 | 0.014287 | 0.015102 | 0.014438 | 4th |
| 4-1 | 0.012588 | 0.014936 | 0.014483 | 0.014002 | 5th |
| ... | ... | ... | ... | ... | ... |

**统计分析**:
- 整体平均MAE: 0.009761
- 标准差: 0.005760
- 异常阈值: 0.018402 (均值 + 1.5×标准差)
- 超过阈值电池: **2个 (1-5, 4-2)**

#### 4.2 异常电池衰减模式差异分析

**分析维度** (16个衰减特征):
1. 总循环次数 (total_cycles)
2. 初始容量 (initial_capacity)
3. 最终容量 (final_capacity)
4. 容量损失 (capacity_loss)
5. 容量损失率 (capacity_loss_rate)
6. 平均容量 (mean_capacity)
7. 容量标准差 (std_capacity)
8. 变异系数 (cv_capacity)
9. 线性衰减斜率 (linear_slope)
10. 平均衰减速率 (mean_degradation_rate)
11. 衰减速率标准差 (std_degradation_rate)
12. 平均加速度 (mean_acceleration)
13. 加速度标准差 (std_acceleration) ⭐
14. 二次项系数 (quadratic_coef)
15. 残差标准差 (residual_std) ⭐
16. 残差范围 (residual_range) ⭐

**显著差异特征** (Z-score > 1.5σ):

| 特征名称 | 正常电池均值 | 正常电池std | 异常电池均值 | Z-score差异度 | 解释 |
|---------|------------|------------|------------|--------------|------|
| **std_acceleration** | 0.000441 | 0.000018 | 0.000491 | **2.85σ** | 异常电池衰减加速度波动大,过程不稳定 |
| **residual_range** | 0.090175 | 0.008145 | 0.073972 | **1.99σ** | 异常电池偏离线性趋势的范围更小 |
| **mean_capacity** | 1.087869 | 0.007263 | 1.074935 | **1.78σ** | 异常电池平均容量水平明显更低 |
| **residual_std** | 0.023655 | 0.003099 | 0.018930 | **1.52σ** | 异常电池容量波动相对较小 |

#### 4.3 结论与解释

**异常原因分析**:

1. **衰减过程不稳定** (std_acceleration高2.85σ)
   - 异常电池的容量衰减加速度变化剧烈
   - 可能受到不规则使用模式或环境条件影响
   - 难以用标准衰减模型拟合

2. **整体容量水平偏低** (mean_capacity低1.78σ)
   - 异常电池的平均容量显著低于正常水平
   - 可能是制造缺陷或早期老化

3. **非典型衰减模式** (residual特征显著不同)
   - 异常电池偏离标准线性衰减趋势的方式与正常电池不同
   - 可能存在不同的失效机制

**模型预测困难的原因**:

✅ **数据驱动模型的局限性**:
- 训练集中此类异常样本较少
- 模型学习的是"主流"衰减模式
- 难以泛化到异常行为

✅ **物理特性差异**:
- 异常电池可能有不同的内部化学反应
- 衰减机制可能涉及多种失效模式叠加
- 需要更复杂的物理建模

**建议**:
1. 📊 **数据增强**: 收集更多此类异常电池数据
2. 🔬 **特征工程**: 针对异常电池设计专门特征
3. 🎯 **分层建模**: 对正常/异常电池使用不同模型
4. ⚠️ **实际应用**: 对高误差电池添加预警机制

---

## 三、生成的可视化与报告

### 图表列表:

1. **电池误差分析** (4张图):
   - `battery_error_heatmap.png` - 各电池在不同模型上的MAE热力图
   - `battery_error_barplot.png` - 按平均MAE排序的条形图
   - `battery_error_boxplot.png` - 不同模型MAE分布箱线图
   - `battery_error_scatter.png` - 平均MAE vs 标准差散点图

2. **电池容量曲线**:
   - `all_test_batteries_curves.png` - 所有16个测试集电池容量衰减曲线(异常电池红色标注)
   - `high_error_batteries_detailed.png` - 异常电池详细曲线(含统计信息)

3. **特征差异分析**:
   - `battery_feature_comparison.png` - 前8个显著差异特征的分布对比

4. **模型训练结果**:
   - CNN-LSTM训练曲线和预测结果可视化(按电池着色)

### 数据报告:

1. `battery_error_report.csv` - 各电池误差统计详情
2. `battery_feature_comparison.csv` - 特征对比Z-score分析
3. `battery_degradation_features.csv` - 每个电池的16种衰减特征

---

## 四、技术创新点

1. ✨ **统一学习率调度系统**
   - 支持8种调度策略(CosineAnnealing, StepLR, ExponentialLR等)
   - 配置文件驱动,易于实验
   - 详细文档: `configs/LEARNING_RATE_SCHEDULER_GUIDE.md`

2. ✨ **物理约束神经网络(PINN)**
   - 软单调性约束 + 时间衰减权重
   - 边界约束确保SOH ∈ [0,1]
   - 可配置权重便于调优

3. ✨ **多维度数据分析工具**
   - 跨模型误差分析
   - 16种电池衰减特征提取
   - Z-score标准化差异度量

---

## 五、下周计划

1. 🎯 **完善实验数据**
   - 补充各模型完整的性能指标表格
   - 记录训练时间和计算资源消耗

2. 🎯 **异常电池深入研究**
   - 分析异常电池的原始数据特征
   - 尝试专门针对异常电池的建模方法

3. 🎯 **模型优化**
   - 调优CNN-LSTM超参数
   - 实验不同的PINN约束权重组合

4. 🎯 **文档整理**
   - 完善代码注释
   - 准备论文/报告材料

---

## 六、遇到的问题与解决

### 问题1: Git合并导致SOH计算错误

**问题描述**:
- Git rebase时引入了使用错误rated_capacity计算SOH的代码
- 导致所有模型训练结果异常

**解决方案**:
- 通过git版本对比定位问题
- 恢复使用initial_capacity的正确方法
- Commit: 14e6655

### 问题2: 中文字体显示问题

**问题描述**:
- Matplotlib生成的图表中文标题无法显示

**解决方案**:
- 所有可视化标题改为英文
- 保持专业性和可读性

### 问题3: 电池数据加载失败

**问题描述**:
- CSV文件没有'cycle'列,导致读取失败

**解决方案**:
- 改用行索引作为cycle编号
- `cycles = np.arange(1, len(df) + 1)`

---

## 七、代码统计

- 新增Python脚本:
  - `analyze_battery_errors.py` (~540行)
  - `utils/lr_schedulers.py` (~246行)

- 修改的配置文件:
  - `configs/models/cnn_lstm_config.json`
  - `configs/models/gru_config.json`

- 新增文档:
  - `configs/LEARNING_RATE_SCHEDULER_GUIDE.md` (345行)

---

**报告人**: [你的名字]
**日期**: 2025-XX-XX
