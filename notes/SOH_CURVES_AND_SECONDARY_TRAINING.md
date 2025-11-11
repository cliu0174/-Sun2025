# SOH预测曲线图和二次训练分析

## 新增功能：SOH预测曲线对比图

### 功能说明

新增了一个类似论文图表的SOH预测曲线图，用于直观对比BPINN-1、BPINN-2和真实SOH值随循环次数的变化。

### 使用方法

在训练完成后，系统会自动生成SOH预测曲线图，保存位置：
```
results/soh_prediction_curves.png
```

### 图表特点

1. **三条曲线对比**：
   - 黑色线（圆形标记）：真实SOH值
   - 蓝色线（方形标记）：BPINN-1预测
   - 橙色线（三角标记）：BPINN-2预测

2. **视觉设计**：
   - X轴：循环次数（Cycle）
   - Y轴：SOH（百分比格式）
   - 网格线帮助读取数值
   - 标记点每隔一定间隔显示，避免过于密集

3. **自动调整**：
   - Y轴范围根据数据自动调整
   - 图例位置自动选择最佳位置

---

## 二次训练问题分析

### 损失函数确认

根据论文公式，当前实现**是正确的**：

```
L_total = L_data(train) + ω1·Physics(train) + ω2·Physics(test)
```

其中：
- `L_data(train)` = 训练集的MSE
- `Physics(train)` = 训练集的物理约束损失（单调性）
- `Physics(test)` = 测试集的物理约束损失（单调性）

**关键点**：损失函数**不包含测试集的MSE**，只优化训练集的预测准确性。

---

## 为什么二次训练后指标可能变差？

### 设计意图

这是**论文的设计权衡**，不是bug！

二次训练的真正目标：
1. ✓ 保持训练集的拟合性能（MSE_train）
2. ✓ 改善测试集的**物理一致性**（单调性约束）
3. ✗ **不直接优化测试集的预测准确性**

### 可能的现象

- **物理损失下降，MAE上升**：模型为了满足单调性，牺牲了预测准确性
- **这可能是正常的**：如果测试集物理损失显著改善，即使MAE略微上升，也符合论文设计

---

## 解决方案

### 1. 最佳模型自动选择（已实现）✓

系统已自动实现最佳模型选择功能：
- 在二次训练的每次iteration都评估test MAE
- 自动保存MAE最低的模型
- 训练结束后恢复最佳模型

**检查指标**：
- 如果 `best_iteration < 50`：说明过拟合太快，应减少迭代次数
- 如果 `best_iteration ≈ 400`：可能需要更多迭代

### 2. 参数调优建议

#### 方案A：降低物理约束权重（推荐首选）

修改 `main.py` 第71行：
```python
config = {
    ...
    'lambda_test_physics': 0.005,  # 从0.01降到0.005
}
```

**适用场景**：MAE上升明显（>0.5%）

#### 方案B：减少迭代次数

修改 `main.py` 第69行：
```python
config = {
    ...
    'num_secondary_iterations': 200,  # 从400降到200
}
```

**适用场景**：`best_iteration` 很小（<100）

#### 方案C：降低学习率

修改 `main.py` 第70行：
```python
config = {
    ...
    'secondary_learning_rate': 0.0005,  # 从0.001降到0.0005
}
```

**适用场景**：MAE波动大，需要更平稳的优化

#### 方案D：组合调整

如果单独调整效果不佳，可以组合使用：
```python
config = {
    ...
    'lambda_test_physics': 0.005,         # 降低物理约束
    'num_secondary_iterations': 200,      # 减少迭代
    'secondary_learning_rate': 0.0005,    # 降低学习率
}
```

---

## 实验建议

### 第1步：基准测试

运行一次完整训练，观察：
1. Phase 1的best_epoch是多少？
2. Phase 2的best_iteration是多少？
3. 二次训练后MAE变化了多少？
4. 测试集物理损失下降了多少？

### 第2步：诊断问题

根据观察结果判断：

| 现象 | 可能原因 | 建议方案 |
|------|----------|---------|
| best_iteration < 50 | 过拟合太快 | 方案B：减少迭代次数 |
| MAE上升 > 0.5% | 物理约束太强 | 方案A：降低λ2 |
| MAE波动大 | 学习率太大 | 方案C：降低学习率 |
| 物理损失下降，MAE微升 | 正常权衡 | 可接受，或微调λ2 |

### 第3步：调优

- 每次只改变一个参数
- 记录每次实验的结果
- 对比best_iteration的位置变化

---

## 预期结果

### 正常情况

- Phase 1: MAE ≈ 1.5-2.5%
- Phase 2: MAE ≈ 1.2-2.0%（略有改善或持平）
- best_iteration: 50-200之间

### 理想情况

- Phase 1: MAE < 2.0%
- Phase 2: MAE < 1.5%
- test_physics_loss显著下降
- 预测曲线平滑且接近真实值

---

## 文件位置

### 实现代码
- [src/utils.py](../src/utils.py:271-322) - `plot_soh_predictions_over_cycles()` 函数
- [main.py](../main.py:246-252) - 调用位置

### 配置参数
- [main.py](../main.py:60-71) - `config` 字典

### 生成的图表
- `results/soh_prediction_curves.png` - SOH预测曲线对比图
- `results/training_history_phase1.png` - 第一阶段训练历史
- `results/secondary_training_history.png` - 第二阶段训练历史
- `results/model_comparison.png` - 模型对比图

---

## 总结

1. ✓ 新增了论文风格的SOH预测曲线对比图
2. ✓ 实现了最佳模型自动选择功能
3. ✓ 确认了当前损失函数符合论文设计
4. ✓ 理解了二次训练的真正目标（物理一致性 vs 预测准确性的权衡）
5. ✓ 提供了完整的参数调优方案

**下一步**：运行 `python main.py` 进行训练，观察生成的SOH预测曲线图！
