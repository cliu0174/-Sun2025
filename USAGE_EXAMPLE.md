# BPINN使用示例

## 快速开始

### 1. 运行完整训练

```bash
python main.py
```

这将执行：
- ✓ Phase 1: 主训练（2000 epochs）
- ✓ Phase 2: 二次训练（400 iterations）
- ✓ 自动选择最佳模型
- ✓ 生成所有图表和结果

### 2. 查看结果

训练完成后，检查 `results/` 目录：

```
results/
├── soh_prediction_curves.png        # ⭐ 新增：SOH预测曲线对比图
├── training_history_phase1.png      # Phase 1训练历史
├── secondary_training_history.png   # Phase 2训练历史
├── predictions_phase1.png            # BPINN-1预测散点图
├── predictions_phase2.png            # BPINN-2预测散点图
├── model_comparison.png              # 模型对比
├── bpinn_model_phase1.pth           # Phase 1最佳模型
└── bpinn_model.pth                   # 最终模型
```

### 3. 理解输出

#### 控制台输出示例

```
======================================================================
Phase 1: Primary Training (BPINN-1)
======================================================================
Training: 100%|████████████| 2000/2000 [10:30<00:00, 3.17it/s]

*** New best model at epoch 1156: MAE=0.016234, RMSE=0.020123 ***

Epoch 2000/2000
Train Loss: 0.000234 (Data: 0.000230, Physics: 0.000004)
Test MAE: 0.017845, RMSE: 0.021234
Current Best: Epoch 1156, MAE=0.016234, RMSE=0.020123

=== Restored best model from epoch 1156 ===
Best MAE: 0.016234, Best RMSE: 0.020123

======================================================================
Evaluating BPINN-1 (After Primary Training)
======================================================================
Best model was from epoch 1156
Best MAE: 0.016234, Best RMSE: 0.020123

==================================================
BPINN-1 Performance:
==================================================
MAE:  0.0162 (1.62%)
RMSE: 0.0201 (2.01%)
==================================================

======================================================================
Phase 2: Secondary Training (BPINN-2)
======================================================================
Secondary Training: 100%|████████████| 400/400 [01:20<00:00, 4.98it/s]

*** New best model at iteration 87: MAE=0.015123, RMSE=0.019234 ***

Iteration 400/400
Total Loss: 0.000198
Train MSE: 0.000195, Train Physics: 0.000001
Test Physics Loss: 0.000002
Test MAE: 0.016234, RMSE: 0.020123
Current Best: Iteration 87, MAE=0.015123, RMSE=0.019234

=== Restored best model from iteration 87 ===
Best MAE: 0.015123, Best RMSE: 0.019234

======================================================================
Evaluating BPINN-2 (After Secondary Training)
======================================================================
Best model was from iteration 87
Best MAE: 0.015123, Best RMSE: 0.019234

==================================================
BPINN-2 Performance:
==================================================
MAE:  0.0151 (1.51%)
RMSE: 0.0192 (1.92%)
==================================================
```

#### 关键指标解读

1. **Best Epoch/Iteration**: 最佳模型出现的位置
   - 如果best_epoch远小于2000：可能过拟合
   - 如果best_iteration < 50：二次训练迭代太多

2. **MAE和RMSE**:
   - MAE < 2%：良好
   - MAE < 1.5%：优秀
   - MAE < 0.5%：接近论文水平（需要更好的特征提取）

3. **Physics Loss**:
   - 应该趋近于0
   - Phase 2的test_physics_loss应该明显下降

---

## 参数调整

### 场景1：Phase 1过拟合

**现象**：best_epoch < 1000

**解决**：
```python
# main.py
config = {
    ...
    'num_epochs': 1500,  # 减少epoch数
}
```

### 场景2：Phase 2效果变差

**现象**：Phase 2 MAE > Phase 1 MAE

**解决**：
```python
# main.py
config = {
    ...
    'lambda_test_physics': 0.005,      # 降低物理约束
    'num_secondary_iterations': 200,   # 减少迭代次数
}
```

### 场景3：训练太慢

**解决**：
```python
# main.py
config = {
    ...
    'num_epochs': 1000,                # 减少epoch
    'num_secondary_iterations': 200,   # 减少iteration
}
```

---

## 查看SOH预测曲线

打开 `results/soh_prediction_curves.png`，你会看到类似论文的图表：

```
     ┌─────────────────────────────────────┐
 100%│   Real (黑色圆点)                    │
     │   BPINN-1 (蓝色方块)                 │
  90%│   BPINN-2 (橙色三角)                 │
     │         ╲                            │
  80%│          ╲╲                          │
 SOH │           ╲╲                         │
  70%│            ╲╲╲                       │
     │              ╲╲                      │
  60%│               ╲                      │
     └─────────────────────────────────────┘
       0    20    40    60    80    100
                   Cycle
```

**理想情况**：
- BPINN-2的线最接近Real
- 三条线平滑，没有大幅波动
- BPINN-2比BPINN-1更稳定

---

## 常见问题

### Q1: 为什么Phase 2反而变差了？

**A**: 这是正常的设计权衡。Phase 2的目标是改善物理一致性（单调性），不是直接优化MAE。如果：
- test_physics_loss显著下降
- MAE只是轻微上升（<0.5%）

那么这是可接受的。如果MAE上升太多，参考"参数调整"部分。

### Q2: 如何知道训练是否成功？

**A**: 检查以下指标：
1. ✓ MAE < 2%
2. ✓ RMSE < 2.5%
3. ✓ Physics loss接近0
4. ✓ SOH曲线平滑

### Q3: 如何与论文结果对比？

**A**: 论文中报告的是各个电池单独测试的结果：
- B05: MAE < 0.4%, RMSE < 0.5%
- B06: MAE < 0.3%, RMSE < 0.4%
- B07: MAE < 0.4%, RMSE < 0.5%

当前实现是混合测试，所以指标可能在1.5-2.5%范围是正常的。

### Q4: 如何改善结果？

**A**:
1. **检查IC特征提取**：确保MATLAB提取的特征正确
2. **调整参数**：参考"参数调整"部分
3. **增加数据**：使用更多电池数据
4. **分别训练**：为每个电池单独训练模型

---

## 下一步

1. ✓ 运行 `python main.py`
2. ✓ 查看 `results/soh_prediction_curves.png`
3. ✓ 根据结果调整参数
4. ✓ 重新训练并对比

如有问题，查看：
- [notes/IMPLEMENTATION_NOTES.md](notes/IMPLEMENTATION_NOTES.md) - 实现细节
- [notes/SOH_CURVES_AND_SECONDARY_TRAINING.md](notes/SOH_CURVES_AND_SECONDARY_TRAINING.md) - 详细分析
