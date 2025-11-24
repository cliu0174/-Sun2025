# 回滚到MAE判断标准 - 修改总结

**修改日期**: 2025-11-20
**操作**: 从val_loss回滚到val_mae作为最佳模型判断标准

---

## 修改内容

### 1. 初始化变量 (第329-332行)
```python
# 修改前
best_val_loss = float('inf')

# 修改后
best_val_mae = float('inf')
```

### 2. 保存最佳模型逻辑 (第396-400行)
```python
# 修改前
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_epoch = epoch + 1
    best_model_state = {...}
    patience_counter = 0

# 修改后
if val_mae < best_val_mae:
    best_val_mae = val_mae
    best_epoch = epoch + 1
    best_model_state = {...}
    patience_counter = 0
```

### 3. 打印信息 (第407-413行)
```python
# 修改前
print(f"  Val Loss:   {val_loss:.6f} (Best: {best_val_loss:.6f})")
print(f"  Val MAE:    {val_mae*100:.4f}%")

# 修改后
print(f"  Val Loss:   {val_loss:.6f}")
print(f"  Val MAE:    {val_mae*100:.4f}% (Best: {best_val_mae*100:.4f}%)")
```

### 4. Early Stopping提示 (第418-419行)
```python
# 修改前
print(f"早停触发！验证集loss连续{patience}个epoch未改善")
print(f"最佳epoch: {best_epoch}, 最佳val_loss: {best_val_loss:.6f}")

# 修改后
print(f"早停触发！验证集MAE连续{patience}个epoch未改善")
print(f"最佳epoch: {best_epoch}, 最佳val_mae: {best_val_mae*100:.4f}%")
```

### 5. 恢复最佳模型提示 (第428行)
```python
# 修改前
print(f"已恢复最佳模型 (Epoch {best_epoch}, Val Loss: {best_val_loss:.6f})")

# 修改后
print(f"已恢复最佳模型 (Epoch {best_epoch}, Val MAE: {best_val_mae*100:.4f}%)")
```

### 6. 保存模型信息 (第466行)
```python
# 修改前
wrapper.best_loss = best_val_loss

# 修改后
wrapper.best_mae = best_val_mae
```

### 7. 保存结果字典 (第490行)
```python
# 修改前
'best_val_loss': float(best_val_loss),

# 修改后
'best_val_mae': float(best_val_mae),
```

---

## 当前训练策略

### 最佳模型选择
- **判断标准**: Validation MAE (平均绝对误差百分比)
- **选择逻辑**: 选择验证集MAE最小的epoch的模型
- **Early Stopping**: 基于MAE连续N个epoch不改善时触发

### 训练流程
```
for each epoch:
    1. 训练 (计算train_loss)
    2. 更新学习率 (scheduler.step())
    3. 验证 (计算val_loss, val_mae, val_rmse)
    4. 如果 val_mae < best_val_mae:
          保存模型
          重置patience计数器
    5. 如果 patience_counter > patience:
          触发early stopping
```

### 打印示例
```
Epoch [10/200]
  LR:         0.004667
  Train Loss: 0.012345
  Val Loss:   0.011234
  Val MAE:    3.4567% (Best: 3.2100%)
  Val RMSE:   4.5678%
  Best Epoch: 8
  Patience:   2/30
```

---

## 为什么使用MAE而不是Loss？

### 使用MAE的理由
1. **直观性**: MAE直接反映预测误差的百分比，更容易理解
2. **实用性**: MAE是最终评估指标，直接优化目标更合理
3. **鲁棒性**: MAE对异常值不如MSE敏感

### 原始论文使用Loss的理由
1. **训练目标一致性**: 优化器优化的就是loss
2. **理论基础**: Loss是训练的直接目标函数
3. **数值稳定性**: Loss的数值范围可能更稳定

### 两种方法的区别
- **Loss (MSE)**: 误差的平方，对大误差更敏感
- **MAE**: 误差的绝对值，更平衡地对待各种误差

在实践中，两种方法通常会选择相近的模型，但在边界情况下可能有差异。

---

## 注意事项

1. **配置文件未修改**: patience仍然是30（你在配置中修改的值）
2. **Scheduler时机**: 仍然在validation之前调用（已修正）
3. **Early Stopping逻辑**: 完全一致，只是判断指标从loss改为mae
4. **测试集评估**: 不受影响，仍然使用最佳模型在测试集评估

---

## 当前配置总结

```json
{
  "training": {
    "num_epochs": 200,
    "batch_size": 512,
    "learning_rate": 0.002,
    "scheduler": {
      "enabled": true,
      "warmup_epochs": 30,
      "warmup_lr": 0.002,
      "base_lr": 0.01,
      "final_lr": 0.0001
    },
    "early_stopping": {
      "enabled": true,
      "patience": 30,  // 注意：这是30，不是10
      "min_delta": 1e-5
    }
  }
}
```

**最佳模型判断**: ✅ 基于 **val_mae**
**Early Stopping**: ✅ 基于 **val_mae**
**Patience**: ⚠️ 30个epoch（比原始论文的10更宽松）

---

**修改完成**: 2025-11-20
**状态**: ✅ 已回滚到MAE判断标准
**可以训练**: ✅ 是
