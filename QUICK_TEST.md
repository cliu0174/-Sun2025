# 快速测试指南

## ✅ 系统已修复！

训练脚本现在可以正常运行了。

## 🚀 立即开始

### 方式1: 完整训练（2500 epochs，约需要几分钟）

```bash
python train_single_model.py
```

### 方式2: 快速测试（100 epochs，约需要30秒）

临时修改配置文件 `configs/models/cnn_config.json`:

```json
{
  "training": {
    "num_epochs": 100,    // 改为 100（快速测试）
    "batch_size": 64,
    "learning_rate": 0.001
  }
}
```

然后运行:
```bash
python train_single_model.py
```

测试完成后，记得改回 `"num_epochs": 2500`

---

## 📊 预期输出

训练过程会显示：

```
使用设备: cuda
======================================================================
训练模型: CNN on HUST Battery 1-1
======================================================================

[1/6] 加载配置...
[2/6] 加载数据...
  原始特征数: 16
  训练样本数: 1115
  测试样本数: 372

[3/6] 特征选择...
  筛选后特征数: 6

[4/6] 创建数据加载器...
[5/6] 创建模型...
  模型参数: 2,881

[6/6] 开始训练...
  Training: 100%|████████| 2500/2500
```

---

## 📂 结果保存位置

```
results/results_hust/1-1/cnn/
├── model_checkpoint.pth       # 训练好的模型
├── config.json                # 使用的配置
├── results.pkl                # 详细结果
├── training_history.png       # 训练曲线图
└── predictions.png            # 预测对比图
```

---

## 🎯 训练不同的模型

### 训练FNN

编辑 `train_single_model.py` 底部：
```python
MODEL_TYPE = 'fnn'    # 改这里
BATTERY_ID = '1-1'
```

### 训练LSTM

```python
MODEL_TYPE = 'lstm'   # 改这里
BATTERY_ID = '1-1'
```

### 训练BPINN

```python
MODEL_TYPE = 'bpinn'  # 改这里
BATTERY_ID = '1-1'
```

### 换其他电池数据

```python
MODEL_TYPE = 'cnn'
BATTERY_ID = '3-1'    # 改这里
```

---

## 💡 常用配置调整

### 加快训练（用于测试）

```json
{
  "training": {
    "num_epochs": 100,     // 减少epoch
    "batch_size": 128      // 增大batch size
  }
}
```

### 提高精度（用于最终训练）

```json
{
  "training": {
    "num_epochs": 3000,    // 增加epoch
    "batch_size": 32       // 减小batch size
  }
}
```

### 防止过拟合

```json
{
  "architecture": {
    "dropout_rate": 0.3    // 增加dropout
  }
}
```

---

## 🔧 常见问题

### Q: 提示找不到CUDA？

**A**: 将 `DEVICE` 改为 `'cpu'`:
```python
DEVICE = 'cpu'  # train_single_model.py 底部
```

### Q: 训练太慢？

**A**: 三个解决方案：
1. 减少epochs: `num_epochs: 500`
2. 增大batch size: `batch_size: 128`
3. 使用GPU: 确保 `DEVICE = 'cuda'`

### Q: 内存不足？

**A**: 减小batch size:
```json
{
  "training": {
    "batch_size": 16   // 从64减到16
  }
}
```

### Q: 想看详细的训练日志？

**A**: 训练过程会实时显示，每100个epoch打印一次结果。

---

## 📈 查看结果

### 查看训练曲线

训练完成后，打开：
```
results/results_hust/1-1/cnn/training_history.png
```

### 查看预测对比

打开：
```
results/results_hust/1-1/cnn/predictions.png
```

### 加载结果数据

```python
import pickle

with open('results/results_hust/1-1/cnn/results.pkl', 'rb') as f:
    results = pickle.load(f)

print(f"MAE: {results['mae']*100:.4f}%")
print(f"RMSE: {results['rmse']*100:.4f}%")
```

---

## 🎓 下一步

1. **完成首次测试**
   ```bash
   python train_single_model.py
   ```

2. **查看结果是否满意**
   - 查看图表
   - 检查MAE/RMSE值

3. **如果满意，批量对比所有模型**
   ```bash
   python train_comparison.py
   ```

4. **根据对比结果选择最佳模型**

---

## 📚 完整文档

- [训练脚本详细指南](TRAINING_GUIDE.md)
- [模型工厂快速入门](notes/MODEL_FACTORY_QUICK_START.md)
- [完整API文档](notes/MODEL_FACTORY_GUIDE.md)

---

**准备好了吗？开始训练！** 🚀

```bash
python train_single_model.py
```
