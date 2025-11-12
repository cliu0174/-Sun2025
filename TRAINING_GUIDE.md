# 训练脚本使用指南

## 🚀 快速开始

### 方法1: 训练单个模型（推荐新手）

```bash
python train_single_model.py
```

**默认配置**: CNN模型在HUST 1-1数据上训练

**自定义配置**: 编辑 `train_single_model.py` 底部的参数

```python
MODEL_TYPE = 'cnn'      # 改成 'fnn', 'lstm', 'bpinn'
BATTERY_ID = '1-1'      # 改成其他电池ID
DEVICE = 'cuda'         # 或 'cpu'
```

### 方法2: 批量对比多个模型（推荐实验）

```bash
python train_comparison.py
```

**默认**: 训练所有4个模型 (FNN, CNN, LSTM, BPINN)

**自定义**: 编辑 `train_comparison.py` 底部的参数

```python
MODEL_TYPES = ['fnn', 'cnn']  # 只训练FNN和CNN
BATTERY_ID = '1-1'
```

---

## 📂 输出结果

### 单模型训练输出

```
results/results_hust/1-1/cnn/
├── model_checkpoint.pth       # 模型检查点
├── config.json                # 训练配置
├── results.pkl                # 详细结果
├── training_history.png       # 训练曲线
└── predictions.png            # 预测对比图
```

### 批量对比输出

```
results/comparison_hust/1-1/
├── comparison_table.csv       # 对比表格
├── comparison_report.md       # 详细报告
├── metrics_comparison.png     # 指标对比
├── training_curves_comparison.png  # 训练曲线对比
└── predictions_comparison.png # 预测对比
```

---

## 🎯 使用场景

### 场景1: 训练CNN在HUST 1-1数据上

```bash
# 直接运行（使用默认配置）
python train_single_model.py
```

### 场景2: 训练FNN在HUST 3-1数据上

编辑 `train_single_model.py`:
```python
MODEL_TYPE = 'fnn'
BATTERY_ID = '3-1'
```

然后运行:
```bash
python train_single_model.py
```

### 场景3: 对比所有模型在HUST 1-1上的性能

```bash
python train_comparison.py
```

### 场景4: 只对比FNN和BPINN

编辑 `train_comparison.py`:
```python
MODEL_TYPES = ['fnn', 'bpinn']
```

然后运行:
```bash
python train_comparison.py
```

---

## ⚙️ 配置调整

### 修改训练参数

所有训练参数在配置文件中：`configs/models/`

例如，修改CNN的学习率：

编辑 `configs/models/cnn_config.json`:
```json
{
  "training": {
    "learning_rate": 0.0005,  // 从0.001改为0.0005
    "batch_size": 32,         // 从64改为32
    "num_epochs": 3000        // 从2500改为3000
  }
}
```

### 临时覆盖参数（不修改配置文件）

在 `train_single_model.py` 中，找到创建wrapper的部分，添加：

```python
# 加载配置后
config = ConfigLoader.load_model_config(model_type)

# 临时修改
config['training']['learning_rate'] = 0.0005
config['training']['batch_size'] = 32

# 然后创建wrapper
wrapper = UnifiedModelWrapper(
    model_type=model_type,
    input_size=input_size,
    config=config,  # 使用修改后的配置
    device=device
)
```

---

## 📊 查看结果

### 查看训练曲线

```python
# 在训练完成后，图片自动保存
# 查看: results/results_hust/1-1/cnn/training_history.png
```

### 加载保存的模型

```python
from models import UnifiedModelWrapper

# 加载检查点
wrapper = UnifiedModelWrapper.load_checkpoint(
    'results/results_hust/1-1/cnn/model_checkpoint.pth'
)

# 使用模型进行预测
model = wrapper.model
model.eval()
# predictions = model(test_data)
```

### 读取结果数据

```python
import pickle

# 加载结果
with open('results/results_hust/1-1/cnn/results.pkl', 'rb') as f:
    results = pickle.load(f)

print(f"MAE: {results['mae']}")
print(f"RMSE: {results['rmse']}")
print(f"Best Epoch: {results['best_epoch']}")

# 访问预测值和真实值
predictions = results['predictions']
targets = results['targets']
```

---

## 🔧 高级用法

### 1. 训练时动态调整学习率

在 `train_single_model.py` 中，学习率调度器已经支持：

编辑配置文件启用：
```json
{
  "training": {
    "scheduler": {
      "enabled": true,     // 改为true
      "type": "StepLR",
      "step_size": 500,    // 每500个epoch降低学习率
      "gamma": 0.9         // 降低到原来的90%
    }
  }
}
```

### 2. 特征选择

配置文件中已支持：
```json
{
  "feature_selection": {
    "enabled": true,               // 是否启用
    "correlation_threshold": 0.5,  // 相关性阈值
    "top_k": 6                     // 选择前k个特征
  }
}
```

### 3. 早停（Early Stopping）

```json
{
  "training": {
    "early_stopping": {
      "enabled": true,      // 启用早停
      "patience": 100,      // 容忍100个epoch不改善
      "min_delta": 1e-5     // 最小改善阈值
    }
  }
}
```

---

## 💡 常见问题

### Q1: 训练太慢？

**解决方案**:
1. 减少epoch数量: `num_epochs: 1000`
2. 增加batch size: `batch_size: 128`
3. 使用GPU: 确保 `DEVICE = 'cuda'`

### Q2: 过拟合？

**解决方案**:
1. 增加dropout: `dropout_rate: 0.3`
2. 减少模型大小: `hidden_sizes: [32, 16]`
3. 使用更多训练数据: `train_ratio: 0.8`

### Q3: 欠拟合？

**解决方案**:
1. 增加模型容量: `hidden_sizes: [128, 64, 32]`
2. 降低dropout: `dropout_rate: 0.1`
3. 训练更多轮: `num_epochs: 5000`

### Q4: 如何在不同电池数据上训练？

只需修改 `BATTERY_ID`:
```python
BATTERY_ID = '3-1'  # 或其他电池ID
```

### Q5: 如何保证实验可复现？

系统自动保存完整配置：
- 配置文件: `config.json`
- 随机种子: 在训练脚本中可设置
- 模型检查点: `model_checkpoint.pth`

---

## 📈 性能优化建议

### 针对小数据集（<1000样本）

```json
{
  "architecture": {
    "hidden_sizes": [32, 16],
    "dropout_rate": 0.3
  },
  "training": {
    "batch_size": 16,
    "learning_rate": 0.0001,
    "num_epochs": 3000
  }
}
```

### 针对大数据集（>5000样本）

```json
{
  "architecture": {
    "hidden_sizes": [128, 64, 32],
    "dropout_rate": 0.2
  },
  "training": {
    "batch_size": 128,
    "learning_rate": 0.001,
    "num_epochs": 2000
  }
}
```

### BPINN特殊配置

```json
{
  "architecture": {
    "hidden_sizes": [10, 10, 10]  // 保持小网络
  },
  "training": {
    "num_epochs": 2000,
    "lambda_physics": 0.01        // 物理约束权重
  },
  "secondary_training": {
    "enabled": true,              // 启用二次训练
    "num_iterations": 400
  }
}
```

---

## 🎓 完整工作流程示例

### 步骤1: 训练单个模型测试

```bash
# 快速测试CNN
python train_single_model.py
```

### 步骤2: 查看结果并调整参数

查看 `results/results_hust/1-1/cnn/training_history.png`

如果需要调整，修改 `configs/models/cnn_config.json`

### 步骤3: 批量对比所有模型

```bash
python train_comparison.py
```

### 步骤4: 分析对比结果

查看 `results/comparison_hust/1-1/comparison_report.md`

### 步骤5: 选择最佳模型并微调

根据对比结果，选择最佳模型，然后：
1. 调整该模型的超参数
2. 重新训练获得最佳性能

---

## 🔗 相关文档

- [模型工厂使用指南](notes/MODEL_FACTORY_GUIDE.md) - 详细的API文档
- [快速入门](notes/MODEL_FACTORY_QUICK_START.md) - 5分钟快速入门
- [示例代码](example_model_factory_usage.py) - 基础示例

---

## ✅ 检查清单

开始训练前，确认：

- [ ] 数据文件存在: `data/HUST data/1-1.csv`
- [ ] 配置文件正确: `configs/models/cnn_config.json`
- [ ] GPU可用（可选）: `torch.cuda.is_available()`
- [ ] 输出目录权限: `results/`

---

**现在就开始训练吧！**

```bash
# 单模型训练
python train_single_model.py

# 或批量对比
python train_comparison.py
```
