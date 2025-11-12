# 模型工厂系统 - 快速入门

## 🎯 三种使用方式

### 方式1: 最简单（推荐新手）

```python
from models import ModelFactory

# 创建模型（自动加载配置）
model = ModelFactory.create_model('fnn', input_size=6)
```

### 方式2: 自定义参数（推荐实验）

```python
from models import ModelFactory

# 覆盖部分参数
model = ModelFactory.create_model(
    model_type='fnn',
    input_size=6,
    hidden_sizes=[128, 64, 32],  # 自定义隐藏层
    dropout_rate=0.3             # 自定义dropout
)
```

### 方式3: 完整包装器（推荐生产）

```python
from models import UnifiedModelWrapper

# 创建完整训练环境
wrapper = UnifiedModelWrapper(
    model_type='fnn',
    input_size=6,
    device='cuda'
)

# 直接获取所有组件
model = wrapper.model
optimizer = wrapper.get_optimizer()
criterion = wrapper.criterion
```

---

## 🔥 常用代码片段

### 创建所有模型类型

```python
for model_type in ['fnn', 'cnn', 'lstm', 'bpinn']:
    model = ModelFactory.create_model(model_type, input_size=6)
    print(f"{model_type.upper()}: {sum(p.numel() for p in model.parameters())} params")
```

### 查看配置

```python
from models import ConfigLoader

config = ConfigLoader.load_model_config('fnn')
ConfigLoader.print_config(config)
```

### 保存和加载模型

```python
# 保存
wrapper.save_checkpoint('model.pth', epoch=100)

# 加载
wrapper = UnifiedModelWrapper.load_checkpoint('model.pth')
```

---

## 📂 支持的模型

| 模型 | 标识 | 特点 |
|------|------|------|
| FNN | `'fnn'` | 简单前馈网络，适合快速实验 |
| CNN | `'cnn'` | 1D卷积，适合特征提取 |
| LSTM | `'lstm'` | 时序建模，适合序列数据 |
| BPINN | `'bpinn'` | 物理约束，适合SOH预测 |

---

## 🎓 回答你的问题

### 问题1: 不同数据集是否要调参？

**答：需要。** 建议调整：
- `batch_size`: 根据样本数量（小数据集用16-32，大数据集用64-128）
- `learning_rate`: 根据收敛情况（通常0.0001-0.001）
- `num_epochs`: 根据过拟合情况

**不建议改变**:
- `hidden_sizes`: 保持一致以便公平对比

```python
# NASA数据集（小数据集）
model_nasa = ModelFactory.create_model('fnn', input_size=6)
# 训练时: batch_size=32, epochs=2500

# HUST数据集（大数据集）
model_hust = ModelFactory.create_model('fnn', input_size=8)
# 训练时: batch_size=64, epochs=2000
```

### 问题2: 统一模型库？

**答：已完成！** 就是现在这个系统。

```python
# 统一接口创建任何模型
model = ModelFactory.create_model(
    model_type='fnn',  # 改成'cnn', 'lstm', 'bpinn'即可切换
    input_size=6
)
```

### 问题3: 动态参数？

**答：支持！** 两种方式：

```python
# 方式1: kwargs覆盖
model = ModelFactory.create_model('fnn', input_size=6, hidden_sizes=[256, 128])

# 方式2: 修改配置
config = ConfigLoader.load_model_config('fnn')
config['architecture']['hidden_sizes'] = [256, 128]
model = ModelFactory.create_model('fnn', input_size=6, config=config)
```

### 问题4: 保护现有结果？

**答：完全兼容！**

- ✅ 旧代码不受影响
- ✅ 新系统使用不同的导入方式
- ✅ 配置文件管理，结果可追溯

```python
# 旧代码（继续工作）
from models import FNN
model = FNN(input_size=6, hidden_sizes=[64, 32, 16])

# 新代码（推荐）
from models import ModelFactory
model = ModelFactory.create_model('fnn', input_size=6)
```

### 问题5: 公平对比？

**答：推荐两阶段。**

**阶段1 - 公平对比（证明算法优势）:**
```python
# 统一关键参数
common_params = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 2000
}

# 在配置文件中设置相同的参数
# 或在训练时使用相同的训练循环
```

**阶段2 - 各自最优（寻找最佳性能）:**
```python
# 每个模型独立调优
fnn_params = {'learning_rate': 0.001, 'hidden_sizes': [128, 64, 32]}
lstm_params = {'learning_rate': 0.0005, 'hidden_size': 64}
```

---

## 📊 完整训练示例

```python
from models import UnifiedModelWrapper
from data_loaders import load_hust_battery_data
import torch

# 1. 加载数据
data_dict = load_hust_battery_data('1-1')

# 2. 创建模型
wrapper = UnifiedModelWrapper(
    model_type='fnn',
    input_size=data_dict['train_features'].shape[1],
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 3. 获取训练组件
model = wrapper.model
optimizer = wrapper.get_optimizer(learning_rate=0.001)
criterion = wrapper.criterion

# 4. 训练（伪代码）
for epoch in range(2000):
    # 训练步骤...
    pass

# 5. 保存
wrapper.save_checkpoint('results/fnn_model.pth', epoch=2000)
```

---

## 💡 最佳实践

1. **使用配置文件管理实验**
   ```python
   ConfigLoader.save_config(config, f'results/exp_{exp_id}_config.json')
   ```

2. **对比实验时打印配置**
   ```python
   ConfigLoader.print_config(config)
   ```

3. **检查点包含完整信息**
   ```python
   wrapper.save_checkpoint('model.pth', epoch=100, optimizer_state=optimizer.state_dict())
   ```

4. **使用统一接口便于切换**
   ```python
   # 只需改一行就能切换模型
   model_type = 'fnn'  # 'cnn', 'lstm', 'bpinn'
   model = ModelFactory.create_model(model_type, input_size=6)
   ```

---

## 🔗 更多信息

- 详细文档: [MODEL_FACTORY_GUIDE.md](MODEL_FACTORY_GUIDE.md)
- 示例代码: [example_model_factory_usage.py](../example_model_factory_usage.py)
- 配置文件: [configs/models/](../configs/models/)

---

**开始使用吧！运行示例查看效果：**
```bash
python example_model_factory_usage.py
```
