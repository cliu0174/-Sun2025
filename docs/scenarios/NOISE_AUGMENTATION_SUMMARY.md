# 数据增强功能总结

## ✅ 已完成的修改

### 1. 新增文件

#### [utils/data_augmentation.py](utils/data_augmentation.py)
- **功能**: 数据增强工具函数
- **包含**:
  - `add_gaussian_noise()`: 添加高斯噪声
  - `random_drop_samples()`: 随机丢弃样本
  - `add_degradation_noise()`: 完整的退化噪声流程
  - `NOISE_PRESETS`: 预设的噪声级别 (light/medium/heavy)
  - `get_noise_preset()`: 获取预设参数

#### [docs/DATA_AUGMENTATION_GUIDE.md](docs/DATA_AUGMENTATION_GUIDE.md)
- **内容**: 完整的使用指南
- **包含**: 实验设计、使用方法、噪声级别说明、注意事项

#### [examples/train_with_noise.py](examples/train_with_noise.py)
- **功能**: 示例脚本,展示如何运行对比实验
- **包含**: 4个实验函数和结果对比

### 2. 修改的文件

#### [train_cross_battery.py](train_cross_battery.py)

**修改1**: `prepare_cross_battery_data()` 函数
```python
def prepare_cross_battery_data(all_data, train_batteries, val_batteries, test_batteries,
                               add_noise=False, noise_level='medium', noise_seed=42):
    # 新增: 对训练集和验证集添加噪声
    # 测试集保持干净
```

**修改2**: `train_cross_battery_model()` 函数
```python
def train_cross_battery_model(
    # ... 其他参数
    add_noise=False,             # 新增: 噪声开关
    noise_level='medium',        # 新增: 噪声级别
    noise_seed=None              # 新增: 噪声种子
):
```

**修改3**: 主函数 `__main__`
```python
# 新增数据增强开关
ADD_NOISE = False           # 一键开关
NOISE_LEVEL = 'medium'      # 噪声级别选择
```

## 🎯 核心设计

### 噪声只加在训练/验证集

```
训练流程:
  ┌─────────────┐
  │ 加载原始数据 │
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │ 合并+标准化  │
  └──────┬──────┘
         │
    ┌────▼────┐
    │ add_noise? │
    └─┬────┬──┘
      │    │
    Yes   No
      │    │
  ┌───▼────▼───┐
  │ 训练集: +噪声│
  │ 验证集: +噪声│
  │ 测试集: 干净 │ ← 关键!
  └─────────────┘
```

### 预设噪声级别

| 级别 | 特征噪声σ | 容量噪声σ | 丢弃率 | 用途 |
|------|----------|----------|--------|------|
| `light` | 0.01 | 0.005 | 5% | 轻度退化 |
| `medium` | 0.05 | 0.02 | 15% | **典型场景** (推荐) |
| `heavy` | 0.10 | 0.05 | 30% | 压力测试 |

## 🔧 使用方法

### 快速开始 (推荐)

只需修改 [train_cross_battery.py](train_cross_battery.py) 的一个变量:

```python
# 第1424行
ADD_NOISE = True  # 改为 True 启用噪声
```

然后正常运行:
```bash
python train_cross_battery.py
```

### 切换噪声级别

```python
NOISE_LEVEL = 'light'    # 轻度噪声
NOISE_LEVEL = 'medium'   # 中度噪声 (推荐)
NOISE_LEVEL = 'heavy'    # 重度噪声
```

### 完全向后兼容

如果不修改任何代码,默认 `ADD_NOISE = False`,所有行为与之前完全一致。

## 📊 实验设计

### 验证物理约束的鲁棒性

| 实验 | 训练数据 | 物理约束 | 测试集MAE | 改善幅度 |
|------|---------|---------|-----------|---------|
| 1 | 干净 | ❌ | X% | - (基线) |
| 2 | 干净 | ✅ | Y% | X% - Y% |
| 3 | 噪声 | ❌ | Z% | - (噪声基线) |
| 4 | 噪声 | ✅ | W% | Z% - W% |

**关键对比**:
- 如果 `(Z% - W%)` > `(X% - Y%)`, 说明**物理约束在噪声环境下更有价值**!

### 运行步骤

1. **实验1**: `ADD_NOISE=False` + 配置文件关闭物理约束
2. **实验2**: `ADD_NOISE=False` + 配置文件启用物理约束
3. **实验3**: `ADD_NOISE=True` + 配置文件关闭物理约束
4. **实验4**: `ADD_NOISE=True` + 配置文件启用物理约束

## 🎨 输出示例

启用噪声时的输出:

```
======================================================================
跨电池训练: CNN_LSTM
数据划分: Train/Val/Test = 60%/20%/20%
数据增强: 启用 (中度噪声 (15% drop, σ_feat=0.05, σ_targ=0.02))
======================================================================

...

======================================================================
数据增强: 中度噪声 (15% drop, σ_feat=0.05, σ_targ=0.02)
======================================================================

对训练集添加噪声:
======================================================================
数据退化增强 (模拟实测数据质量问题)
======================================================================
原始样本数: 50000
特征噪声: σ=0.0500, 实际噪声水平=0.0502
容量噪声: σ=0.0200, 实际噪声水平=0.0199
随机丢弃: 7500 个样本 (15.0%)
剩余样本数: 42500
======================================================================

对验证集添加噪声:
...

======================================================================
⚠️  测试集保持干净 (用于公平对比)
======================================================================
```

## ⚙️ 技术细节

### 1. 高斯噪声

```python
noise = np.random.normal(0, noise_std, size=data.shape)
noisy_data = data + noise
```

### 2. 容量值裁剪

```python
# 容量噪声后裁剪到 [0, 1.2]
noisy_targets = np.clip(noisy_targets, 0.0, 1.2)
```

### 3. 随机丢弃保持时序

```python
# 丢弃后保持索引顺序
keep_indices = np.random.choice(n_total, size=n_keep, replace=False)
keep_indices = np.sort(keep_indices)  # 保持时序
```

### 4. 不同随机种子

```python
# 训练集: seed = noise_seed
# 验证集: seed = noise_seed + 1000  # 避免完全相同
```

## ⚠️ 注意事项

### ✅ DO (应该做)
- 训练集和验证集加噪声
- 测试集保持干净
- 使用固定的 `noise_seed` 确保可复现
- 记录每个实验的噪声参数

### ❌ DON'T (不要做)
- ❌ 不要在测试集上加噪声
- ❌ 不要忘记记录噪声级别
- ❌ 不要在不同实验间改变 `noise_seed`

## 📁 项目结构

```
1111-soh/
├── utils/
│   └── data_augmentation.py          # 新增: 数据增强工具
├── docs/
│   └── DATA_AUGMENTATION_GUIDE.md    # 新增: 使用指南
├── examples/
│   └── train_with_noise.py           # 新增: 示例脚本
├── train_cross_battery.py            # 修改: 添加噪声开关
└── NOISE_AUGMENTATION_SUMMARY.md     # 本文档
```

## 🚀 快速参考

### 启用噪声
```python
ADD_NOISE = True
NOISE_LEVEL = 'medium'
```

### 关闭噪声 (恢复默认)
```python
ADD_NOISE = False
```

### 测试工具函数
```bash
python utils/data_augmentation.py
```

### 查看完整指南
参见 [docs/DATA_AUGMENTATION_GUIDE.md](docs/DATA_AUGMENTATION_GUIDE.md)

## 🎓 理论基础

### 为什么要加噪声?

1. **鲁棒性测试**: 实际采集的电池数据常有噪声和缺失
2. **验证物理约束**: 物理约束应该能帮助模型过滤噪声,学到更本质的规律
3. **对比实验**: 干净数据 vs 噪声数据,有无物理约束的对比

### 为什么测试集不加噪声?

1. **公平对比**: 测试集是评估标准,必须一致
2. **实际场景**: 我们希望模型在训练数据差的情况下,仍能在真实数据上表现好
3. **可复现**: 保持测试集不变,可以与之前的实验对比

## 📞 问题反馈

如有问题,请检查:
1. [docs/DATA_AUGMENTATION_GUIDE.md](docs/DATA_AUGMENTATION_GUIDE.md) - 完整使用指南
2. [utils/data_augmentation.py](utils/data_augmentation.py) - 工具函数源码
3. [examples/train_with_noise.py](examples/train_with_noise.py) - 示例代码

---

**版本**: 1.0
**日期**: 2025-12-12
**状态**: ✅ 完成并测试
