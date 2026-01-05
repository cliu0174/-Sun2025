## 数据集迁移指南: HUST → MIT

### 概述

你的代码现在支持两个数据集:
- **HUST数据集**: 77个电池，平均~1800 cycles
- **MIT数据集**: 125个电池，平均~685 cycles

所有训练代码、模型、物理约束都可以无缝迁移！

---

## 快速开始

### 1. 在MIT数据集上训练

#### 方法A: 使用专用脚本 (推荐)

```bash
# 训练CNN-LSTM
python train_mit.py

# 脚本会自动:
# 1. 加载MIT数据集 (125个电池)
# 2. 划分60%/20%/20%
# 3. 使用相同的训练流程
# 4. 保存结果到 results/mit_cnn_lstm/
```

#### 方法B: 修改现有脚本

只需修改 `train_cross_battery.py` 中的 `load_all_batteries` 调用:

```python
# 原来 (HUST)
battery_names, all_data = load_all_batteries(
    data_dir='data/HUST data'
)

# 改为 (MIT)
from data_loaders import load_all_mit_batteries
battery_names, all_data = load_all_mit_batteries(
    data_dir='data/MIT data'
)
```

其他代码**完全不用改**！

---

## 数据集对比

### HUST vs MIT 数据特征

| 特性 | HUST | MIT |
|------|------|-----|
| 电池数量 | 77 | 125 |
| 平均Cycles | ~1800 | ~685 |
| Cycles范围 | 1000+ | 36 - 2156 |
| 特征数 | 16 | 16 (相同!) |
| 目标值 | SOH (已归一化) | capacity → SOH (需计算) |

**关键点**: 两个数据集的**特征维度相同**(16维)，所以:
- ✅ 模型架构完全兼容
- ✅ 预训练模型可迁移
- ✅ 物理约束直接适用

---

## XGBoost在MIT数据集上训练

### 方法1: 修改训练脚本

编辑 `train_xgboost_baseline.py`，在 `train_xgboost_baseline` 函数开头修改:

```python
# 找到这一行 (约第51行)
battery_names, all_data = load_all_batteries(...)

# 改为
from data_loaders import load_all_mit_batteries
battery_names, all_data = load_all_mit_batteries(
    data_dir='data/MIT data',
    apply_cleaning=apply_cleaning
)
```

然后运行:
```bash
python train_xgboost_baseline.py
```

### 方法2: 创建MIT专用版本

复制并修改:
```bash
cp train_xgboost_baseline.py train_xgboost_mit.py
# 然后修改load_all_batteries → load_all_mit_batteries
python train_xgboost_mit.py
```

---

## 数据加载器详情

### MIT数据加载器特点

```python
from data_loaders import load_all_mit_batteries

# 加载所有MIT电池
battery_names, all_data = load_all_mit_batteries(
    data_dir='data/MIT data',
    apply_cleaning=False  # 可选: 3-sigma清洗
)

# 返回格式与HUST完全相同:
# battery_names: ['2017-05-12_b1', '2017-05-12_b2', ...]
# all_data: {
#     '2017-05-12_b1': (features, soh, capacity),
#     ...
# }
```

**自动处理**:
- ✅ 从3个批次加载: `2017-05-12`, `2017-06-30`, `2018-04-12`
- ✅ SOH自动计算: `SOH = capacity / initial_capacity`
- ✅ 电池命名规范化: `2017-05-12_battery-1` → `2017-05-12_b1`

---

## 完整实验流程示例

### 场景: 在两个数据集上对比模型性能

```bash
# 1. 在HUST上训练
python train_cross_battery.py  # 或你的脚本

# 2. 在MIT上训练
python train_mit.py

# 3. 对比结果
python compare_datasets.py  # (可自己编写)
```

### 预期结果差异

由于MIT数据集特点不同，可能会看到:
- **RMSE**: MIT可能略高 (电池多样性更大)
- **训练速度**: MIT更快 (平均cycles更少)
- **泛化能力**: MIT测试集更能体现跨电池泛化

---

## 物理约束迁移

**好消息**: 物理约束无需修改！

```python
# 物理约束在MIT上同样适用
# 因为SOH的物理特性是一致的:
# - 单调递减
# - 边界约束 [0, 1]
# - 平滑变化

# 使用带物理约束的配置
model = ModelFactory.create_model(
    model_type='cnn_lstm',  # 自动加载config中的physics_constraints
    input_size=16
)
```

---

## 文件结构

迁移后的文件组织:

```
├── data/
│   ├── HUST data/           # HUST数据集
│   │   ├── 1-1.csv
│   │   └── ...
│   └── MIT data/            # MIT数据集
│       ├── 2017-05-12/
│       ├── 2017-06-30/
│       └── 2018-04-12/
│
├── data_loaders/
│   ├── data_loader_hust.py  # HUST加载器
│   └── data_loader_mit.py   # MIT加载器 (新增)
│
├── train_cross_battery.py   # HUST训练脚本
├── train_mit.py             # MIT训练脚本 (新增)
├── train_xgboost_baseline.py
│
└── results/
    ├── cross_battery/       # HUST结果
    │   └── cnn_lstm/
    └── mit_cnn_lstm/        # MIT结果 (新增)
```

---

## 常见问题

### Q1: MIT和HUST的SOH范围不同怎么办?

A: 已自动处理！两个加载器都返回归一化的SOH:
- HUST: `SOH = capacity / 初始容量`
- MIT: `SOH = capacity / 初始容量` (相同逻辑)

### Q2: MIT有些电池cycles很少(36个)怎么办?

A: 有两种策略:
```python
# 策略1: 过滤掉cycles太少的电池
battery_names_filtered = [
    name for name in battery_names
    if len(all_data[name][0]) >= 100  # 至少100 cycles
]

# 策略2: 保留所有，但注意window_size
# 如果window_size=40，最少36-cycle的电池会被自然排除
```

### Q3: 如何在两个数据集上都测试XGBoost?

A: 修改 `train_xgboost_baseline.py`:

```python
# 添加dataset参数
def train_xgboost_baseline(
    model_type='xgboost_simple',
    dataset='hust',  # 新增: 'hust' 或 'mit'
    ...
):
    # 根据dataset加载不同数据
    if dataset == 'hust':
        battery_names, all_data = load_all_batteries(...)
    elif dataset == 'mit':
        from data_loaders import load_all_mit_batteries
        battery_names, all_data = load_all_mit_batteries(...)
```

### Q4: 能否混合两个数据集训练?

A: 可以！合并两个数据集:

```python
# 加载HUST
hust_names, hust_data = load_all_batteries(...)

# 加载MIT
mit_names, mit_data = load_all_mit_batteries(...)

# 合并
all_names = hust_names + mit_names
all_data = {**hust_data, **mit_data}

# 然后正常训练
```

**注意**: 需要处理电池命名冲突（MIT已用batch前缀避免）

---

## 性能对比建议

### 论文实验设计

```python
# 实验组1: HUST数据集
# - XGBoost_Simple: X.XX%
# - XGBoost_Enhanced: X.XX%
# - CNN-LSTM: X.XX%
# - CNN-LSTM + Physics: X.XX%

# 实验组2: MIT数据集
# - XGBoost_Simple: X.XX%
# - XGBoost_Enhanced: X.XX%
# - CNN-LSTM: X.XX%
# - CNN-LSTM + Physics: X.XX%

# 对比结论:
# 1. 方法在不同数据集上的稳定性
# 2. 数据集特性对性能的影响
# 3. 物理约束在不同数据分布下的增益
```

---

## 总结

✅ **已实现**:
- MIT数据加载器
- MIT训练脚本
- 完全兼容现有模型和物理约束

✅ **迁移步骤** (3步):
1. 确保MIT数据在 `data/MIT data/`
2. 运行 `python train_mit.py`
3. 查看结果 `results/mit_cnn_lstm/`

✅ **零修改迁移**:
- 模型架构不变
- 物理约束不变
- 训练流程不变
- 特征维度相同

🚀 **扩展性**: 未来添加新数据集，只需:
1. 创建对应的 `data_loader_xxx.py`
2. 实现 `load_all_xxx_batteries()` 函数
3. 返回相同格式: `(features, soh, capacity)`
