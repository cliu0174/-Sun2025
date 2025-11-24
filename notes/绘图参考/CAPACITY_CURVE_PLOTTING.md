# 绘制容量衰减曲线 - 使用指南

## 📌 概述

这个脚本可以绘制HUST数据集中电池容量随循环次数变化的曲线，支持单个电池或多个电池的对比分析。

**脚本位置**: `plot_hust_capacity_curves.py`

---

## 🚀 快速开始

### 1. 绘制单个电池的容量曲线

```bash
# 绘制电池 1-1
python plot_hust_capacity_curves.py --battery 1-1

# 绘制电池 2-5
python plot_hust_capacity_curves.py --battery 2-5

# 绘制电池 10-3
python plot_hust_capacity_curves.py --battery 10-3
```

**输出**:
- 一个PNG图表（包含容量曲线和统计信息）
- 保存位置: `results/capacity_curves/1_1_capacity_curve.png`

---

### 2. 绘制多个电池的对比曲线

```bash
# 绘制多个电池
python plot_hust_capacity_curves.py --batteries 1-1 1-2 1-3 1-4 1-5

# 绘制B电池组所有电池
python plot_hust_capacity_curves.py --battery-group B

# 绘制第1组所有电池
python plot_hust_capacity_curves.py --battery-group 1
```

**输出**:
- `capacity_comparison.png` - 所有电池的容量曲线对比
- `capacity_statistics.png` - 4个统计柱状图

---

### 3. 自定义保存位置

```bash
# 保存到自定义路径
python plot_hust_capacity_curves.py --battery 1-1 \
    --save-path my_results/battery_1_1.png

# 保存到自定义目录
python plot_hust_capacity_curves.py --batteries 1-1 1-2 \
    --save-dir my_plots/
```

---

### 4. 高级选项

```bash
# 不显示图表，只保存
python plot_hust_capacity_curves.py --battery 1-1 --no-show

# 提高图表分辨率 (默认150 DPI)
python plot_hust_capacity_curves.py --battery 1-1 --dpi 300

# 自定义数据目录
python plot_hust_capacity_curves.py --battery 1-1 \
    --data-dir data/my_hust_data/
```

---

## 📊 输出示例

### 单个电池模式

生成一张图表，包含：
- ✅ 容量随循环次数的衰减曲线
- ✅ 初始容量
- ✅ 最终容量
- ✅ 总容量损失（绝对值和百分比）
- ✅ 总循环数
- ✅ 网格线和图例

**示例统计信息**:
```
电池 1-1 容量统计:
  初始容量: 1.168928 Ah
  最终容量: 1.061500 Ah
  容量损失: 0.107428 Ah (9.19%)
  总循环数: 1130
```

---

### 多电池对比模式

生成两张图表：

#### 1️⃣ capacity_comparison.png - 容量曲线对比
显示所有电池的容量曲线在同一坐标系中：
- 不同颜色代表不同电池
- 清晰看出容量衰减速率差异

#### 2️⃣ capacity_statistics.png - 统计信息（2×2网格）

| 左上 | 右上 |
|------|------|
| **初始容量对比** | **最终容量对比** |
| 柱状图显示各电池的初始容量 | 柱状图显示各电池的最终容量 |

| 左下 | 右下 |
|------|------|
| **容量损失对比（Ah）** | **容量损失百分比** |
| 柱状图显示绝对容量损失 | 柱状图显示百分比损失 |

**示例表格输出**:
```
电池       初始容量       最终容量       损失(Ah)      损失(%)  循环数
----------------------------------------------------------------------
1-1        1.168928       1.061500       0.107428     9.19     1130
1-2        1.169856       1.062300       0.107556     9.19     1130
1-3        1.169752       1.063100       0.106652     9.12     1130
```

---

## 🎨 图表定制

### 修改图表大小

在脚本中修改 `figsize` 参数：

```python
# 单个电池 - 默认 (10, 6)
fig, _ = plot_single_capacity_curve(battery_name, data_path, figsize=(12, 7))

# 多电池对比 - 默认 (14, 8)
fig, ax, data = plot_multiple_capacity_curves(batteries, figsize=(16, 9))
```

### 修改颜色

在脚本中查找并修改颜色设置：

```python
# 单个电池线条颜色（默认蓝色）
ax.plot(cycles, capacity, 'b-', linewidth=2)  # 改为其他颜色: 'r-', 'g-', etc.

# 柱状图颜色（默认绿色）
axes[0, 0].bar(..., color='green', alpha=0.7)  # 改为其他颜色
```

---

## 📈 函数说明

### `plot_single_capacity_curve()`
```python
def plot_single_capacity_curve(battery_name, data_path, figsize=(10, 6))
```
绘制单个电池的容量衰减曲线。

**参数**:
- `battery_name`: 电池名称 (e.g., '1-1')
- `data_path`: CSV文件路径
- `figsize`: 图表大小 (width, height)

**返回**: `(capacity, cycles)` 元组

---

### `plot_multiple_capacity_curves()`
```python
def plot_multiple_capacity_curves(battery_names, data_dir='data/HUST data', figsize=(14, 8))
```
绘制多个电池的容量曲线对比。

**参数**:
- `battery_names`: 电池名称列表
- `data_dir`: 数据目录
- `figsize`: 图表大小

**返回**: `(fig, ax, data_dict)` - 图表、坐标轴和数据字典

---

### `plot_capacity_statistics()`
```python
def plot_capacity_statistics(battery_names, data_dir='data/HUST data', figsize=(12, 6))
```
绘制统计信息（4个子图）。

**参数**:
- `battery_names`: 电池名称列表
- `data_dir`: 数据目录
- `figsize`: 图表大小

**返回**: `(fig, stats)` - 图表和统计字典

---

### `get_battery_list()`
```python
def get_battery_list(battery_group=None, data_dir='data/HUST data')
```
获取电池列表。

**参数**:
- `battery_group`: 电池组前缀 (e.g., '1', 'B') 或 None (所有)
- `data_dir`: 数据目录

**返回**: 排序后的电池名称列表

---

## 💻 Python代码中使用

### 示例1: 绘制单个电池

```python
from plot_hust_capacity_curves import plot_single_capacity_curve

# 绘制电池 1-1
fig, (capacity, cycles) = plot_single_capacity_curve(
    battery_name='1-1',
    data_path='data/HUST data/1-1.csv',
    figsize=(12, 7)
)

# 自定义处理
print(f"Initial capacity: {capacity[0]}")
print(f"Final capacity: {capacity[-1]}")

# 保存
plt.savefig('my_plot.png', dpi=300)
plt.show()
```

### 示例2: 绘制多电池对比

```python
from plot_hust_capacity_curves import plot_multiple_capacity_curves, plot_capacity_statistics

# 绘制对比
fig1, ax, data = plot_multiple_capacity_curves(
    battery_names=['1-1', '1-2', '1-3', '1-4'],
    data_dir='data/HUST data'
)

# 绘制统计
fig2, stats = plot_capacity_statistics(
    battery_names=['1-1', '1-2', '1-3', '1-4'],
    data_dir='data/HUST data'
)

plt.show()
```

### 示例3: 提取数据用于后续分析

```python
from plot_hust_capacity_curves import get_battery_list, plot_multiple_capacity_curves

# 获取第1组所有电池
batteries = get_battery_list(battery_group='1')

# 绘制对比
fig, ax, data_dict = plot_multiple_capacity_curves(batteries)

# 提取和处理数据
for battery_name, data in data_dict.items():
    print(f"{battery_name}:")
    print(f"  Capacity loss: {data['loss']:.6f} Ah")
    print(f"  Total cycles: {len(data['cycles'])}")
```

---

## 🔍 数据格式

HUST数据集的CSV格式：

```
voltage mean, voltage std, ... , capacity
3.455707,     0.044239,    ... , 1.168928
3.456173,     0.044987,    ... , 1.169865
...
```

- **第一列 ~ 第16列**: 特征（电压、电流统计量等）
- **第17列**: `capacity` - 电池容量（Ah）
- **每一行**: 一个循环的数据

---

## 📁 输出文件结构

```
results/
└── capacity_curves/           # 默认保存目录
    ├── 1_1_capacity_curve.png        # 单个电池曲线
    ├── 1_2_capacity_curve.png
    ├── capacity_comparison.png       # 多电池对比
    └── capacity_statistics.png       # 统计信息
```

---

## 🎯 常见用法

### 场景1: 分析某个电池的容量衰减

```bash
python plot_hust_capacity_curves.py --battery 1-1 --no-show
# 保存到 results/capacity_curves/1_1_capacity_curve.png
```

### 场景2: 对比同一组电池的容量差异

```bash
python plot_hust_capacity_curves.py --battery-group 1
# 生成 capacity_comparison.png 和 capacity_statistics.png
```

### 场景3: 对比B组全部电池

```bash
python plot_hust_capacity_curves.py --battery-group B --dpi 300
# 高分辨率输出
```

### 场景4: 制作论文图表

```bash
python plot_hust_capacity_curves.py --battery 1-1 \
    --save-path papers/figures/capacity_degradation.png \
    --dpi 300 --no-show
```

---

## ⚙️ 命令行参数完整列表

| 参数 | 说明 | 示例 |
|------|------|------|
| `--battery` | 单个电池 | `--battery 1-1` |
| `--batteries` | 多个电池 | `--batteries 1-1 1-2 1-3` |
| `--battery-group` | 电池组 | `--battery-group 1` |
| `--data-dir` | 数据目录 | `--data-dir data/HUST\ data` |
| `--save-dir` | 保存目录 | `--save-dir my_results/` |
| `--save-path` | 自定义保存路径 | `--save-path plot.png` |
| `--no-show` | 不显示图表 | `--no-show` |
| `--dpi` | 图表分辨率 | `--dpi 300` |

---

## 🐛 故障排除

### 问题: 找不到数据文件
```
Error: Data file not found: data/HUST data/1-1.csv
```
**解决**: 检查数据目录路径，使用 `--data-dir` 指定正确路径

### 问题: 电池名称格式错误
```
Warning: Data file not found for battery 1_1
```
**解决**: 电池名称应使用 `-` 分隔符，如 `1-1` 而非 `1_1`

### 问题: 图表显示不出来
**解决**: 使用 `--no-show` 只保存不显示，或检查Matplotlib后端设置

---

## 📊 数据统计示例

对于电池1-1的完整统计：
```
电池: 1-1
  初始容量: 1.168928 Ah
  最终容量: 1.061500 Ah
  容量损失: 0.107428 Ah
  损失百分比: 9.19%
  总循环数: 1130
```

这表示该电池在1130个循环后，容量衰减了约9.19%。

