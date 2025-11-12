# 容量衰减曲线绘制 - 快速参考

## 📌 脚本位置
```
plot_hust_capacity_curves.py
```

---

## 🚀 最常用的命令

### 1️⃣ 绘制单个电池
```bash
python plot_hust_capacity_curves.py --battery 1-1
```
**输出**: `results/capacity_curves/1_1_capacity_curve.png`

### 2️⃣ 绘制多个电池对比
```bash
python plot_hust_capacity_curves.py --batteries 1-1 1-2 1-3
```
**输出**: 
- `capacity_comparison.png` - 所有电池曲线对比
- `capacity_statistics.png` - 统计信息（4个图）

### 3️⃣ 绘制整个电池组
```bash
python plot_hust_capacity_curves.py --battery-group 1
```
**说明**: 绘制第1组的所有电池（1-1到1-8）

### 4️⃣ 高分辨率保存（论文用）
```bash
python plot_hust_capacity_curves.py --battery 1-1 --dpi 300 --no-show
```

---

## 📊 输出内容

### 单个电池
✅ 容量随循环次数的衰减曲线
✅ 初始容量、最终容量、容量损失、循环数（显示在图表上）
✅ PNG图表

### 多个电池
✅ 所有电池的容量曲线对比（超叠显示）
✅ 4个统计柱状图（初始、最终、损失Ah、损失%）
✅ 数据统计表格（打印输出）

---

## 🎯 常见使用场景

| 场景 | 命令 |
|------|------|
| 查看电池1-1的容量衰减 | `python plot_hust_capacity_curves.py --battery 1-1` |
| 对比1-1, 1-2, 1-3三个电池 | `python plot_hust_capacity_curves.py --batteries 1-1 1-2 1-3` |
| 查看B组所有电池的对比 | `python plot_hust_capacity_curves.py --battery-group B` |
| 保存为高分辨率图表 | `python plot_hust_capacity_curves.py --battery 1-1 --dpi 300` |
| 自定义保存位置 | `python plot_hust_capacity_curves.py --battery 1-1 --save-path my_plot.png` |

---

## 💡 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--battery <name>` | 单个电池 | `--battery 1-1` |
| `--batteries <names...>` | 多个电池 | `--batteries 1-1 1-2 1-3` |
| `--battery-group <group>` | 电池组（前缀） | `--battery-group 1` |
| `--save-dir <path>` | 保存目录 | `--save-dir my_results/` |
| `--save-path <path>` | 保存文件路径 | `--save-path plot.png` |
| `--dpi <number>` | 分辨率 | `--dpi 300` |
| `--no-show` | 不显示图表 | `--no-show` |
| `--data-dir <path>` | 数据目录 | `--data-dir data/HUST\ data` |

---

## 📈 在Python中使用

```python
from plot_hust_capacity_curves import plot_single_capacity_curve

# 绘制单个电池
capacity, cycles = plot_single_capacity_curve('1-1', 'data/HUST data/1-1.csv')

# 获取数据
print(f"初始容量: {capacity[0]}")
print(f"最终容量: {capacity[-1]}")
print(f"总循环数: {len(capacity)}")

# 保存图表
import matplotlib.pyplot as plt
plt.savefig('my_plot.png', dpi=300)
```

---

## 📁 输出文件

所有图表默认保存到: `results/capacity_curves/`

```
results/capacity_curves/
├── 1_1_capacity_curve.png      # 单个电池
├── 1_2_capacity_curve.png
├── capacity_comparison.png     # 多电池对比
└── capacity_statistics.png     # 统计信息
```

---

## ✅ 脚本已验证

```
电池 1-1 容量统计:
  初始容量: 1.169526 Ah
  最终容量: 0.880193 Ah
  容量损失: 0.289334 Ah (24.74%)
  总循环数: 1487

✓ 图表已成功生成
```

---

## 📌 电池编号范围

HUST数据集包含以下电池：

| 组别 | 电池 | 范围 |
|------|------|------|
| 第1组 | 1-1到1-8 | 8个 |
| 第2组 | 2-2到2-8 | 7个（无2-1） |
| 第3组 | 3-1到3-8 | 8个 |
| ... | ... | ... |
| 第10组 | 10-1到10-7 | 7个（无10-8） |

**总数**: 77个电池

