# 全HUST电芯容量衰减曲线绘图指南

## 概述

`plot_all_hust_batteries.py` 脚本用于绘制所有HUST LFP电芯（共77个）的容量衰减曲线。该脚本提供了两种可视化模式：

1. **简洁模式**：所有77个电芯的容量曲线叠放在一张图中
2. **详细统计模式**：包含容量曲线和统计分布直方图的四子图模式

## 快速开始

### 基础用法

```bash
# 绘制所有电芯，简洁模式
python plot_all_hust_batteries.py

# 不显示图表窗口，仅保存文件
python plot_all_hust_batteries.py --no-show

# 高分辨率输出（300 DPI）
python plot_all_hust_batteries.py --dpi 300 --no-show
```

### 详细统计模式

```bash
# 生成包含统计分布的四子图
python plot_all_hust_batteries.py --include-stats --no-show

# 结合其他选项
python plot_all_hust_batteries.py --include-stats --dpi 300 --no-show
```

## 详细命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data-dir` | str | `data/HUST data` | 数据目录路径 |
| `--save-dir` | str | `results/capacity_curves` | 图表保存目录 |
| `--save-path` | str | None | 自定义保存路径（覆盖默认路径） |
| `--dpi` | int | 100 | 保存图表的分辨率（DPI） |
| `--colormap` | str | `tab20` | 颜色映射方案 |
| `--figsize` | float | 16 10 | 图表大小（宽 高） |
| `--no-show` | flag | - | 不显示图表窗口 |
| `--include-stats` | flag | - | 生成包含统计信息的详细图表 |
| `--save-stats` | str | None | 保存统计数据到CSV文件 |

## 颜色映射选项

支持的颜色映射：`tab20`、`tab20c`、`tab10`、`hsv`、`jet`

```bash
# 使用不同的颜色映射
python plot_all_hust_batteries.py --colormap tab20c --no-show
python plot_all_hust_batteries.py --colormap hsv --no-show
```

## 自定义图表大小

```bash
# 更大的图表（20x12 英寸）
python plot_all_hust_batteries.py --figsize 20 12 --no-show

# 更小的图表（12x8 英寸）
python plot_all_hust_batteries.py --figsize 12 8 --no-show
```

## 导出统计数据

```bash
# 将统计数据保存到CSV文件
python plot_all_hust_batteries.py --save-stats results/battery_stats.csv --no-show
```

## 输出文件

脚本生成以下输出文件：

### 简洁模式
- **all_batteries_overlay.png** - 所有77个电芯的容量曲线（单个图表）
- 文件大小：约 746 KB（100 DPI）

### 详细统计模式
- **all_batteries_overlay.png** - 简洁模式图表
- **all_batteries_with_stats.png** - 包含统计分布的四子图
  - 上方：所有电芯的容量曲线
  - 左下：初始容量分布直方图
  - 右下：容量损失百分比分布直方图

## 生成的统计汇总

脚本会在控制台输出以下统计信息：

```
Metric               Min             Max             Mean            Std Dev
--------------------------------------------------------------------------------
Initial Cap (Ah)     1.163193        1.231382        1.189159        0.015267
Final Cap (Ah)       0.879694        0.903331        0.881548        0.003956
Capacity Loss (Ah)   0.267540        0.351036        0.307611        0.016419
Loss Percent (%)     22.85           28.51           25.86           1.06
Cycle Count          1123            2672            1874.9          384.3
```

## 数据特性

### HUST数据集概览
- **总电芯数**：77个
- **成功加载率**：100%（77/77）

### 容量衰减统计
- **初始容量**：1.16～1.23 Ah，平均 1.19 Ah
- **最终容量**：0.88～0.90 Ah，平均 0.88 Ah
- **容量损失**：22.85～28.51%，平均 25.86%
- **循环次数**：1,123～2,672 次，平均 1,875 次

## 实际使用示例

### 场景1：快速生成简洁图表
```bash
python plot_all_hust_batteries.py --no-show
# 输出：all_batteries_overlay.png
```

### 场景2：生成出版质量图表（高分辨率）
```bash
python plot_all_hust_batteries.py --dpi 300 --figsize 18 12 --no-show
# 输出：高质量PNG文件
```

### 场景3：完整分析（包含统计）
```bash
python plot_all_hust_batteries.py --include-stats --dpi 300 --save-stats results/stats.csv --no-show
# 输出：
#   - all_batteries_overlay.png
#   - all_batteries_with_stats.png
#   - results/stats.csv
```

### 场景4：自定义保存路径
```bash
python plot_all_hust_batteries.py --save-path my_output/hust_overview.png --no-show
# 输出：my_output/hust_overview.png
```

## 主要功能

### 1. 自动电芯扫描
- 自动从数据目录扫描并识别所有CSV文件
- 支持自定义数据目录路径

### 2. 多颜色支持
- 自动为每个电芯分配唯一颜色
- 循环使用颜色映射（超过20个电芯时）
- 支持多种预设颜色映射

### 3. 统计信息
- 计算每个电芯的初始容量、最终容量、容量损失
- 提供全局统计汇总（最小值、最大值、平均值、标准差）
- 可选输出为CSV格式

### 4. 图表注解
- 图表标题和轴标签
- 统计信息文本框（显示总电芯数、平均损失、循环范围）
- 网格线辅助
- 多列图例避免重叠

## 故障排除

### 问题1：找不到数据文件
```bash
# 确保数据目录存在，或指定正确路径
python plot_all_hust_batteries.py --data-dir "path/to/your/data"
```

### 问题2：图表窗口无法显示
```bash
# 使用 --no-show 选项，只保存文件
python plot_all_hust_batteries.py --no-show
```

### 问题3：图表过于拥挤
```bash
# 增加图表大小
python plot_all_hust_batteries.py --figsize 20 14 --no-show
```

## 性能信息

- **处理时间**：< 10 秒（77个电芯）
- **内存占用**：< 500 MB
- **输出文件大小**：
  - 简洁模式：约 746 KB（100 DPI）
  - 详细统计：约 421 KB（100 DPI）

## 代码结构

### 主要函数

#### `get_all_hust_batteries(data_dir)`
自动扫描数据目录，返回排序后的电芯名称列表。

#### `plot_all_batteries_overlay(batteries, data_dir, figsize, colormap)`
绘制所有电芯的容量曲线叠放图。
- **返回值**：(fig, ax, stats_dict)

#### `plot_all_batteries_with_statistics(batteries, data_dir, figsize)`
绘制包含统计分布的四子图。
- **返回值**：(fig, stats_dict)

#### `print_summary_statistics(stats_dict)`
输出统计汇总表到控制台。

## 相关脚本

- **plot_hust_capacity_curves.py** - 绘制单个或多个电芯的容量曲线
- **plot_capacity_curves.py** - 绘制IC数据集的容量曲线
- **main_hust_baseline.py** - HUST数据集的基线模型训练脚本

## 更新日志

### v1.0 (2025-11-12)
- 初始版本
- 支持两种可视化模式（简洁和详细统计）
- 完整的命令行参数支持
- 自动统计汇总功能
- CSV导出功能

## 许可和致谢

此脚本用于HUST电池数据集的可视化分析。

---

**最后更新**：2025-11-12
