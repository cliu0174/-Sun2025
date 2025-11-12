# 快速参考：全HUST电芯绘图脚本

## 脚本功能
绘制所有77个HUST电芯的容量衰减曲线，支持两种可视化模式。

## 最常用命令

```bash
# 1. 生成基础图表（推荐）
python plot_all_hust_batteries.py --no-show

# 2. 生成高分辨率图表（出版用）
python plot_all_hust_batteries.py --dpi 300 --no-show

# 3. 生成详细统计图表
python plot_all_hust_batteries.py --include-stats --dpi 300 --no-show

# 4. 导出统计数据为CSV
python plot_all_hust_batteries.py --save-stats stats.csv --no-show
```

## 输出文件位置
- `results/capacity_curves/all_batteries_overlay.png` - 简洁模式
- `results/capacity_curves/all_batteries_with_stats.png` - 详细统计模式

## 关键统计数据
| 指标 | 最小值 | 最大值 | 平均值 | 标准差 |
|------|-------|-------|-------|-------|
| 初始容量(Ah) | 1.163 | 1.231 | 1.189 | 0.015 |
| 最终容量(Ah) | 0.880 | 0.903 | 0.882 | 0.004 |
| 损失百分比(%) | 22.85 | 28.51 | 25.86 | 1.06 |
| 循环次数 | 1123 | 2672 | 1875 | 384 |

## 参数速查表

| 参数 | 效果 | 示例 |
|------|------|------|
| `--dpi 300` | 提高输出分辨率 | 用于出版 |
| `--figsize 20 12` | 自定义图表大小 | 更大视图 |
| `--colormap tab20c` | 更多颜色 | 77个电芯更易区分 |
| `--include-stats` | 添加统计直方图 | 分布分析 |
| `--save-stats file.csv` | 导出数据 | 进一步分析 |
| `--no-show` | 不弹出窗口 | 批量处理 |

## 图表说明

### 简洁模式图表
```
图表内容：
├─ 所有77个电芯的容量曲线
├─ 不同颜色区分各电芯
└─ 右上角：统计信息框
```

### 详细统计模式图表
```
四子图布局：
├─ 上方(全宽)：所有电芯容量曲线
├─ 左下：初始容量分布直方图
│         └─ 红色虚线：平均值
├─ 右下：损失百分比分布直方图
│         └─ 深红虚线：平均值
```

## 常见使用场景

### 场景A：数据探索
```bash
python plot_all_hust_batteries.py
# 快速查看所有电芯衰减趋势
```

### 场景B：论文/报告
```bash
python plot_all_hust_batteries.py --dpi 300 --figsize 18 12 --no-show
# 高质量出版图表
```

### 场景C：完整分析
```bash
python plot_all_hust_batteries.py --include-stats --dpi 300 --save-stats stats.csv --no-show
# 图表+统计+数据导出
```

### 场景D：自定义路径
```bash
python plot_all_hust_batteries.py --save-path my_output.png --no-show
# 保存到指定位置
```

## 性能
- 处理时间：< 10 秒
- 内存占用：< 500 MB
- 文件大小：746 KB（100 DPI）/ 421 KB（统计版）

## 故障排除

| 问题 | 解决方案 |
|------|--------|
| 图表显示窗口冻结 | 添加 `--no-show` |
| 找不到数据 | 检查 `data/HUST data/` 目录 |
| 图表太拥挤 | 用 `--figsize 20 14` 增大 |
| 颜色难以区分 | 用 `--colormap tab20c` |

## 数据统计信息

### 数据集特性
- **总电芯数**：77
- **均成功加载**：77/77 (100%)

### 容量衰减特性
- 所有电芯都显示线性递减趋势
- 损失百分比相对集中（22.85%-28.51%）
- 循环寿命存在较大差异（1123-2672次）

## 配套脚本
- `plot_hust_capacity_curves.py` - 单/多电芯绘图
- `main_hust_baseline.py` - 基线模型训练
- `data_analysis/analyze_hust_features.py` - 特征分析

---
**快速链接**
- 完整文档：`notes/ALL_BATTERIES_PLOTTING.md`
- 脚本位置：`plot_all_hust_batteries.py`
- 输出目录：`results/capacity_curves/`

