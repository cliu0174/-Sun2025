# 项目文件结构整理说明

## 📁 新的项目结构

### 核心目录

```
1111-soh/
├── data_loaders/               # ✨ 新建 - 数据加载模块
│   ├── __init__.py
│   ├── data_loader.py          # 通用数据加载
│   ├── data_loader_hust.py     # HUST数据集加载
│   └── data_loader_per_battery.py  # 按电池加载
│
├── models/                     # ✨ 新建 - 模型和训练器
│   ├── __init__.py
│   ├── baseline_models.py      # FNN, CNN, LSTM 模型定义
│   ├── model.py               # BPINN 模型
│   └── model_trainer.py       # 统一训练器 + ConfigLoader + ModelFactory
│
├── data_analysis/             # ✨ 新建 - 数据分析脚本
│   ├── __init__.py
│   ├── analyze_hust_features.py    # 特征相关性分析
│   ├── plot_comparison_results.py  # 对比结果可视化
│   └── plot_per_battery_results.py # 按电池结果可视化
│
├── src/                       # 保留 - 工具和通用函数
│   ├── utils.py              # 绘图、输出、目录管理等
│   ├── train.py              # 训练函数
│   ├── evaluate.py           # 评估函数
│   └── feature_selector.py   # 特征选择模块
│
├── configs/                   # 配置文件
│   └── models/
│       ├── fnn_config.json
│       ├── cnn_config.json
│       └── lstm_config.json
│
├── data/                      # 数据文件
│   ├── B05_IC.csv
│   ├── B06_IC.csv
│   ├── B07_IC.csv
│   └── HUST data/           # HUST电池数据集
│
├── results/                   # 训练结果
│
├── notes/                     # 文档和笔记
│
├── test/                      # 测试文件
│
└── main_*.py                  # 主训练脚本
    ├── main.py               # BPINN训练
    ├── main_per_battery.py   # BPINN按电池训练
    ├── main_comparison.py    # FNN/CNN/LSTM对比（Per-Battery）
    └── main_hust_baseline.py # ⭐ FNN/CNN/LSTM统一训练（HUST数据集）
```

## 🎯 变更内容

### 1. 创建的新文件夹

| 文件夹 | 用途 | 包含的文件 |
|------|------|---------|
| `data_loaders/` | 数据加载模块 | 3个数据加载脚本 + `__init__.py` |
| `models/` | 模型和训练 | 3个模型文件 + `__init__.py` |
| `data_analysis/` | 数据分析可视化 | 3个分析脚本 + `__init__.py` |

### 2. 移动的文件

```
src/data_loader.py           → data_loaders/data_loader.py
src/data_loader_hust.py      → data_loaders/data_loader_hust.py
src/data_loader_per_battery.py → data_loaders/data_loader_per_battery.py

src/baseline_models.py       → models/baseline_models.py
src/model.py                 → models/model.py
src/model_trainer.py         → models/model_trainer.py

analyze_hust_features.py     → data_analysis/analyze_hust_features.py
plot_comparison_results.py   → data_analysis/plot_comparison_results.py
plot_per_battery_results.py  → data_analysis/plot_per_battery_results.py
```

### 3. 保留在 `src/` 的文件

```
src/
├── utils.py               # 通用工具函数（保留）
├── train.py               # 通用训练函数（保留）
├── evaluate.py            # 评估函数（保留）
└── feature_selector.py    # 特征选择模块（保留）
```

## 📝 导入更新

### 旧导入方式
```python
from src.data_loader_hust import load_single_hust_battery
from src.baseline_models import FNN, CNN, LSTM
from src.model_trainer import ConfigLoader, ModelTrainer
```

### 新导入方式
```python
from data_loaders import load_single_hust_battery
from models import FNN, CNN, LSTM, ConfigLoader, ModelTrainer
```

### 受影响的脚本

已更新的脚本：
- ✅ `main.py`
- ✅ `main_per_battery.py`
- ✅ `main_comparison.py`
- ✅ `main_hust_comparison.py`
- ✅ `main_hust_baseline.py`
- ✅ `data_analysis/analyze_hust_features.py`
- ✅ `models/model_trainer.py`

## 🔧 编码兼容性修复

同步修复了Windows GBK编码不兼容的问题：
- ✅ 替换了所有特殊Unicode字符（✓、❌等）为ASCII字符
- ✅ 将中文打印语句改为英文
- ✅ 确保在Windows PowerShell上正常运行

## ✨ 功能验证

训练脚本测试结果：
```bash
python main_hust_baseline.py --model FNN --battery 1-1 --epochs 10 --no-cuda
```

✅ 特征选择：成功选择前6个最相关特征
✅ 训练运行：成功完成10个epoch
✅ 模型保存：成功保存到 `results/results_hust/1-1/fnn/`
✅ 结果汇总：成功显示训练汇总信息

## 📚 使用示例

### 训练基准模型

```bash
# 训练FNN
python main_hust_baseline.py --model FNN --battery 1-1 --epochs 200

# 训练CNN
python main_hust_baseline.py --model CNN --battery 1-1 --epochs 200

# 训练LSTM
python main_hust_baseline.py --model LSTM --battery 1-1 --epochs 200

# 多电池训练
python main_hust_baseline.py --model FNN --batteries 1-1 1-2 1-3 --epochs 200
```

### 数据分析

```bash
# 特征相关性分析
python data_analysis/analyze_hust_features.py
```

## 📊 项目优势

- **模块化**：清晰的关注点分离
- **易维护**：相关功能归类到专门的文件夹
- **易扩展**：新的加载器、模型、分析脚本可轻松添加
- **专业**：符合Python项目标准结构
