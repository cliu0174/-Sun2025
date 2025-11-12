# HUST电池SOH估计项目 - 精简版目录结构

## 📁 推荐的项目结构（仅HUST相关）

```
1111-soh/
│
├── 📊 data/                          # 数据目录
│   └── HUST data/                    # HUST数据集
│       ├── 1-1.csv                   # 电池1-1数据
│       ├── 1-2.csv
│       ├── ...
│       └── 10-8.csv                  # 共77组电池
│
├── ⚙️ configs/                       # 配置文件
│   └── models/                       # 模型配置
│       ├── fnn_config.json           # FNN配置
│       ├── cnn_config.json           # CNN配置
│       ├── lstm_config.json          # LSTM配置
│       └── bpinn_config.json         # BPINN配置
│
├── 🏭 models/                        # 模型定义
│   ├── __init__.py                   # 导出接口
│   ├── baseline_models.py            # FNN/CNN/LSTM定义
│   ├── model.py                      # BPINN模型定义
│   ├── model_factory.py              # ⭐ 统一模型工厂（核心）
│   └── model_trainer.py              # 旧版训练器（向后兼容）
│
├── 📥 data_loaders/                  # 数据加载器
│   ├── __init__.py
│   ├── data_loader_hust.py           # ⭐ HUST数据加载器（主要使用）
│   └── data_loader.py                # NASA数据加载器（可删除）
│
├── 🔧 src/                           # 工具函数
│   ├── feature_selector.py           # 特征选择
│   ├── train.py                      # 训练函数（BPINN用）
│   ├── evaluate.py                   # 评估函数
│   └── utils.py                      # 工具函数
│
├── 📊 data_analysis/                 # 数据分析脚本
│   ├── __init__.py
│   ├── analyze_hust_features.py      # 特征分析
│   ├── plot_comparison_results.py    # 对比图表
│   └── plot_per_battery_results.py   # 单电池结果图
│
├── 🚀 训练脚本（核心）
│   ├── train_single_model.py         # ⭐ 单电池训练
│   ├── train_cross_battery.py        # ⭐ 跨电池训练（6:2:2）
│   ├── train_baseline_comparison.py  # ⭐ 基准模型对比
│   ├── train_comparison.py           # 批量对比训练
│   └── compare_training_modes.py     # 训练模式对比
│
├── 📈 可视化脚本
│   ├── plot_hust_capacity_curves.py  # 容量曲线绘制
│   └── read_pkl.py                   # 读取结果工具
│
├── 🧪 示例和测试
│   └── example_model_factory_usage.py # 模型工厂使用示例
│
├── 📖 文档
│   ├── README_MODEL_FACTORY.md       # ⭐ 系统总结
│   ├── TRAINING_GUIDE.md             # ⭐ 训练指南
│   ├── CROSS_BATTERY_GUIDE.md        # ⭐ 跨电池训练指南
│   ├── QUICK_TEST.md                 # 快速测试指南
│   │
│   └── notes/                        # 详细文档
│       ├── MODEL_FACTORY_GUIDE.md    # API完整文档
│       ├── MODEL_FACTORY_QUICK_START.md # 快速入门
│       ├── PROJECT_STRUCTURE.md       # 项目结构说明
│       ├── FEATURE_SELECTION_SUMMARY.md # 特征选择总结
│       └── ...                        # 其他研究笔记
│
├── 📊 results/                       # 实验结果
│   ├── results_hust/                 # 单电池训练结果
│   │   └── 1-1/                      # 电池1-1
│   │       ├── fnn/                  # FNN结果
│   │       ├── cnn/
│   │       ├── lstm/
│   │       └── bpinn/
│   │
│   ├── cross_battery/                # ⭐ 跨电池训练结果
│   │   ├── fnn/
│   │   ├── cnn/
│   │   ├── lstm/
│   │   └── bpinn/
│   │
│   ├── baseline_comparison/          # ⭐ 基准模型对比结果
│   │   ├── baseline_comparison.csv
│   │   ├── metrics_comparison.png
│   │   └── BASELINE_COMPARISON_REPORT.md
│   │
│   ├── feature_analysis/             # 特征分析结果
│   │   ├── 1-1_correlation_heatmap.png
│   │   └── ...
│   │
│   └── capacity_curves/              # 容量曲线图
│       └── ...
│
├── 🗑️ 可删除的文件（NASA相关）
│   ├── main.py                       # NASA BPINN训练（可删除）
│   ├── main_comparison.py            # NASA对比（可删除）
│   ├── main_per_battery.py           # NASA单电池（可删除）
│   ├── data_loaders/data_loader.py   # NASA数据加载器（可删除）
│   └── data/B05_IC.csv, B06_IC.csv, B07_IC.csv  # NASA数据（可删除）
│
├── 📄 配置文件
│   ├── .gitignore
│   ├── requirements.txt              # Python依赖
│   └── README.md                     # 项目说明
│
└── 🔧 其他
    ├── .claude/                      # Claude配置（可选）
    └── test/                         # 测试文件（可选）
```

---

## 🎯 **最小核心文件（必需）**

如果只保留最核心的HUST相关代码：

```
1111-soh/
├── data/HUST data/          # 77组电池数据
├── configs/models/          # 4个模型配置JSON
├── models/                  # 模型定义
│   ├── __init__.py
│   ├── baseline_models.py
│   ├── model.py
│   └── model_factory.py     # ⭐ 核心
├── data_loaders/            # 数据加载
│   ├── __init__.py
│   └── data_loader_hust.py  # ⭐ 核心
├── src/                     # 工具函数
│   ├── feature_selector.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── train_single_model.py         # ⭐ 单电池训练
├── train_cross_battery.py        # ⭐ 跨电池训练
├── train_baseline_comparison.py  # ⭐ 基准对比
├── TRAINING_GUIDE.md             # ⭐ 使用指南
└── requirements.txt
```

---

## 🗑️ **可以删除的文件列表**

### NASA相关文件
```bash
# 主训练脚本（NASA）
main.py
main_comparison.py
main_hust_baseline.py
main_hust_comparison.py
main_per_battery.py

# NASA数据加载器
data_loaders/data_loader.py
data_loaders/data_loader_per_battery.py

# NASA数据文件
data/B05_IC.csv
data/B06_IC.csv
data/B07_IC.csv

# NASA相关结果
results/NASA_BPINN/
results/results_per_battery/B05_results.pkl
```

### 其他可删除文件
```bash
# 旧的虚拟环境（如果不用）
bpinn_env/

# 旧的分析脚本（已重构到data_analysis/）
analyze_hust_features.py
plot_comparison_results.py
plot_per_battery_results.py

# 旧的文档（已移到notes/）
ANALYSIS_CONCLUSION.md
FEATURE_SELECTION_QUICK_GUIDE.md
FEATURE_SELECTION_SUMMARY.md
MODEL_LIBRARY_README.md
```

---

## 📋 **删除NASA文件的命令**

如果你想清理项目，可以运行：

```bash
# 进入项目目录
cd d:\Projects\1111-soh

# 删除NASA主训练脚本
rm main.py main_comparison.py main_hust_baseline.py main_hust_comparison.py main_per_battery.py

# 删除NASA数据加载器（保留HUST的）
rm data_loaders/data_loader.py data_loaders/data_loader_per_battery.py

# 删除NASA数据文件（如果有）
rm data/B*.csv

# 删除NASA结果（可选）
rm -rf results/NASA_BPINN/

# 删除旧的虚拟环境（可选）
rm -rf bpinn_env/

# 删除已移动的旧文件
rm analyze_hust_features.py plot_comparison_results.py plot_per_battery_results.py
```

---

## ✅ **推荐的最终项目结构（精简版）**

```
1111-soh/                           # 项目根目录
│
├── 📊 data/
│   └── HUST data/                  # 77组电池数据
│
├── ⚙️ configs/models/               # 模型配置（4个JSON）
│
├── 🏭 models/                      # 模型定义
│   ├── baseline_models.py          # FNN/CNN/LSTM
│   ├── model.py                    # BPINN
│   └── model_factory.py            # 统一工厂
│
├── 📥 data_loaders/
│   └── data_loader_hust.py         # HUST数据加载
│
├── 🔧 src/                         # 工具函数
│   ├── feature_selector.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── 🚀 核心训练脚本
│   ├── train_single_model.py       # 单电池训练
│   ├── train_cross_battery.py      # 跨电池训练（6:2:2）
│   └── train_baseline_comparison.py # 基准对比
│
├── 📖 文档
│   ├── README_MODEL_FACTORY.md     # 系统说明
│   ├── TRAINING_GUIDE.md           # 训练指南
│   ├── CROSS_BATTERY_GUIDE.md      # 跨电池指南
│   └── notes/                      # 详细文档
│
└── 📊 results/                     # 实验结果
    ├── results_hust/               # 单电池结果
    ├── cross_battery/              # 跨电池结果
    └── baseline_comparison/        # 对比结果
```

---

## 📝 **文件用途速查表**

| 文件 | 用途 | 是否必需 |
|------|------|---------|
| `train_single_model.py` | 单电池训练 | ⭐ 必需 |
| `train_cross_battery.py` | 跨电池训练（6:2:2） | ⭐ 必需 |
| `train_baseline_comparison.py` | 基准模型对比 | ⭐ 必需 |
| `models/model_factory.py` | 统一模型工厂 | ⭐ 必需 |
| `data_loaders/data_loader_hust.py` | HUST数据加载 | ⭐ 必需 |
| `configs/models/*.json` | 模型配置 | ⭐ 必需 |
| `main.py` | NASA BPINN训练 | ❌ 可删除 |
| `data_loaders/data_loader.py` | NASA数据加载 | ❌ 可删除 |

---

## 🎓 **使用建议**

### 日常使用的文件

```
1. 单电池快速测试
   → train_single_model.py

2. 跨电池泛化评估
   → train_cross_battery.py

3. 基准模型对比实验
   → train_baseline_comparison.py

4. 调整模型参数
   → configs/models/*.json

5. 查看使用方法
   → TRAINING_GUIDE.md
```

---

这个精简结构只保留HUST相关的核心代码，删除了所有NASA相关文件，使项目更加清晰！需要我帮你执行删除操作吗？