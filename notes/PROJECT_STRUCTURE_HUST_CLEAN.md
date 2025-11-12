# HUST电池SOH估计项目 - 清理后的目录结构

## 📁 推荐保留的文件结构

```
1111-soh/
│
├── 📊 data/                           # 数据目录
│   └── HUST data/                     # ⭐ 保留：HUST数据集
│       ├── 1-1.csv
│       ├── 1-2.csv
│       ├── ...
│       └── 10-8.csv                   # 共77组电池数据
│
├── ⚙️ configs/                        # ⭐ 保留：配置文件
│   └── models/
│       ├── fnn_config.json
│       ├── cnn_config.json
│       ├── lstm_config.json
│       └── bpinn_config.json
│
├── 🏭 models/                         # ⭐ 保留：模型定义
│   ├── __init__.py
│   ├── baseline_models.py            # FNN/CNN/LSTM定义
│   ├── model.py                      # BPINN模型
│   └── model_factory.py              # 统一模型工厂（核心）
│
├── 📥 data_loaders/                   # ⭐ 保留：数据加载器
│   ├── __init__.py
│   └── data_loader_hust.py           # HUST数据加载器
│
├── 🔧 src/                            # ⭐ 保留：工具函数
│   ├── __init__.py
│   ├── feature_selector.py           # 特征选择
│   ├── train.py                      # 训练函数（BPINN用）
│   ├── evaluate.py                   # 评估函数
│   └── utils.py                      # 工具函数
│
├── 📊 data_analysis/                  # ⭐ 保留：数据分析脚本
│   ├── __init__.py
│   ├── analyze_hust_features.py
│   ├── plot_comparison_results.py
│   └── plot_per_battery_results.py
│
├── 🚀 核心训练脚本                     # ⭐ 保留
│   ├── train_single_model.py         # 单电池训练
│   ├── train_cross_battery.py        # 跨电池训练（6:2:2）
│   ├── train_baseline_comparison.py  # 基准模型对比
│   ├── train_comparison.py           # 批量对比训练
│   └── compare_training_modes.py     # 训练模式对比
│
├── 📈 可视化和工具脚本                 # ⭐ 保留
│   ├── plot_hust_capacity_curves.py  # 容量曲线绘制
│   ├── read_pkl.py                   # 读取结果工具
│   └── example_model_factory_usage.py # 模型工厂示例
│
├── 📖 核心文档                        # ⭐ 保留
│   ├── README.md                     # 项目说明
│   ├── README_MODEL_FACTORY.md       # 模型工厂总结
│   ├── TRAINING_GUIDE.md             # 训练指南
│   ├── CROSS_BATTERY_GUIDE.md        # 跨电池训练指南
│   ├── QUICK_TEST.md                 # 快速测试指南
│   └── PROJECT_STRUCTURE_HUST_ONLY.md # 项目结构文档
│
├── 📖 notes/                          # ⭐ 保留：详细文档和研究笔记
│   ├── MODEL_FACTORY_GUIDE.md
│   ├── MODEL_FACTORY_QUICK_START.md
│   ├── PROJECT_STRUCTURE.md
│   ├── FEATURE_SELECTION_SUMMARY.md
│   ├── FEATURE_SELECTION_QUICK_GUIDE.md
│   ├── ANALYSIS_CONCLUSION.md
│   ├── BASELINE_COMPARISON_SUMMARY.md
│   ├── BASELINE_DOCUMENTATION_INDEX.md
│   ├── BASELINE_MODEL_REVIEW.md
│   ├── BASELINE_OPTIMIZATION_GUIDE.md
│   ├── BASELINE_REVIEW_FINAL_REPORT.md
│   ├── BASELINE_REVIEW_FINDINGS.md
│   ├── CAPACITY_CURVE_PLOTTING.md
│   ├── CAPACITY_QUICK_START.md
│   ├── MODEL_LIBRARY_README.md
│   ├── QUICK_OPTIMIZATION_CHECKLIST.md
│   └── ...
│
├── 📊 results/                        # ⭐ 保留：实验结果
│   ├── results_hust/                 # 单电池训练结果
│   │   └── 1-1/
│   │       ├── fnn/
│   │       ├── cnn/
│   │       ├── lstm/
│   │       └── bpinn/
│   ├── cross_battery/                # 跨电池训练结果
│   │   ├── fnn/
│   │   ├── cnn/
│   │   ├── lstm/
│   │   └── bpinn/
│   ├── baseline_comparison/          # 基准模型对比结果
│   ├── feature_analysis/             # 特征分析结果
│   └── capacity_curves/              # 容量曲线图
│
├── 🔧 其他配置文件                    # ⭐ 保留
│   ├── .gitignore
│   ├── requirements.txt
│   └── .claude/                      # Claude配置（可选）
│
└── 📝 项目管理文件                    # ⭐ 保留
    ├── PROJECT_STRUCTURE_HUST_ONLY.md
    └── PROJECT_STRUCTURE_HUST_CLEAN.md  # 本文件
```

---

## 🗑️ 需要删除的文件（NASA相关）

### NASA主训练脚本
```bash
main.py                    # NASA BPINN训练
main_comparison.py         # NASA对比
main_hust_baseline.py      # 已被train_baseline_comparison.py替代
main_hust_comparison.py    # 已被train_comparison.py替代
main_per_battery.py        # NASA单电池训练
```

### NASA数据加载器
```bash
data_loaders/data_loader.py              # NASA数据加载器
data_loaders/data_loader_per_battery.py  # NASA单电池数据加载器
```

### NASA数据文件
```bash
data/B05_IC.csv
data/B06_IC.csv
data/B07_IC.csv
# 或任何 data/B*.csv 文件
```

### NASA实验结果
```bash
results/NASA_BPINN/         # NASA相关结果（如果有）
results/results_per_battery/B05_results.pkl  # NASA结果（如果有）
```

### 旧的分析脚本（已移到data_analysis/）
```bash
analyze_hust_features.py    # 已移到 data_analysis/
plot_comparison_results.py  # 已移到 data_analysis/
plot_per_battery_results.py # 已移到 data_analysis/
diagnose_training.py        # 诊断脚本（可删除）
```

### 旧的虚拟环境
```bash
bpinn_env/                  # 旧的虚拟环境目录
```

### 临时文件
```bash
nul                         # 临时文件
__pycache__/               # Python缓存（会自动生成）
*.pyc                      # Python缓存文件
```

---

## 🧹 清理命令（Windows PowerShell）

```powershell
# 进入项目目录
cd d:\Projects\1111-soh

# === 删除NASA主训练脚本 ===
Remove-Item -Path "main.py" -ErrorAction SilentlyContinue
Remove-Item -Path "main_comparison.py" -ErrorAction SilentlyContinue
Remove-Item -Path "main_hust_baseline.py" -ErrorAction SilentlyContinue
Remove-Item -Path "main_hust_comparison.py" -ErrorAction SilentlyContinue
Remove-Item -Path "main_per_battery.py" -ErrorAction SilentlyContinue

# === 删除NASA数据加载器 ===
Remove-Item -Path "data_loaders\data_loader.py" -ErrorAction SilentlyContinue
Remove-Item -Path "data_loaders\data_loader_per_battery.py" -ErrorAction SilentlyContinue

# === 删除NASA数据文件 ===
Remove-Item -Path "data\B*.csv" -ErrorAction SilentlyContinue

# === 删除NASA结果目录 ===
Remove-Item -Path "results\NASA_BPINN" -Recurse -Force -ErrorAction SilentlyContinue

# === 删除旧的分析脚本（已移动） ===
Remove-Item -Path "analyze_hust_features.py" -ErrorAction SilentlyContinue
Remove-Item -Path "plot_comparison_results.py" -ErrorAction SilentlyContinue
Remove-Item -Path "plot_per_battery_results.py" -ErrorAction SilentlyContinue
Remove-Item -Path "diagnose_training.py" -ErrorAction SilentlyContinue

# === 删除旧虚拟环境 ===
Remove-Item -Path "bpinn_env" -Recurse -Force -ErrorAction SilentlyContinue

# === 删除临时文件 ===
Remove-Item -Path "nul" -ErrorAction SilentlyContinue

Write-Host "✅ 清理完成！" -ForegroundColor Green
```

---

## 🧹 清理命令（Linux/Mac Bash）

```bash
#!/bin/bash
# 进入项目目录
cd d:/Projects/1111-soh

# === 删除NASA主训练脚本 ===
rm -f main.py main_comparison.py main_hust_baseline.py main_hust_comparison.py main_per_battery.py

# === 删除NASA数据加载器 ===
rm -f data_loaders/data_loader.py data_loaders/data_loader_per_battery.py

# === 删除NASA数据文件 ===
rm -f data/B*.csv

# === 删除NASA结果目录 ===
rm -rf results/NASA_BPINN/

# === 删除旧的分析脚本 ===
rm -f analyze_hust_features.py plot_comparison_results.py plot_per_battery_results.py diagnose_training.py

# === 删除旧虚拟环境 ===
rm -rf bpinn_env/

# === 删除临时文件 ===
rm -f nul

echo "✅ 清理完成！"
```

---

## 📋 清理后的最小核心文件清单

如果只保留**绝对必需**的HUST核心代码：

```
1111-soh/
├── data/HUST data/                   # 77组电池数据 ✅
├── configs/models/                   # 4个模型配置JSON ✅
├── models/                           # 模型定义 ✅
│   ├── __init__.py
│   ├── baseline_models.py
│   ├── model.py
│   └── model_factory.py             # 核心
├── data_loaders/                     # 数据加载 ✅
│   ├── __init__.py
│   └── data_loader_hust.py
├── src/                              # 工具函数 ✅
│   ├── feature_selector.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── train_single_model.py             # 单电池训练 ✅
├── train_cross_battery.py            # 跨电池训练 ✅
├── train_baseline_comparison.py      # 基准对比 ✅
├── TRAINING_GUIDE.md                 # 使用指南 ✅
├── CROSS_BATTERY_GUIDE.md            # 跨电池指南 ✅
└── requirements.txt                  # 依赖 ✅
```

**这13个组件就是最小可运行系统！**

---

## 📊 文件统计

### 保留文件
- **核心代码**: 约20个Python文件
- **配置文件**: 4个JSON配置
- **文档**: 约15个Markdown文档
- **数据**: 77个CSV文件（HUST数据）
- **结果**: results/目录（实验结果）

### 删除文件
- **NASA训练脚本**: 5个Python文件
- **NASA数据加载器**: 2个Python文件
- **NASA数据**: 3个CSV文件
- **旧分析脚本**: 4个Python文件
- **旧虚拟环境**: bpinn_env/目录

**预计节省空间**: 约100-500MB（主要是虚拟环境）

---

## ✅ 清理后的项目优势

1. **结构清晰** - 只保留HUST相关代码
2. **易于维护** - 减少了50%的文件数量
3. **避免混淆** - 不会误用NASA相关脚本
4. **文档齐全** - 保留所有HUST相关文档
5. **功能完整** - 所有HUST实验功能都可用

---

## 🎯 清理后如何使用

### 单电池训练（快速测试）
```bash
python train_single_model.py
```

### 跨电池训练（评估泛化）
```bash
python train_cross_battery.py
```

### 基准模型对比（研究实验）
```bash
python train_baseline_comparison.py
```

---

## 📝 清理检查清单

清理前，请确认：

- [ ] 已备份重要数据
- [ ] NASA相关结果已归档（如果需要）
- [ ] 确认不再需要NASA相关代码
- [ ] 已阅读清理命令，理解每一步操作

**建议**: 先在Git中创建新分支再清理，方便回退

```bash
git checkout -b cleanup-nasa-files
# 执行清理命令
git add .
git commit -m "Remove NASA-related files, keep only HUST code"
```

---

## 🔄 如果需要恢复

如果误删或需要恢复NASA相关代码：

```bash
# 回退到清理前
git checkout main

# 或恢复特定文件
git checkout main -- main.py
```

---

这个清理后的项目结构专注于HUST数据集研究，更加简洁高效！

需要我帮你执行清理命令吗？
