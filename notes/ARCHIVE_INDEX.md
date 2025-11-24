# Notes 归档索引

## 📋 保留的参考文档

以下文档可能包含有用的分析数据和详细说明，建议保留供参考：

### 🔬 分析报告（推荐保留）
1. **ANALYSIS_CONCLUSION.md** - 综合分析结论
2. **BASELINE_REVIEW_FINAL_REPORT.md** - 基准模型最终报告
3. **FEATURE_SELECTION_ANALYSIS.md** - 特征选择分析

### 📊 对比研究（推荐保留）
4. **BASELINE_COMPARISON_SUMMARY.md** - 基准模型对比总结
5. **DATASET_COMPARISON.md** - 数据集对比

### 🎨 绘图参考（可选保留）
6. **CAPACITY_CURVE_PLOTTING.md** - 容量曲线绘图详解
7. **ALL_BATTERIES_PLOTTING.md** - 所有电池绘图

### 🔧 优化指南（可选保留）
8. **BASELINE_OPTIMIZATION_GUIDE.md** - 详细优化指南
9. **FEATURE_SELECTION_QUICK_GUIDE.md** - 特征选择快速指南

---

## 🗑️ 建议删除的文档

以下文档内容已合并到主目录的 **USAGE_GUIDE.md**，可以安全删除：

### 重复的快速开始
- QUICK_START.md
- QUICKSTART.md
- QUICK_TEST.md
- CAPACITY_QUICK_START.md
- MODEL_FACTORY_QUICK_START.md

### 重复的训练指南
- TRAINING_GUIDE.md
- BASELINE_TRAINING_GUIDE.md
- CROSS_BATTERY_GUIDE.md
- PER_BATTERY_USAGE.md
- USAGE_EXAMPLE.md

### 重复的模型工厂
- MODEL_FACTORY_GUIDE.md
- README_MODEL_FACTORY.md
- MODEL_LIBRARY_README.md

### 重复的项目结构
- PROJECT_STRUCTURE.md
- PROJECT_STRUCTURE_HUST_CLEAN.md
- PROJECT_STRUCTURE_HUST_ONLY.md
- BASELINE_DOCUMENTATION_INDEX.md

### 重复的对比文档
- BASELINE_COMPARISON.md (保留 SUMMARY 版本即可)
- COMPARISON_USAGE.md

### 重复的优化文档
- QUICK_OPTIMIZATION_CHECKLIST.md (保留 GUIDE 版本即可)

### 重复的特征选择
- FEATURE_SELECTION_SUMMARY.md (保留 ANALYSIS 版本即可)

### 重复的绘图
- PLOTTING_QUICK_REF.md

### 重复的评审
- BASELINE_MODEL_REVIEW.md (保留 FINAL_REPORT 即可)
- BASELINE_REVIEW_FINDINGS.md (保留 FINAL_REPORT 即可)

### 其他重复
- IMPLEMENTATION_NOTES.md
- DATA_SPLIT_AND_VISUALIZATION.md
- SOH_CURVES_AND_SECONDARY_TRAINING.md
- HUST_BPINN_ANALYSIS.md

---

## 🎯 精简后的 notes/ 结构

```
notes/
├── README.md                           # 本说明文件
├── ARCHIVE_INDEX.md                    # 归档索引（本文件）
│
├── 分析报告/
│   ├── ANALYSIS_CONCLUSION.md
│   ├── BASELINE_REVIEW_FINAL_REPORT.md
│   └── FEATURE_SELECTION_ANALYSIS.md
│
├── 对比研究/
│   ├── BASELINE_COMPARISON_SUMMARY.md
│   └── DATASET_COMPARISON.md
│
├── 绘图参考/
│   ├── CAPACITY_CURVE_PLOTTING.md
│   └── ALL_BATTERIES_PLOTTING.md
│
└── 优化指南/
    ├── BASELINE_OPTIMIZATION_GUIDE.md
    └── FEATURE_SELECTION_QUICK_GUIDE.md
```

**总计**: 从 37 个文档精简到 11 个核心文档

---

## 📝 一键清理命令

如果你确定要删除重复文档，可以运行：

```bash
# 进入 notes 目录
cd notes/

# 删除重复的快速开始文档
rm -f QUICK_START.md QUICKSTART.md QUICK_TEST.md CAPACITY_QUICK_START.md MODEL_FACTORY_QUICK_START.md

# 删除重复的训练指南
rm -f TRAINING_GUIDE.md BASELINE_TRAINING_GUIDE.md CROSS_BATTERY_GUIDE.md PER_BATTERY_USAGE.md USAGE_EXAMPLE.md

# 删除重复的模型工厂
rm -f MODEL_FACTORY_GUIDE.md README_MODEL_FACTORY.md MODEL_LIBRARY_README.md

# 删除重复的项目结构
rm -f PROJECT_STRUCTURE.md PROJECT_STRUCTURE_HUST_CLEAN.md PROJECT_STRUCTURE_HUST_ONLY.md BASELINE_DOCUMENTATION_INDEX.md

# 删除其他重复文档
rm -f BASELINE_COMPARISON.md COMPARISON_USAGE.md QUICK_OPTIMIZATION_CHECKLIST.md
rm -f FEATURE_SELECTION_SUMMARY.md PLOTTING_QUICK_REF.md
rm -f BASELINE_MODEL_REVIEW.md BASELINE_REVIEW_FINDINGS.md
rm -f IMPLEMENTATION_NOTES.md DATA_SPLIT_AND_VISUALIZATION.md
rm -f SOH_CURVES_AND_SECONDARY_TRAINING.md HUST_BPINN_ANALYSIS.md

# 返回项目根目录
cd ..
```

**警告**: 删除前请确认你不需要这些文档！

---

## 💾 备份建议

如果你不确定是否要删除，可以先备份：

```bash
# 创建备份
mkdir notes_backup_$(date +%Y%m%d)
cp -r notes/* notes_backup_$(date +%Y%m%d)/

# 然后再删除重复文档
```

---

**创建日期**: 2025-11-14
