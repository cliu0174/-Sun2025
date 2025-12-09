# 项目文档索引

## 📚 文档结构

```
docs/
├── siamese/          # Siamese/Pairwise 模式相关文档
├── triplet/          # Triplet 模式相关文档
├── analysis/         # 分析和诊断文档
├── guides/           # 使用指南和教程
├── plans/            # 工作计划和实施方案
├── reports/          # 项目报告和周报
├── README.md         # 本文件（文档索引）
├── CLEANUP_SUMMARY.md                    # 文档清理记录
└── ARCHITECTURE_VISUALIZATION_PROMPT.md  # 架构可视化prompt
```

---

## 🔄 Siamese (孪生/配对) 模式

**适用场景**: 一阶平滑性约束，消除相邻预测间的突变

### 核心文档
- **[SIAMESE_使用指南.md](siamese/SIAMESE_使用指南.md)** - 中文完整使用指南 ⭐ 推荐
- **[SIAMESE_USAGE_GUIDE.md](siamese/SIAMESE_USAGE_GUIDE.md)** - 英文使用指南
- **[SIAMESE_TRAINING_PROCESS_EXPLAINED.md](siamese/SIAMESE_TRAINING_PROCESS_EXPLAINED.md)** - 训练流程详解
- **[CROSS_BATTERY_SIAMESE_SUMMARY.md](siamese/CROSS_BATTERY_SIAMESE_SUMMARY.md)** - 跨电池训练总结

### 关键特性
- 配对采样: (t, t+k)
- 一阶约束: smoothness = (pred_next - pred_t)²
- 适用于: 去除相邻点间的跳变

---

## 🎯 Triplet (三元组) 模式

**适用场景**: 二阶曲率约束，消除锯齿模式和方向突变

### 核心文档
- **[TRIPLET_IMPLEMENTATION_COMPLETE.md](triplet/TRIPLET_IMPLEMENTATION_COMPLETE.md)** - 完整实现说明 ⭐ 推荐
- **[TRIPLET_FIXES.md](triplet/TRIPLET_FIXES.md)** - 问题修复记录
- **[CURVATURE_LOSS_ZERO_EXPLANATION.md](triplet/CURVATURE_LOSS_ZERO_EXPLANATION.md)** - 曲率损失为0的完整解释

### 关键特性
- 三元组采样: (t, t+k, t+2k)
- 一阶约束: monotonicity (两个转换)
- 二阶约束: curvature = pred_3 - 2×pred_2 + pred_1
- 适用于: 消除锯齿，强制平滑衰减曲线

### 技术优势
- 相比Pairwise: 增加二阶约束
- 物理意义: 同时约束速度和加速度
- 效果: 更平滑的预测曲线

---

## 📊 分析文档

### 深度分析
- **[PINN_Analysis_Report.md](analysis/PINN_Analysis_Report.md)** - 物理信息神经网络分析报告
- **[MONOTONIC_LOSS_ANALYSIS.md](analysis/MONOTONIC_LOSS_ANALYSIS.md)** - 单调性损失深度分析
- **[SMOOTHNESS_CONSTRAINT_EXPLANATION.md](analysis/SMOOTHNESS_CONSTRAINT_EXPLANATION.md)** - 平滑性约束机制解释

### 问题诊断
这些文档帮助理解为什么某些损失为0，以及这是否正常。

---

## 📅 工作计划

- **[DECEMBER_WORK_PLAN.md](plans/DECEMBER_WORK_PLAN.md)** - 12月详细工作计划
- **[DECEMBER_PLAN_PRESENTATION.md](plans/DECEMBER_PLAN_PRESENTATION.md)** - 计划展示版本
- **[PHYSICS_CONSTRAINTS_IMPLEMENTATION_PLAN.md](plans/PHYSICS_CONSTRAINTS_IMPLEMENTATION_PLAN.md)** - 物理约束实施方案

---

## 📖 使用指南

- **[PHYSICS_CONSTRAINTS_USAGE.md](guides/PHYSICS_CONSTRAINTS_USAGE.md)** - 物理约束使用指南
- **[PHYSICS_CONSTRAINTS_SUMMARY.md](guides/PHYSICS_CONSTRAINTS_SUMMARY.md)** - 物理约束功能总结
- **[QUICK_START_PHYSICS.md](guides/QUICK_START_PHYSICS.md)** - 物理约束快速上手
- **[USAGE_GUIDE.md](guides/USAGE_GUIDE.md)** - 通用使用指南
- **[DATA_CLEANING_GUIDE.md](guides/DATA_CLEANING_GUIDE.md)** - 数据清洗指南

---

## 📊 项目报告

- **[weekly_report.md](reports/weekly_report.md)** - 周报
- **[PROJECT_STRUCTURE.md](reports/PROJECT_STRUCTURE.md)** - 项目结构说明

---

## 🎨 可视化

- **[ARCHITECTURE_VISUALIZATION_PROMPT.md](ARCHITECTURE_VISUALIZATION_PROMPT.md)** - 完整架构可视化prompt
- **[../viz_prompts/](../viz_prompts/)** - 分步骤架构图生成prompt
  - 01_data_pipeline.md - 数据处理流程图
  - (更多待添加)

---

## 📝 元文档

- **[CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md)** - 文档清理记录（2025-12-09）

---

## 🚀 快速开始

### 使用Siamese模式
```json
{
  "physics_constraints": {
    "enabled": true,
    "siamese_sampling": {
      "enabled": true,
      "split_threshold": 300,
      "step_k": 1
    }
  }
}
```

参考: [SIAMESE_使用指南.md](siamese/SIAMESE_使用指南.md)

### 使用Triplet模式
```json
{
  "physics_constraints": {
    "enabled": true,
    "triplet_sampling": {
      "enabled": true,
      "split_threshold": 300,
      "step_k": 1
    },
    "curvature_weight": 0.1
  }
}
```

参考: [TRIPLET_IMPLEMENTATION_COMPLETE.md](triplet/TRIPLET_IMPLEMENTATION_COMPLETE.md)

---

## 💡 选择建议

| 场景 | 推荐模式 | 原因 |
|------|---------|------|
| 预测有相邻跳变 | Siamese | 一阶平滑即可 |
| 预测有锯齿模式 | Triplet | 需要二阶约束 |
| 追求极致平滑 | Triplet | 更强的物理约束 |
| 计算资源有限 | Siamese | 更快的训练速度 |

---

## 📝 文档更新记录

- **2025-12-09**: 重组文档结构，分类归档
- **2025-12-09**: 完成Triplet实现和文档
- **2025-12-08**: 完成Siamese系列文档
- **2025-12-03**: 创建12月工作计划

---

## 🔗 相关资源

- **主项目README**: [../README.md](../README.md)
- **项目总结**: [../PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md)
- **架构可视化**: [../ARCHITECTURE_VISUALIZATION_PROMPT.md](../ARCHITECTURE_VISUALIZATION_PROMPT.md)
