# PI-CNNLSTM 电池 SOH 估计项目 — Claude 记忆卡

> 本文件在新会话中会自动加载。阅读后请优先查阅下方详细计划，避免重复讨论已确定的方案。

## 项目一句话概述

基于**物理一致性约束**的电池 SOH 估计（PI-CNNLSTM），核心场景是**部分生命周期监督**（Partial Lifecycle Supervision）——即只有电池生命周期的部分区段有 SOH 标签。数据集为 HUST 77 块电池、14 维特征。

## 当前状态（2026-04-18 更新）

- **基线**：**baseline-v2.2**（CNN-LSTM + 软单调 + 边界约束 + 60/20/20划分 + 种子42）
- **监督场景**：**部分生命周期监督**（label masking 已实现，保留样本只 mask 标签）
- **监督比例矩阵**：`[1.0, 0.7, 0.5, 0.3]`
- **当前阶段**：**Stage 0-3 代码全部完成，所有实验待服务器运行；Stage 4 待开发**
- **代码现状**：
  - ✅ Stage 0：部分监督 + 物理约束 + `run_baseline_v22.py`（20 次）
  - ✅ Stage 1：M1 注意力（`models/modules/attention.py`）+ M2 MC Dropout（`models/modules/mc_dropout.py`）+ M7 置信区间（`evaluation/uncertainty_eval.py`）
  - ✅ Stage 2：M4 速率连续性（`physics_loss.py::smoothness_loss`）+ M5 自适应权重（`models/adaptive_loss.py`）+ `config_override` 支持
  - ✅ Stage 3：M6 伪标签（`training/pseudo_labeling.py::PseudoLabelManager`）
  - 🔜 Stage 4：Exp-07 全模块集成（待开发）
- **实验脚本**：`run_baseline_v22.py` + `experiments/run_exp01~06.py`（共 7 个，总计 ~95 次运行）
- **下一步行动**：① 开发 Stage 4（Exp-07）；② 有服务器后按顺序跑实验
- **参考论文**：9 篇 PDF 在 `paper/` 目录（已 .gitignore）

## ⚠️ 重要：已制定的改进计划

**完整详细计划位于：[`docs/IMPROVEMENT_PLAN.md`](docs/IMPROVEMENT_PLAN.md)**

未来任何关于"下一步怎么改进"、"要加哪些模块"、"论文怎么写"的讨论，**请先读这份文档**，不要重新发散讨论。

### 计划核心速览（最终组合）

```
架构：CNN-LSTM + 循环级注意力（M1）
损失：自适应权重（M5）+ 软单调 + 边界 + 速率连续性（M4）
训练：不确定性引导伪标签扩展部分监督（M6）
评估：MC Dropout 置信区间（M2+M7）+ 特征缺失鲁棒性（M8）
```

### 8 个模块（按优先级）

| ID | 模块 | 类型 | 优先级 |
|----|------|------|--------|
| M1 | 循环级注意力 | 架构 | P0 |
| M2 | MC Dropout | 架构 | P0 |
| M3 | 个体归一化 | 架构 | P2 |
| M4 | 速率连续性约束 | 损失 | P0 |
| M5 | 自适应损失权重 | 损失 | P0 |
| M6 | 不确定性伪标签 ⭐核心 | 训练 | P1 |
| M7 | 置信区间评估 | 评估 | P0 |
| M8 | 特征缺失鲁棒性 | 评估 | P1 |

### 实施路线图（5 个 Stage）

- Stage 0：基线锁定
- Stage 1：架构增强（Exp-01, 02）
- Stage 2：损失函数精化（Exp-03, 04, 05）
- Stage 3：训练策略升级（Exp-06）⭐核心
- Stage 4：全模块集成（Exp-07）
- Stage 5：鲁棒性验证（Exp-08）

**总预计工期：10 天**

## 用户偏好

- 用户喜欢**模块化、可插拔**的代码组织
- 喜欢**逐模块消融验证**，而不是一次上全套
- 喜欢**中文交流**，论文中文撰写
- 邮箱：liuchang2262@gmail.com

## 关键代码入口

- `train_cross_battery.py` — 主训练脚本（支持 M5 自适应权重、M6 伪标签、config_override）
- `run_baseline_v22.py` — Stage 0 基线扫描（20 次批量实验，断点续跑）
- `models/physics_loss.py` — 物理约束损失（含 supervision_mask、forward_components）
- `models/adaptive_loss.py` — M5 自适应损失权重（AdaptivePhysicsLoss）
- `training/pseudo_labeling.py` — M6 伪标签管理器（PseudoLabelManager）
- `models/modules/attention.py` — M1 循环级注意力（CycleAttention）
- `models/modules/mc_dropout.py` — M2 MC Dropout（MCDropout + mc_predict）
- `evaluation/uncertainty_eval.py` — M7 置信区间评估指标
- `data_loaders/data_loader_hust.py` — 数据加载（含 is_labeled 字段）
- `configs/models/` — 所有实验配置（baseline/attention/mc/rate_smoothness/adaptive_weight/pseudo_label）
- `experiments/` — Exp-01~06 实验脚本

## 新会话启动检查清单

1. [ ] 读完本文件
2. [ ] 读 `docs/IMPROVEMENT_PLAN.md`
3. [ ] 检查 `experiments/` 目录，确认当前 Stage 进度
4. [ ] 如果用户问"下一步做什么"，参考计划中的"Stage X"章节
