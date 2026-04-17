# PI-CNNLSTM 电池 SOH 估计项目 — Claude 记忆卡

> 本文件在新会话中会自动加载。阅读后请优先查阅下方详细计划，避免重复讨论已确定的方案。

## 项目一句话概述

基于**物理一致性约束**的电池 SOH 估计（PI-CNNLSTM），核心场景是**部分生命周期监督**（Partial Lifecycle Supervision）——即只有电池生命周期的部分区段有 SOH 标签。数据集为 HUST 77 块电池、14 维特征。

## 当前状态（2026-04-17 更新）

- **基线**：**baseline-v2.2**（CNN-LSTM + 软单调 + 边界约束 + 60/20/20划分 + 种子42）
- **监督场景**：**部分生命周期监督**（label masking 已实现，保留样本只 mask 标签）
- **监督比例矩阵**：`[1.0, 0.7, 0.5, 0.3]`
- **当前阶段**：**Stage 1 代码完成，所有实验待运行**
- **代码现状**：
  - ✅ 部分监督机制已实现（`generate_supervision_mask()` + `is_labeled` 字段 + masked MSE）
  - ✅ `run_baseline_v22.py` 已就绪（5 seeds × 4 ratios = 20 次，支持断点续跑）
  - ✅ 烟雾测试通过（ratio=1.0 和 ratio=0.5 均验证正常）
  - ⚠️ `cnn_lstm_config.json` 中 `physics_constraints.enabled` 当前为 `false`（用户手动调整）
  - ℹ️ `utils/data_augmentation` 不存在，scenario1-4 相关代码不可用（当前不需要）
- **下一步行动**：运行 `python run_baseline_v22.py` 得到 20 次基线数字，再推进 Stage 1
- **参考论文**：9 篇 PDF 在 `paper/` 目录（已.gitignore）

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

- `train_cross_battery.py` — 当前主训练脚本（baseline-v2.2）
- `run_baseline_v22.py` — Stage 0 基线扫描（20 次批量实验）
- `models/physics_loss.py` — 物理约束损失（含 supervision_mask 支持）
- `data_loaders/data_loader_hust.py` — 数据加载（含 is_labeled 字段）
- `configs/models/cnn_lstm_config.json` — 模型与物理约束配置
- `models/` — 模型定义目录
- `feature_extraction.py` — 14 维特征提取
- `compare_models.py` — 模型对比

## 新会话启动检查清单

1. [ ] 读完本文件
2. [ ] 读 `docs/IMPROVEMENT_PLAN.md`
3. [ ] 检查 `experiments/` 目录，确认当前 Stage 进度
4. [ ] 如果用户问"下一步做什么"，参考计划中的"Stage X"章节
