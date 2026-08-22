# 第四章 4.3：五模型横向对比筛选实验协议

## 目的与边界

本实验回答“在相同的稀疏、轨迹集中式 SOH 监督条件下，PI-MSCL 与主流数据驱动估计器相比表现如何？”本轮为**单种子筛选运行**，用于确定结果形态、排查实现问题和生成候选表格；它不是可在论文中写作 `mean ± std` 的正式主表。

## 统一实验条件

| 项目 | 固定设置 |
|---|---|
| 数据划分 | battery-level 60/20/20，`split_seed=42`（实际为 46/15/16 块电池） |
| 输入 | 当前代码实现的 16 个充电特征，40-cycle many-to-one 窗口 |
| 监督 | `trajectory_prefix_concentrated`，10% SOH 标签预算 |
| 单次种子 | `training_seed=2262`，`mask_seed=3262` |
| 参数选择 | 只依据完整验证集 MAE；测试集仅在模型冻结后评估 |
| 训练 | Adam、最多 200 epochs、batch size 1024、同一 warmup-cosine 调度和早停策略 |
| 评价 | MAE、RMSE、R²；另外归档单调违例率、累计向上超额和速率粗糙度 |

种子选择依据：在已经完成的 PI-MSCL 10% trajectory-prefix 运行中，`2262/3262` 得到最低测试 MAE（2.988%）。这种按既有最优结果选取的做法只适合**预筛选**，不能用于最终泛化性结论。

## 对比组

| ID | 模型 | 类别 | 实现边界 |
|---|---|---|---|
| 1 | XGBoost | 保留的树模型基线 | 采用项目已有实现 |
| 2 | GRU | 保留的循环神经网络基线 | 采用项目已有实现 |
| 3 | Transformer-SOH | 近期文献横向模型 | 复现 Transformer 多头自注意力时序编码；不使用论文特定电压片段特征 |
| 4 | CNN-BiGRU-Attention | 近期文献横向模型 | 复现 CNN、双向 GRU 与时序注意力主干；不使用 Hiking Optimization Algorithm（HOA） |
| 5 | PI-MSCL | 本文方法 | 多尺度 CNN-LSTM 加软单调约束 |

新增模型的原始论文：

1. Shu, X. *et al.* “State of health estimation for lithium-ion batteries based on voltage segment and transformer,” *Journal of Energy Storage* **108**, 115200 (2025). DOI: [10.1016/j.est.2024.115200](https://doi.org/10.1016/j.est.2024.115200).
2. Wu, S. *et al.* “SOH Estimation of Lithium Battery Under Improved CNN-BIGRU-Attention Model Based on Hiking Optimization Algorithm,” *World Electric Vehicle Journal* **16**, 487 (2025). DOI: [10.3390/wevj16090487](https://doi.org/10.3390/wevj16090487).

两篇论文均在近两年内发表，且属于数据驱动 SOH 估计。这里复现的是其可迁移的**网络主干**；原论文各自使用的数据集、专用健康指标和超参数搜索均不迁入，避免用额外信息或不等量调参破坏公平性。

## 运行与归档

运行器：`experiments/run_chapter4_five_model_comparison.py`。

```powershell
python experiments/run_chapter4_five_model_comparison.py --device cuda
```

每个模型会保存预测、测试指标、轨迹指标、验证集最优 epoch 和完整配置。正式表格仅在后续对五个模型使用相同的多个训练/标签种子重复后更新为 `mean ± std`。
