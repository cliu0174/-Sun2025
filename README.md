# PI-CNNLSTM — 基于物理一致性约束的锂电池 SOH 估计

基于 **HUST 数据集**（77 块电池，14 维特征），使用 CNN-LSTM 网络结合物理约束损失函数估计锂离子电池健康状态（SOH）。核心场景为**部分生命周期监督**（Partial Lifecycle Supervision）——只有电池生命周期的部分区段有 SOH 标签。

---

## 环境准备

```bash
pip install -r requirements.txt
# Python 3.8+，PyTorch 2.0+，推荐 CUDA GPU
```

---

## 快速开始

```bash
python train_cross_battery.py
```

训练完成后，结果（模型权重、预测图、性能指标）自动保存到 `results/cross_battery/cnn_lstm/`。

在脚本底部的 `__main__` 块中调整参数：

```python
MODEL_TYPE = 'cnn_lstm'   # 可选: 'fnn', 'cnn', 'lstm', 'gru', 'bilstm', 'bigru', 'rescnn'
SEED       = 42
```

---

## 常用运行入口速查

### MS-PI-CNNLSTM / PI-MS-CNN-LSTM

MS-PI-CNNLSTM（多尺度 CNN + LSTM + 物理一致性约束）的主实验脚本是：

```bash
python experiments/run_exp09_ms_full_stack.py
```

该脚本对应 Exp-09，多尺度模型类型为 `ms_cnn_lstm_v2`，基础配置文件为 `configs/models/ms_cnn_lstm_v2_config.json`，结果输出到 `experiments/exp09_ms_full_stack/`。

Exp-09 内部包含三组主要配置：

| 实验 ID | 模型含义 | 备注 |
|------|------|------|
| `Exp09a_ms_full_stack_w03` | MS-CNN-LSTM + MC Dropout + 速率连续性 + 物理约束 | 较强物理约束，`monotonic_weight=0.3` |
| `Exp09b_ms_full_stack_w01` | MS-CNN-LSTM + MC Dropout + 速率连续性 + 物理约束 | 较轻物理约束，`monotonic_weight=0.1` |
| `Exp09c_ms_pi_only` | **PI-MS-CNN-LSTM / MS-PI-CNNLSTM** | 论文主方法常用版本：多尺度 CNN-LSTM + 软单调物理约束 |

如果只想在通用训练入口中调用多尺度模型，可使用 `model_type='ms_cnn_lstm_v2'`，并在 `config_override` 中打开 `architecture.use_multiscale=True` 和相应 `physics_constraints`。

### 之前生成论文图 / 示意图的脚本

以下脚本主要用于“生图”，不一定都需要重新训练；优先看“是否需 GPU”列：

| 脚本 | 主要用途 | 是否需 GPU | 输出 |
|------|------|------|------|
| `scripts/plot_paper_figures.py` | 第四章主绘图脚本，一次生成 MAE 趋势、消融热力图、单调违反率、鲁棒性折线图等 6 张核心图 | 否，使用内置汇总数据 | `figures/` |
| `scripts/plot_exp09c_advantages.py` | 生成 Exp09c / PI-MS-CNN-LSTM 核心优势图，包括全模型横评、监督稀疏度交互效应、复杂度陷阱、鲁棒性优势 | 否，使用内置汇总数据 | `figures/` |
| `scripts/plot_mae_trend.py` | 生成不同监督比例下 MAE 趋势图 | 否，使用内置数据 | `figures/fig4_9_mae_trend.{png,pdf}` |
| `scripts/plot_battery_trajectory.py` | 生成单电池 SOH 预测轨迹对比图，读取预测缓存 `.npz` | 否，缓存存在即可 | `figures/fig4_8_soh_*.{png,pdf}` |
| `scripts/plot_model_comparison.py` | 生成图4-6散点图和图4-7指标柱状图；完整运行会训练 XGBoost/LSTM/CNN-LSTM/PI-MS-CNN-LSTM，`--plot-only` 只读缓存出图 | 完整运行需 GPU；`--plot-only` 不需要 | `figures/fig4_6_scatter.{png,pdf}`、`figures/fig4_7_bar.{png,pdf}` |
| `scripts/run_and_plot_predictions.py` | 训练 7 个对比模型并生成多模型预测曲线和散点图；支持 `--plot-only` | 完整运行需 GPU；`--plot-only` 不需要 | `figures/`、`figures/pred_cache/` |
| `scripts/plot_anomaly_detection.py` | CRT 数据集异常检测与 SOH 轨迹图，含 5 类退化场景 × 3 种严重程度 | 视缓存/数据而定 | `figures/` |
| `scripts/plot_crt_dataset.py` | CRT 自建数据集概览示意图 | 否 | `docs/crt_dataset_overview.png` |
| `scripts/plot_hust_anomaly_trajectories.py` | HUST 异常电池轨迹可视化 | 否，需本地数据/结果 | `figures/` |
| `scripts/generate_anomaly_score_two_panel_svg.py` | 生成异常分数双面板 SVG 示意图 | 否 | `outputs/figures/anomaly_score_two_panel_no_title.svg` |
| `docs/plot_degradation_scenarios.py` | 生成多种电池退化/异常场景示意图 | 否 | `docs/` |
| `docs/plot_degradation_scenarios_self_trajectory.py` | 生成自轨迹形式的退化场景示意图 | 否 | `docs/` |
| `docs/plot_15_anomaly_curves.py` | 生成 15 条异常退化曲线示意图 | 否 | `docs/` |

架构图相关文件位于 `docs/`：`MS-PI-CNNLSTM_architecture.tex`、`MS-PI-CNNLSTM_architecture.drawio`，已生成预览包括 `MS-PI-CNNLSTM_architecture.png/pdf`。

---

## 项目结构

```
1111-soh/
├── train_cross_battery.py          # 主训练脚本（核心入口，支持自适应权重、伪标签、config_override）
├── run_baseline_v22.py             # Stage 0 基线批量扫描（20 次，断点续跑）
├── inference.py                    # 模型推理
├── compare_models.py               # 多模型横向对比
├── feature_extraction.py           # 14 维特征提取逻辑
│
├── models/
│   ├── cnn_lstm.py                 # CNN-LSTM 模型（基础版）
│   ├── cnn_lstm_v2.py              # CNN-LSTM v2（多尺度 CNN）
│   ├── baseline_models.py          # FNN / CNN / LSTM / GRU 等
│   ├── model_factory.py            # 模型工厂
│   ├── physics_loss.py             # 物理约束损失（软单调 + 边界 + 速率连续性）
│   ├── adaptive_loss.py            # M5 自适应损失权重（AdaptivePhysicsLoss）
│   └── modules/
│       ├── attention.py            # M1 循环级注意力（CycleAttention）
│       └── mc_dropout.py           # M2 MC Dropout（MCDropout + mc_predict）
│
├── training/
│   └── pseudo_labeling.py          # M6 不确定性引导伪标签（PseudoLabelManager）
│
├── evaluation/
│   ├── anomaly_detection.py        # M8 在线异常检测（trajectory_deviation 方法）
│   ├── robustness_eval.py          # M8 特征缺失鲁棒性（三类扰动）
│   ├── uncertainty_eval.py         # M7 置信区间评估（PICP / MPIW / Spearman）
│   └── physics_viz.py              # 物理约束可视化工具
│
├── data_loaders/
│   └── data_loader_hust.py         # HUST 数据加载与窗口化（含 is_labeled 字段）
│
├── configs/models/                 # 各模型 JSON 配置
│   └── cnn_lstm_config.json        # 当前默认配置
│
├── utils/
│   └── lr_schedulers.py            # 学习率调度器
│
├── experiments/                    # Exp-01~12 实验脚本（见下方说明）
├── scripts/                        # 绘图 / 报告生成工具脚本（见下方说明）
│
├── data/HUST data/                 # 原始数据（77 块电池 .csv，需自行获取）
├── results/                        # 训练输出（模型权重、预测图、指标）
├── figures/                        # 论文插图输出目录
└── outputs/                        # 文档输出目录（.docx / .tex）
```

---

## 实验脚本（experiments/）

| 脚本 | 说明 |
|------|------|
| `run_exp01_attention.py` | M1 循环级注意力消融实验 |
| `run_exp02_mc_dropout.py` | M2 MC Dropout 不确定性消融实验 |
| `run_exp03_rate_smoothness.py` | M4 速率连续性约束消融实验 |
| `run_exp04_adaptive_weight.py` | M5 自适应损失权重消融实验 |
| `run_exp05_combined_loss.py` | M4+M5 联合损失消融实验 |
| `run_exp06_pseudo_label.py` | M6 伪标签训练策略消融实验 |
| `run_exp07_full_stack.py` | Full Stack 集成（4 比例 × 5 种子，含 M7 PICP/MPIW） |
| `run_exp08_robustness.py` | M8 鲁棒性验证（3 类扰动，Baseline vs Full Stack） |
| `run_exp09_ms_full_stack.py` | **MS-PI-CNNLSTM 主运行脚本**；Exp09 多尺度 CNN-LSTM / PI-MS-CNN-LSTM，包含 Exp09a/Exp09b/Exp09c，结果输出到 `experiments/exp09_ms_full_stack/` |
| `run_exp10_uncertainty.py` | Exp10 不确定性评估 |
| `run_exp11_robustness.py` | Exp11 鲁棒性扩展实验 |
| `run_exp12_anomaly_detection.py` | Exp12 在线异常检测完整实验 |
| `run_ablation_single_module.py` | 单模块逐一消融（快速验证） |
| `run_exp_arch_mvp.py` | 架构 MVP 快速验证 |
| `run_m6_verify.py` / `run_m6_multiseed.py` / `run_m6_r05_probe.py` | M6 伪标签专项验证 |
| `mvp_anomaly_verify.py` | 异常检测 MVP 快速验证 |
| `visualize_ablation.py` | 消融结果可视化 |
| `analyze_results.py` | 实验结果批量分析 |
| `plot_best_seed.py` | 绘制最优种子的预测曲线 |
| `verify_physics_switch.py` | 验证物理约束开关是否生效 |
| `verify_transformer_reconstruction.py` | Transformer 架构重构验证 |

---

## 工具脚本（scripts/）

### 绘图脚本

| 脚本 | 用途 | 输出 |
|------|------|------|
| `plot_paper_figures.py` | **论文第四章主绘图脚本**，生成全部 6 张插图（MAE 趋势、消融热力图、单调违反率、鲁棒性折线图等），无需重跑实验 | `figures/` |
| `plot_exp09c_advantages.py` | 生成 PI-MS-CNN-LSTM (Exp09c) 核心优势对比图（4 张：全模型横评、监督稀疏度交互效应、复杂度陷阱、鲁棒性优势） | `figures/` |
| `plot_mae_trend.py` | 图4-9：不同监督比例下各模型 MAE 变化趋势折线图（数据硬编码，无需训练） | `figures/fig4_9_mae_trend.{png,pdf}` |
| `plot_battery_trajectory.py` | 图4-8：单电池 SOH 预测轨迹对比（CNN-LSTM vs PI-MSCL），从缓存 .npz 读取，支持 `--battery` 指定电池 | `figures/fig4_8_soh_*.{png,pdf}` |
| `plot_model_comparison.py` | 图4-6 & 图4-7：训练 4 个对比模型并生成散点图（pred vs true）和 RMSE/MAE/R² 柱状对比图，支持 `--plot-only` 仅从缓存出图 | `figures/fig4_6_scatter.{png,pdf}` `figures/fig4_7_bar.{png,pdf}` |
| `plot_anomaly_detection.py` | 图4-11 & 图4-12 & 表4-9：CRT 数据集上 SOH 轨迹与异常检测（5 种退化场景 × 3 种严重程度），含 N 连击报警逻辑 | `figures/` |
| `plot_crt_dataset.py` | CRT 自建数据集概览图：部分生命周期监督下的退化轨迹示意 | `docs/crt_dataset_overview.png` |
| `plot_hust_anomaly_trajectories.py` | HUST 数据集上异常电池轨迹可视化 | `figures/` |
| `generate_anomaly_score_two_panel_svg.py` | 生成异常分数双面板示意 SVG（无需数据，纯示意图） | `outputs/figures/anomaly_score_two_panel_no_title.svg` |

### 数据导出脚本

| 脚本 | 用途 | 输出 |
|------|------|------|
| `export_table48.py` | 导出表4-8：不同监督比例（r=1.0/0.7/0.5/0.3）下 CNN-LSTM vs PI-MSCL 的 MAE/RMSE/R² 对比，多种子均值，支持断点续跑 | `figures/` + 控制台 |
| `run_and_plot_predictions.py` | **需 GPU**：训练 7 个对比模型并生成 Fig7（多模型预测曲线）和 Fig8（散点图），配置 r=0.3/seed=929，支持 `--plot-only` | `figures/` |

### 文档生成脚本

| 脚本 | 用途 | 输出 |
|------|------|------|
| `build_0521_report.py` | 生成 0521 阶段汇报 Word 文档（含图表、结果） | `outputs/` |
| `build_0604_report.py` | 在 0521 汇报模板基础上更新生成 0604 阶段汇报文档 | `outputs/0604汇报.docx` |
| `build_chapter4_revised_docx.py` | 生成第四章全文扩写版 Word 文档（参考文献增强稿） | `outputs/` |
| `build_opening_report_soh_supplement_docx.py` | 生成开题报告 SOH 已有研究补充建议 Word 文档 | `outputs/开题报告_SOH已有研究补充建议.docx` |
| `build_opening_report_soh_supplement_docx_from_md.py` | 从 Markdown 源文件生成开题报告补充建议 Word 文档（v3） | `outputs/开题报告_SOH已有研究补充建议_v3.docx` |
| `build_soc_proposal_adapted_docx.py` | 生成 SOC 研究基础开题报告适配版 Word 文档 | `outputs/SOC已有研究基础_开题报告适配版.docx` |
| `rewrite_chapter4_boss_feedback.py` | 根据导师反馈重构第四章，从源 .docx 生成重构稿 | `outputs/第四章_老总反馈重构稿.docx` |
| `docx2tex.py` | 将第四章 .docx 转换为 XeLaTeX/ctex 格式 .tex 文件（中文学位论文） | `*.tex` |

---

## 模型与配置

### CNN-LSTM 架构

CNN 提取局部特征（电压/电流/温度曲线），LSTM 建模容量衰减的长期时序依赖。

主要超参数在 `configs/models/cnn_lstm_config.json` 中配置：

```json
{
  "architecture": {
    "hidden_size": 64,
    "cnn_channels": [256, 128],
    "kernel_size": 7,
    "dropout_rate": 0.4
  },
  "training": {
    "num_epochs": 200,
    "batch_size": 256,
    "learning_rate": 0.0005
  },
  "data": {
    "window_size": 40
  }
}
```

### 物理约束损失

在标准 MSE 基础上加入三项物理先验：

- **软单调性约束**：惩罚 SOH 预测值上升，符合电池不可逆衰退的物理规律
- **边界约束**：将预测值约束在 [0, 1] 范围内
- **速率连续性约束**（M4）：惩罚相邻时步的 SOH 变化率突变

```json
"physics_constraints": {
    "enabled": true,
    "monotonic_weight": 0.1,
    "boundary_weight": 0.05,
    "rate_smoothness_weight": 0.02
}
```

### 数据划分

按电池随机划分，**train / val / test = 60% / 20% / 20%**（约 46 / 15 / 16 块），种子固定（seed=42）确保可复现。

### 部分监督设置

通过 `is_labeled` 字段对标签进行 masking，物理约束在所有样本上生效，监督损失仅在有标签样本上计算。监督比例矩阵：`[1.0, 0.7, 0.5, 0.3]`。

---

## 支持的模型类型

| model_type | 说明 |
|-----------|------|
| `cnn_lstm` | CNN-LSTM 混合（默认）|
| `lstm` | 标准 LSTM |
| `gru` | 标准 GRU |
| `bilstm` | 双向 LSTM |
| `bigru` | 双向 GRU |
| `cnn` | 纯 CNN |
| `fnn` / `mlp` | 全连接网络 |
| `rescnn` | 残差 CNN |

切换方式：修改 `train_cross_battery.py` 中的 `MODEL_TYPE`，或直接修改对应 `configs/models/*.json`。

---

## 数据集

**HUST 锂离子电池数据集**（需自行获取，不含在仓库中）

- 77 块电池，每块约 100–600 个充放电循环
- 14 维特征：充放电容量、能量、内阻、温度等
- 放置路径：`data/HUST data/*.csv`

---

## 参考文献

Sun, G., Liu, Y., & Liu, X. (2025). A method for estimating lithium-ion battery state of health based on physics-informed machine learning. *Journal of Power Sources*, 627, 235767.

---

## 联系

liuchang2262@gmail.com
