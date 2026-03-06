# Battery SOH Estimation with Physics-Informed Neural Networks

基于物理信息神经网络的锂离子电池健康状态（SOH）估计项目。

## 项目特点

- **物理约束**: 融合单调性和曲率约束的物理信息神经网络
- **数据退化场景**: 4种数据退化场景模拟真实传感器故障
- **跨电池训练**: 支持77组电池的跨电池迁移学习
- **可视化**: 完整的训练和预测可视化分析

## 项目结构

```
.
├── train_cross_battery.py          # 主训练脚本（跨电池训练）
│
├── models/                         # 模型定义
│   ├── baseline_models.py         # LSTM, GRU, CNN, BiLSTM, BiGRU
│   ├── cnn_lstm.py                # CNN-LSTM 混合模型
│   ├── seq2seq_models.py          # Seq2Seq 系列模型
│   ├── model_factory.py           # 模型工厂和配置加载器
│   └── physics_loss.py            # 物理约束损失函数
│
├── data_loaders/                   # 数据加载模块
│   ├── data_loader_hust.py        # HUST 数据集加载器
│   └── data_loader_mit.py         # MIT 数据集加载器
│
├── utils/                          # 工具模块
│   ├── data_augmentation.py       # 数据退化场景实现
│   ├── lr_schedulers.py           # 学习率调度器
│   ├── data/                      # 数据处理工具
│   │   ├── data_cleaning.py       # 数据清洗
│   │   └── HUST_dataloader.py     # HUST 数据加载工具
│   ├── features/
│   │   └── feature_extraction.py  # 特征提取
│   ├── inference/
│   │   └── inference.py           # 推理脚本
│   ├── plotting/
│   │   └── plot_hust_capacity_curves.py  # 容量曲线绘图
│   ├── training/                  # 其他训练脚本
│   │   ├── train_mit.py
│   │   ├── train_seq2seq.py
│   │   ├── train_single_model.py
│   │   ├── train_with_physics.py
│   │   └── train_xgboost_baseline.py
│   └── testing/                   # 测试脚本
│       ├── test_mit_loader.py
│       ├── test_monotonic_weights.py
│       └── test_xgboost_baseline.py
│
├── configs/                        # 模型配置文件（JSON）
│   └── models/
│
├── data/                           # 数据目录
│   └── HUST data/                 # 77 组电池 CSV 数据
│
├── results/                        # 训练结果
│   └── cross_battery/             # 跨电池训练结果
│
├── docs/                           # 文档
│   ├── guides/                    # 使用指南
│   ├── scenarios/                 # 数据退化场景文档
│   ├── siamese/                   # Siamese 模式文档
│   ├── analysis/                  # 分析报告
│   └── README.md                  # 文档导航
│
└── PROJECT_SUMMARY.md              # 项目详细总结
```

## 快速开始

### 1. 基础训练

```bash
# 默认配置训练（Scenario 2，稀疏采样）
python train_cross_battery.py

# 使用 Scenario 4（连续循环缺失）
python train_cross_battery.py --degradation_scenario scenario4 --cycle_drop_rate 0.3 --cycle_drop_num_gaps 2
```

### 2. 批量参数测试

```bash
# 测试多个单调性权重组合
python utils/testing/test_monotonic_weights.py
```

### 3. 模型推理

```bash
python utils/inference/inference.py

# 参考指南: docs/guides/INFERENCE_QUICK_START.md
```

## 文档导航

| 类别 | 路径 | 说明 |
|------|------|------|
| 使用指南 | [docs/guides/USAGE_GUIDE.md](docs/guides/USAGE_GUIDE.md) | 主使用指南 |
| 推理快速开始 | [docs/guides/INFERENCE_QUICK_START.md](docs/guides/INFERENCE_QUICK_START.md) | 推理脚本用法 |
| 物理约束 | [docs/guides/PHYSICS_CONSTRAINTS_USAGE.md](docs/guides/PHYSICS_CONSTRAINTS_USAGE.md) | 物理约束配置 |
| 数据增强 | [docs/scenarios/DATA_AUGMENTATION_GUIDE.md](docs/scenarios/DATA_AUGMENTATION_GUIDE.md) | 4种退化场景总指南 |
| Scenario 4 | [docs/scenarios/SCENARIO4_USAGE.md](docs/scenarios/SCENARIO4_USAGE.md) | 连续循环缺失场景 |
| Siamese 模式 | [docs/siamese/SIAMESE_USAGE_GUIDE.md](docs/siamese/SIAMESE_USAGE_GUIDE.md) | 跨电池 Siamese 训练 |
| PINN 分析 | [docs/analysis/PINN_Analysis_Report.md](docs/analysis/PINN_Analysis_Report.md) | PINN 分析报告 |

## 核心功能

### 数据退化场景

模拟真实场景下的传感器故障和数据缺失：

- **Scenario 1**: 噪声 + 随机丢弃
- **Scenario 2**: 规律稀疏采样（如定期 HPPC 测试）
- **Scenario 3**: 随机缺失（随机传感器读取失败）
- **Scenario 4**: 连续循环缺失（传感器系统故障）

### 物理约束

- **单调性约束**: 确保 SOH 随循环次数单调递减（软约束 + 容差机制）
- **平滑性约束**: 消除预测曲线的跳变
- **曲率约束**: 进一步约束二阶导数，获得更平滑的曲线
- **时间衰减权重**: 近期循环数据权重更高

### 模型库

10 个深度学习模型，统一通过 `ModelFactory` 接口创建：

- 基础循环网络：LSTM, GRU, BiLSTM, BiGRU
- 卷积网络：CNN, ResCNN, CNN-LSTM, CNN-BiLSTM, PI-CNN-LSTM
- Seq2Seq：LSTM/GRU/BiLSTM/BiGRU Seq2Seq
- 传统机器学习：FNN, XGBoost

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA（推荐，用于 GPU 加速）

详细依赖见 `requirements.txt`

## 参考文献

Sun, G., Liu, Y., & Liu, X. (2025). A method for estimating lithium-ion battery state of health based on physics-informed machine learning. *Journal of Power Sources*, 627, 235767.
