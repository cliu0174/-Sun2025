# 项目目录结构

```
1111-soh/
├── README.md                        # 项目主文档
├── USAGE_GUIDE.md                   # 使用指南
├── requirements.txt                 # Python 依赖
│
├── train_cross_battery.py           # 跨电池训练主脚本
├── train_single_model.py            # 单电池训练脚本
├── train_seq2seq.py                 # Seq2Seq 模型训练脚本
├── plot_hust_capacity_curves.py     # 容量曲线绘图工具
│
├── configs/                         # 配置文件目录
│   └── models/                      # 模型配置
│       ├── *_config.json           # 基准模型配置 (FNN, CNN, LSTM, GRU, etc.)
│       └── *_seq2seq_config.json   # Seq2Seq 模型配置
│
├── data/                            # 数据目录
│   └── HUST data/                   # HUST 电池数据集 (77组)
│
├── data_loaders/                    # 数据加载器
│   ├── __init__.py
│   └── data_loader_hust.py
│
├── models/                          # 模型定义
│   ├── __init__.py
│   ├── baseline_models.py          # 基准模型 (Many-to-One)
│   ├── seq2seq_models.py           # Seq2Seq 模型 (Many-to-Many)
│   └── model_factory.py            # 统一模型工厂
│
├── results/                         # 训练结果
│   ├── cross_battery/              # 跨电池训练结果
│   └── seq2seq/                    # Seq2Seq 训练结果
│
├── notes/                           # 项目文档和笔记
│   ├── README.md                   # 文档索引
│   ├── ARCHIVE_INDEX.md            # 归档索引
│   ├── 优化指南/                   # 优化相关指南
│   ├── 分析报告/                   # 分析报告
│   ├── 对比研究/                   # 对比研究
│   └── 绘图参考/                   # 绘图参考
│
├── tests/                           # 测试文件
│   ├── README.md
│   ├── test_*.py                   # 各类测试
│   └── archive/                    # 归档的测试
│
└── PINN4SOH/                        # PINN 参考实现（外部项目）
```

## 核心功能

### 训练脚本
- **train_cross_battery.py**: 跨电池训练（推荐使用）
- **train_single_model.py**: 单电池训练
- **train_seq2seq.py**: Seq2Seq 模型专用训练

### 模型类型
**基准模型（Many-to-One）:**
- FNN, CNN, LSTM, GRU, BiLSTM, BiGRU, MLP, ResCNN

**Seq2Seq 模型（Many-to-Many）:**
- LSTMSeq2Seq, GRUSeq2Seq, BiLSTMSeq2Seq, BiGRUSeq2Seq

### 数据集
- HUST: 77组 LFP 电池，每组 1000+ 充放电循环
- 跨电池训练: 60% train / 20% val / 20% test

## 快速开始

查看 `USAGE_GUIDE.md` 获取详细使用说明。
