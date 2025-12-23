# Battery SOH Estimation with Physics-Informed Neural Networks

基于物理信息神经网络的锂离子电池健康状态（SOH）估计项目。

## 🎯 项目特点

- 🔬 **物理约束**: 融合单调性和曲率约束的物理信息神经网络
- 📊 **数据退化场景**: 4种数据退化场景模拟真实传感器故障
- 🧪 **批量测试**: 自动化多参数批量测试和性能分析
- 🔄 **跨电池训练**: 支持跨电池迁移学习
- 📈 **可视化**: 完整的训练和预测可视化分析

## 📁 项目结构

```
.
├── train_cross_battery.py      # 主训练脚本（跨电池训练）
├── test_monotonic_weights.py   # 批量测试脚本（支持多参数测试）
├── inference.py                 # 推理脚本
├── utils/
│   ├── data_augmentation.py    # 数据退化场景实现
│   └── ...                      # 其他工具
├── models/                      # 模型定义
├── data/                        # 数据目录
├── results/                     # 训练结果
├── docs/                        # 📚 完整文档
│   ├── scenarios/               # 数据退化场景文档
│   ├── batch_testing/           # 批量测试文档
│   ├── siamese/                 # Siamese模式文档
│   ├── triplet/                 # Triplet模式文档
│   └── README.md                # 文档导航 ⭐
└── PROJECT_SUMMARY.md           # 项目详细总结
```

## 🚀 快速开始

### 1. 基础训练

```bash
# 默认配置训练（Scenario 2，稀疏采样）
python train_cross_battery.py

# 使用 Scenario 4（连续循环缺失）
python train_cross_battery.py --degradation_scenario scenario4
```

### 2. 批量参数测试

```bash
# 测试多个单调性权重
python test_monotonic_weights.py

# 查看详细使用说明
# 参考: docs/batch_testing/QUICK_START_WEIGHT_TEST.md
```

### 3. 模型推理

```bash
# 单电池推理
python inference.py

# 查看推理指南
# 参考: docs/guides/INFERENCE_QUICK_START.md
```

## 📖 文档导航

所有详细文档位于 `docs/` 目录，推荐阅读顺序：

1. **新手入门**
   - [项目总结](PROJECT_SUMMARY.md) - 项目整体架构和功能
   - [文档索引](docs/README.md) - 完整文档导航

2. **数据退化场景** (`docs/scenarios/`)
   - [数据增强总指南](docs/scenarios/DATA_AUGMENTATION_GUIDE.md) ⭐
   - [Scenario 1: 噪声增强](docs/scenarios/NOISE_AUGMENTATION_SUMMARY.md)
   - [Scenario 2: 稀疏采样](docs/scenarios/SPARSE_SAMPLING_MANUAL_CONTROL.md)
   - [Scenario 3: 随机缺失](docs/scenarios/RANDOM_MISSING_GUIDE.md)
   - [Scenario 4: 连续缺失](docs/scenarios/SCENARIO4_USAGE.md) ⭐ 最新

3. **批量测试** (`docs/batch_testing/`)
   - [批量测试总结](docs/batch_testing/BATCH_TESTING_SUMMARY.md) ⭐
   - [快速开始](docs/batch_testing/QUICK_START_WEIGHT_TEST.md)
   - [Scenario 4 批量测试](docs/scenarios/SCENARIO4_BATCH_TESTING.md) ⭐ 最新

4. **物理约束模式** (`docs/siamese/`, `docs/triplet/`)
   - Siamese模式: 一阶平滑性约束
   - Triplet模式: 二阶曲率约束

## 🔬 核心功能

### 数据退化场景

模拟真实场景下的传感器故障和数据缺失：

- **Scenario 1**: 噪声 + 随机丢弃
- **Scenario 2**: 规律稀疏采样（如定期HPPC测试）
- **Scenario 3**: 随机缺失（随机传感器读取失败）
- **Scenario 4**: 连续循环缺失（传感器系统故障）⭐ 最新

### 批量参数测试

自动化测试框架，支持：
- 多个单调性权重批量测试
- **Scenario 4 多参数批量测试** ⭐ 最新
  - 同时测试多个丢弃率 (drop_rates)
  - 同时测试多个缺失段数 (num_gaps)
  - 自动生成所有参数组合
  - 6子图综合可视化分析

示例：
```python
# test_monotonic_weights.py 配置
CONFIG = {
    'degradation_scenario': 'scenario4',
    'cycle_drop_rates': [0.2, 0.3, 0.5],      # 3个丢弃率
    'cycle_drop_num_gaps_list': [1, 2, 3],    # 3个缺失段数
    'monotonic_weights': [0.0, 0.3, 0.5],     # 3个权重
    # 总计: 3 × 3 × 3 = 27 个实验
}
```

### 物理约束

- **单调性约束**: 确保SOH随循环次数单调递减
- **平滑性约束**: 消除预测曲线的跳变
- **曲率约束**: 进一步约束二阶导数，获得更平滑的曲线

## 📊 实验结果

训练结果自动保存在 `results/` 目录：
- 训练历史曲线
- 预测对比图
- 最优/最差电池性能分析
- 批量测试对比图表

## 🛠️ 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA (推荐，用于GPU加速)

详细依赖见 `requirements.txt`

## 📝 最近更新

- **2025-12-23**: 添加 Scenario 4 (连续循环缺失) 完整功能
- **2025-12-23**: 添加 Scenario 4 多参数批量测试支持
- **2025-12-23**: 大规模项目清理和文档重组
- **2025-12-18**: 完善批量测试框架
- **2025-12-15**: 添加 Scenario 3 (随机缺失)

## 📚 参考文献

Sun, G., Liu, Y., & Liu, X. (2025). A method for estimating lithium-ion battery state of health based on physics-informed machine learning. Journal of Power Sources, 627, 235767.

## 📧 联系方式

如有问题或建议，请查看 [docs/README.md](docs/README.md) 获取详细文档。
