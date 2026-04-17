# PI-CNNLSTM — 基于物理一致性约束的锂电池 SOH 估计

基于 **HUST 数据集**（77 块电池，14 维特征），使用 CNN-LSTM 网络结合物理约束损失函数估计锂离子电池健康状态（SOH）。

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

## 项目结构

```
1111-soh/
├── train_cross_battery.py      # 主训练脚本（核心入口）
├── inference.py                # 模型推理
├── compare_models.py           # 多模型横向对比
├── feature_extraction.py       # 14 维特征提取逻辑
│
├── models/
│   ├── cnn_lstm.py             # CNN-LSTM 模型
│   ├── baseline_models.py      # FNN / CNN / LSTM / GRU 等
│   ├── model_factory.py        # 模型工厂
│   └── physics_loss.py         # 物理约束损失函数
│
├── data_loaders/
│   └── data_loader_hust.py     # HUST 数据加载与窗口化
│
├── configs/models/             # 各模型 JSON 配置
│   └── cnn_lstm_config.json    # 当前默认配置
│
├── utils/
│   └── lr_schedulers.py        # 学习率调度器
│
├── data/HUST data/             # 原始数据（77 块电池 .csv）
└── results/                    # 训练输出
```

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

在标准 MSE 基础上加入两项物理先验：

- **软单调性约束**：惩罚 SOH 预测值上升，符合电池不可逆衰退的物理规律
- **边界约束**：将预测值约束在 [0, 1] 范围内

```json
"physics_constraints": {
    "enabled": true,
    "monotonic_weight": 0.1,
    "boundary_weight": 0.05
}
```

### 数据划分

按电池随机划分，**train / val / test = 60% / 20% / 20%**（约 46 / 15 / 16 块），种子固定（seed=42）确保可复现。

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
