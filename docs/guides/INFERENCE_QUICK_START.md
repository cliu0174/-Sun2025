# 推理快速上手 - 超简单！

## 🎯 只需2步

### 第1步：修改配置

打开 `inference.py`，找到配置区域（大约第16-30行）：

```python
CONFIG = {
    # 模型设置
    'model_path': 'results/cross_battery/CNN_LSTM/model_checkpoint.pth',  # 👈 你的模型路径
    'data_dir': 'data/HUST data',                                          # 👈 数据目录
    'output_dir': 'inference_results',                                     # 👈 结果保存位置

    # 推理模式选择（三选一）
    'mode': 'single',  # 👈 改这里！'single', 'batch', 或 'test_set'

    # 单电池模式配置（mode='single'时使用）
    'single_battery_id': '4-3',  # 👈 改这里！要预测的电池ID

    # 批量模式配置（mode='batch'时使用）
    'batch_batteries': ['4-3', '6-2', '10-1', '3-8'],  # 👈 改这里！电池列表
}
```

### 第2步：运行

```bash
python inference.py
```

**就这么简单！** 🎉

---

## 📋 三种模式详解

### 模式1: 单电池预测（最常用）

预测一个电池的SOH曲线。

**配置**:
```python
'mode': 'single',
'single_battery_id': '4-3',  # 改成你要预测的电池ID
```

**输出**:
- `inference_results/4-3/4-3_predictions.png` - 4张图表
- `inference_results/4-3/4-3_results.json` - 详细结果

**使用场景**: 快速查看某个电池的预测效果

---

### 模式2: 批量预测

一次预测多个电池。

**配置**:
```python
'mode': 'batch',
'batch_batteries': ['4-3', '6-2', '10-1', '3-8'],  # 改成你的电池列表
```

**输出**:
- `inference_results/batch/individual/` - 每个电池的详细结果
- `inference_results/batch/summary_report.json` - 汇总统计
- `inference_results/batch/batch_results_summary.png` - 对比图表

**使用场景**: 对比多个电池的预测效果

---

### 模式3: 测试集预测

自动预测模型的所有测试集电池。

**配置**:
```python
'mode': 'test_set',
# 无需其他配置，自动读取 battery_split.json
```

**输出**: 与批量模式相同

**使用场景**: 评估模型在测试集上的整体表现

---

## 💡 使用示例

### 示例1: 预测单个电池

```python
# 1. 打开 inference.py
# 2. 修改配置:
CONFIG = {
    'model_path': 'results/cross_battery/CNN_LSTM/model_checkpoint.pth',
    'data_dir': 'data/HUST data',
    'output_dir': 'inference_results',
    'mode': 'single',                # 单电池模式
    'single_battery_id': '6-2',      # 预测 6-2 电池
    'batch_batteries': [],
}
# 3. 运行: python inference.py
```

### 示例2: 预测多个电池

```python
CONFIG = {
    'model_path': 'results/cross_battery/CNN_LSTM/model_checkpoint.pth',
    'data_dir': 'data/HUST data',
    'output_dir': 'inference_results',
    'mode': 'batch',                 # 批量模式
    'single_battery_id': '',
    'batch_batteries': ['4-3', '6-2', '10-1'],  # 预测这3个电池
}
```

### 示例3: 评估测试集

```python
CONFIG = {
    'model_path': 'results/cross_battery/CNN_LSTM/model_checkpoint.pth',
    'data_dir': 'data/HUST data',
    'output_dir': 'inference_results',
    'mode': 'test_set',              # 测试集模式
    'single_battery_id': '',
    'batch_batteries': [],
}
```

---

## 📊 结果说明

### 图表内容

**单电池图表**（4个子图）:
1. 时间序列 - 预测vs真实SOH随循环变化
2. 散点图 - 预测vs真实值（对角线=完美）
3. 误差曲线 - 误差随时间变化
4. 误差分布 - 误差直方图

**批量图表**（4个子图）:
1. MAE柱状图 - 各电池MAE对比
2. R²柱状图 - 各电池R²对比
3. 汇总散点图 - 所有电池预测vs真实
4. 总体误差分布 - 所有电池误差汇总

### 评估指标

| 指标 | 说明 | 越小越好/越大越好 |
|------|------|------------------|
| MAE | 平均绝对误差 | ↓ 越小越好 |
| RMSE | 均方根误差 | ↓ 越小越好 |
| MAPE | 平均百分比误差 | ↓ 越小越好 |
| R² | 决定系数 | ↑ 越大越好（接近1） |
| Max_Error | 最大误差 | ↓ 越小越好 |

---

## 🔧 常见问题

### Q: 如何查看可用的电池ID？

**方法1**: 查看数据目录
```bash
dir "data\HUST data\*.xlsx"
```

**方法2**: 在Python中查看
```python
from data_loaders.data_loader_hust import load_all_hust_batteries
batteries = load_all_hust_batteries(data_dir='data/HUST data')
print(list(batteries.keys()))
```

### Q: 找不到模型文件？

确认路径正确：
- 模型训练完成后保存在 `results/cross_battery/{模型类型}/model_checkpoint.pth`
- 例如：`results/cross_battery/CNN_LSTM/model_checkpoint.pth`

### Q: 推理速度慢？

- **减小batch_size**: 默认256，可以改小（但不影响结果）
- **使用GPU**: 如果有CUDA，会自动使用GPU加速
- **批量模式**: 比单独运行多次单电池快很多

### Q: 图表没有显示？

- 图表会自动保存为PNG文件
- 如果需要交互式显示，确保安装了matplotlib并配置正确
- 可以注释掉 `plt.show()` 只保存不显示

---

## ⚡ 提示与技巧

### 技巧1: 快速切换模式

在配置中只需改一行：
```python
'mode': 'single',    # 改成 'batch' 或 'test_set'
```

### 技巧2: 保存不同实验结果

```python
'output_dir': 'inference_results/experiment_001',  # 用不同目录名
```

### 技巧3: 对比不同模型

```python
# 实验1: CNN-LSTM模型
'model_path': 'results/cross_battery/CNN_LSTM/model_checkpoint.pth',
'output_dir': 'inference_results/cnn_lstm',

# 实验2: LSTM模型
'model_path': 'results/cross_battery/LSTM/model_checkpoint.pth',
'output_dir': 'inference_results/lstm',
```

---

## 📝 完整工作流程

```
1. 训练模型
   python train_cross_battery.py
   ↓
2. 修改 inference.py 配置
   设置模型路径和预测模式
   ↓
3. 运行推理
   python inference.py
   ↓
4. 查看结果
   打开 inference_results/ 目录
   ↓
5. 分析图表和指标
   根据需要调整模型或数据
```

---

## 🎓 进阶使用

如果需要更复杂的功能，可以直接修改 `inference.py` 中的函数，例如：

- 自定义图表样式
- 添加更多评估指标
- 导出不同格式的结果
- 集成到其他系统

**所有功能都在一个文件里，方便修改！**

---

**最后更新**: 2025-12-09
**难度**: ⭐ 超简单
**推荐指数**: ⭐⭐⭐⭐⭐
