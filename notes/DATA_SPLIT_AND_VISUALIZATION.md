# 数据集划分和可视化说明

## 数据集划分方式

### 当前实现（代码：src/data_loader.py:52-66）

```python
对于每个电池文件（B05_IC.csv, B06_IC.csv, B07_IC.csv）：
    读取数据
    前60%循环 → 训练集
    后40%循环 → 测试集

合并：
    训练集 = B05训练 + B06训练 + B07训练
    测试集 = B05测试 + B06测试 + B07测试
```

### 具体示例

假设每个电池有100个循环：

```
B05: 循环1-100
  训练集: 循环1-60
  测试集: 循环61-100

B06: 循环1-100
  训练集: 循环1-60
  测试集: 循环61-100

B07: 循环1-100
  训练集: 循环1-60
  测试集: 循环61-100

最终：
  训练集 = 180个样本（60+60+60）
  测试集 = 120个样本（40+40+40）
```

### 为什么这样划分？

1. **时序独立性**：每个电池的前60%用于训练，后40%用于测试
2. **防止数据泄露**：不会用未来的循环数据训练历史循环
3. **符合论文**：论文中提到60/40划分

---

## 各部分数据的用途

### 训练集（60%，180个样本）

#### Phase 1 - 主训练（BPINN-1）
```
作用：训练神经网络参数（权重W和偏置M）
损失函数：L = MSE(train) + λ₁·Physics(train)
训练2000个epoch
```

**具体过程**：
1. 每个batch: 输入特征 → 网络 → 预测SOH
2. 计算数据损失: |预测SOH - 真实SOH|²
3. 计算物理损失: 检查单调性约束
4. 反向传播，更新参数

#### Phase 2 - 二次训练（BPINN-2）
```
作用：计算训练集上的MSE和物理损失
损失函数：L = MSE(train) + λ₁·Physics(train) + λ₂·Physics(test)
训练400个iteration
```

**具体过程**：
1. 使用全部训练集数据（不分batch）
2. 计算训练集MSE（保持拟合性能）
3. 计算训练集物理损失（保持物理一致性）
4. 同时计算测试集物理损失（改善测试集单调性）

### 测试集（40%，120个样本）

#### Phase 1 - 评估
```
作用：评估模型泛化性能
指标：MAE, RMSE
不参与训练，只用于评估
```

#### Phase 2 - 在线优化
```
作用：计算测试集物理损失，用于微调模型
注意：不计算测试集MSE损失
目的：改善测试集的物理一致性（单调性）
```

---

## 可视化问题分析

### 问题：为什么图表不连续？

你提供的图片显示了**不连续的三段曲线**，这是因为：

#### 原因1：数据来自三个电池
```
测试集包含：
  B05: 循环61-100 (SOH: 0.85-0.70)
  B06: 循环61-100 (SOH: 0.90-0.75)
  B07: 循环61-100 (SOH: 0.80-0.65)

如果按原始顺序绘制：
  X轴: 0-40 (B05) | 40-80 (B06) | 80-120 (B07)
  会出现不连续！
```

#### 原因2：没有真实循环编号
```
当前实现：
  X轴 = 样本索引 (0, 1, 2, ..., 119)

应该是：
  X轴 = 真实循环编号 (需要从CSV中读取)
```

### 修正后的方案

#### 方案1：按SOH排序（已实现）✓

```python
# 将所有测试样本按SOH从高到低排序
sorted_indices = np.argsort(real_soh)[::-1]
real_soh_sorted = real_soh[sorted_indices]
pred_bpinn1_sorted = pred_bpinn1[sorted_indices]
pred_bpinn2_sorted = pred_bpinn2[sorted_indices]

# X轴：排序后的样本索引
# Y轴：SOH值
```

**优点**：
- 创建平滑的退化曲线
- 便于观察预测趋势
- 不需要修改数据集

**缺点**：
- X轴不是真实循环编号
- 混合了三个电池的数据

#### 方案2：分电池绘制（推荐，但需要修改数据集）

```python
# 需要在CSV中添加 battery_id 列
# 然后分别绘制：

图1: B05的测试集 (循环61-100 vs SOH)
图2: B06的测试集 (循环61-100 vs SOH)
图3: B07的测试集 (循环61-100 vs SOH)
```

**优点**：
- 符合论文方法（论文中是单电池结果）
- X轴是真实循环编号
- 更准确反映预测性能

**缺点**：
- 需要修改数据集，添加battery_id列
- 需要修改data_loader.py

---

## 当前图表的正确解读

### 修正后的图表（按SOH排序）

```
标题: SOH Prediction Comparison on Test Set
X轴: Test Sample Index (Sorted by SOH)
Y轴: SOH (%)

说明框:
  "Data from 3 batteries (B05, B06, B07) test sets
   Sorted by SOH for visualization"
```

**如何解读**：
1. **平滑性**：曲线越平滑，预测越稳定
2. **接近程度**：预测线越接近Real线，精度越高
3. **趋势一致性**：三条线应该都是单调下降的（SOH退化）

### 理想图表应该是什么样？

```
 100%│ ●━━━●━━━●━━━●━━━●          Real (黑色)
     │ ▲━━━▲━━━▲━━━▲━━━▲          BPINN-2 (橙色) ← 最接近
  90%│ ■━━━■━━━■━━━■━━━■          BPINN-1 (蓝色)
     │   ╲    ╲    ╲    ╲
  80%│    ╲    ╲    ╲    ╲         三条线都平滑下降
 SOH │     ╲    ╲    ╲    ╲        BPINN-2比BPINN-1更接近Real
  70%│      ╲    ╲    ╲    ╲
     │       ╲    ╲    ╲    ╲
  60%│        ╲    ╲    ╲    ╲
     └────────────────────────────
       0    20    40    60    80
              Sample Index
```

---

## 如何实现论文中的图表？

### 方法1：修改数据集（推荐）

**步骤**：

1. **在MATLAB中提取IC特征时**，记录battery_id和cycle:
   ```matlab
   results_table = table(battery_id, cycle, y_h, V_h, k_l, k_r, t_G, Q_G, SOH);
   ```

2. **修改data_loader.py**，保留battery_id和cycle:
   ```python
   # 读取时保留额外列
   df = pd.read_csv(file_path)
   battery_id = df['battery_id']
   cycle = df['cycle']
   ```

3. **新增绘图函数**，按电池分别绘制:
   ```python
   def plot_soh_per_battery_with_cycles(results, battery_ids, cycles):
       for battery_id in unique(battery_ids):
           mask = (battery_ids == battery_id)
           plt.plot(cycles[mask], real_soh[mask], label=f'{battery_id} Real')
           plt.plot(cycles[mask], pred[mask], label=f'{battery_id} Pred')
   ```

### 方法2：使用当前实现（临时方案）

**当前修改**已经实现：
- ✓ 按SOH排序，创建平滑曲线
- ✓ 添加说明文字，解释数据来源
- ✓ 可以直观对比BPINN-1和BPINN-2

**使用建议**：
1. 用于快速对比整体预测性能
2. 观察BPINN-2是否比BPINN-1更接近真实值
3. 检查预测曲线的平滑性

**不足**：
- ❌ X轴不是真实循环编号
- ❌ 混合了多个电池的数据
- ❌ 不能准确反映时序性能

---

## 总结

### 数据流向图

```
原始数据 (3个电池 × 100循环)
    ↓
划分 (每个电池60/40)
    ↓
    ├─→ 训练集 (180样本) → Phase 1: 训练网络参数
    │                    → Phase 2: 计算MSE(train)+Physics(train)
    │
    └─→ 测试集 (120样本) → Phase 1: 评估MAE/RMSE
                         → Phase 2: 计算Physics(test)
```

### 当前图表说明

- **X轴**：测试样本索引（按SOH排序）
- **Y轴**：SOH值
- **数据**：三个电池的测试集混合
- **用途**：快速对比BPINN-1 vs BPINN-2的整体性能

### 改进方向

如果想要更准确的可视化：
1. ✅ 短期：使用当前排序后的图表
2. 🔧 长期：修改数据集，添加battery_id和cycle列
3. 📊 最终：实现按电池分别绘制真实循环vs SOH

---

## 参考

- 数据加载代码：[src/data_loader.py](../src/data_loader.py:40-94)
- 绘图函数：[src/utils.py](../src/utils.py:271-340)
- 论文：Sun et al. (2025) - Section 3.2 Data Preparation
