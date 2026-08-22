# Exp-12 电池异常检测模块：完整归档文档

> **文档用途**：项目归档 + 论文第四章参考资料 + 后续 Agent 接手指引
> **撰写日期**：2026-05-27
> **负责人**：liuchang2262@gmail.com
> **对应代码分支**：paper-framework

---

## 一、背景与动机

### 1.1 研究问题

本项目的核心任务是 **SOH（State of Health，电池健康度）估计**，使用 PI-MS-CNN-LSTM 模型在 HUST 数据集（77 块 LFP 电池，14 维特征）上实现准确预测。

Exp-12 是在此基础上提出的**扩展性验证实验**：

> **能否复用已训练的 SOH 预测模型，在不进行任何额外训练的情况下，实现对电池异常退化的自动检测？**

这一思路被称为"**零额外成本异常感知**"——模型权重固定，仅利用推理阶段产生的预测信号来判断当前电池是否出现异常。

### 1.2 核心假设

模型在正常电池数据上训练后，学到了"正常特征轨迹 → SOH"的映射关系。当异常电池的特征输入时：

- 预测的 SOH 值会偏离正常退化规律（比应有的下降更快、更突然）
- 输入特征本身会偏离训练集统计分布
- 预测序列的变化速率会出现异常

这三类信号均可在不修改模型的前提下直接提取。

### 1.3 与主线实验的关系

Exp-12 是**附加验证实验**，不是论文核心贡献。核心贡献是：

- 多尺度 CNN-LSTM 架构（Exp-09c）的 SOH 估计精度
- 部分监督场景下的性能
- 物理约束的辅助价值

若 Exp-12 最终检测性能不理想（AUC < 0.65），可从论文中删除而不影响主线。若性能良好，可在第四章末尾作为"应用扩展"补充约半页。

---

## 二、方法路线全貌

### 2.1 总体流程

```
已训练的 SOH 预测模型（权重固定）
        │
        ├─── 正常电池特征序列  →  模型推理  →  clean_predictions
        │
        └─── 异常注入后特征序列 →  模型推理  →  fault_predictions
                    ↑
              [5 类退化场景注入]
        
        ↓
对 fault_predictions 计算 5 维异常分数（逐时间步）
        ↓
设定报警阈值（FPR=5% 对应的第 95 百分位）
        ↓
评估：AUC / Det@FPR5% / det_delay
```

### 2.2 方案演进历史

| 阶段 | 方案 | 问题 | 结果 |
|------|------|------|------|
| **Legacy 模式** | 跨电池注入：将"供体电池"末期特征混入测试电池 | 供体索引 Bug：早期故障时刻实际使用的是供体的健康早期特征（无效注入）| AUC ≈ 0.5（随机水平）|
| **Bug 修复后 Legacy** | 修复 `_donor_terminal_sequence()`，始终取供体末期 20% 特征 | 注入后模型对特征差异不敏感，预测结果几乎不变 | AUC 仍 ≈ 0.5 |
| **Self-trajectory 模式** ✅ | 跨时间注入：用同一电池自身的末期状态注入至故障点 | combined 信号被 input_zscore 污染 | trajectory_deviation 单信号 AUC 0.65–0.75 ✅ |

**根本原因分析**：Legacy 模式失败是因为不同电池之间的特征分布差异（个体差异）远小于时间维度上的特征演变（老化差异）。模型见过各种电池的正常特征，因此对跨电池混入的特征不敏感。换用同一电池自身末期特征后，时间维度的特征偏移真实反映了"提前老化"，模型的预测差异显著。

---

## 三、五类退化场景

### 3.1 设计原则

所有场景均采用 **self-trajectory（同电池轨迹重排）** 方式：

- 无需真实失效电池数据
- 保留电池个体特性（同一块电池的不同时间段）
- 通过将"未来"状态提前注入至故障点，模拟各类退化加速

故障注入位置统一设置在 **50% 生命周期处**（`fault_cycle = int(T * 0.5)`）。

严重度分三级：`mild`（轻度）/ `moderate`（中度）/ `severe`（重度）。

### 3.2 场景详细说明

---

#### A1：突发老化（Sudden Aging）

**物理背景**：机械冲击、过充/过放、短路等突发性事件造成不可逆内部损伤，电池容量在某一时刻突然大幅降低，之后以新的（更低）基线继续正常退化。

**SOH 曲线特征**：某循环后出现 SOH 骤降台阶（3%–8%），之后恢复平稳衰减。

**注入实现**：

```python
def inject_self_jump_drop(features, targets, fault_cycle, severity):
    # 根据 severity 确定目标 SOH 落差
    soh_drop = {'mild': 0.03, 'moderate': 0.05, 'severe': 0.08}[severity]
    target_soh = targets[fault_cycle] - soh_drop
    
    # 在后续序列中找到 SOH 降至目标值的位置 s0
    for s in range(fault_cycle + 1, T):
        if targets[s] <= target_soh:
            s0 = s; break
    
    # 跳过中间段：fault_cycle 之前正常 + s0 之后末期
    corrupted = concat(features[:fault_cycle], features[s0:])
```

**检测难度**：⭐ 容易（SOH 曲线突变，trajectory_deviation 立即感知）

**实测 AUC**（severe）：**~0.73**，verdict: **GOOD**

---

#### A2：容量拐点（Knee-point）

**物理背景**：电池容量衰减在正常使用时呈近似线性，但在活性锂不可逆损失积累至临界点时，退化速率急剧加快，呈现"膝盖弯曲"形态。常见于大量循环后的老化末期。

**SOH 曲线特征**：前期缓慢，拐点后斜率突然变陡，加速下滑。

**注入实现**：

```python
def inject_self_knee_sampling(features, fault_cycle, severity):
    step = {'mild': 2, 'moderate': 3, 'severe': 4}[severity]
    
    # 故障点后进行跳采样：每 step 个真实循环取一个点
    # 使得单位伪循环内 SOH 下降更快，模拟退化加速
    tail_idx = np.arange(fault_cycle, T, step)
    corrupted = concat(features[:fault_cycle], features[tail_idx])
```

**检测难度**：⭐⭐ 较易（曲线斜率有明显转折）

**实测 AUC**（severe）：**~0.70**，verdict: **WEAK/GOOD**

---

#### A3：簇内不均衡（Cell Imbalance）

**物理背景**：电池组由多个单体串联/并联。若某块单体因温度梯度、制造偏差或局部过载而加速老化，整体容量受到短板效应拖累，SOH 持续偏低于理论预期。

**SOH 曲线特征**：从故障点开始，SOH 曲线持续低于正常轨迹，差距随时间**线性扩大**（始终偏低，不会偏高——因为混入的是"更老的未来状态"，必然比当前状态更低）。

**注入实现**：

```python
def inject_self_imbalance_drift(features, fault_cycle, severity):
    max_alpha = {'mild': 0.25, 'moderate': 0.40, 'severe': 0.55}[severity]
    max_offset = int({'mild': 0.10, 'moderate': 0.18, 'severe': 0.25}[severity] * T)
    
    for t in range(fault_cycle, T):
        progress = (t - fault_cycle) / (T - fault_cycle)
        alpha_t = max_alpha * progress          # 线性增长的混合比例
        future_idx = t + int(max_offset * progress)
        future_feat = features[min(future_idx, T-1)]  # 超出时末期外推
        corrupted[t] = (1 - alpha_t) * features[t] + alpha_t * future_feat
```

**检测难度**：⭐⭐⭐ 中等（变化缓慢但持续，trajectory_deviation 能感知累积偏离）

**实测 AUC**（severe，trajectory_deviation 信号）：**~0.654**

---

#### A4：析锂（Lithium Plating）

**物理背景**：低温或大电流快充时，锂离子嵌入负极速度不及析出速度，导致金属锂在负极表面沉积（"析锂"）。析出的金属锂不可逆，既造成活性锂损失（容量突降），又可能刺穿隔膜（引发短路）。特征是**台阶突降 + 之后加速衰减**的双重形态。

**SOH 曲线特征**：故障点处出现小台阶（2%–5.5%），台阶后继续加速退化。

**注入实现**：

```python
def inject_self_lithium_plating(features, targets, fault_cycle, severity):
    # 先找台阶跳跃位置（比 sudden_drop 台阶小）
    soh_drop = {'mild': 0.02, 'moderate': 0.035, 'severe': 0.055}[severity]
    step = {'mild': 2, 'moderate': 2, 'severe': 3}[severity]
    
    # 跳至 s0（台阶），然后跳采样（加速）= A1 + A2 的组合
    tail_idx = np.arange(s0, T, step)
    corrupted = concat(features[:fault_cycle], features[tail_idx])
```

**检测难度**：⭐⭐ 较易（突变 + 加速双重特征）

**实测 AUC**（severe）：**~0.736**，verdict: **GOOD**

---

#### A5：内阻增长（Internal Resistance Rise）

**物理背景**：随使用次数增加，SEI（固体电解质界面）膜持续增厚，电解液副反应产物积累，电极颗粒接触电阻升高，综合导致电池内阻缓慢增大。内阻增大使充放电效率降低，可用容量缓慢减少，退化曲线斜率**二次方式**加大（早期几乎不可察，后期加速明显）。

**SOH 曲线特征**：故障后曲线斜率极缓慢变陡，无折点无阶跃，早期与正常轨迹几乎重合。

**注入实现**：

```python
def inject_self_resistance_rise(features, fault_cycle, severity):
    max_alpha = {'mild': 0.20, 'moderate': 0.35, 'severe': 0.55}[severity]
    offset = int({'mild': 0.08, 'moderate': 0.15, 'severe': 0.22}[severity] * T)
    
    for t in range(fault_cycle, T):
        progress = (t - fault_cycle) / (T - fault_cycle)
        alpha_t = max_alpha * progress ** 2     # 二次增长：早期慢，后期快
        future_feat = features[min(t + offset, T-1)]
        corrupted[t] = (1 - alpha_t) * features[t] + alpha_t * future_feat
```

**检测难度**：⭐⭐⭐⭐ 最难（早期信号极弱，trajectory_deviation 才能感知）

**实测 AUC**（severe，trajectory_deviation 信号）：**~0.668**

---

### 3.3 五类场景对比

| 场景 | 注入方式 | SOH 变化形态 | 检测难度 | 实测 AUC（traj） |
|------|---------|------------|---------|----------------|
| A1 突发老化 | 截断+拼接末期 | 骤降台阶（大） | ⭐ | 0.745 |
| A2 容量拐点 | 跳采样加速 | 斜率突增 | ⭐⭐ | 0.697 |
| A3 簇内不均衡 | 线性混入未来状态 | 持续线性偏低 | ⭐⭐⭐ | 0.654 |
| A4 析锂 | 截断+拼接+跳采样 | 台阶（小）+加速 | ⭐⭐ | 0.736 |
| A5 内阻增长 | 二次混入未来状态 | 斜率极缓慢加大 | ⭐⭐⭐⭐ | 0.668 |

---

## 四、五个检测信号

检测信号由 `evaluation/anomaly_detection.py` 中的 `compute_anomaly_scores()` 函数计算，对每个时间步 t 输出一个分数，分数越高表示当前时刻越可疑。

### 4.1 信号详细说明

---

#### 信号 1：input_zscore（输入特征偏离度）

**计算方式**：

```python
zscore_all = |features[t] - train_mean| / train_std  # (14,)
input_zscore[t] = max(zscore_all)                      # 取最大维度的偏离
```

**含义**：与训练集统计分布相比，当前输入特征偏离了多少个标准差。偏离越大，当前电池状态越不像"正常"。

**局限**：正常电池在生命末期，特征值自然偏离训练集均值（因为训练时末期样本少），导致该信号在正常时段也较高，形成**系统性假阳性**。这是 combined 信号失效的主要原因。

**实测 AUC**：0.56–0.73（各场景差异大，末期容易误报）

---

#### 信号 2：mono_violation（单调违规）

**计算方式**：

```python
soh_diff = predictions[t] - predictions[t-1]
mono_violation[t] = max(soh_diff, 0)   # 只记录上升量
```

**含义**：电池健康度符合热力学不可逆原理，只能单调递减。若模型预测出 SOH 上升，说明当前输入特征与正常退化规律严重不符。

**局限**：正常预测也存在随机抖动（小幅上升），信号噪声较大；对渐进型异常（A3、A5）不敏感。

**实测 AUC**：≈ 0.5（接近随机，区分能力弱）

---

#### 信号 3：rate_anomaly（退化速率突变）

**计算方式**：

```python
soh_rate = diff(predictions)  # 相邻步差分
for t in range(window, T):
    local_mean = mean(soh_rate[t-window:t])
    local_std  = std(soh_rate[t-window:t])
    rate_anomaly[t] = |soh_rate[t] - local_mean| / local_std
```

**含义**：当前退化速率与最近 20 步的局部均值相比，偏离了多少个局部标准差。检测退化速度的突变。

**局限**：窗口 20 步时局部统计噪声较大；对渐进型场景（A3、A5）不敏感；与 drop_anomaly 有功能重叠。

**实测 AUC**：≈ 0.5

---

#### 信号 4：drop_anomaly（异常骤降）

**计算方式**：

```python
drop_rate = max(-diff(predictions), 0)   # 只保留下跌量
for t in range(window, T):
    local_mean = mean(drop_rate[t-window:t])
    local_std  = std(drop_rate[t-window:t])
    drop_anomaly[t] = max(0, (drop_rate[t] - local_mean) / local_std)
```

**含义**：当前 SOH 下跌幅度是否显著超过近期局部基准。专门检测异常加速下跌，与 mono_violation（检测异常上升）互补。

**局限**：对渐进型异常同样不敏感；局部窗口短时噪声大。

**实测 AUC**：≈ 0.5

---

#### 信号 5：trajectory_deviation（轨迹偏差）⭐ 最有效信号

**计算方式**：

```python
for t in range(window, T):
    x = np.arange(window)
    y = predictions[t-window:t]          # 最近 window 步的预测值
    slope, intercept = polyfit(x, y, 1)  # 线性拟合当前趋势
    expected = intercept + slope * window # 外推"下一步应该在哪"
    trajectory_deviation[t] = |predictions[t] - expected|
```

**含义**：用过去 20 步的 SOH 预测值拟合一条趋势线，外推"按当前趋势，下一步应该落在哪里"，然后计算实际值与预期值的绝对偏差。

**直觉**：如果电池一直在以某个速率正常退化，预测曲线应该平滑延续这个趋势。一旦出现突变（A1、A4）或累积偏移（A3、A5），实际值就会偏离趋势线外推值，该信号立即感知。

**为什么最有效**：
- 突发型（A1、A4）：故障点处 SOH 骤降，实际值远低于趋势外推值，偏差瞬间跳大
- 渐进型（A3、A5）：偏离虽缓慢，但每步都在偏离，累积后偏差持续增大
- 正常时段：SOH 平稳退化，趋势外推误差小，分数保持低位

**实测 AUC**：**0.654–0.745**（全部 5 类场景均有效）

---

#### 综合信号：combined / combined_traj / combined_weighted

```python
# 原始 combined（有问题）
combined = max(normalize(input_zscore),
               normalize(mono_violation),
               normalize(rate_anomaly),
               normalize(drop_anomaly),
               normalize(trajectory_deviation))

# combined_traj（推荐，直接使用最强信号）
combined_traj = normalize(trajectory_deviation)

# combined_weighted（备选，主辅结合）
combined_weighted = 0.7 * normalize(trajectory_deviation) + 0.3 * normalize(input_zscore)
```

**combined 失效原因**：`max()` 策略下，任一信号偏高都会拉高综合分。input_zscore 在正常末期自然偏高，导致 combined 在正常时段误报率飙升，AUC 反而低于随机（< 0.5）。

**推荐使用 combined_traj** 作为最终检测信号。

---

### 4.2 信号效果汇总

基于 Exp-09c r=0.3 seed=7 severe 模式的实测 AUC：

| 场景 | input_zscore | mono_violation | rate_anomaly | drop_anomaly | **trajectory_deviation** | combined |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| sudden_aging | 0.731 | 0.549 | 0.500 | 0.507 | **0.745** | 0.729 |
| knee_point | 0.562 | 0.482 | 0.505 | 0.520 | **0.697** | 0.584 |
| imbalance | 0.593 | 0.553 | 0.500 | 0.502 | **0.654** | 0.397 ❌ |
| li_plating | 0.679 | 0.481 | 0.506 | 0.520 | **0.736** | 0.696 |
| resistance_rise | 0.601 | 0.565 | 0.498 | 0.493 | **0.668** | 0.425 ❌ |

---

## 五、异常识别完整流程

### 5.1 离线阶段（训练，一次完成）

1. 用正常电池数据训练 PI-MS-CNN-LSTM，获得权重文件 `model.pth`
2. 记录训练集各特征的均值（`feat_mean`）和标准差（`feat_std`），用于后续 z-score 计算

### 5.2 在线监测阶段（推理，逐循环进行）

```
新的充电数据到达
        │
        ↓
Step 1：模型推理
  输入：当前循环的 14 维特征
  输出：当前 SOH 预测值（append 到历史序列中）
        │
        ↓
Step 2：计算 5 维异常分数
  input_zscore     ← 特征 z-score（与训练集统计量对比）
  mono_violation   ← SOH 是否上升（违反单调性）
  rate_anomaly     ← 退化速率是否突变（局部 z-score）
  drop_anomaly     ← 下跌幅度是否异常（局部 z-score）
  trajectory_deviation ← 实际值与趋势外推值之差（核心信号）
        │
        ↓
Step 3：对比报警阈值
  阈值来源：正常时段分数的第 95 百分位（对应 FPR=5%）
  判定规则：
    单点判定：当前分数 > 阈值 → 标记为异常
    连续确认（工程优化）：连续 N 次 > 阈值 → 确认为异常
        │
        ↓
Step 4：输出检测结果
  - 是否报警（bool）
  - 各信号的当前分数
  - 已连续超限次数
```

### 5.3 报警线确定方式（FPR=5%）

```python
# 用正常时段（故障注入点之前）的分数确定阈值
normal_scores = scores[:fault_cycle]
threshold = np.percentile(normal_scores, 95)  # 第 95 百分位

# 这保证了：正常时段中最多 5% 的时刻会误超阈值（FPR ≤ 5%）
```

### 5.4 连续确认机制（降低误报）

```python
n_confirm = 3   # 连续 3 次超线才确认报警
count = 0
for t in range(T):
    if score[t] > threshold:
        count += 1
        if count >= n_confirm:
            alarm = True; break
    else:
        count = 0   # 中断则重新计数
```

**效果**：单点误报率 5%，连续 3 次误报概率为 0.05³ = 0.012%（降低约 400 倍），代价是检测延迟增加 N 个循环。

---

## 六、评价指标

### 6.1 AUC（Area Under ROC Curve）

**定义**：随机抽取一个异常时刻和一个正常时刻，前者的异常分数高于后者的概率。

**计算**：对比 `scores_clean`（正常时段分数）和 `scores_fault`（故障后时段分数），用 sklearn 的 `roc_auc_score` 计算。

**解读**：
- AUC = 0.5：模型无区分能力（等同于随机猜测）
- AUC = 0.7：实验中 sudden_aging 的水平，有实用价值
- AUC = 1.0：完美区分（理论上限）

**判断标准**（本实验）：> 0.70 GOOD | 0.58–0.70 WEAK | < 0.58 FAIL

### 6.2 Det@FPR5%（Detection Rate at FPR ≤ 5%）

**定义**：在误报率（FPR）不超过 5% 的约束下，能正确检出的真实异常时刻比例。

**计算过程**：
1. 画 ROC 曲线（横轴=FPR，纵轴=TPR）
2. 找到 FPR ≤ 5% 区间内最右边的点
3. 读取该点对应的 TPR 值

**工程含义**：每 100 个正常充电循环最多允许 5 次误报警，此时能检出多少真实故障时刻？

**解读参考**：
- 5%：随机猜测水平（无效）
- 10%：比随机高 2 倍（有一定参考价值）
- 15%：sudden_aging 实测水平，比随机高 3 倍（工程上有意义）

### 6.3 det_delay（检测延迟，单位：循环数）

**定义**：从故障注入点（fault_cycle）到首次触发报警所经历的充电循环数。

**计算**：

```python
# 用 FPR=5% 对应的阈值，从 fault_cycle 开始向后扫描
for t in range(fault_cycle, T):
    if score[t] >= threshold_fpr5:
        det_delay = t - fault_cycle
        break
# 若整段均未检出，则 det_delay = -1
```

**解读**：
- 越小越好（响应越快）
- = -1：整段均未检出（最差情况）
- 突发型场景（A1、A4）通常在数十循环内报警
- 渐进型场景（A3、A5）延迟更长，甚至未能检出

---

## 七、实验结果

### 7.1 主要数值结果

**配置**：self_trajectory 注入模式，severe 严重度，4 模型 × 3 ratios × 3 seeds = 36 次训练

**各场景 combined AUC 对比（4 模型，r=0.3~1.0）**：

| 场景 | Eneg1 | E0 | A1 | Exp09c |
|------|:---:|:---:|:---:|:---:|
| sudden_aging | 0.732 | 0.734 | 0.736 | 0.733 |
| knee_point | 0.591 | 0.593 | 0.604 | 0.598 |
| imbalance | 0.407 | 0.428 | 0.404 | 0.421 |
| li_plating | — | — | — | ~0.70 |
| resistance_rise | — | — | — | ~0.43 |

**关键观察**：
- 4 个模型 AUC 差距 < 0.01，说明检测能力来自方法本身，而非模型架构
- ratio=1.0/0.5/0.3 的检测能力几乎一致（差距 ±0.005）
- imbalance/resistance_rise 的 combined AUC < 0.5，是被 input_zscore 拖累的结果

### 7.2 Det@FPR5% 实测（severe）

| 场景 | 各模型均值 |
|------|----------|
| sudden_aging | ~15%（是随机水平 3 倍）|
| knee_point | ~10%（是随机水平 2 倍）|
| imbalance | ~8.6%（接近随机）|

### 7.3 trajectory_deviation 单信号结果（Exp09c r=0.3 seed=7）

| 场景 | AUC | 判断 |
|------|:---:|:---:|
| sudden_aging | **0.745** | GOOD ✅ |
| knee_point | **0.697** | WEAK |
| imbalance | **0.654** | WEAK |
| li_plating | **0.736** | GOOD ✅ |
| resistance_rise | **0.668** | WEAK |

**结论**：trajectory_deviation 是目前唯一跨场景稳定有效的信号，5 类场景 AUC 均 ≥ 0.65。

---

## 八、已识别问题与改进方向

### 问题 1：combined 信号定义不合理（已知，影响中等）

**现象**：combined = max(5 信号) 对渐进场景（A3、A5）AUC < 0.5

**根因**：input_zscore 在正常末期自然偏高，max() 被其"污染"

**修复方案**：
- 方案 A（已实现）：直接用 `combined_traj = trajectory_deviation`
- 方案 B：`0.7 × traj + 0.3 × input_zscore` 加权（`combined_weighted`，已实现）
- 方案 C：让 input_zscore 不参与 combined

### 问题 2：window_size = 20 待优化

当前 trajectory_deviation 使用 20 步窗口。窗口太短时局部拟合噪声大，太长时响应延迟增加。建议测试 window_size = 10 / 20 / 30 的差异。

### 问题 3：rate_anomaly / drop_anomaly / mono_violation 无实际贡献

三个信号在所有场景下 AUC ≈ 0.5，既不能单独用于检测，参与 combined 也会引入噪声。可考虑从 combined 中移除，仅保留 trajectory_deviation 和 input_zscore（有限参与）。

### 问题 4：渐进型场景检测能力有限（根本问题）

A3（imbalance）和 A5（resistance_rise）的 trajectory_deviation AUC 约 0.65–0.67，属于"有效但不强"。根本原因是渐进型注入的信号幅度本身就很微弱，模型预测的随机误差与异常信号在量级上接近。改进方向：增大 window_size、使用自适应基线而非固定线性外推。

### 改进优先级

| 优先级 | 改进项 | 预计效果 |
|-------|-------|---------|
| P0 | 将最终报告改为使用 `trajectory_deviation` 代替 `combined` | 立即消除渐进场景 AUC < 0.5 问题 |
| P1 | 测试 window_size 从 20 调至 10/30 | 可能提升 0.01–0.03 AUC |
| P2 | 引入连续 N 次确认机制（N=3） | 工程实用性大幅提升，误报率降低 400 倍 |
| P3 | 对最佳模型（Exp09c）做异常分数时序可视化 | 论文图表，直观展示信号在故障前后的跳变 |

---

## 九、论文写法建议

### 9.1 章节定位

建议在论文第四章末尾增加一节：**"4.X 基于 SOH 预测信号的零训练代价异常检测"**

约占篇幅：0.5–1 页正文 + 1 张图（异常分数时序图）+ 1 张结果表

### 9.2 可写入论文的核心发现

1. **"零额外训练成本"思路被验证**：不需要真实失效数据，不需要额外训练，复用 SOH 预测模型即可实现异常感知。

2. **trajectory_deviation 是核心贡献**：基于"局部趋势外推偏差"的检测信号，对 5 类场景 AUC 均 ≥ 0.65，是目前最稳定有效的检测维度。

3. **数值结果**：突发型场景（sudden_aging、li_plating）AUC ~0.73，Det@FPR5% ~15%（随机基准的 3 倍）；渐进型场景 AUC ~0.65，仍优于随机。

### 9.3 限制条件（必须在论文中说明）

1. 对渐进型异常（A3、A5）的检测能力相对有限（AUC ~0.65，Det@FPR5% 接近随机）
2. 所有异常均为合成数据，未在真实失效电池上验证
3. 物理约束对异常检测能力几乎无贡献（4 模型 AUC 差距 < 0.01）
4. 当前方案只能检测"比正常退化更快"的异常，无法识别"假性恢复"等复杂场景

### 9.4 建议图表

- **图 1**：5 类退化场景 SOH 示意图（已生成：`docs/exp12_degradation_scenarios_self_trajectory.png`）
- **图 2**（推荐新增）：某块测试电池的 trajectory_deviation 分数时序图，标注故障注入点和报警线
- **表 1**：5 类场景下各信号 AUC 汇总表（已有数据）

---

## 十、关键代码入口

| 文件 | 功能 |
|------|------|
| `evaluation/anomaly_detection.py` | 全部注入函数、评分函数、评估函数 |
| `experiments/run_exp12_anomaly_detection.py` | 完整批量实验（4 模型 × 3 ratios × 3 seeds） |
| `experiments/mvp_anomaly_verify.py` | 快速验证脚本（加载已有 checkpoint，无需重新训练） |
| `docs/plot_degradation_scenarios_self_trajectory.py` | 生成 5 类场景 SOH 示意图 |
| `docs/anomaly_detection_demo.html` | 交互式演示页面（动态展示各场景 SOH 曲线+报警） |

### 运行 MVP 验证（不需要重新训练）

```bash
# self_trajectory 模式（推荐）
python experiments/mvp_anomaly_verify.py --injection_mode self_trajectory --severity severe

# 指定具体 checkpoint
python experiments/mvp_anomaly_verify.py \
    --injection_mode self_trajectory \
    --ckpt_dir experiments/exp12_anomaly_detection/E0_baseline/ratio0p5/seed929
```

### 输出格式说明

```
场景名                inzscore  mono  rate  drop  traj  combined  verdict
  → Det@FPR5%         xx.x%    ...
  → delay(cycles)        12    ...
```

---

## 十一、当前状态（2026-05-27）

| 项目 | 状态 |
|------|------|
| 注入函数实现 | ✅ 5 类 self-trajectory 注入函数全部完成 |
| 检测信号实现 | ✅ 5 维信号 + 3 种 combined 方案 |
| MVP 验证脚本 | ✅ 支持 AUC / Det@FPR5% / det_delay 三指标输出 |
| 服务器全量实验 | ✅ 已完成（36 次训练），结果记录于 `exp12_results_20260521.md` |
| combined 修复 | ⚠️ 代码已实现 combined_traj，但全量汇总表尚未用新信号重跑 |
| det_delay 数据 | ⚠️ 代码已支持输出，但服务器结果未回传，本地无缓存 checkpoint |
| 论文写作 | ⏳ 待决策是否加入主线 |
