# Exp-12 异常检测模块：工作进度总结

> **写给下一个 Agent 的交接文档**
> 最后更新：2026-05-21
> 当前负责人：liuchang2262@gmail.com

---

## 一、Exp-12 是什么

**目标**：验证已训练的 PI-MS-CNN-LSTM 模型能否在**零额外训练成本**下，通过推理时产生的异常信号（特征偏离、预测速率突变、物理约束违规）检测电池簇中的退化异常电芯。

**核心假设**：模型在正常数据上训练后，学到了"正常特征轨迹 → SOH"的映射；当异常特征输入时，预测值或特征统计量会偏离正常范围，从而触发检测信号。

**不需要真实失效电池**：所有"异常"数据均为合成——用真实 HUST 数据的特征序列按照 5 种物理退化场景的逻辑人工注入。

---

## 二、实验设计

### 2.1 检测流程

```
训练（一次，固定权重）
    ↓
对每块测试电池生成两条序列：
    clean_features   → 模型推理 → clean_preds
    corrupt_features → 模型推理 → fault_preds
    ↓
计算 4 个异常分数（逐时间步）：
    input_zscore   = 特征偏离训练集分布
    mono_violation = 预测 SOH 上升（违反物理）
    rate_anomaly   = 预测速率突变（z-score）
    combined       = max(三者归一化后)
    ↓
评估：ROC-AUC / Det@FPR5% / 检测延迟
```

### 2.2 五种退化场景

| 场景 | 退化类型 | 实现方式 | 需要供体 |
|------|---------|---------|---------|
| `sudden_aging` | 突发加速老化 | 混入供体末期特征（固定 α） | ✅ |
| `knee_point` | 容量拐点突破 | 混入自身末期特征（固定 α） | ❌ |
| `imbalance` | 簇内不均衡加剧 | 混入供体末期特征（α 线性增长） | ✅ |
| `li_plating` | 析锂台阶突降 | 阶跃 step_α + 持续 post_α | ✅ |
| `resistance_rise` | 内阻渐进增长 | 向自身末期漂移（二次增长） | ❌ |

严重度：mild(0.2) / moderate(0.4) / severe(0.7)，故障注入位置：50% 生命周期

### 2.3 对比模型（2×2 矩阵）

```
           无物理约束      有物理约束（软单调）
单路CNN     Eneg1           E0
多尺度CNN   A1              Exp09c★（最终方案）
```

### 2.4 关键文件

```
evaluation/anomaly_detection.py         ← 注入函数 + 评分 + 评估
experiments/run_exp12_anomaly_detection.py  ← 完整实验脚本（4 模型 × 2 ratios × 3 seeds）
experiments/mvp_anomaly_verify.py       ← MVP 快速验证（无需重新训练，加载已有 checkpoint）
docs/exp12_degradation_scenarios.png    ← 5 种场景 SOH 曲线示意图
```

---

## 三、已发现并修复的 Bug

### Bug：供体特征索引错误（已修复，commit a31229d）

**问题**：`inject_cell_failure` / `inject_imbalance_escalation` / `inject_lithium_plating` 使用了以下索引逻辑：

```python
donor_idx = min(T_donor - 1 - (T - 1 - t), T_donor - 1)
donor_idx = max(0, donor_idx)
```

当供体电池寿命 < 测试电池剩余寿命时（`T_donor < T - fault_cycle`），早期故障时刻的 `donor_idx` 被 clamp 到 0，即**使用供体的健康早期特征**而非末期退化特征。这等于向正常电池注入另一块正常电池的早期数据，不产生任何可检测的异常。

**修复**：新增 `_donor_terminal_sequence()` 辅助函数，始终取供体末期 20% 的特征循环填充：

```python
def _donor_terminal_sequence(donor_features, n_cycles):
    T_donor = len(donor_features)
    terminal_start = max(0, int(T_donor * 0.80))
    terminal = donor_features[terminal_start:]
    if len(terminal) >= n_cycles:
        return terminal[-n_cycles:]
    else:
        repeats = (n_cycles // len(terminal)) + 1
        return np.tile(terminal, (repeats, 1))[-n_cycles:]
```

---

## 四、当前实验结果

### 4.1 服务器全量结果（修复前，已过期）

Exp-12 在服务器上已完整跑完一遍（24 训练轮次），但使用的是 bug 版本。
旧结果：**所有场景 AUC ≈ 0.5（等于随机）**。
→ 旧结果目录 `experiments/exp12_anomaly_detection/` 需要删除重跑。

### 4.2 本地 MVP 结果（修复后，2026-05-21）

模型：E0 baseline（cnn_lstm，seed=42，ratio=0.5）
配置：severe 严重度，8 块测试电池，2 个供体

| 场景 | AUC | 判断 |
|------|-----|------|
| sudden_aging | 0.479 | ❌ FAIL |
| knee_point | 0.542 | ❌ FAIL |
| imbalance | 0.457 | ❌ FAIL |
| li_plating | 0.449 | ❌ FAIL |
| resistance_rise | 0.572 | ❌ FAIL |

**bug 修复后依然全部失败**，说明问题不在于用了哪段供体特征。

---

## 五、根本原因分析

### 5.1 HUST 数据集的特殊性

HUST 使用的是 **LFP（磷酸铁锂）电池**，其化学特性决定了：

- 容量在大多数寿命区间内**线性平稳衰减**
- CC-CV 充电曲线的统计特征（均值/标准差/偏度/峰度）随寿命变化**极为平缓**
- 末期特征与中期特征在统计分布上差距很小

这意味着：
```
注入"供体末期特征" → 这些特征本质上仍在训练分布内
→ input_zscore 低（不显著）
→ 模型对混合特征预测的 SOH 与正常情况差距小
→ rate_anomaly 无显著突变
→ AUC ≈ 0.5
```

### 5.2 当前方法的本质局限

当前检测信号都基于**全局统计比较**，而非**当前生命周期阶段的局部比较**：

- `input_zscore`：比较特征与整个训练集均值——末期特征被训练集中大量末期样本"规范化"掉了
- `rate_anomaly`：比较当前速率与局部历史——模型对混合特征的预测变化平滑，无尖峰
- `mono_violation`：只检测 SOH 上升——异常主要表现为 SOH **下跌加速**，不触发此信号

---

## 六、下一步改进方向

用户明确表示**希望继续提升而不是放弃**，以下是可行的改进方向（从易到难）：

### 方向 A：改进异常分数定义（推荐优先尝试）

**问题**：当前 `rate_anomaly` 和 `input_zscore` 使用全局基准，对 LFP 特征变化不敏感。

**改进**：引入**循环编号感知的局部基准**：
- 对每块电池，维护一条"预期退化轨迹"（用前 N 个循环外推）
- `trajectory_deviation[t]` = 当前预测值 vs 外推预期值的偏差
- 这个信号对突发场景（①②④）应该更敏感

```python
# 示例：用最近 K 步预测的线性回归外推下一步
def compute_trajectory_deviation(preds, window=20):
    deviation = np.zeros(len(preds))
    for t in range(window, len(preds)):
        x = np.arange(window)
        y = preds[t-window:t]
        slope, intercept = np.polyfit(x, y, 1)
        expected = intercept + slope * window
        deviation[t] = abs(preds[t] - expected)
    return deviation
```

### 方向 B：改进异常注入使特征变化更显著

**问题**：对 LFP 而言，即使注入末期特征，电压/电流统计量变化仍小。

**改进**：在注入时**放大特征偏差**，而不只是线性混合：
```python
# 在注入时将正常特征和供体特征的差距放大 k 倍
delta = donor_terminal - normal_features[t]
corrupted[t] = normal_features[t] + k * alpha * delta  # k > 1 放大
```
这使注入的特征超出训练分布，z-score 更高，detection 信号更强。但需要评估这是否还符合物理真实性。

### 方向 C：换用更强的异常信号（重构误差）

**思路**：训练一个自编码器（Autoencoder）在正常数据上重建充电特征。推理时，重建误差大 → 异常。这是无监督异常检测的标准方法，但需要额外训练。

### 方向 D：缩小问题范围，只检测最突出的场景

`sudden_aging`（α=0.7，severe）本应是最容易检测的——70% 末期特征混入正常特征。但 AUC 仍为 0.479。

**建议先验证**：直接打印一块测试电池在 fault_cycle 前后的特征均值，确认注入后特征是否**真的**发生了可测量的变化：

```python
# 诊断代码
nf = info['features']      # (T, F)
cf = inject_cell_failure(nf, donor_feat, fault_cycle, 'severe')
print("故障前特征均值:", nf[:fault_cycle].mean(axis=0))
print("故障后特征均值:", cf[fault_cycle:].mean(axis=0))
print("均值差异:", (cf[fault_cycle:] - nf[fault_cycle:]).mean(axis=0))
```

如果差异几乎为零，说明需要从数据层面重新审视（可能特征被标准化了，导致注入失效）。

---

## 七、待完成工作清单

| 优先级 | 任务 | 文件 |
|-------|------|------|
| P0 | **诊断**：打印故障前后特征均值差异，确认注入有效 | mvp 脚本 |
| P1 | 实现 `trajectory_deviation` 信号（方向 A） | `evaluation/anomaly_detection.py` |
| P1 | 在 MVP 上验证新信号是否改善 AUC | `experiments/mvp_anomaly_verify.py` |
| P2 | 如果 AUC > 0.65，在服务器删旧结果重跑完整 Exp-12 | 服务器 |
| P3 | 分析最终结果，决定是否写入论文 | — |

---

## 八、关键代码片段

### 运行 MVP（不需要重新训练）
```bash
# 先确认服务器有已有 checkpoint
ls experiments/exp12_anomaly_detection/*/ratio*/seed*/model.pth

# 运行 MVP（2~5 分钟）
python experiments/mvp_anomaly_verify.py --severity severe
```

### 服务器重跑完整实验（修复后）
```bash
rm -rf experiments/exp12_anomaly_detection/
git pull origin paper-framework
python experiments/run_exp12_anomaly_detection.py
```

### 诊断注入有效性
```python
# 在 Python 交互或脚本中快速验证
import numpy as np
from evaluation.anomaly_detection import inject_cell_failure, _donor_terminal_sequence

# 检查注入后特征是否真的变化
nf = battery_data[test_bid]['features']        # (T, 6 or 16)
df = battery_data[donor_id]['features']
fault_cycle = int(len(nf) * 0.5)

cf = inject_cell_failure(nf, df, fault_cycle, 'severe')
diff = (cf[fault_cycle:] - nf[fault_cycle:]).mean(axis=0)
print("各维特征均值变化:", diff)
print("L1 总变化:", np.abs(diff).sum())
```

---

## 九、论文上下文

Exp-12 是**附加/验证性实验**，不是论文核心贡献。核心是：
- MS-PI-CNNLSTM（Exp09c）的 SOH 估计精度
- 部分监督场景下的性能
- 物理约束的场景性价值

**如果 Exp-12 最终无法取得有效 AUC（> 0.65）**，论文直接不写，不影响主线。
**如果能取得有效 AUC**，可在第四章末尾加半页作为"应用扩展"。
