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

## 六、下一步改进方向（2026-05-21 最新共识）

用户提出一个关键判断：当前 Exp-12 失败很可能不是检测逻辑本身完全错误，而是**异常注入后的特征变化不够显著**，没有触发模型预测 SOH 出现明显落差。新的优先方向是：不再主要依赖跨电池供体混合，而是用**同一块电池自身生命周期轨迹的跳跃、抽取、采样和未来状态混合**来构造异常。

核心原则：

```text
异常后的 features 必须仍来自同一块电池的真实后期轨迹，
但通过重排生命周期片段，人为制造 SOH 骤降、拐点、加速或渐进偏离。
```

这样做的好处：
- 保留同一电池个体特性，避免跨电池供体差异干扰；
- 异常特征来自真实观测轨迹，不是任意噪声或无物理含义的放大扰动；
- 可以直接验证用户的核心假设：如果模型学到了 `feature → SOH` 映射，那么同电池后期特征拼接后，预测 SOH 应出现落差或斜率突增。

新增示意图已生成，**不要覆盖旧图**：

```text
docs/exp12_degradation_scenarios_self_trajectory.png
docs/plot_degradation_scenarios_self_trajectory.py
```

### 方向 A：同电池生命周期轨迹重构（推荐优先实现）

#### A1. SOH 骤降 / Single-cell sudden drop

用户明确想模拟“电池 SOH 骤降”的情景。构造方式：

```python
corrupted_features = concat(features[:t0], features[s0:])
corrupted_soh      = concat(soh[:t0],      soh[s0:])
```

含义：
- `t0` 前正常退化；
- `t0` 处发生突发不可逆损伤；
- 损伤后状态等价于跳过一段生命周期，直接进入 `s0` 之后；
- 后续仍沿同一电池自身退化轨迹继续衰减。

参数建议：
- `t0 = int(0.50 * T)`
- `s0` 不固定比例优先，建议按 SOH 落差选择：`soh[t0] - soh[s0] ≈ 0.03 / 0.05 / 0.08` 对应 mild / moderate / severe
- 如果找不到满足落差的 `s0`，再 fallback 到比例位置，如 `0.60T / 0.68T / 0.75T`

注意：该构造会让异常序列长度变短，评估时 `fault_start = t0`，正样本为 `corrupted[fault_start:]`。

#### A2. 容量拐点 / Capacity knee-point

用户确认“后半段时间压缩”可以通过**抽取/跳采样**实现：

```python
tail_idx = np.arange(t0, T, step)
corrupted_features = concat(features[:t0], features[tail_idx])
corrupted_soh      = concat(soh[:t0],      soh[tail_idx])
```

含义：
- `t0` 前正常；
- `t0` 后仍沿同一电池轨迹退化；
- 但每个观测点跨过更多真实生命周期，因此单位 pseudo-cycle 内 SOH 下降更快；
- 模拟容量拐点后斜率突然变陡。

严重度建议：

```text
mild:     step = 2
moderate: step = 3
severe:   step = 4
```

也可以用连续比例 `gamma = 1.5 / 2.0 / 3.0` 做非整数抽样。

#### A3. 簇内不均衡加剧 / Inter-cell imbalance

改成“同电池未来状态渐进混合”，不用跨电池供体：

```python
for t >= t0:
    progress = (t - t0) / (T - t0)
    alpha_t = max_alpha * progress
    future_idx = min(T - 1, t + int(max_offset * progress))
    corrupted[t] = (1 - alpha_t) * features[t] + alpha_t * features[future_idx]
```

含义：
- 异常不是一步跳变，而是越来越像同一电池更老阶段；
- 模拟簇内不均衡逐渐放大、BMS 观测轨迹逐步偏离正常单体。

严重度建议：

```text
mild:     max_alpha=0.25, max_offset≈0.10T
moderate: max_alpha=0.40, max_offset≈0.18T
severe:   max_alpha=0.55, max_offset≈0.25T
```

#### A4. 析锂台阶 / Lithium plating

构造成“先跳跃，再加速采样”：

```python
corrupted_features = concat(
    features[:t0],
    features[s0:T:step],
)
corrupted_soh = concat(
    soh[:t0],
    soh[s0:T:step],
)
```

含义：
- `t0` 发生析锂导致的不可逆台阶式损伤；
- 损伤后直接进入同电池更后期状态；
- 后续尾段继续跳采样，模拟台阶后加速退化。

建议与 sudden drop 区分：
- sudden drop：`features[:t0] + features[s0:]`，台阶后保持原始后期速度；
- lithium plating：`features[:t0] + features[s0:T:step]`，台阶后继续加速。

#### A5. 内阻渐进增长 / Internal resistance rise

改成“二次漂移到未来状态”：

```python
for t >= t0:
    progress = (t - t0) / (T - t0)
    alpha_t = max_alpha * progress ** 2
    future_idx = min(T - 1, t + offset)
    corrupted[t] = (1 - alpha_t) * features[t] + alpha_t * features[future_idx]
```

含义：
- 早期几乎看不出异常；
- 后期越来越偏向同一电池更老状态；
- 模拟内阻逐渐增长导致的后期斜率增加。

### 方向 B：异常分数定义同步升级

**问题**：当前 `rate_anomaly` 和 `input_zscore` 使用全局基准，对 LFP 特征变化不敏感。

**改进 1：新增 SOH 骤降分数 `drop_anomaly`**

当前 `mono_violation` 检测的是 SOH 上升，不适合检测 SOH 突然下降。建议新增：

```python
drop = np.zeros(T)
drop[1:] = np.maximum(predictions[:-1] - predictions[1:], 0)
```

再做局部标准化：

```python
drop_anomaly[t] = (drop[t] - mean(drop[t-K:t])) / std(drop[t-K:t])
```

该信号直接对应用户关心的判断逻辑：

```text
如果根据当前输入特征预测出的 SOH 与上一时刻 SOH 存在较大落差/斜率突增，
就判别出现退化异常。
```

**改进 2：引入循环编号感知的局部基准 `trajectory_deviation`**

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

MVP 里不要只打印 `combined`，应同时输出：

```text
input_zscore / rate_anomaly / drop_anomaly / trajectory_deviation / combined
```

并额外打印诊断量：

```text
pred_drop_at_fault = preds_fault[fault_start-1] - preds_fault[fault_start]
mean_pred_gap_after_fault = mean(preds_clean_aligned - preds_fault_aligned)
feature_l1_change_after_fault
```

### 方向 C：旧方案作为备选，不再优先

旧方案包括：
- 放大跨电池供体特征偏差；
- 继续改进全局 z-score；
- 自编码器重构误差。

这些方案暂时降级为备选。下一位 Agent 应优先实现“同电池轨迹重构”版本，因为它更贴近用户要模拟的 SOH 骤降和拐点情景。

### 实现建议

推荐在 `evaluation/anomaly_detection.py` 中新增一组函数，不要直接删除原有 5 个注入函数：

```text
inject_self_jump_drop()
inject_self_knee_sampling()
inject_self_imbalance_drift()
inject_self_lithium_plating()
inject_self_resistance_rise()
```

然后给 `run_anomaly_detection_for_battery()` 增加参数，例如：

```python
injection_mode: str = 'legacy'  # legacy / self_trajectory
targets: Optional[np.ndarray] = None
```

`self_trajectory` 模式需要传入当前电池的真实 SOH `targets`，用于选择 `s0` 或构造 `corrupted_soh` 诊断曲线。模型推理仍然只使用 `corrupted_features`。

---

## 七、待完成工作清单

| 优先级 | 任务 | 文件 |
|-------|------|------|
| P0 | 实现 5 个 `self_trajectory` 注入函数，先不要删除 legacy 注入函数 | `evaluation/anomaly_detection.py` |
| P0 | MVP 增加 `--injection_mode self_trajectory`，并把当前电池 `targets` 传入检测流程 | `experiments/mvp_anomaly_verify.py` |
| P0 | 新增/打印 `drop_anomaly`，验证 SOH 骤降是否能被预测落差识别 | `evaluation/anomaly_detection.py` / MVP |
| P1 | 实现 `trajectory_deviation`，并和 `drop_anomaly` 分开输出 AUC | `evaluation/anomaly_detection.py` |
| P1 | 在 MVP 上同时打印各信号 AUC，不要只看 `combined` | `experiments/mvp_anomaly_verify.py` |
| P1 | 诊断打印：fault 点预测落差、fault 后 clean-vs-fault 预测差、特征 L1 变化 | `experiments/mvp_anomaly_verify.py` |
| P2 | 若 `self_jump_drop` / `self_lithium_plating` AUC > 0.65，再扩展完整 Exp-12 | `experiments/run_exp12_anomaly_detection.py` |
| P3 | 分析最终结果，决定是否写入论文；若仍失败，Exp-12 不写入主线 | — |

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
