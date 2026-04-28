# 消融实验第六轮结果快照（2026-04-28）

> 承接第五轮（`ablation_results_round5_20260427.md`）  
> 本轮关键事件：修复 Eneg1_no_physics NaN 梯度污染 bug，获得完整对照组数据

---

## 实验配置

- **5 seeds** × **3 ratios** = `[1.0, 0.5, 0.3]`，7 组实验共 105 次运行，全部成功
- **物理约束参数**：`monotonic_weight=0.3, tolerance=0.005, boundary_weight=0, min_cycle=300`
- **本轮新增**：Eneg1_no_physics 全部 15 次正常完成（修 NaN 梯度 bug 后）

---

## 本轮修复记录

### Bug：Eneg1 ratio<1.0 全部失败，报 matplotlib 直方图错误

**根因**（两层）：

1. **一层：NaN 梯度污染**  
   `train_cross_battery.py` 部分监督路径：
   ```python
   # 旧代码（有 bug）
   diff_sq = (predictions - targets) ** 2   # targets 含 NaN → diff_sq 含 NaN
   loss = torch.where(finite_mask, diff_sq, zeros)  # 前向屏蔽了，但 autograd 图已污染
   # → backward() 产生 NaN 梯度 → optimizer.step 把所有参数变 NaN
   ```
   **修复**（commit `9abacb9`）：先把 NaN target 替换为 0 再做减法
   ```python
   safe_targets = torch.where(finite_mask, targets, torch.zeros_like(targets))
   diff_sq = (predictions - safe_targets) ** 2
   loss = torch.where(finite_mask, diff_sq, torch.zeros_like(diff_sq)).sum() / n_labeled
   ```

2. **二层：绘图崩溃掩盖了根因**  
   NaN 参数导致预测全 NaN，`plt.hist(nan_errors)` 抛 `ValueError: autodetected range of [nan, nan] is not finite`，error.json 里看到的是绘图错误而非训练错误。  
   **防御修复**（commit `e772036`）：绘图前过滤 NaN。

**为何 ratio=1.0 正常**：mask 全 True，走 `criterion(predictions, targets)` 快路径，不触发 NaN target 路径。

---

## MAE 结果（5 seeds 均值，单位原始值/百分比，括号内 Δ% vs E0）

| 组 | ratio=1.0 MAE | Δ% | ratio=0.5 MAE | Δ% | ratio=0.3 MAE | Δ% | 综合判断 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Eneg1 无物理约束** | 0.00808 | +2.03% | 0.00800 | +0.94% | 0.00822 | +1.16% | ⚠️ MAE 反优于 E0，见注 |
| **E0 Baseline** | 0.00824 | — | 0.00807 | — | 0.00832 | — | 基线 |
| **E1 +M1 注意力** | 0.00824 | +0.05% | 0.00863 | **−6.89%** | 0.00860 | −3.35% | ❌ 部分监督退化 |
| **E2 +M2 MC Dropout** | **0.00788** | **+4.42%** | 0.00816 | −1.08% | 0.00825 | +0.79% | ✅ r=1.0 最佳 |
| **E3 +M4 速率连续性** | 0.00837 | −1.54% | 0.00823 | −1.95% | 0.00845 | −1.56% | ⚠️ MAE 微退，物理↑ |
| **E4 +M5 自适应权重** | 0.00923 | **−12.02%** | 0.00849 | −5.20% | 0.00880 | −5.78% | ❌❌ 全场最差 |
| **E5 +M2+M6 伪标签** | **0.00799** | **+3.09%** | 0.00827 | −2.45% | 0.00887 | **−6.59%** | ❌ r=0.3 核心场景退化 |

> **Eneg1 MAE 低于 E0 的解释**：软单调约束（E0）以约 +1.2% MAE 换取单调违反率降低 ~2%，这是正常的约束-精度权衡，不是 E0 变差了。

---

## 单调违反率（配对比例，越低越好）

| 组 | ratio=1.0 | ratio=0.5 | ratio=0.3 | 综合判断 |
|---|:---:|:---:|:---:|---|
| Eneg1 无物理约束 | 6.18% | 8.15% | 9.10% | 最高（无约束） |
| **E0 Baseline** | **5.40%** | 5.07% | 7.74% | 基线 |
| E1 +M1 注意力 | 5.65% | **12.84%** ❌ | 6.81% | r=0.5 飙升 2.5× |
| E2 +M2 MC Dropout | 8.46% | 5.45% | 10.15% | r=1.0 偏高 |
| **E3 +M4 速率连续性** | 9.17% | **2.89%** ✅ | 9.25% | r=0.5 全场最低 |
| E4 +M5 自适应权重 | 11.51% ❌ | 4.78% | 7.43% | r=1.0 最差 |
| E5 +M2+M6 伪标签 | 7.61% | 6.82% | **18.39%** ❌ | r=0.3 全场最高 |

---

## 模块定性结论

### ✅ M2 MC Dropout — 保留
- r=1.0 时 +4.42%，是单独模块中 MAE 提升最显著的
- r<1.0 时基本持平（±1%）
- **论文定位**：全监督 / 足量标签场景下的精度提升模块 + 置信区间估计基础

### ✅ M4 速率连续性 — 保留（物理一致性故事线）
- MAE 一律退化 ~1.5%，不适合当精度卖点
- r=0.5 时单调违反率 2.89%（E0 的 57%），是物理保真度最佳模块
- **论文定位**：物理一致性约束贡献，与 MAE 是不同维度

### ❌ M1 注意力 — 待彻查（计划 B 的目标之一）
- r=1.0 几乎无效（+0.05%）
- r=0.5 时 MAE −6.9%、mono_viol 飙到 12.84%
- **疑因**：注意力在标签稀疏时过拟合到少量有标签循环的局部模式，破坏单调先验
- **计划 B**：打印注意力权重分布，检查低 ratio 下是否对有标签样本过度集中

### ❌❌ M5 自适应权重 — 待彻查（计划 B 最高优先级）
- r=1.0 时 MAE −12.02%（接近 2× 基线标准差，基本确认训练崩）
- 单调违反率 r=1.0 时 11.51%（E0 的 2.1×），说明单调性权重被学歪
- **疑因**：`init_log_vars=[0, 1.2, 5, 5]` 初始 monotonic/smoothness 权重被大幅压低，
  梯度信号失衡后 log_var 继续学歪
- **计划 B**：打印训练过程中 4 个 log_var 的变化曲线，确认是否单调权重持续趋零

### ❌ M6 伪标签（+M2）— 待彻查（计划 B）
- r=1.0 时 +3.09%（此时无未标注样本，等同于 M2 ensemble 效果）
- r=0.5 时 −2.45%，r=0.3 时 −6.59%（核心场景，越需要伪标签越退化）
- r=0.3 单调违反率 18.39%（全场最高），伪标签引入大量噪声
- **疑因**：低标签率下 MC Dropout 自身预测不准，置信度阈值（当前 `uncertainty<0.01`）太宽松
- **计划 B**：打印伪标签命中率（pseudo label 与真实 SOH 误差分布），确认阈值调整方向

---

## 后续策略选择

### 计划 B（当前决定：先执行）
**目标**：彻查 M5 log_var 训练曲线 + M6 伪标签命中率，定位退化根因

具体行动：
1. **M5 诊断**：在 `adaptive_loss.py` 加 log_var 记录，单跑一次 ratio=1.0，画 4 个权重的 epoch 曲线
2. **M6 诊断**：在 `pseudo_labeling.py` 的 `_select_pseudo_labels` 加命中率统计（pseudo_soh vs true_soh），单跑 ratio=0.3
3. **M1 诊断**：打印注意力权重 entropy，确认是否退化为 one-hot（过拟合标志）

B 的结果决定后续：
- 若 M5 可修（调整初始化 / 加 warmup）→ **选项 A**（跑 Full Stack，含修复后 M5）
- 若 M5/M6 根本不适合当前场景 → **选项 C**（重构 Full Stack，只保留 M2+M4）

### 计划 A（B 后可能）
直接跑 Exp-07 Full Stack（含全部模块），把消融负结果纳入论文作对比

### 计划 C（B 后可能）
修改 Exp-07 Full Stack 配置，去掉 M5，只保留 M2+M4（或 M2+M4+M6 提高阈值）

---

## Raw Data 索引

完整 metrics.json 位于服务器：
```
/data2/Documents/lc/1111-soh/experiments/ablation_single_module/metrics.json
```

Git commits（本轮）：
- `9abacb9` — fix(supervision): 修复部分监督 loss 反向传播 NaN 梯度污染参数
- `e772036` — fix(plot): 过滤 NaN/Inf 误差再绘直方图
