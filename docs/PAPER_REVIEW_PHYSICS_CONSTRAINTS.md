# 参考论文物理约束实现方式综述（2026-04-23）

## 已读论文列表

| # | 文件 | 作者 | 期刊/年份 | 核心方法 |
|---|------|------|-----------|----------|
| 1 | paper1_text.txt (s41598-025-30602-4) | Salem & Mohamed | Sci. Rep. 2025 | LSTM + PPO 强化学习混合 |
| 2 | paper2_text.txt (s41467-024-48779-z) | Wang et al. | Nature Comms 2024 | PINN + 软单调约束 + PDE |
| 3 | paper3_text.txt = S0360544 | Tian et al. | Energy 2025 | 通用 PINN SOH 框架 + UDE |
| 4 | S0951832 | Zhu et al. | Reliab. Eng. 2025 | 贝叶斯校准 PINN + MC Dropout（二次电池） |
| 5 | S2352152 | Zhang et al. | J. Energy Storage | TS-PINN，IC 曲线特征 + 自适应权重 |
| 6 | batteries-09-00301 | Singh et al. | Batteries 2023 | 菲克扩散 PDE（SOC 估计） |
| 7 | batteries-11-00049 | Deng et al. | Batteries 2025 | BPNN+MLP PINN，梯度单调约束 |
| 8 | s41598-026-37850-y | Kumar et al. | Sci. Rep. 2026 | 温度+阻抗感知 LSTM-PINN |
| 9 | s11581-025-06805-0 | Wang Lei et al. | Ionics 2026 | 部分可观性 PINN，自适应权重 |

---

## 一、单调性约束的实现方式

### 1.1 文献中实际使用的方法

**方法A：相邻步比较（主流方法）**

```
L_mono = Σ_{k=1}^{M-1} ReLU(û_{k+1} - û_k)
```

- **Wang et al. 2024**（Nature Comms）：明确使用相邻步差值，对"预测值上升"施加惩罚
- **Tian et al. 2025**（Energy）：同上，公式 `L_mono = Σ ReLU(û_{k+1} - û_k)`
- **Kumar et al. 2026**（Sci. Rep.）：采用独立的 `λ_mono * L_mono` 项，消融实验证实去除后出现"伪回弹"伪影

**方法B：梯度约束（Deng et al. 2025）**

```
L_mono = (1/N) * Σ ReLU(∂ŜOH / ∂X_IC)
```

该方法约束的不是预测序列的时序单调性，而是**预测 SOH 关于健康因子的偏导数方向**（SOH 应随 IC 峰面积减小而降低）。适用于以健康因子为直接输入的 MLP 架构，与时序模型结构不同。

**方法C：RL 奖励函数（Salem & Mohamed 2025）**

将单调性编码进 PPO 奖励函数，不在损失函数中显式施加。

**方法D：PDE 残差（Zhu et al., Zhang et al., Kumar et al.）**

```
L_pde = ||∂u/∂t - G_θ(t, x, u)||²
```

通过拟合退化动力学方程（如阿伦尼乌斯温度依赖模型、幂律衰减模型）间接约束单调性，不直接比较相邻步输出。

### 1.2 核心结论：无一篇论文使用"时间窗口范围"比较

**我们当前的 `temporal_max_step=20` 设计（在时间差 ≤20 的样本对之间比较）在文献中没有先例。** 所有使用显式单调约束的论文均采用**相邻步**（k 与 k+1）比较，或全序列排序后比较。

### 1.3 对本项目的直接启示

当前代码：
```python
# 筛选同一电池、cycle_diff <= temporal_max_step 的样本对
valid_mask = (same_battery) & (cycle_diff <= self.temporal_max_step) & (cycle_diff > 0)
```

**问题**：shuffle=True 时，同一批次中相同电池的窗口平均循环间隔 ≈178，远超 20，导致 valid_mask 命中率极低，约束几乎不生效。

**推荐修改**：改为对批次内同一电池的所有样本，按循环索引排序后做相邻比较：
```python
# 对每块电池，按 cycle_idx 排序后取相邻差
sorted_preds = predictions[sorted_idx]
L_mono = ReLU(sorted_preds[1:] - sorted_preds[:-1] - tolerance).mean()
```

---

## 二、物理约束的其他形式

### 2.1 PDE 型（高复杂度）

- **Zhu et al.**：`∂u/∂t ≈ G_θ(t, x, u)` — 用神经网络逼近退化动力学方程的右端项
- **Kumar et al.**：`dSOH/dt = -αβ_eff*(1 - α_eff*t)^(β-1)` — 幂律退化模型
- **Singh et al.**：菲克扩散方程（Li+ 在固相中的扩散，针对 SOC 不是 SOH）

这类方法需要预先假设退化方程结构，复杂度高，对 HUST 数据不一定适用。

### 2.2 边界约束

Wang et al.、Kumar et al. 均包含 `SOH ∈ [0, 1]` 的边界约束，与我们的 `boundary_loss` 一致。

### 2.3 平滑约束（速率连续性）

Wang Lei et al.（Ionics 2026）：
```
L_smooth = 可选时序梯度平滑正则化项
```
描述与我们的 M4 `smoothness_loss` 一致。Zhang et al. TS-PINN 的 PDE 残差也间接起到平滑作用。

---

## 三、时间衰减权重/注意力机制

**结论：文献中没有任何一篇论文在物理约束损失中使用时间衰减权重或注意力机制。**

这意味着我们讨论的 `TemporalPhysicsAttention`（对近期样本对施加更大惩罚）是**原创设计**，在文献中无先例。如果实现，需要在论文中自行论证其合理性，而无法援引他人。

---

## 四、批次随机打乱与物理约束的冲突

文献中所有 PINN 论文均**未讨论**是否对数据进行 shuffle。推测大多数工作：
- 按时序（trajectory）顺序输入，不做 shuffle
- 或以电池级别（trajectory-level）批次为单位训练

**我们的设计（cycle 级别 shuffle=True）是训练效率优先的工程选择**，但代价是单调约束需要重新设计为"排序后相邻比较"而非直接使用 mini-batch 内的顺序关系。

---

## 五、MC Dropout 与不确定性量化

仅 **Zhu et al.（paper5）** 使用 MC Dropout：
- 训练和推理阶段均开启 Dropout
- 多次随机前向传播 → 均值 + 方差 → 95% 置信区间
- 用于第二次生命电池（二手电池）的不确定性校准

方法与我们的 M2+M7 设计完全一致，可直接引用为 MC Dropout 不确定性量化的依据。

---

## 六、自适应损失权重

### 6.1 文献中的实现

- **Zhang et al. TS-PINN**：基于梯度统计的自适应权重，PDE 项从初始 1/3 最终稳定到 0.45，数据项权重随训练进行自然增大
- **Wang Lei et al.**：`λ1, λ2, λ3` 在训练中动态调整，"adaptive weight-balancing strategy"（未详细说明方法）

### 6.2 Kendall 不确定性权重（M5 方案）

**无任何论文使用 Kendall et al. (2018) 同方差不确定性加权框架**来平衡多任务损失。这是我们 M5 的**原创贡献**，但需注意：

- 当前 M5 的 `init_log_vars = [0.0, 0.0, 0.0, 0.0]` 导致初始物理权重远高于 baseline（10-20 倍）
- 需修复为：`init_log_vars = [0.0, ln(0.1), ln(0.05), 0.0]` ≈ `[0.0, 2.303, 2.996, 0.0]`
- 还需加入 `l2_reg = 0.01` 防止权重归零

---

## 七、部分监督场景的独特性

**我们的"部分生命周期监督"设定在文献中最接近的工作：**

- Wang Lei et al.（Ionics 2026）：随机 mask 输入特征（30%-70% 缺失）——但缺失的是**特征**，不是 SOH 标签
- Zhu et al.：第二次生命电池使用历史不完整——但不是显式的 label masking

**我们的设定（生命周期连续区间内 SOH 标签缺失）在文献中没有完全对应的工作**，是论文创新点之一。结合物理约束对无标签样本的弱监督作用，有较强的差异化价值。

---

## 八、对改进单调约束的具体建议

基于文献综述，建议将 `monotonic_loss` 改为标准的相邻步比较：

```python
def monotonic_loss(self, predictions, battery_ids, cycle_indices, supervision_mask=None):
    loss = torch.tensor(0.0, device=predictions.device)
    n_valid = 0
    
    bid_tensor, n_batteries = self._encode_battery_ids(battery_ids, predictions.device)
    
    for uid in range(n_batteries):
        mask = (bid_tensor == uid).nonzero(as_tuple=True)[0]
        if len(mask) < 2:
            continue
        
        # 按 cycle_idx 排序
        cycles = cycle_indices[mask]
        sort_idx = cycles.argsort()
        sorted_preds = predictions[mask[sort_idx]].squeeze(-1)
        
        # 相邻步单调约束（标准文献方法）
        diffs = sorted_preds[1:] - sorted_preds[:-1]  # 应为负值（递减）
        violations = torch.relu(diffs + self.tolerance)  # tolerance ≈ 0.001
        loss = loss + violations.sum()
        n_valid += len(violations)
    
    return loss / max(n_valid, 1)
```

这一改动解决了 `temporal_max_step=20` + `shuffle=True` 导致的约束几乎不生效的问题。

---

## 九、总结对比表

| 论文 | 单调约束 | PDE 约束 | 边界约束 | 平滑约束 | 自适应权重 | MC Dropout |
|------|----------|----------|----------|----------|------------|------------|
| Wang et al. 2024 | ✅ 相邻步 | ✅ | ✅ | — | — | — |
| Tian et al. 2025 | ✅ 相邻步 | ✅ UDE | — | — | — | — |
| Zhu et al. 2025 | — | ✅ PDE | — | — | — | ✅ |
| Zhang et al. TS-PINN | — | ✅ PDE | — | — | ✅ 梯度统计 | — |
| Deng et al. 2025 | ✅ 梯度 | ✅ | — | — | — | — |
| Kumar et al. 2026 | ✅ 相邻步 | ✅ 幂律 | — | — | 固定 λ | — |
| Wang Lei et al. 2026 | — | — | — | ✅ | ✅ 动态 | — |
| **本文（PI-CNNLSTM）** | **⚠️ 需修复** | — | **✅** | **✅ M4** | **✅ M5 Kendall** | **✅ M2** |

