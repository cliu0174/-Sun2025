# M6 伪标签验证完整记录（2026-04-28~29）

> 承接第七轮消融（`ablation_results_round7_20260428.md`）  
> 本文档记录 M6 三方向修复验证的完整过程（4 Rounds + 多 seed + r=0.5 探针），  
> 最终结论：**M6 不进 Full Stack；仅在 r=1.0 消融表中报告**

---

## 背景

第七轮消融确认 M6 在 r=0.3 的 MAE 退化 −6.59%，根因为"负偏差自我强化"：
初始轻微低估 → 伪标签偏低 → 模型学偏 → 下一轮更偏 → 循环放大。
σ 坍缩三重防护已生效，遗留问题是偏差方向性错误。

提出三方向修复：**EMA 平滑**（α=0.7）+ **单调性过滤** + **λ 自适应**。

验证脚本：`experiments/run_m6_verify.py`  
参照基准：E0（seed=42, r=0.3）MAE = **1.1708%**

---

## Round 1（warmup=30，已失效）

**问题**：best_epoch < warmup_epochs（seed=42 的 best_epoch=15 < warmup=30），  
伪标签从未影响最佳 checkpoint。结果无效，需降低 warmup。

**处置**：warmup 降至 10，重新验证。

---

## Round 2（warmup=10）

**问题**：seed=42 的 best_epoch=8 < warmup=10，仍然收敛过早。  
composite selection 尚未引入，纯 val_mae 选 best，伪标签生效前就锁定了 checkpoint。

**处置**：引入 composite model selection（score = val_mae + γ × mono_viol，min_epoch 门控）。

---

## Round 3（composite selection，γ=0.1）

**配置**：warmup=10，γ=0.1，min_epoch=10

| 组 | best_epoch | MAE | Δ vs E0 | mono_viol |
|---|---:|---:|---:|---:|
| E0 Baseline | 44 | 1.1708% | 基准 | 17.50% |
| E5_base_sel | **11** | **6.853%** ❌ | −485% | 0.32% |
| E5_ema_sel | **11** | **6.853%** ❌ | −485% | 0.32% |
| E5_filter_sel | 115 | 1.248% | −6.60% | 11.99% |
| E5_lambda_sel | 101 | **1.146%** | **+2.09%** ✅ | 8.10% |
| E5_all_sel | 84 | 1.179% | −0.71% | **2.92%** |

**Bug 发现**：E5_base_sel / E5_ema_sel 卡在 epoch 11（best_val_mae=6.2%）。  
根因：composite 模式 `best_val_score=inf`，第一个合法 epoch 必然中标；  
之后 patience 累加，early stopping 把模型锁死在那个糟糕 checkpoint。

**有效发现**：
- E5_lambda_sel（γ 帮其获得 mono_viol 信用）best_epoch=101，MAE=1.146%，**优于 E0**
- E5_all_sel MAE=1.179%，在 ±1% 内（达标），mono_viol 从 17.5% 降至 2.92%

**处置**：修复 min_epoch 前 patience 门控 bug；γ 归零（回归纯 MAE 目标）。

---

## Round 4（γ=0，纯 MAE + min_epoch 门控）

**配置**：warmup=10，γ=0.0，min_epoch=10

### Round 4a（首跑，存在 warmup 预跟踪 bug）

全部 `best_epoch=0`，`best_val_mae=Infinity`。  
根因：warmup 期"预跟踪 best_val_score"导致 post-warmup 永远无法保存 checkpoint。  
（伪标签训练使 val_mae 在 warmup 期短暂升高，预设基线过低）

**处置**：去掉预跟踪，warmup 期只冻结 patience_counter=0，进入 min_epoch 后从 inf 重新竞争。

### Round 4b（修复后，最终结果）

| 组 | best_epoch | MAE | Δ vs E0 | 备注 |
|---|---:|---:|---:|---|
| E0 Baseline | 44 | 1.1708% | 基准 | — |
| E5_base_g0 | **10** | 1.4646% | −25.1% | 早停 epoch 20 |
| E5_ema_g0 | **10** | 1.4646% | −25.1% | 与 base 完全一致 |
| E5_lambda_g0 | **10** | 1.4646% | −25.1% | 与 base 完全一致 |
| **E5_filter_g0** | **109** | **1.1701%** | **+0.06%** ✅ | 单调过滤稳定训练 |
| **E5_all_g0** | **108** | **1.1175%** | **+4.55%** ✅ | 本轮最优 |

**关键发现**：
- base / ema / lambda 三组结果完全相同（elapsed_sec 各异，不是缓存问题）。  
  原因：warmup=10，伪标签在 epoch 10 刚生成、尚未参与训练；EMA/lambda 只影响 epoch 11+，  
  但 early stopping 在 epoch 20 触发，best_epoch=10 选了预伪标签状态。
- **单调过滤是唯一有效的稳定剂**：过滤后的伪标签不引发 val_mae 退化，训练持续到 epoch 108/109。
- EMA 和 λ 只在过滤保证稳定的前提下才有贡献（all vs filter：1.1175% vs 1.1701%，差 0.53%）。

---

## 多 seed 验证（E5_all_g0，seeds=[42, 123, 34]，r=0.3）

**脚本**：`experiments/run_m6_multiseed.py`

| seed | E0 MAE | E5_all_g0 MAE | Δ | best_epoch |
|---:|---:|---:|---:|---:|
| 42 | 1.1708% | 1.1175% | **+4.55%** ✅ | 108 |
| 123 | 0.9505% | 1.0643% | **−11.97%** ❌ | **10** |
| 34 | 0.6641% | 1.0165% | **−53.07%** ❌ | **10** |
| **均值** | **0.9284%** | **1.0661%** | **−14.83%** ❌ | — |

**统计**：  
- E0 std = 0.2075%（高方差，seed 敏感）  
- E5 std = 0.0412%（低方差，M6 把所有 seed 拉向中等水平 ~1.06%）  
- delta std = 24.23%（极高，M6 效果对 seed 高度不稳定）

**结论**：仅 1/3 seed 有效。seed=42 是"幸运 seed"，非稳定现象。

---

## r=0.5 探针（seed=42）

**脚本**：`experiments/run_m6_r05_probe.py`

```
E0  MAE = 1.0829%   best_ep=76
E5  MAE = 1.2974%   best_ep=10
Δ       = −19.81%
```

伪标签诊断（6 次更新）：

| 更新 | 接受MAE | 偏差 | 全体 unlabeled MAE |
|---:|---:|---:|---:|
| 1（ep11） | 0.43% | +0.11% | 0.66% |
| 2（ep16） | 0.43% | **−0.24%** | **2.20%** ← 突变×3 |
| 3（ep21） | 0.41% | −0.11% | 1.41% |
| 4（ep26） | **0.59%** | **−0.53%** | 2.17% |
| 5（ep31） | 0.34% | −0.08% | 1.13% |
| 6（ep36） | **0.69%** | **−0.66%** | 1.70%（std 突然 3.1%） |

**关键诊断**：全体 unlabeled MAE 在 0.66%~2.20% 之间剧烈震荡，远超 accepted MAE（0.34%~0.69%）。  
接受率机械固定在 20%（n_accepted 每次恒为 8194），只过滤相对置信度，不过滤绝对质量。  
best_epoch=10，early stopping 再次在 epoch 20 触发。

**结论**：M6 在 r=0.5 同样无效。

---

## 最终结论

### M6 适用边界

| 监督比例 | 结果 | 根因 |
|---|---|---|
| r=1.0 | ✅ +3.09%（原消融，稳定） | 模型充分训练，伪标签质量高且稳定 |
| r=0.5 | ❌ −19.81%（1 seed） | 初始模型质量不足，伪标签 unlabeled MAE 震荡 |
| r=0.3 | ❌ 均值 −14.83%（3 seeds） | 同上，且 seed 高度敏感（std=24.23%） |

### 根因分析

低监督率下的恶性循环：  
1. 模型见过的真实标签少 → 对无标注样本预测差（pseudo_mae_all_unlabeled 高）  
2. 过滤只选相对置信度高的 20%，但绝对质量仍差  
3. 低质量伪标签引入负偏差 → val_mae 在 epoch 10 后上升  
4. early stopping 在 epoch 20 触发，best_epoch=10（伪标签尚未参与训练）  
5. 最终返回的是无伪标签状态，退化到比 E0 更差（因 pseudo_label 配置改变了训练动态）

r=1.0 时无此问题：模型充分训练后预测质量高，过滤 + EMA 能产生稳定的高质量伪标签。

### 处置决定

- **Full Stack 不含 M6**（M2+M4 确定）
- **Exp-07 配置需更新**：去掉 M1/M5/M6
- **论文中 M6 的写法**：作为消融项报告 r=1.0 有效（+3.09%），在 limitation 章节说明低监督率下的失效边界及其根因

---

## 代码修改记录（本轮新增）

| 文件 | 改动 | 目的 |
|------|------|------|
| `train_cross_battery.py` | composite model selection（min_epoch 门控，γ 参数）| 解决 best_epoch < warmup 问题 |
| `train_cross_battery.py` | min_epoch 前只冻结 patience，不预跟踪 best_val_score | 修复 best_epoch=0 bug |
| `train_cross_battery.py` | early stopping `> patience` → `>= patience` | off-by-one 修复 |
| `experiments/run_m6_verify.py` | Round 1~4 验证脚本（逐步迭代） | — |
| `experiments/run_m6_multiseed.py` | 多 seed 稳定性验证 | — |
| `experiments/run_m6_r05_probe.py` | r=0.5 快速探针 | — |

---

## Raw Data 索引

```
experiments/m6_verify/summary_g0.json          # Round 4 汇总
experiments/m6_multiseed/summary_multiseed.json # 多 seed 汇总
experiments/m6_r05_probe/summary.json          # r=0.5 探针结果
```

相关文档：
- `docs/ablation_results_round7_20260428.md` — 第七轮消融（M6 偏差问题定位）
- `docs/IMPROVEMENT_PLAN.md` — 总体计划（已更新 Full Stack 最终配置）
