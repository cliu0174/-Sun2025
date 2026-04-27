"""
物理约束后验分析工具

提供独立于训练脚本的物理违规统计函数，可直接接入消融实验流程。
输入 numpy 数组（predictions, targets, battery_ids），
无需 cycle_indices（假设测试集已按 cycle 顺序排列，shuffle=False 下成立）。
"""

import numpy as np
from collections import defaultdict
from typing import Optional


def compute_physics_violations(
    predictions,
    targets,
    battery_ids,
    tolerance: float = 0.01,
) -> dict:
    """
    计算预测结果在物理约束上的后验违规统计。

    Args:
        predictions : (N,) 或 (N,1) array-like，预测 SOH
        targets     : (N,) 或 (N,1) array-like，真实 SOH
        battery_ids : list/array，长度 N，每个样本对应的电池 ID
        tolerance   : 单调性软约束容忍量（与训练时保持一致，默认 0.01）

    Returns:
        dict，包含以下 key：
            mono_violation_rate     单调性违规配对比例 (%)
            mono_violation_mean     违规样本的平均超量 (SOH 单位)
            boundary_violation_rate 超出 [0,1] 的样本比例 (%)
            delta_soh_mean          平均 |ΔSOH| per step（反映预测平滑性）
            delta_soh_std           |ΔSOH| 的标准差
            n_batteries             覆盖的电池数
            n_pairs                 单调性统计的总配对数
    """
    preds = np.asarray(predictions, dtype=float).flatten()
    bids  = np.asarray(battery_ids)

    # 按电池分组（保持原顺序，测试 loader shuffle=False 时即 cycle 顺序）
    groups = defaultdict(list)
    for i, bid in enumerate(bids):
        groups[bid].append(i)

    mono_viol_amounts = []
    n_pairs_total     = 0
    delta_sohs        = []

    for bid, indices in sorted(groups.items()):
        p     = preds[indices]
        diffs = np.diff(p)           # diff[t] = pred[t+1] - pred[t]

        delta_sohs.extend(np.abs(diffs).tolist())

        # 单调性违规：SOH 应递减，violation 当 diff > tolerance
        violations = np.maximum(0.0, diffs - tolerance)
        mono_viol_amounts.extend(violations.tolist())
        n_pairs_total += len(diffs)

    viol_arr     = np.array(mono_viol_amounts)
    n_violations = int(np.sum(viol_arr > 0))

    boundary_viol = int(np.sum((preds < 0) | (preds > 1)))

    mono_viol_mean = float(viol_arr[viol_arr > 0].mean()) if n_violations > 0 else 0.0

    return {
        'mono_violation_rate':      round(n_violations / max(1, n_pairs_total) * 100, 4),
        'mono_violation_mean':      round(mono_viol_mean, 6),
        'boundary_violation_rate':  round(boundary_viol / max(1, len(preds)) * 100, 4),
        'delta_soh_mean':           round(float(np.mean(delta_sohs)) if delta_sohs else 0.0, 6),
        'delta_soh_std':            round(float(np.std(delta_sohs))  if delta_sohs else 0.0, 6),
        'n_batteries':              len(groups),
        'n_pairs':                  n_pairs_total,
    }
