"""
Exp-12：基于物理约束的电芯异常检测
====================================

核心思路：已训练的 PI-MS-CNN-LSTM 模型学到了"正常退化模式"。
当电池簇中某电芯突发失效时，其特征向量偏离正常模式，
模型的物理约束违反信号（单调违规率、速率突变）可作为零额外成本的异常检测器。

模块功能：
  1. 异常注入：跨电芯特征替换（模拟簇内电芯突发失效）
  2. 异常评分：三维评分（输入 z-score / 单调违规 / 速率突变）
  3. 检测评估：ROC-AUC / Detection Rate / 检测延迟
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


# ============================================================
# 1. 异常注入：跨电芯特征替换
# ============================================================

def inject_cell_failure(
    normal_features: np.ndarray,
    donor_features: np.ndarray,
    fault_cycle: int,
    severity: str = 'moderate',
    alpha: Optional[float] = None,
) -> np.ndarray:
    """
    模拟电池簇中单电芯突发失效：从 fault_cycle 开始，
    将正常电芯的特征与失效供体电芯的末期特征混合。

    Args:
        normal_features: (T, F) 正常电芯的完整特征序列
        donor_features:  (T', F) 供体电芯（失效源）的特征序列
        fault_cycle:     故障注入时刻（窗口级索引）
        severity:        'mild' / 'moderate' / 'severe'
        alpha:           直接指定混合比例（覆盖 severity）

    Returns:
        corrupted: (T, F) 注入故障后的特征序列
    """
    severity_map = {
        'mild':     0.2,   # 5 芯簇中 1 芯轻度退化
        'moderate': 0.4,   # 3 芯簇中 1 芯严重退化
        'severe':   0.7,   # 失效电芯主导测量
    }

    if alpha is None:
        alpha = severity_map[severity]

    corrupted = normal_features.copy()
    T = len(normal_features)
    T_donor = len(donor_features)

    # 从供体电芯取末期特征（最后一段）
    for t in range(fault_cycle, T):
        # 供体索引：从末期倒数映射
        donor_idx = min(T_donor - 1 - (T - 1 - t), T_donor - 1)
        donor_idx = max(0, donor_idx)
        corrupted[t] = (1 - alpha) * normal_features[t] + alpha * donor_features[donor_idx]

    return corrupted


def select_donor_batteries(
    all_battery_data: Dict[str, np.ndarray],
    all_battery_targets: Dict[str, np.ndarray],
    n_donors: int = 5,
) -> List[str]:
    """
    选择退化最严重的电芯作为故障供体。

    Args:
        all_battery_data:    {battery_id: features (T, F)}
        all_battery_targets: {battery_id: targets (T,)}
        n_donors: 选择数量

    Returns:
        donor_ids: 退化最严重的电芯 ID 列表
    """
    # 按最终 SOH 升序排列（SOH 最低的作为供体）
    final_soh = {bid: targets[-1] for bid, targets in all_battery_targets.items()
                 if len(targets) > 0}
    sorted_batteries = sorted(final_soh.items(), key=lambda x: x[1])
    return [bid for bid, _ in sorted_batteries[:n_donors]]


# ============================================================
# 2. 异常评分
# ============================================================

def compute_anomaly_scores(
    predictions: np.ndarray,
    features: np.ndarray,
    feature_history_mean: np.ndarray,
    feature_history_std: np.ndarray,
    window_size: int = 5,
) -> Dict[str, np.ndarray]:
    """
    计算三维异常分数（逐窗口）。

    Args:
        predictions:         (T,) 模型预测 SOH 序列
        features:            (T, F) 输入特征序列
        feature_history_mean: (F,) 训练集特征均值（用于 z-score 基准）
        feature_history_std:  (F,) 训练集特征标准差
        window_size:          滑动窗口大小（用于局部统计）

    Returns:
        scores: {
            'input_zscore':   (T,) 输入层异常分数（最大维度 z-score）
            'mono_violation': (T,) 单调违规信号（1=违规, 0=正常）
            'rate_anomaly':   (T,) 速率突变分数
            'combined':       (T,) 综合分数
        }
    """
    T = len(predictions)

    # ── 分数 1：输入层 z-score ──
    # 相对训练集统计量的 z-score，取每步的最大维度
    safe_std = np.where(feature_history_std > 1e-8, feature_history_std, 1.0)
    zscore_all = np.abs(features - feature_history_mean) / safe_std  # (T, F)
    input_zscore = np.max(zscore_all, axis=-1)  # (T,)

    # ── 分数 2：单调违规 ──
    # SOH 应单调递减，任何上升都是违规信号
    mono_violation = np.zeros(T)
    if T > 1:
        soh_diff = np.diff(predictions)  # (T-1,)
        mono_violation[1:] = np.maximum(soh_diff, 0)  # 上升量

    # ── 分数 3：速率突变 ──
    # 退化速率的局部异常度
    rate_anomaly = np.zeros(T)
    if T > 2:
        soh_rate = np.diff(predictions)  # (T-1,)
        for t in range(window_size, T - 1):
            local_rates = soh_rate[max(0, t - window_size):t]
            if len(local_rates) > 1 and np.std(local_rates) > 1e-8:
                rate_anomaly[t + 1] = abs(soh_rate[t] - np.mean(local_rates)) / np.std(local_rates)

    # ── 综合分数 ──
    # 标准化后取最大值
    def _safe_normalize(x):
        xmin, xmax = x.min(), x.max()
        if xmax - xmin < 1e-10:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin)

    combined = np.maximum.reduce([
        _safe_normalize(input_zscore),
        _safe_normalize(mono_violation) * 2.0,  # 物理违规权重更高
        _safe_normalize(rate_anomaly),
    ])

    return {
        'input_zscore':   input_zscore,
        'mono_violation': mono_violation,
        'rate_anomaly':   rate_anomaly,
        'combined':       combined,
    }


# ============================================================
# 3. 检测评估
# ============================================================

def evaluate_detection(
    scores_clean: np.ndarray,
    scores_fault: np.ndarray,
    fault_start: int,
) -> Dict[str, float]:
    """
    评估异常检测性能。

    Args:
        scores_clean: (T,) 正常电芯的异常分数序列
        scores_fault: (T,) 故障电芯的异常分数序列
        fault_start:  故障注入起始窗口索引

    Returns:
        metrics: {
            'auc':            ROC-AUC
            'det_rate_fpr5':  Detection Rate @ FPR=5%
            'det_delay':      检测延迟（从 fault_start 到首次超阈值的窗口数）
        }
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    # 构建二分类标签
    # 负样本：clean 的全部 + fault 在 fault_start 之前的部分
    # 正样本：fault 在 fault_start 及之后的部分
    neg_scores = np.concatenate([scores_clean, scores_fault[:fault_start]])
    pos_scores = scores_fault[fault_start:]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return {'auc': float('nan'), 'det_rate_fpr5': float('nan'), 'det_delay': -1}

    y_true = np.concatenate([np.zeros(len(neg_scores)), np.ones(len(pos_scores))])
    y_score = np.concatenate([neg_scores, pos_scores])

    # ROC-AUC
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float('nan')

    # Detection Rate @ FPR=5%
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        # 找 FPR <= 5% 的最大 TPR
        mask = fpr <= 0.05
        det_rate_fpr5 = tpr[mask][-1] if mask.any() else 0.0
    except Exception:
        det_rate_fpr5 = float('nan')

    # 检测延迟：用 FPR=5% 对应的阈值
    try:
        idx_5 = np.where(fpr <= 0.05)[0][-1]
        threshold_5 = thresholds[idx_5]

        # 从 fault_start 开始找首次超阈值
        det_delay = -1
        for t in range(fault_start, len(scores_fault)):
            if scores_fault[t] >= threshold_5:
                det_delay = t - fault_start
                break
    except Exception:
        det_delay = -1

    return {
        'auc':           float(auc),
        'det_rate_fpr5': float(det_rate_fpr5),
        'det_delay':     int(det_delay),
    }


def run_anomaly_detection_for_battery(
    model: torch.nn.Module,
    normal_features: np.ndarray,
    donor_features: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    fault_cycle: int,
    severity: str,
    window_size: int = 40,
    device: str = 'cuda',
) -> Dict:
    """
    对单块电池执行完整的异常检测流程：注入 → 推理 → 评分 → 评估。

    Args:
        model:           训练好的模型
        normal_features: (T_raw, F) 正常电芯原始特征（已标准化）
        donor_features:  (T_donor, F) 供体电芯原始特征（已标准化）
        feature_mean:    (F,) 训练集特征均值
        feature_std:     (F,) 训练集特征标准差
        fault_cycle:     故障注入时刻（原始 cycle 索引）
        severity:        严重程度
        window_size:     模型窗口大小
        device:          计算设备

    Returns:
        result: 包含评分和评估指标的字典
    """
    # 1. 注入异常
    corrupted_features = inject_cell_failure(
        normal_features, donor_features, fault_cycle, severity
    )

    # 2. 窗口化并推理
    def _predict_sequence(features_seq):
        """对特征序列做窗口化推理，返回预测 SOH 序列"""
        raw = getattr(model, 'model', model)
        raw.eval()

        T_raw = len(features_seq)
        if T_raw < window_size:
            return np.array([])

        windows = []
        for i in range(T_raw - window_size + 1):
            windows.append(features_seq[i:i + window_size])
        x = torch.FloatTensor(np.array(windows)).to(device)

        preds = []
        batch_size = 512
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                batch = x[start:start + batch_size]
                out = raw(batch)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                preds.append(out.squeeze(-1).cpu().numpy())

        return np.concatenate(preds)

    preds_clean = _predict_sequence(normal_features)
    preds_fault = _predict_sequence(corrupted_features)

    if len(preds_clean) == 0 or len(preds_fault) == 0:
        return {'error': 'sequence too short'}

    # 特征也需要对齐到窗口输出长度
    # 窗口化后第 i 个输出对应原始 cycle [i, i+window_size-1] 的最后一步
    T_out = len(preds_clean)
    feat_aligned = normal_features[window_size - 1:window_size - 1 + T_out]
    feat_fault_aligned = corrupted_features[window_size - 1:window_size - 1 + T_out]

    # fault_cycle 映射到输出序列索引
    fault_start_out = max(0, fault_cycle - window_size + 1)
    fault_start_out = min(fault_start_out, T_out - 1)

    # 3. 计算异常分数
    scores_clean = compute_anomaly_scores(
        preds_clean, feat_aligned, feature_mean, feature_std
    )
    scores_fault = compute_anomaly_scores(
        preds_fault, feat_fault_aligned, feature_mean, feature_std
    )

    # 4. 评估检测性能（各分数维度 + 综合）
    metrics = {}
    for score_key in ['input_zscore', 'mono_violation', 'rate_anomaly', 'combined']:
        det = evaluate_detection(
            scores_clean[score_key],
            scores_fault[score_key],
            fault_start_out,
        )
        metrics[score_key] = det

    return {
        'fault_cycle':    fault_cycle,
        'fault_start_out': fault_start_out,
        'severity':       severity,
        'T_output':       T_out,
        'metrics':        metrics,
        'preds_clean':    preds_clean,
        'preds_fault':    preds_fault,
        'scores_clean':   {k: v.tolist() for k, v in scores_clean.items()},
        'scores_fault':   {k: v.tolist() for k, v in scores_fault.items()},
    }
