"""
Exp-12：基于物理约束的电芯异常检测
====================================

核心思路：已训练的 PI-MS-CNN-LSTM 模型学到了"正常退化模式"。
当电池簇中某电芯突发失效时，其特征向量偏离正常模式，
模型的物理约束违反信号（单调违规率、速率突变）可作为零额外成本的异常检测器。

模块功能：
  1. 异常注入：5 种退化场景（见下）
  2. 异常评分：四维评分（输入 z-score / 单调违规 / 速率突变 / 综合）
  3. 检测评估：ROC-AUC / Detection Rate@FPR5% / 检测延迟

支持的 5 种退化场景（scenario 参数）：
  sudden_aging    — 单电芯突发加速老化（跨电芯特征替换，原始场景）
  knee_point      — 容量拐点突破（自身末期特征替换，无需供体）
  imbalance       — 簇内不均衡加剧（alpha 线性增长的渐进漂移）
  li_plating      — 析锂台阶突降（阶跃 + 持续混合）
  resistance_rise — 内阻渐进增长（特征向末期二次漂移，无需供体）
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


# ============================================================
# 1. 异常注入：跨电芯特征替换
# ============================================================

def _donor_terminal_sequence(donor_features: np.ndarray, n_cycles: int) -> np.ndarray:
    """
    从供体电芯取末期连续特征序列，长度为 n_cycles。
    始终从供体末期（最后 20%）往前取，避免误用供体健康早期数据。

    原始 bug：供体索引用"从末尾倒数映射"，当供体寿命 < 剩余故障周期时
    会 clamp 到供体起始（健康早期），导致注入的是正常特征而非异常特征。
    """
    T_donor = len(donor_features)
    # 取供体末期 20% 的起点，确保拿到退化特征
    terminal_start = max(0, int(T_donor * 0.80))
    terminal = donor_features[terminal_start:]   # 末期特征段，至少 20%

    # 如果末期段足够长，直接取最后 n_cycles；否则循环填充
    if len(terminal) >= n_cycles:
        return terminal[-n_cycles:]
    else:
        # 不足时从末期段循环复制
        repeats = (n_cycles // len(terminal)) + 1
        tiled = np.tile(terminal, (repeats, 1))
        return tiled[-n_cycles:]


def inject_cell_failure(
    normal_features: np.ndarray,
    donor_features: np.ndarray,
    fault_cycle: int,
    severity: str = 'moderate',
    alpha: Optional[float] = None,
) -> np.ndarray:
    """
    场景 1 — 单电芯突发加速老化（原始场景）

    从 fault_cycle 开始，将正常电芯的特征与供体末期特征混合。
    混合比 alpha 固定，模拟突发性失效。
    """
    severity_map = {
        'mild':     0.2,
        'moderate': 0.4,
        'severe':   0.7,
    }
    if alpha is None:
        alpha = severity_map[severity]

    corrupted = normal_features.copy()
    T = len(normal_features)
    n_fault = T - fault_cycle

    # 始终取供体末期特征（修复：不再从头映射）
    donor_seq = _donor_terminal_sequence(donor_features, n_fault)

    for i, t in enumerate(range(fault_cycle, T)):
        corrupted[t] = (1 - alpha) * normal_features[t] + alpha * donor_seq[i]

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
# 1b. 新增退化场景注入函数
# ============================================================

def inject_knee_point(
    normal_features: np.ndarray,
    fault_cycle: int,
    severity: str = 'moderate',
) -> np.ndarray:
    """
    场景 2 — 容量拐点突破（Capacity Knee-point）

    电芯在 fault_cycle 处进入加速退化阶段，使用电芯自身末期特征
    作为"加速后状态"吸引子，不需要供体电芯。

    物理机制：SEI 膜失控增厚 / 析锂累积达临界值 / 电解液耗尽
    曲线特征：折点前正常，折点后斜率骤增（SOH 出现明显弯折）
    最敏感信号：rate_anomaly（速率突变）
    """
    severity_map = {'mild': 0.20, 'moderate': 0.40, 'severe': 0.65}
    alpha = severity_map.get(severity, 0.40)

    T = len(normal_features)
    corrupted = normal_features.copy()

    # 取电芯自身末期 15% 的特征均值作为拐点后状态
    terminal_start = max(fault_cycle + 1, int(T * 0.85))
    if terminal_start >= T:
        terminal_start = T - 1
    terminal_mean = normal_features[terminal_start:].mean(axis=0)

    for t in range(fault_cycle, T):
        corrupted[t] = (1 - alpha) * normal_features[t] + alpha * terminal_mean

    return corrupted


def inject_imbalance_escalation(
    normal_features: np.ndarray,
    donor_features: np.ndarray,
    fault_cycle: int,
    severity: str = 'moderate',
) -> np.ndarray:
    """
    场景 3 — 簇内不均衡加剧（Inter-cell Imbalance Escalation）

    混合比例 alpha 从 0 线性增大到 max_alpha，供体电芯代表
    簇中退化最严重的电芯，模拟 BMS 簇级测量的渐进漂移。

    物理机制：热梯度导致的差异老化 / 制造差异激活
    曲线特征：从 fault_cycle 开始缓慢偏离正常，差距持续扩大
    最敏感信号：input_zscore（特征渐进漂移）
    """
    severity_map = {'mild': 0.20, 'moderate': 0.40, 'severe': 0.65}
    max_alpha = severity_map.get(severity, 0.40)

    T = len(normal_features)
    n_fault = T - fault_cycle
    corrupted = normal_features.copy()

    donor_seq = _donor_terminal_sequence(donor_features, n_fault)
    duration = max(1, n_fault - 1)

    for i, t in enumerate(range(fault_cycle, T)):
        alpha_t = max_alpha * i / duration        # 线性增长
        corrupted[t] = (1 - alpha_t) * normal_features[t] + alpha_t * donor_seq[i]

    return corrupted


def inject_lithium_plating(
    normal_features: np.ndarray,
    donor_features: np.ndarray,
    fault_cycle: int,
    severity: str = 'moderate',
) -> np.ndarray:
    """
    场景 4 — 析锂台阶突降（Lithium Plating Step-drop）

    fault_cycle 处出现一个大幅阶跃（step_alpha），之后以较小但
    持续的混合比（post_alpha）继续衰减，形成"台阶 + 加速"的双重特征。

    物理机制：高倍率 / 低温充电导致锂沉积，累积后突发不可逆容量损失
    曲线特征：明显台阶 + 台阶后更陡的斜率
    最敏感信号：mono_violation（阶跃破坏单调性）+ rate_anomaly
    """
    severity_map = {
        'mild':     (0.20, 0.15),
        'moderate': (0.45, 0.30),
        'severe':   (0.70, 0.55),
    }
    step_alpha, post_alpha = severity_map.get(severity, (0.45, 0.30))

    T = len(normal_features)
    n_fault = T - fault_cycle
    corrupted = normal_features.copy()

    donor_seq = _donor_terminal_sequence(donor_features, n_fault)

    for i, t in enumerate(range(fault_cycle, T)):
        alpha = step_alpha if i == 0 else post_alpha   # 第一步大跳，后续持续混合
        corrupted[t] = (1 - alpha) * normal_features[t] + alpha * donor_seq[i]

    return corrupted


def inject_resistance_rise(
    normal_features: np.ndarray,
    fault_cycle: int,
    severity: str = 'moderate',
) -> np.ndarray:
    """
    场景 5 — 内阻渐进增长（Progressive Internal Resistance Rise）

    特征向量向电芯自身末期状态渐进漂移，漂移量随时间二次增长
    （早期慢、后期快），不需要供体电芯。

    物理机制：SEI 持续增厚 / 电极颗粒开裂 / 接触电阻升高
    曲线特征：斜率持续增大，无折点无阶跃，最难早期检测
    最敏感信号：rate_anomaly（速率持续加大）
    """
    severity_map = {'mild': 0.15, 'moderate': 0.30, 'severe': 0.55}
    max_drift = severity_map.get(severity, 0.30)

    T = len(normal_features)
    corrupted = normal_features.copy()

    # 用末期 10% 特征均值作为漂移目标
    terminal_start = max(fault_cycle + 1, int(T * 0.90))
    if terminal_start >= T:
        terminal_start = T - 1
    drift_target = normal_features[terminal_start:].mean(axis=0)

    duration = max(1, T - fault_cycle)
    for t in range(fault_cycle, T):
        progress = (t - fault_cycle) / duration
        drift_alpha = max_drift * (progress ** 2)   # 二次增长：早期慢，后期加速
        corrupted[t] = (1 - drift_alpha) * normal_features[t] + drift_alpha * drift_target

    return corrupted


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
        _safe_normalize(mono_violation),   # 与其他信号等权（异常主要表现为下跌而非上升）
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
    donor_features: Optional[np.ndarray],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    fault_cycle: int,
    severity: str,
    scenario: str = 'sudden_aging',
    window_size: int = 40,
    device: str = 'cuda',
) -> Dict:
    """
    对单块电池执行完整的异常检测流程：注入 → 推理 → 评分 → 评估。

    Args:
        model:           训练好的模型
        normal_features: (T_raw, F) 正常电芯原始特征（已标准化）
        donor_features:  (T_donor, F) 供体电芯特征；knee_point / resistance_rise 场景传 None
        feature_mean:    (F,) 训练集特征均值
        feature_std:     (F,) 训练集特征标准差
        fault_cycle:     故障注入时刻（原始 cycle 索引）
        severity:        严重程度（'mild' / 'moderate' / 'severe'）
        scenario:        退化场景（见模块文档）
        window_size:     模型窗口大小
        device:          计算设备

    Returns:
        result: 包含评分和评估指标的字典
    """
    # 1. 根据 scenario 选择注入函数
    if scenario == 'sudden_aging':
        corrupted_features = inject_cell_failure(
            normal_features, donor_features, fault_cycle, severity)
    elif scenario == 'knee_point':
        corrupted_features = inject_knee_point(
            normal_features, fault_cycle, severity)
    elif scenario == 'imbalance':
        corrupted_features = inject_imbalance_escalation(
            normal_features, donor_features, fault_cycle, severity)
    elif scenario == 'li_plating':
        corrupted_features = inject_lithium_plating(
            normal_features, donor_features, fault_cycle, severity)
    elif scenario == 'resistance_rise':
        corrupted_features = inject_resistance_rise(
            normal_features, fault_cycle, severity)
    else:
        raise ValueError(f"Unknown scenario: {scenario!r}")

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
