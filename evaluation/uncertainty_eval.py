"""
M7：置信区间评估指标
====================
基于 MC Dropout 采样结果计算不确定性量化指标。

指标：
    PICP  (Prediction Interval Coverage Probability)
          95% 置信区间内真实值的覆盖比例，理论值应接近 0.95
    MPIW  (Mean Prediction Interval Width)
          置信区间平均宽度，越小越好（精度高、不确定性小）
    ENCE  (Expected Normalized Calibration Error)
          不确定性校准误差，越小越好
    Spearman(|error|, std)
          误差绝对值与预测标准差的 Spearman 相关系数，
          正相关说明模型在误差大的地方也给出了更大的不确定性（好的校准）
"""

import numpy as np
from scipy import stats


def compute_picp(targets: np.ndarray,
                 pred_mean: np.ndarray,
                 pred_std: np.ndarray,
                 confidence: float = 0.95) -> float:
    """
    计算预测区间覆盖概率（PICP）。

    Args:
        targets:    真实值，shape (N,)
        pred_mean:  预测均值，shape (N,)
        pred_std:   预测标准差，shape (N,)
        confidence: 置信水平（默认 0.95）

    Returns:
        picp: 覆盖比例，理想值 = confidence
    """
    z = stats.norm.ppf((1 + confidence) / 2)  # 1.96 for 95%
    lower = pred_mean - z * pred_std
    upper = pred_mean + z * pred_std
    covered = ((targets >= lower) & (targets <= upper)).mean()
    return float(covered)


def compute_mpiw(pred_mean: np.ndarray,
                 pred_std: np.ndarray,
                 confidence: float = 0.95) -> float:
    """
    计算预测区间平均宽度（MPIW）。

    Args:
        pred_mean:  预测均值，shape (N,)
        pred_std:   预测标准差，shape (N,)
        confidence: 置信水平（默认 0.95）

    Returns:
        mpiw: 平均宽度（SOH 单位，即 [0,1] 量纲）
    """
    z = stats.norm.ppf((1 + confidence) / 2)
    width = 2 * z * pred_std
    return float(width.mean())


def compute_spearman(targets: np.ndarray,
                     pred_mean: np.ndarray,
                     pred_std: np.ndarray) -> float:
    """
    计算 |误差| 与预测标准差的 Spearman 相关系数。

    正值（尤其 > 0.3）说明不确定性与误差正相关，即模型"知道自己哪里不确定"。
    """
    abs_error = np.abs(targets - pred_mean)
    rho, _ = stats.spearmanr(abs_error, pred_std)
    return float(rho)


def uncertainty_report(targets: np.ndarray,
                        pred_mean: np.ndarray,
                        pred_std: np.ndarray,
                        confidence: float = 0.95,
                        verbose: bool = True) -> dict:
    """
    完整不确定性评估报告。

    Args:
        targets:    真实值，shape (N,)
        pred_mean:  MC Dropout 预测均值，shape (N,)
        pred_std:   MC Dropout 预测标准差，shape (N,)
        confidence: 置信水平（默认 0.95）
        verbose:    是否打印结果

    Returns:
        report: 包含所有指标的字典
    """
    picp    = compute_picp(targets, pred_mean, pred_std, confidence)
    mpiw    = compute_mpiw(pred_mean, pred_std, confidence)
    spearman = compute_spearman(targets, pred_mean, pred_std)

    # 点估计误差（用 mean 作为点预测）
    mae  = float(np.mean(np.abs(targets - pred_mean)))
    rmse = float(np.sqrt(np.mean((targets - pred_mean) ** 2)))

    report = {
        'picp':          picp,
        'mpiw':          mpiw,
        'spearman_corr': spearman,
        'mae':           mae,
        'rmse':          rmse,
        'confidence':    confidence,
        'n_samples':     len(targets),
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"不确定性评估报告（{confidence*100:.0f}% 置信区间）")
        print(f"{'='*50}")
        print(f"  PICP  : {picp:.4f}  （理想值 = {confidence:.2f}）")
        print(f"  MPIW  : {mpiw*100:.4f}%  （区间平均宽度）")
        print(f"  Spearman(|err|, std): {spearman:.4f}")
        print(f"  MAE   : {mae*100:.4f}%")
        print(f"  RMSE  : {rmse*100:.4f}%")

    return report
