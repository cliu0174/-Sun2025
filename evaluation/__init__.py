"""评估工具集（M7：置信区间评估 | M8：特征缺失鲁棒性）"""
from .uncertainty_eval import compute_picp, compute_mpiw, uncertainty_report
from .robustness_eval import (
    evaluate_robustness,
    print_robustness_report,
    perturb_mask_features,
    perturb_gaussian_noise,
    perturb_drift,
)

__all__ = [
    # M7
    'compute_picp', 'compute_mpiw', 'uncertainty_report',
    # M8
    'evaluate_robustness', 'print_robustness_report',
    'perturb_mask_features', 'perturb_gaussian_noise', 'perturb_drift',
]
