"""
M8：特征缺失鲁棒性评估
========================
在测试集上模拟真实工况中的传感器故障/缺失/漂移场景，
对比 Baseline 与 Full Stack 方法的性能衰减幅度，
验证物理约束和伪标签在噪声环境下的鲁棒性提升。

三类扰动场景：
    Scene A  随机特征置零（模拟传感器完全失效）
             mask_k = 0 / 1 / 2 / 3 维特征（随机选取）
    Scene B  高斯噪声污染（模拟传感器测量误差）
             sigma = 0.01 / 0.05 / 0.1（相对标准归一化特征的幅度）
    Scene C  传感器系统性漂移（模拟长期标定偏差）
             drift = +0.05 / +0.1 / +0.2（叠加到所有特征上的常量偏置）

评估指标：
    - MAE_clean     无扰动的干净测试集误差
    - MAE_perturbed 扰动后的误差
    - delta_MAE     = MAE_perturbed - MAE_clean（衰减量）
    - rel_degradation = delta_MAE / MAE_clean × 100%（相对衰减比）
"""

import numpy as np
import torch
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────────────────────
# 场景 A：随机特征置零
# ──────────────────────────────────────────────────────────────

def perturb_mask_features(x: torch.Tensor,
                          n_mask: int,
                          rng: np.random.RandomState) -> torch.Tensor:
    """
    将输入张量的 n_mask 个特征维度随机置零。

    Args:
        x:      形状 (B, T, F) 或 (B, F)
        n_mask: 要屏蔽的特征数量（沿最后一维）
        rng:    NumPy 随机状态（保证可复现）

    Returns:
        x_perturbed: 置零后的张量，形状与输入相同
    """
    if n_mask <= 0:
        return x.clone()

    x_p = x.clone()
    n_feat = x.shape[-1]
    n_mask = min(n_mask, n_feat)
    mask_dims = rng.choice(n_feat, size=n_mask, replace=False)

    x_p[..., mask_dims] = 0.0
    return x_p


# ──────────────────────────────────────────────────────────────
# 场景 B：高斯噪声
# ──────────────────────────────────────────────────────────────

def perturb_gaussian_noise(x: torch.Tensor,
                            sigma: float,
                            rng: np.random.RandomState) -> torch.Tensor:
    """
    对输入张量叠加零均值高斯噪声。

    Args:
        x:     形状 (B, T, F) 或 (B, F)
        sigma: 噪声标准差（相对于标准化特征的尺度）
        rng:   NumPy 随机状态

    Returns:
        x_perturbed: 加噪后的张量
    """
    if sigma <= 0:
        return x.clone()

    noise = torch.from_numpy(rng.normal(0, sigma, x.shape).astype(np.float32))
    return x + noise.to(x.device)


# ──────────────────────────────────────────────────────────────
# 场景 C：系统性漂移
# ──────────────────────────────────────────────────────────────

def perturb_drift(x: torch.Tensor, drift: float) -> torch.Tensor:
    """
    对所有特征施加常量偏置，模拟传感器系统性漂移。

    Args:
        x:     形状 (B, T, F) 或 (B, F)
        drift: 偏置量（直接叠加到标准化特征上）

    Returns:
        x_perturbed: 漂移后的张量
    """
    return x + drift


# ──────────────────────────────────────────────────────────────
# 核心评估函数
# ──────────────────────────────────────────────────────────────

def _eval_loader(model: torch.nn.Module,
                 loader: DataLoader,
                 device: str,
                 perturb_fn=None) -> dict:
    """
    在 DataLoader 上计算 MAE / RMSE，可选地对每个 batch 施加扰动。

    Args:
        model:       已训练好的模型（eval 模式）
        loader:      测试集 DataLoader
        device:      'cuda' 或 'cpu'
        perturb_fn:  接受 (x_tensor) 返回 (x_perturbed) 的函数；None 表示无扰动

    Returns:
        {'mae': float, 'rmse': float, 'n': int}
    """
    model.eval()
    total_ae = 0.0
    total_se = 0.0
    n = 0

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict) and 'window' in batch:
                x = batch['window'].to(device)
                y = batch['target_soh'].to(device)
            elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                x = batch[0].to(device)
                y = batch[1].to(device)
            else:
                continue

            if perturb_fn is not None:
                x = perturb_fn(x)

            pred = model(x)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            pred = pred.squeeze(-1)
            y    = y.squeeze(-1)

            total_ae += torch.abs(pred - y).sum().item()
            total_se += ((pred - y) ** 2).sum().item()
            n        += y.numel()

    if n == 0:
        return {'mae': float('nan'), 'rmse': float('nan'), 'n': 0}

    return {
        'mae':  total_ae / n,
        'rmse': (total_se / n) ** 0.5,
        'n':    n,
    }


def evaluate_robustness(model: torch.nn.Module,
                        test_loader: DataLoader,
                        device: str,
                        seed: int = 42) -> dict:
    """
    完整鲁棒性评估：对三类扰动的各强度等级逐一测试。

    Args:
        model:       已训练好的模型
        test_loader: 测试集 DataLoader（batch 含 'window' 和 'target_soh'）
        device:      计算设备
        seed:        随机种子，保证扰动可复现

    Returns:
        results: 嵌套 dict，结构为:
            {
              'clean': {'mae': ..., 'rmse': ...},
              'scene_a': {
                  'n_mask_1': {'mae': ..., 'rmse': ..., 'delta_mae': ..., 'rel_deg': ...},
                  'n_mask_2': {...},
                  'n_mask_3': {...},
              },
              'scene_b': {
                  'sigma_001': {...},
                  'sigma_005': {...},
                  'sigma_010': {...},
              },
              'scene_c': {
                  'drift_005': {...},
                  'drift_010': {...},
                  'drift_020': {...},
              },
            }
    """
    results = {}

    # ── 干净基准 ──────────────────────────────────────────────
    clean = _eval_loader(model, test_loader, device)
    results['clean'] = clean
    mae_clean = clean['mae']

    # ── Scene A：随机特征置零 ─────────────────────────────────
    results['scene_a'] = {}
    for n_mask in [1, 2, 3]:
        rng = np.random.RandomState(seed + n_mask)

        def _fn_a(x, _nm=n_mask, _rng=rng):
            return perturb_mask_features(x, _nm, _rng)

        metrics = _eval_loader(model, test_loader, device, perturb_fn=_fn_a)
        delta = metrics['mae'] - mae_clean
        metrics['delta_mae'] = delta
        metrics['rel_degradation'] = (delta / mae_clean * 100) if mae_clean > 0 else float('nan')
        results['scene_a'][f'n_mask_{n_mask}'] = metrics

    # ── Scene B：高斯噪声 ─────────────────────────────────────
    results['scene_b'] = {}
    for sigma in [0.01, 0.05, 0.10]:
        rng = np.random.RandomState(seed + int(sigma * 1000))

        def _fn_b(x, _sigma=sigma, _rng=rng):
            return perturb_gaussian_noise(x, _sigma, _rng)

        metrics = _eval_loader(model, test_loader, device, perturb_fn=_fn_b)
        delta = metrics['mae'] - mae_clean
        metrics['delta_mae'] = delta
        metrics['rel_degradation'] = (delta / mae_clean * 100) if mae_clean > 0 else float('nan')
        key = f"sigma_{str(sigma).replace('.', '').ljust(3, '0')}"
        results['scene_b'][key] = metrics

    # ── Scene C：系统性漂移 ───────────────────────────────────
    results['scene_c'] = {}
    for drift in [0.05, 0.10, 0.20]:

        def _fn_c(x, _d=drift):
            return perturb_drift(x, _d)

        metrics = _eval_loader(model, test_loader, device, perturb_fn=_fn_c)
        delta = metrics['mae'] - mae_clean
        metrics['delta_mae'] = delta
        metrics['rel_degradation'] = (delta / mae_clean * 100) if mae_clean > 0 else float('nan')
        key = f"drift_{str(drift).replace('.', '').ljust(3, '0')}"
        results['scene_c'][key] = metrics

    return results


def print_robustness_report(results: dict, model_name: str = '') -> None:
    """打印鲁棒性评估报告。"""
    header = f"鲁棒性评估报告{'  — ' + model_name if model_name else ''}"
    print(f"\n{'='*65}")
    print(header)
    print(f"{'='*65}")

    clean_mae = results['clean']['mae'] * 100
    print(f"  干净基准 MAE : {clean_mae:.4f}%")

    def _row(label, m):
        mae_p = m['mae'] * 100
        delta = m.get('delta_mae', float('nan')) * 100
        rel   = m.get('rel_degradation', float('nan'))
        return (f"  {label:<25} MAE={mae_p:7.4f}%  "
                f"Δ={delta:+7.4f}%  rel_deg={rel:+6.1f}%")

    print(f"\n  [场景 A：随机特征置零]")
    for k, v in results.get('scene_a', {}).items():
        n = k.split('_')[-1]
        print(_row(f"mask {n} 维", v))

    print(f"\n  [场景 B：高斯噪声]")
    for k, v in results.get('scene_b', {}).items():
        sigma_str = k.replace('sigma_', 'σ=0.')
        print(_row(sigma_str, v))

    print(f"\n  [场景 C：系统性漂移]")
    for k, v in results.get('scene_c', {}).items():
        drift_str = k.replace('drift_', 'drift=+0.')
        print(_row(drift_str, v))

    print()
