"""
scripts/export_table48.py
=========================
导出表4-8：部分监督条件下 CNN-LSTM vs PI-MSCL 多指标对比
  - 监督比例：r = 1.0 / 0.7 / 0.5 / 0.3
  - 指标：MAE(%) / RMSE(%) / R²
  - 每个配置运行 N_SEEDS 次取均值（默认 5 次）
  - 结果缓存到 figures/pred_cache/table48_*.json，支持断点续跑

用法：
    python scripts/export_table48.py             # 完整跑（首次约 30~60 min GPU）
    python scripts/export_table48.py --seeds 1   # 快速验证，每配置只跑 1 次
    python scripts/export_table48.py --device cpu
"""

import os, sys, json, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CACHE_DIR  = os.path.join(ROOT, 'figures', 'pred_cache')
OUTPUT_DIR = os.path.join(ROOT, 'figures')
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 实验配置 ──────────────────────────────────────────────────────
RATIOS  = [1.0, 0.7, 0.5, 0.3]
SEEDS   = [42, 123, 34, 999, 1024]   # 默认 5 个种子

_NO_PI = {'enabled': False, 'monotonic_weight': 0.0,
          'boundary_weight': 0.0, 'smoothness_weight': 0.0}
_PI_CONFIG = {
    'enabled': True, 'monotonic_weight': 0.3, 'boundary_weight': 0.0,
    'smoothness_weight': 0.0, 'monotonic_tolerance': 0.005,
    'min_cycle': 300, 'base_loss_weight': 1.0,
}

MODELS = {
    'cnn_lstm': {
        'model_type': 'cnn_lstm',
        'override'  : {'physics_constraints': _NO_PI},
        'label'     : 'CNN-LSTM',
    },
    'pi_ms_cnn_lstm': {
        'model_type': 'ms_cnn_lstm_v2',
        'override'  : {
            'architecture'       : {'use_multiscale': True, 'per_window_norm': False},
            'physics_constraints': _PI_CONFIG,
        },
        'label': 'PI-MSCL',
    },
}


def _cache_path(exp_id, ratio, seed):
    return os.path.join(
        CACHE_DIR,
        f'table48_{exp_id}_r{ratio}_s{seed}.json'
    )


def run_one(exp_id, cfg, ratio, seed, device):
    """训练并返回 {mae, rmse, r2}，结果缓存到 JSON。"""
    cp = _cache_path(exp_id, ratio, seed)
    if os.path.exists(cp):
        with open(cp) as f:
            return json.load(f)

    from train_cross_battery import train_cross_battery_model
    print(f'  [TRAIN] {exp_id}  r={ratio}  seed={seed}')

    _, result, _ = train_cross_battery_model(
        model_type        = cfg['model_type'],
        device            = device,
        seed              = seed,
        supervision_ratio = ratio,
        supervision_seed  = None,
        config_override   = cfg['override'],
    )

    rec = {
        'mae' : float(result['test_mae']),
        'rmse': float(result['test_rmse']),
        'r2'  : float(result['test_r2']),
    }
    with open(cp, 'w') as f:
        json.dump(rec, f)
    print(f'    MAE={rec["mae"]*100:.4f}%  RMSE={rec["rmse"]*100:.4f}%  R²={rec["r2"]:.4f}')
    return rec


def collect(seeds, device):
    """收集所有配置的结果，返回嵌套字典 results[exp_id][ratio] = {mae,rmse,r2}（均值）。"""
    results = {eid: {} for eid in MODELS}

    for exp_id, cfg in MODELS.items():
        for ratio in RATIOS:
            recs = []
            for seed in seeds:
                rec = run_one(exp_id, cfg, ratio, seed, device)
                recs.append(rec)

            results[exp_id][ratio] = {
                'mae_mean' : np.mean([r['mae']  for r in recs]),
                'rmse_mean': np.mean([r['rmse'] for r in recs]),
                'r2_mean'  : np.mean([r['r2']   for r in recs]),
                'mae_std'  : np.std( [r['mae']  for r in recs]),
                'n'        : len(recs),
            }

    return results


def print_and_save(results):
    """打印表格并保存 CSV。"""
    import csv

    print('\n' + '='*90)
    print('表4-8  部分监督条件下 CNN-LSTM vs PI-MSCL 多指标对比')
    print('='*90)
    print(f"{'监督比例':<10} "
          f"{'CNN-LSTM':^36}  {'PI-MSCL':^36}  {'ΔMAE':>6}")
    print(f"{'':^10} "
          f"{'MAE(%)':>8} {'RMSE(%)':>9} {'R²':>7}  "
          f"{'MAE(%)':>8} {'RMSE(%)':>9} {'R²':>7}  {'(%)':>6}")
    print('-'*90)

    rows = []
    for ratio in RATIOS:
        c = results['cnn_lstm'][ratio]
        p = results['pi_ms_cnn_lstm'][ratio]
        delta = (c['mae_mean'] - p['mae_mean']) * 100   # 正数 = PI-MSCL 更好
        rows.append({
            'ratio'      : ratio,
            'cnn_mae'    : c['mae_mean']  * 100,
            'cnn_rmse'   : c['rmse_mean'] * 100,
            'cnn_r2'     : c['r2_mean'],
            'pi_mae'     : p['mae_mean']  * 100,
            'pi_rmse'    : p['rmse_mean'] * 100,
            'pi_r2'      : p['r2_mean'],
            'delta_mae'  : delta,
        })
        print(f"r={ratio:<7.1f} "
              f"{c['mae_mean']*100:>8.4f} {c['rmse_mean']*100:>9.4f} {c['r2_mean']:>7.4f}  "
              f"{p['mae_mean']*100:>8.4f} {p['rmse_mean']*100:>9.4f} {p['r2_mean']:>7.4f}  "
              f"{delta:>+6.4f}")

    print('='*90)
    print('注：ΔMAE = CNN-LSTM MAE − PI-MSCL MAE（正值表示 PI-MSCL 更优）')

    # 保存 CSV
    csv_path = os.path.join(OUTPUT_DIR, 'table4_8_partial_supervision.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=[
            'ratio',
            'cnn_mae', 'cnn_rmse', 'cnn_r2',
            'pi_mae',  'pi_rmse',  'pi_r2',
            'delta_mae',
        ])
        w.writeheader()
        for r in rows:
            w.writerow({k: f'{v:.4f}' for k, v in r.items()})
    print(f'[OK] CSV → {csv_path}')


def main():
    parser = argparse.ArgumentParser(description='Export Table 4-8')
    parser.add_argument('--seeds', type=int, default=len(SEEDS),
                        help=f'使用前 N 个种子（默认 {len(SEEDS)}）')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    seeds  = SEEDS[:args.seeds]
    device = args.device

    print(f'配置：{len(MODELS)} 模型 × {len(RATIOS)} 比例 × {len(seeds)} 种子'
          f' = {len(MODELS)*len(RATIOS)*len(seeds)} 次训练')
    print(f'种子：{seeds}  设备：{device}\n')

    results = collect(seeds, device)
    print_and_save(results)


if __name__ == '__main__':
    main()
