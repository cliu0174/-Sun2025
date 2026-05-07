"""
论文预测曲线绘图 —— 需在服务器运行（需要 GPU 训练）

功能：
  1. 跑 E0 (PI-CNN-LSTM) 和 Exp09c (PI-MS-CNN-LSTM)，各 1 次
  2. 保存逐电池预测值到 .npz
  3. 生成 Fig7（预测曲线对比）和 Fig8（散点图）

配置：r=0.3, seed=929（最优配置）
耗时估计：每组 ~5 分钟，共 ~10 分钟

用法：
  python scripts/run_and_plot_predictions.py                # 完整运行
  python scripts/run_and_plot_predictions.py --plot-only     # 只从缓存画图
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# 添加项目根目录到 sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

CACHE_DIR = os.path.join(ROOT, 'figures', 'pred_cache')
OUTPUT_DIR = os.path.join(ROOT, 'figures')
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================================
# 配置
# ============================================================
SEED = 929
RATIO = 0.3
DEVICE = 'cuda'

CONFIGS = {
    'E0': {
        'model_type': 'cnn_lstm',
        'override': None,
        'label': 'E0 (PI-CNN-LSTM)',
    },
    'Exp09c': {
        'model_type': 'ms_cnn_lstm_v2',
        'override': {
            'training': {
                'use_physics': True,
                'physics_loss': {
                    'monotonic_weight': 0.3,
                    'boundary_weight': 0.0,
                    'tolerance': 0.005,
                    'min_cycle': 300,
                },
            },
            'model': {
                'mc_dropout': {'enabled': False},
            },
        },
        'label': 'Exp09c (PI-MS-CNN-LSTM)',
    },
}


# ============================================================
# 运行推理并缓存
# ============================================================
def run_and_cache(exp_id, config):
    """训练模型并缓存预测结果"""
    cache_file = os.path.join(CACHE_DIR, f'{exp_id}_r{RATIO}_s{SEED}.npz')

    if os.path.exists(cache_file):
        print(f'  [CACHE HIT] {cache_file}')
        return np.load(cache_file, allow_pickle=True)

    from train_cross_battery import train_cross_battery_model

    print(f'\n  [TRAIN] {exp_id}: {config["label"]}')
    print(f'          ratio={RATIO}, seed={SEED}')

    model, results, data_dict = train_cross_battery_model(
        model_type=config['model_type'],
        seed=SEED,
        device=DEVICE,
        supervision_ratio=RATIO,
        supervision_seed=None,
        config_override=config['override'],
    )

    preds = np.array(results['predictions'])
    targets = np.array(results['targets'])
    battery_ids = np.array(results['battery_ids']) if results.get('battery_ids') is not None else None

    np.savez(cache_file,
             predictions=preds,
             targets=targets,
             battery_ids=battery_ids)

    print(f'  [SAVED] {cache_file}')
    print(f'          MAE={results["test_mae"]:.4f}%, R2={results["test_r2"]:.4f}')
    return np.load(cache_file, allow_pickle=True)


# ============================================================
# Fig 7: 预测曲线对比（逐电池）
# ============================================================
def plot_fig7(data_e0, data_exp09c):
    """选 3 块代表性电池展示预测 vs 真值曲线"""

    bids_e0 = data_e0['battery_ids']
    preds_e0 = data_e0['predictions']
    targets_e0 = data_e0['targets']

    bids_exp09c = data_exp09c['battery_ids']
    preds_exp09c = data_exp09c['predictions']
    targets_exp09c = data_exp09c['targets']

    # 获取所有测试电池
    unique_batteries = sorted(set(bids_e0))

    # 计算每块电池的 MAE 差值（Exp09c - E0），选 3 块代表性电池
    battery_info = []
    for bid in unique_batteries:
        mask_e0 = bids_e0 == bid
        mask_exp09c = bids_exp09c == bid
        if mask_e0.sum() > 0 and mask_exp09c.sum() > 0:
            mae_e0 = np.mean(np.abs(preds_e0[mask_e0] - targets_e0[mask_e0]))
            mae_exp09c = np.mean(np.abs(preds_exp09c[mask_exp09c] - targets_exp09c[mask_exp09c]))
            n_samples = mask_e0.sum()
            battery_info.append({
                'bid': bid,
                'mae_e0': mae_e0,
                'mae_exp09c': mae_exp09c,
                'improvement': (mae_e0 - mae_exp09c) / mae_e0 * 100,
                'n_samples': n_samples,
            })

    battery_info.sort(key=lambda x: x['improvement'], reverse=True)

    # 选 3 块：最大改善、中位改善、最小改善（或退化）
    n = len(battery_info)
    selected = [
        battery_info[0],            # 最大改善
        battery_info[n // 2],       # 中位
        battery_info[-1],           # 最小改善/退化
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for ax_idx, info in enumerate(selected):
        ax = axes[ax_idx]
        bid = info['bid']

        mask_e0 = bids_e0 == bid
        mask_exp09c = bids_exp09c == bid

        t_e0 = targets_e0[mask_e0]
        p_e0 = preds_e0[mask_e0]
        t_exp09c = targets_exp09c[mask_exp09c]
        p_exp09c = preds_exp09c[mask_exp09c]

        cycles = np.arange(len(t_e0))

        ax.plot(cycles, t_e0, 'k-', linewidth=1.5, label='True SOH', alpha=0.8)
        ax.plot(cycles, p_e0, '--', color='#2196F3', linewidth=1.2,
               label=f'E0 (MAE={info["mae_e0"]*100:.3f}%)')
        ax.plot(cycles[:len(p_exp09c)], p_exp09c, '--', color='#4CAF50', linewidth=1.2,
               label=f'Exp09c (MAE={info["mae_exp09c"]*100:.3f}%)')

        ax.set_xlabel('Cycle Window')
        ax.set_title(f'Battery {bid}\n(Improvement: {info["improvement"]:+.1f}%)',
                    fontsize=10)
        ax.legend(fontsize=7, loc='lower left')

    axes[0].set_ylabel('SOH')
    plt.suptitle(f'Prediction Curves: E0 vs Exp09c (r={RATIO}, seed={SEED})', fontsize=12)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'fig7_prediction_curves.png')
    plt.savefig(path)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f'  [OK] Fig 7 saved: {path}')


# ============================================================
# Fig 8: 散点图（pred vs true）
# ============================================================
def plot_fig8(data_e0, data_exp09c):
    """pred vs true 散点图，E0 vs Exp09c 并排"""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    datasets = [
        ('E0 (PI-CNN-LSTM)', data_e0, '#2196F3'),
        ('Exp09c (PI-MS-CNN-LSTM)', data_exp09c, '#4CAF50'),
    ]

    for ax_idx, (label, data, color) in enumerate(datasets):
        ax = axes[ax_idx]

        preds = data['predictions']
        targets = data['targets']

        mae = np.mean(np.abs(preds - targets)) * 100
        r2 = 1 - np.sum((preds - targets)**2) / np.sum((targets - np.mean(targets))**2)

        ax.scatter(targets, preds, s=3, alpha=0.3, c=color, edgecolors='none')

        # 对角线
        lims = [min(targets.min(), preds.min()) - 0.02,
                max(targets.max(), preds.max()) + 0.02]
        ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5, label='Ideal (y=x)')

        # 指标文字
        ax.text(0.05, 0.92, f'MAE = {mae:.4f}%\nR$^2$ = {r2:.4f}',
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('True SOH')
        ax.set_ylabel('Predicted SOH')
        ax.set_title(label, fontweight='bold')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Prediction Scatter Plot (r={RATIO}, seed={SEED})', fontsize=12)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'fig8_scatter_plot.png')
    plt.savefig(path)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f'  [OK] Fig 8 saved: {path}')


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true',
                       help='Only plot from cached .npz files (skip training)')
    args = parser.parse_args()

    print('=' * 60)
    print('Paper Prediction Figures (Fig 7 & Fig 8)')
    print(f'Config: ratio={RATIO}, seed={SEED}')
    print('=' * 60)

    if args.plot_only:
        # 从缓存加载
        cache_e0 = os.path.join(CACHE_DIR, f'E0_r{RATIO}_s{SEED}.npz')
        cache_exp09c = os.path.join(CACHE_DIR, f'Exp09c_r{RATIO}_s{SEED}.npz')

        if not os.path.exists(cache_e0) or not os.path.exists(cache_exp09c):
            print('ERROR: Cache files not found. Run without --plot-only first.')
            sys.exit(1)

        data_e0 = np.load(cache_e0, allow_pickle=True)
        data_exp09c = np.load(cache_exp09c, allow_pickle=True)
    else:
        # 训练并缓存
        data_e0 = run_and_cache('E0', CONFIGS['E0'])
        data_exp09c = run_and_cache('Exp09c', CONFIGS['Exp09c'])

    print('\nGenerating figures...')
    plot_fig7(data_e0, data_exp09c)
    plot_fig8(data_e0, data_exp09c)

    print('\n' + '=' * 60)
    print(f'All done. Figures in: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
