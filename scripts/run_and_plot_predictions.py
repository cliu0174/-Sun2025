"""
论文预测曲线绘图 —— 需在服务器运行（需要 GPU 训练）

功能：
  1. 训练 7 个模型（LSTM / GRU / CNN-LSTM 无物理 / E0 / E2+MC / A1 无物理 / Exp09c）
  2. 保存逐电池预测值到 .npz（支持断点续跑）
  3. 生成 Fig7（多模型预测曲线对比）和 Fig8（多模型散点图）

配置：r=0.3, seed=929（Exp09c 最优场景）
耗时估计：每组 ~5 分钟，共 ~35 分钟（缓存命中则跳过）

用法：
  python scripts/run_and_plot_predictions.py                # 完整运行
  python scripts/run_and_plot_predictions.py --plot-only     # 只从缓存画图
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
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

_PI = {
    'enabled': True,
    'monotonic_weight': 0.3,
    'boundary_weight': 0.0,
    'smoothness_weight': 0.0,
    'monotonic_tolerance': 0.005,
    'min_cycle': 300,
}
_NO_PI = {
    'enabled': False,
    'monotonic_weight': 0.0,
    'boundary_weight': 0.0,
    'smoothness_weight': 0.0,
}

# 7 个模型，从弱到强排列
CONFIGS = OrderedDict([
    ('B1_LSTM', {
        'model_type': 'lstm',
        'override': {'physics_constraints': _NO_PI},
        'label': 'LSTM',
        'color': '#BDBDBD',
        'linestyle': ':',
    }),
    ('B2_GRU', {
        'model_type': 'gru',
        'override': {'physics_constraints': _NO_PI},
        'label': 'GRU',
        'color': '#9E9E9E',
        'linestyle': ':',
    }),
    ('Eneg1', {
        'model_type': 'cnn_lstm',
        'override': {'physics_constraints': _NO_PI},
        'label': 'CNN-LSTM (no physics)',
        'color': '#FF9800',
        'linestyle': '--',
    }),
    ('E0', {
        'model_type': 'cnn_lstm',
        'override': {'physics_constraints': {**_PI}},
        'label': 'PI-CNN-LSTM (E0)',
        'color': '#2196F3',
        'linestyle': '--',
    }),
    ('E2_MC', {
        'model_type': 'cnn_lstm_mc',
        'override': {
            'physics_constraints': {**_PI},
            'model': {'mc_dropout': {'enabled': True, 'rate': 0.1}},
        },
        'label': 'PI-CNN-LSTM+MC (E2)',
        'color': '#F44336',
        'linestyle': '--',
    }),
    ('A1', {
        'model_type': 'ms_cnn_lstm_v2',
        'override': {
            'architecture': {'use_multiscale': True, 'per_window_norm': False},
            'physics_constraints': _NO_PI,
        },
        'label': 'MS-CNN-LSTM (A1)',
        'color': '#FF7043',
        'linestyle': '-.',
    }),
    ('Exp09c', {
        'model_type': 'ms_cnn_lstm_v2',
        'override': {
            'architecture': {'use_multiscale': True, 'per_window_norm': False},
            'physics_constraints': {**_PI, 'base_loss_weight': 1.0},
        },
        'label': 'PI-MS-CNN-LSTM [Ours]',
        'color': '#4CAF50',
        'linestyle': '-',
    }),
])


# ============================================================
# 运行推理并缓存
# ============================================================
def run_and_cache(exp_id, config):
    """训练模型并缓存预测结果"""
    cache_file = os.path.join(CACHE_DIR, f'{exp_id}_r{RATIO}_s{SEED}.npz')

    if os.path.exists(cache_file):
        print(f'  [CACHE HIT] {exp_id}')
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
             battery_ids=battery_ids,
             mae=results['test_mae'],
             r2=results['test_r2'])

    print(f'  [SAVED] {cache_file}')
    print(f'          MAE={results["test_mae"]*100:.4f}%, R2={results["test_r2"]:.4f}')
    return np.load(cache_file, allow_pickle=True)


# ============================================================
# Fig 7: 多模型预测曲线对比（选 1 块代表性电池）
# ============================================================
def plot_fig7(all_data):
    """在 1 块电池上展示所有模型的预测 vs 真值"""

    # 用 Exp09c 的数据找一块 Exp09c 改善最大的电池
    ref_key = 'Exp09c'
    base_key = 'E0'

    bids_ref = all_data[ref_key]['battery_ids']
    preds_ref = all_data[ref_key]['predictions']
    targets_ref = all_data[ref_key]['targets']
    bids_base = all_data[base_key]['battery_ids']
    preds_base = all_data[base_key]['predictions']
    targets_base = all_data[base_key]['targets']

    unique_batteries = sorted(set(bids_ref))

    # 找改善最大的电池
    best_bid, best_improve = None, -999
    for bid in unique_batteries:
        m_ref = bids_ref == bid
        m_base = bids_base == bid
        if m_ref.sum() > 20 and m_base.sum() > 20:
            mae_ref = np.mean(np.abs(preds_ref[m_ref] - targets_ref[m_ref]))
            mae_base = np.mean(np.abs(preds_base[m_base] - targets_base[m_base]))
            improve = (mae_base - mae_ref) / mae_base * 100
            if improve > best_improve:
                best_improve = improve
                best_bid = bid

    # 画图：2 块电池（最大改善 + 中位）
    battery_maes = []
    for bid in unique_batteries:
        m_ref = bids_ref == bid
        m_base = bids_base == bid
        if m_ref.sum() > 20 and m_base.sum() > 20:
            mae_base = np.mean(np.abs(preds_base[m_base] - targets_base[m_base]))
            mae_ref = np.mean(np.abs(preds_ref[m_ref] - targets_ref[m_ref]))
            battery_maes.append((bid, (mae_base - mae_ref) / mae_base * 100))

    battery_maes.sort(key=lambda x: x[1], reverse=True)
    selected_bids = [battery_maes[0][0], battery_maes[len(battery_maes)//2][0]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax_idx, sel_bid in enumerate(selected_bids):
        ax = axes[ax_idx]

        # 真值（用 Exp09c 的 targets，所有模型相同）
        mask = bids_ref == sel_bid
        true_soh = targets_ref[mask]
        cycles = np.arange(len(true_soh))
        ax.plot(cycles, true_soh, 'k-', linewidth=2.0, label='True SOH', zorder=10)

        # 各模型预测
        for exp_id, cfg in CONFIGS.items():
            data = all_data[exp_id]
            m = data['battery_ids'] == sel_bid
            if m.sum() == 0:
                continue
            p = data['predictions'][m]
            mae_val = np.mean(np.abs(p - true_soh[:len(p)])) * 100

            lw = 2.0 if exp_id == 'Exp09c' else 1.0
            alpha = 1.0 if exp_id == 'Exp09c' else 0.7
            zorder = 8 if exp_id == 'Exp09c' else 3

            ax.plot(cycles[:len(p)], p,
                   color=cfg['color'], linestyle=cfg['linestyle'],
                   linewidth=lw, alpha=alpha, zorder=zorder,
                   label=f'{cfg["label"]} ({mae_val:.3f}%)')

        ax.set_xlabel('Cycle Window')
        ax.set_title(f'Battery {sel_bid}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=6.5, loc='lower left', ncol=1, framealpha=0.9)

    axes[0].set_ylabel('SOH')
    plt.suptitle(f'Prediction Curves: 7 Models Comparison (r={RATIO}, seed={SEED})',
                fontsize=13)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'fig7_prediction_curves.png')
    plt.savefig(path)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f'  [OK] Fig 7 saved: {path}')


# ============================================================
# Fig 8: 多模型散点图（pred vs true）
# ============================================================
def plot_fig8(all_data):
    """7 个模型的 pred vs true 散点图"""

    n_models = len(CONFIGS)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.flatten()

    for ax_idx, (exp_id, cfg) in enumerate(CONFIGS.items()):
        ax = axes_flat[ax_idx]
        data = all_data[exp_id]

        preds = data['predictions']
        targets = data['targets']
        mae = np.mean(np.abs(preds - targets)) * 100
        r2 = 1 - np.sum((preds - targets)**2) / np.sum((targets - np.mean(targets))**2)

        ax.scatter(targets, preds, s=2, alpha=0.25, c=cfg['color'], edgecolors='none')

        lims = [min(targets.min(), preds.min()) - 0.02,
                max(targets.max(), preds.max()) + 0.02]
        ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.4)

        ax.text(0.05, 0.92, f'MAE={mae:.4f}%\nR$^2$={r2:.4f}',
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        title = cfg['label']
        if exp_id == 'Exp09c':
            title += ' *'
        ax.set_title(title, fontsize=9, fontweight='bold' if exp_id == 'Exp09c' else 'normal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')

        if ax_idx % 4 == 0:
            ax.set_ylabel('Predicted SOH')
        if ax_idx >= 4:
            ax.set_xlabel('True SOH')

    # 隐藏第 8 个空白子图
    axes_flat[-1].axis('off')

    plt.suptitle(f'Prediction Scatter: 7 Models (r={RATIO}, seed={SEED})', fontsize=13)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'fig8_scatter_plot.png')
    plt.savefig(path)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f'  [OK] Fig 8 saved: {path}')


# ============================================================
# Fig 9: MAE 横向柱状图（一目了然的排名）
# ============================================================
def plot_fig9(all_data):
    """7 个模型在 r=0.3 的 MAE 排名柱状图"""
    fig, ax = plt.subplots(figsize=(8, 5))

    model_maes = []
    for exp_id, cfg in CONFIGS.items():
        data = all_data[exp_id]
        mae = np.mean(np.abs(data['predictions'] - data['targets'])) * 100
        model_maes.append((exp_id, cfg['label'], mae, cfg['color']))

    # 按 MAE 从大到小排序（最好的在最下面）
    model_maes.sort(key=lambda x: x[2], reverse=True)

    names = [m[1] for m in model_maes]
    values = [m[2] for m in model_maes]
    colors = [m[3] for m in model_maes]

    bars = ax.barh(range(len(model_maes)), values, color=colors,
                   edgecolor='white', linewidth=0.8, height=0.6)

    # Exp09c 高亮边框
    for i, m in enumerate(model_maes):
        if m[0] == 'Exp09c':
            bars[i].set_edgecolor('#1B5E20')
            bars[i].set_linewidth(2.5)

    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
               f'{val:.4f}%', va='center', fontsize=9,
               fontweight='bold' if model_maes[i][0] == 'Exp09c' else 'normal')

    ax.set_yticks(range(len(model_maes)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('MAE (%)')
    ax.set_title(f'Model Ranking at r={RATIO} (seed={SEED})', fontsize=12, fontweight='bold')
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig9_model_ranking.png')
    plt.savefig(path)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f'  [OK] Fig 9 saved: {path}')


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true',
                       help='Only plot from cached .npz files (skip training)')
    args = parser.parse_args()

    print('=' * 60)
    print('Paper Prediction Figures (7 Models)')
    print(f'Config: ratio={RATIO}, seed={SEED}')
    print(f'Models: {list(CONFIGS.keys())}')
    print('=' * 60)

    all_data = {}

    if args.plot_only:
        missing = []
        for exp_id in CONFIGS:
            cache_file = os.path.join(CACHE_DIR, f'{exp_id}_r{RATIO}_s{SEED}.npz')
            if os.path.exists(cache_file):
                all_data[exp_id] = np.load(cache_file, allow_pickle=True)
            else:
                missing.append(exp_id)
        if missing:
            print(f'ERROR: Cache missing for: {missing}')
            print('Run without --plot-only first.')
            sys.exit(1)
    else:
        for i, (exp_id, cfg) in enumerate(CONFIGS.items()):
            print(f'\n[{i+1}/{len(CONFIGS)}] {cfg["label"]}')
            all_data[exp_id] = run_and_cache(exp_id, cfg)

    print('\nGenerating figures...')
    plot_fig7(all_data)
    plot_fig8(all_data)
    plot_fig9(all_data)

    print('\n' + '=' * 60)
    print(f'Done. 3 figures saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
