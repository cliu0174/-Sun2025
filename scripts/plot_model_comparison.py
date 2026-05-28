"""
图4-6 & 图4-7 生成脚本
================================
功能：
  1. 训练 4 个对比模型（XGBoost / LSTM / CNN-LSTM / PI-MS-CNN-LSTM）
  2. 缓存预测结果（支持断点续跑 --plot-only）
  3. 生成图4-6：测试集散点图（pred vs true SOH，2×2 布局）
  4. 生成图4-7：RMSE / MAE / R² 柱状对比图

运行方式：
  python scripts/plot_model_comparison.py            # 完整训练 + 出图
  python scripts/plot_model_comparison.py --plot-only  # 仅从缓存出图

配置：r=1.0（全监督）, seed=929
输出：figures/fig4_6_scatter.{png,pdf}
      figures/fig4_7_bar.{png,pdf}
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import OrderedDict

# ── 项目根目录 ───────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── 输出目录 ─────────────────────────────────────────────────────
CACHE_DIR  = os.path.join(ROOT, 'figures', 'pred_cache')
OUTPUT_DIR = os.path.join(ROOT, 'figures')
os.makedirs(CACHE_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 全局配置 ──────────────────────────────────────────────────────
SEED   = 929
RATIO  = 1.0          # 全监督
DEVICE = 'cuda'

# ── 图形样式 ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'    : 'serif',
    'font.size'      : 10,
    'figure.dpi'     : 150,
    'savefig.dpi'    : 300,
    'savefig.bbox'   : 'tight',
    'axes.grid'      : True,
    'grid.alpha'     : 0.25,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── 模型定义（有序，表格顺序）────────────────────────────────────
_PI_CONFIG = {
    'enabled'           : True,
    'monotonic_weight'  : 0.3,
    'boundary_weight'   : 0.0,
    'smoothness_weight' : 0.0,
    'monotonic_tolerance': 0.005,
    'min_cycle'         : 300,
    'base_loss_weight'  : 1.0,
}
_NO_PI = {
    'enabled'           : False,
    'monotonic_weight'  : 0.0,
    'boundary_weight'   : 0.0,
    'smoothness_weight' : 0.0,
}

MODELS = OrderedDict([
    ('xgboost', {
        'model_type' : 'xgboost_simple',
        'override'   : None,
        'label'      : 'XGBoost',
        'color'      : '#63C5B5',   # teal green
        'marker'     : 'o',
        'is_xgb'     : True,
    }),
    ('lstm', {
        'model_type' : 'lstm',
        'override'   : {'physics_constraints': _NO_PI},
        'label'      : 'LSTM',
        'color'      : '#8BA7C7',   # steel blue
        'marker'     : 's',
        'is_xgb'     : False,
    }),
    ('cnn_lstm', {
        'model_type' : 'cnn_lstm',
        'override'   : {'physics_constraints': _NO_PI},
        'label'      : 'CNN-LSTM',
        'color'      : '#F5C542',   # golden yellow
        'marker'     : '^',
        'is_xgb'     : False,
    }),
    ('pi_ms_cnn_lstm', {
        'model_type' : 'ms_cnn_lstm_v2',
        'override'   : {
            'architecture'       : {'use_multiscale': True, 'per_window_norm': False},
            'physics_constraints': _PI_CONFIG,
        },
        'label'      : 'PI-MS-CNN-LSTM',
        'color'      : '#F4831F',   # orange (替换原灰色)
        'marker'     : 'D',
        'is_xgb'     : False,
    }),
])


# ════════════════════════════════════════════════════════════════════
# 训练 & 缓存
# ════════════════════════════════════════════════════════════════════

def cache_path(exp_id: str) -> str:
    return os.path.join(CACHE_DIR, f'fig46_{exp_id}_r{RATIO}_s{SEED}.npz')


def train_and_cache(exp_id: str, cfg: dict) -> dict:
    """训练模型并将预测结果缓存到 .npz，已有缓存则直接读取。"""
    fpath = cache_path(exp_id)
    if os.path.exists(fpath):
        print(f'  [CACHE HIT] {exp_id}')
        return dict(np.load(fpath, allow_pickle=True))

    print(f'\n  [TRAIN] {exp_id}: {cfg["label"]}  (seed={SEED}, r={RATIO})')

    if cfg['is_xgb']:
        from train_xgboost_baseline import train_xgboost_baseline
        _, result, _ = train_xgboost_baseline(
            model_type  = cfg['model_type'],
            train_ratio = 0.6,
            val_ratio   = 0.2,
            test_ratio  = 0.2,
            device      = 'cpu',
            seed        = SEED,
        )
    else:
        from train_cross_battery import train_cross_battery_model
        _, result, _ = train_cross_battery_model(
            model_type       = cfg['model_type'],
            device           = DEVICE,
            seed             = SEED,
            supervision_ratio= RATIO,
            supervision_seed = None,
            config_override  = cfg['override'],
        )

    preds   = np.array(result['predictions'],  dtype=np.float32)
    targets = np.array(result['targets'],       dtype=np.float32)
    bids_raw = result.get('battery_ids')
    bids    = np.array(bids_raw, dtype=object) if bids_raw is not None else None

    np.savez(fpath,
             predictions = preds,
             targets     = targets,
             battery_ids = bids,
             rmse        = result['test_rmse'],
             mae         = result['test_mae'],
             r2          = result['test_r2'])

    print(f'  [DONE]  MAE={result["test_mae"]*100:.4f}%  '
          f'RMSE={result["test_rmse"]*100:.4f}%  R²={result["test_r2"]:.4f}')
    return dict(np.load(fpath, allow_pickle=True))


# ════════════════════════════════════════════════════════════════════
# 图4-6：散点图
# ════════════════════════════════════════════════════════════════════

def plot_fig46(all_data: dict):
    """
    2×2 pred-vs-true 散点图，样式对齐截图：
    - 标题行：模型名 + RMSE=x%, R²=y（加粗）
    - 右上角 Ideal 虚线图例
    - 各子图颜色与柱状图一致
    """
    ids   = list(MODELS.keys())
    n     = len(ids)
    ncols = 2
    nrows = (n + ncols - 1) // ncols   # 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 8.5))
    axes_flat = axes.flatten()

    for idx, exp_id in enumerate(ids):
        ax  = axes_flat[idx]
        cfg = MODELS[exp_id]
        d   = all_data[exp_id]
        p   = d['predictions'].astype(float)
        t   = d['targets'].astype(float)

        rmse = float(d['rmse']) * 100
        r2   = float(d['r2'])

        # 散点（小点 + 低透明度，密集时显层次）
        ax.scatter(t, p, s=2, alpha=0.25, c=cfg['color'],
                   edgecolors='none', rasterized=True)

        # 对角理想线
        lo = min(t.min(), p.min()) - 0.005
        hi = max(t.max(), p.max()) + 0.005
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0, label='Ideal')

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')

        # 标题：模型名 + 指标（对齐截图格式）
        ax.set_title(f'{cfg["label"]}\n'
                     f'RMSE={rmse:.3f}%, R²={r2:.4f}',
                     fontsize=10, fontweight='bold', pad=6)

        ax.set_xlabel('True SOH',      fontsize=9)
        ax.set_ylabel('Predicted SOH', fontsize=9)
        ax.legend(fontsize=8, loc='upper left', framealpha=0.7,
                  handlelength=1.5)

        # 刻度字号
        ax.tick_params(axis='both', labelsize=8)

    # 隐藏多余子图（模型数为奇数时）
    for k in range(n, nrows * ncols):
        axes_flat[k].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 1])

    for ext in ('png', 'pdf'):
        fpath = os.path.join(OUTPUT_DIR, f'fig4_6_scatter.{ext}')
        plt.savefig(fpath)
        print(f'  [OK] 图4-6 → {fpath}')
    plt.close()


# ════════════════════════════════════════════════════════════════════
# 图4-7：柱状对比图
# ════════════════════════════════════════════════════════════════════

def plot_fig47(all_data: dict):
    """
    1×3 柱状图：RMSE / MAE / R²，样式对齐截图：
    - 数值标注在柱顶（百分号格式）
    - 各柱颜色与散点图一致
    - y 轴从 0 开始（RMSE/MAE），R² 从适当下限开始
    """
    ids    = list(MODELS.keys())
    labels = [MODELS[e]['label'] for e in ids]
    colors = [MODELS[e]['color'] for e in ids]

    rmse_vals = [float(all_data[e]['rmse']) * 100 for e in ids]
    mae_vals  = [float(all_data[e]['mae'])  * 100 for e in ids]
    r2_vals   = [float(all_data[e]['r2'])          for e in ids]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))

    metrics = [
        ('RMSE Comparison', 'RMSE (%)', rmse_vals, True,  '{:.3f}%'),
        ('MAE Comparison',  'MAE (%)',  mae_vals,  True,  '{:.3f}%'),
        ('R² Comparison',   'R²',       r2_vals,   False, '{:.4f}'),
    ]

    x     = np.arange(len(labels))
    bar_w = 0.55

    for ax, (title, ylabel, vals, lower_better, fmt) in zip(axes, metrics):
        bars = ax.bar(x, vals,
                      width     = bar_w,
                      color     = colors,
                      edgecolor = 'white',
                      linewidth = 0.6)

        # 数值标注在柱顶
        ymax = max(vals)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.015,
                    fmt.format(val),
                    ha='center', va='bottom', fontsize=9)

        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.tick_params(axis='y', labelsize=8)

        # y 轴范围
        if ylabel == 'R²':
            ymin_val = min(vals)
            ax.set_ylim(max(0, ymin_val - 0.02), 1.0 + (1.0 - ymin_val) * 0.25)
        else:
            ax.set_ylim(0, ymax * 1.18)

        # 只保留水平网格
        ax.yaxis.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

    plt.tight_layout()

    for ext in ('png', 'pdf'):
        fpath = os.path.join(OUTPUT_DIR, f'fig4_7_bar.{ext}')
        plt.savefig(fpath)
        print(f'  [OK] 图4-7 → {fpath}')
    plt.close()


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Generate Fig4-6 & Fig4-7')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip training, load from cache only')
    parser.add_argument('--device', default=DEVICE,
                        help='Compute device (default: cuda)')
    args = parser.parse_args()

    # 允许命令行覆盖设备
    global DEVICE
    DEVICE = args.device

    print('=' * 60)
    print('  Fig4-6 & Fig4-7 — Model Comparison')
    print(f'  Models : {list(MODELS.keys())}')
    print(f'  Config : r={RATIO}, seed={SEED}, device={DEVICE}')
    print('=' * 60)

    all_data = {}

    if args.plot_only:
        # 只从缓存读
        missing = []
        for exp_id in MODELS:
            fp = cache_path(exp_id)
            if os.path.exists(fp):
                all_data[exp_id] = dict(np.load(fp, allow_pickle=True))
                print(f'  [LOAD] {exp_id}  '
                      f'MAE={float(all_data[exp_id]["mae"])*100:.4f}%')
            else:
                missing.append(exp_id)
        if missing:
            print(f'\nERROR: 缓存缺失 → {missing}')
            print('请先不加 --plot-only 运行一次完整训练。')
            sys.exit(1)
    else:
        for exp_id, cfg in MODELS.items():
            print(f'\n[{list(MODELS.keys()).index(exp_id)+1}/{len(MODELS)}] '
                  f'{cfg["label"]}')
            all_data[exp_id] = train_and_cache(exp_id, cfg)

    print('\n生成图表...')
    plot_fig46(all_data)
    plot_fig47(all_data)

    print('\n' + '=' * 60)
    print(f'  完成。输出目录：{OUTPUT_DIR}')
    print('  图4-6 → fig4_6_scatter.png / .pdf')
    print('  图4-7 → fig4_7_bar.png / .pdf')


if __name__ == '__main__':
    main()
