"""
plot_best_seed.py
=================
在 seeds=[929, 2262, 7] 中找出各自最优 seed，然后生成：
  左图：M6 伪标签(+M2) at r=1.0  ——含 MC Dropout ±1σ 置信区间
  右图：A1 多尺度 CNN          at r=0.5  ——逐电池预测曲线

输出：experiments/best_seed_predictions.png
用法：
    python experiments/plot_best_seed.py
    python experiments/plot_best_seed.py --skip-train   # 从已有缓存加载，不重新训练
"""

import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from train_cross_battery import train_cross_battery_model
from models.modules.mc_dropout import mc_predict

# ─── 参数 ────────────────────────────────────────────────────────────────
SEEDS   = [929, 2262, 7]
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = os.path.join(os.path.dirname(__file__), 'best_seed_plots')
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_DIR = os.path.join(OUT_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# ─── 实验配置 ─────────────────────────────────────────────────────────────
EXPS = {
    'M6_r1p0': {
        'label'      : 'M6 伪标签 (+M2)',
        'ratio'      : 1.0,
        'model_type' : 'cnn_lstm_pseudo_label',  # CNN_LSTM_MC + M6
        'override'   : {},                        # 使用 config 文件默认值
        'use_mc'     : True,
        'n_mc'       : 50,
    },
    'A1_r0p5': {
        'label'      : '多尺度 CNN (A1)',
        'ratio'      : 0.5,
        'model_type' : 'ms_cnn_lstm_v2',
        'override'   : {},
        'use_mc'     : False,
        'n_mc'       : 0,
    },
}

# ─── 缓存工具 ────────────────────────────────────────────────────────────
def cache_path(exp_key, seed):
    return os.path.join(CACHE_DIR, f'{exp_key}_seed{seed}.npz')

def save_cache(exp_key, seed, mae, preds, targets, battery_ids, mc_std=None):
    path = cache_path(exp_key, seed)
    arrays = dict(preds=preds, targets=targets, battery_ids=np.array(battery_ids, dtype=object))
    if mc_std is not None:
        arrays['mc_std'] = mc_std
    np.savez(path, mae=np.array([mae]), **arrays)
    return path

def load_cache(exp_key, seed):
    path = cache_path(exp_key, seed)
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return {
        'mae'        : float(d['mae'][0]),
        'preds'      : d['preds'],
        'targets'    : d['targets'],
        'battery_ids': list(d['battery_ids']),
        'mc_std'     : d['mc_std'] if 'mc_std' in d else None,
    }

# ─── 训练单次并收集结果 ────────────────────────────────────────────────────
def run_one(exp_key, seed, cfg, skip_if_cached=True):
    """
    训练/加载一次实验，返回 result dict。
    当 use_mc=True 时额外做 MC Dropout 采样，返回 mc_std。
    """
    cached = load_cache(exp_key, seed)
    if skip_if_cached and cached is not None:
        print(f"  [CACHE] {exp_key} seed={seed}  MAE={cached['mae']*100:.4f}%")
        return cached

    print(f"\n  [TRAIN] {exp_key}  seed={seed}  ratio={cfg['ratio']}  device={DEVICE}")
    t0 = time.time()

    wrapper, results, data_dict = train_cross_battery_model(
        model_type       = cfg['model_type'],
        seed             = seed,
        device           = DEVICE,
        supervision_ratio= cfg['ratio'],
        supervision_seed = None,
        config_override  = cfg['override'],
    )
    elapsed = time.time() - t0
    mae = float(results['test_mae'])
    print(f"  [DONE]  {exp_key} seed={seed}  MAE={mae*100:.4f}%  ({elapsed:.0f}s)")

    preds      = np.array(results['predictions'])
    targets    = np.array(results['targets'])
    battery_ids= list(results['battery_ids']) if results['battery_ids'] is not None else []

    # MC Dropout 置信区间：对测试集逐电池重新采样
    mc_std = None
    if cfg['use_mc'] and wrapper is not None:
        model = wrapper.model if hasattr(wrapper, 'model') else wrapper
        model.eval()

        # 用 data_dict 里的 test_features 和 test_battery_ids 重建 per-sample tensors
        test_feat  = data_dict.get('test_features')    # (N, T, F) or (N, F)
        test_bids  = data_dict.get('test_battery_ids') # (N,)

        if test_feat is not None:
            x_tensor = torch.tensor(test_feat, dtype=torch.float32).to(DEVICE)
            # 批量 mc_predict（避免显存爆）
            batch_sz = 512
            means_list, stds_list = [], []
            for i in range(0, len(x_tensor), batch_sz):
                xb = x_tensor[i:i+batch_sz]
                with torch.no_grad():
                    m, s = mc_predict(model, xb, n_samples=cfg['n_mc'])
                means_list.append(m.cpu().numpy().squeeze(-1))
                stds_list.append(s.cpu().numpy().squeeze(-1))
            mc_std = np.concatenate(stds_list, axis=0)   # (N,)

            # 用 mc_mean 替换原始 predictions（保持一致）
            mc_mean = np.concatenate(means_list, axis=0)
            preds   = mc_mean

            # 更新 battery_ids（来自 data_dict，更可靠）
            if test_bids is not None:
                battery_ids = list(test_bids)

    save_cache(exp_key, seed, mae, preds, targets, battery_ids, mc_std)
    return {
        'mae': mae, 'preds': preds, 'targets': targets,
        'battery_ids': battery_ids, 'mc_std': mc_std,
    }

# ─── 找最优 seed ──────────────────────────────────────────────────────────
def find_best_seed(exp_key, cfg, skip_if_cached):
    results_by_seed = {}
    for seed in SEEDS:
        r = run_one(exp_key, seed, cfg, skip_if_cached=skip_if_cached)
        if r is not None:
            results_by_seed[seed] = r

    best_seed = min(results_by_seed, key=lambda s: results_by_seed[s]['mae'])
    maes = {s: results_by_seed[s]['mae']*100 for s in SEEDS}
    print(f"\n  [{exp_key}] MAE by seed: " + ", ".join(f"seed={s}: {v:.4f}%" for s, v in maes.items()))
    print(f"  [{exp_key}] 最优 seed = {best_seed}  MAE = {maes[best_seed]:.4f}%")
    return best_seed, results_by_seed[best_seed]

# ─── 按电池整理数据 ───────────────────────────────────────────────────────
def split_by_battery(preds, targets, battery_ids, mc_std=None):
    """返回 {battery_id: {'preds', 'targets', 'mae', 'std'(可选)}} 字典"""
    bid_arr = np.array(battery_ids)
    unique  = sorted(set(battery_ids))
    out = {}
    for bid in unique:
        mask = bid_arr == bid
        p = preds[mask]; t = targets[mask]
        mae = float(np.mean(np.abs(p - t)))
        entry = {'preds': p, 'targets': t, 'mae': mae}
        if mc_std is not None:
            entry['std'] = mc_std[mask]
        out[bid] = entry
    return out

def pick_batteries(battery_dict, n=4, strategy='spread'):
    """
    选取 n 个电池用于展示。
    strategy='spread'：按 MAE 分位数取，既有好的也有差的，更有代表性。
    """
    sorted_bids = sorted(battery_dict, key=lambda b: battery_dict[b]['mae'])
    total = len(sorted_bids)
    if total <= n:
        return sorted_bids
    if strategy == 'spread':
        idxs = np.linspace(0, total-1, n, dtype=int)
        return [sorted_bids[i] for i in idxs]
    elif strategy == 'best':
        return sorted_bids[:n]
    else:
        return sorted_bids[:n]

# ─── 绘图 ─────────────────────────────────────────────────────────────────
COLORS = {
    'true'   : '#2c7bb6',   # 蓝
    'pred'   : '#d7191c',   # 红
    'ci'     : '#fdae61',   # 橙（置信区间填充）
    'ci_edge': '#f46d43',
}

def draw_battery_panel(ax, bid, data, use_mc, show_xlabel, show_ylabel, panel_idx):
    p = data['preds']; t = data['targets']
    cycles = np.arange(len(p))
    mae  = np.mean(np.abs(p - t))
    rmse = np.sqrt(np.mean((p - t) ** 2))
    r2   = 1 - np.sum((t - p)**2) / (np.sum((t - np.mean(t))**2) + 1e-12)

    ax.plot(cycles, t, '-', color=COLORS['true'],  lw=2.0, alpha=0.85, label='True SOH')
    ax.plot(cycles, p, '--', color=COLORS['pred'], lw=1.8, alpha=0.85, label='Predicted')

    if use_mc and 'std' in data and data['std'] is not None:
        s = data['std']
        ax.fill_between(cycles, p - s, p + s,
                        color=COLORS['ci'], alpha=0.35, label=r'±1$\sigma$')
        ax.fill_between(cycles, p - 2*s, p + 2*s,
                        color=COLORS['ci'], alpha=0.15, label=r'±2$\sigma$')

    ymin = min(t.min(), p.min()) - 0.03
    ymax = max(t.max(), p.max()) + 0.03
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-1, len(p))

    title = f'Battery {bid}'
    ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
    ax.text(0.98, 0.04,
            f'MAE={mae*100:.3f}%\nRMSE={rmse*100:.3f}%\nR²={r2:.4f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, color='#333333',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.8))
    ax.grid(True, alpha=0.25, lw=0.8)
    ax.tick_params(labelsize=8)
    if show_xlabel:
        ax.set_xlabel('Cycle Index', fontsize=9)
    if show_ylabel:
        ax.set_ylabel('SOH', fontsize=9)


def make_figure(m6_best_seed, m6_data, a1_best_seed, a1_data, output_path, n_show=4):
    """
    两行布局：
      上行：M6+M2  r=1.0，n_show 个子图
      下行：A1 MS-CNN  r=0.5，n_show 个子图
    """
    m6_batteries = split_by_battery(
        m6_data['preds'], m6_data['targets'], m6_data['battery_ids'],
        mc_std=m6_data['mc_std']
    )
    a1_batteries = split_by_battery(
        a1_data['preds'], a1_data['targets'], a1_data['battery_ids'],
    )

    m6_selected = pick_batteries(m6_batteries, n=n_show, strategy='spread')
    a1_selected = pick_batteries(a1_batteries, n=n_show, strategy='spread')

    fig = plt.figure(figsize=(5 * n_show, 9))
    fig.patch.set_facecolor('white')

    # 两行，每行 n_show 列
    gs = gridspec.GridSpec(2, n_show, figure=fig, hspace=0.42, wspace=0.32,
                           top=0.91, bottom=0.08)

    # 上行 title
    m6_overall_mae = m6_data['mae'] * 100
    a1_overall_mae = a1_data['mae'] * 100

    row_titles = [
        f'M6 伪标签 (+M2 MC Dropout)  |  ratio=1.0  |  seed={m6_best_seed}  |  MAE={m6_overall_mae:.4f}%',
        f'多尺度 CNN (A1)  |  ratio=0.5  |  seed={a1_best_seed}  |  MAE={a1_overall_mae:.4f}%',
    ]
    row_colors = ['#1a6b3c', '#7b2d8b']

    for row_i, (selected, batteries, use_mc, row_title, rc) in enumerate([
        (m6_selected, m6_batteries, True,  row_titles[0], row_colors[0]),
        (a1_selected, a1_batteries, False, row_titles[1], row_colors[1]),
    ]):
        for col_i, bid in enumerate(selected):
            ax = fig.add_subplot(gs[row_i, col_i])
            draw_battery_panel(
                ax, bid, batteries[bid], use_mc=use_mc,
                show_xlabel=(row_i == 1),
                show_ylabel=(col_i == 0),
                panel_idx=col_i,
            )

        # 每行左侧标注方法名
        fig.text(0.005, 0.75 - row_i * 0.5, row_title,
                 fontsize=10, fontweight='bold', color=rc,
                 rotation=0, va='center',
                 transform=fig.transFigure)

    # 上行图例（含置信区间）
    legend_elements_m6 = [
        Line2D([0],[0], color=COLORS['true'], lw=2, label='True SOH'),
        Line2D([0],[0], color=COLORS['pred'], lw=2, ls='--', label='Predicted SOH'),
        Patch(facecolor=COLORS['ci'], alpha=0.5, label=r'±1$\sigma$ (MC Dropout)'),
        Patch(facecolor=COLORS['ci'], alpha=0.2, label=r'±2$\sigma$ (MC Dropout)'),
    ]
    legend_elements_a1 = [
        Line2D([0],[0], color=COLORS['true'], lw=2, label='True SOH'),
        Line2D([0],[0], color=COLORS['pred'], lw=2, ls='--', label='Predicted SOH'),
    ]

    axes_row0 = [fig.add_subplot(gs[0, 0])]   # dummy for legend placement
    fig.legend(handles=legend_elements_m6,
               loc='upper right', bbox_to_anchor=(0.99, 0.98),
               fontsize=8.5, framealpha=0.9, ncol=4)

    # 总标题
    fig.suptitle('SOH Prediction Results — Best Seed per Experiment',
                 fontsize=13, fontweight='bold', y=0.985)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n[DONE] 图已保存：{output_path}")
    return output_path


# ─── 入口 ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-train', action='store_true',
                        help='只从缓存加载，不重新训练（缓存不存在时会报错）')
    parser.add_argument('--n-show', type=int, default=4,
                        help='每行展示几个电池（默认4）')
    args = parser.parse_args()

    skip = args.skip_train

    print("=" * 65)
    print("  Step 1/4  训练/加载  M6+M2  r=1.0  (seeds=%s)" % SEEDS)
    print("=" * 65)
    m6_best_seed, m6_best = find_best_seed('M6_r1p0', EXPS['M6_r1p0'], skip_if_cached=not skip)

    print("\n" + "=" * 65)
    print("  Step 2/4  训练/加载  A1 MS-CNN  r=0.5  (seeds=%s)" % SEEDS)
    print("=" * 65)
    a1_best_seed, a1_best = find_best_seed('A1_r0p5', EXPS['A1_r0p5'], skip_if_cached=not skip)

    print("\n" + "=" * 65)
    print("  Step 3/4  汇总最优 Seed")
    print("=" * 65)
    print(f"  M6+M2  r=1.0  → 最优 seed={m6_best_seed}  MAE={m6_best['mae']*100:.4f}%")
    print(f"  A1     r=0.5  → 最优 seed={a1_best_seed}  MAE={a1_best['mae']*100:.4f}%")

    # 保存 seed 记录
    seed_record = {
        'M6_r1p0': {'best_seed': int(m6_best_seed), 'mae_pct': round(m6_best['mae']*100, 4),
                    'all_seeds': SEEDS},
        'A1_r0p5': {'best_seed': int(a1_best_seed), 'mae_pct': round(a1_best['mae']*100, 4),
                    'all_seeds': SEEDS},
    }
    record_path = os.path.join(OUT_DIR, 'best_seeds.json')
    with open(record_path, 'w', encoding='utf-8') as f:
        json.dump(seed_record, f, indent=2, ensure_ascii=False)
    print(f"  [INFO] 最优 seed 记录已保存：{record_path}")

    print("\n" + "=" * 65)
    print("  Step 4/4  生成对比图")
    print("=" * 65)
    output_path = os.path.join(OUT_DIR, 'best_seed_predictions.png')
    make_figure(m6_best_seed, m6_best, a1_best_seed, a1_best, output_path,
                n_show=args.n_show)


if __name__ == '__main__':
    main()
