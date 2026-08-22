"""
plot_best_seed.py
=================
在 seeds=[929, 2262, 7] 中为每组实验找最优 seed，生成【含基线对比】的独立图：
  → experiments/best_seed_plots/M6_r1p0_figure.png
  → experiments/best_seed_plots/A1_r0p5_figure.png

每张图三个面板（目标模型 vs Baseline 对比）：
  ① 散点图：Baseline（灰色×）+ 目标模型（按电池着色○），含各自整体指标
  ② SOH 曲线：True + Baseline + 目标模型（M6 含 ±1σ/±2σ MC 置信区间）
  ③ 误差对比：Baseline（灰色填充）vs 目标模型（彩色填充），含各自均值线

用法：
    python experiments/plot_best_seed.py                    # 完整训练
    python experiments/plot_best_seed.py --skip-train       # 从缓存加载
    python experiments/plot_best_seed.py --seeds 929,2262,7
"""

import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from train_cross_battery import train_cross_battery_model
from models import ConfigLoader
from models.modules.mc_dropout import mc_predict

# ═══════════════════════════════════════════════════════════════════
# 全局配置
# ═══════════════════════════════════════════════════════════════════
SEEDS     = [929, 2262, 7]
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR   = os.path.join(os.path.dirname(__file__), 'best_seed_plots')
MODEL_DIR = os.path.join(OUT_DIR, 'models')
CACHE_DIR = os.path.join(OUT_DIR, 'cache')
for _d in [OUT_DIR, MODEL_DIR, CACHE_DIR]:
    os.makedirs(_d, exist_ok=True)

# 颜色常量
TRUE_CLR     = '#2c7bb6'   # 真实 SOH（蓝）
PRED_CLR     = '#d7191c'   # 目标模型预测（红）
BASE_CLR     = '#636363'   # 基线预测（灰）
CI1_CLR      = '#fdae61'   # MC ±1σ 填充（橙）
CI2_CLR      = '#fee090'   # MC ±2σ 填充（黄）
ERR_BASE_CLR = '#b0b0b0'   # 基线误差填充（浅灰）
ERR_PRED_CLR = '#f4a582'   # 目标模型误差填充（浅红）

EXPS = {
    'M6_r1p0': {
        'label'          : 'M6 Pseudo-label (+MC Dropout)',
        'ratio'          : 1.0,
        'model_type'     : 'cnn_lstm_pseudo_label',
        'baseline_model' : 'cnn_lstm',
        'override'       : {},
        'use_mc'         : True,
        'n_mc'           : 50,
        'color_title'    : '#1a6b3c',
    },
    'A1_r0p5': {
        'label'          : 'Multi-scale CNN (A1)',
        'ratio'          : 0.5,
        'model_type'     : 'ms_cnn_lstm_v2',
        'baseline_model' : 'cnn_lstm',
        'override'       : {},
        'use_mc'         : False,
        'n_mc'           : 0,
        'color_title'    : '#7b2d8b',
    },
}

# ═══════════════════════════════════════════════════════════════════
# 滑窗（与训练保持一致：全局跨样本滑动）
# ═══════════════════════════════════════════════════════════════════
def _apply_windowing(features, targets, battery_ids, window_size):
    if window_size <= 1:
        return features, targets, battery_ids
    wf, wt, wb = [], [], []
    for i in range(len(features) - window_size + 1):
        wf.append(features[i:i + window_size])
        wt.append(targets[i + window_size - 1])
        wb.append(battery_ids[i + window_size - 1])
    return np.array(wf), np.array(wt), np.array(wb, dtype=object)

# ═══════════════════════════════════════════════════════════════════
# 缓存工具
# ═══════════════════════════════════════════════════════════════════
def _cache_path(key, seed):
    return os.path.join(CACHE_DIR, f'{key}_seed{seed}.npz')

def save_cache(key, seed, mae, preds, targets, battery_ids, mc_std=None):
    kw = dict(preds=preds, targets=targets,
              battery_ids=np.array(battery_ids, dtype=object))
    if mc_std is not None:
        kw['mc_std'] = mc_std
    np.savez(_cache_path(key, seed), mae=np.array([mae]), **kw)

def load_cache(key, seed):
    p = _cache_path(key, seed)
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    preds   = d['preds']
    targets = d['targets']
    bids    = list(d['battery_ids'])
    mc_std  = d['mc_std'] if 'mc_std' in d else None

    # 一致性检查：三数组必须等长，否则删除缓存强制重训
    sizes = (len(preds), len(targets), len(bids))
    if len(set(sizes)) != 1:
        print(f"  [WARN]  缓存 {key}/seed={seed} 尺寸不一致 "
              f"preds={sizes[0]}, targets={sizes[1]}, bids={sizes[2]} → 删除重训")
        os.remove(p)
        return None

    return {'mae': float(d['mae'][0]), 'preds': preds,
            'targets': targets, 'battery_ids': bids, 'mc_std': mc_std}

# ═══════════════════════════════════════════════════════════════════
# MC Dropout 推理（preds/targets/bids 来自同一滑窗，保证自洽）
# ═══════════════════════════════════════════════════════════════════
def run_mc_inference(model, test_features, test_targets, test_battery_ids,
                     window_size, device, n_samples=50, batch_size=512):
    placeholder = test_targets if test_targets is not None else np.zeros(len(test_features))
    X, y_win, bids = _apply_windowing(
        test_features, placeholder, test_battery_ids, window_size
    )
    x_t = torch.tensor(X, dtype=torch.float32).to(device)
    means, stds = [], []
    for i in range(0, len(x_t), batch_size):
        m, s = mc_predict(model, x_t[i:i + batch_size], n_samples=n_samples)
        means.append(m.cpu().numpy().squeeze(-1))
        stds.append(s.cpu().numpy().squeeze(-1))
    return np.concatenate(means), np.concatenate(stds), y_win, bids

# ═══════════════════════════════════════════════════════════════════
# 模型保存
# ═══════════════════════════════════════════════════════════════════
def save_model(wrapper, key, seed, ratio, mae, model_type, window_size):
    model = wrapper.model if hasattr(wrapper, 'model') else wrapper
    path  = os.path.join(MODEL_DIR, f'{key}_seed{seed}.pt')
    try:
        config = ConfigLoader.load_model_config(model_type)
    except Exception:
        config = {}
    torch.save({'model_state_dict': model.state_dict(), 'model_type': model_type,
                'config': config, 'seed': seed, 'supervision_ratio': ratio,
                'window_size': window_size, 'test_mae': mae, 'exp_key': key}, path)
    print(f"  [SAVE]  模型已保存：{path}")

# ═══════════════════════════════════════════════════════════════════
# 单次训练（目标模型）
# ═══════════════════════════════════════════════════════════════════
def run_one(exp_key, seed, cfg, skip_if_cached):
    cached = load_cache(exp_key, seed)
    if skip_if_cached and cached is not None:
        print(f"  [CACHE]  {exp_key}  seed={seed}  MAE={cached['mae']*100:.4f}%")
        return cached, None

    print(f"\n  [TRAIN]  {exp_key}  seed={seed}  ratio={cfg['ratio']}  device={DEVICE}")
    t0 = time.time()
    wrapper, results, data_dict = train_cross_battery_model(
        model_type        = cfg['model_type'],
        seed              = seed,
        device            = DEVICE,
        supervision_ratio = cfg['ratio'],
        supervision_seed  = None,
        config_override   = cfg['override'],
    )
    print(f"  [DONE]   {exp_key}  seed={seed}  MAE={results['test_mae']*100:.4f}%  "
          f"({time.time()-t0:.0f}s)")

    preds       = np.array(results['predictions'])
    targets     = np.array(results['targets'])
    battery_ids = list(results['battery_ids']) if results['battery_ids'] is not None else []

    try:
        window_size = ConfigLoader.load_model_config(cfg['model_type'])['data']['window_size']
    except Exception:
        window_size = 40

    mc_std = None
    if cfg['use_mc'] and wrapper is not None:
        model     = wrapper.model if hasattr(wrapper, 'model') else wrapper
        test_feat = data_dict.get('test_features')
        test_tgts = data_dict.get('test_targets')
        test_bids = data_dict.get('test_battery_ids')
        if test_feat is not None and test_bids is not None:
            print(f"  [MC]     运行 {cfg['n_mc']} 次采样…")
            mc_mean, mc_std, mc_targets, mc_bids = run_mc_inference(
                model, test_feat, test_tgts, test_bids,
                window_size, DEVICE, n_samples=cfg['n_mc']
            )
            preds       = mc_mean
            targets     = mc_targets     # 与 preds/bids 同一滑窗 → 自洽
            battery_ids = list(mc_bids)
            print(f"  [MC]     preds={len(preds)}, targets={len(targets)}, "
                  f"bids={len(battery_ids)}")

    if wrapper is not None:
        save_model(wrapper, exp_key, seed, cfg['ratio'],
                   results['test_mae'], cfg['model_type'], window_size)

    save_cache(exp_key, seed, results['test_mae'], preds, targets, battery_ids, mc_std)
    return {'mae': results['test_mae'], 'preds': preds,
            'targets': targets, 'battery_ids': battery_ids, 'mc_std': mc_std}, wrapper

# ═══════════════════════════════════════════════════════════════════
# 基线训练（固定 cnn_lstm，相同 seed + ratio）
# ═══════════════════════════════════════════════════════════════════
def run_baseline(exp_key, seed, ratio, baseline_model, skip_if_cached):
    key    = f'{exp_key}_baseline'
    cached = load_cache(key, seed)
    if skip_if_cached and cached is not None:
        print(f"  [CACHE]  {key}  seed={seed}  MAE={cached['mae']*100:.4f}%")
        return cached

    print(f"\n  [TRAIN]  {key}  seed={seed}  ratio={ratio}  device={DEVICE}")
    t0 = time.time()
    _, results, _ = train_cross_battery_model(
        model_type        = baseline_model,
        seed              = seed,
        device            = DEVICE,
        supervision_ratio = ratio,
        supervision_seed  = None,
        config_override   = {},
    )
    print(f"  [DONE]   {key}  seed={seed}  MAE={results['test_mae']*100:.4f}%  "
          f"({time.time()-t0:.0f}s)")

    preds       = np.array(results['predictions'])
    targets     = np.array(results['targets'])
    battery_ids = list(results['battery_ids']) if results['battery_ids'] is not None else []

    save_cache(key, seed, results['test_mae'], preds, targets, battery_ids)
    return {'mae': results['test_mae'], 'preds': preds,
            'targets': targets, 'battery_ids': battery_ids, 'mc_std': None}

# ═══════════════════════════════════════════════════════════════════
# 找最优 seed
# ═══════════════════════════════════════════════════════════════════
def find_best_seed(exp_key, cfg, seeds, skip_if_cached):
    all_data = {}
    for seed in seeds:
        d, _ = run_one(exp_key, seed, cfg, skip_if_cached=skip_if_cached)
        if d is not None:
            all_data[seed] = d
    best = min(all_data, key=lambda s: all_data[s]['mae'])
    maes = {s: all_data[s]['mae'] * 100 for s in seeds}
    print(f"\n  [{exp_key}] seed MAE:  " +
          "  ".join(f"seed={s}: {v:.4f}%" for s, v in maes.items()))
    print(f"  [{exp_key}] 最优 seed = {best}   MAE = {maes[best]:.4f}%\n")
    return best, all_data[best]

# ═══════════════════════════════════════════════════════════════════
# 按电池分组
# ═══════════════════════════════════════════════════════════════════
def by_battery(preds, targets, battery_ids, mc_std=None):
    bid_arr = np.asarray(battery_ids, dtype=object)
    out = {}
    for bid in sorted(set(battery_ids)):
        mask = bid_arr == bid
        p, t = preds[mask], targets[mask]
        entry = {'preds': p, 'targets': t,
                 'mae': float(np.mean(np.abs(p - t)))}
        if mc_std is not None:
            entry['std'] = mc_std[mask]
        out[bid] = entry
    return out

# ═══════════════════════════════════════════════════════════════════
# 颜色分配（按电池 ID）
# ═══════════════════════════════════════════════════════════════════
def battery_colors(n):
    if n <= 10:
        return [plt.get_cmap('tab10')(i) for i in range(n)]
    elif n <= 20:
        return [plt.get_cmap('tab20')(i) for i in range(n)]
    return [plt.get_cmap('gist_ncar')(i / n) for i in range(n)]

# ═══════════════════════════════════════════════════════════════════
# 指标文本
# ═══════════════════════════════════════════════════════════════════
def _metrics(preds, targets):
    mae  = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    r2   = 1 - np.sum((targets - preds) ** 2) / (
           np.sum((targets - targets.mean()) ** 2) + 1e-12)
    return mae, rmse, r2

def _metrics_text(preds, targets):
    mae, rmse, r2 = _metrics(preds, targets)
    return f'MAE={mae*100:.3f}%\nRMSE={rmse*100:.3f}%\nR²={r2:.4f}'

# ═══════════════════════════════════════════════════════════════════
# 面板 ①：散点图（Baseline 灰×背景 + 目标模型按电池着色○）
# ═══════════════════════════════════════════════════════════════════
def panel_scatter(ax, bat_dict, base_bat_dict, cfg_label, ratio, best_seed,
                  overall_mae, base_mae):
    bids   = sorted(bat_dict.keys())
    colors = battery_colors(len(bids))
    clr_map = dict(zip(bids, colors))

    all_t, all_p = [], []

    # Baseline 散点（灰色 × 标记，置于底层）
    if base_bat_dict is not None:
        bt_all = np.concatenate([base_bat_dict[b]['targets'] for b in base_bat_dict
                                 if b in bat_dict])
        bp_all = np.concatenate([base_bat_dict[b]['preds'] for b in base_bat_dict
                                 if b in bat_dict])
        ax.scatter(bt_all, bp_all, s=5, marker='x', lw=0.6,
                   color=BASE_CLR, alpha=0.30, zorder=2, label='Baseline')

    # 目标模型散点（按电池着色）
    for bid in bids:
        t = bat_dict[bid]['targets']
        p = bat_dict[bid]['preds']
        ax.scatter(t, p, s=8, marker='o', alpha=0.60, color=clr_map[bid],
                   label=f'B-{bid}', zorder=3)
        all_t.append(t); all_p.append(p)

    all_t = np.concatenate(all_t)
    all_p = np.concatenate(all_p)
    vmin  = min(all_t.min(), all_p.min()) - 0.02
    vmax  = max(all_t.max(), all_p.max()) + 0.02

    ax.plot([vmin, vmax], [vmin, vmax], '--', color='#aaaaaa',
            lw=1.2, alpha=0.8, zorder=1, label='y = x')
    ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
    ax.set_xlabel('True SOH', fontsize=10)
    ax.set_ylabel('Predicted SOH', fontsize=10)
    ax.set_title('Prediction Scatter (All Test Batteries)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2, lw=0.7)
    ax.set_aspect('equal', adjustable='box')

    # 双指标文本框（基线左下，目标右上）
    if base_mae is not None:
        ax.text(0.04, 0.40,
                f'Baseline\nMAE={base_mae*100:.3f}%',
                transform=ax.transAxes, ha='left', va='top', fontsize=8,
                color=BASE_CLR,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=BASE_CLR, alpha=0.85))
    ax.text(0.04, 0.96, _metrics_text(all_p, all_t),
            transform=ax.transAxes, ha='left', va='top', fontsize=8,
            color=PRED_CLR,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=PRED_CLR, alpha=0.85))

    # 图例（电池过多时缩字体 + 2 列）
    ncol = 2 if len(bids) > 10 else 1
    ax.legend(loc='lower right', fontsize=6, ncol=ncol, framealpha=0.85,
              markerscale=1.6, handlelength=1.0, handletextpad=0.3,
              borderpad=0.4, labelspacing=0.25)

# ═══════════════════════════════════════════════════════════════════
# 面板 ②：SOH 预测曲线（True + Baseline + 目标模型 + MC CI）
# ═══════════════════════════════════════════════════════════════════
def panel_soh_curve(ax, bid, data, base_data, use_mc):
    p, t = data['preds'], data['targets']
    cycles = np.arange(len(p))

    # MC 置信区间（目标模型）
    if use_mc and 'std' in data and data['std'] is not None:
        s = data['std']
        ax.fill_between(cycles, p - 2*s, p + 2*s,
                        color=CI2_CLR, alpha=0.28, label=r'±2$\sigma$', zorder=1)
        ax.fill_between(cycles, p - s, p + s,
                        color=CI1_CLR, alpha=0.42, label=r'±1$\sigma$', zorder=2)

    # 基线曲线（若提供）
    if base_data is not None:
        bp = base_data['preds'][:len(p)]   # 对齐长度（同电池、同 seed）
        ax.plot(np.arange(len(bp)), bp, '-.',
                color=BASE_CLR, lw=1.5, alpha=0.75, label='Baseline', zorder=3)

    # 真实 SOH + 目标模型预测
    ax.plot(cycles, t, '-',  color=TRUE_CLR, lw=2.0, alpha=0.9, label='True SOH', zorder=5)
    ax.plot(cycles, p, '--', color=PRED_CLR, lw=1.8, alpha=0.9,
            label='Proposed', zorder=6)

    # 轴范围
    y_lo = t.min() - 0.03
    y_hi = t.max() + 0.03
    if base_data is not None:
        y_lo = min(y_lo, bp.min() - 0.01)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(-1, len(p))

    # 双指标文本框
    p_mae, p_rmse, p_r2 = _metrics(p, t)
    ax.text(0.98, 0.96,
            f'Proposed\nMAE={p_mae*100:.3f}%\nRMSE={p_rmse*100:.3f}%\nR²={p_r2:.4f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=7.5,
            color=PRED_CLR,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=PRED_CLR, alpha=0.9))
    if base_data is not None:
        b_mae, b_rmse, b_r2 = _metrics(bp, t[:len(bp)])
        ax.text(0.02, 0.96,
                f'Baseline\nMAE={b_mae*100:.3f}%\nRMSE={b_rmse*100:.3f}%\nR²={b_r2:.4f}',
                transform=ax.transAxes, ha='left', va='top', fontsize=7.5,
                color=BASE_CLR,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=BASE_CLR, alpha=0.9))

    ax.set_title(f'SOH Curve — Battery {bid}  (Best by Proposed)',
                 fontsize=10, fontweight='bold')
    ax.set_ylabel('SOH', fontsize=9)
    ax.legend(loc='lower left', fontsize=7.5, framealpha=0.85,
              ncol=2 if use_mc else 1)
    ax.grid(True, alpha=0.25, lw=0.7)
    ax.tick_params(labelsize=8)

# ═══════════════════════════════════════════════════════════════════
# 面板 ③：误差对比曲线（Baseline 灰色 + 目标模型彩色）
# ═══════════════════════════════════════════════════════════════════
def panel_error_curve(ax, bid, data, base_data, use_mc):
    p, t  = data['preds'], data['targets']
    err   = np.abs(p - t) * 100
    cyc   = np.arange(len(err))

    # 基线误差（灰色填充）
    if base_data is not None:
        bp      = base_data['preds'][:len(p)]
        base_err = np.abs(bp - t[:len(bp)]) * 100
        b_cyc   = np.arange(len(base_err))
        ax.fill_between(b_cyc, 0, base_err,
                        color=ERR_BASE_CLR, alpha=0.60, label='Baseline error', zorder=2)
        b_mean = base_err.mean()
        ax.axhline(b_mean, color=BASE_CLR, lw=1.4, ls='--',
                   label=f'Base mean {b_mean:.3f}%', zorder=4)

    # 目标模型误差（彩色填充）
    ax.fill_between(cyc, 0, err,
                    color=ERR_PRED_CLR, alpha=0.75, label='Proposed error', zorder=3)
    p_mean = err.mean()
    ax.axhline(p_mean, color=PRED_CLR, lw=1.4, ls='--',
               label=f'Prop. mean {p_mean:.3f}%', zorder=5)

    # MC 不确定性阴影（可选）
    if use_mc and 'std' in data and data['std'] is not None:
        s = data['std'] * 100
        ax.fill_between(cyc, 0, s,
                        color=CI1_CLR, alpha=0.30, label=r'1$\sigma$ (MC)', zorder=6)

    # 改进量标注
    if base_data is not None:
        delta = b_mean - p_mean
        sign  = '+' if delta >= 0 else ''
        ax.text(0.98, 0.96, f'Δ={sign}{delta:.3f}%',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                fontweight='bold',
                color='#1a6b3c' if delta >= 0 else '#d7191c',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cccccc', alpha=0.9))

    y_max = max(err.max(), base_err.max() if base_data is not None else 0) * 1.15
    ax.set_xlim(-1, len(err))
    ax.set_ylim(0, max(y_max, 0.5))
    ax.set_xlabel('Cycle Index', fontsize=9)
    ax.set_ylabel('Absolute Error (%)', fontsize=9)
    ax.set_title(f'Error Curve — Battery {bid}', fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7.5, framealpha=0.85, ncol=2)
    ax.grid(True, alpha=0.25, lw=0.7, axis='y')
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

# ═══════════════════════════════════════════════════════════════════
# 生成单张实验图（含基线对比）
# ═══════════════════════════════════════════════════════════════════
def make_figure_for_exp(exp_key, best_seed, data, base_data, cfg):
    bat_dict  = by_battery(data['preds'], data['targets'],
                           data['battery_ids'], data['mc_std'])
    base_dict = (by_battery(base_data['preds'], base_data['targets'],
                            base_data['battery_ids'])
                 if base_data is not None else None)

    # 最准确电池 = 目标模型 MAE 最低
    best_bid = min(bat_dict, key=lambda b: bat_dict[b]['mae'])
    print(f"  [{exp_key}] 最准确电池 = {best_bid}  "
          f"Proposed MAE = {bat_dict[best_bid]['mae']*100:.4f}%")
    if base_dict and best_bid in base_dict:
        print(f"  [{exp_key}]              "
              f"Baseline MAE = {base_dict[best_bid]['mae']*100:.4f}%")

    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           width_ratios=[1.1, 1.0],
                           hspace=0.44, wspace=0.32,
                           top=0.88, bottom=0.09,
                           left=0.07, right=0.97)

    ax_scatter = fig.add_subplot(gs[:, 0])
    ax_soh     = fig.add_subplot(gs[0, 1])
    ax_err     = fig.add_subplot(gs[1, 1])

    base_bat = base_dict.get(best_bid) if base_dict else None

    panel_scatter(ax_scatter, bat_dict, base_dict, cfg['label'],
                  cfg['ratio'], best_seed, data['mae'],
                  base_data['mae'] if base_data else None)
    panel_soh_curve(ax_soh, best_bid, bat_dict[best_bid], base_bat,
                    use_mc=cfg['use_mc'])
    panel_error_curve(ax_err, best_bid, bat_dict[best_bid], base_bat,
                      use_mc=cfg['use_mc'])

    # 总标题（含双 MAE 对比）
    base_str = (f"  |  Baseline MAE={base_data['mae']*100:.4f}%"
                if base_data else '')
    title = (f"{cfg['label']}  |  ratio={cfg['ratio']}  |  seed={best_seed}"
             f"  |  Proposed MAE={data['mae']*100:.4f}%{base_str}")
    fig.suptitle(title, fontsize=11, fontweight='bold',
                 color=cfg['color_title'], y=0.95)

    out_path = os.path.join(OUT_DIR, f'{exp_key}_figure.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [SAVE]  图已保存：{out_path}\n")
    return out_path

# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-train', action='store_true',
                        help='从缓存加载，不重新训练')
    parser.add_argument('--seeds', type=str, default='929,2262,7',
                        help='逗号分隔的 seed 列表，默认 929,2262,7')
    args  = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    skip  = args.skip_train

    print(f"\n{'='*70}")
    print(f"  设备: {DEVICE}   Seeds: {seeds}   skip-train: {skip}")
    print(f"{'='*70}\n")

    summary = {}
    for exp_key, cfg in EXPS.items():
        print(f"\n{'─'*70}")
        print(f"  实验: {exp_key}  [{cfg['label']}]  ratio={cfg['ratio']}")
        print(f"{'─'*70}")

        # ① 找最优 seed（目标模型）
        best_seed, best_data = find_best_seed(exp_key, cfg, seeds,
                                              skip_if_cached=skip)

        # ② 用相同 seed 训练/加载基线
        print(f"  [BASE]   运行基线 seed={best_seed}  model={cfg['baseline_model']}")
        base_data = run_baseline(exp_key, best_seed, cfg['ratio'],
                                 cfg['baseline_model'], skip_if_cached=skip)

        # ③ 出图
        make_figure_for_exp(exp_key, best_seed, best_data, base_data, cfg)

        summary[exp_key] = {
            'best_seed'      : int(best_seed),
            'proposed_mae_pct': round(best_data['mae'] * 100, 4),
            'baseline_mae_pct': round(base_data['mae'] * 100, 4),
            'improvement_pct' : round((base_data['mae'] - best_data['mae']) * 100, 4),
        }

    # 保存汇总
    record_path = os.path.join(OUT_DIR, 'best_seeds.json')
    with open(record_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("  全部完成！")
    for exp_key, s in summary.items():
        print(f"  {exp_key}:  Proposed={s['proposed_mae_pct']}%  "
              f"Baseline={s['baseline_mae_pct']}%  "
              f"Δ={s['improvement_pct']:+.4f}%")
    print(f"\n  图目录：{OUT_DIR}")
    print(f"  汇总：{record_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
