"""
plot_best_seed.py
=================
在 seeds=[929, 2262, 7] 中为每组实验找最优 seed，分别生成独立图：
  → experiments/best_seed_plots/M6_r1p0_figure.png
  → experiments/best_seed_plots/A1_r0p5_figure.png

每张图包含三个面板：
  ① 预测误差散点图（所有测试电池，按电池 ID 不同颜色区分）
  ② 最准确电池的 SOH 预测曲线（M6 含 ±1σ/±2σ MC Dropout 置信区间）
  ③ 最准确电池的逐循环绝对误差曲线

训练好的模型自动保存：
  → experiments/best_seed_plots/models/{exp_key}_seed{best_seed}.pt
    内含 state_dict / model_type / config / seed / ratio / window_size / MAE

用法：
    python experiments/plot_best_seed.py                    # 完整训练
    python experiments/plot_best_seed.py --skip-train       # 跳过训练，从缓存加载
    python experiments/plot_best_seed.py --seeds 929,2262,7 # 指定 seed 列表
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
from matplotlib.ticker import MaxNLocator

from train_cross_battery import train_cross_battery_model
from models import ConfigLoader
from models.modules.mc_dropout import mc_predict

# ═══════════════════════════════════════════════════════════════════
# 全局配置
# ═══════════════════════════════════════════════════════════════════
SEEDS   = [929, 2262, 7]
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = os.path.join(os.path.dirname(__file__), 'best_seed_plots')
MODEL_DIR = os.path.join(OUT_DIR, 'models')
CACHE_DIR = os.path.join(OUT_DIR, 'cache')
for d in [OUT_DIR, MODEL_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

EXPS = {
    'M6_r1p0': {
        'label'      : 'M6 伪标签 (+M2 MC Dropout)',
        'ratio'      : 1.0,
        'model_type' : 'cnn_lstm_pseudo_label',
        'override'   : {},
        'use_mc'     : True,
        'n_mc'       : 50,
        'color_title': '#1a6b3c',
    },
    'A1_r0p5': {
        'label'      : '多尺度 CNN (A1)',
        'ratio'      : 0.5,
        'model_type' : 'ms_cnn_lstm_v2',
        'override'   : {},
        'use_mc'     : False,
        'n_mc'       : 0,
        'color_title': '#7b2d8b',
    },
}

# ═══════════════════════════════════════════════════════════════════
# 滑窗函数（与 train_cross_battery.py 保持一致：全局跨样本滑动）
# ═══════════════════════════════════════════════════════════════════
def _apply_windowing(features, targets, battery_ids, window_size):
    """全局滑窗，不切断电池边界（与训练保持一致）。"""
    if window_size <= 1:
        return features, targets, battery_ids
    wf, wt, wb = [], [], []
    for i in range(len(features) - window_size + 1):
        wf.append(features[i:i + window_size])
        wt.append(targets[i + window_size - 1])
        wb.append(battery_ids[i + window_size - 1])
    return np.array(wf), np.array(wt), np.array(wb, dtype=object)

# ═══════════════════════════════════════════════════════════════════
# 模型保存工具
# ═══════════════════════════════════════════════════════════════════
def save_model(wrapper, exp_key, seed, ratio, mae, model_type, window_size):
    """保存 state_dict 与训练元信息，方便后续重载推理。"""
    model = wrapper.model if hasattr(wrapper, 'model') else wrapper
    save_path = os.path.join(MODEL_DIR, f'{exp_key}_seed{seed}.pt')
    # 读取完整 config（方便重建模型）
    try:
        config = ConfigLoader.load_model_config(model_type)
    except Exception:
        config = {}
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type'      : model_type,
        'config'          : config,
        'seed'            : seed,
        'supervision_ratio': ratio,
        'window_size'     : window_size,
        'test_mae'        : mae,
        'exp_key'         : exp_key,
    }, save_path)
    print(f"  [SAVE] 模型已保存：{save_path}")
    return save_path

# ═══════════════════════════════════════════════════════════════════
# 缓存工具
# ═══════════════════════════════════════════════════════════════════
def _cache_path(exp_key, seed):
    return os.path.join(CACHE_DIR, f'{exp_key}_seed{seed}.npz')

def save_cache(exp_key, seed, mae, preds, targets, battery_ids, mc_std=None):
    kw = dict(preds=preds, targets=targets,
              battery_ids=np.array(battery_ids, dtype=object))
    if mc_std is not None:
        kw['mc_std'] = mc_std
    np.savez(_cache_path(exp_key, seed), mae=np.array([mae]), **kw)

def load_cache(exp_key, seed):
    p = _cache_path(exp_key, seed)
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    preds   = d['preds']
    targets = d['targets']
    bids    = list(d['battery_ids'])
    mc_std  = d['mc_std'] if 'mc_std' in d else None

    # ── 一致性检查：三个主数组必须等长 ──────────────────────────────
    sizes = (len(preds), len(targets), len(bids))
    if len(set(sizes)) != 1:
        print(f"  [WARN]  缓存 {exp_key}/seed={seed} 尺寸不一致 "
              f"preds={sizes[0]}, targets={sizes[1]}, bids={sizes[2]} → 删除并重训")
        os.remove(p)
        return None                 # 触发重新训练

    return {
        'mae'        : float(d['mae'][0]),
        'preds'      : preds,
        'targets'    : targets,
        'battery_ids': bids,
        'mc_std'     : mc_std,
    }

# ═══════════════════════════════════════════════════════════════════
# MC Dropout 批量推理
# ═══════════════════════════════════════════════════════════════════
def run_mc_inference(model, test_features, test_targets, test_battery_ids,
                     window_size, device, n_samples=50, batch_size=512):
    """
    对 test_features（已缩放，2D: N×F）先滑窗，再 MC 采样。

    注意：preds / targets / bids 全部来自同一次 _apply_windowing，
    保证三者长度一致，不依赖外部 results['targets']。

    返回 (mc_mean, mc_std, windowed_targets, windowed_bids)，均为 numpy。
    """
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
# 单次训练 / 缓存加载
# ═══════════════════════════════════════════════════════════════════
def run_one(exp_key, seed, cfg, skip_if_cached):
    cached = load_cache(exp_key, seed)
    if skip_if_cached and cached is not None:
        print(f"  [CACHE]  {exp_key}  seed={seed}  MAE={cached['mae']*100:.4f}%")
        return cached, None     # (data, wrapper) → wrapper=None 表示已缓存

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

    # window_size：从 config 读取
    try:
        window_size = ConfigLoader.load_model_config(cfg['model_type'])['data']['window_size']
    except Exception:
        window_size = 40

    # MC Dropout 置信区间（仅 M2 模型）
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
            # preds / targets / battery_ids 全部来自同一次滑窗 → 长度自洽
            preds       = mc_mean
            targets     = mc_targets   # ← 关键：随 MC 窗口同步更新
            battery_ids = list(mc_bids)
            print(f"  [MC]     preds={len(preds)}, targets={len(targets)}, "
                  f"bids={len(battery_ids)}")

    # 保存模型
    if wrapper is not None:
        save_model(wrapper, exp_key, seed, cfg['ratio'],
                   results['test_mae'], cfg['model_type'], window_size)

    save_cache(exp_key, seed, results['test_mae'], preds, targets, battery_ids, mc_std)
    data = {
        'mae': results['test_mae'], 'preds': preds,
        'targets': targets, 'battery_ids': battery_ids, 'mc_std': mc_std,
    }
    return data, wrapper

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
    maes = {s: all_data[s]['mae']*100 for s in seeds}
    print(f"\n  [{exp_key}] seed MAE:  " +
          "  ".join(f"seed={s}: {v:.4f}%" for s, v in maes.items()))
    print(f"  [{exp_key}] 最优 seed = {best}   MAE = {maes[best]:.4f}%\n")
    return best, all_data[best]

# ═══════════════════════════════════════════════════════════════════
# 数据整理：按电池分组
# ═══════════════════════════════════════════════════════════════════
def by_battery(preds, targets, battery_ids, mc_std=None):
    """返回 {bid: {'preds','targets','mae','std'(可选)}} dict"""
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
# 颜色分配（支持大量电池）
# ═══════════════════════════════════════════════════════════════════
def battery_colors(n):
    if n <= 10:
        cmap = plt.get_cmap('tab10')
        return [cmap(i) for i in range(n)]
    elif n <= 20:
        cmap = plt.get_cmap('tab20')
        return [cmap(i) for i in range(n)]
    else:
        cmap = plt.get_cmap('gist_ncar')
        return [cmap(i / n) for i in range(n)]

# ═══════════════════════════════════════════════════════════════════
# 绘图核心函数
# ═══════════════════════════════════════════════════════════════════
TRUE_CLR  = '#2c7bb6'
PRED_CLR  = '#d7191c'
CI1_CLR   = '#fdae61'
CI2_CLR   = '#fee090'
ERR_CLR   = '#636363'

def _metrics_text(preds, targets):
    mae  = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets)**2))
    r2   = 1 - np.sum((targets - preds)**2) / (np.sum((targets - targets.mean())**2) + 1e-12)
    return f'MAE={mae*100:.3f}%\nRMSE={rmse*100:.3f}%\nR²={r2:.4f}', mae, rmse, r2


def panel_scatter(ax, bat_dict, exp_label, ratio, best_seed, overall_mae):
    """面板①：预测误差散点图（pred vs true，按电池着色）"""
    bids   = sorted(bat_dict.keys())
    colors = battery_colors(len(bids))
    clr_map = dict(zip(bids, colors))

    all_t, all_p = [], []
    for bid in bids:
        t = bat_dict[bid]['targets']
        p = bat_dict[bid]['preds']
        ax.scatter(t, p, s=8, alpha=0.55, color=clr_map[bid],
                   label=f'Battery {bid}', zorder=3)
        all_t.append(t); all_p.append(p)

    all_t = np.concatenate(all_t)
    all_p = np.concatenate(all_p)
    vmin  = min(all_t.min(), all_p.min()) - 0.02
    vmax  = max(all_t.max(), all_p.max()) + 0.02

    # 对角线 y = x
    ax.plot([vmin, vmax], [vmin, vmax], '--', color='#888888',
            lw=1.2, alpha=0.8, zorder=2, label='y = x')

    ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
    ax.set_xlabel('True SOH', fontsize=10)
    ax.set_ylabel('Predicted SOH', fontsize=10)
    ax.set_title('Prediction Error Scatter\n(All Test Batteries)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2, lw=0.7)
    ax.set_aspect('equal', adjustable='box')

    # 整体指标文本框
    txt, *_ = _metrics_text(all_p, all_t)
    ax.text(0.04, 0.96, txt, transform=ax.transAxes,
            ha='left', va='top', fontsize=8.5, color='#222222',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#aaaaaa', alpha=0.9))

    # 图例（电池过多时用小字体 + 2 列）
    ncol = 2 if len(bids) > 10 else 1
    ax.legend(loc='lower right', fontsize=6.5, ncol=ncol,
              framealpha=0.85, markerscale=1.8,
              handlelength=1.0, handletextpad=0.4,
              borderpad=0.5, labelspacing=0.3)


def panel_soh_curve(ax, bid, data, use_mc, show_xlabel=False):
    """面板②：最准确电池 SOH 预测曲线（含 MC 置信区间）"""
    p, t = data['preds'], data['targets']
    cycles = np.arange(len(p))

    if use_mc and 'std' in data and data['std'] is not None:
        s = data['std']
        ax.fill_between(cycles, p - 2*s, p + 2*s,
                        color=CI2_CLR, alpha=0.30, label=r'±2$\sigma$', zorder=1)
        ax.fill_between(cycles, p - s, p + s,
                        color=CI1_CLR, alpha=0.45, label=r'±1$\sigma$', zorder=2)

    ax.plot(cycles, t, '-',  color=TRUE_CLR, lw=2.0, alpha=0.9, label='True SOH',    zorder=4)
    ax.plot(cycles, p, '--', color=PRED_CLR, lw=1.8, alpha=0.9, label='Predicted',   zorder=5)

    ymin = min(t.min(), p.min()) - 0.03
    ymax = max(t.max(), p.max()) + 0.03
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-1, len(p))

    txt, *_ = _metrics_text(p, t)
    ax.text(0.98, 0.96, txt, transform=ax.transAxes,
            ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#aaaaaa', alpha=0.9))

    ax.set_title(f'SOH Curve — Battery {bid}  (Best Predicted)',
                 fontsize=10, fontweight='bold')
    ax.set_ylabel('SOH', fontsize=9)
    if show_xlabel:
        ax.set_xlabel('Cycle Index', fontsize=9)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.25, lw=0.7)
    ax.tick_params(labelsize=8)


def panel_error_curve(ax, bid, data, use_mc):
    """面板③：逐循环绝对误差曲线"""
    p, t = data['preds'], data['targets']
    err  = np.abs(p - t)
    cycles = np.arange(len(err))

    ax.bar(cycles, err * 100, width=1.0, color=ERR_CLR, alpha=0.55, label='|Error|')
    ax.plot(cycles, err * 100, '-', color='#333333', lw=0.8, alpha=0.5)

    # 均值线
    mean_err = err.mean() * 100
    ax.axhline(mean_err, color='#d7191c', lw=1.5, ls='--',
               label=f'Mean {mean_err:.3f}%')

    if use_mc and 'std' in data and data['std'] is not None:
        s = data['std'] * 100
        ax.fill_between(cycles, 0, s, color=CI1_CLR, alpha=0.35, label=r'1$\sigma$ (MC)')

    ax.set_xlim(-1, len(err))
    ax.set_ylim(0, max(err.max()*100 * 1.15, 0.5))
    ax.set_xlabel('Cycle Index', fontsize=9)
    ax.set_ylabel('Absolute Error (%)', fontsize=9)
    ax.set_title(f'Error Curve — Battery {bid}', fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.25, lw=0.7, axis='y')
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


# ═══════════════════════════════════════════════════════════════════
# 生成单张实验图
# ═══════════════════════════════════════════════════════════════════
def make_figure_for_exp(exp_key, best_seed, data, cfg):
    """
    三面板布局：
      左列（宽）：散点图
      右列（窄）：上=SOH曲线，下=误差曲线
    """
    bat_dict = by_battery(
        data['preds'], data['targets'], data['battery_ids'], data['mc_std']
    )

    # 最准确电池 = MAE 最小
    best_bid = min(bat_dict, key=lambda b: bat_dict[b]['mae'])
    print(f"  [{exp_key}] 最准确电池 = {best_bid}  "
          f"MAE = {bat_dict[best_bid]['mae']*100:.4f}%")

    fig = plt.figure(figsize=(15, 8))
    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           width_ratios=[1.1, 1.0],
                           hspace=0.42, wspace=0.32,
                           top=0.88, bottom=0.10,
                           left=0.07, right=0.97)

    ax_scatter = fig.add_subplot(gs[:, 0])   # 左列，占两行
    ax_soh     = fig.add_subplot(gs[0, 1])   # 右上
    ax_err     = fig.add_subplot(gs[1, 1])   # 右下

    # 绘制三个面板
    panel_scatter(ax_scatter, bat_dict, cfg['label'],
                  cfg['ratio'], best_seed, data['mae'])
    panel_soh_curve(ax_soh, best_bid, bat_dict[best_bid],
                    use_mc=cfg['use_mc'], show_xlabel=False)
    panel_error_curve(ax_err, best_bid, bat_dict[best_bid],
                      use_mc=cfg['use_mc'])

    # 总标题
    title = (f"{cfg['label']}  |  ratio={cfg['ratio']}  |  "
             f"best seed={best_seed}  |  overall MAE={data['mae']*100:.4f}%")
    fig.suptitle(title, fontsize=12, fontweight='bold',
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
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    skip  = args.skip_train

    print(f"\n{'='*65}")
    print(f"  设备: {DEVICE}   Seeds: {seeds}   skip-train: {skip}")
    print(f"{'='*65}\n")

    best_seeds = {}
    for exp_key, cfg in EXPS.items():
        print(f"\n{'─'*65}")
        print(f"  实验: {exp_key}  [{cfg['label']}]  ratio={cfg['ratio']}")
        print(f"{'─'*65}")
        best_seed, best_data = find_best_seed(exp_key, cfg, seeds,
                                              skip_if_cached=skip)
        best_seeds[exp_key] = {'seed': int(best_seed),
                               'mae_pct': round(best_data['mae']*100, 4)}
        make_figure_for_exp(exp_key, best_seed, best_data, cfg)

    # 保存最优 seed 汇总
    record_path = os.path.join(OUT_DIR, 'best_seeds.json')
    with open(record_path, 'w', encoding='utf-8') as f:
        json.dump(best_seeds, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*65}")
    print("  全部完成！输出文件：")
    for exp_key in EXPS:
        print(f"    图：{OUT_DIR}/{exp_key}_figure.png")
        seed = best_seeds[exp_key]['seed']
        print(f"  模型：{MODEL_DIR}/{exp_key}_seed{seed}.pt")
    print(f"  汇总：{record_path}")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
