"""
图4-8：单电池 SOH 预测轨迹对比（CNN-LSTM vs PI-MSCL）
=======================================================
从 plot_model_comparison.py 生成的缓存（.npz）读取，无需重新训练。
输出两张独立图：
  fig4_8_soh_{bid}.{png,pdf}    — SOH 轨迹对比
  fig4_8_err_{bid}.{png,pdf}    — 预测误差轨迹

用法：
    # 默认输出电池 3-3
    python scripts/plot_battery_trajectory.py

    # 指定任意电池 ID
    python scripts/plot_battery_trajectory.py --battery 3-3

    # 列出所有可用电池
    python scripts/plot_battery_trajectory.py --list
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR  = os.path.join(ROOT, 'figures', 'pred_cache')
OUTPUT_DIR = os.path.join(ROOT, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 与 plot_model_comparison.py 保持一致 ─────────────────────────
SEED  = 42
RATIO = 1.0

MODELS = {
    'cnn_lstm': {
        'label' : 'CNN-LSTM',
        'color' : '#F5C542',
        'lw'    : 1.4,
        'ls'    : '--',
        'zorder': 3,
        'alpha' : 0.9,
    },
    'pi_ms_cnn_lstm': {
        'label' : 'PI-MSCL',
        'color' : '#F4831F',
        'lw'    : 2.0,
        'ls'    : '-',
        'zorder': 5,
        'alpha' : 1.0,
    },
}

# ── 图形全局样式 ──────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'serif',
    'font.size'        : 10,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})


# ════════════════════════════════════════════════════════════════
# 加载缓存
# ════════════════════════════════════════════════════════════════

def load_cache():
    data = {}
    missing = []
    for exp_id in MODELS:
        fpath = os.path.join(CACHE_DIR, f'fig46_{exp_id}_r{RATIO}_s{SEED}.npz')
        if os.path.exists(fpath):
            data[exp_id] = dict(np.load(fpath, allow_pickle=True))
            if data[exp_id]['battery_ids'] is not None:
                data[exp_id]['battery_ids'] = np.array(
                    [str(b) for b in data[exp_id]['battery_ids']]
                )
        else:
            missing.append(fpath)
    if missing:
        print('ERROR: 以下缓存文件不存在，请先运行 plot_model_comparison.py：')
        for p in missing:
            print(f'  {p}')
        sys.exit(1)
    return data


# ════════════════════════════════════════════════════════════════
# 电池辅助
# ════════════════════════════════════════════════════════════════

def available_batteries(data):
    ref_ids = set(data['cnn_lstm']['battery_ids'])
    for exp_id in MODELS:
        ref_ids &= set(data[exp_id]['battery_ids'])
    return sorted(ref_ids)


def _metrics(pred, true):
    """返回 (rmse%, mae%) 两个浮点数。"""
    err  = pred - true
    rmse = np.sqrt(np.mean(err ** 2)) * 100
    mae  = np.mean(np.abs(err))       * 100
    return rmse, mae


# ════════════════════════════════════════════════════════════════
# 图1：SOH 轨迹对比
# ════════════════════════════════════════════════════════════════

def plot_soh(data, bid, tag='Unseen'):
    ref_id   = 'cnn_lstm'
    ref_mask = data[ref_id]['battery_ids'] == bid
    true_soh = data[ref_id]['targets'][ref_mask].astype(float)
    n_pts    = len(true_soh)
    x        = np.arange(n_pts)

    fig, ax = plt.subplots(figsize=(8, 4.2))

    # 真值
    ax.plot(x, true_soh, color='black', lw=1.8, ls='-',
            zorder=10, label='True SOH')

    # 两个模型
    for exp_id, cfg in MODELS.items():
        d    = data[exp_id]
        mask = d['battery_ids'] == bid
        pred = d['predictions'][mask].astype(float)
        nn   = min(len(pred), n_pts)
        rmse, mae = _metrics(pred[:nn], true_soh[:nn])

        ax.plot(x[:nn], pred[:nn],
                color   = cfg['color'],
                lw      = cfg['lw'],
                ls      = cfg['ls'],
                alpha   = cfg['alpha'],
                zorder  = cfg['zorder'],
                label   = f'{cfg["label"]} (RMSE={rmse:.3f}%)')

    ax.set_xlabel('Window Index', fontsize=10)
    ax.set_ylabel('SOH',          fontsize=10)
    ax.set_title(f'Battery {bid} ({tag})', fontsize=11, fontweight='bold')
    ax.set_xlim(0, n_pts - 1)
    ax.legend(fontsize=9, loc='lower left',
              framealpha=0.85, edgecolor='#CCCCCC')
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    safe_bid = bid.replace('/', '_')
    for ext in ('png', 'pdf'):
        fpath = os.path.join(OUTPUT_DIR, f'fig4_8_soh_{safe_bid}.{ext}')
        plt.savefig(fpath)
        print(f'[OK] → {fpath}')
    plt.close()


# ════════════════════════════════════════════════════════════════
# 图2：预测误差轨迹
# ════════════════════════════════════════════════════════════════

def plot_error(data, bid, tag='Unseen'):
    ref_id   = 'cnn_lstm'
    ref_mask = data[ref_id]['battery_ids'] == bid
    true_soh = data[ref_id]['targets'][ref_mask].astype(float)
    n_pts    = len(true_soh)
    x        = np.arange(n_pts)

    fig, ax = plt.subplots(figsize=(8, 3.5))

    # 零线
    ax.axhline(0, color='red', lw=1.2, ls='--', zorder=1, label='Zero Error')

    # 两个模型的误差
    for exp_id, cfg in MODELS.items():
        d    = data[exp_id]
        mask = d['battery_ids'] == bid
        pred = d['predictions'][mask].astype(float)
        nn   = min(len(pred), n_pts)
        rmse, mae = _metrics(pred[:nn], true_soh[:nn])
        err = (pred[:nn] - true_soh[:nn]) * 100    # 有符号误差，单位 %

        ax.plot(x[:nn], err,
                color  = cfg['color'],
                lw     = cfg['lw'] * 0.9,
                ls     = cfg['ls'],
                alpha  = cfg['alpha'],
                zorder = cfg['zorder'],
                label  = f'{cfg["label"]} (RMSE={rmse:.3f}%, MAE={mae:.3f}%)')

    ax.set_xlabel('Window Index', fontsize=10)
    ax.set_ylabel('Pred - True',  fontsize=10)
    ax.set_title(f'Battery {bid} Prediction Error ({tag})',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0, n_pts - 1)
    ax.legend(fontsize=9, loc='lower left',
              framealpha=0.85, edgecolor='#CCCCCC')
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    safe_bid = bid.replace('/', '_')
    for ext in ('png', 'pdf'):
        fpath = os.path.join(OUTPUT_DIR, f'fig4_8_err_{safe_bid}.{ext}')
        plt.savefig(fpath)
        print(f'[OK] → {fpath}')
    plt.close()


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Plot SOH trajectory: CNN-LSTM vs PI-MSCL')
    parser.add_argument('--battery', default='3-3',
                        metavar='BID',
                        help='电池 ID（默认：3-3）')
    parser.add_argument('--tag', default='Unseen',
                        help='括号内标注文字（默认：Unseen）')
    parser.add_argument('--list', action='store_true',
                        help='列出所有可用电池后退出')
    args = parser.parse_args()

    print('加载缓存...')
    data = load_cache()

    if args.list:
        bats = available_batteries(data)
        print(f'共 {len(bats)} 块电池：')
        for b in bats:
            print(f'  {b}')
        return

    bid = args.battery
    available = set(available_batteries(data))
    if bid not in available:
        print(f'ERROR: 电池 "{bid}" 不在缓存中。')
        print('可用电池：')
        for b in sorted(available):
            print(f'  {b}')
        sys.exit(1)

    print(f'\n绘制电池：{bid}  tag={args.tag}')
    plot_soh  (data, bid, tag=args.tag)
    plot_error(data, bid, tag=args.tag)

    print(f'\n完成。输出目录：{OUTPUT_DIR}')


if __name__ == '__main__':
    main()
