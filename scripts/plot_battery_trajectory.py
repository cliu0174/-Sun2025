"""
图4-8：单电池 SOH 预测轨迹对比（CNN-LSTM vs PI-MSCL）
=======================================================
从 plot_model_comparison.py 生成的缓存（.npz）读取，无需重新训练。

用法：
    # 自动选电池（改善最大 + 中位各一块，2×1 布局）
    python scripts/plot_battery_trajectory.py

    # 指定一块电池
    python scripts/plot_battery_trajectory.py --battery CS2_36

    # 指定两块电池
    python scripts/plot_battery_trajectory.py --battery CS2_36 CS2_38

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

# ── 与 plot_model_comparison.py 保持一致 ────────────────────────
SEED  = 42
RATIO = 1.0

MODELS = {
    'cnn_lstm': {
        'label' : 'CNN-LSTM',
        'color' : '#F5C542',
        'lw'    : 1.5,
        'ls'    : '--',
        'zorder': 3,
    },
    'pi_ms_cnn_lstm': {
        'label' : 'PI-MSCL',
        'color' : '#F4831F',
        'lw'    : 2.2,
        'ls'    : '-',
        'zorder': 5,
    },
}

# ── 图形全局样式 ─────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'serif',
    'font.size'        : 10,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
    'axes.grid'        : True,
    'grid.alpha'       : 0.25,
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
            # battery_ids 可能是 object 数组，统一转 str
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
# 电池筛选逻辑
# ════════════════════════════════════════════════════════════════

def available_batteries(data):
    """返回两个模型都有数据的电池列表（按 id 排序）。"""
    ref_ids = set(data['cnn_lstm']['battery_ids'])
    for exp_id in MODELS:
        ref_ids &= set(data[exp_id]['battery_ids'])
    return sorted(ref_ids)


def select_batteries_auto(data, n=2):
    """
    自动选 n 块电池：
      - 按 PI-MSCL 相对于 CNN-LSTM 的 MAE 改善幅度排序
      - 返回 [改善最大, 中位改善] 各一块（n=2）
    """
    batteries = available_batteries(data)

    improvements = []
    for bid in batteries:
        maes = {}
        for exp_id in MODELS:
            d = data[exp_id]
            mask = d['battery_ids'] == bid
            if mask.sum() < 10:
                break
            maes[exp_id] = np.mean(np.abs(
                d['predictions'][mask] - d['targets'][mask]
            ))
        else:
            base = maes['cnn_lstm']
            ours = maes['pi_ms_cnn_lstm']
            rel_improve = (base - ours) / base * 100   # 正值 = PI-MSCL 更好
            improvements.append((bid, rel_improve))

    improvements.sort(key=lambda x: x[1], reverse=True)

    if len(improvements) == 0:
        print('WARNING: 没有找到有效电池，使用全部可用电池的前两块。')
        return batteries[:n]

    selected = [improvements[0][0]]   # 改善最大
    if n >= 2 and len(improvements) > 1:
        mid_idx = len(improvements) // 2
        selected.append(improvements[mid_idx][0])  # 中位

    print(f'自动选电池（按 PI-MSCL vs CNN-LSTM MAE 改善排序）：')
    for bid, imp in improvements[:5]:
        marker = ' ◀ 已选' if bid in selected else ''
        print(f'  {bid}  改善 {imp:+.2f}%{marker}')
    print(f'  ...')
    mid_bid = improvements[mid_idx][0] if len(improvements) > 1 else None
    if mid_bid and mid_bid not in improvements[:5]:
        imp_mid = dict(improvements)[mid_bid]
        print(f'  {mid_bid}  改善 {imp_mid:+.2f}%  ◀ 已选（中位）')

    return selected[:n]


# ════════════════════════════════════════════════════════════════
# 绘图
# ════════════════════════════════════════════════════════════════

def plot_trajectory(data, battery_list, out_prefix='fig4_8_trajectory'):
    """
    对每块电池绘制上下两图：
      上：SOH 预测轨迹（真值 + 两模型）
      下：预测误差轨迹（pred - true）
    多块电池时并排排列（列数 = len(battery_list)）。
    """
    n_bats = len(battery_list)
    fig, axes = plt.subplots(
        2, n_bats,
        figsize=(6.0 * n_bats, 7.0),
        gridspec_kw={'height_ratios': [2.5, 1.2], 'hspace': 0.08},
        squeeze=False,
    )

    # 参考模型（用来取真值）
    ref_id = 'cnn_lstm'

    for col, bid in enumerate(battery_list):
        ax_soh = axes[0][col]
        ax_err = axes[1][col]

        ref_mask = data[ref_id]['battery_ids'] == bid
        true_soh = data[ref_id]['targets'][ref_mask].astype(float)
        n_pts    = len(true_soh)
        cycles   = np.arange(n_pts)

        # ── 上图：SOH 轨迹 ──────────────────────────────
        ax_soh.plot(cycles, true_soh,
                    color='#333333', lw=1.8, ls='-',
                    zorder=10, label='True SOH')

        for exp_id, cfg in MODELS.items():
            d    = data[exp_id]
            mask = d['battery_ids'] == bid
            pred = d['predictions'][mask].astype(float)
            nn   = min(len(pred), n_pts)
            mae  = np.mean(np.abs(pred[:nn] - true_soh[:nn])) * 100

            ax_soh.plot(cycles[:nn], pred[:nn],
                        color=cfg['color'], lw=cfg['lw'], ls=cfg['ls'],
                        zorder=cfg['zorder'],
                        label=f'{cfg["label"]}  (MAE={mae:.3f}%)')

        ax_soh.set_ylabel('SOH', fontsize=10)
        ax_soh.set_title(f'Battery  {bid}', fontsize=11, fontweight='bold', pad=6)
        ax_soh.legend(fontsize=8.5, loc='lower left',
                      framealpha=0.85, edgecolor='#CCCCCC')
        ax_soh.tick_params(labelbottom=False)   # x 刻度留给下图

        # ── 下图：误差轨迹 ──────────────────────────────
        for exp_id, cfg in MODELS.items():
            d    = data[exp_id]
            mask = d['battery_ids'] == bid
            pred = d['predictions'][mask].astype(float)
            nn   = min(len(pred), n_pts)
            err  = (pred[:nn] - true_soh[:nn]) * 100  # 有符号误差，单位 %

            ax_err.plot(cycles[:nn], err,
                        color=cfg['color'], lw=cfg['lw'] * 0.85,
                        ls=cfg['ls'], zorder=cfg['zorder'],
                        label=cfg['label'])

        ax_err.axhline(0, color='#666666', lw=0.8, ls='--', zorder=1)
        ax_err.set_xlabel('Cycle Window', fontsize=10)
        ax_err.set_ylabel('Error (%)', fontsize=9)
        ax_err.tick_params(labelsize=8)

        # 对齐 x 轴范围
        ax_soh.set_xlim(0, n_pts - 1)
        ax_err.set_xlim(0, n_pts - 1)

    plt.tight_layout()

    for ext in ('png', 'pdf'):
        fpath = os.path.join(OUTPUT_DIR, f'{out_prefix}.{ext}')
        plt.savefig(fpath)
        print(f'[OK] → {fpath}')

    plt.close()


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Plot SOH trajectory: CNN-LSTM vs PI-MSCL')
    parser.add_argument('--battery', nargs='+', default=None,
                        metavar='BID',
                        help='指定电池 ID（可多个），不指定则自动选取')
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

    if args.battery:
        battery_list = args.battery
        # 验证电池是否存在
        available = set(available_batteries(data))
        for bid in battery_list:
            if bid not in available:
                print(f'WARNING: 电池 "{bid}" 在缓存中不存在，将跳过。')
        battery_list = [b for b in battery_list if b in available]
        if not battery_list:
            print('ERROR: 所有指定电池均不存在于缓存中。')
            sys.exit(1)
    else:
        battery_list = select_batteries_auto(data, n=2)

    print(f'\n绘制电池：{battery_list}')
    plot_trajectory(data, battery_list)

    print(f'\n完成。输出目录：{OUTPUT_DIR}')


if __name__ == '__main__':
    main()
