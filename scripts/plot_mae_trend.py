"""
图4-9：不同监督比例下各模型MAE变化趋势对比图
================================================
数据来自表4-7（硬编码，无需训练）。
输出：figures/fig4_9_mae_trend.png / .pdf

用法：
    python scripts/plot_mae_trend.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 图形全局样式（与 plot_model_comparison.py 保持一致）────────────
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

# ── 表4-7 数据（MAE %）────────────────────────────────────────────
RATIOS = [1.0, 0.7, 0.5, 0.3]

DATA = {
    'XGBoost' : {'mae': [1.097, 1.139, 1.174, 1.238],
                 'color': '#63C5B5', 'marker': 'o',
                 'lw': 1.5, 'ls': '--', 'zorder': 3},
    'LSTM'    : {'mae': [1.164, 1.197, 1.268, 1.296],
                 'color': '#8BA7C7', 'marker': 's',
                 'lw': 1.5, 'ls': '--', 'zorder': 3},
    'CNN-LSTM': {'mae': [1.059, 1.147, 1.164, 1.186],
                 'color': '#F5C542', 'marker': '^',
                 'lw': 1.5, 'ls': '--', 'zorder': 3},
    'PI-MSCL' : {'mae': [1.101, 1.115, 1.137, 1.134],
                 'color': '#F4831F', 'marker': 'D',
                 'lw': 2.5, 'ls': '-',  'zorder': 5},   # 加粗突出
}

# ── 绘图 ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

def push_apart(positions, min_gap=0.022, iterations=200):
    """将一组 y 坐标迭代推开，确保相邻间距 >= min_gap，同时保持相对顺序。"""
    pos = sorted(range(len(positions)), key=lambda i: positions[i])
    arr = [positions[p] for p in pos]
    for _ in range(iterations):
        moved = False
        for i in range(len(arr) - 1):
            gap = arr[i + 1] - arr[i]
            if gap < min_gap:
                mid = (arr[i] + arr[i + 1]) / 2
                arr[i]     = mid - min_gap / 2
                arr[i + 1] = mid + min_gap / 2
                moved = True
        if not moved:
            break
    result = [0.0] * len(positions)
    for sorted_i, orig_i in enumerate(pos):
        result[orig_i] = arr[sorted_i]
    return result


x = np.array(RATIOS)
names  = list(DATA.keys())
colors = [DATA[n]['color'] for n in names]

# 先画折线
for name, cfg in DATA.items():
    ax.plot(x, cfg['mae'],
            color     = cfg['color'],
            marker    = cfg['marker'],
            linestyle = cfg['ls'],
            linewidth = cfg['lw'],
            markersize= 7,
            zorder    = cfg['zorder'],
            label     = name)

# 逐 x 位置智能放置标签
for xi_idx, xi in enumerate(x):
    y_data = [DATA[n]['mae'][xi_idx] for n in names]

    # 初始偏移：奇偶交替上/下
    order   = sorted(range(len(y_data)), key=lambda i: y_data[i])
    offsets = [0.0] * len(names)
    for rank, orig_i in enumerate(order):
        offsets[orig_i] = +0.011 if rank % 2 == 1 else -0.011

    raw_label_y = [y_data[i] + offsets[i] for i in range(len(names))]
    adj_label_y = push_apart(raw_label_y, min_gap=0.022)

    for i, name in enumerate(names):
        yi       = y_data[i]
        label_y  = adj_label_y[i]
        va       = 'bottom' if label_y >= yi else 'top'
        ax.text(xi, label_y, f'{yi:.3f}',
                ha='center', va=va,
                fontsize=7.5,
                color=DATA[name]['color'],
                fontweight='bold' if name == 'PI-MSCL' else 'normal')

# ── 坐标轴设置 ────────────────────────────────────────────────────
ax.set_xlabel('Supervision Ratio  r', fontsize=11)
ax.set_ylabel('MAE (%)', fontsize=11)

ax.set_xticks(RATIOS)
ax.set_xticklabels([str(r) for r in RATIOS], fontsize=10)

# x 轴反向（左=全监督，右=极稀疏），符合论文叙事方向
ax.invert_xaxis()
ax.set_xlim(1.05, 0.25)

ymin = min(v for d in DATA.values() for v in d['mae'])
ymax = max(v for d in DATA.values() for v in d['mae'])
ax.set_ylim(ymin - 0.05, ymax + 0.06)

# 只保留水平虚线网格
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# 图例
ax.legend(fontsize=9, loc='upper right',
          framealpha=0.85, edgecolor='#CCCCCC',
          handlelength=2.0)


plt.tight_layout()

for ext in ('png', 'pdf'):
    fpath = os.path.join(OUTPUT_DIR, f'fig4_9_mae_trend.{ext}')
    plt.savefig(fpath)
    print(f'[OK] → {fpath}')

plt.close()
