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
    'CNN-LSTM': {'mae': [1.059, 1.097, 1.144, 1.186],
                 'color': '#F5C542', 'marker': '^',
                 'lw': 1.5, 'ls': '--', 'zorder': 3},
    'PI-MSCL' : {'mae': [1.101, 1.174, 1.180, 1.134],
                 'color': '#F4831F', 'marker': 'D',
                 'lw': 2.5, 'ls': '-',  'zorder': 5},   # 加粗突出
}

# ── 绘图 ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

x = np.array(RATIOS)

for name, cfg in DATA.items():
    y = cfg['mae']
    ax.plot(x, y,
            color     = cfg['color'],
            marker    = cfg['marker'],
            linestyle = cfg['ls'],
            linewidth = cfg['lw'],
            markersize= 7,
            zorder    = cfg['zorder'],
            label     = name)

    # 数值标注（PI-MSCL 标在上方，其余标在下方，避免重叠）
    for xi, yi in zip(x, y):
        offset = 0.010 if name == 'PI-MSCL' else -0.016
        va     = 'bottom' if name == 'PI-MSCL' else 'top'
        ax.text(xi, yi + offset, f'{yi:.3f}',
                ha='center', va=va,
                fontsize=7.5,
                color=cfg['color'],
                fontweight='bold' if name == 'PI-MSCL' else 'normal')

# ── 坐标轴设置 ────────────────────────────────────────────────────
ax.set_xlabel('Supervision Ratio  r', fontsize=11)
ax.set_ylabel('MAE (%)', fontsize=11)
ax.set_title('MAE vs. Supervision Ratio — 4 Models', fontsize=12, fontweight='bold', pad=10)

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

# 添加稀疏度方向注释
ax.annotate('← Full supervision          Sparse →',
            xy=(0.5, -0.13), xycoords='axes fraction',
            ha='center', fontsize=8, color='#777777',
            style='italic')

plt.tight_layout()

for ext in ('png', 'pdf'):
    fpath = os.path.join(OUTPUT_DIR, f'fig4_9_mae_trend.{ext}')
    plt.savefig(fpath)
    print(f'[OK] → {fpath}')

plt.close()
