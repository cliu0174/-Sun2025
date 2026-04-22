"""
消融实验结果可视化脚本
读取 metrics.json，生成对比图表
"""

import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 读取数据 ──────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, 'metrics.json'), encoding='utf-8') as f:
    data = json.load(f)

RATIOS = [str(r) for r in data['ratios']]          # ['0.5', '0.3']
EXPS   = [e for e in data['experiments'] if e['by_ratio']]  # 过滤无结果的 E5

labels  = [e['label']  for e in EXPS]
exp_ids = [e['exp_id'] for e in EXPS]
n_exp   = len(EXPS)

def get(metric_mean, metric_std, ratio):
    means, stds = [], []
    for e in EXPS:
        d = e['by_ratio'].get(ratio, {})
        means.append(d.get(metric_mean, np.nan))
        stds.append(d.get(metric_std,  np.nan))
    return np.array(means), np.array(stds)

mae_m  = {r: get('mae_mean',  'mae_std',  r) for r in RATIOS}
rmse_m = {r: get('rmse_mean', 'rmse_std', r) for r in RATIOS}
r2_m   = {r: get('r2_mean',   'r2_mean',  r) for r in RATIOS}   # r2 无 std 字段

# rel_improvement（正 = 优于 baseline）
def get_rel(ratio):
    vals = []
    for e in EXPS:
        d = e['by_ratio'].get(ratio, {})
        vals.append(d.get('rel_improvement', None))
    return vals

rel = {r: get_rel(r) for r in RATIOS}

# ── 颜色方案 ──────────────────────────────────────────────────────────────────
COLORS = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
RATIO_COLORS = {'0.5': '#2196F3', '0.3': '#FF9800'}

# ── Figure 主布局 ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#F8F9FA')
gs  = GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38,
               top=0.91, bottom=0.07, left=0.07, right=0.97)

fig.suptitle('消融实验结果对比  (Seeds: 999/123/34 × Ratios: 0.5/0.3)',
             fontsize=15, fontweight='bold', y=0.975)

x = np.arange(n_exp)
w = 0.35

# ── 辅助：柱状图（双 ratio） ──────────────────────────────────────────────────
def bar_dual(ax, means_05, stds_05, means_03, stds_03,
             ylabel, title, higher_better=False, yunit=''):
    b1 = ax.bar(x - w/2, means_05, w, yerr=stds_05,
                color=RATIO_COLORS['0.5'], alpha=0.82, capsize=4,
                label='ratio=0.5', error_kw={'linewidth': 1.2})
    b2 = ax.bar(x + w/2, means_03, w, yerr=stds_03,
                color=RATIO_COLORS['0.3'], alpha=0.82, capsize=4,
                label='ratio=0.3', error_kw={'linewidth': 1.2})

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, rotation=20, ha='right')
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_facecolor('#FFFFFF')

    # 在柱顶标值
    for bars, means in [(b1, means_05), (b2, means_03)]:
        for bar, val in zip(bars, means):
            if not np.isnan(val):
                txt = f'{val*1000:.2f}' if yunit == '×10⁻³' else f'{val:.4f}'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00005,
                        txt, ha='center', va='bottom', fontsize=6.5, rotation=60)

    # 标 baseline 参考线
    ax.axvline(x=0 + w/2 + 0.02, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

# ── Row 0: MAE & RMSE & R² 柱状图 ────────────────────────────────────────────
ax_mae  = fig.add_subplot(gs[0, 0])
ax_rmse = fig.add_subplot(gs[0, 1])
ax_r2   = fig.add_subplot(gs[0, 2])

bar_dual(ax_mae,
         mae_m['0.5'][0], mae_m['0.5'][1],
         mae_m['0.3'][0], mae_m['0.3'][1],
         'MAE', 'MAE 对比', yunit='×10⁻³')

bar_dual(ax_rmse,
         rmse_m['0.5'][0], rmse_m['0.5'][1],
         rmse_m['0.3'][0], rmse_m['0.3'][1],
         'RMSE', 'RMSE 对比', yunit='×10⁻³')

# R² 无 std，单独画
r2_05 = np.array([e['by_ratio']['0.5']['r2_mean'] for e in EXPS])
r2_03 = np.array([e['by_ratio']['0.3']['r2_mean'] for e in EXPS])
ax_r2.bar(x - w/2, r2_05, w, color=RATIO_COLORS['0.5'], alpha=0.82, label='ratio=0.5')
ax_r2.bar(x + w/2, r2_03, w, color=RATIO_COLORS['0.3'], alpha=0.82, label='ratio=0.3')
ax_r2.set_title('R² 对比（越高越好）', fontsize=11, fontweight='bold', pad=8)
ax_r2.set_ylabel('R²', fontsize=9)
ax_r2.set_xticks(x); ax_r2.set_xticklabels(labels, fontsize=8.5, rotation=20, ha='right')
ax_r2.legend(fontsize=8, framealpha=0.7)
ax_r2.set_ylim(0.940, 0.982)
ax_r2.grid(axis='y', linestyle='--', alpha=0.4)
ax_r2.set_facecolor('#FFFFFF')
for i, (v5, v3) in enumerate(zip(r2_05, r2_03)):
    ax_r2.text(i - w/2, v5 + 0.0003, f'{v5:.4f}', ha='center', va='bottom', fontsize=6.5, rotation=60)
    ax_r2.text(i + w/2, v3 + 0.0003, f'{v3:.4f}', ha='center', va='bottom', fontsize=6.5, rotation=60)

# ── Row 1: 相对提升热力图 + 折线图 ───────────────────────────────────────────
ax_heat  = fig.add_subplot(gs[1, :2])
ax_delta = fig.add_subplot(gs[1, 2])

# 相对提升热力图
rel_matrix = np.array([
    [e['by_ratio'][r].get('rel_improvement') or np.nan for r in RATIOS]
    for e in EXPS
], dtype=float)  # shape: (n_exp, 2)

from matplotlib.colors import TwoSlopeNorm
vmax = np.nanmax(np.abs(rel_matrix[1:]))   # 排除 baseline 自身（全 nan）
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax_heat.imshow(rel_matrix.T, aspect='auto', cmap='RdYlGn', norm=norm)
plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.03, label='相对 MAE 提升 (%)')
ax_heat.set_xticks(range(n_exp)); ax_heat.set_xticklabels(labels, fontsize=9, rotation=20, ha='right')
ax_heat.set_yticks([0, 1]); ax_heat.set_yticklabels(['ratio=0.5', 'ratio=0.3'], fontsize=9)
ax_heat.set_title('相对 MAE 提升热力图（vs Baseline，正值=提升）', fontsize=11, fontweight='bold', pad=8)
for i in range(n_exp):
    for j, r in enumerate(RATIOS):
        v = rel_matrix[i, j]
        if not np.isnan(v):
            txt = f'{v:+.1f}%'
            color = 'black' if abs(v) < vmax * 0.6 else 'white'
            ax_heat.text(i, j, txt, ha='center', va='center',
                        fontsize=9, fontweight='bold', color=color)

# Delta MAE 折线（按 ratio 分开）
for r, color, marker in [('0.5', RATIO_COLORS['0.5'], 'o'), ('0.3', RATIO_COLORS['0.3'], 's')]:
    vals = [e['by_ratio'][r].get('delta_mae', None) for e in EXPS]
    ys   = [v * 1000 if v is not None else np.nan for v in vals]
    ax_delta.plot(range(n_exp), ys, color=color, marker=marker,
                  linewidth=1.8, markersize=6, label=f'ratio={r}')
ax_delta.axhline(0, color='gray', linestyle='--', linewidth=1)
ax_delta.set_title('MAE 绝对提升量（×10⁻³）', fontsize=11, fontweight='bold', pad=8)
ax_delta.set_ylabel('Δ MAE (×10⁻³)', fontsize=9)
ax_delta.set_xticks(range(n_exp)); ax_delta.set_xticklabels(labels, fontsize=8, rotation=20, ha='right')
ax_delta.legend(fontsize=8, framealpha=0.7)
ax_delta.grid(linestyle='--', alpha=0.4)
ax_delta.set_facecolor('#FFFFFF')
ax_delta.fill_between(range(n_exp),
    [e['by_ratio']['0.5'].get('delta_mae', 0) or 0 for e in EXPS],
    0, alpha=0.12, color=RATIO_COLORS['0.5'])

# ── Row 2: 数值汇总表 ─────────────────────────────────────────────────────────
ax_tbl = fig.add_subplot(gs[2, :])
ax_tbl.axis('off')

col_labels = ['实验', '说明',
              'MAE@0.5', 'RMSE@0.5', 'R²@0.5', 'Δ MAE@0.5',
              'MAE@0.3', 'RMSE@0.3', 'R²@0.3', 'Δ MAE@0.3']

rows = []
for e in EXPS:
    d5 = e['by_ratio'].get('0.5', {})
    d3 = e['by_ratio'].get('0.3', {})
    def fmt_mae(d):
        m, s = d.get('mae_mean', np.nan), d.get('mae_std', np.nan)
        return f'{m*1000:.3f}±{s*1000:.3f}' if not np.isnan(m) else '—'
    def fmt_rmse(d):
        m, s = d.get('rmse_mean', np.nan), d.get('rmse_std', np.nan)
        return f'{m*1000:.3f}±{s*1000:.3f}' if not np.isnan(m) else '—'
    def fmt_r2(d):
        v = d.get('r2_mean', np.nan)
        return f'{v:.4f}' if not np.isnan(v) else '—'
    def fmt_delta(d):
        v = d.get('rel_improvement', None)
        return f'{v:+.2f}%' if v is not None else '—'
    rows.append([
        e['label'], e['note'],
        fmt_mae(d5), fmt_rmse(d5), fmt_r2(d5), fmt_delta(d5),
        fmt_mae(d3), fmt_rmse(d3), fmt_r2(d3), fmt_delta(d3),
    ])

tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                   cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)

# 样式：表头
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#2C3E50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

# 样式：交替行 + 高亮最优
best_mae5_idx = int(np.nanargmin([e['by_ratio']['0.5'].get('mae_mean', np.inf) for e in EXPS]))
best_mae3_idx = int(np.nanargmin([e['by_ratio']['0.3'].get('mae_mean', np.inf) for e in EXPS]))

for i, row in enumerate(rows):
    bg = '#EBF5FB' if i % 2 == 0 else '#FDFEFE'
    for j in range(len(col_labels)):
        tbl[i+1, j].set_facecolor(bg)
        # 高亮最优 MAE
        if i == best_mae5_idx and j == 2:
            tbl[i+1, j].set_facecolor('#A9DFBF')
        if i == best_mae3_idx and j == 6:
            tbl[i+1, j].set_facecolor('#A9DFBF')

ax_tbl.set_title('数值汇总表  (MAE/RMSE 单位: ×10⁻³，mean±std，绿色=该 ratio 最优 MAE)',
                 fontsize=10, fontweight='bold', pad=6)

# ── 保存 ─────────────────────────────────────────────────────────────────────
out_path = os.path.join(script_dir, 'ablation_results.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'已保存: {out_path}')
plt.show()
