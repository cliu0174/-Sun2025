"""
生成 self-trajectory 版本的 5 类退化场景 SOH 示意图。

直接调用 evaluation/anomaly_detection.py 里的真实注入函数，
对模拟的"正常电池 SOH 轨迹"做同样的操作，确保示意图与实际代码 100% 一致。

输出：docs/exp12_degradation_scenarios_self_trajectory.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluation.anomaly_detection import (
    inject_self_jump_drop, inject_self_knee_sampling,
    inject_self_imbalance_drift, inject_self_lithium_plating,
    inject_self_resistance_rise,
)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
N = 500

def smooth(x, w=8):
    padded = np.pad(x, w // 2, mode='edge')
    return np.convolve(padded, np.ones(w) / w, mode='valid')[:len(x)]

# ── 构造模拟的"正常电池"SOH 与特征 ──
cycles_full = np.arange(N)
soh_normal  = 1.0 - 0.00045 * cycles_full - 2e-7 * cycles_full**2
soh_normal += np.random.normal(0, 0.002, N)
soh_normal  = np.clip(smooth(soh_normal, 6), 0, 1)

# 用 SOH 当"features[:, 0]"，让注入函数对 features 做同样的操作就直接得到 SOH 异常轨迹
features = soh_normal.reshape(-1, 1).astype(np.float64)
targets  = soh_normal.copy()

t0 = N // 2
SEVERITY = 'severe'

# ── 绘图 ──
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
axes = axes.flatten()

c_normal = '#888888'
c_fault  = '#E63946'
c_onset  = '#2196F3'
c_drift  = '#FF9800'
c_quad   = '#9C27B0'


def style_axis(ax, ylim=(0.55, 1.05), ylabel=False):
    ax.set_ylim(*ylim)
    ax.set_xlabel('Pseudo-cycle')
    if ylabel:
        ax.set_ylabel('SOH')
    ax.grid(True, alpha=0.3)


# ── Panel 0: 正常退化（参照） ──
ax = axes[0]
ax.plot(cycles_full, soh_normal, color=c_normal, lw=2)
ax.axhline(0.8, color='gray', ls='--', alpha=0.4, lw=1)
ax.text(10, 0.815, 'EOL threshold (0.8)', fontsize=7, color='gray')
ax.set_title('(0) Normal Degradation\n[Reference: same cell]',
             fontsize=10, fontweight='bold')
style_axis(ax, ylabel=True)


# ── A1 SOH Sudden Drop ──
ax = axes[1]
cf, fc = inject_self_jump_drop(features, targets, t0, SEVERITY)
soh_a1 = cf[:, 0]
x_a1   = np.arange(len(soh_a1))

ax.plot(cycles_full, soh_normal, color=c_normal, lw=2, ls='--',
        alpha=0.45, label='Original full life')
ax.plot(x_a1, soh_a1, color=c_fault, lw=2,
        label=f'A1 jump-drop ({SEVERITY})')
ax.axvline(fc, color=c_onset, ls=':', lw=1.5)
ax.annotate(f'Skip middle life\n→ jump to later state',
            xy=(fc, soh_a1[fc]),
            xytext=(fc + 30, soh_a1[fc] + 0.09),
            arrowprops=dict(arrowstyle='->', color=c_onset),
            fontsize=8, color=c_onset)
ax.annotate('', xy=(fc + 4, soh_a1[fc]),
            xytext=(fc - 4, soh_normal[fc]),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.2))
ax.text(fc + 9, (soh_normal[fc] + soh_a1[fc]) / 2, 'drop',
        fontsize=8, color='gray')
ax.set_title('A1) SOH Sudden Drop\n[concat(soh[:t0], soh[s0:])]',
             fontsize=10, fontweight='bold')
style_axis(ax)
ax.legend(fontsize=7, loc='lower left')


# ── A2 Capacity Knee-point ──
ax = axes[2]
cf, fc = inject_self_knee_sampling(features, t0, SEVERITY)
soh_a2 = cf[:, 0]
x_a2   = np.arange(len(soh_a2))

ax.plot(cycles_full, soh_normal, color=c_normal, lw=2, ls='--',
        alpha=0.45, label='Original full life')
_step_show = {'mild': 2, 'moderate': 3, 'severe': 4}[SEVERITY]
ax.plot(x_a2, soh_a2, color=c_fault, lw=2,
        label=f'A2 knee (step={_step_show})')
ax.axvline(fc, color=c_onset, ls=':', lw=1.5)
ax.annotate('Tail subsampled\n→ slope steepens',
            xy=(fc + 30, soh_a2[fc + 30]),
            xytext=(fc - 130, soh_a2[fc + 30] - 0.08),
            arrowprops=dict(arrowstyle='->', color=c_onset),
            fontsize=8, color=c_onset)
ax.set_title('A2) Capacity Knee-point\n[concat(soh[:t0], soh[t0::step])]',
             fontsize=10, fontweight='bold')
style_axis(ax, ylim=(0.35, 1.05))
ax.legend(fontsize=7, loc='upper right')


# ── A3 Inter-cell Imbalance ──
ax = axes[3]
cf = inject_self_imbalance_drift(features, t0, SEVERITY)
soh_a3 = cf[:, 0]

ax.plot(cycles_full, soh_normal, color=c_normal, lw=2, ls='--',
        alpha=0.45, label='Normal')
ax.plot(cycles_full, soh_a3, color=c_drift, lw=2,
        label=f'A3 imbalance drift ({SEVERITY})')
ax.fill_between(cycles_full[t0:], soh_normal[t0:], soh_a3[t0:],
                color=c_drift, alpha=0.15, label='Growing gap')
ax.axvline(t0, color=c_onset, ls=':', lw=1.5)
ax.annotate(f'Linear drift to future\n→ gap grows + endpoint extrapolated',
            xy=(t0 + 130, soh_a3[t0 + 130]),
            xytext=(t0 + 20, soh_a3[t0 + 130] + 0.10),
            arrowprops=dict(arrowstyle='->', color=c_onset),
            fontsize=8, color=c_onset)
ax.set_title('A3) Inter-cell Imbalance\n[Linear future-state mixing]',
             fontsize=10, fontweight='bold')
style_axis(ax, ylabel=True)
ax.legend(fontsize=7, loc='lower left')


# ── A4 Lithium Plating ──
ax = axes[4]
cf, fc = inject_self_lithium_plating(features, targets, t0, SEVERITY)
soh_a4 = cf[:, 0]
x_a4   = np.arange(len(soh_a4))

ax.plot(cycles_full, soh_normal, color=c_normal, lw=2, ls='--',
        alpha=0.45, label='Original full life')
ax.plot(x_a4, soh_a4, color=c_fault, lw=2,
        label=f'A4 Li plating ({SEVERITY})')
ax.axvline(fc, color=c_onset, ls=':', lw=1.5)
ax.annotate('Step + skip sampling\n(jump + acceleration)',
            xy=(fc, soh_a4[fc]),
            xytext=(fc + 40, soh_a4[fc] + 0.12),
            arrowprops=dict(arrowstyle='->', color=c_onset),
            fontsize=8, color=c_onset)
ax.annotate('', xy=(fc + 4, soh_a4[fc]),
            xytext=(fc - 4, soh_normal[fc]),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.2))
ax.text(fc + 9, (soh_normal[fc] + soh_a4[fc]) / 2, 'step',
        fontsize=8, color='gray')
ax.set_title('A4) Lithium Plating\n[step-drop + skip sampling]',
             fontsize=10, fontweight='bold')
style_axis(ax, ylim=(0.35, 1.05))
ax.legend(fontsize=7, loc='upper right')


# ── A5 Internal Resistance Rise ──
ax = axes[5]
cf = inject_self_resistance_rise(features, t0, SEVERITY)
soh_a5 = cf[:, 0]

ax.plot(cycles_full, soh_normal, color=c_normal, lw=2, ls='--',
        alpha=0.45, label='Normal')
ax.plot(cycles_full, soh_a5, color=c_quad, lw=2,
        label=f'A5 R increase ({SEVERITY})')
ax.fill_between(cycles_full[t0:], soh_normal[t0:], soh_a5[t0:],
                color=c_quad, alpha=0.12, label='Quadratic gap')
ax.axvline(t0, color=c_onset, ls=':', lw=1.5)
ax.annotate('Quadratic drift\n→ slow start, fast end',
            xy=(int(N * 0.78), soh_a5[int(N * 0.78)]),
            xytext=(int(N * 0.78) - 150, soh_a5[int(N * 0.78)] - 0.08),
            arrowprops=dict(arrowstyle='->', color=c_quad),
            fontsize=8, color=c_quad)
ax.set_title('A5) Internal Resistance Rise\n[Quadratic future-state mixing]',
             fontsize=10, fontweight='bold')
style_axis(ax, ylim=(0.45, 1.05))
ax.legend(fontsize=7, loc='upper right')


plt.suptitle(f'Exp-12 Self-Trajectory Degradation Scenarios  '
             f'(severity = {SEVERITY}, generated from actual inject_self_* code)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
out_path = 'docs/exp12_degradation_scenarios_self_trajectory.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')
