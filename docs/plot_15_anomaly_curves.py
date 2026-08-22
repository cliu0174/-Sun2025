"""
生成 5 种异常场景 × 3 个严重度 = 15 条异常 SOH 曲线 + 1 条正常参照。

使用真实 HUST 电池数据：从测试集中挑一块寿命较长的电池作为底子，
对其真实 SOH 序列调用 inject_self_* 函数得到异常版本。

输出：docs/exp12_15_anomaly_curves.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
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

# ── 加载真实 HUST 电池数据 ──
# 选 5-3：测试集里寿命最长的（约 2650 cycle）
BATTERY_ID = '5-3'
CSV_PATH = f'data/HUST data/{BATTERY_ID}.csv'

df = pd.read_csv(CSV_PATH)
print(f"Battery: {BATTERY_ID}, total cycles = {len(df)}")
print(f"Columns: {df.columns.tolist()[:5]}...")

# 容量 → SOH（按额定容量 1.1 Ah 归一化）
RATED_CAPACITY = 1.1
soh_normal = (df['capacity'].values / RATED_CAPACITY).astype(np.float64)
soh_normal = np.clip(soh_normal, 0, 1.2)   # 允许略大于 1（初始活化期）

# 输入特征（16 维）
feature_columns = [
    'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
    'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
    'current mean', 'current std', 'current kurtosis', 'current skewness',
    'CV Q', 'CV charge time', 'current slope', 'current entropy',
]
features_full = df[feature_columns].values.astype(np.float64)

N = len(soh_normal)
cycles_full = np.arange(N)
t0 = N // 2   # 50% 寿命

print(f"SOH range: {soh_normal.min():.3f} ~ {soh_normal.max():.3f}")
print(f"Fault injection at cycle t0 = {t0} (50% lifecycle)")

# ── 5 种场景 × 3 个严重度 ──
scenarios = [
    ('A1 Sudden Drop',         'sudden_aging',    inject_self_jump_drop,         True),
    ('A2 Capacity Knee',       'knee_point',      inject_self_knee_sampling,     False),
    ('A3 Imbalance Drift',     'imbalance',       inject_self_imbalance_drift,   False),
    ('A4 Lithium Plating',     'li_plating',      inject_self_lithium_plating,   True),
    ('A5 Resistance Rise',     'resistance_rise', inject_self_resistance_rise,   False),
]
severities = ['mild', 'moderate', 'severe']

scenario_colors = {
    'A1 Sudden Drop':     ['#FFB3B3', '#FF6666', '#CC0000'],
    'A2 Capacity Knee':   ['#FFDDAA', '#FFAA44', '#CC6600'],
    'A3 Imbalance Drift': ['#FFEE88', '#FFCC22', '#998800'],
    'A4 Lithium Plating': ['#FFB3FF', '#CC44CC', '#660066'],
    'A5 Resistance Rise': ['#B3D9FF', '#4488CC', '#003366'],
}

# 注入函数操作的是 SOH 序列（作为单维 feature）
# 关键：用 SOH 当 features[:, 0]，让 inject_self_* 对 SOH 做轨迹重排
soh_as_feature = soh_normal.reshape(-1, 1).astype(np.float64)
targets        = soh_normal.copy()

# ── 绘图：5 行（场景）× 3 列（严重度）= 15 子图 ──
fig, axes = plt.subplots(5, 3, figsize=(15, 14), sharex=False)

ylim_low = max(0.4, soh_normal.min() - 0.05)

for i, (label, key, inject_fn, needs_targets) in enumerate(scenarios):
    for j, sev in enumerate(severities):
        ax = axes[i, j]

        if needs_targets:
            result = inject_fn(soh_as_feature, targets, t0, sev)
        else:
            result = inject_fn(soh_as_feature, t0, sev)

        if isinstance(result, tuple):
            cf, fc = result
            soh_corrupted = cf[:, 0]
            fault_pos = fc
        else:
            cf = result
            soh_corrupted = cf[:, 0]
            fault_pos = t0
        x_corrupted = np.arange(len(soh_corrupted))

        # 正常参照（真实曲线）
        ax.plot(cycles_full, soh_normal, color='#888888', lw=1.2, ls='--',
                alpha=0.5, label=f'Normal ({BATTERY_ID})')
        # 异常曲线
        color = scenario_colors[label][j]
        ax.plot(x_corrupted, soh_corrupted, color=color, lw=1.8, label=sev)
        ax.axvline(fault_pos, color='#2196F3', ls=':', lw=1, alpha=0.7)

        if j == 0:
            ax.set_ylabel(f'{label}\nSOH', fontsize=9, fontweight='bold')
        if i == 0:
            ax.set_title(f'severity = {sev}', fontsize=10, fontweight='bold')
        if i == 4:
            ax.set_xlabel('Pseudo-cycle')

        len_change = len(soh_corrupted) - N
        len_label = (f'len: {N}→{len(soh_corrupted)} ({len_change:+d})'
                     if len_change != 0 else f'len: {N}')
        ax.text(0.97, 0.05, len_label, transform=ax.transAxes,
                fontsize=7, color='gray', ha='right')

        ax.set_ylim(ylim_low, soh_normal.max() + 0.05)
        ax.set_xlim(0, N + 50)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='lower left')

plt.suptitle(f'15 Anomaly SOH Curves on Real HUST Battery [{BATTERY_ID}, '
             f'lifetime={N} cycles, fault@{t0}]\n'
             f'(generated from inject_self_* code; gray dashed = real normal trajectory)',
             fontsize=12, fontweight='bold', y=1.00)
plt.tight_layout()
out_path = 'docs/exp12_15_anomaly_curves.png'
plt.savefig(out_path, dpi=130, bbox_inches='tight')
print(f'\nSaved: {out_path}')
