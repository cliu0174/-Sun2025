import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
N = 500
cycles = np.arange(N)


def smooth(x, w=8):
    padded = np.pad(x, w // 2, mode='edge')
    return np.convolve(padded, np.ones(w) / w, mode='valid')[:len(x)]


def interp_to_length(y, n):
    src = np.arange(len(y))
    dst = np.linspace(0, len(y) - 1, n)
    return np.interp(dst, src, y)


# Reference trajectory from one battery.
soh_normal = 1.0 - 0.00045 * cycles - 2e-7 * cycles**2
soh_normal += np.random.normal(0, 0.002, N)
soh_normal = np.clip(smooth(soh_normal, 6), 0, 1)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

c_normal = '#888888'
c_fault = '#E63946'
c_onset = '#2196F3'
c_gap = '#B0B0B0'
c_imbal = '#FF9800'
c_resist = '#9C27B0'

titles = [
    '(0) Normal Degradation\n[Reference: same cell]',
    '(1) SOH Sudden Drop\n[Self trajectory skip]',
    '(2) Capacity Knee-point\n[Post-onset tail sampling]',
    '(3) Inter-cell Imbalance\n[Future-state mixing]',
    '(4) Lithium Plating\n[Step-drop + tail sampling]',
    '(5) Internal Resistance Rise\n[Quadratic future drift]',
]


def style_axis(ax, ylim=(0.55, 1.05), ylabel=False):
    ax.set_ylim(*ylim)
    ax.set_xlabel('Pseudo-cycle')
    if ylabel:
        ax.set_ylabel('SOH')
    ax.grid(True, alpha=0.3)


# Panel 0: normal degradation.
ax = axes[0]
ax.plot(cycles, soh_normal, color=c_normal, lw=2)
ax.axhline(0.8, color='gray', ls='--', alpha=0.4, lw=1)
ax.text(10, 0.815, 'EOL threshold (0.8)', fontsize=7, color='gray')
ax.set_title(titles[0], fontsize=10, fontweight='bold')
style_axis(ax, ylabel=True)

# Panel 1: sudden drop by skipping middle life segment.
ax = axes[1]
t0 = int(N * 0.50)
s0 = int(N * 0.68)
soh_jump = np.concatenate([soh_normal[:t0], soh_normal[s0:]])
x_jump = np.arange(len(soh_jump))
ax.plot(cycles, soh_normal, color=c_normal, lw=2, ls='--', alpha=0.45, label='Original full life')
ax.plot(x_jump, soh_jump, color=c_fault, lw=2, label='Constructed fault')
ax.axvline(t0, color=c_onset, ls=':', lw=1.5)
ax.annotate('Skip middle life\nSOH sudden drop',
            xy=(t0, soh_jump[t0]), xytext=(t0 + 40, soh_jump[t0] + 0.08),
            arrowprops=dict(arrowstyle='->', color=c_onset), fontsize=8, color=c_onset)
ax.annotate('', xy=(t0 + 8, soh_jump[t0]),
            xytext=(t0 + 8, soh_normal[t0]),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.2))
ax.text(t0 + 14, (soh_normal[t0] + soh_jump[t0]) / 2, 'drop', fontsize=8, color='gray')
ax.set_title(titles[1], fontsize=10, fontweight='bold')
style_axis(ax)
ax.legend(fontsize=7, loc='lower left')

# Panel 2: knee point - 按真实长度画，不再插值（修复：之前的 interp_to_length 导致末端反而比正常高）
ax = axes[2]
knee = int(N * 0.50)
sample_step = 3
tail_idx = np.arange(knee, N, sample_step)
soh_knee = np.concatenate([soh_normal[:knee], soh_normal[tail_idx]])   # 直接拼接，长度变短
x_knee = np.arange(len(soh_knee))
ax.plot(cycles, soh_normal, color=c_normal, lw=2, ls='--', alpha=0.45, label='Normal (orig)')
ax.plot(x_knee, soh_knee,   color=c_fault,  lw=2, label=f'Knee-point (step={sample_step})')
ax.axvline(knee, color=c_onset, ls=':', lw=1.5)
ax.annotate(f'Tail step={sample_step}\nslope x{sample_step} steeper',
            xy=(knee + 30, soh_knee[knee + 30]),
            xytext=(knee - 120, soh_knee[knee + 30] - 0.08),
            arrowprops=dict(arrowstyle='->', color=c_onset), fontsize=8, color=c_onset)
ax.set_title(titles[2], fontsize=10, fontweight='bold')
style_axis(ax, ylim=(0.35, 1.05))
ax.legend(fontsize=7, loc='upper right')

# Panel 3: imbalance by gradually mixing current state with future state.
ax = axes[3]
start = int(N * 0.40)
max_alpha = 0.55
max_offset = 90
soh_imbal = soh_normal.copy()
for t in range(start, N):
    progress = (t - start) / max(1, N - start - 1)
    alpha = max_alpha * progress
    future_idx = min(N - 1, t + int(max_offset * progress))
    soh_imbal[t] = (1 - alpha) * soh_normal[t] + alpha * soh_normal[future_idx]
ax.plot(cycles, soh_normal, color=c_normal, lw=2, ls='--', alpha=0.45, label='Normal')
ax.plot(cycles, soh_imbal, color=c_imbal, lw=2, label='Imbalance drift')
ax.fill_between(cycles[start:], soh_normal[start:], soh_imbal[start:],
                color=c_imbal, alpha=0.15, label='Growing gap')
ax.axvline(start, color=c_onset, ls=':', lw=1.5)
ax.annotate('Mixing ratio grows\nwith future-state offset',
            xy=(start + 120, soh_imbal[start + 120]),
            xytext=(start + 40, soh_imbal[start + 120] + 0.08),
            arrowprops=dict(arrowstyle='->', color=c_onset), fontsize=8, color=c_onset)
ax.set_title(titles[3], fontsize=10, fontweight='bold')
style_axis(ax, ylabel=True)
ax.legend(fontsize=7, loc='lower left')

# Panel 4: lithium plating - 同样按真实长度画，不再插值
ax = axes[4]
plate_at = int(N * 0.45)
plate_s0 = int(N * 0.60)
plate_step = 2
plate_idx = np.arange(plate_s0, N, plate_step)
soh_plate = np.concatenate([soh_normal[:plate_at], soh_normal[plate_idx]])   # 直接拼接，变短
x_plate = np.arange(len(soh_plate))
ax.plot(cycles, soh_normal, color=c_normal, lw=2, ls='--', alpha=0.45, label='Normal (orig)')
ax.plot(x_plate, soh_plate, color=c_fault,  lw=2, label=f'Li plating (step={plate_step})')
ax.axvline(plate_at, color=c_onset, ls=':', lw=1.5)
ax.annotate('Step drop + skip sampling\n(accelerated decay)',
            xy=(plate_at, soh_plate[plate_at]),
            xytext=(plate_at + 50, soh_plate[plate_at] + 0.12),
            arrowprops=dict(arrowstyle='->', color=c_onset), fontsize=8, color=c_onset)
# 台阶箭头
ax.annotate('', xy=(plate_at + 2, soh_plate[plate_at]),
            xytext=(plate_at - 2, soh_normal[plate_at]),
            arrowprops=dict(arrowstyle='->', color=c_fault, lw=1.5))
ax.text(plate_at + 8, (soh_normal[plate_at] + soh_plate[plate_at]) / 2,
        'step', fontsize=8, color='gray')
ax.set_title(titles[4], fontsize=10, fontweight='bold')
style_axis(ax, ylim=(0.35, 1.05))
ax.legend(fontsize=7, loc='upper right')

# Panel 5: resistance rise by quadratic drift toward a future state.
ax = axes[5]
rise_start = int(N * 0.35)
max_alpha = 0.75
offset = 120
soh_resist = soh_normal.copy()
for t in range(rise_start, N):
    progress = (t - rise_start) / max(1, N - rise_start - 1)
    alpha = max_alpha * progress**2
    future_idx = min(N - 1, t + offset)
    soh_resist[t] = (1 - alpha) * soh_normal[t] + alpha * soh_normal[future_idx]
ax.plot(cycles, soh_normal, color=c_normal, lw=2, ls='--', alpha=0.45, label='Normal')
ax.plot(cycles, soh_resist, color=c_resist, lw=2, label='R increase')
ax.axvline(rise_start, color=c_onset, ls=':', lw=1.5)
ax.annotate('Quadratic drift\ntoward later state',
            xy=(int(N * 0.73), soh_resist[int(N * 0.73)]),
            xytext=(int(N * 0.55), soh_resist[int(N * 0.73)] - 0.08),
            arrowprops=dict(arrowstyle='->', color=c_resist), fontsize=8, color=c_resist)
ax.set_title(titles[5], fontsize=10, fontweight='bold')
style_axis(ax, ylim=(0.35, 1.05))
ax.legend(fontsize=7, loc='upper right')

plt.suptitle('Self-Trajectory SOH Degradation Scenarios for Exp-12 Anomaly Detection',
             fontsize=12, fontweight='bold')
plt.tight_layout()
out_path = 'docs/exp12_degradation_scenarios_self_trajectory.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')
