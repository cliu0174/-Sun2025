import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
N = 500

def smooth(x, w=8):
    # 用边缘值填充，避免首尾边界下坠
    padded = np.pad(x, w // 2, mode='edge')
    return np.convolve(padded, np.ones(w) / w, mode='valid')[:len(x)]

cycles = np.arange(N)

# 正常退化基准
soh_normal = 1.0 - 0.00045 * cycles - 2e-7 * cycles**2
soh_normal += np.random.normal(0, 0.002, N)
soh_normal = np.clip(smooth(soh_normal, 6), 0, 1)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

c_normal = '#888888'
c_fault  = '#E63946'
c_onset  = '#2196F3'

titles = [
    '(0) Normal Degradation\n[Reference]',
    '(1) Single-Cell Sudden Aging\n[Already in Exp-12]',
    '(2) Capacity Knee-point',
    '(3) Inter-cell Imbalance\n[Gradual cluster drift]',
    '(4) Lithium Plating\n[Step-drop + acceleration]',
    '(5) Internal Resistance Rise\n[Progressive slope increase]',
]

# ── Panel 0: 正常退化 ──
ax = axes[0]
ax.plot(cycles, soh_normal, color=c_normal, lw=2)
ax.axhline(0.8, color='gray', ls='--', alpha=0.4, lw=1)
ax.text(10, 0.815, 'EOL threshold (0.8)', fontsize=7, color='gray')
ax.set_title(titles[0], fontsize=10, fontweight='bold')
ax.set_ylim(0.55, 1.05); ax.set_xlabel('Cycle'); ax.set_ylabel('SOH')
ax.grid(True, alpha=0.3)

# ── Panel 1: 单电芯加速老化（突变，50%处） ──
ax = axes[1]
fault_at = N // 2
soh_fault = soh_normal.copy()
for t in range(fault_at, N):
    alpha = 0.5
    donor_val = 0.72 - 0.0003 * (t - fault_at)
    soh_fault[t] = (1 - alpha) * soh_normal[t] + alpha * donor_val
soh_fault = np.clip(soh_fault, 0, 1)
ax.plot(cycles, soh_normal, color=c_normal, lw=2, ls='--', label='Normal', alpha=0.5)
ax.plot(cycles, soh_fault, color=c_fault, lw=2, label='Fault cell')
ax.axvline(fault_at, color=c_onset, ls=':', lw=1.5)
ax.annotate('Fault injected\n(50% lifecycle)', xy=(fault_at, soh_fault[fault_at]),
            xytext=(fault_at + 40, soh_fault[fault_at] + 0.06),
            arrowprops=dict(arrowstyle='->', color=c_onset), fontsize=8, color=c_onset)
ax.set_title(titles[1], fontsize=10, fontweight='bold')
ax.set_ylim(0.55, 1.05); ax.set_xlabel('Cycle')
ax.legend(fontsize=7, loc='lower left'); ax.grid(True, alpha=0.3)

# ── Panel 2: 容量拐点 ──
ax = axes[2]
knee = int(N * 0.55)
soh_knee = soh_normal.copy()
for t in range(knee, N):
    extra = 0.0009 * (t - knee)
    soh_knee[t] = soh_normal[t] - extra
soh_knee = np.clip(soh_knee, 0, 1)
ax.plot(cycles, soh_normal, color=c_normal, lw=2, ls='--', label='Normal', alpha=0.5)
ax.plot(cycles, soh_knee, color=c_fault, lw=2, label='Knee-point')
ax.axvline(knee, color=c_onset, ls=':', lw=1.5)
ax.annotate('Knee point\n(sudden\nacceleration)', xy=(knee, soh_knee[knee]),
            xytext=(knee - 100, soh_knee[knee] - 0.08),
            arrowprops=dict(arrowstyle='->', color=c_onset), fontsize=8, color=c_onset)
ax.set_title(titles[2], fontsize=10, fontweight='bold')
ax.set_ylim(0.35, 1.05); ax.set_xlabel('Cycle')
ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)

# ── Panel 3: 簇内不均衡加剧（渐进漂移） ──
ax = axes[3]
fault_start = int(N * 0.4)
soh_imbal = soh_normal.copy()
for t in range(fault_start, N):
    alpha_t = 0.4 * (t - fault_start) / (N - fault_start)
    soh_imbal[t] = soh_normal[t] - alpha_t * 0.15
soh_imbal = np.clip(soh_imbal, 0, 1)
ax.plot(cycles, soh_normal, color=c_normal, lw=2, ls='--', label='Normal', alpha=0.5)
ax.plot(cycles, soh_imbal, color='#FF9800', lw=2, label='Imbalance grows')
ax.fill_between(cycles[fault_start:], soh_normal[fault_start:],
                soh_imbal[fault_start:], alpha=0.15, color='#FF9800', label='Growing gap')
ax.axvline(fault_start, color=c_onset, ls=':', lw=1.5)
ax.annotate('Imbalance begins\n(gradual drift)', xy=(fault_start, soh_imbal[fault_start]),
            xytext=(fault_start + 50, soh_imbal[fault_start] + 0.06),
            arrowprops=dict(arrowstyle='->', color=c_onset), fontsize=8, color=c_onset)
ax.set_title(titles[3], fontsize=10, fontweight='bold')
ax.set_ylim(0.55, 1.05); ax.set_xlabel('Cycle')
ax.legend(fontsize=7, loc='lower left'); ax.grid(True, alpha=0.3)

# ── Panel 4: 析锂台阶 ──
ax = axes[4]
step_at = int(N * 0.45)
step_size = 0.07
soh_plating = soh_normal.copy()
for t in range(step_at, N):
    extra_rate = 0.00035 * (t - step_at)
    soh_plating[t] = soh_normal[t] - step_size - extra_rate
soh_plating = np.clip(soh_plating, 0, 1)
ax.plot(cycles, soh_normal, color=c_normal, lw=2, ls='--', label='Normal', alpha=0.5)
ax.plot(cycles[:step_at+1], soh_plating[:step_at+1], color=c_fault, lw=2)
ax.plot(cycles[step_at:], soh_plating[step_at:], color=c_fault, lw=2, label='Li Plating')
ax.axvline(step_at, color=c_onset, ls=':', lw=1.5)
ax.annotate('Sudden step-drop\nthen faster decay',
            xy=(step_at + 5, soh_plating[step_at + 5]),
            xytext=(step_at + 50, soh_plating[step_at + 5] + 0.08),
            arrowprops=dict(arrowstyle='->', color=c_onset), fontsize=8, color=c_onset)
# 标注台阶高度
ax.annotate('', xy=(step_at + 15, soh_plating[step_at + 5]),
            xytext=(step_at + 15, soh_normal[step_at + 5]),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.2))
ax.text(step_at + 18, (soh_normal[step_at+5] + soh_plating[step_at+5]) / 2,
        f'-{step_size:.0%}', fontsize=8, color='gray')
ax.set_title(titles[4], fontsize=10, fontweight='bold')
ax.set_ylim(0.35, 1.05); ax.set_xlabel('Cycle')
ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)

# ── Panel 5: 内阻渐进增长 ──
ax = axes[5]
fault_start5 = int(N * 0.35)
soh_resist = soh_normal.copy()
for t in range(fault_start5, N):
    extra = 0.00045 * ((t - fault_start5) / N) * (t - fault_start5)
    soh_resist[t] = soh_normal[t] - extra
soh_resist = np.clip(soh_resist, 0, 1)
ax.plot(cycles, soh_normal, color=c_normal, lw=2, ls='--', label='Normal', alpha=0.5)
ax.plot(cycles, soh_resist, color='#9C27B0', lw=2, label='R increase')
ax.axvline(fault_start5, color=c_onset, ls=':', lw=1.5)
ax.annotate('Slope gradually\nsteepens', xy=(int(N*0.65), soh_resist[int(N*0.65)]),
            xytext=(int(N*0.65) - 120, soh_resist[int(N*0.65)] - 0.06),
            arrowprops=dict(arrowstyle='->', color='#9C27B0'), fontsize=8, color='#9C27B0')
ax.set_title(titles[5], fontsize=10, fontweight='bold')
ax.set_ylim(0.35, 1.05); ax.set_xlabel('Cycle')
ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)

plt.suptitle('Battery SOH Degradation Scenarios for Exp-12 Anomaly Detection',
             fontsize=12, fontweight='bold')
plt.tight_layout()
out_path = 'docs/exp12_degradation_scenarios.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')
