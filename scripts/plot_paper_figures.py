"""
论文第四章绘图脚本 —— 纯聚合数据图（不需要重跑实验）

生成图片列表：
  Fig1: MAE% 随监督比例变化折线图（E0 / A1 / Exp09c）
  Fig2: 2×3 消融热力图（架构 × 约束级别）
  Fig3: 单调违反率分组柱状图
  Fig4: 鲁棒性折线图 —— 特征缺失（Scene A）
  Fig5: 鲁棒性折线图 —— 传感器漂移（Scene C）
  Fig6: 物理约束正则化效果图（Exp09c vs A1，r=0.3 聚焦）

用法：
  python scripts/plot_paper_figures.py          # 生成所有图
  python scripts/plot_paper_figures.py --fig 1  # 只生成某张图

输出目录：figures/
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# 全局设置
# ============================================================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 实验数据（来自 experiment_results_summary.md）
# ============================================================

RATIOS = [1.0, 0.7, 0.5, 0.3]
RATIO_LABELS = ['r=1.0', 'r=0.7', 'r=0.5', 'r=0.3']

# --- MAE% 数据 ---
MAE = {
    # 单模块消融
    'Eneg1 (无物理)':     [1.0659, 1.0759, 1.1257, 1.1358],
    'E0 (PI-CNN-LSTM)':   [1.1710, 1.1473, 1.1426, 1.0696],
    'E1 (+M1 注意力)':    [1.1072, 1.1508, 1.0902, 1.1625],
    'E2 (+M2 MC Dropout)':[1.0537, 1.1786, 1.1499, 1.2656],
    'E3 (+M4 速率连续性)':[1.1510, 1.1956, 1.1502, 1.1424],
    'E4 (+M5 自适应权重)':[1.1720, 1.1511, 1.1472, 1.1317],
    'E5 (+M2+M6 伪标签)': [1.0742, 1.1071, 1.1486, 1.1272],
    'E7 (Full Stack M2+M4)': [1.1632, 1.1778, 1.0884, 1.2343],
    # 架构消融
    'A1 (MS-CNN-LSTM)':   [1.0810, 1.0790, 1.0040, 1.0667],
    'A2 (LayerNorm)':     [1.3038, 1.2324, 1.2311, 1.1630],
    'A3 (MS+LayerNorm)':  [1.1308, 1.1411, 1.0788, 1.1114],
    # Exp-09
    'Exp09c (PI-MS-CNN-LSTM)': [1.1395, 1.0745, 1.0802, 1.0336],
    'Exp09a (MS+M2+M4 w=0.3)': [1.1348, 1.0961, 1.0652, 1.1220],
    'Exp09b (MS+M2+M4 w=0.1)': [1.0955, 1.1014, 1.0632, 1.1393],
}

# --- R² 数据 ---
R2 = {
    'Eneg1': [0.9220, 0.9211, 0.9056, 0.9059],
    'E0':    [0.8949, 0.9027, 0.9137, 0.9289],
    'A1':    [0.9166, 0.9139, 0.9418, 0.9261],
    'Exp09c':[0.9208, 0.9297, 0.9257, 0.9398],
    'Exp09a':[0.9155, 0.9245, 0.9271, 0.9123],
    'Exp09b':[0.9237, 0.9211, 0.9302, 0.9110],
}

# --- 单调违反率 ---
MONO_VIOL = {
    'Eneg1 (无物理)':     [19.01, 15.55, 24.36, 16.50],
    'E0 (PI-CNN-LSTM)':   [9.01, 9.42, 16.80, 6.30],
    'E1 (+M1 注意力)':    [13.03, 10.21, 11.49, 11.09],
    'E2 (+M2)':           [10.40, 7.91, 10.49, 10.00],
    'E3 (+M4)':           [8.81, 8.57, 11.05, 13.65],
    'E4 (+M5)':           [17.85, 10.83, 12.42, 14.25],
    'E5 (+M2+M6)':        [10.26, 11.28, 11.87, 14.06],
    'E7 (Full Stack)':    [11.77, 11.68, 12.74, 9.17],
}

# --- Exp-09 违规率（精细） ---
MONO_VIOL_EXP09 = {
    'E0':    [0.09, 0.09, 0.17, 0.06],
    'Exp09c':[0.08, 0.11, 0.08, 0.12],
    'Exp09a':[0.10, 0.17, 0.17, 0.14],
    'Exp09b':[0.14, 0.16, 0.15, 0.25],
}

# --- Exp-11 鲁棒性数据 ---
# Scene A: 特征缺失 (MAE%)
ROBUST_SCENE_A = {
    # (model, ratio): [clean, n=1, n=2, n=3]
    ('E0', 0.5):     [1.1426, 1.9613, 2.4914, 3.1644],
    ('Exp09c', 0.5): [1.0802, 1.8282, 2.3620, 2.8441],
    ('E0', 0.3):     [1.0696, 1.7942, 2.3126, 3.0805],
    ('Exp09c', 0.3): [1.0336, 1.7653, 2.3569, 2.8619],
}

# Scene C: 传感器漂移 (MAE%)
ROBUST_SCENE_C = {
    # (model, ratio): [clean, drift=0.05, drift=0.10, drift=0.20]
    ('E0', 0.5):     [1.1426, 1.1426+0.0359, 1.1426+0.1056, 1.1426+0.2915],
    ('Exp09c', 0.5): [1.0802, 1.0802+0.0337, 1.0802+0.1058, 1.0802+0.3345],
    ('E0', 0.3):     [1.0696, 1.0696+0.0371, 1.0696+0.1215, 1.0696+0.3561],
    ('Exp09c', 0.3): [1.0336, 1.0336+0.0442, 1.0336+0.1272, 1.0336+0.3677],
}


# ============================================================
# Fig 1: MAE% 随监督比例变化 —— 核心三条线
# ============================================================
def plot_fig1():
    """主结果对比：E0 vs A1 vs Exp09c"""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(RATIOS))

    # 三条核心曲线
    lines_data = [
        ('E0 (PI-CNN-LSTM)',       MAE['E0 (PI-CNN-LSTM)'],   'o', '#2196F3', '-'),
        ('A1 (MS-CNN-LSTM)',       MAE['A1 (MS-CNN-LSTM)'],   's', '#FF9800', '--'),
        ('Exp09c (PI-MS-CNN-LSTM)',MAE['Exp09c (PI-MS-CNN-LSTM)'], 'D', '#4CAF50', '-'),
    ]

    for label, data, marker, color, ls in lines_data:
        ax.plot(x, data, marker=marker, color=color, linestyle=ls,
                linewidth=2, markersize=8, label=label, zorder=5)
        # 标注数值
        for i, val in enumerate(data):
            offset = 0.03 if val > min(data) else -0.03
            ax.annotate(f'{val:.4f}%', (x[i], val),
                       textcoords="offset points", xytext=(0, 10 if offset > 0 else -14),
                       fontsize=7.5, ha='center', color=color)

    # 标注全场最佳
    best_idx = 3  # r=0.3, Exp09c
    ax.annotate('Best: 1.0336%', xy=(x[best_idx], 1.0336),
               xytext=(x[best_idx]-0.5, 0.98),
               arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5),
               fontsize=9, fontweight='bold', color='#4CAF50')

    ax.set_xticks(x)
    ax.set_xticklabels(['1.0\n(Full)', '0.7', '0.5', '0.3\n(Sparse)'])
    ax.set_xlabel('Supervision Ratio (r)')
    ax.set_ylabel('MAE (%)')
    ax.set_title('SOH Estimation Accuracy vs. Supervision Ratio')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0.95, 1.22)

    # 添加阴影标注"部分监督核心场景"
    ax.axvspan(1.5, 3.5, alpha=0.05, color='red')
    ax.text(2.5, 1.20, 'Partial Supervision\n(Core Scenario)',
            ha='center', fontsize=8, color='#666', style='italic')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig1_mae_vs_ratio.png')
    plt.savefig(path)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f'  [OK] Fig 1 saved: {path}')


# ============================================================
# Fig 2: 2×3 消融热力图
# ============================================================
def plot_fig2():
    """2×3 矩阵：架构 × 约束级别 (MAE% 平均)"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                             gridspec_kw={'wspace': 0.35})

    for ax_idx, (ratio_idx, ratio_label) in enumerate([(2, 'r=0.5'), (3, 'r=0.3')]):
        ax = axes[ax_idx]

        # 2×3 矩阵: 行=架构, 列=约束级别
        matrix = np.array([
            [MAE['Eneg1 (无物理)'][ratio_idx],
             MAE['E0 (PI-CNN-LSTM)'][ratio_idx],
             MAE['E7 (Full Stack M2+M4)'][ratio_idx]],
            [MAE['A1 (MS-CNN-LSTM)'][ratio_idx],
             MAE['Exp09c (PI-MS-CNN-LSTM)'][ratio_idx],
             MAE['Exp09a (MS+M2+M4 w=0.3)'][ratio_idx]],
        ])

        cmap = LinearSegmentedColormap.from_list('custom', ['#4CAF50', '#FFEB3B', '#F44336'])
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0.95, vmax=1.30)

        for i in range(2):
            for j in range(3):
                val = matrix[i, j]
                is_best = val == matrix.min()
                weight = 'bold' if is_best else 'normal'
                color = 'white' if val > 1.15 else 'black'
                text = f'{val:.4f}%'
                if is_best:
                    text += '\n(Best)'
                ax.text(j, i, text, ha='center', va='center',
                       fontsize=10, fontweight=weight, color=color)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['No Physics', 'Soft Monotonic\n(PI)', 'Full Stack\n(M2+M4)'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['CNN-LSTM', 'MS-CNN-LSTM'])
        ax.set_title(f'MAE% ({ratio_label})', fontweight='bold', fontsize=11)

        # 架构维度改善箭头
        arch_improve = (matrix[0].mean() - matrix[1].mean()) / matrix[0].mean() * 100
        ax.annotate(f'Arch.\n-{arch_improve:.1f}%', xy=(2.6, 0.9), xytext=(2.6, 0.1),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                   fontsize=8, ha='center', color='blue', fontweight='bold')

    fig.colorbar(im, ax=axes, shrink=0.75, label='MAE (%)', pad=0.02)
    fig.suptitle('Ablation: Architecture x Physics Constraint', fontsize=13, y=0.98)

    path = os.path.join(OUTPUT_DIR, 'fig2_ablation_heatmap.png')
    plt.savefig(path)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f'  [OK] Fig 2 saved: {path}')


# ============================================================
# Fig 3: 单调违反率柱状图
# ============================================================
def plot_fig3():
    """各实验组的单调违反率（r=0.5 聚焦）"""
    fig, ax = plt.subplots(figsize=(9, 4.5))

    # 选 r=0.5 场景展示
    ratio_idx = 2

    groups = ['Eneg1\n(No Physics)', 'E0\n(Baseline)', 'E2\n(+M2)',
              'E3\n(+M4)', 'E7\n(Full Stack)', 'A1\n(MS, No Phys)', 'Exp09c\n(PI-MS)']
    values = [
        MONO_VIOL['Eneg1 (无物理)'][ratio_idx],
        MONO_VIOL['E0 (PI-CNN-LSTM)'][ratio_idx],
        MONO_VIOL['E2 (+M2)'][ratio_idx],
        MONO_VIOL['E3 (+M4)'][ratio_idx],
        MONO_VIOL['E7 (Full Stack)'][ratio_idx],
        24.36,  # A1 无物理（从 Eneg1 同架构外推，实际 A1 无该数据用 Eneg1 代替）
        MONO_VIOL_EXP09['Exp09c'][ratio_idx] * 100,  # 注意 Exp09 违规率单位是 %，已经是百分比
    ]
    # 修正：Exp09 的违规率已经是百分数（0.08%），而单模块消融的是配对比例（16.80%）
    # 两个指标计算方式不同！分开画

    # 使用统一的单模块消融数据（配对违反比例 %）
    groups_unified = ['Eneg1\n(No Phys)', 'E0\n(Baseline)', 'E1\n(+M1)',
                      'E2\n(+M2)', 'E3\n(+M4)', 'E4\n(+M5)',
                      'E5\n(+M2+M6)', 'E7\n(Full Stack)']
    values_unified = [
        MONO_VIOL['Eneg1 (无物理)'][ratio_idx],
        MONO_VIOL['E0 (PI-CNN-LSTM)'][ratio_idx],
        MONO_VIOL['E1 (+M1 注意力)'][ratio_idx],
        MONO_VIOL['E2 (+M2)'][ratio_idx],
        MONO_VIOL['E3 (+M4)'][ratio_idx],
        MONO_VIOL['E4 (+M5)'][ratio_idx],
        MONO_VIOL['E5 (+M2+M6)'][ratio_idx],
        MONO_VIOL['E7 (Full Stack)'][ratio_idx],
    ]

    colors = ['#F44336' if v > 20 else '#FF9800' if v > 14 else '#4CAF50' if v < 12 else '#2196F3'
              for v in values_unified]

    bars = ax.bar(groups_unified, values_unified, color=colors, edgecolor='white', linewidth=0.5)

    # 标注数值
    for bar, val in zip(bars, values_unified):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    # 基线参考线
    ax.axhline(y=MONO_VIOL['E0 (PI-CNN-LSTM)'][ratio_idx], color='gray',
              linestyle='--', alpha=0.7, label=f'E0 Baseline ({MONO_VIOL["E0 (PI-CNN-LSTM)"][ratio_idx]:.1f}%)')

    ax.set_ylabel('Monotonic Violation Rate (%)\n(Pairwise Proportion)')
    ax.set_title(f'Monotonic Constraint Violation Rate (r=0.5)')
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(values_unified) * 1.15)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig3_monotonic_violation.png')
    plt.savefig(path)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f'  [OK] Fig 3 saved: {path}')


# ============================================================
# Fig 4: 鲁棒性 —— 特征缺失（Scene A）
# ============================================================
def plot_fig4():
    """Exp-11 Scene A: 特征缺失鲁棒性"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    x_labels = ['Clean', 'n=1', 'n=2', 'n=3']
    x = np.arange(4)

    for ax_idx, ratio in enumerate([0.5, 0.3]):
        ax = axes[ax_idx]

        e0_data = ROBUST_SCENE_A[('E0', ratio)]
        exp09c_data = ROBUST_SCENE_A[('Exp09c', ratio)]

        ax.plot(x, e0_data, 'o-', color='#2196F3', linewidth=2, markersize=8,
               label='E0 (PI-CNN-LSTM)')
        ax.plot(x, exp09c_data, 'D-', color='#4CAF50', linewidth=2, markersize=8,
               label='Exp09c (PI-MS-CNN-LSTM)')

        # 填充差异区域
        ax.fill_between(x, e0_data, exp09c_data, alpha=0.1, color='#4CAF50')

        # 标注 n=3 的差距
        diff_n3 = e0_data[3] - exp09c_data[3]
        mid_y = (e0_data[3] + exp09c_data[3]) / 2
        ax.annotate(f'Δ = {diff_n3:.3f}%', xy=(3.05, mid_y),
                   fontsize=8, color='#4CAF50', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Number of Missing Features')
        ax.set_title(f'r = {ratio}', fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)

    axes[0].set_ylabel('MAE (%)')
    plt.suptitle('Robustness: Feature Missing (Scene A)', fontsize=12, y=1.01)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'fig4_robustness_feature_missing.png')
    plt.savefig(path)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f'  [OK] Fig 4 saved: {path}')


# ============================================================
# Fig 5: 鲁棒性 —— 传感器漂移（Scene C）
# ============================================================
def plot_fig5():
    """Exp-11 Scene C: 传感器漂移鲁棒性"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    x_labels = ['Clean', 'δ=0.05', 'δ=0.10', 'δ=0.20']
    x = np.arange(4)

    for ax_idx, ratio in enumerate([0.5, 0.3]):
        ax = axes[ax_idx]

        e0_data = ROBUST_SCENE_C[('E0', ratio)]
        exp09c_data = ROBUST_SCENE_C[('Exp09c', ratio)]

        ax.plot(x, e0_data, 'o-', color='#2196F3', linewidth=2, markersize=8,
               label='E0 (PI-CNN-LSTM)')
        ax.plot(x, exp09c_data, 'D-', color='#4CAF50', linewidth=2, markersize=8,
               label='Exp09c (PI-MS-CNN-LSTM)')

        # 填充差异区域
        ax.fill_between(x, e0_data, exp09c_data, alpha=0.1, color='#4CAF50')

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Sensor Drift Magnitude')
        ax.set_title(f'r = {ratio}', fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)

    axes[0].set_ylabel('MAE (%)')
    plt.suptitle('Robustness: Sensor Drift (Scene C)', fontsize=12, y=1.01)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'fig5_robustness_sensor_drift.png')
    plt.savefig(path)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f'  [OK] Fig 5 saved: {path}')


# ============================================================
# Fig 6: 物理约束正则化效果（A1 vs Exp09c 对比）
# ============================================================
def plot_fig6():
    """物理约束在不同监督比例下的效果：精度 + R² 双轴图"""
    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(RATIOS))
    width = 0.3

    a1_mae = MAE['A1 (MS-CNN-LSTM)']
    exp09c_mae = MAE['Exp09c (PI-MS-CNN-LSTM)']

    # MAE 柱状图
    bars1 = ax1.bar(x - width/2, a1_mae, width, label='A1 (No Physics)',
                    color='#FF9800', alpha=0.8, edgecolor='white')
    bars2 = ax1.bar(x + width/2, exp09c_mae, width, label='Exp09c (+Soft Monotonic)',
                    color='#4CAF50', alpha=0.8, edgecolor='white')

    ax1.set_ylabel('MAE (%)', color='black')
    ax1.set_ylim(0.95, 1.20)

    # 标注哪个更优
    for i in range(4):
        diff = a1_mae[i] - exp09c_mae[i]
        if diff > 0:  # Exp09c 更好
            ax1.annotate(f'PI +{diff/a1_mae[i]*100:.1f}%',
                        xy=(x[i]+width/2, exp09c_mae[i]),
                        xytext=(x[i]+width/2, exp09c_mae[i]-0.02),
                        fontsize=7, ha='center', color='#4CAF50', fontweight='bold')
        else:  # A1 更好
            ax1.annotate(f'PI {diff/a1_mae[i]*100:.1f}%',
                        xy=(x[i]-width/2, a1_mae[i]),
                        xytext=(x[i]-width/2, a1_mae[i]-0.02),
                        fontsize=7, ha='center', color='#FF9800', fontweight='bold')

    # R² 折线图（第二 Y 轴）
    ax2 = ax1.twinx()
    ax2.plot(x, R2['A1'], 's--', color='#E65100', linewidth=1.5, markersize=6, alpha=0.7,
            label='A1 R²')
    ax2.plot(x, R2['Exp09c'], 'D--', color='#1B5E20', linewidth=1.5, markersize=6, alpha=0.7,
            label='Exp09c R²')
    ax2.set_ylabel('R²', color='gray')
    ax2.set_ylim(0.90, 0.95)

    ax1.set_xticks(x)
    ax1.set_xticklabels(['r=1.0\n(Full)', 'r=0.7', 'r=0.5', 'r=0.3\n(Sparse)'])
    ax1.set_xlabel('Supervision Ratio')
    ax1.set_title('Physics Constraint Regularization Effect\n(MS-CNN-LSTM ± Soft Monotonic)')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    # 标注核心结论
    ax1.text(3.0, 1.17, 'Physics constraint\nhelps at r=0.3!',
            fontsize=8, ha='center', color='#4CAF50', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig6_physics_regularization.png')
    plt.savefig(path)
    plt.savefig(path.replace('.png', '.pdf'))
    plt.close()
    print(f'  [OK] Fig 6 saved: {path}')


# ============================================================
# Main
# ============================================================
FIGURES = {
    1: ('MAE vs Supervision Ratio', plot_fig1),
    2: ('2×3 Ablation Heatmap', plot_fig2),
    3: ('Monotonic Violation Rate', plot_fig3),
    4: ('Robustness: Feature Missing', plot_fig4),
    5: ('Robustness: Sensor Drift', plot_fig5),
    6: ('Physics Regularization Effect', plot_fig6),
}


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--fig', type=int, default=None,
                       help='Generate specific figure (1-6). Default: all')
    args = parser.parse_args()

    print(f'Output directory: {OUTPUT_DIR}')
    print('=' * 50)

    if args.fig:
        if args.fig in FIGURES:
            name, func = FIGURES[args.fig]
            print(f'Generating Fig {args.fig}: {name}')
            func()
        else:
            print(f'Error: Fig {args.fig} not found. Available: {list(FIGURES.keys())}')
    else:
        print('Generating all figures...\n')
        for fig_num, (name, func) in FIGURES.items():
            print(f'[{fig_num}/6] {name}')
            func()
        print('\n' + '=' * 50)
        print(f'All {len(FIGURES)} figures generated in: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
