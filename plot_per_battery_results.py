"""
Per-battery结果可视化脚本（论文Fig. 4风格）。

为每个电池绘制：
- SOH预测曲线（Cycle vs SOH）
- 对比BPINN-1, BPINN-2和Real值
"""

import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('src')


def plot_soh_curves_per_battery(all_results, save_path='results_per_battery/soh_comparison.png'):
    """
    绘制每个电池的SOH预测曲线（论文Fig. 4风格）。

    Args:
        all_results: main_per_battery.py返回的结果字典
        save_path: 保存路径
    """
    n_batteries = len(all_results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (battery_name, results) in enumerate(all_results.items()):
        ax = axes[idx]

        # 获取数据
        test_cycles = results['test_cycles']
        real_soh = results['results_phase1']['targets']  # Real SOH
        pred_bpinn1 = results['results_phase1']['predictions']
        pred_bpinn2 = results['results_phase2']['predictions']

        # 绘制曲线
        ax.plot(test_cycles, real_soh, 'k-', linewidth=2.5,
                label='Real', marker='o', markersize=4,
                markevery=max(1, len(test_cycles)//10), alpha=0.8)

        ax.plot(test_cycles, pred_bpinn1, '-', color='#1f77b4', linewidth=2,
                label='BPINN-1', marker='s', markersize=4,
                markevery=max(1, len(test_cycles)//10), alpha=0.8)

        ax.plot(test_cycles, pred_bpinn2, '-', color='#ff7f0e', linewidth=2,
                label='BPINN-2', marker='^', markersize=4,
                markevery=max(1, len(test_cycles)//10), alpha=0.8)

        # 格式化
        ax.set_xlabel('Cycle', fontsize=11, fontweight='bold')
        ax.set_ylabel('SOH', fontsize=11, fontweight='bold')
        ax.set_title(f'{battery_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Y轴格式化为百分比
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

        # 添加性能指标文本
        mae1 = results['results_phase1']['mae'] * 100
        rmse1 = results['results_phase1']['rmse'] * 100
        mae2 = results['results_phase2']['mae'] * 100
        rmse2 = results['results_phase2']['rmse'] * 100

        textstr = f'BPINN-1: MAE={mae1:.2f}%, RMSE={rmse1:.2f}%\n'
        textstr += f'BPINN-2: MAE={mae2:.2f}%, RMSE={rmse2:.2f}%'

        ax.text(0.02, 0.02, textstr, transform=ax.transAxes,
                fontsize=8, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # 如果电池少于4个，隐藏多余的子图
    for idx in range(n_batteries, 4):
        axes[idx].axis('off')

    plt.suptitle('SOH Prediction Comparison (Per-Battery Training)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SOH curves saved to: {save_path}")

    plt.show()


def plot_error_comparison(all_results, save_path='results_per_battery/error_comparison.png'):
    """
    绘制各电池的误差对比条形图。

    Args:
        all_results: 结果字典
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    batteries = list(all_results.keys())
    mae1_values = [all_results[b]['results_phase1']['mae'] * 100 for b in batteries]
    mae2_values = [all_results[b]['results_phase2']['mae'] * 100 for b in batteries]
    rmse1_values = [all_results[b]['results_phase1']['rmse'] * 100 for b in batteries]
    rmse2_values = [all_results[b]['results_phase2']['rmse'] * 100 for b in batteries]

    x = np.arange(len(batteries))
    width = 0.35

    # MAE对比
    ax1.bar(x - width/2, mae1_values, width, label='BPINN-1',
            alpha=0.8, color='#1f77b4')
    ax1.bar(x + width/2, mae2_values, width, label='BPINN-2',
            alpha=0.8, color='#ff7f0e')
    ax1.set_xlabel('Battery', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MAE (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Absolute Error Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(batteries)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # RMSE对比
    ax2.bar(x - width/2, rmse1_values, width, label='BPINN-1',
            alpha=0.8, color='#1f77b4')
    ax2.bar(x + width/2, rmse2_values, width, label='BPINN-2',
            alpha=0.8, color='#ff7f0e')
    ax2.set_xlabel('Battery', fontsize=12, fontweight='bold')
    ax2.set_ylabel('RMSE (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Root Mean Square Error Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(batteries)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error comparison saved to: {save_path}")

    plt.show()


def main():
    """主函数：加载结果并绘图。"""
    results_file = 'results_per_battery/all_results.pkl'

    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        print("Please run main_per_battery.py first to generate results.")
        return

    # 加载结果
    print(f"Loading results from: {results_file}")
    with open(results_file, 'rb') as f:
        all_results = pickle.load(f)

    print(f"Found results for {len(all_results)} batteries")

    # 绘制SOH曲线
    print("\nGenerating SOH prediction curves...")
    plot_soh_curves_per_battery(all_results)

    # 绘制误差对比
    print("\nGenerating error comparison...")
    plot_error_comparison(all_results)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
