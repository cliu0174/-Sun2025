"""
可视化对比结果：BPINN vs FNN vs CNN vs LSTM

生成对比图表。
"""

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_comparison_bar_chart(all_results, battery_name, save_path=None):
    """
    绘制模型对比条形图。

    Args:
        all_results: 所有模型的结果字典
        battery_name: 电池名称
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    models = list(all_results.keys())
    mae_values = [all_results[m]['mae'] * 100 for m in models]
    rmse_values = [all_results[m]['rmse'] * 100 for m in models]

    x = np.arange(len(models))
    width = 0.6

    # Define colors
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    # MAE comparison
    bars1 = ax1.bar(x, mae_values, width, color=colors[:len(models)], alpha=0.8)
    ax1.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax1.set_ylabel('MAE (%)', fontsize=13, fontweight='bold')
    ax1.set_title(f'{battery_name} - Mean Absolute Error', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, mae_values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                f'{val:.3f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # RMSE comparison
    bars2 = ax2.bar(x, rmse_values, width, color=colors[:len(models)], alpha=0.8)
    ax2.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax2.set_ylabel('RMSE (%)', fontsize=13, fontweight='bold')
    ax2.set_title(f'{battery_name} - Root Mean Square Error', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, rmse_values)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                f'{val:.3f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison chart saved to: {save_path}")

    plt.show()


def plot_comparison_table(all_results, battery_name, save_path=None):
    """
    绘制模型对比表格。

    Args:
        all_results: 所有模型的结果字典
        battery_name: 电池名称
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')

    # Prepare data
    models = list(all_results.keys())
    mae_values = [f"{all_results[m]['mae'] * 100:.4f}%" for m in models]
    rmse_values = [f"{all_results[m]['rmse'] * 100:.4f}%" for m in models]

    # Create table
    table_data = []
    table_data.append(['Model', 'MAE', 'RMSE'])
    for i, model in enumerate(models):
        table_data.append([model, mae_values[i], rmse_values[i]])

    # Highlight best values
    best_mae_idx = np.argmin([all_results[m]['mae'] for m in models]) + 1
    best_rmse_idx = np.argmin([all_results[m]['rmse'] for m in models]) + 1

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.3, 0.3])

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)

    # Style header
    for i in range(3):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white', fontsize=14)

    # Highlight best cells
    table[(best_mae_idx, 1)].set_facecolor('#2ecc71')
    table[(best_mae_idx, 1)].set_text_props(weight='bold')
    table[(best_rmse_idx, 2)].set_facecolor('#2ecc71')
    table[(best_rmse_idx, 2)].set_text_props(weight='bold')

    plt.title(f'{battery_name} - Model Comparison', fontsize=16, fontweight='bold', pad=20)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison table saved to: {save_path}")

    plt.show()


def main():
    """主函数：加载并可视化对比结果。"""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize comparison results')
    parser.add_argument('--battery', type=str, default='B05',
                        choices=['B05', 'B06', 'B07'],
                        help='Battery to visualize (default: B05)')
    args = parser.parse_args()

    results_file = f'results_comparison/{args.battery}/comparison_results.pkl'

    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        print(f"Please run: python main_comparison.py --battery {args.battery}")
        return

    # Load results
    print(f"Loading results from: {results_file}")
    with open(results_file, 'rb') as f:
        all_results = pickle.load(f)

    print(f"Found results for {len(all_results)} models")

    # Print summary
    print(f"\n{'Model':<12} {'MAE (%)':<12} {'RMSE (%)':<12}")
    print("-" * 36)
    for model_name, results in all_results.items():
        mae = results['mae'] * 100
        rmse = results['rmse'] * 100
        print(f"{model_name:<12} {mae:<12.4f} {rmse:<12.4f}")

    # Plot bar chart
    print("\nGenerating comparison bar chart...")
    save_path_bar = f'results_comparison/{args.battery}/comparison_bar_chart.png'
    plot_comparison_bar_chart(all_results, args.battery, save_path=save_path_bar)

    # Plot table
    print("\nGenerating comparison table...")
    save_path_table = f'results_comparison/{args.battery}/comparison_table.png'
    plot_comparison_table(all_results, args.battery, save_path=save_path_table)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
