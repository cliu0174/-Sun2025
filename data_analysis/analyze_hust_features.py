"""
HUST数据集特征分析脚本

分析16个特征与容量(SOH)的相关性，包括:
- 皮尔逊相关系数
- 斯皮尔曼相关系数（单调性）
- 可视化散点图
- 特征重要性排序
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data_loaders import load_single_hust_battery


def analyze_feature_correlations(battery_file, save_dir='results/feature_analysis'):
    """
    分析单个电池的特征相关性。

    Args:
        battery_file: 电池CSV文件路径
        save_dir: 结果保存目录

    Returns:
        相关性分析结果字典
    """
    # 加载数据
    data_dict = load_single_hust_battery(battery_file, train_ratio=1.0, normalize_target=True)
    battery_name = data_dict['battery_name']

    print(f"\n{'='*70}")
    print(f"分析电池: {battery_name}")
    print(f"{'='*70}")
    print(f"总样本数: {len(data_dict['train_capacity'])}")
    print(f"SOH范围: {data_dict['train_capacity'].min():.4f} - {data_dict['train_capacity'].max():.4f}")

    # 创建DataFrame
    feature_names = data_dict['feature_names']
    features = data_dict['train_features']
    capacity = data_dict['train_capacity']

    df = pd.DataFrame(features, columns=feature_names)
    df['SOH'] = capacity

    # 1. 计算皮尔逊相关系数（线性相关）
    print(f"\n{'='*70}")
    print("1. 皮尔逊相关系数分析 (线性相关)")
    print(f"{'='*70}")

    pearson_corr = []
    for feature in feature_names:
        corr_result = stats.pearsonr(df[feature], df['SOH'])
        corr = float(corr_result[0])
        p_value = float(corr_result[1])
        pearson_corr.append({
            'feature': feature,
            'correlation': corr,
            'abs_correlation': abs(corr),
            'p_value': p_value,
            'significant': 'Yes' if p_value < 0.001 else 'No'
        })

    pearson_df = pd.DataFrame(pearson_corr).sort_values('abs_correlation', ascending=False)

    print(f"\n{'特征':<25} {'相关系数':<12} {'绝对值':<12} {'显著性':<10} {'关系':<10}")
    print("-" * 70)
    for _, row in pearson_df.iterrows():
        direction = "正相关" if row['correlation'] > 0 else "负相关"
        print(f"{row['feature']:<25} {row['correlation']:<12.6f} {row['abs_correlation']:<12.6f} "
              f"{row['significant']:<10} {direction:<10}")

    # 2. 计算斯皮尔曼相关系数（单调性）
    print(f"\n{'='*70}")
    print("2. 斯皮尔曼相关系数分析 (单调相关)")
    print(f"{'='*70}")

    spearman_corr = []
    for feature in feature_names:
        corr_result = stats.spearmanr(df[feature], df['SOH'])
        corr = float(corr_result[0])
        p_value = float(corr_result[1])
        spearman_corr.append({
            'feature': feature,
            'correlation': corr,
            'abs_correlation': abs(corr),
            'p_value': p_value,
            'significant': 'Yes' if p_value < 0.001 else 'No'
        })

    spearman_df = pd.DataFrame(spearman_corr).sort_values('abs_correlation', ascending=False)

    print(f"\n{'特征':<25} {'相关系数':<12} {'绝对值':<12} {'显著性':<10} {'关系':<10}")
    print("-" * 70)
    for _, row in spearman_df.iterrows():
        direction = "正相关 ↗" if row['correlation'] > 0 else "负相关 ↘"
        print(f"{row['feature']:<25} {row['correlation']:<12.6f} {row['abs_correlation']:<12.6f} "
              f"{row['significant']:<10} {direction:<10}")

    # 3. Top-10特征详细分析
    print(f"\n{'='*70}")
    print("3. Top-10 最强相关特征 (按皮尔逊相关系数)")
    print(f"{'='*70}")

    top_features = pearson_df.head(10)

    print(f"\n{'排名':<6} {'特征':<25} {'皮尔逊':<12} {'斯皮尔曼':<12} {'物理约束潜力':<15}")
    print("-" * 70)

    for idx, (_, row) in enumerate(top_features.iterrows(), 1):
        feature = row['feature']
        pearson_val = row['correlation']
        spearman_val = spearman_df[spearman_df['feature'] == feature]['correlation'].values[0]

        # 判断物理约束潜力
        if abs(pearson_val) > 0.8 and abs(spearman_val) > 0.8:
            potential = "*** Strong"
        elif abs(pearson_val) > 0.6 and abs(spearman_val) > 0.6:
            potential = "** Medium"
        else:
            potential = "* Weak"

        print(f"{idx:<6} {feature:<25} {pearson_val:<12.6f} {spearman_val:<12.6f} {potential:<15}")

    # 4. 创建可视化
    os.makedirs(save_dir, exist_ok=True)

    # 4.1 相关系数对比图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 皮尔逊相关系数
    ax1 = axes[0]
    colors = ['green' if x > 0 else 'red' for x in pearson_df['correlation']]
    ax1.barh(range(len(pearson_df)), pearson_df['correlation'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(pearson_df)))
    ax1.set_yticklabels(pearson_df['feature'], fontsize=9)
    ax1.set_xlabel('Pearson Correlation Coefficient', fontsize=11)
    ax1.set_title(f'Pearson Correlation with SOH\n({battery_name})', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.grid(axis='x', alpha=0.3)

    # 斯皮尔曼相关系数
    ax2 = axes[1]
    colors = ['green' if x > 0 else 'red' for x in spearman_df['correlation']]
    ax2.barh(range(len(spearman_df)), spearman_df['correlation'], color=colors, alpha=0.7)
    ax2.set_yticks(range(len(spearman_df)))
    ax2.set_yticklabels(spearman_df['feature'], fontsize=9)
    ax2.set_xlabel('Spearman Correlation Coefficient', fontsize=11)
    ax2.set_title(f'Spearman Correlation with SOH\n({battery_name})', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{battery_name}_correlation_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved correlation comparison: {save_dir}/{battery_name}_correlation_comparison.png")
    plt.close()

    # 4.2 Top-6特征散点图
    top6_features = pearson_df.head(6)['feature'].values

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(top6_features):
        ax = axes[idx]

        # 原始数据散点图
        ax.scatter(df[feature], df['SOH'], alpha=0.5, s=10, color='blue', label='Data points')

        # 拟合线（线性回归）
        z = np.polyfit(df[feature], df['SOH'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[feature].min(), df[feature].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Linear fit')

        # 相关系数
        corr = pearson_df[pearson_df['feature'] == feature]['correlation'].values[0]

        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('SOH', fontsize=10)
        ax.set_title(f'{feature}\nPearson: {corr:.4f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle(f'Top-6 Feature vs SOH Scatter Plots ({battery_name})',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{battery_name}_top6_scatter.png'), dpi=150, bbox_inches='tight')
    print(f"[OK] Saved Top-6 scatter plots: {save_dir}/{battery_name}_top6_scatter.png")
    plt.close()

    # 4.3 相关性矩阵热力图（包含SOH）
    # 只选择top-10特征以保持可读性
    top10_features = pearson_df.head(10)['feature'].values
    corr_matrix = df[list(top10_features) + ['SOH']].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f'Correlation Matrix (Top-10 Features + SOH)\n({battery_name})',
              fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{battery_name}_correlation_matrix.png'), dpi=150, bbox_inches='tight')
    print(f"[OK] Saved correlation matrix: {save_dir}/{battery_name}_correlation_matrix.png")
    plt.close()

    # 4.4 SOH随时间变化 + Top-3特征变化
    top3_features = pearson_df.head(3)['feature'].values

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # SOH变化
    ax0 = axes[0]
    ax0.plot(df['SOH'], linewidth=2, color='black', label='SOH')
    ax0.set_ylabel('SOH', fontsize=11, fontweight='bold')
    ax0.set_title(f'SOH and Top-3 Features over Cycles ({battery_name})',
                  fontsize=13, fontweight='bold')
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=10)

    # Top-3特征变化
    for idx, feature in enumerate(top3_features):
        ax = axes[idx + 1]
        corr = pearson_df[pearson_df['feature'] == feature]['correlation'].values[0]

        # 归一化特征值到[0, 1]
        normalized = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())

        ax.plot(normalized, linewidth=2, label=f'{feature} (normalized)', color=f'C{idx}')
        ax.set_ylabel(f'{feature}\n(normalized)', fontsize=10)
        ax.set_xlabel('Cycle' if idx == 2 else '', fontsize=10)
        ax.grid(True, alpha=0.3)

        # 显示相关系数
        direction = "↗" if corr > 0 else "↘"
        ax.text(0.02, 0.95, f'Corr: {corr:.4f} {direction}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{battery_name}_time_series.png'), dpi=150, bbox_inches='tight')
    print(f"[OK] Saved time series plot: {save_dir}/{battery_name}_time_series.png")
    plt.close()

    # 4.5 单行相关系数热力图（类似您提供的图片）
    fig, ax = plt.subplots(figsize=(18, 2))

    # 按原始顺序排列相关系数
    correlations = []
    for feature in feature_names:
        corr = pearson_df[pearson_df['feature'] == feature]['correlation'].values[0]
        correlations.append(corr)

    correlations_array = np.array(correlations).reshape(1, -1)

    # 创建热力图
    im = ax.imshow(correlations_array, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

    # 设置刻度和标签
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels([str(i) for i in range(1, len(feature_names) + 1)], fontsize=10)
    ax.set_yticks([0])
    ax.set_yticklabels(['HUST'], fontsize=12, fontweight='bold')

    # 在每个格子中显示相关系数值
    for i in range(len(feature_names)):
        text_color = 'white' if abs(correlations[i]) > 0.5 else 'black'
        ax.text(i, 0, f'{correlations[i]:.2f}',
               ha="center", va="center", color=text_color, fontsize=9, fontweight='bold')

    # 添加特征名称作为x轴标签（在底部）
    ax.set_xlabel('Feature Index', fontsize=11, fontweight='bold')
    ax.set_title(f'Pearson Correlation with SOH - Battery {battery_name}',
                 fontsize=13, fontweight='bold', pad=15)

    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.15, aspect=30)
    cbar.set_label('Correlation Coefficient', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{battery_name}_correlation_heatmap.png'),
                dpi=200, bbox_inches='tight')
    print(f"[OK] Saved correlation heatmap: {save_dir}/{battery_name}_correlation_heatmap.png")
    plt.close()

    # 4.6 带特征名称的详细版本
    fig, ax = plt.subplots(figsize=(20, 3))

    # 按相关系数绝对值排序
    sorted_indices = pearson_df['feature'].tolist()
    sorted_correlations = []
    for feature in sorted_indices:
        corr = pearson_df[pearson_df['feature'] == feature]['correlation'].values[0]
        sorted_correlations.append(corr)

    sorted_array = np.array(sorted_correlations).reshape(1, -1)

    # 创建热力图
    im = ax.imshow(sorted_array, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

    # 设置刻度和标签
    ax.set_xticks(np.arange(len(sorted_indices)))
    ax.set_xticklabels(sorted_indices, rotation=45, ha='right', fontsize=9)
    ax.set_yticks([0])
    ax.set_yticklabels(['HUST'], fontsize=12, fontweight='bold')

    # 在每个格子中显示相关系数值
    for i in range(len(sorted_indices)):
        text_color = 'white' if abs(sorted_correlations[i]) > 0.5 else 'black'
        ax.text(i, 0, f'{sorted_correlations[i]:.3f}',
               ha="center", va="center", color=text_color, fontsize=8, fontweight='bold')

    ax.set_title(f'Pearson Correlation with SOH (Sorted by Absolute Value) - Battery {battery_name}',
                 fontsize=13, fontweight='bold', pad=15)

    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.2, aspect=40)
    cbar.set_label('Correlation Coefficient', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{battery_name}_correlation_heatmap_detailed.png'),
                dpi=200, bbox_inches='tight')
    print(f"[OK] Saved detailed correlation heatmap: {save_dir}/{battery_name}_correlation_heatmap_detailed.png")
    plt.close()

    # 5. 生成Markdown报告
    report_path = os.path.join(save_dir, f'{battery_name}_analysis_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# HUST电池特征分析报告\n\n")
        f.write(f"**电池**: {battery_name}\n\n")
        f.write(f"**样本数**: {len(df)}\n\n")
        f.write(f"**SOH范围**: {df['SOH'].min():.4f} - {df['SOH'].max():.4f}\n\n")
        f.write(f"---\n\n")

        f.write(f"## 📊 皮尔逊相关系数排名 (Top-10)\n\n")
        f.write(f"| 排名 | 特征 | 相关系数 | 绝对值 | 关系 | 物理约束潜力 |\n")
        f.write(f"|------|------|----------|--------|------|-------------|\n")

        for idx, (_, row) in enumerate(pearson_df.head(10).iterrows(), 1):
            feature = row['feature']
            corr = row['correlation']
            abs_corr = row['abs_correlation']
            direction = "正相关 ↗" if corr > 0 else "负相关 ↘"

            spearman_val = spearman_df[spearman_df['feature'] == feature]['correlation'].values[0]
            if abs(corr) > 0.8 and abs(spearman_val) > 0.8:
                potential = "*** Strong"
            elif abs(corr) > 0.6 and abs(spearman_val) > 0.6:
                potential = "** Medium"
            else:
                potential = "* Weak"

            f.write(f"| {idx} | **{feature}** | {corr:.6f} | {abs_corr:.6f} | {direction} | {potential} |\n")

        f.write(f"\n---\n\n")
        f.write(f"## 🎯 BPINN物理约束建议\n\n")
        f.write(f"基于相关性分析，以下特征适合作为物理约束:\n\n")

        strong_features = pearson_df[pearson_df['abs_correlation'] > 0.7].head(5)
        for idx, (_, row) in enumerate(strong_features.iterrows(), 1):
            feature = row['feature']
            corr = row['correlation']
            constraint = f"∂SOH/∂{feature} > 0" if corr > 0 else f"∂SOH/∂{feature} < 0"
            f.write(f"{idx}. **{feature}** (相关性: {corr:.4f})\n")
            f.write(f"   - 约束: `{constraint}` (单调性)\n\n")

        f.write(f"\n---\n\n")
        f.write(f"## 📈 可视化图表\n\n")
        f.write(f"1. [相关系数对比图]({battery_name}_correlation_comparison.png)\n")
        f.write(f"2. [Top-6散点图]({battery_name}_top6_scatter.png)\n")
        f.write(f"3. [相关性矩阵]({battery_name}_correlation_matrix.png)\n")
        f.write(f"4. [时间序列图]({battery_name}_time_series.png)\n")

    print(f"[OK] Saved analysis report: {report_path}")

    # 返回结果
    results = {
        'battery_name': battery_name,
        'pearson': pearson_df,
        'spearman': spearman_df,
        'dataframe': df
    }

    return results


def compare_multiple_batteries(battery_files, save_dir='results/feature_analysis'):
    """
    对比多个电池的特征相关性。

    Args:
        battery_files: 电池文件路径列表
        save_dir: 结果保存目录
    """
    all_results = []

    for battery_file in battery_files:
        if os.path.exists(battery_file):
            results = analyze_feature_correlations(battery_file, save_dir)
            all_results.append(results)

    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("跨电池对比分析")
        print(f"{'='*70}")

        # 汇总Top-5特征在各电池上的相关性
        feature_importance = {}
        for results in all_results:
            for _, row in results['pearson'].head(5).iterrows():
                feature = row['feature']
                if feature not in feature_importance:
                    feature_importance[feature] = []
                feature_importance[feature].append(row['abs_correlation'])

        print(f"\n{'特征':<25} {'平均绝对相关':<15} {'出现次数':<10}")
        print("-" * 50)

        sorted_features = sorted(feature_importance.items(),
                                key=lambda x: np.mean(x[1]), reverse=True)
        for feature, correlations in sorted_features:
            avg_corr = np.mean(correlations)
            count = len(correlations)
            print(f"{feature:<25} {avg_corr:<15.6f} {count:<10}")

    return all_results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze HUST battery feature correlations')
    parser.add_argument('--battery', type=str, default='1-1',
                        help='Battery to analyze (e.g., 1-1, 1-2, ..., 10-8)')
    parser.add_argument('--multiple', action='store_true',
                        help='Analyze multiple batteries (1-1, 2-1, 3-1)')
    args = parser.parse_args()

    data_dir = 'data/HUST data'
    save_dir = 'results/feature_analysis'

    if args.multiple:
        # 分析多个电池
        battery_files = [
            os.path.join(data_dir, '1-1.csv'),
            os.path.join(data_dir, '2-1.csv'),
            os.path.join(data_dir, '3-1.csv')
        ]
        compare_multiple_batteries(battery_files, save_dir)
    else:
        # 分析单个电池
        battery_file = os.path.join(data_dir, f'{args.battery}.csv')
        if not os.path.exists(battery_file):
            print(f"错误: 找不到文件 {battery_file}")
            return

        analyze_feature_correlations(battery_file, save_dir)

    print(f"\n{'='*70}")
    print("分析完成！")
    print(f"{'='*70}")
    print(f"结果保存在: {save_dir}/")
    print("\n查看可视化图表和Markdown报告以获取详细分析。")


if __name__ == "__main__":
    main()
