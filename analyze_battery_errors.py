"""
分析并定位在多个模型上表现异常的电池
识别哪些电池在不同模型上都有较高的预测误差
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


def load_model_results(model_name):
    """加载单个模型的预测结果"""
    results_path = f'results/cross_battery/{model_name}/results.pkl'

    try:
        with open(results_path, 'rb') as f:
            results = pickle.load(f)

        predictions = np.array(results['predictions']).flatten()
        targets = np.array(results['targets']).flatten()
        battery_ids = results.get('battery_ids', None)

        return predictions, targets, battery_ids
    except Exception as e:
        print(f"无法加载 {model_name}: {e}")
        return None, None, None


def analyze_battery_errors(models=['cnn_lstm', 'gru', 'bigru']):
    """分析各电池在不同模型上的误差"""

    print("="*80)
    print("电池预测误差分析")
    print("="*80)

    # 存储所有模型的结果
    all_results = {}

    # 加载所有模型的结果
    for model in models:
        predictions, targets, battery_ids = load_model_results(model)

        if predictions is not None and battery_ids is not None:
            all_results[model] = {
                'predictions': predictions,
                'targets': targets,
                'battery_ids': battery_ids
            }
            print(f"[OK] 加载 {model.upper()}: {len(predictions)} 个样本")

    if not all_results:
        print("未找到任何模型结果！")
        return

    print()

    # 分析每个模型在各电池上的误差
    battery_errors = {}  # {battery_id: {model: mae}}

    for model, data in all_results.items():
        predictions = data['predictions']
        targets = data['targets']
        battery_ids = data['battery_ids']

        # 计算误差
        errors = np.abs(predictions - targets)

        # 按电池分组
        unique_batteries = np.unique(battery_ids)

        for battery in unique_batteries:
            if battery not in battery_errors:
                battery_errors[battery] = {}

            mask = battery_ids == battery
            battery_mae = np.mean(errors[mask])
            battery_errors[battery][model] = battery_mae

    # 转换为DataFrame便于分析
    df = pd.DataFrame(battery_errors).T
    df.index.name = 'Battery'

    # 计算每个电池的平均误差和标准差
    df['Mean_MAE'] = df.mean(axis=1)
    df['Std_MAE'] = df.std(axis=1)
    df['Max_MAE'] = df.max(axis=1)
    df['Min_MAE'] = df.min(axis=1)

    # 按平均误差排序
    df_sorted = df.sort_values('Mean_MAE', ascending=False)

    # 打印结果表格
    print("\n" + "="*80)
    print("各电池在不同模型上的MAE统计")
    print("="*80)
    print(df_sorted.to_string())

    # 识别异常电池（取前2个误差最大的）
    overall_mean = df['Mean_MAE'].mean()
    overall_std = df['Mean_MAE'].std()
    threshold = overall_mean + 1.5 * overall_std

    # 只取前2个最异常的电池
    problematic_batteries = df_sorted.head(2).index.tolist()

    print("\n" + "="*80)
    print("异常电池识别")
    print("="*80)
    print(f"整体平均MAE: {overall_mean:.6f}")
    print(f"标准差: {overall_std:.6f}")
    print(f"异常阈值: {threshold:.6f} (平均 + 1.5×标准差)")
    print(f"\n异常电池数量: {len(problematic_batteries)} / {len(df)}")

    if problematic_batteries:
        print("\n异常电池列表:")
        for battery in problematic_batteries:
            mean_mae = df.loc[battery, 'Mean_MAE']
            max_mae = df.loc[battery, 'Max_MAE']
            print(f"  {battery}: 平均MAE={mean_mae:.6f}, 最大MAE={max_mae:.6f}")
    else:
        print("\n未发现明显异常的电池！")

    # 可视化：拆分为4个独立图表
    print("\n生成可视化...")

    model_cols = [col for col in df.columns if col not in ['Mean_MAE', 'Std_MAE', 'Max_MAE', 'Min_MAE']]

    # 图1: 热力图
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_sorted[model_cols], annot=True, fmt='.4f', cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'MAE'})
    ax1.set_title('Battery MAE Heatmap Across Models', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Battery', fontsize=12)
    plt.tight_layout()
    plt.savefig('battery_error_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Heatmap saved: battery_error_heatmap.png")
    plt.close()

    # 图2: 条形图
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    df_sorted['Mean_MAE'].plot(kind='barh', ax=ax2, color='steelblue')
    ax2.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    ax2.set_title('Average MAE by Battery', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Mean MAE', fontsize=12)
    ax2.set_ylabel('Battery', fontsize=12)
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('battery_error_barplot.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Bar plot saved: battery_error_barplot.png")
    plt.close()

    # 图3: 箱线图
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    df[model_cols].boxplot(ax=ax3, rot=45)
    ax3.set_title('MAE Distribution Across Models', fontsize=14, fontweight='bold')
    ax3.set_ylabel('MAE', fontsize=12)
    ax3.set_xlabel('Model', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('battery_error_boxplot.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Box plot saved: battery_error_boxplot.png")
    plt.close()

    # 图4: 散点图
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    scatter = ax4.scatter(df['Mean_MAE'], df['Std_MAE'],
                         c=df['Mean_MAE'], cmap='YlOrRd',
                         s=100, alpha=0.6, edgecolors='black')

    # 标注异常电池
    for battery in problematic_batteries:
        x = df.loc[battery, 'Mean_MAE']
        y = df.loc[battery, 'Std_MAE']
        ax4.annotate(battery, (x, y), fontsize=9,
                    xytext=(5, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax4.set_title('Battery Error Consistency Analysis', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Mean MAE (higher = harder to predict)', fontsize=12)
    ax4.set_ylabel('Std MAE (higher = inconsistent)', fontsize=12)
    ax4.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Mean MAE')
    plt.tight_layout()
    plt.savefig('battery_error_scatter.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Scatter plot saved: battery_error_scatter.png")
    plt.close()

    # 保存详细报告
    report_path = 'battery_error_report.csv'
    df_sorted.to_csv(report_path)
    print(f"[OK] 详细报告已保存: {report_path}")

    # 分析异常电池的共性
    if problematic_batteries:
        print("\n" + "="*80)
        print("异常电池详细分析")
        print("="*80)

        for battery in problematic_batteries:
            print(f"\n电池 {battery}:")
            print(f"  - 平均MAE: {df.loc[battery, 'Mean_MAE']:.6f}")
            print(f"  - 标准差: {df.loc[battery, 'Std_MAE']:.6f}")
            print(f"  - 最好模型: {df.loc[battery, model_cols].idxmin()} (MAE={df.loc[battery, model_cols].min():.6f})")
            print(f"  - 最差模型: {df.loc[battery, model_cols].idxmax()} (MAE={df.loc[battery, model_cols].max():.6f})")

    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

    return df_sorted, problematic_batteries


def visualize_all_test_batteries(df_results, mae_threshold=0.012, data_dir='data/HUST data'):
    """可视化所有测试集电池的容量衰减曲线，标注高误差电池"""

    print(f"\n生成所有测试集电池容量曲线 (标注MAE > {mae_threshold})...")

    # 获取所有测试集电池
    all_batteries = df_results.index.tolist()

    # 识别高误差电池
    high_error_batteries = df_results[df_results['Mean_MAE'] > mae_threshold].index.tolist()

    print(f"  总计测试集电池: {len(all_batteries)}")
    print(f"  高误差电池 (MAE > {mae_threshold}): {len(high_error_batteries)}")
    if high_error_batteries:
        print(f"  高误差电池列表: {high_error_batteries}")

    # 计算子图布局
    ncols = 4
    nrows = (len(all_batteries) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5*nrows))
    axes = axes.flatten() if len(all_batteries) > 1 else [axes]

    for idx, battery in enumerate(all_batteries):
        ax = axes[idx]
        battery_file = os.path.join(data_dir, f'{battery}.csv')

        try:
            import pandas as pd
            df = pd.read_csv(battery_file)

            # The cycle number is the row index (starting from 1)
            cycles = np.arange(1, len(df) + 1)
            capacity = df['capacity'].values

            # 根据误差选择颜色
            is_high_error = battery in high_error_batteries
            color = 'red' if is_high_error else 'steelblue'
            linewidth = 2.5 if is_high_error else 1.5

            ax.plot(cycles, capacity, color=color, linewidth=linewidth, alpha=0.8)

            # 标题包含MAE信息
            mae = df_results.loc[battery, 'Mean_MAE']
            title = f'{battery} (MAE={mae:.4f})'
            if is_high_error:
                title += ' [HIGH ERROR]'

            ax.set_title(title, fontsize=11, fontweight='bold' if is_high_error else 'normal',
                        color='red' if is_high_error else 'black')
            ax.set_xlabel('Cycle', fontsize=10)
            ax.set_ylabel('Capacity (Ah)', fontsize=10)
            ax.grid(alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, f'Failed to load\n{battery}\n{e}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_title(f'{battery} [ERROR]', fontsize=11, color='gray')

    # 隐藏多余的子图
    for idx in range(len(all_batteries), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'All Test Batteries Capacity Degradation (Red = MAE > {mae_threshold})',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = 'all_test_batteries_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] All batteries curves saved: {output_path}")
    plt.close()


def visualize_high_error_batteries_detailed(df_results, mae_threshold=0.012, data_dir='data/HUST data'):
    """单独详细可视化高误差电池"""

    high_error_batteries = df_results[df_results['Mean_MAE'] > mae_threshold].index.tolist()

    if not high_error_batteries:
        print(f"\nNo batteries with MAE > {mae_threshold}")
        return

    print(f"\n生成高误差电池详细曲线...")

    fig, axes = plt.subplots(len(high_error_batteries), 1,
                            figsize=(14, 5*len(high_error_batteries)))

    if len(high_error_batteries) == 1:
        axes = [axes]

    for idx, battery in enumerate(high_error_batteries):
        ax = axes[idx]
        battery_file = os.path.join(data_dir, f'{battery}.csv')

        try:
            import pandas as pd
            df = pd.read_csv(battery_file)

            # The cycle number is the row index (starting from 1)
            cycles = np.arange(1, len(df) + 1)
            capacity = df['capacity'].values
            mae = df_results.loc[battery, 'Mean_MAE']

            ax.plot(cycles, capacity, marker='o', markersize=4, linewidth=2.5,
                   color='red', label=f'Battery {battery}')
            ax.set_xlabel('Cycle', fontsize=12)
            ax.set_ylabel('Capacity (Ah)', fontsize=12)
            ax.set_title(f'Battery {battery} - Mean MAE: {mae:.4f} [HIGH ERROR]',
                        fontsize=14, fontweight='bold', color='red')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=11)

            # 添加统计信息
            capacity_range = capacity.max() - capacity.min()
            ax.text(0.02, 0.98, f'Cycles: {len(cycles)}\nCapacity Range: {capacity_range:.4f} Ah',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        except Exception as e:
            ax.text(0.5, 0.5, f'Failed to load {battery}\n{e}',
                   ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    output_path = 'high_error_batteries_detailed.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] High error batteries detailed curves saved: {output_path}")
    plt.close()


def analyze_battery_degradation_patterns(df_results, problematic_batteries, data_dir='data/HUST data'):
    """分析异常电池与正常电池在容量衰减模式上的差异"""

    print("\n" + "="*80)
    print("电池衰减模式差异分析")
    print("="*80)

    all_batteries = df_results.index.tolist()
    normal_batteries = [b for b in all_batteries if b not in problematic_batteries]

    # 存储每个电池的特征
    battery_features = {}

    for battery in all_batteries:
        battery_file = os.path.join(data_dir, f'{battery}.csv')

        try:
            df = pd.read_csv(battery_file)
            capacity = df['capacity'].values
            cycles = np.arange(1, len(df) + 1)

            # 计算多种衰减特征
            features = {
                'total_cycles': len(capacity),
                'initial_capacity': capacity[0],
                'final_capacity': capacity[-1],
                'capacity_loss': capacity[0] - capacity[-1],
                'capacity_loss_rate': (capacity[0] - capacity[-1]) / capacity[0],
                'mean_capacity': np.mean(capacity),
                'std_capacity': np.std(capacity),
                'cv_capacity': np.std(capacity) / np.mean(capacity),  # 变异系数
            }

            # 线性拟合衰减趋势
            z = np.polyfit(cycles, capacity, 1)
            features['linear_slope'] = z[0]  # 线性衰减斜率

            # 计算容量变化的加速度（二阶差分）
            capacity_diff = np.diff(capacity)
            capacity_accel = np.diff(capacity_diff)
            features['mean_degradation_rate'] = np.mean(capacity_diff)
            features['std_degradation_rate'] = np.std(capacity_diff)
            features['mean_acceleration'] = np.mean(capacity_accel)
            features['std_acceleration'] = np.std(capacity_accel)

            # 计算非线性程度（二次拟合的二次项系数）
            z2 = np.polyfit(cycles, capacity, 2)
            features['quadratic_coef'] = z2[0]  # 二次项系数

            # 计算容量衰减的波动性
            residuals = capacity - np.polyval(z, cycles)
            features['residual_std'] = np.std(residuals)
            features['residual_range'] = np.max(residuals) - np.min(residuals)

            battery_features[battery] = features

        except Exception as e:
            print(f"  警告: 无法加载 {battery}: {e}")

    # 转换为DataFrame
    features_df = pd.DataFrame(battery_features).T

    # 计算正常电池的统计特征
    normal_features = features_df.loc[normal_batteries]
    problematic_features = features_df.loc[problematic_batteries]

    print(f"\n正常电池数量: {len(normal_batteries)}")
    print(f"异常电池数量: {len(problematic_batteries)}")
    print(f"异常电池列表: {problematic_batteries}")

    # 对每个特征进行比较
    print("\n" + "-"*80)
    print("特征对比分析 (异常电池 vs 正常电池)")
    print("-"*80)
    print(f"{'特征名称':<25} {'正常均值':>12} {'正常std':>12} {'异常均值':>12} {'差异度':>12}")
    print("-"*80)

    feature_comparison = {}

    for feature in features_df.columns:
        normal_mean = normal_features[feature].mean()
        normal_std = normal_features[feature].std()
        problematic_mean = problematic_features[feature].mean()

        # 计算Z-score (标准化差异)
        if normal_std > 0:
            z_score = abs(problematic_mean - normal_mean) / normal_std
        else:
            z_score = 0

        feature_comparison[feature] = {
            'normal_mean': normal_mean,
            'normal_std': normal_std,
            'problematic_mean': problematic_mean,
            'z_score': z_score
        }

        print(f"{feature:<25} {normal_mean:>12.6f} {normal_std:>12.6f} {problematic_mean:>12.6f} {z_score:>12.2f}σ")

    # 找出差异最显著的特征 (Z-score > 1.5)
    print("\n" + "-"*80)
    print("显著差异特征 (|Z-score| > 1.5σ)")
    print("-"*80)

    significant_features = {k: v for k, v in feature_comparison.items() if v['z_score'] > 1.5}

    if significant_features:
        for feature, stats in sorted(significant_features.items(), key=lambda x: x[1]['z_score'], reverse=True):
            print(f"\n{feature}:")
            print(f"  正常电池: {stats['normal_mean']:.6f} ± {stats['normal_std']:.6f}")
            print(f"  异常电池: {stats['problematic_mean']:.6f}")
            print(f"  差异度: {stats['z_score']:.2f}σ")
    else:
        print("未发现显著差异特征")

    # 可视化特征对比
    print("\n生成特征对比可视化...")

    # 选择最重要的8个特征进行可视化
    top_features = sorted(feature_comparison.items(), key=lambda x: x[1]['z_score'], reverse=True)[:8]
    top_feature_names = [f[0] for f in top_features]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(top_feature_names):
        ax = axes[idx]

        # 绘制正常电池的分布
        normal_values = normal_features[feature].values
        problematic_values = problematic_features[feature].values

        ax.hist(normal_values, bins=10, alpha=0.6, label='Normal Batteries', color='steelblue', edgecolor='black')

        # 标注异常电池的值
        for i, (battery, value) in enumerate(zip(problematic_batteries, problematic_values)):
            ax.axvline(value, color='red', linestyle='--', linewidth=2,
                      label=f'{battery}' if i == 0 else '')
            ax.text(value, ax.get_ylim()[1]*0.9, f'{battery}\n{value:.4f}',
                   rotation=0, ha='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        z_score = feature_comparison[feature]['z_score']
        ax.set_title(f'{feature}\n(Z-score: {z_score:.2f}σ)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('Feature Distribution: Problematic vs Normal Batteries',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('battery_feature_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Feature comparison plot saved: battery_feature_comparison.png")
    plt.close()

    # 保存详细的特征对比报告
    comparison_df = pd.DataFrame(feature_comparison).T
    comparison_df = comparison_df.sort_values('z_score', ascending=False)
    comparison_df.to_csv('battery_feature_comparison.csv')
    print(f"[OK] Feature comparison report saved: battery_feature_comparison.csv")

    # 保存每个电池的详细特征
    features_df['is_problematic'] = features_df.index.isin(problematic_batteries)
    features_df['mean_mae'] = df_results['Mean_MAE']
    features_df = features_df.sort_values('mean_mae', ascending=False)
    features_df.to_csv('battery_degradation_features.csv')
    print(f"[OK] Battery features saved: battery_degradation_features.csv")

    print("\n" + "="*80)
    print("差异分析完成！")
    print("="*80)

    return features_df, feature_comparison


if __name__ == "__main__":
    # 分析电池误差
    df_results, problematic_batteries = analyze_battery_errors()

    # 可视化所有测试集电池的容量曲线（标注MAE > 0.018的，即前两个最异常的）
    visualize_all_test_batteries(df_results, mae_threshold=0.018)

    # 单独详细可视化高误差电池
    visualize_high_error_batteries_detailed(df_results, mae_threshold=0.018)

    # 分析异常电池与正常电池的差异
    battery_features, feature_comparison = analyze_battery_degradation_patterns(
        df_results, problematic_batteries
    )

    print("\n所有分析完成！请查看生成的图表和CSV文件。")
