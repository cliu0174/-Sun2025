"""
按电池单独计算相关系数，再取平均
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义CC和CV阶段的特征
CC_FEATURES = [
    'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
    'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy'
]

CV_FEATURES = [
    'current mean', 'current std', 'current kurtosis', 'current skewness',
    'CV Q', 'CV charge time', 'current slope', 'current entropy'
]

ALL_FEATURES = CC_FEATURES + CV_FEATURES


def calculate_battery_correlation(df, features, target='capacity'):
    """计算单个电池的特征与目标变量的相关系数"""
    correlations = {}
    for feature in features:
        corr = df[feature].corr(df[target])
        correlations[feature] = corr
    return correlations


def analyze_per_battery(data_dir='data_features'):
    """按电池单独计算相关系数"""
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    csv_files.sort()

    # 存储每个电池的相关系数
    all_battery_corr = {}

    for csv_file in csv_files:
        battery_id = os.path.basename(csv_file).replace('.csv', '')
        df = pd.read_csv(csv_file)

        # 计算该电池的相关系数
        corr = calculate_battery_correlation(df, ALL_FEATURES)
        all_battery_corr[battery_id] = corr

    # 转换为DataFrame
    corr_df = pd.DataFrame(all_battery_corr).T
    corr_df.index.name = 'battery_id'

    return corr_df


def plot_single_battery(battery_id, data_dir='data_features'):
    """展示单个电池的相关性分析"""
    csv_path = os.path.join(data_dir, f'{battery_id}.csv')
    df = pd.read_csv(csv_path)

    print(f"\n{'='*60}")
    print(f"电池 {battery_id} 数据概览")
    print(f"{'='*60}")
    print(f"循环数: {len(df)}")
    print(f"容量范围: {df['capacity'].min():.4f} ~ {df['capacity'].max():.4f} Ah")

    # 计算相关系数
    cc_corr = calculate_battery_correlation(df, CC_FEATURES)
    cv_corr = calculate_battery_correlation(df, CV_FEATURES)

    print(f"\n{'='*60}")
    print(f"电池 {battery_id} CC阶段特征与SOH相关系数")
    print(f"{'='*60}")
    for feat, corr in sorted(cc_corr.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat:25s}: {corr:+.4f}")

    print(f"\n{'='*60}")
    print(f"电池 {battery_id} CV阶段特征与SOH相关系数")
    print(f"{'='*60}")
    for feat, corr in sorted(cv_corr.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat:25s}: {corr:+.4f}")

    # 绘制热力图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # CC阶段热力图
    cc_corr_matrix = pd.DataFrame({
        'Feature': CC_FEATURES,
        'Correlation': [cc_corr[f] for f in CC_FEATURES]
    }).set_index('Feature')

    sns.heatmap(cc_corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=axes[0],
                cbar_kws={'label': 'Correlation'})
    axes[0].set_title(f'电池 {battery_id} CC阶段特征与SOH相关性', fontsize=14, fontweight='bold')

    # CV阶段热力图
    cv_corr_matrix = pd.DataFrame({
        'Feature': CV_FEATURES,
        'Correlation': [cv_corr[f] for f in CV_FEATURES]
    }).set_index('Feature')

    sns.heatmap(cv_corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=axes[1],
                cbar_kws={'label': 'Correlation'})
    axes[1].set_title(f'电池 {battery_id} CV阶段特征与SOH相关性', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'correlation_{battery_id}.png', dpi=150, bbox_inches='tight')
    print(f"\n热力图已保存至: correlation_{battery_id}.png")
    plt.show()

    return cc_corr, cv_corr


def plot_average_correlation(corr_df):
    """绘制所有电池平均相关系数的热力图"""
    # 计算平均值和标准差
    mean_corr = corr_df.mean()
    std_corr = corr_df.std()

    print(f"\n{'='*60}")
    print(f"所有电池平均相关系数 (n={len(corr_df)})")
    print(f"{'='*60}")

    print("\nCC阶段特征:")
    print("-" * 50)
    for feat in CC_FEATURES:
        print(f"  {feat:25s}: {mean_corr[feat]:+.4f} ± {std_corr[feat]:.4f}")

    print("\nCV阶段特征:")
    print("-" * 50)
    for feat in CV_FEATURES:
        print(f"  {feat:25s}: {mean_corr[feat]:+.4f} ± {std_corr[feat]:.4f}")

    # 绘制热力图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # CC阶段
    cc_data = pd.DataFrame({
        'Feature': CC_FEATURES,
        'Mean Correlation': [mean_corr[f] for f in CC_FEATURES]
    }).set_index('Feature')

    sns.heatmap(cc_data, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=axes[0],
                cbar_kws={'label': 'Correlation'})
    axes[0].set_title('CC阶段特征与SOH相关性\n(所有电池平均)', fontsize=14, fontweight='bold')

    # CV阶段
    cv_data = pd.DataFrame({
        'Feature': CV_FEATURES,
        'Mean Correlation': [mean_corr[f] for f in CV_FEATURES]
    }).set_index('Feature')

    sns.heatmap(cv_data, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=axes[1],
                cbar_kws={'label': 'Correlation'})
    axes[1].set_title('CV阶段特征与SOH相关性\n(所有电池平均)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('correlation_average.png', dpi=150, bbox_inches='tight')
    print(f"\n平均相关系数热力图已保存至: correlation_average.png")
    plt.show()

    return mean_corr, std_corr


if __name__ == '__main__':
    # 1. 展示电池1-1的相关性分析
    print("\n" + "=" * 70)
    print(" 电池 1-1 单独分析 ")
    print("=" * 70)
    plot_single_battery('1-1')

    # 2. 计算所有电池的相关系数
    print("\n" + "=" * 70)
    print(" 计算所有电池的相关系数并取平均 ")
    print("=" * 70)
    corr_df = analyze_per_battery()

    # 3. 绘制平均相关系数热力图
    mean_corr, std_corr = plot_average_correlation(corr_df)
