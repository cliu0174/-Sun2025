"""
CC和CV阶段特征与SOH相关性分析
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

def load_all_data(data_dir='data_features'):
    """加载所有电池特征数据"""
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    all_data = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        battery_id = os.path.basename(csv_file).replace('.csv', '')
        df['battery_id'] = battery_id
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"加载了 {len(csv_files)} 个电池的数据，共 {len(combined_df)} 个循环")
    return combined_df

def calculate_correlation(df, features, target='capacity'):
    """计算特征与目标变量的相关系数"""
    correlations = {}
    for feature in features:
        corr = df[feature].corr(df[target])
        correlations[feature] = corr
    return correlations

def plot_correlation_heatmaps(df):
    """绘制CC和CV阶段特征与SOH的相关性热力图"""
    # 计算相关系数
    cc_corr = calculate_correlation(df, CC_FEATURES)
    cv_corr = calculate_correlation(df, CV_FEATURES)

    # 打印相关系数
    print("\n" + "="*60)
    print("CC阶段特征与SOH(capacity)的相关系数:")
    print("="*60)
    for feat, corr in sorted(cc_corr.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat:25s}: {corr:+.4f}")

    print("\n" + "="*60)
    print("CV阶段特征与SOH(capacity)的相关系数:")
    print("="*60)
    for feat, corr in sorted(cv_corr.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat:25s}: {corr:+.4f}")

    # 创建热力图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # CC阶段热力图
    cc_corr_matrix = pd.DataFrame({
        'Feature': CC_FEATURES,
        'Correlation with SOH': [cc_corr[f] for f in CC_FEATURES]
    }).set_index('Feature')

    sns.heatmap(cc_corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=axes[0],
                cbar_kws={'label': 'Correlation'})
    axes[0].set_title('CC阶段特征与SOH相关性', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')

    # CV阶段热力图
    cv_corr_matrix = pd.DataFrame({
        'Feature': CV_FEATURES,
        'Correlation with SOH': [cv_corr[f] for f in CV_FEATURES]
    }).set_index('Feature')

    sns.heatmap(cv_corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=axes[1],
                cbar_kws={'label': 'Correlation'})
    axes[1].set_title('CV阶段特征与SOH相关性', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')

    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"\n热力图已保存至: correlation_heatmap.png")
    plt.show()

    # 绘制完整的特征相关性矩阵热力图
    plot_full_correlation_matrix(df)

    return cc_corr, cv_corr

def plot_full_correlation_matrix(df):
    """绘制所有特征之间的相关性矩阵"""
    all_features = CC_FEATURES + CV_FEATURES + ['capacity']
    corr_matrix = df[all_features].corr()

    fig, ax = plt.subplots(figsize=(14, 12))

    mask = np.zeros_like(corr_matrix)
    # 不使用上三角mask，显示完整矩阵

    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=ax,
                square=True, linewidths=0.5,
                cbar_kws={'label': 'Correlation', 'shrink': 0.8})

    ax.set_title('所有特征相关性矩阵 (CC + CV + SOH)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('full_correlation_matrix.png', dpi=150, bbox_inches='tight')
    print(f"完整相关性矩阵已保存至: full_correlation_matrix.png")
    plt.show()

if __name__ == '__main__':
    # 加载数据
    df = load_all_data('data_features')

    # 绘制热力图
    cc_corr, cv_corr = plot_correlation_heatmaps(df)
