"""
特征筛选模块：根据相关性选择特征。

提供根据皮尔逊相关系数筛选特征的功能，
简化神经网络输入维度。
"""

import numpy as np
import pandas as pd
from scipy import stats


def get_top_correlated_features(data_dict, correlation_threshold=0.5):
    """
    提取与SOH相关系数绝对值大于阈值的特征。

    Args:
        data_dict: load_single_hust_battery()返回的字典
        correlation_threshold: 相关系数绝对值阈值 (默认0.5)

    Returns:
        Dictionary:
        - selected_features: 选中的特征索引列表
        - feature_names: 选中的特征名列表
        - correlations: 选中特征与SOH的相关系数
        - num_features: 选中的特征数
    """
    feature_names = data_dict['feature_names']
    train_features = data_dict['train_features']
    train_capacity = data_dict['train_capacity']

    # 计算每个特征与SOH的皮尔逊相关系数
    correlations = []
    for i, feature_name in enumerate(feature_names):
        feature_values = train_features[:, i]
        corr_result = stats.pearsonr(feature_values, train_capacity)
        corr = float(corr_result[0])
        p_value = float(corr_result[1])
        
        correlations.append({
            'index': i,
            'name': feature_name,
            'correlation': corr,
            'abs_correlation': abs(corr),
            'p_value': p_value
        })

    # 按绝对相关系数排序
    correlations = sorted(correlations, key=lambda x: x['abs_correlation'], reverse=True)

    # 筛选满足阈值的特征
    selected = [c for c in correlations if c['abs_correlation'] >= correlation_threshold]

    selected_indices = [c['index'] for c in selected]
    selected_names = [c['name'] for c in selected]
    selected_correlations = [c['correlation'] for c in selected]

    # 如果没有特征满足阈值，至少选择Top-5特征
    if len(selected_indices) == 0:
        print(f"Warning: No features with |correlation| >= {correlation_threshold}")
        print(f"Selecting top 5 features instead...")
        selected = correlations[:5]
        selected_indices = [c['index'] for c in selected]
        selected_names = [c['name'] for c in selected]
        selected_correlations = [c['correlation'] for c in selected]

    return {
        'selected_indices': selected_indices,
        'feature_names': selected_names,
        'correlations': selected_correlations,
        'num_features': len(selected_indices),
        'all_correlations': correlations  # 保存所有特征的相关系数信息用于参考
    }


def filter_features(features, feature_indices):
    """
    根据索引筛选特征。

    Args:
        features: numpy array of shape (n_samples, n_all_features)
        feature_indices: 要保留的特征索引列表

    Returns:
        Filtered features of shape (n_samples, len(feature_indices))
    """
    return features[:, feature_indices]


def print_feature_selection_report(selection_result, correlation_threshold=0.5):
    """
    打印特征筛选报告。

    Args:
        selection_result: get_top_correlated_features()返回的字典
        correlation_threshold: 使用的相关系数阈值
    """
    print("\n" + "=" * 70)
    print("特征筛选报告")
    print("=" * 70)

    print(f"\n筛选条件: |相关系数| >= {correlation_threshold}")
    print(f"\n选中特征数: {selection_result['num_features']}")
    print(f"特征索引: {selection_result['selected_indices']}")

    print(f"\n{'#':<3} {'特征名':<30} {'相关系数':<12} {'绝对值':<12} {'关系':<10}")
    print("-" * 70)

    for i, (name, corr) in enumerate(zip(
            selection_result['feature_names'],
            selection_result['correlations']), 1):
        direction = "Positive" if corr > 0 else "Negative"
        print(f"{i:<3} {name:<30} {corr:<12.6f} {abs(corr):<12.6f} {direction:<10}")

    print("\n" + "-" * 70)
    print(f"所有特征相关性信息 (按绝对相关系数排序):\n")
    print(f"{'特征':<30} {'相关系数':<12} {'绝对值':<12} {'p值':<12}")
    print("-" * 70)

    for corr_info in selection_result['all_correlations']:
        p_sig = "Yes" if corr_info['p_value'] < 0.05 else "No"
        print(f"{corr_info['name']:<30} {corr_info['correlation']:<12.6f} "
              f"{corr_info['abs_correlation']:<12.6f} {corr_info['p_value']:<12.6e} {p_sig}")

    print("=" * 70)


def create_filtered_data_dict(data_dict, feature_indices):
    """
    创建新的数据字典，只包含选定的特征。

    Args:
        data_dict: 原始数据字典
        feature_indices: 选定的特征索引列表

    Returns:
        新的数据字典，包含筛选后的特征
    """
    selected_feature_names = [data_dict['feature_names'][i] for i in feature_indices]

    filtered_dict = {
        'train_features': data_dict['train_features'][:, feature_indices],
        'train_capacity': data_dict['train_capacity'],
        'test_features': data_dict['test_features'][:, feature_indices],
        'test_capacity': data_dict['test_capacity'],
        'scaler': data_dict['scaler'],
        'battery_name': data_dict['battery_name'],
        'feature_names': selected_feature_names,
        'n_train': data_dict['n_train'],
        'n_test': data_dict['n_test'],
        'rated_capacity': data_dict['rated_capacity'],
        'normalize_target': data_dict['normalize_target'],
        'original_feature_count': len(data_dict['feature_names']),
        'selected_feature_count': len(feature_indices),
        'selected_feature_indices': feature_indices
    }

    return filtered_dict


if __name__ == "__main__":
    # Test feature selector
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from src.data_loader_hust import load_single_hust_battery

    # Load test data
    file_path = 'data/HUST data/1-1.csv'
    if os.path.exists(file_path):
        data_dict = load_single_hust_battery(file_path, train_ratio=0.75, normalize_target=True)

        print(f"Battery: {data_dict['battery_name']}")
        print(f"Total features: {len(data_dict['feature_names'])}")

        # Test with threshold 0.5
        selection_result = get_top_correlated_features(data_dict, correlation_threshold=0.5)
        print_feature_selection_report(selection_result, correlation_threshold=0.5)

        # Test filtered data
        filtered_data = create_filtered_data_dict(data_dict, selection_result['selected_indices'])
        print(f"\nFiltered data shape:")
        print(f"  Train features: {filtered_data['train_features'].shape}")
        print(f"  Test features: {filtered_data['test_features'].shape}")
        print(f"  Selected features: {filtered_data['feature_names']}")

        print("\n✅ Feature selector test passed!")
    else:
        print(f"❌ File not found: {file_path}")
