"""
MIT数据集加载器

数据集: MIT电池老化数据集
- 3个批次: 2017-05-12, 2017-06-30, 2018-04-12
- 每个批次包含多个电池
- 特征: 16个充电统计特征
- 目标: capacity (容量值，需归一化为SOH)
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


def load_single_mit_battery(battery_path, train_ratio=1.0, normalize_target=True, apply_cleaning=False):
    """
    加载单个MIT电池的数据（格式兼容HUST加载器）

    Args:
        battery_path: 电池CSV文件路径
        train_ratio: 训练集比例 (默认1.0，全部作为训练集)
        normalize_target: 是否归一化目标值为SOH (默认True)
        apply_cleaning: 是否应用3-Sigma数据清洗

    Returns:
        Dictionary包含：
        - train_features: 训练集特征 (标准化后)
        - train_capacity: 训练集SOH/容量
        - test_features: 测试集特征
        - test_capacity: 测试集SOH/容量
        - scaler: StandardScaler对象
        - cleaning_stats: 清洗统计 (可选)
    """
    # 读取CSV
    df = pd.read_csv(battery_path)

    # MIT数据集特征列名
    feature_columns = [
        'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
        'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
        'current mean', 'current std', 'current kurtosis', 'current skewness',
        'CV Q', 'CV charge time', 'current slope', 'current entropy'
    ]

    # 目标列
    target_column = 'capacity'

    # 检查列是否存在
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {battery_path}")

    # 数据清洗（可选）
    cleaning_stats = None
    if apply_cleaning:
        from .data_loader_hust import clean_3_sigma
        df_temp = df[feature_columns + [target_column]].copy()
        df, cleaning_stats = clean_3_sigma(df_temp, verbose=False)

    # 划分训练集和测试集
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # 提取特征和容量
    train_features = train_df[feature_columns].values
    train_capacity = train_df[target_column].values

    test_features = test_df[feature_columns].values if len(test_df) > 0 else np.array([]).reshape(0, len(feature_columns))
    test_capacity = test_df[target_column].values if len(test_df) > 0 else np.array([])

    # 标准化特征 (使用训练集的统计量)
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)

    # 处理测试集为空的情况（train_ratio=1.0）
    if len(test_features) > 0:
        test_features_scaled = scaler.transform(test_features)
    else:
        test_features_scaled = np.array([]).reshape(0, len(feature_columns))

    # 归一化目标值为SOH (可选)
    if normalize_target:
        # 使用初始容量归一化
        initial_capacity = train_df[target_column].iloc[0]
        train_capacity_normalized = train_capacity / initial_capacity
        if len(test_capacity) > 0:
            test_capacity_normalized = test_capacity / initial_capacity
        else:
            test_capacity_normalized = np.array([])
    else:
        train_capacity_normalized = train_capacity
        test_capacity_normalized = test_capacity

    result = {
        'train_features': train_features_scaled,
        'train_capacity': train_capacity_normalized,
        'test_features': test_features_scaled,
        'test_capacity': test_capacity_normalized,
        'scaler': scaler
    }

    if cleaning_stats:
        result['cleaning_stats'] = cleaning_stats

    return result


def load_all_mit_batteries(data_dir='data/MIT data', apply_cleaning=False):
    """
    加载所有MIT电池数据（格式兼容HUST加载器）

    Args:
        data_dir: MIT数据集根目录
        apply_cleaning: 是否应用数据清洗

    Returns:
        battery_names: list of str, 电池名称列表
        all_data: dict, {battery_name: {...}}，格式与HUST相同
    """
    battery_names = []
    all_data = {}

    # 统计清洗效果
    total_removed = 0
    total_original = 0

    # 遍历所有批次
    batches = ['2017-05-12', '2017-06-30', '2018-04-12']

    for batch in batches:
        batch_dir = os.path.join(data_dir, batch)
        if not os.path.exists(batch_dir):
            print(f"[WARNING] Batch directory not found: {batch_dir}")
            continue

        # 遍历该批次的所有电池文件
        for filename in sorted(os.listdir(batch_dir)):
            if filename.endswith('.csv'):
                # 提取电池名称，例如 "2017-05-12_battery-1" -> "2017-05-12_b1"
                battery_name = filename.replace('.csv', '').replace('battery-', 'b')
                battery_path = os.path.join(batch_dir, filename)

                try:
                    data = load_single_mit_battery(
                        battery_path,
                        train_ratio=1.0,
                        normalize_target=True,
                        apply_cleaning=apply_cleaning
                    )

                    battery_names.append(battery_name)
                    all_data[battery_name] = data

                    # 统计清洗效果
                    if apply_cleaning and data.get('cleaning_stats'):
                        stats = data['cleaning_stats']
                        total_removed += stats['total_removed']
                        total_original += stats['original_size']

                except Exception as e:
                    print(f"[ERROR] Failed to load {battery_name}: {e}")
                    continue

    print(f"\n[OK] Successfully loaded {len(battery_names)} MIT batteries")

    # 打印电池cycle数统计
    cycle_counts = [len(all_data[name]['train_features']) for name in battery_names]
    print(f"  Min cycles: {min(cycle_counts)}")
    print(f"  Max cycles: {max(cycle_counts)}")
    print(f"  Mean cycles: {np.mean(cycle_counts):.1f}")

    # 打印清洗统计
    if apply_cleaning and total_original > 0:
        overall_rate = (total_removed / total_original) * 100
        print(f"[3-Sigma清洗] 原始样本: {total_original}, 删除: {total_removed}, 删除率: {overall_rate:.2f}%")

    return battery_names, all_data


class MITBatteryDatasetWithMetadata(Dataset):
    """
    MIT电池数据集(带元数据)

    返回格式:
    {
        'window': (window_size, feature_dim) tensor,
        'target_soh': (1,) tensor,
        'battery_id': str,
        'cycle_idx': int tensor
    }
    """
    def __init__(self, X, y, battery_ids, cycle_indices):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)  # (N, 1)
        self.battery_ids = battery_ids
        self.cycle_indices = torch.LongTensor(cycle_indices)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'window': self.X[idx],
            'target_soh': self.y[idx],
            'battery_id': self.battery_ids[idx],
            'cycle_idx': self.cycle_indices[idx]
        }


def create_mit_dataloaders(data_dict, batch_size=64, window_size=40):
    """
    创建MIT数据集的DataLoader

    Args:
        data_dict: prepare_mit_data()返回的字典
        batch_size: 批次大小
        window_size: 窗口大小

    Returns:
        train_loader, val_loader, test_loader
    """
    from .data_loader_hust import apply_windowing_with_metadata

    train_loader = DataLoader(
        MITBatteryDatasetWithMetadata(
            data_dict['X_train'],
            data_dict['y_train'],
            data_dict['battery_ids_train'],
            data_dict['cycle_indices_train']
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    val_loader = DataLoader(
        MITBatteryDatasetWithMetadata(
            data_dict['X_val'],
            data_dict['y_val'],
            data_dict['battery_ids_val'],
            data_dict['cycle_indices_val']
        ),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    test_loader = DataLoader(
        MITBatteryDatasetWithMetadata(
            data_dict['X_test'],
            data_dict['y_test'],
            data_dict['battery_ids_test'],
            data_dict['cycle_indices_test']
        ),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    return train_loader, val_loader, test_loader
