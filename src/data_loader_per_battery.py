"""
Per-battery data loader for BPINN.
按论文方法：每个电池单独划分训练集和测试集。
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os


class BatteryDataset(Dataset):
    """PyTorch dataset for battery data."""

    def __init__(self, features, soh, p_ic):
        """
        Initialize dataset.

        Args:
            features: Feature matrix (N, 6)
            soh: SOH values (N,)
            p_ic: Peak IC values (N,)
        """
        self.features = torch.FloatTensor(features)
        self.soh = torch.FloatTensor(soh)
        self.p_ic = torch.FloatTensor(p_ic)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.soh[idx], self.p_ic[idx]


def load_single_battery_data(file_path, train_ratio=0.75):
    """
    加载单个电池数据并划分训练/测试集。

    按论文方法：
    - 训练集：前70-80%的循环（默认75%）
    - 测试集：后20-30%的循环
    - 不跨电池混合数据

    Args:
        file_path: CSV文件路径
        train_ratio: 训练集比例（默认0.75，即75%）

    Returns:
        Dictionary包含：
        - train_features, train_soh, train_p_ic: 训练集
        - test_features, test_soh, test_p_ic: 测试集
        - train_cycles, test_cycles: 循环编号（如果CSV中有）
        - scaler: StandardScaler对象
        - battery_name: 电池名称
    """
    # 读取数据
    df = pd.read_csv(file_path)
    battery_name = os.path.basename(file_path).split('_')[0]  # 'B05'

    # 检查是否有循环编号列
    has_cycle = 'Cycle' in df.columns or 'cycle' in df.columns
    cycle_col = 'Cycle' if 'Cycle' in df.columns else ('cycle' if 'cycle' in df.columns else None)

    # 划分训练/测试集（按时序）
    n_total = len(df)
    n_train = int(n_total * train_ratio)

    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]

    # 提取6个IC特征
    feature_columns = ['y_h', 'V_h', 'k_l', 'k_r', 't_G', 'Q_G']

    train_features = train_df[feature_columns].values
    train_soh = train_df['SOH'].values
    train_p_ic = train_df['y_h'].values  # Peak IC

    test_features = test_df[feature_columns].values
    test_soh = test_df['SOH'].values
    test_p_ic = test_df['y_h'].values

    # 标准化特征（使用训练集的统计量）
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    result = {
        'train_features': train_features_scaled,
        'train_soh': train_soh,
        'train_p_ic': train_p_ic,
        'test_features': test_features_scaled,
        'test_soh': test_soh,
        'test_p_ic': test_p_ic,
        'scaler': scaler,
        'battery_name': battery_name,
        'feature_columns': feature_columns,
        'n_train': n_train,
        'n_test': len(test_df)
    }

    # 如果有循环编号，也保存
    if has_cycle:
        result['train_cycles'] = train_df[cycle_col].values
        result['test_cycles'] = test_df[cycle_col].values
    else:
        # 如果没有循环列，创建索引
        result['train_cycles'] = np.arange(n_train)
        result['test_cycles'] = np.arange(n_train, n_total)

    return result


def create_data_loaders_for_battery(data_dict, batch_size=32, shuffle_train=True):
    """
    为单个电池创建DataLoader。

    Args:
        data_dict: load_single_battery_data返回的字典
        batch_size: batch大小
        shuffle_train: 是否shuffle训练集

    Returns:
        train_loader, test_loader
    """
    train_dataset = BatteryDataset(
        data_dict['train_features'],
        data_dict['train_soh'],
        data_dict['train_p_ic']
    )

    test_dataset = BatteryDataset(
        data_dict['test_features'],
        data_dict['test_soh'],
        data_dict['test_p_ic']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),  # 测试集一次全部加载
        shuffle=False
    )

    return train_loader, test_loader


def load_all_batteries(file_paths, train_ratio=0.75):
    """
    加载所有电池数据（per-battery模式）。

    Args:
        file_paths: CSV文件路径列表
        train_ratio: 训练集比例

    Returns:
        Dictionary: {battery_name: data_dict}
    """
    all_data = {}

    for file_path in file_paths:
        data = load_single_battery_data(file_path, train_ratio)
        battery_name = data['battery_name']
        all_data[battery_name] = data

        print(f"\nLoaded {battery_name}:")
        print(f"  Train: {data['n_train']} samples")
        print(f"  Test:  {data['n_test']} samples")
        print(f"  Train SOH range: {data['train_soh'].min():.3f} - {data['train_soh'].max():.3f}")
        print(f"  Test SOH range:  {data['test_soh'].min():.3f} - {data['test_soh'].max():.3f}")

    return all_data


if __name__ == "__main__":
    # 测试代码
    print("Testing per-battery data loader...")

    file_paths = [
        'data/B05_IC.csv',
        'data/B06_IC.csv',
        'data/B07_IC.csv'
    ]

    # 加载所有电池
    all_data = load_all_batteries(file_paths, train_ratio=0.75)

    # 测试单个电池的DataLoader
    battery_name = 'B05'
    train_loader, test_loader = create_data_loaders_for_battery(
        all_data[battery_name], batch_size=16
    )

    print(f"\n{battery_name} DataLoaders:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # 测试一个batch
    features, soh, p_ic = next(iter(train_loader))
    print(f"\n  Batch shape:")
    print(f"    Features: {features.shape}")
    print(f"    SOH: {soh.shape}")
    print(f"    P-IC: {p_ic.shape}")

    print("\nData loader test passed!")
