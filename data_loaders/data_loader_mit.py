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


def load_single_mit_battery(battery_path, apply_cleaning=False):
    """
    加载单个MIT电池的数据

    Args:
        battery_path: 电池CSV文件路径
        apply_cleaning: 是否应用3-Sigma数据清洗

    Returns:
        features: (n_cycles, 16) numpy array
        soh: (n_cycles,) numpy array, 归一化的SOH值
        capacity: (n_cycles,) numpy array, 原始容量值
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

    # 提取特征和容量
    features = df[feature_columns].values
    capacity = df[target_column].values

    # 数据清洗（可选）
    if apply_cleaning:
        from .data_loader_hust import clean_3_sigma
        df_temp = df[feature_columns + [target_column]].copy()
        df_clean = clean_3_sigma(df_temp, verbose=False)
        features = df_clean[feature_columns].values
        capacity = df_clean[target_column].values

    # 计算SOH (归一化容量)
    # SOH = 当前容量 / 初始容量
    initial_capacity = capacity[0]  # 第一个cycle的容量作为初始容量
    soh = capacity / initial_capacity

    return features, soh, capacity


def load_all_mit_batteries(data_dir='data/MIT data', apply_cleaning=False):
    """
    加载所有MIT电池数据

    Args:
        data_dir: MIT数据集根目录
        apply_cleaning: 是否应用数据清洗

    Returns:
        battery_names: list of str, 电池名称列表
        all_data: dict, {battery_name: (features, soh, capacity)}
    """
    battery_names = []
    all_data = {}

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
                    features, soh, capacity = load_single_mit_battery(
                        battery_path,
                        apply_cleaning=apply_cleaning
                    )

                    battery_names.append(battery_name)
                    all_data[battery_name] = (features, soh, capacity)

                except Exception as e:
                    print(f"[ERROR] Failed to load {battery_name}: {e}")
                    continue

    print(f"\n[OK] Successfully loaded {len(battery_names)} MIT batteries")

    # 打印电池cycle数统计
    cycle_counts = [len(all_data[name][0]) for name in battery_names]
    print(f"  Min cycles: {min(cycle_counts)}")
    print(f"  Max cycles: {max(cycle_counts)}")
    print(f"  Mean cycles: {np.mean(cycle_counts):.1f}")

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
