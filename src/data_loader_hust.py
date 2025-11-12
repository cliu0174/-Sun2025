"""
HUST数据集加载器 (Per-Battery模式)

数据集: 77个LFP电池，每个电池有1000+个充电循环数据
特征: 16个充电统计特征
目标: Capacity (容量值)
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


def load_single_hust_battery(file_path, train_ratio=0.75, normalize_target=True):
    """
    加载单个HUST电池数据并划分训练/测试集。

    Args:
        file_path: CSV文件路径 (例如: 'data/HUST data/1-1.csv')
        train_ratio: 训练集比例 (默认0.75，即75%)
        normalize_target: 是否归一化目标值为SOH (默认True)

    Returns:
        Dictionary包含：
        - train_features, train_capacity: 训练集
        - test_features, test_capacity: 测试集
        - scaler: StandardScaler对象
        - battery_name: 电池名称 (例如: '1-1')
        - feature_names: 特征名列表
        - rated_capacity: 额定容量 (1.1 Ah)
    """
    # 读取数据
    df = pd.read_csv(file_path)
    battery_name = os.path.basename(file_path).replace('.csv', '')

    # 16个输入特征
    feature_columns = [
        'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
        'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
        'current mean', 'current std', 'current kurtosis', 'current skewness',
        'CV Q', 'CV charge time', 'current slope', 'current entropy'
    ]

    # 目标变量
    target_column = 'capacity'

    # 检查所有列是否存在
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {file_path}: {missing_cols}")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {file_path}")

    # 按时序划分训练/测试集
    n_total = len(df)
    n_train = int(n_total * train_ratio)

    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]

    # 提取特征和目标
    train_features = train_df[feature_columns].values
    train_capacity = train_df[target_column].values

    test_features = test_df[feature_columns].values
    test_capacity = test_df[target_column].values

    # 标准化特征 (使用训练集的统计量)
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)

    # 处理测试集为空的情况（train_ratio=1.0）
    if len(test_features) > 0:
        test_features_scaled = scaler.transform(test_features)
    else:
        test_features_scaled = np.array([]).reshape(0, len(feature_columns))

    # 额定容量 (LFP电池: 1.1 Ah)
    rated_capacity = 1.1

    # 归一化目标值为SOH (可选)
    if normalize_target:
        # 方法1: 使用额定容量
        # train_capacity_normalized = train_capacity / rated_capacity
        # test_capacity_normalized = test_capacity / rated_capacity

        # 方法2: 使用初始容量 (更准确)
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
        'scaler': scaler,
        'battery_name': battery_name,
        'feature_names': feature_columns,
        'n_train': n_train,
        'n_test': len(test_df),
        'rated_capacity': rated_capacity,
        'normalize_target': normalize_target
    }

    return result


def load_all_hust_batteries(data_dir='data/HUST data', train_ratio=0.75,
                            normalize_target=True, max_batteries=None):
    """
    加载所有HUST电池数据。

    Args:
        data_dir: HUST数据目录
        train_ratio: 训练集比例
        normalize_target: 是否归一化目标值
        max_batteries: 最多加载多少个电池 (None表示全部)

    Returns:
        Dictionary: {battery_name: data_dict}
    """
    all_data = {}

    # 获取所有CSV文件
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

    if max_batteries is not None:
        csv_files = csv_files[:max_batteries]

    print(f"Found {len(csv_files)} battery files")

    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        try:
            data = load_single_hust_battery(file_path, train_ratio, normalize_target)
            all_data[data['battery_name']] = data

            print(f"Loaded {data['battery_name']}: "
                  f"Train={data['n_train']}, Test={data['n_test']}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    return all_data


class HUSTBatteryDataset(Dataset):
    """PyTorch Dataset for HUST battery data."""

    def __init__(self, features, capacity):
        """
        Args:
            features: numpy array of shape (n_samples, 16)
            capacity: numpy array of shape (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.capacity = torch.FloatTensor(capacity)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.capacity[idx]


def create_hust_dataloaders(data_dict, batch_size=64, shuffle_train=True):
    """
    创建PyTorch DataLoader。

    Args:
        data_dict: load_single_hust_battery()返回的字典
        batch_size: Batch大小
        shuffle_train: 是否打乱训练集

    Returns:
        train_loader, test_loader
    """
    train_dataset = HUSTBatteryDataset(
        data_dict['train_features'],
        data_dict['train_capacity']
    )

    test_dataset = HUSTBatteryDataset(
        data_dict['test_features'],
        data_dict['test_capacity']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


if __name__ == "__main__":
    """测试数据加载器"""
    print("Testing HUST data loader...")
    print("=" * 70)

    # 测试加载单个电池
    file_path = 'data/HUST data/1-1.csv'
    if os.path.exists(file_path):
        data = load_single_hust_battery(file_path, train_ratio=0.75, normalize_target=True)

        print(f"\nBattery: {data['battery_name']}")
        print(f"  Train samples: {data['n_train']}")
        print(f"  Test samples: {data['n_test']}")
        print(f"  Features: {len(data['feature_names'])}")
        print(f"  Feature names: {data['feature_names'][:3]}...")
        print(f"  Train capacity range: {data['train_capacity'].min():.4f} - {data['train_capacity'].max():.4f}")
        print(f"  Test capacity range: {data['test_capacity'].min():.4f} - {data['test_capacity'].max():.4f}")
        print(f"  Normalized: {data['normalize_target']}")

        # 测试DataLoader
        train_loader, test_loader = create_hust_dataloaders(data, batch_size=64)
        print(f"\nDataLoader created:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")

        # 测试一个batch
        features, capacity = next(iter(train_loader))
        print(f"\nSample batch:")
        print(f"  Features shape: {features.shape}")
        print(f"  Capacity shape: {capacity.shape}")

        print("\n✅ Data loader test passed!")
    else:
        print(f"❌ File not found: {file_path}")

    # 测试加载多个电池
    print("\n" + "=" * 70)
    print("Testing batch loading (first 3 batteries)...")
    print("=" * 70)

    data_dir = 'data/HUST data'
    if os.path.exists(data_dir):
        all_data = load_all_hust_batteries(data_dir, max_batteries=3)
        print(f"\n✅ Loaded {len(all_data)} batteries successfully!")

        # 统计信息
        total_train = sum(d['n_train'] for d in all_data.values())
        total_test = sum(d['n_test'] for d in all_data.values())
        print(f"\nTotal statistics:")
        print(f"  Total train samples: {total_train}")
        print(f"  Total test samples: {total_test}")
    else:
        print(f"❌ Directory not found: {data_dir}")
