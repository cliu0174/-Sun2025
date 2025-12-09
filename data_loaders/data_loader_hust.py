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


def apply_windowing_with_metadata(features, targets, window_size, battery_id, mode='many_to_one'):
    """
    应用滑动窗口并附加 battery_id 和 cycle_idx（用于物理约束）

    Args:
        features: (total_cycles, feature_dim) 特征数组
        targets: (total_cycles,) 目标值数组
        window_size: 窗口大小
        battery_id: str, 电池名称（如 "1-1"）
        mode: 'many_to_one' 或 'many_to_many'

    Returns:
        X: (N, window_size, feature_dim) 输入窗口
        y: (N,) for many_to_one 或 (N, window_size, 1) for many_to_many
        battery_ids: (N,) 每个窗口的电池ID
        cycle_indices: (N,) 每个窗口对应的cycle索引
    """
    X_windows = []
    y_windows = []
    battery_ids = []
    cycle_indices = []

    total_cycles = len(features)

    for i in range(total_cycles - window_size + 1):
        # 输入窗口
        X_windows.append(features[i:i+window_size])

        if mode == 'many_to_one':
            # Many-to-One: 预测窗口最后一个点的目标值
            y_windows.append(targets[i+window_size-1])
            # cycle_idx = 窗口最后一个点的原始cycle索引
            cycle_idx = i + window_size - 1

        elif mode == 'many_to_many':
            # Many-to-Many: 预测整个窗口的目标值
            y_windows.append(targets[i:i+window_size])
            # cycle_idx 定义为窗口的结束位置
            cycle_idx = i + window_size - 1

        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'many_to_one' or 'many_to_many'")

        # 附加元数据
        battery_ids.append(battery_id)
        cycle_indices.append(cycle_idx)

    return (
        np.array(X_windows),
        np.array(y_windows),
        np.array(battery_ids, dtype=object),  # 字符串数组
        np.array(cycle_indices, dtype=np.int32)
    )


def clean_3_sigma(df, verbose=False):
    """
    使用 3-Sigma 规则清洗异常值

    规则: 删除超过 mean ± 3*std 的数据点

    Args:
        df: DataFrame，包含特征和目标
        verbose: 是否打印清洗信息

    Returns:
        cleaned_df: 清洗后的 DataFrame
        stats: 清洗统计信息
    """
    original_size = len(df)

    # 1. 替换无穷大为 NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # 2. 删除含 NaN 的行
    df = df.dropna()
    after_nan = len(df)

    # 3. 对每列应用 3-Sigma 规则
    removed_by_column = {}

    for col in df.columns:
        before = len(df)
        mean = df[col].mean()
        std = df[col].std()

        # 计算上下界
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std

        # 删除超过界限的行
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        removed = before - len(df)
        if removed > 0:
            removed_by_column[col] = removed

    final_size = len(df)

    # 统计信息
    stats = {
        'original_size': original_size,
        'after_nan_removal': after_nan,
        'final_size': final_size,
        'total_removed': original_size - final_size,
        'nan_removed': original_size - after_nan,
        'outliers_removed': after_nan - final_size,
        'removal_rate': (original_size - final_size) / original_size * 100 if original_size > 0 else 0,
        'removed_by_column': removed_by_column
    }

    if verbose:
        print(f"  3-Sigma清洗: {original_size} -> {final_size} "
              f"(删除 {stats['total_removed']}, {stats['removal_rate']:.1f}%)")

    return df, stats


def load_single_hust_battery(file_path, train_ratio=0.75, normalize_target=True, apply_cleaning=False):
    """
    加载单个HUST电池数据并划分训练/测试集。

    Args:
        file_path: CSV文件路径 (例如: 'data/HUST data/1-1.csv')
        train_ratio: 训练集比例 (默认0.75，即75%)
        normalize_target: 是否归一化目标值为SOH (默认True)
        apply_cleaning: 是否应用3-Sigma清洗 (默认False，保持向后兼容)

    Returns:
        Dictionary包含：
        - train_features, train_capacity: 训练集
        - test_features, test_capacity: 测试集
        - scaler: StandardScaler对象
        - battery_name: 电池名称 (例如: '1-1')
        - feature_names: 特征名列表
        - rated_capacity: 额定容量 (1.1 Ah)
        - cleaning_stats: 清洗统计（如果启用清洗）
    """
    # 读取数据
    df = pd.read_csv(file_path)
    battery_name = os.path.basename(file_path).replace('.csv', '')

    # 应用 3-Sigma 清洗（可选）
    cleaning_stats = None
    if apply_cleaning:
        df, cleaning_stats = clean_3_sigma(df, verbose=False)

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
        # 方法1: 使用初始容量 (更准确，推荐使用)
        initial_capacity = train_df[target_column].iloc[0]
        train_capacity_normalized = train_capacity / initial_capacity
        if len(test_capacity) > 0:
            test_capacity_normalized = test_capacity / initial_capacity
        else:
            test_capacity_normalized = np.array([])

        # 方法2: 使用额定容量 (PINN4SOH 方法，已弃用)
        # train_capacity_normalized = train_capacity / rated_capacity
        # if len(test_capacity) > 0:
        #     test_capacity_normalized = test_capacity / rated_capacity
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
        'normalize_target': normalize_target,
        'cleaning_stats': cleaning_stats  # 清洗统计信息（如果启用）
    }

    return result


def load_all_hust_batteries(data_dir='data/HUST data', train_ratio=0.75,
                            normalize_target=True, max_batteries=None, apply_cleaning=False):
    """
    加载所有HUST电池数据。

    Args:
        data_dir: HUST数据目录
        train_ratio: 训练集比例
        normalize_target: 是否归一化目标值
        max_batteries: 最多加载多少个电池 (None表示全部)
        apply_cleaning: 是否应用3-Sigma清洗 (默认False)

    Returns:
        Dictionary: {battery_name: data_dict}
    """
    all_data = {}

    # 获取所有CSV文件
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

    if max_batteries is not None:
        csv_files = csv_files[:max_batteries]

    print(f"Found {len(csv_files)} battery files")

    # 统计清洗效果
    total_removed = 0
    total_original = 0

    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        try:
            data = load_single_hust_battery(file_path, train_ratio, normalize_target, apply_cleaning)
            all_data[data['battery_name']] = data

            msg = f"Loaded {data['battery_name']}: Train={data['n_train']}, Test={data['n_test']}"

            # 如果启用清洗，显示清洗统计
            if apply_cleaning and data['cleaning_stats'] is not None:
                stats = data['cleaning_stats']
                total_removed += stats['total_removed']
                total_original += stats['original_size']
                msg += f" | Cleaned: {stats['total_removed']} removed ({stats['removal_rate']:.1f}%)"

            print(msg)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    # 打印总体清洗统计
    if apply_cleaning and total_original > 0:
        overall_rate = (total_removed / total_original) * 100
        print(f"\n[3-Sigma清洗总计] 原始: {total_original}, 删除: {total_removed}, 删除率: {overall_rate:.2f}%")

    return all_data



class HUSTBatteryDataset(Dataset):
    """
    修正版 Dataset: 支持滑动窗口生成时间序列
    """
    def __init__(self, features, capacity, window_size=10):
        """
        Args:
            features: (N, 16) 原始特征
            capacity: (N, ) 目标容量
            window_size: 时间窗口大小 (例如看过去10个周期)
        """
        self.window_size = window_size
        self.X, self.y = self._create_sequences(features, capacity, window_size)

    def _create_sequences(self, features, capacity, window_size):
        X_seq, y_seq = [], []
        # 从第 window_size 个数据开始，因为需要回头看 window_size 个历史数据
        for i in range(window_size, len(features)):
            # 取过去 window_size 个时间步作为输入
            # 形状: (window_size, 16)
            X_seq.append(features[i-window_size:i]) 
            # 取当前时间步作为预测目标
            y_seq.append(capacity[i])
            
        return torch.FloatTensor(np.array(X_seq)), torch.FloatTensor(np.array(y_seq)).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class HUSTBatteryDatasetWithMetadata(Dataset):
    """
    支持物理约束的 Dataset: 返回 battery_id 和 cycle_idx

    用于在 Many-to-One 模型中应用物理约束

    Supports three sampling modes:
    1. Standard mode: Single sample (siamese_mode=False, triplet_mode=False)
    2. Pairwise mode: Paired samples (siamese_mode=True, triplet_mode=False)
    3. Triplet mode: Three samples (siamese_mode=False, triplet_mode=True)

    IMPORTANT: Pairwise/Triplet modes are ONLY for training/validation.
               For testing, always use mode='test' to ensure single-sample inference.
    """
    def __init__(self, X, y, battery_ids, cycle_indices, siamese_mode=False, triplet_mode=False, step_k=1, mode='train'):
        """
        Args:
            X: (N, window_size, feature_dim) 输入窗口
            y: (N,) 目标值
            battery_ids: (N,) 电池名称数组
            cycle_indices: (N,) cycle 索引数组
            siamese_mode: (bool) 是否启用孪生采样模式 (default=False)
            triplet_mode: (bool) 是否启用三元组采样模式 (default=False)
            step_k: (int) 配对步长 (default=1, 相邻样本)
            mode: (str) 'train', 'val', or 'test' - forces single-sample for test
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1) if y.ndim == 1 else torch.FloatTensor(y)
        self.battery_ids = battery_ids
        self.cycle_indices = torch.LongTensor(cycle_indices)

        # Mode and sampling settings
        self.mode = mode
        self.siamese_mode = siamese_mode
        self.triplet_mode = triplet_mode
        self.step_k = step_k

        # Validate: cannot enable both siamese and triplet
        if self.siamese_mode and self.triplet_mode:
            raise ValueError("Cannot enable both siamese_mode and triplet_mode simultaneously")

        # CRITICAL: Force single-sample mode for testing (no data leakage!)
        if self.mode == 'test':
            self.siamese_mode = False
            self.triplet_mode = False
            if siamese_mode or triplet_mode:
                print(f"[WARNING] Test mode detected: disabling pairwise/triplet mode to prevent data leakage")

        # Build valid indices based on mode
        if self.triplet_mode:
            self.valid_indices = self._build_valid_triplets()
        elif self.siamese_mode:
            self.valid_indices = self._build_valid_pairs()
        else:
            # Standard mode: all indices are valid
            self.valid_indices = list(range(len(self.X)))

    def _build_valid_pairs(self):
        """
        Build list of valid indices for Siamese mode.
        Only keep samples that have a valid next sample from the SAME battery.
        """
        valid_indices = []

        for idx in range(len(self.X) - self.step_k):
            # Check if next sample is from same battery
            if self.battery_ids[idx] == self.battery_ids[idx + self.step_k]:
                # Check if cycles are exactly step_k apart
                cycle_diff = self.cycle_indices[idx + self.step_k] - self.cycle_indices[idx]
                if cycle_diff == self.step_k:
                    valid_indices.append(idx)

        return valid_indices

    def _build_valid_triplets(self):
        """
        Build list of valid indices for Triplet mode.
        Only keep samples that have TWO valid next samples from the SAME battery.
        Returns samples that can form (t, t+k, t+2k) triplets.
        """
        valid_indices = []

        for idx in range(len(self.X) - 2 * self.step_k):
            # Check if both next samples are from same battery
            if (self.battery_ids[idx] == self.battery_ids[idx + self.step_k] and
                self.battery_ids[idx] == self.battery_ids[idx + 2 * self.step_k]):

                # Check if cycles are exactly step_k apart for both transitions
                cycle_diff_1 = self.cycle_indices[idx + self.step_k] - self.cycle_indices[idx]
                cycle_diff_2 = self.cycle_indices[idx + 2 * self.step_k] - self.cycle_indices[idx + self.step_k]

                if cycle_diff_1 == self.step_k and cycle_diff_2 == self.step_k:
                    valid_indices.append(idx)

        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Returns sample(s) based on mode.

        Standard mode (siamese_mode=False, triplet_mode=False):
            Returns single sample dict

        Pairwise mode (siamese_mode=True, triplet_mode=False):
            Returns paired samples dict (t, t+k)

        Triplet mode (siamese_mode=False, triplet_mode=True):
            Returns triplet samples dict (t, t+k, t+2k)
        """
        real_idx = self.valid_indices[idx]

        if self.triplet_mode:
            # Triplet mode: return three samples
            idx_2 = real_idx + self.step_k
            idx_3 = real_idx + 2 * self.step_k

            return {
                'x_1': self.X[real_idx],
                'x_2': self.X[idx_2],
                'x_3': self.X[idx_3],
                'y_1': self.y[real_idx],
                'y_2': self.y[idx_2],
                'y_3': self.y[idx_3],
                'cycle_index': self.cycle_indices[real_idx],
                'battery_id': self.battery_ids[real_idx]
            }
        elif self.siamese_mode:
            # Pairwise mode: return paired samples
            next_idx = real_idx + self.step_k

            return {
                'x_t': self.X[real_idx],
                'x_next': self.X[next_idx],
                'y_t': self.y[real_idx],
                'y_next': self.y[next_idx],
                'cycle_index': self.cycle_indices[real_idx],
                'battery_id': self.battery_ids[real_idx]
            }
        else:
            # Standard mode: return single sample
            return {
                'window': self.X[real_idx],
                'target_soh': self.y[real_idx],
                'battery_id': self.battery_ids[real_idx],
                'cycle_idx': self.cycle_indices[real_idx]
            }


# 更新 DataLoader 创建函数
def create_hust_dataloaders(data_dict, batch_size=64, window_size=10):
    """
    注意：增加了 window_size 参数
    """
    # 训练集需要滑动窗口
    train_dataset = HUSTBatteryDataset(
        data_dict['train_features'],
        data_dict['train_capacity'],
        window_size=window_size
    )

    # 测试集也需要同样的滑动窗口处理
    test_dataset = HUSTBatteryDataset(
        data_dict['test_features'],
        data_dict['test_capacity'],
        window_size=window_size
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
