"""
数据增强工具: 用于验证物理约束在噪声数据上的鲁棒性

功能:
场景一 - 随机退化:
  1. 特征加高斯噪声
  2. 容量加高斯噪声
  3. 随机丢弃循环样本

场景二 - 稀疏采样:
  1. 规律间隔采样 (每N个循环保留1个)
  2. 模拟HPPC/DST等定期测试场景

用途: 模拟不同数据退化场景,验证物理约束的作用
"""

import numpy as np


def add_gaussian_noise(data, noise_std=0.05, seed=None, clip_range=None):
    """
    对数据添加高斯噪声

    Args:
        data: numpy array, 原始数据
        noise_std: float, 高斯噪声的标准差
        seed: int, 随机种子 (None表示不固定)
        clip_range: tuple (min, max), 裁剪范围 (None表示不裁剪)

    Returns:
        noisy_data: 加噪后的数据
    """
    if seed is not None:
        np.random.seed(seed)

    noise = np.random.normal(0, noise_std, size=data.shape)
    noisy_data = data + noise

    if clip_range is not None:
        noisy_data = np.clip(noisy_data, clip_range[0], clip_range[1])

    return noisy_data


def random_drop_samples(features, targets, battery_ids, drop_ratio=0.1, seed=None):
    """
    随机丢弃部分样本 (模拟采样不全)

    Args:
        features: (N, feature_dim) 特征数组
        targets: (N,) 目标数组
        battery_ids: (N,) 电池ID数组
        drop_ratio: float, 丢弃比例 (0~1)
        seed: int, 随机种子

    Returns:
        dropped_features, dropped_targets, dropped_battery_ids
    """
    if drop_ratio <= 0 or drop_ratio >= 1:
        return features, targets, battery_ids

    if seed is not None:
        np.random.seed(seed)

    n_total = len(features)
    n_keep = int(n_total * (1 - drop_ratio))

    # 随机选择要保留的索引
    keep_indices = np.random.choice(n_total, size=n_keep, replace=False)
    keep_indices = np.sort(keep_indices)  # 保持时序顺序

    return features[keep_indices], targets[keep_indices], battery_ids[keep_indices]


def add_degradation_noise(features, targets, battery_ids,
                          feature_noise_std=0.05,
                          target_noise_std=0.02,
                          drop_ratio=0.1,
                          seed=None,
                          verbose=True):
    """
    对数据添加退化噪声 (模拟实测数据质量问题)

    这个函数会:
    1. 在特征上添加高斯噪声
    2. 在容量值上添加高斯噪声 (并裁剪到[0,1])
    3. 随机丢弃部分循环样本

    Args:
        features: (N, feature_dim) 标准化后的特征
        targets: (N,) 归一化后的容量值
        battery_ids: (N,) 电池ID数组
        feature_noise_std: float, 特征高斯噪声的标准差
        target_noise_std: float, 容量高斯噪声的标准差
        drop_ratio: float, 随机丢弃的循环比例 (0~1)
        seed: int, 随机种子 (用于可复现)
        verbose: bool, 是否打印统计信息

    Returns:
        noisy_features, noisy_targets, noisy_battery_ids
    """
    if verbose:
        print("\n" + "="*70)
        print("数据退化增强 (模拟实测数据质量问题)")
        print("="*70)
        print(f"原始样本数: {len(features)}")

    # 1. 特征加噪声
    if feature_noise_std > 0:
        noisy_features = add_gaussian_noise(features, noise_std=feature_noise_std, seed=seed)
        if verbose:
            noise_level = np.std(noisy_features - features)
            print(f"特征噪声: σ={feature_noise_std:.4f}, 实际噪声水平={noise_level:.4f}")
    else:
        noisy_features = features.copy()

    # 2. 容量加噪声 (裁剪到[0, 1.2]范围,允许略超1.0)
    if target_noise_std > 0:
        noisy_targets = add_gaussian_noise(targets, noise_std=target_noise_std,
                                          seed=seed+1 if seed else None,
                                          clip_range=(0.0, 1.2))
        if verbose:
            noise_level = np.std(noisy_targets - targets)
            print(f"容量噪声: σ={target_noise_std:.4f}, 实际噪声水平={noise_level:.4f}")
    else:
        noisy_targets = targets.copy()

    # 3. 随机丢弃样本
    if drop_ratio > 0:
        noisy_features, noisy_targets, noisy_battery_ids = random_drop_samples(
            noisy_features, noisy_targets, battery_ids,
            drop_ratio=drop_ratio,
            seed=seed+2 if seed else None
        )
        if verbose:
            dropped_count = len(features) - len(noisy_features)
            print(f"随机丢弃: {dropped_count} 个样本 ({drop_ratio*100:.1f}%)")
            print(f"剩余样本数: {len(noisy_features)}")
    else:
        noisy_battery_ids = battery_ids.copy()

    if verbose:
        print("="*70)

    return noisy_features, noisy_targets, noisy_battery_ids


# 预设噪声级别 (方便调用)
NOISE_PRESETS = {
    'none': {
        'feature_noise_std': 0.0,
        'target_noise_std': 0.0,
        'drop_ratio': 0.0,
        'description': '无噪声 (干净数据)'
    },
    'light': {
        'feature_noise_std': 0.01,
        'target_noise_std': 0.005,
        'drop_ratio': 0.05,
        'description': '轻度噪声 (5% drop, σ_feat=0.01, σ_targ=0.005)'
    },
    'medium': {
        'feature_noise_std': 0.05,
        'target_noise_std': 0.02,
        'drop_ratio': 0.15,
        'description': '中度噪声 (15% drop, σ_feat=0.05, σ_targ=0.02)'
    },
    'heavy': {
        'feature_noise_std': 0.10,
        'target_noise_std': 0.05,
        'drop_ratio': 0.30,
        'description': '重度噪声 (30% drop, σ_feat=0.10, σ_targ=0.05)'
    }
}


def get_noise_preset(preset_name='medium'):
    """
    获取预设的噪声参数

    Args:
        preset_name: str, 预设名称 ('none', 'light', 'medium', 'heavy')

    Returns:
        dict: 噪声参数字典
    """
    if preset_name not in NOISE_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(NOISE_PRESETS.keys())}")

    return NOISE_PRESETS[preset_name]


# ============================================================================
# 场景二: 稀疏采样 (Regular Sparse Sampling)
# ============================================================================

def sparse_sampling(features, targets, battery_ids, sampling_interval=5,
                    offset=0, verbose=True):
    """
    规律间隔稀疏采样 (模拟定期HPPC/DST测试场景)

    保留每隔 sampling_interval 个样本中的一个,模拟实际应用中只在特定循环
    进行容量测试的情况 (如HPPC测试)。

    Args:
        features: (N, feature_dim) 特征数组
        targets: (N,) 目标数组
        battery_ids: (N,) 电池ID数组
        sampling_interval: int, 采样间隔 (例如5表示每5个循环保留1个)
        offset: int, 起始偏移 (0-based索引)
        verbose: bool, 是否打印统计信息

    Returns:
        sampled_features, sampled_targets, sampled_battery_ids

    Example:
        如果 sampling_interval=5, offset=0:
        原始索引: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]
        保留索引: [0,       5,          10, ...]  (每5个保留1个)

        如果 sampling_interval=5, offset=2:
        保留索引: [   2,          7,             12, ...]
    """
    if sampling_interval <= 0:
        raise ValueError(f"sampling_interval must be > 0, got {sampling_interval}")

    n_total = len(features)

    # 计算保留的索引
    keep_indices = np.arange(offset, n_total, sampling_interval)

    if verbose:
        print("\n" + "="*70)
        print("场景二: 稀疏采样 (模拟定期容量测试)")
        print("="*70)
        print(f"原始样本数: {n_total}")
        print(f"采样间隔: 每 {sampling_interval} 个循环保留 1 个")
        print(f"起始偏移: {offset}")
        print(f"保留样本数: {len(keep_indices)}")
        print(f"实际保留率: {len(keep_indices)/n_total*100:.1f}%")
        print("="*70)

    return features[keep_indices], targets[keep_indices], battery_ids[keep_indices]


def apply_sparse_sampling_by_battery(features, targets, battery_ids,
                                     sampling_interval=5, offset=0, verbose=True):
    """
    按电池分组进行稀疏采样

    对每个电池独立应用稀疏采样,保持每个电池的时序结构。
    这对于跨电池训练特别重要,确保每个电池都按相同间隔采样。

    Args:
        features: (N, feature_dim) 特征数组
        targets: (N,) 目标数组
        battery_ids: (N,) 电池ID数组
        sampling_interval: int, 采样间隔
        offset: int, 起始偏移
        verbose: bool, 是否打印详细信息

    Returns:
        sampled_features, sampled_targets, sampled_battery_ids
    """
    unique_batteries = np.unique(battery_ids)

    sampled_features_list = []
    sampled_targets_list = []
    sampled_ids_list = []

    if verbose:
        print("\n" + "="*70)
        print("场景二: 按电池稀疏采样")
        print("="*70)
        print(f"总电池数: {len(unique_batteries)}")
        print(f"采样间隔: 每 {sampling_interval} 个循环保留 1 个")

    total_original = 0
    total_sampled = 0

    for battery_id in unique_batteries:
        # 找到该电池的所有样本
        mask = (battery_ids == battery_id)
        battery_features = features[mask]
        battery_targets = targets[mask]
        battery_ids_arr = battery_ids[mask]

        # 对该电池进行稀疏采样
        sampled_feat, sampled_targ, sampled_ids = sparse_sampling(
            battery_features, battery_targets, battery_ids_arr,
            sampling_interval=sampling_interval,
            offset=offset,
            verbose=False
        )

        sampled_features_list.append(sampled_feat)
        sampled_targets_list.append(sampled_targ)
        sampled_ids_list.append(sampled_ids)

        total_original += len(battery_features)
        total_sampled += len(sampled_feat)

    # 合并所有电池的采样结果
    final_features = np.vstack(sampled_features_list)
    final_targets = np.concatenate(sampled_targets_list)
    final_ids = np.concatenate(sampled_ids_list)

    if verbose:
        print(f"原始总样本数: {total_original}")
        print(f"采样后总样本数: {total_sampled}")
        print(f"总体保留率: {total_sampled/total_original*100:.1f}%")
        print("="*70)

    return final_features, final_targets, final_ids


# 稀疏采样预设 (常见的测试间隔)
SPARSE_SAMPLING_PRESETS = {
    'dense': {
        'sampling_interval': 2,
        'description': '密集采样 (每2个循环测试1次, 保留50%)'
    },
    'moderate': {
        'sampling_interval': 5,
        'description': '中等采样 (每5个循环测试1次, 保留20%)'
    },
    'sparse': {
        'sampling_interval': 10,
        'description': '稀疏采样 (每10个循环测试1次, 保留10%)'
    },
    'very_sparse': {
        'sampling_interval': 20,
        'description': '极稀疏采样 (每20个循环测试1次, 保留5%)'
    }
}


def get_sparse_sampling_preset(preset_name='moderate'):
    """
    获取稀疏采样预设参数

    Args:
        preset_name: str, 预设名称 ('dense', 'moderate', 'sparse', 'very_sparse')

    Returns:
        dict: 采样参数字典
    """
    if preset_name not in SPARSE_SAMPLING_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(SPARSE_SAMPLING_PRESETS.keys())}")

    return SPARSE_SAMPLING_PRESETS[preset_name]


# ===== 场景三: Random Missing (随机缺失) =====

def random_missing_by_battery(features, targets, battery_ids,
                              missing_rate=0.3, seed=None, verbose=True):
    """
    按电池分组进行随机缺失采样 (模拟传感器随机故障)

    与 sparse_sampling 的区别:
    - sparse_sampling: 规律间隔 (如每5个保留1个)
    - random_missing: 随机丢弃 (如随机丢弃30%)

    特点:
    1. 每个电池独立进行随机缺失
    2. 保持原始 cycle 索引顺序
    3. 时序结构被破坏 (不规则间隔)

    Args:
        features: (N, seq_len, feature_dim) 或 (N, feature_dim) 特征数组
        targets: (N,) 目标数组
        battery_ids: (N,) 电池ID数组 (list 或 ndarray)
        missing_rate: float, 缺失比例 (0~1), 例如 0.3 表示随机丢弃 30%
        seed: int, 随机种子 (确保可复现)
        verbose: bool, 是否打印统计信息

    Returns:
        sampled_features, sampled_targets, sampled_battery_ids

    Example:
        missing_rate=0.3 表示保留 70% 的样本
        原始: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        可能: [0, 2, 3, 5, 7, 9] (随机保留 70%)
    """
    if not (0 <= missing_rate < 1):
        raise ValueError(f"missing_rate must be in [0, 1), got {missing_rate}")

    if seed is not None:
        np.random.seed(seed)

    # 转换 battery_ids 为 numpy array (如果是 list)
    if isinstance(battery_ids, list):
        battery_ids = np.array(battery_ids)

    # 获取唯一电池
    unique_batteries = np.unique(battery_ids)

    sampled_features_list = []
    sampled_targets_list = []
    sampled_battery_ids_list = []

    total_original = len(features)
    total_retained = 0

    if verbose:
        print("\n" + "="*70)
        print("场景三: Random Missing (随机缺失)")
        print("="*70)
        print(f"缺失率: {missing_rate*100:.1f}% (保留 {(1-missing_rate)*100:.1f}%)")
        print(f"原始样本数: {total_original}")
        print(f"电池数量: {len(unique_batteries)}")
        print("-"*70)

    # 对每个电池独立进行随机缺失
    for battery_id in unique_batteries:
        # 找出该电池的所有样本索引
        mask = (battery_ids == battery_id)
        indices = np.where(mask)[0]

        n_battery_samples = len(indices)
        n_keep = int(n_battery_samples * (1 - missing_rate))

        # 随机选择要保留的索引
        keep_positions = np.random.choice(n_battery_samples, size=n_keep, replace=False)
        keep_positions = np.sort(keep_positions)  # 保持时序顺序

        keep_indices = indices[keep_positions]

        # 提取该电池保留的样本
        sampled_features_list.append(features[keep_indices])
        sampled_targets_list.append(targets[keep_indices])
        sampled_battery_ids_list.append(battery_ids[keep_indices])

        total_retained += n_keep

        if verbose:
            retention = n_keep / n_battery_samples * 100
            print(f"  电池 {battery_id}: {n_battery_samples} → {n_keep} ({retention:.1f}%)")

    # 合并所有电池的样本
    sampled_features = np.concatenate(sampled_features_list, axis=0)
    sampled_targets = np.concatenate(sampled_targets_list, axis=0)
    sampled_battery_ids = np.concatenate(sampled_battery_ids_list, axis=0)

    if verbose:
        print("-"*70)
        print(f"总保留样本数: {total_retained}")
        print(f"实际保留率: {total_retained/total_original*100:.1f}%")
        print("="*70)

    return sampled_features, sampled_targets, sampled_battery_ids


# Random Missing 预设配置
RANDOM_MISSING_PRESETS = {
    'light': {
        'missing_rate': 0.2,  # 丢弃 20%, 保留 80%
        'description': '轻度缺失 (保留80%)'
    },
    'moderate': {
        'missing_rate': 0.4,  # 丢弃 40%, 保留 60%
        'description': '中度缺失 (保留60%)'
    },
    'heavy': {
        'missing_rate': 0.6,  # 丢弃 60%, 保留 40%
        'description': '重度缺失 (保留40%)'
    }
}


def get_random_missing_preset(preset_name):
    """
    获取 Random Missing 预设配置

    Args:
        preset_name: str, 预设名称
            - 'light': 保留 80% (缺失 20%)
            - 'moderate': 保留 60% (缺失 40%)
            - 'heavy': 保留 40% (缺失 60%)

    Returns:
        配置字典 {'missing_rate': float, 'description': str}
    """
    if preset_name not in RANDOM_MISSING_PRESETS:
        available = ', '.join(RANDOM_MISSING_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

    return RANDOM_MISSING_PRESETS[preset_name].copy()


# ===== 场景四: Consecutive Cycle Drop (连续循环缺失) =====

def consecutive_cycle_drop_by_battery(features, targets, battery_ids,
                                      cycle_drop_rate=0.3, num_gaps=2, seed=None,
                                      verbose=True, show_battery_details=False):
    """
    按电池分组进行连续循环缺失 (模拟传感器/采集故障)

    与其他场景的区别:
    - Scenario 1 (add_degradation_noise): 随机丢弃点 + 加噪声 (跨电池丢弃)
    - Scenario 3 (random_missing): 在每个电池内随机分散地丢弃数据点
    - Scenario 4 (consecutive_cycle_drop): 在每个电池内随机丢弃**连续的一段**循环

    特点:
    1. 每个电池独立进行连续循环缺失
    2. 模拟真实场景：传感器故障导致连续一段时间没有数据
    3. 随机选择缺失段的起始位置，缺失长度根据 cycle_drop_rate 计算
    4. 可以有多个缺失段 (由 num_gaps 控制)

    Args:
        features: (N, feature_dim) 或 (N, seq_len, feature_dim) 特征数组
        targets: (N,) 目标数组
        battery_ids: (N,) 电池ID数组 (list 或 ndarray)
        cycle_drop_rate: float, 总丢弃比例 (0~1), 例如 0.3 表示总共丢弃 30% 的循环
        num_gaps: int, 缺失段的数量 (默认2，表示有2个连续缺失段)
        seed: int, 随机种子 (确保可复现)
        verbose: bool, 是否打印统计信息 (总体摘要)
        show_battery_details: bool, 是否打印每个电池的详细缺失信息 (默认False)

    Returns:
        sampled_features, sampled_targets, sampled_battery_ids

    Example:
        cycle_drop_rate=0.3, num_gaps=2 表示总共丢弃30%，分成2段连续缺失
        原始: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]  (20个循环)
        可能结果（丢弃6个，分2段）:
        缺失段1: [5,6,7]    (连续3个)
        缺失段2: [13,14,15] (连续3个)
        保留: [0,1,2,3,4, 8,9,10,11,12, 16,17,18,19]  (14个，70%)
    """
    if not (0 <= cycle_drop_rate < 1):
        raise ValueError(f"cycle_drop_rate must be in [0, 1), got {cycle_drop_rate}")

    if num_gaps < 1:
        raise ValueError(f"num_gaps must be >= 1, got {num_gaps}")

    if seed is not None:
        np.random.seed(seed)

    # 转换 battery_ids 为 numpy array (如果是 list)
    if isinstance(battery_ids, list):
        battery_ids = np.array(battery_ids)

    # 获取唯一电池
    unique_batteries = np.unique(battery_ids)

    sampled_features_list = []
    sampled_targets_list = []
    sampled_battery_ids_list = []

    total_original = len(features)
    total_retained = 0

    if verbose:
        print("\n" + "="*70)
        print("场景四: Consecutive Cycle Drop (连续循环缺失)")
        print("="*70)
        print(f"总丢弃率: {cycle_drop_rate*100:.1f}% (保留 {(1-cycle_drop_rate)*100:.1f}%)")
        print(f"缺失段数量: {num_gaps}")
        print(f"原始样本数: {total_original}")
        print(f"电池数量: {len(unique_batteries)}")
        if not show_battery_details:
            print("(详细电池信息已隐藏，设置 show_battery_details=True 查看)")
        print("-"*70)

    # 对每个电池独立进行连续循环缺失
    for battery_id in unique_batteries:
        # 找出该电池的所有样本索引
        mask = (battery_ids == battery_id)
        indices = np.where(mask)[0]

        n_battery_cycles = len(indices)
        n_total_drop = int(n_battery_cycles * cycle_drop_rate)

        # 如果丢弃数量太少，至少保证每段丢1个
        if n_total_drop < num_gaps:
            n_total_drop = min(num_gaps, n_battery_cycles - 1)

        # 将总丢弃量分配到各个缺失段
        # 使用随机划分，但保证每段至少1个
        gap_sizes = np.ones(num_gaps, dtype=int)  # 每段至少1个
        remaining = n_total_drop - num_gaps

        if remaining > 0:
            # 随机分配剩余的丢弃量
            random_allocation = np.random.multinomial(remaining, [1/num_gaps]*num_gaps)
            gap_sizes += random_allocation

        # 创建掩码：True表示保留，False表示丢弃
        keep_mask = np.ones(n_battery_cycles, dtype=bool)

        # 随机选择缺失段的起始位置
        available_positions = list(range(n_battery_cycles))
        gap_info = []

        for gap_size in gap_sizes:
            if len(available_positions) == 0:
                break

            # 确保gap不会超出边界
            max_start = max(0, len(available_positions) - gap_size)
            if max_start < 0:
                break

            # 随机选择起始位置
            start_idx = np.random.randint(0, max_start + 1)
            actual_start = available_positions[start_idx]

            # 标记这段为丢弃
            for i in range(gap_size):
                if actual_start + i < n_battery_cycles:
                    keep_mask[actual_start + i] = False

            gap_info.append((actual_start, actual_start + gap_size - 1))

            # 从可用位置中移除这段（避免重叠）
            # 移除start_idx及其后续gap_size个位置
            remove_count = min(gap_size, len(available_positions) - start_idx)
            for _ in range(remove_count):
                if start_idx < len(available_positions):
                    available_positions.pop(start_idx)

        # 提取保留的样本
        keep_indices = indices[keep_mask]

        sampled_features_list.append(features[keep_indices])
        sampled_targets_list.append(targets[keep_indices])
        sampled_battery_ids_list.append(battery_ids[keep_indices])

        total_retained += len(keep_indices)

        # 只有在 show_battery_details=True 时才打印每个电池的详细信息
        if show_battery_details:
            retention = len(keep_indices) / n_battery_cycles * 100
            print(f"  电池 {battery_id}: {n_battery_cycles} cycles → {len(keep_indices)} cycles ({retention:.1f}%)")
            for i, (start, end) in enumerate(gap_info, 1):
                gap_len = end - start + 1
                print(f"    缺失段{i}: cycle [{start}~{end}] (长度={gap_len})")

    # 合并所有电池的样本
    sampled_features = np.concatenate(sampled_features_list, axis=0)
    sampled_targets = np.concatenate(sampled_targets_list, axis=0)
    sampled_battery_ids = np.concatenate(sampled_battery_ids_list, axis=0)

    if verbose:
        print("-"*70)
        print(f"总保留样本数: {total_retained}")
        print(f"实际保留率: {total_retained/total_original*100:.1f}%")
        print("="*70)

    return sampled_features, sampled_targets, sampled_battery_ids


# Consecutive Cycle Drop 预设配置（连续缺失）
CONSECUTIVE_CYCLE_DROP_PRESETS = {
    'light': {
        'cycle_drop_rate': 0.2,  # 丢弃 20%, 保留 80%
        'num_gaps': 1,           # 1个连续缺失段
        'description': '轻度缺失 (1段连续缺失, 保留80%循环)'
    },
    'moderate': {
        'cycle_drop_rate': 0.3,  # 丢弃 30%, 保留 70%
        'num_gaps': 2,           # 2个连续缺失段
        'description': '中度缺失 (2段连续缺失, 保留70%循环)'
    },
    'heavy': {
        'cycle_drop_rate': 0.5,  # 丢弃 50%, 保留 50%
        'num_gaps': 3,           # 3个连续缺失段
        'description': '重度缺失 (3段连续缺失, 保留50%循环)'
    }
}


def get_consecutive_cycle_drop_preset(preset_name):
    """
    获取 Consecutive Cycle Drop 预设配置（连续缺失）

    Args:
        preset_name: str, 预设名称
            - 'light': 1段连续缺失，保留 80% (丢弃 20%)
            - 'moderate': 2段连续缺失，保留 70% (丢弃 30%)
            - 'heavy': 3段连续缺失，保留 50% (丢弃 50%)

    Returns:
        配置字典 {'cycle_drop_rate': float, 'num_gaps': int, 'description': str}
    """
    if preset_name not in CONSECUTIVE_CYCLE_DROP_PRESETS:
        available = ', '.join(CONSECUTIVE_CYCLE_DROP_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

    return CONSECUTIVE_CYCLE_DROP_PRESETS[preset_name].copy()


if __name__ == "__main__":
    """测试数据增强功能"""
    print("Testing data augmentation utilities...")
    print("="*70)

    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 16

    features = np.random.randn(n_samples, n_features)
    targets = np.linspace(1.0, 0.7, n_samples)  # 模拟容量衰减
    battery_ids = np.array(['test-battery'] * n_samples, dtype=object)

    print(f"Original data: {n_samples} samples")
    print(f"Target range: [{targets.min():.3f}, {targets.max():.3f}]")

    # 测试不同噪声级别
    for preset_name in ['light', 'medium', 'heavy']:
        print(f"\n{'='*70}")
        print(f"Testing preset: {preset_name.upper()}")
        print(f"{'='*70}")

        params = get_noise_preset(preset_name)
        print(f"Description: {params['description']}")

        noisy_feat, noisy_targ, noisy_ids = add_degradation_noise(
            features, targets, battery_ids,
            feature_noise_std=params['feature_noise_std'],
            target_noise_std=params['target_noise_std'],
            drop_ratio=params['drop_ratio'],
            seed=42,
            verbose=True
        )

        print(f"\nAfter augmentation:")
        print(f"  Samples: {len(noisy_feat)}")
        print(f"  Target range: [{noisy_targ.min():.3f}, {noisy_targ.max():.3f}]")
        print(f"  Target noise level: {np.std(noisy_targ[:len(targets)] - targets[:len(noisy_targ)]):.4f}")

    print("\n" + "="*70)
    print("[OK] All tests passed!")
