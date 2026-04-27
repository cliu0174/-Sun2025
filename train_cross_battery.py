"""
跨电池训练脚本：使用所有77组HUST电池数据。

数据划分：train/val/test = 6:2:2
- 训练集: 46个电池 (60%)
- 验证集: 16个电池 (20%)
- 测试集: 15个电池 (20%)
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 导入模型工厂和物理约束
from models import (ModelFactory, ConfigLoader, UnifiedModelWrapper,
                    PhysicsConstrainedLoss, SiamesePhysicsLoss, TripletPhysicsLoss,
                    AdaptivePhysicsLoss)
from models.modules.mc_dropout import MCDropout, mc_predict
from data_loaders import load_single_hust_battery
from data_loaders.data_loader_hust import apply_windowing_with_metadata, HUSTBatteryDatasetWithMetadata
import time


def _has_mc_dropout(model: nn.Module) -> bool:
    """检测模型是否含 MCDropout 层（用于在验证/测试时选择推理策略）。"""
    return any(isinstance(m, MCDropout) for m in model.modules())


def custom_collate_fn(batch):
    """
    自定义 collate function，处理带元数据的 batch（支持标准/孪生/三元组模式）

    注意：此函数必须在模块级别定义，以支持Windows的多进程DataLoader
    """
    # 检查数据格式
    if 'window' in batch[0]:
        # 标准模式
        windows = torch.stack([item['window'] for item in batch])
        targets = torch.cat([item['target_soh'].unsqueeze(0) for item in batch], dim=0)
        battery_ids = [item['battery_id'] for item in batch]
        cycle_indices = torch.stack([item['cycle_idx'] for item in batch])
        # 部分监督 mask（向后兼容：如果 Dataset 没有 is_labeled 字段，默认全 True）
        if 'is_labeled' in batch[0]:
            is_labeled = torch.stack([item['is_labeled'] for item in batch])
        else:
            is_labeled = torch.ones(len(batch), dtype=torch.bool)
        return {
            'window': windows,
            'target_soh': targets,
            'battery_id': battery_ids,
            'cycle_idx': cycle_indices,
            'is_labeled': is_labeled
        }
    elif 'x_1' in batch[0]:
        # 三元组模式
        x_1 = torch.stack([item['x_1'] for item in batch])
        x_2 = torch.stack([item['x_2'] for item in batch])
        x_3 = torch.stack([item['x_3'] for item in batch])
        y_1 = torch.cat([item['y_1'].unsqueeze(0) for item in batch], dim=0)
        y_2 = torch.cat([item['y_2'].unsqueeze(0) for item in batch], dim=0)
        y_3 = torch.cat([item['y_3'].unsqueeze(0) for item in batch], dim=0)
        battery_ids = [item['battery_id'] for item in batch]
        cycle_indices = torch.stack([item['cycle_index'] for item in batch])
        return {
            'x_1': x_1,
            'x_2': x_2,
            'x_3': x_3,
            'y_1': y_1,
            'y_2': y_2,
            'y_3': y_3,
            'battery_id': battery_ids,
            'cycle_idx': cycle_indices
        }
    else:
        # 孪生模式
        x_t = torch.stack([item['x_t'] for item in batch])
        x_next = torch.stack([item['x_next'] for item in batch])
        y_t = torch.cat([item['y_t'].unsqueeze(0) for item in batch], dim=0)
        y_next = torch.cat([item['y_next'].unsqueeze(0) for item in batch], dim=0)
        battery_ids = [item['battery_id'] for item in batch]
        cycle_indices = torch.stack([item['cycle_index'] for item in batch])
        return {
            'x_t': x_t,
            'x_next': x_next,
            'y_t': y_t,
            'y_next': y_next,
            'battery_id': battery_ids,
            'cycle_idx': cycle_indices
        }


def safe_savefig(fig_or_plt, filepath, **kwargs):
    """
    安全保存图片，如果文件被占用则使用带时间戳的文件名

    Args:
        fig_or_plt: matplotlib figure 或 plt 模块
        filepath: 目标文件路径
        **kwargs: 传递给 savefig 的其他参数

    Returns:
        actual_path: 实际保存的文件路径
    """
    try:
        fig_or_plt.savefig(filepath, **kwargs)
        return filepath
    except PermissionError:
        # 文件被占用，添加时间戳创建新文件名
        dir_name = os.path.dirname(filepath)
        base_name = os.path.basename(filepath)
        name, ext = os.path.splitext(base_name)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        new_filepath = os.path.join(dir_name, f"{name}_{timestamp}{ext}")
        fig_or_plt.savefig(new_filepath, **kwargs)
        print(f"  [WARN] 原文件被占用，已保存为新文件")
        return new_filepath


def set_seed(seed=42):
    """设置随机种子确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_all_batteries(data_dir='data/HUST data', apply_cleaning=False):
    """
    加载所有77组HUST电池数据。

    Args:
        data_dir: 数据目录
        apply_cleaning: 是否应用3-Sigma清洗（默认False，保持向后兼容）

    Returns:
        battery_names: 电池名称列表
        all_data: 所有电池的数据字典
    """
    print("="*70)
    print("加载所有HUST电池数据")
    print("="*70)

    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    battery_names = [f.replace('.csv', '') for f in csv_files]

    print(f"找到 {len(battery_names)} 组电池数据")

    # 统计清洗效果
    total_removed = 0
    total_original = 0

    all_data = {}
    for battery_name in tqdm(battery_names, desc="加载数据"):
        file_path = os.path.join(data_dir, f'{battery_name}.csv')
        try:
            data = load_single_hust_battery(file_path, train_ratio=1.0, normalize_target=True, apply_cleaning=apply_cleaning)
            all_data[battery_name] = data

            # 统计清洗效果
            if apply_cleaning and data.get('cleaning_stats'):
                stats = data['cleaning_stats']
                total_removed += stats['total_removed']
                total_original += stats['original_size']

        except Exception as e:
            print(f"警告: 加载 {battery_name} 失败: {e}")

    print(f"成功加载 {len(all_data)} 组电池")

    # 打印清洗统计
    if apply_cleaning and total_original > 0:
        overall_rate = (total_removed / total_original) * 100
        print(f"[3-Sigma清洗] 原始样本: {total_original}, 删除: {total_removed}, 删除率: {overall_rate:.2f}%")

    return battery_names, all_data


def split_batteries(battery_names, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    将电池划分为 train/val/test。

    Args:
        battery_names: 电池名称列表
        train_ratio: 训练集比例 (默认0.6)
        val_ratio: 验证集比例 (默认0.2)
        test_ratio: 测试集比例 (默认0.2)
        seed: 随机种子

    Returns:
        train_batteries, val_batteries, test_batteries: 三个电池名称列表
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 打乱电池顺序
    shuffled_batteries = battery_names.copy()
    random.shuffle(shuffled_batteries)

    n_total = len(shuffled_batteries)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_batteries = shuffled_batteries[:n_train]
    val_batteries = shuffled_batteries[n_train:n_train + n_val]
    test_batteries = shuffled_batteries[n_train + n_val:]

    print("\n" + "="*70)
    print("数据集划分")
    print("="*70)
    print(f"总电池数: {n_total}")
    print(f"训练集: {len(train_batteries)} 个电池 ({len(train_batteries)/n_total*100:.1f}%)")
    print(f"验证集: {len(val_batteries)} 个电池 ({len(val_batteries)/n_total*100:.1f}%)")
    print(f"测试集: {len(test_batteries)} 个电池 ({len(test_batteries)/n_total*100:.1f}%)")

    print(f"\n训练集电池: {train_batteries[:5]}... (显示前5个)")
    print(f"验证集电池: {val_batteries[:5]}... (显示前5个)")
    print(f"测试集电池: {test_batteries[:5]}... (显示前5个)")

    return train_batteries, val_batteries, test_batteries


def generate_supervision_mask(battery_ids, ratio, seed):
    """
    按电池随机生成部分监督 mask（label masking 方式）

    保留所有样本，但对每块电池随机选择 ratio 比例的循环标记为"有标签"。
    未标记的样本仍然参与物理约束计算，但不参与 MSE 监督。

    Args:
        battery_ids: (N,) array/list of battery ids
        ratio: float in (0, 1], 有监督样本的比例
            1.0 = 完全监督（所有样本都有标签）
            0.5 = 部分监督（一半样本有标签）
        seed: int, 随机种子（保证可复现）

    Returns:
        mask: (N,) bool numpy array, True=有标签, False=无标签
    """
    assert 0.0 < ratio <= 1.0, f"supervision_ratio must be in (0, 1], got {ratio}"

    battery_ids = np.asarray(battery_ids)
    n = len(battery_ids)
    mask = np.zeros(n, dtype=bool)

    # 全监督快速路径
    if ratio >= 1.0 - 1e-9:
        mask[:] = True
        return mask

    rng = np.random.RandomState(seed)

    # 按电池独立采样，确保每块电池都保留 ~ratio 比例的标签
    for battery_name in np.unique(battery_ids):
        battery_indices = np.where(battery_ids == battery_name)[0]
        n_battery = len(battery_indices)
        n_labeled = max(1, int(round(n_battery * ratio)))  # 至少保留 1 个标签

        # 在该电池内部随机选择 n_labeled 个作为有标签
        labeled_indices = rng.choice(battery_indices, size=n_labeled, replace=False)
        mask[labeled_indices] = True

    return mask


def prepare_cross_battery_data(all_data, train_batteries, val_batteries, test_batteries,
                               degradation_scenario='none',
                               noise_level='medium',
                               sparse_sampling_level='moderate',
                               sparse_sampling_interval=None,
                               random_missing_level='moderate',
                               random_missing_rate=None,
                               cycle_drop_level='moderate',
                               cycle_drop_rate=None,
                               cycle_drop_num_gaps=None,
                               seed=42,
                               supervision_ratio=1.0,
                               supervision_seed=None):
    """
    准备跨电池的训练/验证/测试数据。

    Args:
        all_data: 所有电池的数据字典
        train_batteries: 训练集电池列表
        val_batteries: 验证集电池列表
        test_batteries: 测试集电池列表
        degradation_scenario: str, 数据退化场景选择
            - 'none': 无退化 (干净数据)
            - 'scenario1': 场景一 - 随机噪声+丢弃
            - 'scenario2': 场景二 - 规律稀疏采样 (Uniform Subsampling)
            - 'scenario3': 场景三 - 随机缺失 (Random Missing)
            - 'scenario4': 场景四 - 连续循环缺失 (Consecutive Cycle Drop)
        noise_level: str, 场景一的噪声级别 ('light', 'medium', 'heavy')
        sparse_sampling_level: str, 场景二的采样级别 ('dense', 'moderate', 'sparse', 'very_sparse')
                              如果 sparse_sampling_interval 不为 None，则忽略此参数
        sparse_sampling_interval: int or None, 场景二的手动间隔设置
                                 如果设置，则直接使用此值，忽略 sparse_sampling_level
                                 例如: interval=3 表示每3个循环保留1个
        random_missing_level: str, 场景三的缺失级别 ('light', 'moderate', 'heavy')
                             如果 random_missing_rate 不为 None，则忽略此参数
        random_missing_rate: float or None, 场景三的手动缺失率设置
                            如果设置，则直接使用此值，忽略 random_missing_level
                            例如: rate=0.4 表示随机丢弃40%数据
        cycle_drop_level: str, 场景四的丢弃级别 ('light', 'moderate', 'heavy')
                         如果 cycle_drop_rate/num_gaps 为 None，则使用此预设级别
        cycle_drop_rate: float or None, 场景四的手动丢弃率设置
                        如果设置，则忽略 cycle_drop_level 的 rate 部分
                        例如: rate=0.3 表示随机丢弃30%循环
        cycle_drop_num_gaps: int or None, 场景四的手动缺失段数量
                            如果设置，则忽略 cycle_drop_level 的 num_gaps 部分
                            例如: num_gaps=2 表示分成2段连续缺失
        seed: int, 随机种子 (确保可复现)

    Returns:
        data_dict: 包含train/val/test的数据字典
    """
    print("\n" + "="*70)
    print("合并数据集")
    print("="*70)

    def merge_batteries(battery_list):
        """合并多个电池的数据，同时记录 battery_id"""
        all_features = []
        all_targets = []
        all_battery_ids = []

        for battery_name in battery_list:
            data = all_data[battery_name]
            features = data['train_features']
            targets = data['train_capacity']

            all_features.append(features)
            all_targets.append(targets)
            # 为每个样本记录来源电池
            all_battery_ids.extend([battery_name] * len(features))

        features = np.vstack(all_features)
        targets = np.concatenate(all_targets)
        battery_ids = np.array(all_battery_ids, dtype=object)

        return features, targets, battery_ids

    # 合并各个集合
    train_features, train_targets, train_battery_ids = merge_batteries(train_batteries)
    val_features, val_targets, val_battery_ids = merge_batteries(val_batteries)
    test_features, test_targets, test_battery_ids = merge_batteries(test_batteries)

    print(f"训练集: {train_features.shape[0]} 个样本")
    print(f"验证集: {val_features.shape[0]} 个样本")
    print(f"测试集: {test_features.shape[0]} 个样本")
    print(f"特征维度: {train_features.shape[1]}")

    # 使用训练集的统计量进行标准化
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    test_features_scaled = scaler.transform(test_features)

    # 数据退化场景应用 (训练集和验证集, 测试集保持干净)
    if degradation_scenario == 'scenario1':
        # 场景一: 随机噪声 + 随机丢弃
        from utils.data_augmentation import add_degradation_noise, get_noise_preset

        noise_params = get_noise_preset(noise_level)
        print(f"\n{'='*70}")
        print(f"场景一: 随机退化 - {noise_params['description']}")
        print(f"{'='*70}")

        # 训练集加噪
        print("\n对训练集添加噪声:")
        train_features_scaled, train_targets, train_battery_ids = add_degradation_noise(
            train_features_scaled, train_targets, train_battery_ids,
            feature_noise_std=noise_params['feature_noise_std'],
            target_noise_std=noise_params['target_noise_std'],
            drop_ratio=noise_params['drop_ratio'],
            seed=seed,
            verbose=True
        )

        # 验证集加噪 (使用不同的种子,避免与训练集完全相同)
        print("\n对验证集添加噪声:")
        val_features_scaled, val_targets, val_battery_ids = add_degradation_noise(
            val_features_scaled, val_targets, val_battery_ids,
            feature_noise_std=noise_params['feature_noise_std'],
            target_noise_std=noise_params['target_noise_std'],
            drop_ratio=noise_params['drop_ratio'],
            seed=seed + 1000,  # 不同的种子
            verbose=True
        )

        print(f"\n{'='*70}")
        print("[NOTE] 测试集保持干净 (用于公平对比)")
        print(f"{'='*70}")

    elif degradation_scenario == 'scenario2':
        # 场景二: 规律稀疏采样
        from utils.data_augmentation import (
            apply_sparse_sampling_by_battery,
            get_sparse_sampling_preset
        )

        # 优先使用手动设置的间隔，否则使用预设级别
        if sparse_sampling_interval is not None:
            sampling_interval = sparse_sampling_interval
            retention_rate = 100.0 / sampling_interval
            description = f"手动间隔 (每{sampling_interval}个循环保留1个, 保留率≈{retention_rate:.1f}%)"
            print(f"\n{'='*70}")
            print(f"场景二: 稀疏采样 - {description}")
            print(f"{'='*70}")
        else:
            sampling_params = get_sparse_sampling_preset(sparse_sampling_level)
            sampling_interval = sampling_params['sampling_interval']
            print(f"\n{'='*70}")
            print(f"场景二: 稀疏采样 - {sampling_params['description']}")
            print(f"{'='*70}")

        # 训练集稀疏采样
        print("\n对训练集进行稀疏采样:")
        train_features_scaled, train_targets, train_battery_ids = apply_sparse_sampling_by_battery(
            train_features_scaled, train_targets, train_battery_ids,
            sampling_interval=sampling_interval,
            offset=0,
            verbose=True
        )

        # 验证集稀疏采样
        print("\n对验证集进行稀疏采样:")
        val_features_scaled, val_targets, val_battery_ids = apply_sparse_sampling_by_battery(
            val_features_scaled, val_targets, val_battery_ids,
            sampling_interval=sampling_interval,
            offset=0,
            verbose=True
        )

        print(f"\n{'='*70}")
        print("[NOTE] 测试集保持干净 (用于公平对比)")
        print(f"{'='*70}")

    elif degradation_scenario == 'scenario3':
        # 场景三: 随机缺失 (Random Missing)
        from utils.data_augmentation import (
            random_missing_by_battery,
            get_random_missing_preset
        )

        # 优先使用手动设置的缺失率，否则使用预设级别
        if random_missing_rate is not None:
            missing_rate = random_missing_rate
            retention_rate = (1 - missing_rate) * 100
            description = f"手动缺失率 (随机丢弃{missing_rate*100:.0f}%, 保留率≈{retention_rate:.1f}%)"
            print(f"\n{'='*70}")
            print(f"场景三: 随机缺失 - {description}")
            print(f"{'='*70}")
        else:
            missing_params = get_random_missing_preset(random_missing_level)
            missing_rate = missing_params['missing_rate']
            print(f"\n{'='*70}")
            print(f"场景三: 随机缺失 - {missing_params['description']}")
            print(f"{'='*70}")

        # 训练集随机缺失
        print("\n对训练集进行随机缺失:")
        train_features_scaled, train_targets, train_battery_ids = random_missing_by_battery(
            train_features_scaled, train_targets, train_battery_ids,
            missing_rate=missing_rate,
            seed=seed,
            verbose=True
        )

        # 验证集随机缺失 (使用不同的种子，避免与训练集完全相同)
        print("\n对验证集进行随机缺失:")
        val_features_scaled, val_targets, val_battery_ids = random_missing_by_battery(
            val_features_scaled, val_targets, val_battery_ids,
            missing_rate=missing_rate,
            seed=seed + 2000,  # 不同的种子
            verbose=True
        )

        print(f"\n{'='*70}")
        print("[NOTE] 测试集保持干净 (用于公平对比)")
        print(f"{'='*70}")

    elif degradation_scenario == 'scenario4':
        # 场景四: 连续循环缺失 (Consecutive Cycle Drop)
        from utils.data_augmentation import (
            consecutive_cycle_drop_by_battery,
            get_consecutive_cycle_drop_preset
        )

        # 优先使用手动设置，否则使用预设级别
        if cycle_drop_rate is not None or cycle_drop_num_gaps is not None:
            # 手动模式：至少设置了一个参数
            preset = get_consecutive_cycle_drop_preset(cycle_drop_level)  # 先获取预设作为默认值
            drop_rate = cycle_drop_rate if cycle_drop_rate is not None else preset['cycle_drop_rate']
            num_gaps = cycle_drop_num_gaps if cycle_drop_num_gaps is not None else preset['num_gaps']

            retention_rate = (1 - drop_rate) * 100
            description = f"手动配置 (丢弃{drop_rate*100:.0f}%, {num_gaps}段连续缺失, 保留率≈{retention_rate:.1f}%)"
            print(f"\n{'='*70}")
            print(f"场景四: 连续循环缺失 - {description}")
            print(f"{'='*70}")
        else:
            # 预设模式
            preset = get_consecutive_cycle_drop_preset(cycle_drop_level)
            drop_rate = preset['cycle_drop_rate']
            num_gaps = preset['num_gaps']
            print(f"\n{'='*70}")
            print(f"场景四: 连续循环缺失 - {preset['description']}")
            print(f"{'='*70}")

        # 训练集连续循环缺失
        print("\n对训练集进行连续循环缺失:")
        train_features_scaled, train_targets, train_battery_ids = consecutive_cycle_drop_by_battery(
            train_features_scaled, train_targets, train_battery_ids,
            cycle_drop_rate=drop_rate,
            num_gaps=num_gaps,
            seed=seed,
            verbose=True
        )

        # 验证集连续循环缺失 (使用不同的种子，避免与训练集完全相同)
        print("\n对验证集进行连续循环缺失:")
        val_features_scaled, val_targets, val_battery_ids = consecutive_cycle_drop_by_battery(
            val_features_scaled, val_targets, val_battery_ids,
            cycle_drop_rate=drop_rate,
            num_gaps=num_gaps,
            seed=seed + 3000,  # 不同的种子
            verbose=True
        )

        print(f"\n{'='*70}")
        print("[NOTE] 测试集保持干净 (用于公平对比)")
        print(f"{'='*70}")

    elif degradation_scenario == 'none':
        print(f"\n{'='*70}")
        print("无数据退化 - 使用干净数据")
        print(f"{'='*70}")

    else:
        raise ValueError(f"Unknown degradation scenario: {degradation_scenario}. "
                        f"Available: 'none', 'scenario1', 'scenario2', 'scenario3', 'scenario4'")

    # ===== 部分监督 mask 生成（Label Masking 方式）=====
    # 只对训练集施加部分监督：val/test 保持完全监督用于公平评估
    if supervision_seed is None:
        supervision_seed = seed  # 默认与主 seed 一致，保证可复现

    train_supervision_mask = generate_supervision_mask(
        train_battery_ids, ratio=supervision_ratio, seed=supervision_seed
    )
    val_supervision_mask = np.ones(len(val_battery_ids), dtype=bool)  # val 完全监督
    test_supervision_mask = np.ones(len(test_battery_ids), dtype=bool)  # test 完全监督

    # 打印统计信息
    n_labeled = train_supervision_mask.sum()
    n_total = len(train_supervision_mask)
    print(f"\n{'='*70}")
    print(f"部分监督配置 (Label Masking)")
    print(f"{'='*70}")
    print(f"  监督比例 (supervision_ratio): {supervision_ratio:.2f}")
    print(f"  种子 (supervision_seed): {supervision_seed}")
    print(f"  训练集有标签样本: {n_labeled} / {n_total} ({100*n_labeled/n_total:.1f}%)")
    print(f"  验证集: 完全监督 ({len(val_supervision_mask)} 样本)")
    print(f"  测试集: 完全监督 ({len(test_supervision_mask)} 样本)")
    if supervision_ratio < 1.0:
        print(f"  [NOTE] 无标签样本仍然参与物理约束计算（单调/边界/平滑）")
        print(f"         但不参与 MSE 监督损失")

    data_dict = {
        'train_features': train_features_scaled,
        'train_targets': train_targets,
        'train_battery_ids': train_battery_ids,
        'train_supervision_mask': train_supervision_mask,
        'val_features': val_features_scaled,
        'val_targets': val_targets,
        'val_battery_ids': val_battery_ids,
        'val_supervision_mask': val_supervision_mask,
        'test_features': test_features_scaled,
        'test_targets': test_targets,
        'test_battery_ids': test_battery_ids,
        'test_supervision_mask': test_supervision_mask,
        'scaler': scaler,
        'n_features': train_features.shape[1],
        'train_batteries': train_batteries,
        'val_batteries': val_batteries,
        'test_batteries': test_batteries,
        'supervision_ratio': supervision_ratio,
        'supervision_seed': supervision_seed,
    }

    return data_dict


def create_dataloaders(data_dict, batch_size=64, window_size=1, seq2seq=False, use_physics=False, siamese_mode=False, triplet_mode=False, step_k=1):
    """
    创建PyTorch DataLoader，支持窗口化数据和物理约束。

    Args:
        data_dict: 数据字典
        batch_size: 批次大小
        window_size: 窗口大小（LSTM/GRU使用>1的值，其他模型使用1）
        seq2seq: 是否使用Seq2Seq模式（Many-to-Many）
        use_physics: 是否使用物理约束（需要 battery_id 和 cycle_idx）
        siamese_mode: 是否使用孪生采样模式（向后兼容：默认False）
        triplet_mode: 是否使用三元组采样模式（向后兼容：默认False）
        step_k: 采样的步长（默认1=相邻）
    """
    from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

    # 根据是否使用物理约束选择不同的数据处理方式
    if use_physics and window_size > 1:
        # 物理约束模式：按每个电池单独窗口化（不跨电池边界）
        print(f"  使用物理约束模式（窗口大小={window_size}，按电池单独窗口化）")

        # 为每个电池单独创建 windowed dataset
        train_datasets = []
        # 获取训练集的部分监督 mask（默认全 True = 完全监督）
        train_sup_mask_full = data_dict.get('train_supervision_mask', None)
        for battery_name in data_dict['train_batteries']:
            # 获取该电池的数据
            mask = data_dict['train_battery_ids'] == battery_name
            features = data_dict['train_features'][mask]
            targets = data_dict['train_targets'][mask]

            if len(features) < window_size:
                continue

            # 应用 windowing + metadata
            X, y, bid, cyc = apply_windowing_with_metadata(
                features, targets, window_size, battery_name, mode='many_to_one'
            )

            # 对齐 supervision mask 到窗口化之后的样本
            # Many-to-one 模式下第 i 个窗口的目标位于 features[i + window_size - 1]
            if train_sup_mask_full is not None:
                battery_sup_mask = train_sup_mask_full[mask]
                # 取窗口目标位置的 mask 作为该窗口的 is_labeled
                windowed_sup_mask = battery_sup_mask[window_size - 1:]
                assert len(windowed_sup_mask) == len(X), \
                    f"mask length mismatch: {len(windowed_sup_mask)} vs {len(X)}"
            else:
                windowed_sup_mask = None

            dataset = HUSTBatteryDatasetWithMetadata(
                X, y, bid, cyc,
                siamese_mode=siamese_mode, triplet_mode=triplet_mode, step_k=step_k, mode='train',
                supervision_mask=windowed_sup_mask
            )
            train_datasets.append(dataset)

        # 验证集
        val_datasets = []
        for battery_name in data_dict['val_batteries']:
            mask = data_dict['val_battery_ids'] == battery_name
            features = data_dict['val_features'][mask]
            targets = data_dict['val_targets'][mask]

            if len(features) < window_size:
                continue

            X, y, bid, cyc = apply_windowing_with_metadata(
                features, targets, window_size, battery_name, mode='many_to_one'
            )
            dataset = HUSTBatteryDatasetWithMetadata(X, y, bid, cyc, siamese_mode=siamese_mode, triplet_mode=triplet_mode, step_k=step_k, mode='val')
            val_datasets.append(dataset)

        # 测试集
        test_datasets = []
        for battery_name in data_dict['test_batteries']:
            mask = data_dict['test_battery_ids'] == battery_name
            features = data_dict['test_features'][mask]
            targets = data_dict['test_targets'][mask]

            if len(features) < window_size:
                continue

            X, y, bid, cyc = apply_windowing_with_metadata(
                features, targets, window_size, battery_name, mode='many_to_one'
            )
            # CRITICAL: Test set MUST use mode='test' to prevent data leakage
            dataset = HUSTBatteryDatasetWithMetadata(X, y, bid, cyc, siamese_mode=False, triplet_mode=False, step_k=step_k, mode='test')
            test_datasets.append(dataset)

        # 合并所有电池的 dataset
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
        test_dataset = ConcatDataset(test_datasets)

        # 创建 DataLoader
        # num_workers=0 保证结果完全一致（单进程）
        # 如需加速可改为 4 或 8（结果仍一致但更快）
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=0,           # 可手动改为4-8加速训练
            pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=0,
            pin_memory=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=0,
            pin_memory=False
        )

        # 物理约束模式下，电池ID在batch的元数据中，这里设为None
        test_battery_ids = None

    else:
        # 标准模式：跨电池窗口化（保持原有训练方式）
        print(f"  使用标准模式（窗口大小={window_size}，跨电池窗口化）")

        def apply_windowing(features, targets, battery_ids, window_size, seq2seq=False):
            """
            对特征进行滑动窗口处理（跨电池边界），同时追踪电池ID

            Args:
                features: 特征数组
                targets: 目标数组
                battery_ids: 电池ID数组（每个样本对应的电池编号）
                window_size: 窗口大小
                seq2seq: 是否为序列到序列模式

            Returns:
                windowed_features, windowed_targets, windowed_battery_ids
            """
            if window_size <= 1:
                return features, targets, battery_ids

            windowed_features = []
            windowed_targets = []
            windowed_battery_ids = []  # 记录每个窗口对应的电池ID

            for i in range(len(features) - window_size + 1):
                window_feat = features[i:i+window_size]
                windowed_features.append(window_feat)

                if seq2seq:
                    window_targ = targets[i:i+window_size]
                    windowed_targets.append(window_targ)
                else:
                    windowed_targets.append(targets[i+window_size-1])

                # 记录目标位置的电池ID（用于着色）
                windowed_battery_ids.append(battery_ids[i+window_size-1])

            return np.array(windowed_features), np.array(windowed_targets), np.array(windowed_battery_ids)

        # 对训练集、验证集、测试集分别进行窗口化处理（跨电池边界）
        train_feat, train_targ, _ = apply_windowing(
            data_dict['train_features'],
            data_dict['train_targets'],
            data_dict['train_battery_ids'],
            window_size,
            seq2seq
        )
        val_feat, val_targ, _ = apply_windowing(
            data_dict['val_features'],
            data_dict['val_targets'],
            data_dict['val_battery_ids'],
            window_size,
            seq2seq
        )
        # 测试集需要保留battery_ids用于着色
        test_feat, test_targ, test_battery_ids = apply_windowing(
            data_dict['test_features'],
            data_dict['test_targets'],
            data_dict['test_battery_ids'],
            window_size,
            seq2seq
        )

        # 处理目标张量形状
        if seq2seq:
            train_targ_tensor = torch.FloatTensor(train_targ).unsqueeze(-1)
            val_targ_tensor = torch.FloatTensor(val_targ).unsqueeze(-1)
            test_targ_tensor = torch.FloatTensor(test_targ).unsqueeze(-1)
            shuffle_train = False
        else:
            train_targ_tensor = torch.FloatTensor(train_targ)
            val_targ_tensor = torch.FloatTensor(val_targ)
            test_targ_tensor = torch.FloatTensor(test_targ)
            shuffle_train = True

        # 创建标准 Dataset 和 DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(train_feat),
            train_targ_tensor
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=0,           # 可手动改为4-8加速训练
            pin_memory=False
        )

        val_dataset = TensorDataset(
            torch.FloatTensor(val_feat),
            val_targ_tensor
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        test_dataset = TensorDataset(
            torch.FloatTensor(test_feat),
            test_targ_tensor
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

    return train_loader, val_loader, test_loader, test_battery_ids


def train_cross_battery_model(
    model_type='cnn',
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    device='cuda',
    seed=42,
    apply_cleaning=False,
    color_by_battery=True,              # 是否按电池着色（默认True）
    highlight_anomalies=True,            # 是否突出显示异常电池（默认True）
    degradation_scenario='none',         # 数据退化场景 ('none', 'scenario1', 'scenario2', 'scenario3', 'scenario4')
    noise_level='medium',                # 场景一噪声级别 ('light', 'medium', 'heavy')
    sparse_sampling_level='moderate',    # 场景二采样级别 ('dense', 'moderate', 'sparse', 'very_sparse')
    sparse_sampling_interval=None,       # 场景二手动间隔 (优先级高于 sparse_sampling_level)
    random_missing_level='moderate',     # 场景三缺失级别 ('light', 'moderate', 'heavy')
    random_missing_rate=None,            # 场景三手动缺失率 (优先级高于 random_missing_level)
    cycle_drop_level='moderate',         # 场景四丢弃级别 ('light', 'moderate', 'heavy')
    cycle_drop_rate=None,                # 场景四手动丢弃率
    cycle_drop_num_gaps=None,            # 场景四手动缺失段数量
    supervision_ratio=1.0,               # 部分监督比例 (1.0=全监督, 0.5=50%有标签)
    supervision_seed=None,               # 部分监督 mask 种子 (None=与主 seed 一致)
    config_override=None,                # dict，深度合并覆盖 config（用于超参扫描）
):
    """
    跨电池训练模型。

    Args:
        model_type: 模型类型 ('fnn', 'cnn', 'lstm', 'gru', 'mlp', 'rescnn')
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        device: 计算设备
        seed: 随机种子
        apply_cleaning: 是否应用3-Sigma数据清洗（默认False，保持向后兼容）
        color_by_battery: 是否按电池着色（默认True）
        highlight_anomalies: 是否突出显示异常电池（默认True）
        degradation_scenario: 数据退化场景 ('none', 'scenario1', 'scenario2', 'scenario3', 'scenario4')
        noise_level: 场景一噪声级别 ('light', 'medium', 'heavy')
        sparse_sampling_level: 场景二采样级别 ('dense', 'moderate', 'sparse', 'very_sparse')
        sparse_sampling_interval: 场景二手动间隔 (如果设置，则忽略 sparse_sampling_level)
                                 例如: interval=3 表示每3个循环保留1个
        random_missing_level: 场景三缺失级别 ('light', 'moderate', 'heavy')
        random_missing_rate: 场景三手动缺失率 (如果设置，则忽略 random_missing_level)
                            例如: rate=0.4 表示随机丢弃40%数据
        cycle_drop_level: 场景四丢弃级别 ('light', 'moderate', 'heavy')
        cycle_drop_rate: 场景四手动丢弃率 (如果设置，则忽略 cycle_drop_level 的 rate 部分)
                        例如: rate=0.3 表示随机丢弃30%循环
        cycle_drop_num_gaps: 场景四手动缺失段数量 (如果设置，则忽略 cycle_drop_level 的 num_gaps 部分)
                            例如: num_gaps=2 表示分成2段连续缺失
    """
    set_seed(seed)

    print("\n" + "="*70)
    print(f"跨电池训练: {model_type.upper()}")
    print(f"数据划分: Train/Val/Test = {train_ratio*100:.0f}%/{val_ratio*100:.0f}%/{test_ratio*100:.0f}%")
    if apply_cleaning:
        print("数据清洗: 启用 (3-Sigma)")

    # 显示退化场景信息
    if degradation_scenario == 'scenario1':
        from utils.data_augmentation import NOISE_PRESETS
        noise_desc = NOISE_PRESETS[noise_level]['description']
        print(f"数据退化: 场景一 ({noise_desc})")
    elif degradation_scenario == 'scenario2':
        from utils.data_augmentation import SPARSE_SAMPLING_PRESETS
        if sparse_sampling_interval is not None:
            retention_rate = 100.0 / sparse_sampling_interval
            print(f"数据退化: 场景二 (手动间隔={sparse_sampling_interval}, 保留率≈{retention_rate:.1f}%)")
        else:
            sampling_desc = SPARSE_SAMPLING_PRESETS[sparse_sampling_level]['description']
            print(f"数据退化: 场景二 ({sampling_desc})")
    elif degradation_scenario == 'scenario3':
        from utils.data_augmentation import RANDOM_MISSING_PRESETS
        if random_missing_rate is not None:
            retention_rate = (1 - random_missing_rate) * 100
            print(f"数据退化: 场景三 (手动缺失率={random_missing_rate*100:.0f}%, 保留率≈{retention_rate:.1f}%)")
        else:
            missing_desc = RANDOM_MISSING_PRESETS[random_missing_level]['description']
            print(f"数据退化: 场景三 ({missing_desc})")
    elif degradation_scenario == 'scenario4':
        from utils.data_augmentation import CONSECUTIVE_CYCLE_DROP_PRESETS
        if cycle_drop_rate is not None or cycle_drop_num_gaps is not None:
            preset = CONSECUTIVE_CYCLE_DROP_PRESETS[cycle_drop_level]
            drop_rate = cycle_drop_rate if cycle_drop_rate is not None else preset['cycle_drop_rate']
            num_gaps = cycle_drop_num_gaps if cycle_drop_num_gaps is not None else preset['num_gaps']
            retention_rate = (1 - drop_rate) * 100
            print(f"数据退化: 场景四 (手动配置: 丢弃{drop_rate*100:.0f}%, {num_gaps}段连续缺失, 保留率≈{retention_rate:.1f}%)")
        else:
            drop_desc = CONSECUTIVE_CYCLE_DROP_PRESETS[cycle_drop_level]['description']
            print(f"数据退化: 场景四 ({drop_desc})")
    else:
        print("数据退化: 无 (干净数据)")

    print("="*70)

    # 1. 加载所有电池数据
    battery_names, all_data = load_all_batteries(apply_cleaning=apply_cleaning)

    # 2. 划分数据集
    train_batteries, val_batteries, test_batteries = split_batteries(
        battery_names,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    # 3. 准备数据
    data_dict = prepare_cross_battery_data(
        all_data, train_batteries, val_batteries, test_batteries,
        degradation_scenario=degradation_scenario,
        noise_level=noise_level,
        sparse_sampling_level=sparse_sampling_level,
        sparse_sampling_interval=sparse_sampling_interval,
        random_missing_level=random_missing_level,
        random_missing_rate=random_missing_rate,
        cycle_drop_level=cycle_drop_level,
        cycle_drop_rate=cycle_drop_rate,
        cycle_drop_num_gaps=cycle_drop_num_gaps,
        seed=seed,
        supervision_ratio=supervision_ratio,
        supervision_seed=supervision_seed
    )

    # 4. 加载模型配置
    config = ConfigLoader.load_model_config(model_type)

    # 深度合并 config_override（用于超参扫描，只覆盖指定字段）
    if config_override:
        def _deep_merge(base: dict, override: dict):
            for k, v in override.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    _deep_merge(base[k], v)
                else:
                    base[k] = v
        _deep_merge(config, config_override)

    print("\n" + "="*70)
    print("模型配置")
    print("="*70)
    print(f"模型类型: {config['model_type']}")
    print(f"隐藏层配置: {config['architecture'].get('hidden_sizes', config['architecture'])}")
    print(f"学习率: {config['training']['learning_rate']}")
    print(f"批次大小: {config['training']['batch_size']}")
    print(f"训练轮数: {config['training']['num_epochs']}")

    # 5. 检查是否启用物理约束
    physics_config = config.get('physics_constraints', {})
    use_physics = physics_config.get('enabled', False)

    if use_physics:
        print("\n" + "="*70)
        print("物理约束配置")
        print("="*70)
        print(f"启用物理约束: True")
        print(f"  基础损失权重: {physics_config.get('base_loss_weight', 1.0)}")
        print(f"  单调性权重: {physics_config.get('monotonic_weight', 0.1)}")
        print(f"  边界权重: {physics_config.get('boundary_weight', 0.05)}")
        print(f"  平滑性权重: {physics_config.get('smoothness_weight', 0.0)}")
        print(f"  单调性容忍度: {physics_config.get('monotonic_tolerance', 0.01)}")
        print(f"  时间衰减启用: {physics_config.get('temporal_decay', {}).get('enabled', True)}")
        print(f"  时间衰减最大步长: {physics_config.get('temporal_decay', {}).get('max_step', 20)}")
        print(f"  时间衰减类型: {physics_config.get('temporal_decay', {}).get('decay_type', 'exp')}")
        print(f"  时间衰减系数: {physics_config.get('temporal_decay', {}).get('decay_alpha', 0.2)}")

        # 读取孪生采样配置（向后兼容：默认关闭）
        siamese_config = physics_config.get('siamese_sampling', {})
        siamese_mode = siamese_config.get('enabled', False)

        # 读取三元组采样配置（向后兼容：默认关闭）
        triplet_config = physics_config.get('triplet_sampling', {})
        triplet_mode = triplet_config.get('enabled', False)

        # 根据启用的模式读取对应的参数
        if triplet_mode:
            split_threshold = triplet_config.get('split_threshold', 200)
            step_k = triplet_config.get('step_k', 1)
        elif siamese_mode:
            split_threshold = siamese_config.get('split_threshold', 300)
            step_k = siamese_config.get('step_k', 1)
        else:
            split_threshold = 300
            step_k = 1

        # Validate: cannot enable both siamese and triplet
        if siamese_mode and triplet_mode:
            raise ValueError("Cannot enable both siamese_sampling and triplet_sampling simultaneously")

        if triplet_mode:
            print(f"\n  三元组采样: 启用")
            print(f"    分段阈值: {split_threshold} cycles")
            print(f"    配对步长: {step_k}")
            print(f"    曲率权重: {physics_config.get('curvature_weight', 0.1)}")
        elif siamese_mode:
            print(f"\n  孪生采样: 启用")
            print(f"    分段阈值: {split_threshold} cycles")
            print(f"    配对步长: {step_k}")
    else:
        print("\n物理约束: 未启用")
        siamese_mode = False
        triplet_mode = False
        split_threshold = 300
        step_k = 1

    # 6. 创建数据加载器（根据模型类型设置窗口大小和Seq2Seq模式）
    # 检测是否为Seq2Seq模型（从配置文件中的model_type判断）
    config_model_type = config['model_type'].lower()
    is_seq2seq = 'seq2seq' in config_model_type

    # 时序模型和CNN混合模型需要窗口化数据
    needs_window = any(model_name in config_model_type for model_name in ['lstm', 'gru', 'cnn'])
    if needs_window or is_seq2seq:
        window_size = config.get('data', {}).get('window_size', 10)
        if is_seq2seq:
            print(f"\n模型 {config_model_type.upper()} 使用 Seq2Seq (Many-to-Many)，窗口大小: {window_size}")
        else:
            print(f"\n模型 {config_model_type.upper()} 使用窗口化数据 (Many-to-One)，窗口大小: {window_size}")
    else:
        window_size = 1
        print(f"\n模型 {config_model_type.upper()} 使用平坦特征（window_size=1）")

    train_loader, val_loader, test_loader, test_battery_ids = create_dataloaders(
        data_dict,
        batch_size=config['training']['batch_size'],
        window_size=window_size,
        seq2seq=is_seq2seq,
        use_physics=use_physics,  # 传递物理约束标志
        siamese_mode=siamese_mode,  # 传递孪生采样标志（向后兼容：默认False）
        triplet_mode=triplet_mode,  # 传递三元组采样标志（向后兼容：默认False）
        step_k=step_k  # 传递配对步长
    )

    # 7. 创建模型
    print("\n创建模型...")
    wrapper = UnifiedModelWrapper(
        model_type=model_type,
        input_size=data_dict['n_features'],
        config=config,
        device=device
    )

    model = wrapper.model
    ModelFactory.print_model_info(model)

    optimizer = wrapper.get_optimizer()

    # 8. 创建损失函数（支持物理约束、孪生采样、三元组采样）
    if use_physics:
        if triplet_mode:
            # 三元组采样模式：使用 TripletPhysicsLoss
            criterion = TripletPhysicsLoss(
                base_loss_weight=physics_config.get('base_loss_weight', 1.0),
                monotonic_weight=physics_config.get('monotonic_weight', 0.1),
                curvature_weight=physics_config.get('curvature_weight', 0.1),
                boundary_weight=physics_config.get('boundary_weight', 0.05),
                split_threshold=split_threshold,
                monotonic_tolerance=physics_config.get('monotonic_tolerance', 0.0),
                verbose=False
            ).to(device)
            print(f"\n损失函数: TripletPhysicsLoss (三元组采样 + 二阶曲率约束)")
            print(f"  分段阈值: {split_threshold} cycles")
            print(f"  曲率权重: {physics_config.get('curvature_weight', 0.1)}")
        elif siamese_mode:
            # 孪生采样模式：使用 SiamesePhysicsLoss
            criterion = SiamesePhysicsLoss(
                base_loss_weight=physics_config.get('base_loss_weight', 1.0),
                monotonic_weight=physics_config.get('monotonic_weight', 0.1),
                smoothness_weight=physics_config.get('smoothness_weight', 0.01),
                boundary_weight=physics_config.get('boundary_weight', 0.05),
                split_threshold=split_threshold,
                monotonic_tolerance=physics_config.get('monotonic_tolerance', 0.0),
                verbose=False
            ).to(device)
            print(f"\n损失函数: SiamesePhysicsLoss (孪生采样 + 分段约束)")
            print(f"  分段阈值: {split_threshold} cycles")
        else:
            # 标准模式：使用 PhysicsConstrainedLoss
            _base_criterion = PhysicsConstrainedLoss(
                base_loss_weight=physics_config.get('base_loss_weight', 1.0),
                monotonic_weight=physics_config.get('monotonic_weight', 0.1),
                boundary_weight=physics_config.get('boundary_weight', 0.0),
                smoothness_weight=physics_config.get('smoothness_weight', 0.0),
                monotonic_tolerance=physics_config.get('monotonic_tolerance', 0.01),
                min_cycle=physics_config.get('min_cycle', 0),
                temporal_decay_enabled=physics_config.get('temporal_decay', {}).get('enabled', True),
                temporal_max_step=physics_config.get('temporal_decay', {}).get('max_step', 20),
                temporal_decay_type=physics_config.get('temporal_decay', {}).get('decay_type', 'exp'),
                temporal_decay_alpha=physics_config.get('temporal_decay', {}).get('decay_alpha', 0.2),
                verbose=False
            ).to(device)

            # M5：自适应损失权重（可选）
            aw_cfg = physics_config.get('adaptive_weight', {})
            if aw_cfg.get('enabled', False):
                criterion = AdaptivePhysicsLoss(
                    physics_loss=_base_criterion,
                    init_log_vars=aw_cfg.get('init_log_vars', None),
                    l2_reg=aw_cfg.get('l2_reg', 0.0),
                ).to(device)
                # log_vars 是可训练参数，需加入优化器
                optimizer.add_param_group({
                    'params': [criterion.log_vars],
                    'lr': config['training']['learning_rate'],
                })
                print("\n损失函数: AdaptivePhysicsLoss (M5 自适应权重)")
                print(f"  l2_reg={aw_cfg.get('l2_reg', 0.0)}, "
                      f"init_log_vars={aw_cfg.get('init_log_vars', [0,0,0,0])}")
            else:
                criterion = _base_criterion
                print("\n损失函数: PhysicsConstrainedLoss (物理约束)")
    else:
        criterion = wrapper.criterion
        print(f"\n损失函数: {type(criterion).__name__} (标准)")

    # 6.5 获取训练轮数并创建学习率调度器
    num_epochs = config['training']['num_epochs']

    from utils.lr_schedulers import create_scheduler
    scheduler, scheduler_type = create_scheduler(optimizer, config, num_epochs)

    if scheduler is not None:
        print("  调度器创建成功 [OK]")

    # 7. 训练模型
    print("\n" + "="*70)
    print("开始训练")
    print("="*70)

    # M6：不确定性伪标签管理器（可选）
    pseudo_manager = None
    pl_cfg = config.get('pseudo_labeling', {})
    if pl_cfg.get('enabled', False):
        from training.pseudo_labeling import PseudoLabelManager
        pseudo_manager = PseudoLabelManager(
            warmup_epochs      = pl_cfg.get('warmup_epochs', 30),
            update_every_k     = pl_cfg.get('update_every_k', 5),
            n_mc_samples       = pl_cfg.get('n_mc_samples', 50),
            threshold_percentile = pl_cfg.get('threshold_percentile', 30.0),
            max_pseudo_ratio   = pl_cfg.get('max_pseudo_ratio', 0.5),
            lambda_pseudo      = pl_cfg.get('lambda_pseudo', 1.0),
            epsilon            = pl_cfg.get('epsilon', 1e-6),
            inference_batch_size = pl_cfg.get('inference_batch_size', 512),
        )
        unlabeled_indices = PseudoLabelManager.get_unlabeled_indices(
            train_loader.dataset
        )
        print(f"\nM6 伪标签已启用: {pseudo_manager}")
        print(f"  无标签样本: {len(unlabeled_indices)}/{len(train_loader.dataset)}")
        if len(unlabeled_indices) == 0:
            print("  ⚠️  supervision_ratio=1.0，无无标签样本，M6 无效")
            pseudo_manager = None

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': []
    }

    # 使用val_mae判断最佳模型
    best_val_mae = float('inf')
    best_epoch = 0
    best_model_state = None

    # Early stopping配置
    patience = config['training']['early_stopping'].get('patience', 10)
    patience_counter = 0

    # M2：检测一次，供验证/测试推理复用
    mc_mode = _has_mc_dropout(model)
    if mc_mode:
        print("  [MCDropout] 检测到 MCDropout 层：验证用 10 次采样均值，测试用 50 次采样均值")

    for epoch in tqdm(range(num_epochs), desc="Training"):
        # M6：Warmup 结束后定期刷新伪标签
        if pseudo_manager is not None and pseudo_manager.should_update(epoch):
            pseudo_manager.update(
                model, train_loader.dataset, unlabeled_indices,
                device, collate_fn=custom_collate_fn,
            )
            model.train()   # 恢复训练模式
        # ===== 训练阶段 =====
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            # 处理不同类型的batch
            # 检查batch类型（dict表示有元数据，tuple表示无元数据）
            if isinstance(batch, dict):
                # 检查数据格式：triplet / pairwise / standard
                if 'x_1' in batch:
                    # Triplet mode: 三元组样本
                    x_1 = batch['x_1'].to(device)
                    x_2 = batch['x_2'].to(device)
                    x_3 = batch['x_3'].to(device)
                    y_1 = batch['y_1'].to(device)
                    y_2 = batch['y_2'].to(device)
                    y_3 = batch['y_3'].to(device)
                    cycle_indices = batch['cycle_idx']

                    optimizer.zero_grad()

                    # 三次前向传播
                    pred_1 = model(x_1)
                    pred_2 = model(x_2)
                    pred_3 = model(x_3)

                    # 三元组模式损失计算（TripletPhysicsLoss）
                    loss = criterion(pred_1, pred_2, pred_3, y_1, y_2, y_3, cycle_indices)

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * x_1.size(0)

                elif 'x_t' in batch:
                    # Pairwise mode: 配对样本
                    x_t = batch['x_t'].to(device)
                    x_next = batch['x_next'].to(device)
                    y_t = batch['y_t'].to(device)
                    y_next = batch['y_next'].to(device)
                    battery_ids = batch['battery_id']
                    cycle_indices = batch['cycle_idx']

                    optimizer.zero_grad()

                    # 两次前向传播
                    pred_t = model(x_t)
                    pred_next = model(x_next)

                    # 孪生模式损失计算（SiamesePhysicsLoss）
                    loss = criterion(pred_t, pred_next, y_t, y_next, cycle_indices)

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * x_t.size(0)
                else:
                    # 标准模式：带元数据的 batch（窗口化数据）
                    features = batch['window'].to(device)
                    targets = batch['target_soh'].to(device)
                    battery_ids = batch['battery_id']
                    cycle_indices = batch['cycle_idx']
                    # 部分监督 mask（向后兼容：没有 is_labeled 则默认 None=全监督）
                    is_labeled = batch.get('is_labeled', None)
                    # 带元数据的batch: targets 已经是 (batch, 1) 形状，不需要 unsqueeze

                    optimizer.zero_grad()
                    predictions = model(features)

                    # 计算损失（传入 supervision_mask 支持部分监督）
                    if use_physics and battery_ids is not None:
                        loss = criterion(predictions, targets, battery_ids, cycle_indices,
                                         supervision_mask=is_labeled)
                    elif is_labeled is not None and not is_labeled.all():
                        # 非物理约束模式下的部分监督：手动计算 masked MSE
                        mask = is_labeled.to(predictions.device).float()
                        if mask.dim() == 1:
                            mask = mask.unsqueeze(1)
                        n_labeled = mask.sum().clamp(min=1.0)
                        loss = ((predictions - targets) ** 2 * mask).sum() / n_labeled
                    else:
                        loss = criterion(predictions, targets)

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * features.size(0)
            else:
                # 传统 tuple batch（window_size=1的情况）
                features, targets = batch
                features = features.to(device)
                targets = targets.to(device)
                battery_ids = None
                cycle_indices = None

                # 只有Many-to-One模型需要unsqueeze (标准模式)
                if not is_seq2seq:
                    targets = targets.unsqueeze(1)

                optimizer.zero_grad()
                predictions = model(features)

                # 计算损失
                loss = criterion(predictions, targets)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * features.size(0)

        train_loss /= len(train_loader.dataset)

        # M6：伪标签附加训练轮次（主训练完成后执行，不影响 train_loss 统计）
        if pseudo_manager is not None and pseudo_manager.n_pseudo > 0:
            pseudo_loader = pseudo_manager.get_pseudo_loader(
                batch_size=config['training']['batch_size']
            )
            if pseudo_loader is not None:
                model.train()
                pseudo_manager.compute_pseudo_loss(model, pseudo_loader, optimizer, device)

        # ===== 更新学习率 (在validation之前，与原始代码一致) =====
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()

        # ===== 验证阶段 =====
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_rmse = 0.0

        # 用于累积物理损失详情
        if use_physics:
            physics_loss_details = {
                'base': 0.0,
                'monotonic': 0.0,
                'boundary': 0.0,
                'smoothness': 0.0,
                'mask_active_ratio': 0.0  # 用于孪生模式
            }
            mask_batch_count = 0  # 用于计算mask平均值的batch计数

        with torch.no_grad():
            for batch in val_loader:
                # 处理不同类型的batch
                # 检查batch类型（dict表示有元数据，tuple表示无元数据）
                if isinstance(batch, dict):
                    # 检查数据格式：triplet / pairwise / standard
                    if 'x_1' in batch:
                        # Triplet mode: 三元组样本
                        x_1 = batch['x_1'].to(device)
                        x_2 = batch['x_2'].to(device)
                        x_3 = batch['x_3'].to(device)
                        y_1 = batch['y_1'].to(device)
                        y_2 = batch['y_2'].to(device)
                        y_3 = batch['y_3'].to(device)
                        cycle_indices = batch['cycle_idx']

                        # 三次前向传播
                        pred_1 = model(x_1)
                        pred_2 = model(x_2)
                        pred_3 = model(x_3)

                        # 三元组模式损失计算（TripletPhysicsLoss）
                        loss = criterion(pred_1, pred_2, pred_3, y_1, y_2, y_3, cycle_indices)

                        # 获取详细损失（TripletPhysicsLoss）
                        details = criterion.get_loss_details()
                        physics_loss_details['base'] += details['base'] * x_1.size(0)
                        physics_loss_details['monotonic'] += details['monotonic'] * x_1.size(0)
                        physics_loss_details['boundary'] += details['boundary'] * x_1.size(0)
                        # Triplet特有：曲率损失
                        if 'curvature' not in physics_loss_details:
                            physics_loss_details['curvature'] = 0.0
                        physics_loss_details['curvature'] += details['curvature'] * x_1.size(0)
                        # 累积mask比例
                        if 'mask_active_ratio' in details:
                            physics_loss_details['mask_active_ratio'] += details['mask_active_ratio']
                            mask_batch_count += 1

                        # 用于MAE/RMSE计算：取三个预测的平均
                        predictions = (pred_1 + pred_2 + pred_3) / 3.0
                        targets = (y_1 + y_2 + y_3) / 3.0

                        # 累积验证损失和MAE/RMSE
                        val_loss += loss.item() * x_1.size(0)
                        val_mae += torch.abs(predictions - targets).sum().item()
                        val_rmse += torch.sqrt(torch.mean((predictions - targets) ** 2)).item() * x_1.size(0)

                    elif 'x_t' in batch:
                        # Pairwise mode: 配对样本
                        x_t = batch['x_t'].to(device)
                        x_next = batch['x_next'].to(device)
                        y_t = batch['y_t'].to(device)
                        y_next = batch['y_next'].to(device)
                        battery_ids = batch['battery_id']
                        cycle_indices = batch['cycle_idx']

                        # 两次前向传播
                        pred_t = model(x_t)
                        pred_next = model(x_next)

                        # 孪生模式损失计算（SiamesePhysicsLoss）
                        loss = criterion(pred_t, pred_next, y_t, y_next, cycle_indices)

                        # 获取详细损失（SiamesePhysicsLoss）
                        details = criterion.get_loss_details()
                        physics_loss_details['base'] += details['base'] * x_t.size(0)
                        physics_loss_details['monotonic'] += details['monotonic'] * x_t.size(0)
                        physics_loss_details['boundary'] += details['boundary'] * x_t.size(0)
                        physics_loss_details['smoothness'] += details['smoothness'] * x_t.size(0)
                        # 累积mask比例（用于孪生模式）
                        if 'mask_active_ratio' in details:
                            physics_loss_details['mask_active_ratio'] += details['mask_active_ratio']
                            mask_batch_count += 1

                        # 计算MAE和RMSE（取两个预测的平均）
                        val_loss += loss.item() * x_t.size(0)
                        val_mae += (torch.mean(torch.abs(pred_t - y_t)).item() +
                                   torch.mean(torch.abs(pred_next - y_next)).item()) / 2 * x_t.size(0)
                        val_rmse += (torch.sqrt(torch.mean((pred_t - y_t) ** 2)).item() +
                                    torch.sqrt(torch.mean((pred_next - y_next) ** 2)).item()) / 2 * x_t.size(0)
                    else:
                        # 标准模式：带元数据的 batch（窗口化数据）
                        features = batch['window'].to(device)
                        targets = batch['target_soh'].to(device)
                        battery_ids = batch['battery_id']
                        cycle_indices = batch['cycle_idx']
                        # 带元数据的batch: targets 已经是 (batch, 1) 形状，不需要 unsqueeze

                        # M2 修复：MCDropout 在 eval 模式下仍激活，用 10 次采样均值
                        # 替代单次噪声预测，保证 early stopping 信号稳定
                        if mc_mode:
                            predictions, _ = mc_predict(model, features, n_samples=10)
                        else:
                            predictions = model(features)

                        # 计算损失
                        if use_physics and battery_ids is not None:
                            loss = criterion(predictions, targets, battery_ids, cycle_indices)
                            # 获取详细损失
                            details = criterion.get_loss_details()
                            physics_loss_details['base'] += details['base'] * features.size(0)
                            physics_loss_details['monotonic'] += details['monotonic'] * features.size(0)
                            physics_loss_details['boundary'] += details['boundary'] * features.size(0)
                            physics_loss_details['smoothness'] += details['smoothness'] * features.size(0)
                        else:
                            loss = criterion(predictions, targets)

                        val_loss += loss.item() * features.size(0)
                        val_mae += torch.mean(torch.abs(predictions - targets)).item() * features.size(0)
                        val_rmse += torch.sqrt(torch.mean((predictions - targets) ** 2)).item() * features.size(0)
                else:
                    # 传统 tuple batch（window_size=1的情况）
                    features, targets = batch
                    features = features.to(device)
                    targets = targets.to(device)
                    battery_ids = None
                    cycle_indices = None

                    # 只有Many-to-One模型需要unsqueeze (标准模式)
                    if not is_seq2seq:
                        targets = targets.unsqueeze(1)

                    predictions = model(features)

                    # 计算损失
                    loss = criterion(predictions, targets)

                    val_loss += loss.item() * features.size(0)
                    val_mae += torch.mean(torch.abs(predictions - targets)).item() * features.size(0)
                    val_rmse += torch.sqrt(torch.mean((predictions - targets) ** 2)).item() * features.size(0)

        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        val_rmse /= len(val_loader.dataset)

        # 计算平均物理损失详情
        if use_physics:
            for key in physics_loss_details:
                if key == 'mask_active_ratio' and mask_batch_count > 0:
                    # mask_active_ratio 按batch数量平均
                    physics_loss_details[key] /= mask_batch_count
                else:
                    # 其他损失按样本数量平均
                    physics_loss_details[key] /= len(val_loader.dataset)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)

        # ===== 保存最佳模型和Early Stopping =====
        patience_counter += 1  # 每个epoch递增

        if val_mae < best_val_mae:  # 基于val_mae判断
            best_val_mae = val_mae
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0  # 重置early stopping计数器

        # 每10个epoch打印一次
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            if scheduler is not None:
                print(f"  LR:         {current_lr:.6f}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Val MAE:    {val_mae*100:.4f}% (Best: {best_val_mae*100:.4f}%)")
            print(f"  Val RMSE:   {val_rmse*100:.4f}%")
            print(f"  Best Epoch: {best_epoch}")
            if config['training']['early_stopping'].get('enabled', False):
                print(f"  Patience:   {patience_counter}/{patience}")

            # 显示物理约束详细损失
            if use_physics:
                print(f"\n  物理约束详细损失:")
                print(f"    基础损失 (MSE):  {physics_loss_details['base']:.6f}")
                print(f"    单调性损失:      {physics_loss_details['monotonic']:.6f}")
                print(f"    边界损失:        {physics_loss_details['boundary']:.6f}")

                # Triplet模式显示曲率损失，Pairwise模式显示平滑性损失
                if triplet_mode and 'curvature' in physics_loss_details:
                    print(f"    曲率损失 (2阶):  {physics_loss_details['curvature']:.6f}")
                else:
                    print(f"    平滑性损失:      {physics_loss_details['smoothness']:.6f}")

                # 显示Mask激活比例（Pairwise和Triplet模式都有）
                if (siamese_mode or triplet_mode) and physics_loss_details['mask_active_ratio'] > 0:
                    print(f"    Mask激活比例:    {physics_loss_details['mask_active_ratio']*100:.1f}% (cycle >= {split_threshold})")

                # 计算加权后的贡献
                # Triplet模式使用curvature，Pairwise模式使用smoothness
                if triplet_mode and 'curvature' in physics_loss_details:
                    second_order_weight = physics_config.get('curvature_weight', 0.1)
                    second_order_loss = physics_loss_details['curvature']
                    second_order_name = "曲率"
                else:
                    second_order_weight = physics_config.get('smoothness_weight', 0.0)
                    second_order_loss = physics_loss_details['smoothness']
                    second_order_name = "平滑性"

                total_weighted = (
                    physics_config.get('base_loss_weight', 1.0) * physics_loss_details['base'] +
                    physics_config.get('monotonic_weight', 0.1) * physics_loss_details['monotonic'] +
                    physics_config.get('boundary_weight', 0.05) * physics_loss_details['boundary'] +
                    second_order_weight * second_order_loss
                )

                print(f"  加权后贡献比例:")
                if total_weighted > 0:
                    base_contrib = physics_config.get('base_loss_weight', 1.0) * physics_loss_details['base']
                    mono_contrib = physics_config.get('monotonic_weight', 0.1) * physics_loss_details['monotonic']
                    bound_contrib = physics_config.get('boundary_weight', 0.05) * physics_loss_details['boundary']
                    second_order_contrib = second_order_weight * second_order_loss

                    print(f"    基础:   {base_contrib:.6f} ({base_contrib/total_weighted*100:.1f}%)")
                    print(f"    单调性: {mono_contrib:.6f} ({mono_contrib/total_weighted*100:.1f}%)")
                    print(f"    边界:   {bound_contrib:.6f} ({bound_contrib/total_weighted*100:.1f}%)")
                    print(f"    {second_order_name}: {second_order_contrib:.6f} ({second_order_contrib/total_weighted*100:.1f}%)")

        # Early stopping检查
        if config['training']['early_stopping'].get('enabled', False):
            if patience_counter > patience:
                print(f"\n早停触发！验证集MAE连续{patience}个epoch未改善")
                print(f"最佳epoch: {best_epoch}, 最佳val_mae: {best_val_mae*100:.4f}%")
                break

    # 8. 恢复最佳模型
    print("\n" + "="*70)
    print("训练完成")
    print("="*70)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"已恢复最佳模型 (Epoch {best_epoch}, Val MAE: {best_val_mae*100:.4f}%)")

    # 9. 测试集评估
    print("\n在测试集上评估...")
    model.eval()

    all_predictions = []
    all_targets = []
    all_battery_ids_from_batch = []  # 用于物理约束模式下收集电池ID

    with torch.no_grad():
        for batch in test_loader:
            # 处理不同类型的batch
            # 检查batch类型（dict表示有元数据，tuple表示无元数据）
            # IMPORTANT: Test set always uses single-sample mode (no siamese pairs)
            #            This ensures realistic inference without data leakage
            if isinstance(batch, dict):
                # Single-sample mode with metadata: expect 'window' key (NOT 'x_t')
                features = batch['window'].to(device)
                targets = batch['target_soh'].cpu().numpy()
                battery_ids_batch = batch['battery_id']  # 提取电池ID
            else:
                # Standard mode: tuple batch (legacy support)
                features, targets = batch
                features = features.to(device)
                targets = targets.cpu().numpy()
                battery_ids_batch = None

            # M2 修复：MCDropout 模型用 50 次采样均值作为点估计，
            # 与 Baseline 的 model.eval() 干净预测保持可比性
            if mc_mode:
                predictions, _ = mc_predict(model, features, n_samples=50)
                predictions = predictions.cpu().numpy()
            else:
                predictions = model(features).cpu().numpy()

            # 处理输出形状
            if is_seq2seq:
                # Seq2Seq: (batch, seq_len, 1) -> 展平为1D
                predictions = predictions.reshape(-1)
                targets = targets.reshape(-1)
            else:
                # Many-to-One: (batch, 1) -> squeeze
                predictions = predictions.squeeze()
                targets = targets.squeeze()  # 确保 targets 也被 squeeze

            all_predictions.extend(predictions)
            all_targets.extend(targets)

            # 收集电池ID（用于着色）
            if battery_ids_batch is not None:
                all_battery_ids_from_batch.extend(battery_ids_batch)

    predictions_np = np.array(all_predictions)
    targets_np = np.array(all_targets)

    # 合并电池ID：物理约束模式从batch收集，标准模式使用test_battery_ids
    if len(all_battery_ids_from_batch) > 0:
        # 物理约束模式：从batch中收集到了电池ID
        final_battery_ids = all_battery_ids_from_batch
    elif test_battery_ids is not None:
        # 标准模式：使用预先计算的test_battery_ids
        final_battery_ids = test_battery_ids
    else:
        final_battery_ids = None

    test_mae = np.mean(np.abs(predictions_np - targets_np))
    test_rmse = np.sqrt(np.mean((predictions_np - targets_np) ** 2))
    test_mape = np.mean(np.abs((predictions_np - targets_np) / targets_np)) * 100

    # 计算R² (决定系数)
    ss_res = np.sum((targets_np - predictions_np) ** 2)  # 残差平方和
    ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)  # 总平方和
    test_r2 = 1 - (ss_res / ss_tot)

    print("\n" + "="*70)
    print("测试集结果")
    print("="*70)
    print(f"MAE:  {test_mae*100:.4f}%")
    print(f"RMSE: {test_rmse*100:.4f}%")
    print(f"MAPE: {test_mape:.4f}%")
    print(f"R2:   {test_r2:.6f}")

    # 10. 保存结果
    results_dir = f'results/cross_battery/{model_type}'
    os.makedirs(results_dir, exist_ok=True)

    # 保存模型
    checkpoint_path = os.path.join(results_dir, 'model_checkpoint.pth')
    wrapper.best_mae = best_val_mae
    wrapper.best_epoch = best_epoch
    wrapper.training_history = history
    wrapper.save_checkpoint(checkpoint_path, epoch=best_epoch)

    # 保存配置和数据划分信息
    import json
    split_info = {
        'train_batteries': train_batteries,
        'val_batteries': val_batteries,
        'test_batteries': test_batteries,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'seed': seed
    }
    with open(os.path.join(results_dir, 'battery_split.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    # 保存结果
    results = {
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_mape': float(test_mape),
        'test_r2': float(test_r2),
        'best_val_mae': float(best_val_mae),
        'best_epoch': best_epoch,
        'predictions': predictions_np.tolist(),
        'targets': targets_np.tolist(),
        'battery_ids': final_battery_ids if isinstance(final_battery_ids, (list, np.ndarray)) else None,  # 保存电池编号
        'history': history
    }

    import pickle
    with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # 绘制图表
    plot_cross_battery_results(history, predictions_np, targets_np,
                                final_battery_ids if color_by_battery else None,
                                results_dir, highlight_anomalies)

    print(f"\n所有结果已保存到: {results_dir}/")
    print("="*70)

    return wrapper, results, data_dict


def plot_cross_battery_results(history, predictions, targets, battery_ids, save_dir, highlight_anomalies=True):
    """
    绘制训练结果

    Args:
        history: 训练历史
        predictions: 预测值
        targets: 真实值
        battery_ids: 电池编号列表
        save_dir: 保存目录
        highlight_anomalies: 是否突出显示异常电池（默认True）
    """

    # 1. 训练曲线
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot([mae * 100 for mae in history['val_mae']])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (%)')
    axes[1].set_title('Validation MAE')
    axes[1].grid(True)

    axes[2].plot([rmse * 100 for rmse in history['val_rmse']])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('RMSE (%)')
    axes[2].set_title('Validation RMSE')
    axes[2].grid(True)

    plt.tight_layout()
    saved_path = safe_savefig(plt, os.path.join(save_dir, 'training_history.png'), dpi=300)
    print(f"  [OK] 训练历史已保存: {os.path.basename(saved_path)}")
    plt.close()

    # 2. 预测对比（根据电池ID着色）
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 检查是否有电池ID信息
    has_battery_ids = battery_ids is not None and len(battery_ids) > 0

    if has_battery_ids:
        # 获取唯一的电池ID
        unique_batteries = sorted(set(battery_ids))
        n_batteries = len(unique_batteries)

        # 定义异常电池列表
        problematic_batteries = ['1-5', '4-2']

        # 使用colormap为不同电池分配颜色
        import matplotlib.cm as cm
        colors = cm.get_cmap('tab20' if n_batteries <= 20 else 'hsv')(np.linspace(0, 1, n_batteries))

        # 为每个电池绘制不同颜色的点
        for i, battery_id in enumerate(unique_batteries):
            mask = np.array([bid == battery_id for bid in battery_ids])

            # 检查是否为异常电池且启用了高亮
            is_problematic = highlight_anomalies and (battery_id in problematic_batteries)

            if is_problematic:
                # 异常电池使用特殊标记：红色星形，更大，边框
                axes[0].scatter(targets[mask], predictions[mask],
                              alpha=0.8, s=50, c='red', marker='*',
                              edgecolors='darkred', linewidths=1.5,
                              label=f'{battery_id} [ANOMALY]', zorder=10)
            else:
                # 正常电池使用默认样式
                axes[0].scatter(targets[mask], predictions[mask],
                              alpha=0.6, s=10, c=[colors[i]],
                              label=f'{battery_id}' if n_batteries <= 10 else None)

        # 只在电池数量<=10时显示图例，或者启用高亮且有异常电池时显示
        show_legend = n_batteries <= 10 or (highlight_anomalies and any(b in problematic_batteries for b in unique_batteries))
        if show_legend:
            axes[0].legend(loc='best', fontsize=8, markerscale=2)
    else:
        # Standard mode: 单一颜色
        axes[0].scatter(targets, predictions, alpha=0.5, s=10)

    axes[0].plot([targets.min(), targets.max()],
                 [targets.min(), targets.max()],
                 'r--', lw=2, label='Perfect')
    axes[0].set_xlabel('True SOH')
    axes[0].set_ylabel('Predicted SOH')
    axes[0].set_title('Test Set Predictions')
    if not has_battery_ids or n_batteries > 10:
        axes[0].legend()
    axes[0].grid(True)

    errors = predictions - targets
    axes[1].hist(errors, bins=50, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Error')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Error Distribution')
    axes[1].grid(True)

    plt.tight_layout()
    saved_path = safe_savefig(plt, os.path.join(save_dir, 'predictions.png'), dpi=300)
    print(f"  [OK] 预测结果已保存: {os.path.basename(saved_path)}")
    plt.close()

    # 3. 绘制最好的3个电池的容量衰减预测曲线
    if battery_ids is not None and len(battery_ids) > 0:
        plot_top_batteries_capacity_curves(predictions, targets, battery_ids, save_dir, top_n=3)


def plot_top_batteries_capacity_curves(predictions, targets, battery_ids, save_dir, top_n=3):
    """
    绘制预测效果最好的前N个电池的容量衰减曲线

    Args:
        predictions: 预测值 (numpy array)
        targets: 真实值 (numpy array)
        battery_ids: 电池编号列表
        save_dir: 保存目录
        top_n: 显示前N个最好的电池 (默认3)
    """
    print(f"\n绘制预测效果最好的前{top_n}个电池容量衰减曲线...")

    # 计算每个电池的MAE
    unique_batteries = sorted(set(battery_ids))
    battery_errors = {}

    for battery_id in unique_batteries:
        mask = np.array([bid == battery_id for bid in battery_ids])
        battery_preds = predictions[mask]
        battery_targets = targets[mask]
        battery_mae = np.mean(np.abs(battery_preds - battery_targets))
        battery_errors[battery_id] = battery_mae

    # 按MAE排序，取前N个最好的
    sorted_batteries = sorted(battery_errors.items(), key=lambda x: x[1])
    top_batteries = [b[0] for b in sorted_batteries[:top_n]]

    print(f"  预测效果最好的{top_n}个电池:")
    for i, (battery_id, mae) in enumerate(sorted_batteries[:top_n], 1):
        print(f"    {i}. {battery_id}: MAE = {mae*100:.4f}%")

    # 创建子图
    _, axes = plt.subplots(1, top_n, figsize=(6*top_n, 5))
    if top_n == 1:
        axes = [axes]

    for idx, battery_id in enumerate(top_batteries):
        ax = axes[idx]

        # 获取该电池的预测和真实值
        mask = np.array([bid == battery_id for bid in battery_ids])
        battery_preds = predictions[mask]
        battery_targets = targets[mask]

        # 创建cycle索引 (假设按顺序)
        cycles = np.arange(len(battery_preds))

        # 绘制真实值和预测值
        ax.plot(cycles, battery_targets, 'o-', label='True SOH',
                color='steelblue', markersize=4, linewidth=2, alpha=0.7)
        ax.plot(cycles, battery_preds, 's-', label='Predicted SOH',
                color='orangered', markersize=3, linewidth=2, alpha=0.7)

        # 计算并显示指标
        mae = np.mean(np.abs(battery_preds - battery_targets))
        rmse = np.sqrt(np.mean((battery_preds - battery_targets) ** 2))
        r2 = 1 - np.sum((battery_targets - battery_preds)**2) / np.sum((battery_targets - np.mean(battery_targets))**2)

        ax.set_xlabel('Cycle Index', fontsize=11)
        ax.set_ylabel('SOH', fontsize=11)
        ax.set_title(f'Battery {battery_id}\nMAE={mae*100:.3f}%, RMSE={rmse*100:.3f}%, R²={r2:.4f}',
                    fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([battery_targets.min() - 0.05, battery_targets.max() + 0.05])

    plt.suptitle(f'Top {top_n} Best Predicted Batteries - Capacity Degradation Curves',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = os.path.join(save_dir, 'top_batteries_predictions.png')
    saved_path = safe_savefig(plt, output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] 顶部电池预测已保存: {os.path.basename(saved_path)}")
    plt.close()


if __name__ == "__main__":
    """
    使用示例：

    跨电池训练，使用77组HUST数据
    train/val/test = 6:2:2 划分
    """

    # ===== 配置参数 =====
    MODEL_TYPE = 'cnn_lstm'         # 模型类型: 'fnn', 'cnn', 'lstm', 'gru', 'bilstm', 'bigru', 'mlp', 'rescnn', 'cnn_lstm'
    TRAIN_RATIO = 0.6           # 训练集比例 (60%)
    VAL_RATIO = 0.2             # 验证集比例 (20%)
    TEST_RATIO = 0.2            # 测试集比例 (20%)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 42             # 随机种子（确保可复现）

    # ===== 部分监督配置 =====
    # 控制训练集中有多少比例的样本拥有 SOH 标签（Label Masking 方式）
    # 1.0 = 完全监督（所有样本都有标签，即 baseline-v2.2 全监督基线）
    # 0.7 / 0.5 / 0.3 = 部分监督（对应 Stage 0 的四个实验条件）
    SUPERVISION_RATIO = 1.0     # 监督比例: 1.0, 0.7, 0.5, 0.3
    SUPERVISION_SEED = None     # mask 随机种子（None=与 SEED 一致，保证可复现）

    # ===== 数据退化场景选择 (验证物理约束在不同场景下的作用) =====
    # 场景选择: 'none', 'scenario1', 'scenario2', 'scenario3', 'scenario4'
    DEGRADATION_SCENARIO = 'none'  # 'none': 无退化 (干净数据)
                                    # 'scenario1': 随机噪声+随机丢弃
                                    # 'scenario2': 规律稀疏采样 (Uniform Subsampling)
                                    # 'scenario3': 随机缺失 (Random Missing)
                                    # 'scenario4': 随机丢弃循环 (Random Cycle Drop)

    # 场景一参数 (仅当 DEGRADATION_SCENARIO='scenario1' 时生效)
    NOISE_LEVEL = 'light'           # 噪声级别: 'light', 'medium', 'heavy'
                                     # light:  5% drop, σ_feat=0.01, σ_targ=0.005
                                     # medium: 15% drop, σ_feat=0.05, σ_targ=0.02
                                     # heavy:  30% drop, σ_feat=0.10, σ_targ=0.05

    # 场景二参数 (仅当 DEGRADATION_SCENARIO='scenario2' 时生效)
    # 方式1: 使用预设级别
    SPARSE_SAMPLING_LEVEL = 'dense'  # 稀疏采样级别: 'dense', 'moderate', 'sparse', 'very_sparse'
                                         # dense:       每2个循环保留1个 (50%)
                                         # moderate:    每5个循环保留1个 (20%)
                                         # sparse:      每10个循环保留1个 (10%)
                                         # very_sparse: 每20个循环保留1个 (5%)

    # 方式2: 手动设置间隔 (如果设置，将忽略 SPARSE_SAMPLING_LEVEL)
    SPARSE_SAMPLING_INTERVAL = 3    # 手动设置采样间隔 (None=使用预设级别, 整数=手动间隔)
                                         # 例如: 3 表示每3个循环保留1个 (保留率≈33.3%)
                                         #      7 表示每7个循环保留1个 (保留率≈14.3%)
                                         #      15 表示每15个循环保留1个 (保留率≈6.7%)

    # 场景三参数 (仅当 DEGRADATION_SCENARIO='scenario3' 时生效)
    # 方式1: 使用预设级别
    RANDOM_MISSING_LEVEL = 'light'  # 随机缺失级别: 'light', 'moderate', 'heavy'
                                       # light:    20% 缺失 (保留80%)
                                       # moderate: 40% 缺失 (保留60%)
                                       # heavy:    60% 缺失 (保留40%)

    # 方式2: 手动设置缺失率 (如果设置，将忽略 RANDOM_MISSING_LEVEL)
    RANDOM_MISSING_RATE = 0.7         # 手动设置缺失率 (None=使用预设级别, 0-1之间的浮点数=手动缺失率)
                                       # 例如: 0.2 表示随机丢弃20%数据 (保留80%)
                                       #      0.35 表示随机丢弃35%数据 (保留65%)
                                       #      0.5 表示随机丢弃50%数据 (保留50%)

    # 场景四参数 (仅当 DEGRADATION_SCENARIO='scenario4' 时生效)
    # 方式1: 使用预设级别
    CYCLE_DROP_LEVEL = 'moderate'    # 连续循环缺失级别: 'light', 'moderate', 'heavy'
                                      # light:    20% 丢弃, 1段连续缺失 (保留80%循环)
                                      # moderate: 30% 丢弃, 2段连续缺失 (保留70%循环)
                                      # heavy:    50% 丢弃, 3段连续缺失 (保留50%循环)

    # 方式2: 手动设置（如果设置，将覆盖 CYCLE_DROP_LEVEL 对应参数）
    CYCLE_DROP_RATE = None            # 手动设置丢弃率 (None=使用预设级别, 0-1之间的浮点数)
                                      # 例如: 0.3 表示丢弃30%循环 (保留70%)
                                      #      0.4 表示丢弃40%循环 (保留60%)

    CYCLE_DROP_NUM_GAPS = None        # 手动设置缺失段数量 (None=使用预设级别, 整数)
                                      # 例如: 1 表示1段连续缺失
                                      #      2 表示2段连续缺失
                                      #      3 表示3段连续缺失

    # ===== 开始训练 =====
    print(f"\n使用设备: {DEVICE}\n")

    wrapper, results, data_dict = train_cross_battery_model(
        model_type=MODEL_TYPE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        device=DEVICE,
        seed=SEED,
        apply_cleaning=False,                # 是否使用3-Sigma数据清洗
        color_by_battery=True,                # 是否按电池着色（True=彩色图，False=单色图）
        highlight_anomalies=False,            # 是否突出显示异常电池（True=红色星形，False=普通显示）
        degradation_scenario=DEGRADATION_SCENARIO,  # 数据退化场景选择
        noise_level=NOISE_LEVEL,              # 场景一: 噪声级别
        sparse_sampling_level=SPARSE_SAMPLING_LEVEL,  # 场景二: 稀疏采样级别
        sparse_sampling_interval=SPARSE_SAMPLING_INTERVAL,  # 场景二: 手动间隔（优先级更高）
        random_missing_level=RANDOM_MISSING_LEVEL,  # 场景三: 随机缺失级别
        random_missing_rate=RANDOM_MISSING_RATE,    # 场景三: 手动缺失率（优先级更高）
        cycle_drop_level=CYCLE_DROP_LEVEL,    # 场景四: 连续循环缺失级别
        cycle_drop_rate=CYCLE_DROP_RATE,      # 场景四: 手动丢弃率（优先级更高）
        cycle_drop_num_gaps=CYCLE_DROP_NUM_GAPS,  # 场景四: 手动缺失段数量（优先级更高）
        supervision_ratio=SUPERVISION_RATIO,  # 部分监督比例 (1.0=全监督基线)
        supervision_seed=SUPERVISION_SEED     # mask 种子 (None=与主 seed 一致)
    )

    print("\n" + "="*70)
    print("跨电池训练完成！")
    print("="*70)
    print(f"\n最终测试集结果:")
    print(f"  MAE:  {results['test_mae']*100:.4f}%")
    print(f"  RMSE: {results['test_rmse']*100:.4f}%")
    print(f"  MAPE: {results['test_mape']:.4f}%")
    print(f"  R2:   {results['test_r2']:.6f}")
    print(f"\n训练集: {len(data_dict['train_batteries'])} 个电池")
    print(f"验证集: {len(data_dict['val_batteries'])} 个电池")
    print(f"测试集: {len(data_dict['test_batteries'])} 个电池")
