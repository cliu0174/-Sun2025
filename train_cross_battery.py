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
from models import ModelFactory, ConfigLoader, UnifiedModelWrapper, PhysicsConstrainedLoss
from data_loaders import load_single_hust_battery
from data_loaders.data_loader_hust import apply_windowing_with_metadata, HUSTBatteryDatasetWithMetadata


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


def prepare_cross_battery_data(all_data, train_batteries, val_batteries, test_batteries):
    """
    准备跨电池的训练/验证/测试数据。

    Args:
        all_data: 所有电池的数据字典
        train_batteries: 训练集电池列表
        val_batteries: 验证集电池列表
        test_batteries: 测试集电池列表

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

    data_dict = {
        'train_features': train_features_scaled,
        'train_targets': train_targets,
        'train_battery_ids': train_battery_ids,
        'val_features': val_features_scaled,
        'val_targets': val_targets,
        'val_battery_ids': val_battery_ids,
        'test_features': test_features_scaled,
        'test_targets': test_targets,
        'test_battery_ids': test_battery_ids,
        'scaler': scaler,
        'n_features': train_features.shape[1],
        'train_batteries': train_batteries,
        'val_batteries': val_batteries,
        'test_batteries': test_batteries
    }

    return data_dict


def create_dataloaders(data_dict, batch_size=64, window_size=1, seq2seq=False, use_physics=False):
    """
    创建PyTorch DataLoader，支持窗口化数据和物理约束。

    Args:
        data_dict: 数据字典
        batch_size: 批次大小
        window_size: 窗口大小（LSTM/GRU使用>1的值，其他模型使用1）
        seq2seq: 是否使用Seq2Seq模式（Many-to-Many）
        use_physics: 是否使用物理约束（需要 battery_id 和 cycle_idx）
    """
    from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

    def custom_collate_fn(batch):
        """自定义 collate function，处理带元数据的 batch"""
        windows = torch.stack([item['window'] for item in batch])
        # 注意：target_soh 已经是 (1,) 形状，用 cat 而不是 stack 来避免额外维度
        targets = torch.cat([item['target_soh'].unsqueeze(0) for item in batch], dim=0)
        battery_ids = [item['battery_id'] for item in batch]
        cycle_indices = torch.stack([item['cycle_idx'] for item in batch])

        return {
            'window': windows,
            'target_soh': targets,
            'battery_id': battery_ids,
            'cycle_idx': cycle_indices
        }

    # 根据是否使用物理约束选择不同的数据处理方式
    if use_physics and window_size > 1:
        # 物理约束模式：按每个电池单独窗口化（不跨电池边界）
        print(f"  使用物理约束模式（窗口大小={window_size}，按电池单独窗口化）")

        # 为每个电池单独创建 windowed dataset
        train_datasets = []
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
            dataset = HUSTBatteryDatasetWithMetadata(X, y, bid, cyc)
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
            dataset = HUSTBatteryDatasetWithMetadata(X, y, bid, cyc)
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
            dataset = HUSTBatteryDatasetWithMetadata(X, y, bid, cyc)
            test_datasets.append(dataset)

        # 合并所有电池的 dataset
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
        test_dataset = ConcatDataset(test_datasets)

        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)

        val_dataset = TensorDataset(
            torch.FloatTensor(val_feat),
            val_targ_tensor
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_dataset = TensorDataset(
            torch.FloatTensor(test_feat),
            test_targ_tensor
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, test_battery_ids


def train_cross_battery_model(
    model_type='cnn',
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    device='cuda',
    seed=42,
    apply_cleaning=False,
    color_by_battery=True,      # 是否按电池着色（默认True）
    highlight_anomalies=True     # 是否突出显示异常电池（默认True）
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
    """
    set_seed(seed)

    print("\n" + "="*70)
    print(f"跨电池训练: {model_type.upper()}")
    print(f"数据划分: Train/Val/Test = {train_ratio*100:.0f}%/{val_ratio*100:.0f}%/{test_ratio*100:.0f}%")
    if apply_cleaning:
        print("数据清洗: 启用 (3-Sigma)")
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
    data_dict = prepare_cross_battery_data(all_data, train_batteries, val_batteries, test_batteries)

    # 4. 加载模型配置
    config = ConfigLoader.load_model_config(model_type)
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
    else:
        print("\n物理约束: 未启用")

    # 6. 创建数据加载器（根据模型类型设置窗口大小和Seq2Seq模式）
    # 检测是否为Seq2Seq模型（从配置文件中的model_type判断）
    config_model_type = config['model_type'].lower()
    is_seq2seq = 'seq2seq' in config_model_type

    # 时序模型需要窗口化数据
    needs_window = any(model_name in config_model_type for model_name in ['lstm', 'gru'])
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
        use_physics=use_physics  # 传递物理约束标志
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

    # 8. 创建损失函数（支持物理约束）
    if use_physics:
        criterion = PhysicsConstrainedLoss(
            base_loss_weight=physics_config.get('base_loss_weight', 1.0),
            monotonic_weight=physics_config.get('monotonic_weight', 0.1),
            boundary_weight=physics_config.get('boundary_weight', 0.05),
            smoothness_weight=physics_config.get('smoothness_weight', 0.0),
            monotonic_tolerance=physics_config.get('monotonic_tolerance', 0.01),
            temporal_decay_enabled=physics_config.get('temporal_decay', {}).get('enabled', True),
            temporal_max_step=physics_config.get('temporal_decay', {}).get('max_step', 20),
            temporal_decay_type=physics_config.get('temporal_decay', {}).get('decay_type', 'exp'),
            temporal_decay_alpha=physics_config.get('temporal_decay', {}).get('decay_alpha', 0.2),
            verbose=False
        ).to(device)
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

    for epoch in tqdm(range(num_epochs), desc="Training"):
        # ===== 训练阶段 =====
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            # 处理不同类型的batch
            # 检查batch类型（dict表示有元数据，tuple表示无元数据）
            if isinstance(batch, dict):
                # 带元数据的 batch（窗口化数据）
                features = batch['window'].to(device)
                targets = batch['target_soh'].to(device)
                battery_ids = batch['battery_id']
                cycle_indices = batch['cycle_idx']
                # 带元数据的batch: targets 已经是 (batch, 1) 形状，不需要 unsqueeze
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
            if use_physics and battery_ids is not None:
                loss = criterion(predictions, targets, battery_ids, cycle_indices)
            else:
                loss = criterion(predictions, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)

        train_loss /= len(train_loader.dataset)

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
                'smoothness': 0.0
            }

        with torch.no_grad():
            for batch in val_loader:
                # 处理不同类型的batch
                # 检查batch类型（dict表示有元数据，tuple表示无元数据）
                if isinstance(batch, dict):
                    # 带元数据的 batch（窗口化数据）
                    features = batch['window'].to(device)
                    targets = batch['target_soh'].to(device)
                    battery_ids = batch['battery_id']
                    cycle_indices = batch['cycle_idx']
                    # 带元数据的batch: targets 已经是 (batch, 1) 形状，不需要 unsqueeze
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

        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        val_rmse /= len(val_loader.dataset)

        # 计算平均物理损失详情
        if use_physics:
            for key in physics_loss_details:
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
                print(f"    平滑性损失:      {physics_loss_details['smoothness']:.6f}")

                # 计算加权后的贡献
                total_weighted = (
                    physics_config.get('base_loss_weight', 1.0) * physics_loss_details['base'] +
                    physics_config.get('monotonic_weight', 0.1) * physics_loss_details['monotonic'] +
                    physics_config.get('boundary_weight', 0.05) * physics_loss_details['boundary'] +
                    physics_config.get('smoothness_weight', 0.0) * physics_loss_details['smoothness']
                )

                print(f"  加权后贡献比例:")
                if total_weighted > 0:
                    base_contrib = physics_config.get('base_loss_weight', 1.0) * physics_loss_details['base']
                    mono_contrib = physics_config.get('monotonic_weight', 0.1) * physics_loss_details['monotonic']
                    bound_contrib = physics_config.get('boundary_weight', 0.05) * physics_loss_details['boundary']
                    smooth_contrib = physics_config.get('smoothness_weight', 0.0) * physics_loss_details['smoothness']

                    print(f"    基础:   {base_contrib:.6f} ({base_contrib/total_weighted*100:.1f}%)")
                    print(f"    单调性: {mono_contrib:.6f} ({mono_contrib/total_weighted*100:.1f}%)")
                    print(f"    边界:   {bound_contrib:.6f} ({bound_contrib/total_weighted*100:.1f}%)")
                    print(f"    平滑性: {smooth_contrib:.6f} ({smooth_contrib/total_weighted*100:.1f}%)")

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
            if isinstance(batch, dict):
                # 物理约束模式：带元数据的 batch
                features = batch['window'].to(device)
                targets = batch['target_soh'].cpu().numpy()
                battery_ids_batch = batch['battery_id']  # 提取电池ID
            else:
                # 标准模式：tuple batch
                features, targets = batch
                features = features.to(device)
                targets = targets.cpu().numpy()
                battery_ids_batch = None

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
    print(f"R²:   {test_r2:.6f}")

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
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300)
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
    plt.savefig(os.path.join(save_dir, 'predictions.png'), dpi=300)
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] 已保存: {output_path}")
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
    SEED = 42                   # 随机种子（确保可复现）

    # ===== 开始训练 =====
    print(f"\n使用设备: {DEVICE}\n")

    wrapper, results, data_dict = train_cross_battery_model(
        model_type=MODEL_TYPE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        device=DEVICE,
        seed=SEED,
        apply_cleaning=False,       # 是否使用3-Sigma数据清洗
        color_by_battery=True,       # 是否按电池着色（True=彩色图，False=单色图）
        highlight_anomalies=False     # 是否突出显示异常电池（True=红色星形，False=普通显示）
    )

    print("\n" + "="*70)
    print("跨电池训练完成！")
    print("="*70)
    print(f"\n最终测试集结果:")
    print(f"  MAE:  {results['test_mae']*100:.4f}%")
    print(f"  RMSE: {results['test_rmse']*100:.4f}%")
    print(f"  MAPE: {results['test_mape']:.4f}%")
    print(f"  R²:   {results['test_r2']:.6f}")
    print(f"\n训练集: {len(data_dict['train_batteries'])} 个电池")
    print(f"验证集: {len(data_dict['val_batteries'])} 个电池")
    print(f"测试集: {len(data_dict['test_batteries'])} 个电池")
