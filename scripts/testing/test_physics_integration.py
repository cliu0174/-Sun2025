"""
测试物理约束集成

快速验证：
1. 数据加载（windowing + metadata）
2. Dataset 创建
3. DataLoader 运行
4. 物理损失计算
"""

import torch
import numpy as np
from data_loaders.data_loader_hust import (
    load_single_hust_battery,
    apply_windowing_with_metadata,
    HUSTBatteryDatasetWithMetadata
)
from models import PhysicsConstrainedLoss


def test_windowing():
    """测试 windowing 功能"""
    print("="*70)
    print("测试 1: Windowing + Metadata")
    print("="*70)

    # 加载一个电池数据
    data = load_single_hust_battery(
        'data/HUST data/1-1.csv',
        train_ratio=0.75,
        normalize_target=True
    )

    print(f"\n电池: {data['battery_name']}")
    print(f"  训练样本数: {data['n_train']}")

    # 应用 windowing
    X, y, battery_ids, cycle_indices = apply_windowing_with_metadata(
        data['train_features'],
        data['train_capacity'],
        window_size=40,
        battery_id='1-1',
        mode='many_to_one'
    )

    print(f"\nWindowing 结果:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  battery_ids shape: {battery_ids.shape}")
    print(f"  cycle_indices shape: {cycle_indices.shape}")
    print(f"\n示例:")
    print(f"  battery_id[0]: {battery_ids[0]}")
    print(f"  cycle_idx[0]: {cycle_indices[0]}")
    print(f"  y[0]: {y[0]:.6f}")

    return X, y, battery_ids, cycle_indices


def test_dataset(X, y, battery_ids, cycle_indices):
    """测试 Dataset"""
    print("\n" + "="*70)
    print("测试 2: Dataset 创建")
    print("="*70)

    dataset = HUSTBatteryDatasetWithMetadata(X, y, battery_ids, cycle_indices)

    print(f"\nDataset 大小: {len(dataset)}")

    # 获取一个样本
    sample = dataset[0]

    print(f"\n样本结构:")
    print(f"  window shape: {sample['window'].shape}")
    print(f"  target_soh shape: {sample['target_soh'].shape}")
    print(f"  battery_id: {sample['battery_id']}")
    print(f"  cycle_idx: {sample['cycle_idx']}")

    return dataset


def test_dataloader(dataset):
    """测试 DataLoader"""
    print("\n" + "="*70)
    print("测试 3: DataLoader")
    print("="*70)

    # 自定义 collate function
    def custom_collate_fn(batch):
        windows = torch.stack([item['window'] for item in batch])
        targets = torch.stack([item['target_soh'] for item in batch])
        battery_ids = [item['battery_id'] for item in batch]
        cycle_indices = torch.stack([item['cycle_idx'] for item in batch])

        return {
            'window': windows,
            'target_soh': targets,
            'battery_id': battery_ids,
            'cycle_idx': cycle_indices
        }

    # 创建 DataLoader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,  # 关键！可以打乱
        collate_fn=custom_collate_fn
    )

    print(f"\nDataLoader 批次数: {len(dataloader)}")

    # 获取一个 batch
    batch = next(iter(dataloader))

    print(f"\nBatch 结构:")
    print(f"  window shape: {batch['window'].shape}")
    print(f"  target_soh shape: {batch['target_soh'].shape}")
    print(f"  battery_id (前5个): {batch['battery_id'][:5]}")
    print(f"  cycle_idx (前5个): {batch['cycle_idx'][:5].tolist()}")

    return batch


def test_physics_loss(batch):
    """测试物理损失计算"""
    print("\n" + "="*70)
    print("测试 4: 物理损失计算")
    print("="*70)

    # 创建物理约束损失
    criterion = PhysicsConstrainedLoss(
        base_loss_weight=1.0,
        monotonic_weight=0.1,
        boundary_weight=0.05,
        smoothness_weight=0.0,
        monotonic_tolerance=0.01,
        temporal_decay_enabled=True,
        temporal_max_step=20,
        temporal_decay_type='exp',
        temporal_decay_alpha=0.2,
        verbose=True  # 开启详细输出
    )

    # 模拟预测值（直接使用目标值）
    predictions = batch['target_soh']
    targets = batch['target_soh']
    battery_ids = batch['battery_id']
    cycle_indices = batch['cycle_idx']

    print(f"\n计算损失...")
    loss = criterion(predictions, targets, battery_ids, cycle_indices)

    print(f"\n损失值: {loss.item():.6f}")
    print(f"\n详细损失:")
    details = criterion.get_loss_details()
    for key, value in details.items():
        print(f"  {key}: {value:.6f}")

    # 测试违反物理规律的情况
    print("\n" + "-"*70)
    print("测试违反物理规律的预测:")
    print("-"*70)

    # 制造一些违反
    predictions_bad = predictions.clone()
    predictions_bad[5] = predictions_bad[5] + 0.02  # 增加 2%

    loss_bad = criterion(predictions_bad, targets, battery_ids, cycle_indices)

    print(f"\n违反损失值: {loss_bad.item():.6f}")
    print(f"\n详细损失:")
    details_bad = criterion.get_loss_details()
    for key, value in details_bad.items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("物理约束集成测试")
    print("="*70)

    try:
        # 测试 1: Windowing
        X, y, battery_ids, cycle_indices = test_windowing()
        print("\n[OK] Windowing 测试通过")

        # 测试 2: Dataset
        dataset = test_dataset(X, y, battery_ids, cycle_indices)
        print("\n[OK] Dataset 测试通过")

        # 测试 3: DataLoader
        batch = test_dataloader(dataset)
        print("\n[OK] DataLoader 测试通过")

        # 测试 4: 物理损失
        test_physics_loss(batch)
        print("\n[OK] 物理损失测试通过")

        print("\n" + "="*70)
        print("所有测试通过！")
        print("="*70)

    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
