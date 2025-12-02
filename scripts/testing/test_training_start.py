"""
测试训练开始部分，检查物理约束是否被启用
"""

import torch
import numpy as np
from models import ConfigLoader, ModelFactory, UnifiedModelWrapper
from models.physics_loss import PhysicsConstrainedLoss
from data_loaders import load_single_hust_battery
from data_loaders.data_loader_hust import apply_windowing_with_metadata, HUSTBatteryDatasetWithMetadata
from torch.utils.data import DataLoader, ConcatDataset

def test_training_start():
    print("="*70)
    print("测试训练启动 - 检查物理约束")
    print("="*70)

    # 1. 加载配置
    config = ConfigLoader.load_model_config('lstm')
    physics_config = config.get('physics_constraints', {})
    use_physics = physics_config.get('enabled', False)

    print(f"\n1. 配置检查")
    print(f"   use_physics = {use_physics}")
    print(f"   monotonic_weight = {physics_config.get('monotonic_weight', 0.1)}")

    # 2. 加载一个电池数据测试
    print(f"\n2. 加载测试数据")
    data = load_single_hust_battery('data/HUST data/1-1.csv', normalize_target=True)

    features = data['train_features']
    targets = data['train_capacity']

    print(f"   features shape: {features.shape}")
    print(f"   targets shape: {targets.shape}")

    # 3. 测试 windowing
    window_size = config['data'].get('window_size', 40)
    print(f"\n3. 测试 windowing (window_size={window_size})")

    if use_physics:
        X, y, bid, cyc = apply_windowing_with_metadata(
            features, targets, window_size, '1-1', mode='many_to_one'
        )
        print(f"   [Physics Mode]")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        print(f"   battery_ids: {len(bid)} items")
        print(f"   cycle_indices: {len(cyc)} items")
        print(f"   Example battery_id: {bid[0]}")
        print(f"   Example cycle_idx: {cyc[0]}")

        # 创建 Dataset
        dataset = HUSTBatteryDatasetWithMetadata(X, y, bid, cyc)

        # 创建 collate_fn
        def custom_collate_fn(batch):
            windows = torch.stack([item['window'] for item in batch])
            targets = torch.cat([item['target_soh'].unsqueeze(0) for item in batch], dim=0)
            battery_ids = [item['battery_id'] for item in batch]
            cycle_indices = torch.stack([item['cycle_idx'] for item in batch])
            return {
                'window': windows,
                'target_soh': targets,
                'battery_id': battery_ids,
                'cycle_idx': cycle_indices
            }

        # 创建 DataLoader
        loader = DataLoader(dataset, batch_size=16, shuffle=True,
                          collate_fn=custom_collate_fn)

        # 获取一个 batch
        batch = next(iter(loader))

        print(f"\n4. 测试 DataLoader")
        print(f"   batch type: {type(batch)}")
        print(f"   batch keys: {list(batch.keys())}")
        print(f"   window shape: {batch['window'].shape}")
        print(f"   target_soh shape: {batch['target_soh'].shape}")
        print(f"   battery_ids: {len(batch['battery_id'])} items")
        print(f"   cycle_indices shape: {batch['cycle_idx'].shape}")

        # 5. 创建损失函数
        print(f"\n5. 测试损失函数")
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
            verbose=True  # 开启详细输出
        )

        print(f"   criterion type: {type(criterion).__name__}")
        print(f"   monotonic_weight: {criterion.monotonic_weight}")

        # 6. 测试损失计算
        print(f"\n6. 测试损失计算")

        # 模拟模型输出
        predictions = torch.randn(batch['window'].shape[0], 1)

        # 提取元数据
        features = batch['window']
        targets = batch['target_soh']
        battery_ids = batch['battery_id']
        cycle_indices = batch['cycle_idx']

        print(f"   predictions shape: {predictions.shape}")
        print(f"   targets shape: {targets.shape}")
        print(f"   battery_ids: {battery_ids}")
        print(f"   cycle_indices: {cycle_indices.tolist()}")

        # 计算损失
        loss = criterion(predictions, targets, battery_ids, cycle_indices)

        print(f"\n7. 损失结果")
        print(f"   total loss: {loss.item():.6f}")

        details = criterion.get_loss_details()
        print(f"\n   详细损失:")
        for key, value in details.items():
            print(f"     {key}: {value:.6f}")

        if details['monotonic'] > 0:
            print(f"\n   SUCCESS! 单调性约束正在工作")
        else:
            print(f"\n   WARNING: 单调性约束为 0，可能批次内样本不足或已满足约束")

    else:
        print(f"\n   [Standard Mode]")
        print(f"   物理约束未启用")

    print("\n" + "="*70)
    print("测试完成")
    print("="*70)


if __name__ == "__main__":
    test_training_start()
