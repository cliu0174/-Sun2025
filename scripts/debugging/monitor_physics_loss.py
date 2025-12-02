"""
监控实际训练中物理约束的作用

在训练循环中添加日志，查看：
1. 每个batch有多少同一电池的样本
2. 产生了多少配对
3. 单调性损失的实际值
"""

import torch
import numpy as np
from models import ConfigLoader
from data_loaders import load_single_hust_battery
from data_loaders.data_loader_hust import apply_windowing_with_metadata, HUSTBatteryDatasetWithMetadata
from torch.utils.data import DataLoader, ConcatDataset
from models.physics_loss import PhysicsConstrainedLoss

def analyze_batch_distribution(data_dir='data/HUST data', window_size=40, batch_size=256):
    """
    分析实际批次中的数据分布
    """
    print("="*70)
    print("分析批次数据分布")
    print("="*70)

    # 加载多个电池
    import os
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])[:10]  # 加载前10个

    all_datasets = []

    for csv_file in csv_files:
        battery_name = csv_file.replace('.csv', '')
        file_path = os.path.join(data_dir, csv_file)

        data = load_single_hust_battery(file_path, normalize_target=True)
        features = data['train_features']
        targets = data['train_capacity']

        X, y, bid, cyc = apply_windowing_with_metadata(
            features, targets, window_size, battery_name, mode='many_to_one'
        )

        dataset = HUSTBatteryDatasetWithMetadata(X, y, bid, cyc)
        all_datasets.append(dataset)

        print(f"  {battery_name}: {len(dataset)} samples")

    # 合并数据集
    combined_dataset = ConcatDataset(all_datasets)
    print(f"\n总样本数: {len(combined_dataset)}")

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
    loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True,
                       collate_fn=custom_collate_fn)

    print(f"\nbatch_size: {batch_size}")
    print(f"总batch数: {len(loader)}")

    # 创建物理约束损失函数
    config = ConfigLoader.load_model_config('lstm')
    physics_config = config.get('physics_constraints', {})

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
        verbose=False  # 关闭详细输出
    )

    # 分析前 10 个 batch
    print("\n" + "="*70)
    print("分析前10个batch")
    print("="*70)

    total_mono_loss = 0
    total_base_loss = 0
    num_batches_analyzed = 0

    for i, batch in enumerate(loader):
        if i >= 10:
            break

        battery_ids = batch['battery_id']
        cycle_indices = batch['cycle_idx']

        # 统计每个电池的样本数
        unique_batteries = set(battery_ids)
        battery_counts = {bid: battery_ids.count(bid) for bid in unique_batteries}

        print(f"\nBatch {i+1}:")
        print(f"  总样本数: {len(battery_ids)}")
        print(f"  电池数: {len(unique_batteries)}")
        print(f"  每个电池样本数: {dict(sorted(battery_counts.items(), key=lambda x: x[1], reverse=True)[:5])}")

        # 模拟预测（使用随机值）
        predictions = torch.randn(len(battery_ids), 1)
        targets = batch['target_soh']

        # 计算损失
        loss = criterion(predictions, targets, battery_ids, cycle_indices)
        details = criterion.get_loss_details()

        print(f"  单调性损失: {details['monotonic']:.6f}")
        print(f"  基础损失: {details['base']:.6f}")
        print(f"  单调性/基础 比例: {details['monotonic'] / (details['base'] + 1e-8):.4f}")

        total_mono_loss += details['monotonic']
        total_base_loss += details['base']
        num_batches_analyzed += 1

    # 统计
    avg_mono_loss = total_mono_loss / num_batches_analyzed
    avg_base_loss = total_base_loss / num_batches_analyzed

    print("\n" + "="*70)
    print("统计结果")
    print("="*70)
    print(f"平均单调性损失: {avg_mono_loss:.6f}")
    print(f"平均基础损失: {avg_base_loss:.6f}")
    print(f"比例: {avg_mono_loss / (avg_base_loss + 1e-8):.4f}")

    # 计算加权后的贡献
    mono_weight = physics_config.get('monotonic_weight', 0.1)
    base_weight = physics_config.get('base_loss_weight', 1.0)

    mono_contribution = avg_mono_loss * mono_weight
    base_contribution = avg_base_loss * base_weight

    print(f"\n加权后对总损失的贡献:")
    print(f"  单调性: {mono_contribution:.6f} ({mono_contribution / (mono_contribution + base_contribution) * 100:.2f}%)")
    print(f"  基础:   {base_contribution:.6f} ({base_contribution / (mono_contribution + base_contribution) * 100:.2f}%)")

    print(f"\n建议:")
    if mono_contribution / base_contribution < 0.1:
        recommended_weight = base_weight * 0.1 / avg_mono_loss
        print(f"  单调性损失贡献太小 ({mono_contribution / base_contribution * 100:.2f}%)")
        print(f"  建议将 monotonic_weight 增加到 {recommended_weight:.1f}")


if __name__ == "__main__":
    analyze_batch_distribution()
