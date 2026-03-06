"""
调试训练缓慢问题
"""

import torch
import time
import numpy as np
from models.physics_loss import PhysicsConstrainedLoss
from models import ConfigLoader
from data_loaders import load_single_hust_battery
from data_loaders.data_loader_hust import apply_windowing_with_metadata, HUSTBatteryDatasetWithMetadata
from torch.utils.data import DataLoader, ConcatDataset

def test_single_batch_time():
    """测试单个 batch 的计算时间"""
    print("="*70)
    print("测试单个 Batch 的计算时间")
    print("="*70)

    # 加载配置
    config = ConfigLoader.load_model_config('lstm')
    physics_config = config.get('physics_constraints', {})

    # 创建损失函数
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
    )

    print(f"\n物理约束配置:")
    print(f"  monotonic_weight: {physics_config.get('monotonic_weight', 0.1)}")
    print(f"  smoothness_weight: {physics_config.get('smoothness_weight', 0.0)}")
    print(f"  max_step: {physics_config.get('temporal_decay', {}).get('max_step', 20)}")

    # 模拟一个典型的 batch
    batch_size = 256
    predictions = torch.randn(batch_size, 1, requires_grad=True).cuda()
    targets = torch.rand(batch_size, 1).cuda()

    # 模拟跨电池数据分布（典型情况）
    num_batteries = 10
    samples_per_battery = batch_size // num_batteries
    battery_ids = []
    cycle_indices_list = []

    for i in range(num_batteries):
        battery_name = f"battery_{i}"
        # 每个电池约 25 个样本，cycle 随机分布
        for _ in range(samples_per_battery):
            battery_ids.append(battery_name)
            cycle_indices_list.append(torch.randint(0, 1000, (1,)).item())

    # 补齐到 batch_size
    remaining = batch_size - len(battery_ids)
    for _ in range(remaining):
        battery_ids.append("battery_0")
        cycle_indices_list.append(torch.randint(0, 1000, (1,)).item())

    cycle_indices = torch.tensor(cycle_indices_list).cuda()

    print(f"\nBatch 信息:")
    print(f"  batch_size: {batch_size}")
    print(f"  电池数: {num_batteries}")
    print(f"  每个电池约 {samples_per_battery} 个样本")

    # 测试前向传播时间
    print(f"\n测试损失计算时间...")

    # 预热
    for _ in range(3):
        loss = criterion(predictions, targets, battery_ids, cycle_indices)

    torch.cuda.synchronize()

    # 正式计时
    num_iterations = 10
    start_time = time.time()

    for _ in range(num_iterations):
        loss = criterion(predictions, targets, battery_ids, cycle_indices)
        torch.cuda.synchronize()

    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations

    print(f"\n前向传播:")
    print(f"  平均时间: {avg_time*1000:.2f} ms")
    print(f"  总损失: {loss.item():.6f}")

    # 测试反向传播时间
    print(f"\n测试反向传播时间...")

    start_time = time.time()

    for _ in range(num_iterations):
        predictions = torch.randn(batch_size, 1, requires_grad=True).cuda()
        loss = criterion(predictions, targets, battery_ids, cycle_indices)
        loss.backward()
        torch.cuda.synchronize()

    end_time = time.time()
    avg_time_backward = (end_time - start_time) / num_iterations

    print(f"  平均时间: {avg_time_backward*1000:.2f} ms")

    # 总时间
    total_time = avg_time + avg_time_backward
    print(f"\n单个 batch 总时间: {total_time*1000:.2f} ms")

    # 估算每个 epoch 时间
    num_batches = 83697 // batch_size  # 训练集样本数 / batch_size
    epoch_time = total_time * num_batches
    print(f"\n估算每个 epoch 时间: {epoch_time:.2f} 秒")

    if epoch_time > 10:
        print(f"\n⚠️  警告: 每个 epoch 需要 {epoch_time:.1f} 秒，这太慢了！")
        print(f"\n可能的原因:")
        print(f"  1. 物理约束计算中的嵌套循环太多")
        print(f"  2. 每个电池样本数太多，导致配对数量爆炸")
        print(f"  3. temporal_max_step 太大")
    else:
        print(f"\n✓ 时间看起来合理")

    return avg_time, avg_time_backward, epoch_time


def test_data_loading_time():
    """测试数据加载时间"""
    print("\n" + "="*70)
    print("测试数据加载时间")
    print("="*70)

    # 加载少量电池测试
    import os
    data_dir = 'data/HUST data'
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])[:5]

    window_size = 40
    all_datasets = []

    start_time = time.time()

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

    combined_dataset = ConcatDataset(all_datasets)

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

    loader = DataLoader(combined_dataset, batch_size=256, shuffle=True,
                       collate_fn=custom_collate_fn)

    end_time = time.time()
    print(f"数据加载和 DataLoader 创建: {end_time - start_time:.2f} 秒")

    # 测试迭代一个 batch
    start_time = time.time()
    batch = next(iter(loader))
    end_time = time.time()

    print(f"获取第一个 batch: {end_time - start_time:.2f} 秒")
    print(f"Batch 大小: {len(batch['battery_id'])}")

    return end_time - start_time


if __name__ == "__main__":
    print("\n调试训练缓慢问题\n")

    try:
        # 测试 1: 单个 batch 计算时间
        forward_time, backward_time, epoch_time = test_single_batch_time()

        # 测试 2: 数据加载时间
        data_time = test_data_loading_time()

        print("\n" + "="*70)
        print("总结")
        print("="*70)
        print(f"前向传播时间: {forward_time*1000:.2f} ms")
        print(f"反向传播时间: {backward_time*1000:.2f} ms")
        print(f"单个 batch 总时间: {(forward_time + backward_time)*1000:.2f} ms")
        print(f"估算每个 epoch: {epoch_time:.2f} 秒")
        print(f"数据加载第一个 batch: {data_time:.2f} 秒")

        if epoch_time > 10:
            print(f"\n❌ 训练太慢！建议优化")
        else:
            print(f"\n✓ 性能可接受")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
