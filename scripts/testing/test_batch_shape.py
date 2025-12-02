"""
快速测试：验证 batch 形状是否正确
"""

import torch
import numpy as np
from data_loaders.data_loader_hust import HUSTBatteryDatasetWithMetadata
from torch.utils.data import DataLoader

def test_batch_shape():
    print("="*70)
    print("测试 Batch 形状")
    print("="*70)

    # 创建模拟数据
    N = 100  # 样本数
    window_size = 40
    feature_dim = 16

    X = np.random.randn(N, window_size, feature_dim)
    y = np.random.rand(N)  # 注意：1D 数组
    battery_ids = ['1-1'] * 50 + ['6-2'] * 50
    cycle_indices = list(range(50)) + list(range(50))

    print(f"\n原始数据形状:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")

    # 创建 Dataset
    dataset = HUSTBatteryDatasetWithMetadata(X, y, battery_ids, cycle_indices)

    # 检查单个样本
    sample = dataset[0]
    print(f"\n单个样本:")
    print(f"  window: {sample['window'].shape}")
    print(f"  target_soh: {sample['target_soh'].shape}")

    # 创建 collate function
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

    # 创建 DataLoader
    batch_size = 16
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                       collate_fn=custom_collate_fn)

    # 获取一个 batch
    batch = next(iter(loader))

    print(f"\nBatch 形状:")
    print(f"  window: {batch['window'].shape}")
    print(f"  target_soh: {batch['target_soh'].shape}")
    print(f"  battery_ids: {len(batch['battery_id'])}")
    print(f"  cycle_indices: {batch['cycle_idx'].shape}")

    # 模拟模型预测
    predictions = torch.randn(batch_size, 1)  # 模型输出 (batch_size, 1)
    targets = batch['target_soh']  # 应该是 (batch_size, 1)

    print(f"\n形状匹配测试:")
    print(f"  predictions: {predictions.shape}")
    print(f"  targets: {targets.shape}")

    # 测试 MSE
    mse = torch.nn.MSELoss()
    try:
        loss = mse(predictions, targets)
        print(f"  ✓ MSE 计算成功，loss = {loss.item():.4f}")
        print(f"\n[SUCCESS] 形状匹配正确！")
        return True
    except Exception as e:
        print(f"  ✗ MSE 计算失败: {e}")
        print(f"\n[FAILED] 形状不匹配！")
        return False


if __name__ == "__main__":
    success = test_batch_shape()

    if success:
        print("\n" + "="*70)
        print("测试通过！可以继续训练")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("测试失败！需要修复")
        print("="*70)
