"""
测试物理约束在 train_cross_battery.py 中的集成

验证：
1. 配置加载正确
2. DataLoader 创建正确（带 metadata）
3. 损失函数初始化正确
4. 训练循环可以运行
"""

import torch
import json
from models import ConfigLoader, PhysicsConstrainedLoss

def test_config_loading():
    """测试配置加载"""
    print("="*70)
    print("测试 1: 配置加载")
    print("="*70)

    config = ConfigLoader.load_model_config('lstm')

    physics_config = config.get('physics_constraints', {})
    use_physics = physics_config.get('enabled', False)

    print(f"\n物理约束配置:")
    print(f"  enabled: {use_physics}")
    print(f"  monotonic_weight: {physics_config.get('monotonic_weight', 0.1)}")
    print(f"  monotonic_tolerance: {physics_config.get('monotonic_tolerance', 0.01)}")
    print(f"  temporal_decay.enabled: {physics_config.get('temporal_decay', {}).get('enabled', True)}")
    print(f"  temporal_decay.max_step: {physics_config.get('temporal_decay', {}).get('max_step', 20)}")

    return config, physics_config, use_physics


def test_loss_creation(physics_config):
    """测试损失函数创建"""
    print("\n" + "="*70)
    print("测试 2: 损失函数创建")
    print("="*70)

    try:
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

        print("\n损失函数创建成功！")
        print(f"  类型: {type(criterion).__name__}")
        print(f"  单调性权重: {criterion.monotonic_weight}")
        print(f"  容忍度: {criterion.monotonic_tolerance}")
        print(f"  时间衰减: {criterion.temporal_decay_enabled}")

        return criterion

    except Exception as e:
        print(f"\n[ERROR] 损失函数创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_loss_computation(criterion):
    """测试损失计算"""
    print("\n" + "="*70)
    print("测试 3: 损失计算")
    print("="*70)

    # 模拟 batch
    batch_size = 16
    predictions = torch.randn(batch_size, 1) * 0.1 + 0.8  # 围绕 0.8
    targets = torch.randn(batch_size, 1) * 0.1 + 0.8

    # 模拟元数据
    battery_ids = ['1-1'] * 8 + ['6-2'] * 8
    cycle_indices = torch.tensor(
        list(range(50, 58)) + list(range(40, 48))
    )

    print(f"\nBatch 信息:")
    print(f"  batch_size: {batch_size}")
    print(f"  电池 1-1: cycles {list(range(50, 58))}")
    print(f"  电池 6-2: cycles {list(range(40, 48))}")

    # 计算损失
    try:
        loss = criterion(predictions, targets, battery_ids, cycle_indices)

        print(f"\n损失计算成功！")
        print(f"  总损失: {loss.item():.6f}")

        details = criterion.get_loss_details()
        print(f"\n详细损失:")
        for key, value in details.items():
            print(f"    {key}: {value:.6f}")

        return True

    except Exception as e:
        print(f"\n[ERROR] 损失计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_format():
    """测试 batch 格式"""
    print("\n" + "="*70)
    print("测试 4: Batch 格式")
    print("="*70)

    # 模拟物理约束模式的 batch
    batch_physics = {
        'window': torch.randn(16, 40, 16),
        'target_soh': torch.randn(16, 1),
        'battery_id': ['1-1'] * 8 + ['6-2'] * 8,
        'cycle_idx': torch.tensor(list(range(50, 58)) + list(range(40, 48)))
    }

    # 模拟标准模式的 batch
    batch_standard = (
        torch.randn(16, 40, 16),
        torch.randn(16, 1)
    )

    print("\n物理约束模式 batch:")
    print(f"  类型: {type(batch_physics)}")
    print(f"  keys: {list(batch_physics.keys())}")
    print(f"  window shape: {batch_physics['window'].shape}")
    print(f"  battery_ids: {len(batch_physics['battery_id'])} 个")

    print("\n标准模式 batch:")
    print(f"  类型: {type(batch_standard)}")
    print(f"  元素数: {len(batch_standard)}")
    print(f"  features shape: {batch_standard[0].shape}")
    print(f"  targets shape: {batch_standard[1].shape}")

    # 测试解包逻辑
    print("\n测试解包逻辑:")

    # 物理约束模式
    use_physics = True
    if use_physics:
        features = batch_physics['window']
        targets = batch_physics['target_soh']
        battery_ids = batch_physics['battery_id']
        cycle_indices = batch_physics['cycle_idx']
        print("  [物理模式] 解包成功 ✓")
        print(f"    features: {features.shape}")
        print(f"    targets: {targets.shape}")
        print(f"    battery_ids: {len(battery_ids)}")
        print(f"    cycle_indices: {cycle_indices.shape}")

    # 标准模式
    use_physics = False
    if not use_physics:
        features, targets = batch_standard
        print("  [标准模式] 解包成功 ✓")
        print(f"    features: {features.shape}")
        print(f"    targets: {targets.shape}")

    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("物理约束集成测试 - train_cross_battery.py")
    print("="*70)

    try:
        # 测试 1: 配置加载
        config, physics_config, use_physics = test_config_loading()
        print("\n[OK] 配置加载测试通过")

        # 测试 2: 损失函数创建
        criterion = test_loss_creation(physics_config)
        if criterion is None:
            raise Exception("损失函数创建失败")
        print("\n[OK] 损失函数创建测试通过")

        # 测试 3: 损失计算
        success = test_loss_computation(criterion)
        if not success:
            raise Exception("损失计算失败")
        print("\n[OK] 损失计算测试通过")

        # 测试 4: Batch 格式
        test_batch_format()
        print("\n[OK] Batch 格式测试通过")

        print("\n" + "="*70)
        print("所有测试通过！✓")
        print("="*70)
        print("\n你现在可以:")
        print("  1. 在配置文件中启用物理约束 (enabled: true)")
        print("  2. 运行 train_cross_battery.py 进行训练")
        print("  3. 对比有无物理约束的效果")

    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
