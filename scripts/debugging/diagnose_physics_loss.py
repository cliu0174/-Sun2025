"""
诊断物理约束损失函数

检查：
1. 物理约束是否被正确调用
2. 各部分损失的实际值
3. monotonic_tolerance 和 max_step 是否生效
"""

import torch
import json
from models import ConfigLoader, PhysicsConstrainedLoss

def diagnose_physics_loss():
    print("="*70)
    print("物理约束损失函数诊断")
    print("="*70)

    # 加载配置
    config = ConfigLoader.load_model_config('lstm')
    physics_config = config.get('physics_constraints', {})

    print("\n当前配置:")
    print(f"  enabled: {physics_config.get('enabled', False)}")
    print(f"  base_loss_weight: {physics_config.get('base_loss_weight', 1.0)}")
    print(f"  monotonic_weight: {physics_config.get('monotonic_weight', 0.1)}")
    print(f"  monotonic_tolerance: {physics_config.get('monotonic_tolerance', 0.01)}")
    print(f"  temporal_decay.enabled: {physics_config.get('temporal_decay', {}).get('enabled', True)}")
    print(f"  temporal_decay.max_step: {physics_config.get('temporal_decay', {}).get('max_step', 20)}")
    print(f"  temporal_decay.decay_alpha: {physics_config.get('temporal_decay', {}).get('decay_alpha', 0.2)}")

    # 创建损失函数（verbose=True 查看详细输出）
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
        verbose=True  # 重要：开启详细输出
    )

    print("\n" + "="*70)
    print("测试场景 1: 明显违反单调性（SOH 持续上升）")
    print("="*70)

    # 场景1：明显违反单调性
    batch_size = 16
    predictions = torch.tensor([
        # 电池 1-1: cycles 100-107, SOH 持续上升 (严重违反)
        0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87,
        # 电池 6-2: cycles 50-57, SOH 单调下降 (正常)
        0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83
    ]).unsqueeze(1)

    targets = predictions + torch.randn(batch_size, 1) * 0.01  # 加噪声

    battery_ids = ['1-1'] * 8 + ['6-2'] * 8
    cycle_indices = torch.tensor(
        list(range(100, 108)) + list(range(50, 58))
    )

    print(f"\n电池 1-1: cycles {list(range(100, 108))}")
    print(f"  预测值: {predictions[:8].squeeze().tolist()}")
    print(f"\n电池 6-2: cycles {list(range(50, 58))}")
    print(f"  预测值: {predictions[8:].squeeze().tolist()}")

    # 计算损失
    loss1 = criterion(predictions, targets, battery_ids, cycle_indices)
    details1 = criterion.get_loss_details()

    print(f"\n损失详情:")
    print(f"  总损失: {details1['total']:.6f}")
    print(f"  基础损失 (MSE): {details1['base']:.6f}")
    print(f"  单调性损失: {details1['monotonic']:.6f}")
    print(f"  边界损失: {details1['boundary']:.6f}")
    print(f"  平滑性损失: {details1['smoothness']:.6f}")

    print("\n" + "="*70)
    print("测试场景 2: 轻微违反单调性（在 tolerance 范围内）")
    print("="*70)

    # 场景2：轻微违反（在 tolerance 范围内）
    predictions2 = torch.tensor([
        # 电池 1-1: cycles 100-107, 轻微波动 ±0.005 (在 tolerance=0.01 范围内)
        0.80, 0.801, 0.799, 0.802, 0.798, 0.803, 0.797, 0.804,
        # 电池 6-2: cycles 50-57, 单调下降
        0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83
    ]).unsqueeze(1)

    targets2 = predictions2 + torch.randn(batch_size, 1) * 0.01

    print(f"\n电池 1-1: cycles {list(range(100, 108))}")
    print(f"  预测值: {predictions2[:8].squeeze().tolist()}")

    loss2 = criterion(predictions2, targets2, battery_ids, cycle_indices)
    details2 = criterion.get_loss_details()

    print(f"\n损失详情:")
    print(f"  总损失: {details2['total']:.6f}")
    print(f"  基础损失 (MSE): {details2['base']:.6f}")
    print(f"  单调性损失: {details2['monotonic']:.6f}")
    print(f"  边界损失: {details2['boundary']:.6f}")

    print("\n" + "="*70)
    print("测试场景 3: 完美单调下降")
    print("="*70)

    # 场景3：完美单调下降
    predictions3 = torch.tensor([
        # 电池 1-1: cycles 100-107, 严格单调下降
        0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76,
        # 电池 6-2: cycles 50-57, 严格单调下降
        0.95, 0.93, 0.91, 0.89, 0.87, 0.85, 0.83, 0.81
    ]).unsqueeze(1)

    targets3 = predictions3 + torch.randn(batch_size, 1) * 0.01

    print(f"\n电池 1-1: cycles {list(range(100, 108))}")
    print(f"  预测值: {predictions3[:8].squeeze().tolist()}")

    loss3 = criterion(predictions3, targets3, battery_ids, cycle_indices)
    details3 = criterion.get_loss_details()

    print(f"\n损失详情:")
    print(f"  总损失: {details3['total']:.6f}")
    print(f"  基础损失 (MSE): {details3['base']:.6f}")
    print(f"  单调性损失: {details3['monotonic']:.6f}")
    print(f"  边界损失: {details3['boundary']:.6f}")

    print("\n" + "="*70)
    print("对比分析")
    print("="*70)

    print(f"\n单调性损失对比:")
    print(f"  场景1 (严重违反): {details1['monotonic']:.6f}")
    print(f"  场景2 (轻微违反): {details2['monotonic']:.6f}")
    print(f"  场景3 (完美下降): {details3['monotonic']:.6f}")

    print(f"\n总损失对比:")
    print(f"  场景1 (严重违反): {details1['total']:.6f}")
    print(f"  场景2 (轻微违反): {details2['total']:.6f}")
    print(f"  场景3 (完美下降): {details3['total']:.6f}")

    # 检查是否有差异
    mono_diff = details1['monotonic'] - details3['monotonic']
    total_diff = details1['total'] - details3['total']

    print(f"\n差异检查:")
    print(f"  单调性损失差异: {mono_diff:.6f}")
    print(f"  总损失差异: {total_diff:.6f}")

    if mono_diff > 0.001:
        print(f"\n✓ 物理约束正在工作！")
        print(f"  违反单调性会导致损失增加 {mono_diff:.6f}")
    else:
        print(f"\n✗ 警告：物理约束可能没有起作用！")
        print(f"  单调性损失差异太小: {mono_diff:.6f}")

    # 测试 max_step 效果
    print("\n" + "="*70)
    print("测试场景 4: max_step 的影响")
    print("="*70)

    # 场景4：远距离配对（超过 max_step）
    batch_size_4 = 4
    predictions4 = torch.tensor([
        # 电池 1-1: cycles [0, 50, 100, 150]
        # 距离: 0->50 (k=50), 0->100 (k=100), 0->150 (k=150)
        0.90, 0.80, 0.85, 0.88  # cycle 100 和 150 违反单调性
    ]).unsqueeze(1)

    targets4 = predictions4 + torch.randn(batch_size_4, 1) * 0.01
    battery_ids_4 = ['1-1'] * 4
    cycle_indices_4 = torch.tensor([0, 50, 100, 150])

    print(f"\n电池 1-1: cycles {cycle_indices_4.tolist()}")
    print(f"  预测值: {predictions4.squeeze().tolist()}")
    print(f"  max_step = {criterion.temporal_max_step}")

    loss4 = criterion(predictions4, targets4, battery_ids_4, cycle_indices_4)
    details4 = criterion.get_loss_details()

    print(f"\n损失详情:")
    print(f"  单调性损失: {details4['monotonic']:.6f}")
    print(f"\n注意：如果 max_step=20，则 cycles 距离超过 20 的配对会被忽略")

    return {
        'scenario1': details1,
        'scenario2': details2,
        'scenario3': details3,
        'scenario4': details4
    }


if __name__ == "__main__":
    results = diagnose_physics_loss()

    print("\n" + "="*70)
    print("诊断建议")
    print("="*70)

    mono1 = results['scenario1']['monotonic']
    mono3 = results['scenario3']['monotonic']

    if mono1 > mono3 + 0.001:
        print("\n✓ 物理约束功能正常")
        print("\n如果训练结果无变化，可能的原因：")
        print("  1. monotonic_weight 太小（尝试增加到 0.5 或 1.0）")
        print("  2. base_loss_weight 太大（物理约束被基础损失淹没）")
        print("  3. 模型已经自然满足单调性（检查训练日志）")
        print("  4. batch 内同一电池样本太少（增大 batch_size）")
    else:
        print("\n✗ 物理约束可能未生效")
        print("\n请检查：")
        print("  1. 配置文件中 'enabled' 是否为 true")
        print("  2. 训练时是否传入了 battery_ids 和 cycle_indices")
        print("  3. use_physics 标志是否正确设置")
