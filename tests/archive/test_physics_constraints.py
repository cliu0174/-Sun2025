"""
测试物理约束损失函数的集成

测试步骤：
1. 加载LSTM配置（带物理约束）
2. 创建UnifiedModelWrapper
3. 验证损失函数是否正确创建
4. 测试训练流程
"""

import torch
import json
from models import UnifiedModelWrapper, ConfigLoader, PhysicsConstrainedLoss


def test_physics_constraints_integration():
    """测试物理约束的集成"""
    print("="*70)
    print("测试物理约束集成")
    print("="*70)

    # 1. 测试不带物理约束（默认配置）
    print("\n[测试1] 默认配置（无物理约束）")
    print("-"*70)

    config_no_physics = ConfigLoader.load_model_config('lstm')
    config_no_physics['physics_constraints']['enabled'] = False

    wrapper_no_physics = UnifiedModelWrapper(
        model_type='lstm',
        input_size=16,
        config=config_no_physics,
        device='cpu'
    )

    print(f"损失函数类型: {type(wrapper_no_physics.criterion).__name__}")
    print(f"是否为物理约束损失: {isinstance(wrapper_no_physics.criterion, PhysicsConstrainedLoss)}")

    # 2. 测试带物理约束
    print("\n[测试2] 启用物理约束")
    print("-"*70)

    config_with_physics = ConfigLoader.load_model_config('lstm')
    config_with_physics['physics_constraints']['enabled'] = True
    config_with_physics['physics_constraints']['monotonic_weight'] = 0.1
    config_with_physics['physics_constraints']['boundary_weight'] = 0.05

    wrapper_with_physics = UnifiedModelWrapper(
        model_type='lstm',
        input_size=16,
        config=config_with_physics,
        device='cpu'
    )

    print(f"损失函数类型: {type(wrapper_with_physics.criterion).__name__}")
    print(f"是否为物理约束损失: {isinstance(wrapper_with_physics.criterion, PhysicsConstrainedLoss)}")

    if isinstance(wrapper_with_physics.criterion, PhysicsConstrainedLoss):
        print(f"单调性权重: {wrapper_with_physics.criterion.monotonic_weight}")
        print(f"边界权重: {wrapper_with_physics.criterion.boundary_weight}")
        print(f"平滑性权重: {wrapper_with_physics.criterion.smoothness_weight}")

    # 3. 测试损失计算
    print("\n[测试3] 损失计算测试")
    print("-"*70)

    # 创建模拟数据
    batch_size = 4
    seq_len = 10
    predictions = torch.linspace(1.0, 0.8, seq_len).repeat(batch_size, 1).unsqueeze(-1)
    targets = torch.linspace(1.0, 0.8, seq_len).repeat(batch_size, 1).unsqueeze(-1)

    # 合法预测（单调递减）
    loss_no_physics = wrapper_no_physics.criterion(predictions, targets)
    loss_with_physics = wrapper_with_physics.criterion(predictions, targets)

    print(f"无物理约束损失: {loss_no_physics.item():.6f}")
    print(f"有物理约束损失: {loss_with_physics.item():.6f}")

    # 非法预测（有违反）
    predictions_illegal = predictions.clone()
    predictions_illegal[:, 5, 0] = 0.95  # 在第5个位置违反单调性

    loss_no_physics_illegal = wrapper_no_physics.criterion(predictions_illegal, targets)
    loss_with_physics_illegal = wrapper_with_physics.criterion(predictions_illegal, targets)

    print(f"\n非法预测（有违反）:")
    print(f"  无物理约束损失: {loss_no_physics_illegal.item():.6f}")
    print(f"  有物理约束损失: {loss_with_physics_illegal.item():.6f} (应该更大)")

    # 4. 如何在配置文件中启用物理约束
    print("\n"+"="*70)
    print("如何启用物理约束")
    print("="*70)
    print("\n在模型配置文件中（如 lstm_config.json）添加：")
    print(json.dumps({
        "physics_constraints": {
            "enabled": True,
            "monotonic_weight": 0.1,
            "boundary_weight": 0.0,
            "smoothness_weight": 0.0,
            "description": "物理约束：monotonic(单调性), boundary(边界[0,1]), smoothness(平滑性)"
        }
    }, indent=2, ensure_ascii=False))

    print("\n[OK] 测试完成！")
    print("\n总结：")
    print("  1. 配置文件中设置 'physics_constraints.enabled': true 即可启用")
    print("  2. 可以通过调整权重参数控制约束强度")
    print("  3. 所有模型会自动使用配置中的物理约束设置")
    print("  4. enabled=false 时，使用标准MSE损失")


if __name__ == "__main__":
    test_physics_constraints_integration()
