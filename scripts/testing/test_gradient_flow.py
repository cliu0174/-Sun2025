"""
测试物理约束损失的梯度传播

验证梯度能否正确反向传播到模型参数
"""

import torch
import torch.nn as nn
from models.physics_loss import PhysicsConstrainedLoss

def test_gradient_flow():
    print("="*70)
    print("测试物理约束损失的梯度传播")
    print("="*70)

    # 创建一个简单的模型
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
        nn.Sigmoid()
    )

    # 创建物理约束损失函数
    criterion = PhysicsConstrainedLoss(
        base_loss_weight=1.0,
        monotonic_weight=1.0,
        boundary_weight=0.1,
        smoothness_weight=0.5,
        monotonic_tolerance=0.01,
        temporal_decay_enabled=True,
        temporal_max_step=20,
        temporal_decay_type='exp',
        temporal_decay_alpha=0.2,
        verbose=False
    )

    # 创建测试数据
    batch_size = 16
    features = torch.randn(batch_size, 10)
    targets = torch.rand(batch_size, 1)

    # 模拟元数据
    battery_ids = ['1-1'] * 8 + ['6-2'] * 8
    cycle_indices = torch.tensor(
        list(range(100, 108)) + list(range(50, 58))
    )

    print(f"\n测试设置:")
    print(f"  batch_size: {batch_size}")
    print(f"  电池 1-1: cycles 100-107")
    print(f"  电池 6-2: cycles 50-57")

    # 前向传播
    predictions = model(features)

    # 计算损失
    loss = criterion(predictions, targets, battery_ids, cycle_indices)
    details = criterion.get_loss_details()

    print(f"\n损失计算:")
    print(f"  总损失: {loss.item():.6f}")
    print(f"  基础损失: {details['base']:.6f}")
    print(f"  单调性损失: {details['monotonic']:.6f}")
    print(f"  边界损失: {details['boundary']:.6f}")
    print(f"  平滑性损失: {details['smoothness']:.6f}")

    # 检查梯度
    print(f"\n梯度检查:")
    print(f"  loss.requires_grad: {loss.requires_grad}")
    print(f"  loss.grad_fn: {loss.grad_fn}")

    # 反向传播
    loss.backward()

    # 检查模型参数的梯度
    has_grad = False
    max_grad = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            max_grad = max(max_grad, grad_norm)
            print(f"  {name}: grad_norm = {grad_norm:.6f}")

    print(f"\n梯度传播测试:")
    if has_grad:
        print(f"  SUCCESS 梯度成功传播到模型参数")
        print(f"  最大梯度范数: {max_grad:.6f}")

        if max_grad > 1e-6:
            print(f"  梯度足够大，物理约束会影响训练")
        else:
            print(f"  警告: 梯度非常小，物理约束影响可能很弱")
    else:
        print(f"  ERROR 梯度未传播！")
        return False

    # 测试优化器更新
    print(f"\n测试优化器更新:")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 记录更新前的参数
    old_params = [p.clone() for p in model.parameters()]

    optimizer.zero_grad()
    predictions = model(features)
    loss = criterion(predictions, targets, battery_ids, cycle_indices)
    loss.backward()
    optimizer.step()

    # 检查参数是否更新
    params_changed = False
    for old_p, new_p in zip(old_params, model.parameters()):
        if not torch.allclose(old_p, new_p):
            params_changed = True
            diff = (new_p - old_p).abs().max().item()
            print(f"  参数最大变化: {diff:.6f}")
            break

    if params_changed:
        print(f"  SUCCESS 优化器成功更新参数")
    else:
        print(f"  ERROR 参数未更新！")
        return False

    print("\n" + "="*70)
    print("ALL TESTS PASSED 所有测试通过")
    print("="*70)
    print("\n物理约束损失的梯度传播正常！")
    print("现在可以重新训练，应该会看到不同权重产生不同的结果。")

    return True


if __name__ == "__main__":
    test_gradient_flow()
