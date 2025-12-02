"""
测试优化后的平滑度损失性能
"""

import torch
import time
from models.physics_loss import PhysicsConstrainedLoss

def test_smoothness_performance():
    print("="*70)
    print("测试平滑度损失性能")
    print("="*70)

    # 创建两个损失函数：一个有平滑度，一个没有
    criterion_with_smooth = PhysicsConstrainedLoss(
        base_loss_weight=1.0,
        monotonic_weight=1.0,
        boundary_weight=0.1,
        smoothness_weight=0.5,  # 启用平滑度
        monotonic_tolerance=0.01,
        temporal_decay_enabled=True,
        temporal_max_step=20,
        temporal_decay_type='exp',
        temporal_decay_alpha=0.2,
        verbose=False
    )

    criterion_without_smooth = PhysicsConstrainedLoss(
        base_loss_weight=1.0,
        monotonic_weight=1.0,
        boundary_weight=0.1,
        smoothness_weight=0.0,  # 禁用平滑度
        monotonic_tolerance=0.01,
        temporal_decay_enabled=True,
        temporal_max_step=20,
        temporal_decay_type='exp',
        temporal_decay_alpha=0.2,
        verbose=False
    )

    # 模拟 batch
    batch_size = 256
    predictions = torch.randn(batch_size, 1, requires_grad=True).cuda()
    targets = torch.rand(batch_size, 1).cuda()

    # 模拟跨电池数据
    num_batteries = 10
    samples_per_battery = batch_size // num_batteries
    battery_ids = []
    cycle_indices_list = []

    for i in range(num_batteries):
        battery_name = f"battery_{i}"
        for j in range(samples_per_battery):
            battery_ids.append(battery_name)
            cycle_indices_list.append(i * 100 + j)

    remaining = batch_size - len(battery_ids)
    for i in range(remaining):
        battery_ids.append("battery_0")
        cycle_indices_list.append(i)

    cycle_indices = torch.tensor(cycle_indices_list).cuda()

    print(f"\nBatch 配置:")
    print(f"  batch_size: {batch_size}")
    print(f"  电池数: {num_batteries}")
    print(f"  每个电池样本数: {samples_per_battery}")

    # 预热
    for _ in range(5):
        _ = criterion_with_smooth(predictions, targets, battery_ids, cycle_indices)
        _ = criterion_without_smooth(predictions, targets, battery_ids, cycle_indices)

    torch.cuda.synchronize()

    # 测试无平滑度
    print(f"\n测试无平滑度约束...")
    times = []
    for _ in range(20):
        start = time.time()
        loss = criterion_without_smooth(predictions, targets, battery_ids, cycle_indices)
        torch.cuda.synchronize()
        times.append(time.time() - start)

    avg_time_without = sum(times) / len(times)
    print(f"  平均时间: {avg_time_without*1000:.2f} ms")

    # 测试有平滑度
    print(f"\n测试有平滑度约束...")
    times = []
    for _ in range(20):
        start = time.time()
        loss = criterion_with_smooth(predictions, targets, battery_ids, cycle_indices)
        torch.cuda.synchronize()
        times.append(time.time() - start)

    avg_time_with = sum(times) / len(times)
    print(f"  平均时间: {avg_time_with*1000:.2f} ms")

    # 对比
    overhead = avg_time_with - avg_time_without
    print(f"\n" + "="*70)
    print("性能对比")
    print("="*70)
    print(f"无平滑度: {avg_time_without*1000:.2f} ms")
    print(f"有平滑度: {avg_time_with*1000:.2f} ms")
    print(f"平滑度开销: {overhead*1000:.2f} ms ({overhead/avg_time_without*100:.1f}%)")

    # 估算 epoch 时间
    num_batches = 83697 // batch_size
    epoch_time_without = avg_time_without * num_batches
    epoch_time_with = avg_time_with * num_batches

    print(f"\n估算每个 epoch 时间:")
    print(f"  无平滑度: {epoch_time_without:.2f} 秒")
    print(f"  有平滑度: {epoch_time_with:.2f} 秒")
    print(f"  差异: {epoch_time_with - epoch_time_without:.2f} 秒")

    if overhead / avg_time_without < 0.5:
        print(f"\n✓ 平滑度开销可接受 (<50%)")
    else:
        print(f"\n⚠ 平滑度开销较大 (>{overhead/avg_time_without*100:.0f}%)")

    # 测试反向传播
    print(f"\n测试反向传播...")
    times_backward = []
    for _ in range(10):
        predictions = torch.randn(batch_size, 1, requires_grad=True).cuda()
        start = time.time()
        loss = criterion_with_smooth(predictions, targets, battery_ids, cycle_indices)
        loss.backward()
        torch.cuda.synchronize()
        times_backward.append(time.time() - start)

    avg_time_backward = sum(times_backward) / len(times_backward)
    print(f"  前向+反向: {avg_time_backward*1000:.2f} ms")

    return avg_time_with, avg_time_backward


if __name__ == "__main__":
    try:
        forward_time, backward_time = test_smoothness_performance()
        print(f"\n总结:")
        print(f"  前向传播: {forward_time*1000:.2f} ms")
        print(f"  前向+反向: {backward_time*1000:.2f} ms")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
