"""
详细分析物理约束损失的性能瓶颈
"""

import torch
import time
from models.physics_loss import PhysicsConstrainedLoss

def profile_each_component():
    """分析每个组件的耗时"""
    print("="*70)
    print("分析物理约束各组件耗时")
    print("="*70)

    # 创建损失函数
    criterion = PhysicsConstrainedLoss(
        base_loss_weight=1.0,
        monotonic_weight=1.0,
        boundary_weight=0.1,
        smoothness_weight=0.0,  # 已禁用
        monotonic_tolerance=0.01,
        temporal_decay_enabled=True,
        temporal_max_step=20,
        temporal_decay_type='exp',
        temporal_decay_alpha=0.2,
        verbose=False
    )

    # 模拟一个 batch
    batch_size = 256
    predictions = torch.randn(batch_size, 1, requires_grad=True).cuda()
    targets = torch.rand(batch_size, 1).cuda()

    # 模拟跨电池数据（实际情况）
    num_batteries = 10
    samples_per_battery = batch_size // num_batteries
    battery_ids = []
    cycle_indices_list = []

    for i in range(num_batteries):
        battery_name = f"battery_{i}"
        for j in range(samples_per_battery):
            battery_ids.append(battery_name)
            # 模拟连续的 cycle（会产生很多配对）
            cycle_indices_list.append(i * 100 + j)

    # 补齐
    remaining = batch_size - len(battery_ids)
    for i in range(remaining):
        battery_ids.append("battery_0")
        cycle_indices_list.append(i)

    cycle_indices = torch.tensor(cycle_indices_list).cuda()

    print(f"\nBatch 配置:")
    print(f"  batch_size: {batch_size}")
    print(f"  电池数: {num_batteries}")
    print(f"  每个电池样本数: {samples_per_battery}")
    print(f"  max_step: {criterion.temporal_max_step}")

    # 计算理论配对数
    # 对于连续的 n 个样本，max_step=20 时，配对数约为 n * min(n, 20) / 2
    theoretical_pairs = 0
    for i in range(num_batteries):
        n = samples_per_battery
        if n >= 2:
            # 每个样本最多与 min(max_step, n-1) 个后续样本配对
            for j in range(n):
                max_pairs = min(criterion.temporal_max_step, n - j - 1)
                theoretical_pairs += max_pairs

    print(f"  理论配对数: {theoretical_pairs}")

    # 预热
    for _ in range(3):
        _ = criterion(predictions, targets, battery_ids, cycle_indices)

    torch.cuda.synchronize()

    # 测试各组件耗时
    components = {
        'base_loss': criterion.mse_loss,
        'boundary_loss': lambda p, t: criterion.boundary_loss(p),
        'monotonic_loss': lambda p, t: criterion.monotonic_loss(p, battery_ids, cycle_indices),
        'total_loss': lambda p, t: criterion(p, t, battery_ids, cycle_indices)
    }

    results = {}

    for name, func in components.items():
        times = []
        for _ in range(10):
            start = time.time()
            if name == 'total_loss':
                loss = func(predictions, targets)
            else:
                loss = func(predictions, targets)
            torch.cuda.synchronize()
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)
        results[name] = avg_time

        print(f"\n{name}:")
        print(f"  平均时间: {avg_time*1000:.2f} ms")
        if hasattr(loss, 'item'):
            print(f"  损失值: {loss.item():.6f}")

    # 分析
    print("\n" + "="*70)
    print("性能分析")
    print("="*70)

    total = results['total_loss']
    base = results['base_loss']
    boundary = results['boundary_loss']
    monotonic = results['monotonic_loss']

    print(f"\n各组件占比:")
    print(f"  基础损失: {base/total*100:.1f}% ({base*1000:.2f} ms)")
    print(f"  边界损失: {boundary/total*100:.1f}% ({boundary*1000:.2f} ms)")
    print(f"  单调性损失: {monotonic/total*100:.1f}% ({monotonic*1000:.2f} ms)")

    if monotonic / total > 0.8:
        print(f"\n⚠ 单调性损失占用了 {monotonic/total*100:.1f}% 的时间!")
        print(f"  理论配对数: {theoretical_pairs}")
        print(f"  建议:")
        print(f"    1. 减小 max_step (从 20 改为 10)")
        print(f"    2. 减小 batch_size (从 256 改为 128)")
        print(f"    3. 优化单调性损失的实现")

    return results


def test_max_step_impact():
    """测试 max_step 对性能的影响"""
    print("\n" + "="*70)
    print("测试 max_step 对性能的影响")
    print("="*70)

    batch_size = 256
    predictions = torch.randn(batch_size, 1, requires_grad=True).cuda()
    targets = torch.rand(batch_size, 1).cuda()

    # 单个电池，连续 cycle
    battery_ids = ['battery_0'] * batch_size
    cycle_indices = torch.arange(batch_size).cuda()

    max_steps = [5, 10, 15, 20, 30]
    times = []

    for max_step in max_steps:
        criterion = PhysicsConstrainedLoss(
            base_loss_weight=1.0,
            monotonic_weight=1.0,
            boundary_weight=0.1,
            smoothness_weight=0.0,
            monotonic_tolerance=0.01,
            temporal_decay_enabled=True,
            temporal_max_step=max_step,
            temporal_decay_type='exp',
            temporal_decay_alpha=0.2,
            verbose=False
        )

        # 预热
        for _ in range(3):
            _ = criterion(predictions, targets, battery_ids, cycle_indices)

        torch.cuda.synchronize()

        # 计时
        start = time.time()
        for _ in range(10):
            loss = criterion(predictions, targets, battery_ids, cycle_indices)
            torch.cuda.synchronize()
        avg_time = (time.time() - start) / 10

        times.append(avg_time)
        print(f"  max_step={max_step:2d}: {avg_time*1000:.2f} ms")

    print(f"\n建议: 使用 max_step=10 可以节省 {(times[-1]-times[1])/times[-1]*100:.1f}% 的时间")


if __name__ == "__main__":
    try:
        # 分析各组件
        results = profile_each_component()

        # 测试 max_step 影响
        test_max_step_impact()

        print("\n" + "="*70)
        print("分析完成")
        print("="*70)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
