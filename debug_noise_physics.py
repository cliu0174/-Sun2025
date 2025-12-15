"""
诊断脚本: 分析噪声对物理约束的影响

用途:
1. 可视化加噪后的数据分布
2. 检查三元组采样的有效性
3. 分析物理约束损失的各个组件
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.data_augmentation import add_degradation_noise, get_noise_preset

def visualize_noise_effect(noise_level='heavy'):
    """可视化噪声对容量曲线的影响"""
    print(f"{'='*70}")
    print(f"可视化 {noise_level.upper()} 噪声效果")
    print(f"{'='*70}")

    # 生成模拟的容量衰减曲线
    n_cycles = 500
    true_capacity = np.linspace(1.0, 0.7, n_cycles)  # 线性衰减
    features = np.random.randn(n_cycles, 16)  # 模拟特征
    battery_ids = np.array(['test-battery'] * n_cycles, dtype=object)

    # 获取噪声参数
    noise_params = get_noise_preset(noise_level)
    print(f"\n噪声参数:")
    print(f"  特征噪声σ: {noise_params['feature_noise_std']}")
    print(f"  容量噪声σ: {noise_params['target_noise_std']}")
    print(f"  丢弃率: {noise_params['drop_ratio']*100:.0f}%")

    # 加噪声
    noisy_features, noisy_capacity, noisy_ids = add_degradation_noise(
        features, true_capacity, battery_ids,
        feature_noise_std=noise_params['feature_noise_std'],
        target_noise_std=noise_params['target_noise_std'],
        drop_ratio=noise_params['drop_ratio'],
        seed=42,
        verbose=False
    )

    # 统计信息
    print(f"\n数据统计:")
    print(f"  原始样本数: {len(true_capacity)}")
    print(f"  加噪后样本数: {len(noisy_capacity)}")
    print(f"  实际丢弃率: {(1 - len(noisy_capacity)/len(true_capacity))*100:.1f}%")

    # 计算单调性违反
    violations = 0
    for i in range(len(noisy_capacity) - 1):
        if noisy_capacity[i+1] > noisy_capacity[i]:
            violations += 1

    violation_rate = violations / (len(noisy_capacity) - 1) * 100
    print(f"\n单调性分析:")
    print(f"  单调性违反次数: {violations}/{len(noisy_capacity)-1}")
    print(f"  违反率: {violation_rate:.2f}%")

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 容量曲线对比
    ax = axes[0, 0]
    cycles_original = np.arange(len(true_capacity))
    # 由于丢弃,需要重建对应的cycle索引
    # 这里简化处理,假设均匀分布
    cycles_noisy = np.linspace(0, len(true_capacity)-1, len(noisy_capacity))

    ax.plot(cycles_original, true_capacity, 'b-', label='真实曲线', linewidth=2, alpha=0.7)
    ax.plot(cycles_noisy, noisy_capacity, 'r.', label='加噪后', markersize=2, alpha=0.6)
    ax.set_xlabel('Cycle Index')
    ax.set_ylabel('Capacity (SOH)')
    ax.set_title(f'容量曲线对比 - {noise_level.upper()} 噪声')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 噪声分布直方图
    ax = axes[0, 1]
    # 计算噪声值(只取前len(noisy_capacity)个对齐)
    noise_values = noisy_capacity[:min(len(noisy_capacity), len(true_capacity))] - true_capacity[:min(len(noisy_capacity), len(true_capacity))]
    ax.hist(noise_values, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='零均值')
    ax.set_xlabel('噪声值 (加噪容量 - 真实容量)')
    ax.set_ylabel('频数')
    ax.set_title(f'容量噪声分布\n(均值={noise_values.mean():.4f}, 标准差={noise_values.std():.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 一阶导数(衰减率)
    ax = axes[1, 0]
    true_diff = np.diff(true_capacity)
    noisy_diff = np.diff(noisy_capacity)

    ax.plot(true_diff, 'b-', label='真实衰减率', linewidth=2, alpha=0.7)
    ax.plot(noisy_diff, 'r.', label='加噪后衰减率', markersize=2, alpha=0.6)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Cycle Index')
    ax.set_ylabel('ΔCapacity (一阶导数)')
    ax.set_title('衰减率对比\n(正值=反弹,违反单调性)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 统计正值(违反单调性)
    positive_ratio = (noisy_diff > 0).sum() / len(noisy_diff) * 100
    ax.text(0.05, 0.95, f'反弹比例: {positive_ratio:.1f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. 二阶导数(曲率)
    ax = axes[1, 1]
    true_diff2 = np.diff(true_diff)
    noisy_diff2 = np.diff(noisy_diff)

    ax.plot(true_diff2, 'b-', label='真实曲率', linewidth=2, alpha=0.7)
    ax.plot(noisy_diff2, 'r.', label='加噪后曲率', markersize=2, alpha=0.6)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Cycle Index')
    ax.set_ylabel('Δ²Capacity (二阶导数)')
    ax.set_title('曲率对比\n(抖动剧烈=锯齿严重)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 统计曲率波动
    curvature_std = noisy_diff2.std()
    ax.text(0.05, 0.95, f'曲率波动σ: {curvature_std:.5f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'noise_analysis_{noise_level}.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ 可视化已保存: noise_analysis_{noise_level}.png")
    plt.close()

    return violation_rate, curvature_std


def compare_noise_levels():
    """对比不同噪声级别"""
    print("\n" + "="*70)
    print("对比不同噪声级别对物理约束的影响")
    print("="*70 + "\n")

    results = {}
    for level in ['light', 'medium', 'heavy']:
        violation_rate, curvature_std = visualize_noise_effect(level)
        results[level] = {'violation_rate': violation_rate, 'curvature_std': curvature_std}
        print("\n")

    # 总结
    print("="*70)
    print("总结: 不同噪声级别的影响")
    print("="*70)
    print(f"{'噪声级别':<12} {'单调性违反率':<20} {'曲率波动':<20}")
    print("-"*70)
    for level in ['light', 'medium', 'heavy']:
        vr = results[level]['violation_rate']
        cs = results[level]['curvature_std']
        print(f"{level:<12} {vr:>6.2f}%             {cs:>10.6f}")

    print("\n建议:")
    print("- 单调性违反率 < 5%: 可以用强物理约束 (monotonic_weight >= 1.0)")
    print("- 单调性违反率 5-15%: 用中等物理约束 (monotonic_weight = 0.3-0.5)")
    print("- 单调性违反率 > 15%: 降低物理约束或降低噪声级别")
    print("\n- 曲率波动大: 降低 curvature_weight 或增加 monotonic_tolerance")


def check_triplet_sampling_viability(noise_level='heavy'):
    """检查三元组采样在噪声下的可行性"""
    print(f"\n{'='*70}")
    print(f"检查三元组采样可行性 - {noise_level.upper()} 噪声")
    print(f"{'='*70}")

    n_cycles = 500
    step_k = 1  # 配对步长

    # 模拟数据
    features = np.random.randn(n_cycles, 16)
    capacity = np.linspace(1.0, 0.7, n_cycles)
    battery_ids = np.array(['test'] * n_cycles, dtype=object)

    # 加噪声
    noise_params = get_noise_preset(noise_level)
    _, noisy_capacity, noisy_ids = add_degradation_noise(
        features, capacity, battery_ids,
        feature_noise_std=noise_params['feature_noise_std'],
        target_noise_std=noise_params['target_noise_std'],
        drop_ratio=noise_params['drop_ratio'],
        seed=42,
        verbose=False
    )

    # 计算可用三元组数量
    n_original_triplets = n_cycles - 2 * step_k
    n_after_drop = len(noisy_capacity)
    n_actual_triplets = max(0, n_after_drop - 2 * step_k)

    print(f"\n三元组采样统计:")
    print(f"  原始可用三元组: {n_original_triplets}")
    print(f"  丢弃后剩余样本: {n_after_drop}")
    print(f"  实际可用三元组: {n_actual_triplets}")
    print(f"  三元组保留率: {n_actual_triplets/n_original_triplets*100:.1f}%")

    if n_actual_triplets < n_original_triplets * 0.5:
        print("\n⚠️  警告: 三元组数量减少超过50%,可能影响训练效果!")
        print("   建议: 降低 drop_ratio 或使用 siamese_sampling 代替")

    return n_actual_triplets / n_original_triplets


if __name__ == "__main__":
    # 1. 对比不同噪声级别
    compare_noise_levels()

    # 2. 检查三元组采样
    print("\n")
    for level in ['light', 'medium', 'heavy']:
        check_triplet_sampling_viability(level)
