"""
测试双场景系统的示例脚本

场景一: 随机噪声 + 随机丢弃
场景二: 规律稀疏采样

目的: 验证两个场景可以独立运行,互不影响
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from utils.data_augmentation import (
    add_degradation_noise, get_noise_preset,
    sparse_sampling, get_sparse_sampling_preset,
    apply_sparse_sampling_by_battery
)


def test_scenario1():
    """测试场景一: 随机噪声 + 随机丢弃"""
    print("\n" + "="*70)
    print("测试场景一: 随机噪声 + 随机丢弃")
    print("="*70)

    # 创建模拟数据
    n_samples = 500
    n_features = 16
    features = np.random.randn(n_samples, n_features)
    targets = np.linspace(1.0, 0.7, n_samples)  # 线性衰减
    battery_ids = np.array(['test-battery'] * n_samples, dtype=object)

    # 测试不同噪声级别
    for level in ['light', 'medium', 'heavy']:
        print(f"\n--- 测试噪声级别: {level.upper()} ---")
        params = get_noise_preset(level)

        noisy_features, noisy_targets, noisy_ids = add_degradation_noise(
            features, targets, battery_ids,
            feature_noise_std=params['feature_noise_std'],
            target_noise_std=params['target_noise_std'],
            drop_ratio=params['drop_ratio'],
            seed=42,
            verbose=True
        )

        # 验证数据完整性
        assert len(noisy_features) == len(noisy_targets) == len(noisy_ids)
        assert len(noisy_features) < len(features)  # 应该有样本被丢弃
        assert noisy_targets.min() >= 0.0 and noisy_targets.max() <= 1.2

        print(f"[OK] {level} 级别测试通过")


def test_scenario2():
    """测试场景二: 规律稀疏采样"""
    print("\n" + "="*70)
    print("测试场景二: 规律稀疏采样")
    print("="*70)

    # 创建模拟数据
    n_samples = 500
    n_features = 16
    features = np.random.randn(n_samples, n_features)
    targets = np.linspace(1.0, 0.7, n_samples)
    battery_ids = np.array(['test-battery'] * n_samples, dtype=object)

    # 测试不同稀疏采样级别
    for level in ['dense', 'moderate', 'sparse', 'very_sparse']:
        print(f"\n--- 测试稀疏采样级别: {level.upper()} ---")
        params = get_sparse_sampling_preset(level)

        sampled_features, sampled_targets, sampled_ids = sparse_sampling(
            features, targets, battery_ids,
            sampling_interval=params['sampling_interval'],
            offset=0,
            verbose=True
        )

        # 验证数据完整性
        assert len(sampled_features) == len(sampled_targets) == len(sampled_ids)
        expected_count = (n_samples + params['sampling_interval'] - 1) // params['sampling_interval']
        assert len(sampled_features) == expected_count

        # 验证采样间隔的正确性
        if level == 'moderate':
            # 检查索引间隔是否为5
            assert len(sampled_features) == 100  # 500 // 5 = 100

        print(f"[OK] {level} 级别测试通过")


def test_scenario2_multi_battery():
    """测试场景二: 多电池稀疏采样"""
    print("\n" + "="*70)
    print("测试场景二: 多电池稀疏采样")
    print("="*70)

    # 创建3个电池的模拟数据
    n_batteries = 3
    samples_per_battery = 200
    n_features = 16

    all_features = []
    all_targets = []
    all_ids = []

    for i in range(n_batteries):
        battery_name = f'battery-{i+1}'
        features = np.random.randn(samples_per_battery, n_features)
        targets = np.linspace(1.0, 0.7, samples_per_battery)
        battery_ids = np.array([battery_name] * samples_per_battery, dtype=object)

        all_features.append(features)
        all_targets.append(targets)
        all_ids.append(battery_ids)

    # 合并
    features = np.vstack(all_features)
    targets = np.concatenate(all_targets)
    battery_ids = np.concatenate(all_ids)

    print(f"\n原始数据: {n_batteries} 个电池, 每个 {samples_per_battery} 样本")

    # 应用稀疏采样
    sampled_features, sampled_targets, sampled_ids = apply_sparse_sampling_by_battery(
        features, targets, battery_ids,
        sampling_interval=5,
        offset=0,
        verbose=True
    )

    # 验证每个电池都被正确采样
    unique_batteries = np.unique(sampled_ids)
    assert len(unique_batteries) == n_batteries

    for battery in unique_batteries:
        mask = (sampled_ids == battery)
        battery_samples = mask.sum()
        expected = samples_per_battery // 5
        assert battery_samples == expected
        print(f"  {battery}: {battery_samples} 样本 (期望 {expected})")

    print(f"[OK] 多电池稀疏采样测试通过")


def test_independence():
    """测试两个场景的独立性"""
    print("\n" + "="*70)
    print("测试场景独立性")
    print("="*70)

    # 创建相同的初始数据
    n_samples = 500
    n_features = 16
    np.random.seed(42)
    features = np.random.randn(n_samples, n_features)
    targets = np.linspace(1.0, 0.7, n_samples)
    battery_ids = np.array(['test'] * n_samples, dtype=object)

    # 场景一处理
    noise_params = get_noise_preset('medium')
    scenario1_features, scenario1_targets, scenario1_ids = add_degradation_noise(
        features.copy(), targets.copy(), battery_ids.copy(),
        feature_noise_std=noise_params['feature_noise_std'],
        target_noise_std=noise_params['target_noise_std'],
        drop_ratio=noise_params['drop_ratio'],
        seed=42,
        verbose=False
    )

    # 场景二处理
    sampling_params = get_sparse_sampling_preset('moderate')
    scenario2_features, scenario2_targets, scenario2_ids = sparse_sampling(
        features.copy(), targets.copy(), battery_ids.copy(),
        sampling_interval=sampling_params['sampling_interval'],
        offset=0,
        verbose=False
    )

    # 验证两个场景产生不同的结果
    print(f"\n原始数据: {len(features)} 样本")
    print(f"场景一结果: {len(scenario1_features)} 样本 (随机丢弃)")
    print(f"场景二结果: {len(scenario2_features)} 样本 (规律采样)")

    # 它们不应该产生相同数量的样本 (除非巧合)
    assert len(scenario1_features) != len(scenario2_features)

    # 场景一有噪声,场景二无噪声
    scenario1_noise = np.abs(scenario1_targets - targets[:len(scenario1_targets)]).mean()
    scenario2_noise = np.abs(scenario2_targets - targets[::sampling_params['sampling_interval']]).mean()

    print(f"\n场景一平均噪声: {scenario1_noise:.6f} (应该 > 0)")
    print(f"场景二平均噪声: {scenario2_noise:.6f} (应该 ≈ 0)")

    assert scenario1_noise > 0.001  # 场景一有明显噪声
    assert scenario2_noise < 1e-10  # 场景二无噪声

    print(f"\n[OK] 场景独立性测试通过: 两个场景产生不同的结果且互不影响")


if __name__ == "__main__":
    print("="*70)
    print("双场景系统测试")
    print("="*70)

    try:
        # 测试场景一
        test_scenario1()

        # 测试场景二
        test_scenario2()

        # 测试多电池场景
        test_scenario2_multi_battery()

        # 测试独立性
        test_independence()

        # 总结
        print("\n" + "="*70)
        print("[OK] 所有测试通过!")
        print("="*70)
        print("\n双场景系统已就绪:")
        print("  - 场景一: 随机噪声 + 随机丢弃 (3个级别)")
        print("  - 场景二: 规律稀疏采样 (4个级别)")
        print("  - 两个场景完全独立,可单独使用")
        print("\n使用方法:")
        print("  1. 修改 train_cross_battery.py 中的 DEGRADATION_SCENARIO 参数")
        print("  2. 'none': 无退化 (干净数据)")
        print("  3. 'scenario1': 场景一 (随机退化)")
        print("  4. 'scenario2': 场景二 (稀疏采样)")

    except AssertionError as e:
        print(f"\n[FAIL] 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
