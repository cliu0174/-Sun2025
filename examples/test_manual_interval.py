"""
测试手动控制稀疏采样间隔

演示如何使用 SPARSE_SAMPLING_INTERVAL 参数手动控制间隔大小
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_augmentation import sparse_sampling, apply_sparse_sampling_by_battery
import numpy as np

def test_manual_intervals():
    """测试不同的手动间隔设置"""

    print("="*70)
    print("测试手动控制稀疏采样间隔")
    print("="*70)

    # 创建测试数据
    n_samples = 100
    features = np.random.randn(n_samples, 10)
    targets = np.linspace(1.0, 0.8, n_samples)
    battery_ids = np.array(['battery_1'] * n_samples)

    print(f"\n原始数据: {n_samples} 个样本")

    # 测试不同的间隔
    intervals = [2, 3, 5, 7, 10, 15, 20]

    print("\n" + "="*70)
    print("不同间隔的采样结果")
    print("="*70)
    print(f"{'间隔':<10} {'保留样本数':<15} {'保留率':<15} {'说明'}")
    print("-"*70)

    for interval in intervals:
        # 稀疏采样
        sampled_features, sampled_targets, sampled_ids = sparse_sampling(
            features, targets, battery_ids,
            sampling_interval=interval,
            offset=0,
            verbose=False
        )

        n_retained = len(sampled_targets)
        retention_rate = n_retained / n_samples * 100

        # 生成说明
        if interval == 2:
            desc = "密集采样"
        elif interval <= 5:
            desc = "中等采样"
        elif interval <= 10:
            desc = "稀疏采样"
        else:
            desc = "极稀疏采样"

        print(f"{interval:<10} {n_retained:<15} {retention_rate:>6.1f}%        {desc}")

    print("\n" + "="*70)
    print("使用建议")
    print("="*70)
    print("""
    实验设计建议：

    1. 轻度稀疏（对比基线）：
       - interval = 2  (保留 50%)
       - interval = 3  (保留 33%)

    2. 中度稀疏（主要测试区）：
       - interval = 5  (保留 20%)
       - interval = 7  (保留 14%)
       - interval = 10 (保留 10%)

    3. 重度稀疏（极限测试）：
       - interval = 15 (保留 6.7%)
       - interval = 20 (保留 5%)
       - interval = 25 (保留 4%)

    用法示例：
    在 train_cross_battery.py 中设置：

    # 使用预设级别
    SPARSE_SAMPLING_LEVEL = 'moderate'  # interval=5
    SPARSE_SAMPLING_INTERVAL = None

    # 或手动设置间隔（优先级更高）
    SPARSE_SAMPLING_LEVEL = 'moderate'  # 将被忽略
    SPARSE_SAMPLING_INTERVAL = 7        # 使用间隔=7
    """)

def demo_usage_examples():
    """演示具体使用案例"""

    print("\n" + "="*70)
    print("具体使用案例")
    print("="*70)

    cases = [
        {
            'name': '案例1: 使用预设级别',
            'level': 'moderate',
            'interval': None,
            'expected': 5,
            'desc': '方便快速测试，使用预定义的间隔'
        },
        {
            'name': '案例2: 手动间隔3（33%保留）',
            'level': 'moderate',  # 将被忽略
            'interval': 3,
            'expected': 3,
            'desc': '测试轻度稀疏场景'
        },
        {
            'name': '案例3: 手动间隔7（14%保留）',
            'level': None,
            'interval': 7,
            'expected': 7,
            'desc': '测试中度稀疏场景'
        },
        {
            'name': '案例4: 手动间隔15（6.7%保留）',
            'level': None,
            'interval': 15,
            'expected': 15,
            'desc': '测试重度稀疏场景'
        }
    ]

    for i, case in enumerate(cases, 1):
        print(f"\n{case['name']}")
        print("-" * 70)
        print(f"说明: {case['desc']}")
        print(f"\n配置:")
        print(f"  SPARSE_SAMPLING_LEVEL = {repr(case['level'])}")
        print(f"  SPARSE_SAMPLING_INTERVAL = {case['interval']}")
        print(f"\n结果:")
        print(f"  实际使用间隔: {case['expected']}")
        retention = 100.0 / case['expected']
        print(f"  保留率: {retention:.1f}%")

if __name__ == "__main__":
    test_manual_intervals()
    demo_usage_examples()

    print("\n" + "="*70)
    print("测试完成")
    print("="*70)
