"""
快速测试单调性权重（简化版）

用于快速验证，只测试3个关键权重点：
- 0.0 (无物理约束)
- 0.1 (标准值)
- 0.2 (较强约束)

运行时间约 30 分钟（3个权重 × 10分钟/个）
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一块GPU

from test_monotonic_weights import CONFIG, main

# 覆盖默认配置
CONFIG.update({
    # 只测试3个关键权重点
    'monotonic_weights': [0.0, 0.1, 0.2],

    # 稀疏采样配置
    'degradation_scenario': 'scenario2',
    'sparse_sampling_interval': 5,  # 保留 20%

    # 输出目录
    'output_dir': 'results/quick_test',
})

if __name__ == "__main__":
    print("="*70)
    print("快速测试模式")
    print("="*70)
    print(f"只测试 {len(CONFIG['monotonic_weights'])} 个权重: {CONFIG['monotonic_weights']}")
    print(f"预计时间: {len(CONFIG['monotonic_weights']) * 10} 分钟")
    print("="*70)

    input("\n按 Enter 键开始测试...")

    main()
