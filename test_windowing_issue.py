"""
测试 Many-to-Many 窗口化问题

展示当前实现的问题，以及正确的实现方案
"""

import numpy as np

def current_windowing(features, targets, window_size):
    """当前的实现（有问题）"""
    windowed_features = []
    windowed_targets = []

    for i in range(len(features) - window_size + 1):
        window_feat = features[i:i+window_size]
        window_targ = targets[i:i+window_size]  # ❌ 问题：完全重叠
        windowed_features.append(window_feat)
        windowed_targets.append(window_targ)

    return np.array(windowed_features), np.array(windowed_targets)


def correct_windowing_future(features, targets, window_size):
    """正确的实现：预测未来窗口"""
    windowed_features = []
    windowed_targets = []

    for i in range(len(features) - window_size * 2 + 1):
        window_feat = features[i:i+window_size]
        window_targ = targets[i+window_size:i+window_size*2]  # ✅ 预测下一个窗口
        windowed_features.append(window_feat)
        windowed_targets.append(window_targ)

    return np.array(windowed_features), np.array(windowed_targets)


def correct_windowing_aligned(features, targets, window_size):
    """正确的实现：与 Many-to-One 对齐"""
    windowed_features = []
    windowed_targets = []

    for i in range(len(features) - window_size + 1):
        window_feat = features[i:i+window_size]
        # 目标序列向后偏移，使得最后一个目标对应最后一个特征
        window_targ = targets[i:i+window_size]  # ✅ 但需要因果mask
        windowed_features.append(window_feat)
        windowed_targets.append(window_targ)

    return np.array(windowed_features), np.array(windowed_targets)


if __name__ == "__main__":
    # 创建测试数据
    features = np.arange(20).reshape(-1, 1)  # [0,1,2,...,19]
    targets = np.arange(20) * 10  # [0,10,20,...,190]

    window_size = 5

    print("="*70)
    print("测试数据:")
    print("="*70)
    print(f"特征: {features.flatten()}")
    print(f"目标: {targets}")
    print(f"窗口大小: {window_size}\n")

    # 测试当前实现
    print("="*70)
    print("1. 当前实现（有问题 ❌）")
    print("="*70)
    feat_curr, targ_curr = current_windowing(features, targets, window_size)
    for i in range(min(3, len(feat_curr))):
        print(f"窗口 {i}:")
        print(f"  特征: {feat_curr[i].flatten()}")
        print(f"  目标: {targ_curr[i]}")
        print(f"  问题: 特征[0]={feat_curr[i][0,0]}, 目标[0]={targ_curr[i][0]}")
        print(f"        → 模型在时刻0只能看到特征0，就要预测目标0 ❌")
        print()

    # 测试正确实现 - 预测未来
    print("="*70)
    print("2. 正确实现：预测未来窗口（推荐 ✅）")
    print("="*70)
    feat_future, targ_future = correct_windowing_future(features, targets, window_size)
    for i in range(min(3, len(feat_future))):
        print(f"窗口 {i}:")
        print(f"  特征: {feat_future[i].flatten()}")
        print(f"  目标: {targ_future[i]}")
        print(f"  优势: 特征窗口 [0-4]，预测目标 [5-9]")
        print(f"        → 模型看到完整的历史信息再预测未来 ✅")
        print()

    # 对比 Many-to-One
    print("="*70)
    print("3. Many-to-One（作为对比）")
    print("="*70)
    for i in range(min(3, len(features) - window_size + 1)):
        window_feat = features[i:i+window_size]
        target = targets[i+window_size-1]
        print(f"窗口 {i}:")
        print(f"  特征: {window_feat.flatten()}")
        print(f"  目标: {target}")
        print(f"  逻辑: 特征窗口 [{i}-{i+4}]，预测最后位置 {i+4} 的目标 ✅")
        print()

    print("="*70)
    print("总结")
    print("="*70)
    print("问题根源:")
    print("  当前 Many-to-Many 实现中，每个时间步的特征和目标是同一时刻的")
    print("  这导致模型在预测时只能看到当前时刻的信息，没有历史累积")
    print()
    print("解决方案:")
    print("  方案 1: 预测未来窗口 (features[i:i+w] → targets[i+w:i+2w])")
    print("          优势：逻辑清晰，每个预测都有完整的历史信息")
    print("          缺点：需要更多数据（数据长度减半）")
    print()
    print("  方案 2: 使用因果mask（特征序列不变，但模型内部mask）")
    print("          优势：数据利用率高")
    print("          缺点：需要修改模型架构（添加causal attention）")
    print()
    print("  推荐：方案 1（预测未来窗口）")
