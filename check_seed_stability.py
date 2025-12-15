"""
检查随机种子对数据划分和训练的影响

分析:
1. 不同种子下的电池划分
2. 各个电池的难度分布
3. 模型初始化的影响
"""

import os
import random
import numpy as np
import json

def check_battery_split_variance(seeds=[42, 999, 123, 456, 789]):
    """检查不同种子下的电池划分差异"""
    print("="*70)
    print("检查随机种子对数据划分的影响")
    print("="*70)

    # 获取所有电池名称
    data_dir = 'data/HUST data'
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    battery_names = [f.replace('.csv', '') for f in csv_files]

    print(f"\n总电池数: {len(battery_names)}")
    print(f"测试种子: {seeds}\n")

    # 记录每个电池在不同种子下被分配到哪个集合
    battery_assignments = {b: [] for b in battery_names}

    train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)

        # 划分
        shuffled = battery_names.copy()
        random.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_batteries = shuffled[:n_train]
        val_batteries = shuffled[n_train:n_train + n_val]
        test_batteries = shuffled[n_train + n_val:]

        # 记录分配
        for b in train_batteries:
            battery_assignments[b].append('train')
        for b in val_batteries:
            battery_assignments[b].append('val')
        for b in test_batteries:
            battery_assignments[b].append('test')

    # 分析稳定性
    stable_batteries = {
        'always_train': [],
        'always_val': [],
        'always_test': [],
        'mixed': []
    }

    for battery, assignments in battery_assignments.items():
        unique_assignments = set(assignments)

        if len(unique_assignments) == 1:
            if 'train' in unique_assignments:
                stable_batteries['always_train'].append(battery)
            elif 'val' in unique_assignments:
                stable_batteries['always_val'].append(battery)
            else:
                stable_batteries['always_test'].append(battery)
        else:
            stable_batteries['mixed'].append(battery)

    # 打印统计
    print("电池分配稳定性分析:")
    print(f"  总是在训练集: {len(stable_batteries['always_train'])} 个")
    print(f"  总是在验证集: {len(stable_batteries['always_val'])} 个")
    print(f"  总是在测试集: {len(stable_batteries['always_test'])} 个")
    print(f"  不稳定(会变): {len(stable_batteries['mixed'])} 个")

    instability_rate = len(stable_batteries['mixed']) / len(battery_names) * 100
    print(f"\n不稳定率: {instability_rate:.1f}%")

    if instability_rate > 50:
        print("\n⚠️  警告: 超过50%的电池在不同种子下分配不同!")
        print("   这会导致模型性能大幅波动")

    # 检查测试集的重叠
    print("\n" + "="*70)
    print("测试集重叠分析 (Seed 42 vs Seed 999)")
    print("="*70)

    # Seed 42
    random.seed(42)
    shuffled_42 = battery_names.copy()
    random.shuffle(shuffled_42)
    test_42 = set(shuffled_42[int(len(shuffled_42)*0.8):])

    # Seed 999
    random.seed(999)
    shuffled_999 = battery_names.copy()
    random.shuffle(shuffled_999)
    test_999 = set(shuffled_999[int(len(shuffled_999)*0.8):])

    overlap = test_42 & test_999
    only_42 = test_42 - test_999
    only_999 = test_999 - test_42

    print(f"Seed 42 测试集: {len(test_42)} 个电池")
    print(f"Seed 999 测试集: {len(test_999)} 个电池")
    print(f"重叠电池: {len(overlap)} 个 ({len(overlap)/len(test_42)*100:.1f}%)")
    print(f"仅在42: {len(only_42)} 个")
    print(f"仅在999: {len(only_999)} 个")

    if len(overlap) / len(test_42) < 0.5:
        print("\n⚠️  警告: 测试集重叠率 < 50%!")
        print("   两个种子测试的是几乎不同的电池,结果不可比!")

    return stable_batteries, test_42, test_999


def analyze_model_init_variance():
    """分析模型初始化的影响"""
    print("\n" + "="*70)
    print("分析模型初始化随机性")
    print("="*70)

    import torch
    import torch.nn as nn

    # 简单模型测试
    class SimpleModel(nn.Module):
        def __init__(self, seed):
            super().__init__()
            torch.manual_seed(seed)
            self.fc = nn.Linear(16, 1)

        def forward(self, x):
            return self.fc(x)

    # 不同种子初始化
    model_42 = SimpleModel(42)
    model_999 = SimpleModel(999)

    # 提取权重
    w_42 = model_42.fc.weight.detach().numpy()
    w_999 = model_999.fc.weight.detach().numpy()

    # 计算差异
    weight_diff = np.abs(w_42 - w_999).mean()

    print(f"初始权重差异 (Seed 42 vs 999): {weight_diff:.6f}")

    if weight_diff > 0.5:
        print("⚠️  初始权重差异较大,会影响收敛路径")

    return weight_diff


def check_noise_impact():
    """检查加噪声时随机种子的影响"""
    print("\n" + "="*70)
    print("检查噪声随机性的影响")
    print("="*70)

    # 模拟数据
    capacity = np.linspace(1.0, 0.7, 100)

    # 两次加噪(不同种子)
    noise_42 = np.random.RandomState(42).normal(0, 0.005, 100)
    noise_999 = np.random.RandomState(999).normal(0, 0.005, 100)

    noisy_42 = capacity + noise_42
    noisy_999 = capacity + noise_999

    # 计算差异
    diff = np.abs(noisy_42 - noisy_999).mean()

    print(f"加噪后数据差异: {diff:.6f}")
    print(f"噪声标准差: 0.005")
    print(f"差异/标准差比: {diff/0.005:.2f}")

    if diff > 0.01:
        print("⚠️  不同种子的噪声差异较大")


def main():
    """运行所有检查"""
    # 1. 数据划分影响
    stable_batteries, test_42, test_999 = check_battery_split_variance()

    # 2. 模型初始化影响
    weight_diff = analyze_model_init_variance()

    # 3. 噪声影响
    check_noise_impact()

    # 总结
    print("\n" + "="*70)
    print("总结: 随机种子导致结果差异的可能原因")
    print("="*70)

    print("\n1. 数据划分差异")
    print(f"   - 不稳定电池: {len(stable_batteries['mixed'])} 个")
    print(f"   - 测试集重叠率: {len(test_42 & test_999)/len(test_42)*100:.1f}%")
    print("   影响: ⚠️⚠️⚠️ 高 (如果重叠率<50%)")

    print("\n2. 模型初始化差异")
    print(f"   - 初始权重差异: {weight_diff:.6f}")
    print("   影响: ⚠️ 中等 (通过训练可以收敛)")

    print("\n3. 噪声差异 (如果启用ADD_NOISE)")
    print("   影响: ⚠️ 中等")

    print("\n" + "="*70)
    print("建议:")
    print("="*70)
    print("1. 固定一个种子进行对比实验 (推荐seed=42)")
    print("2. 运行多个种子取平均 (5-10个种子)")
    print("3. 使用K-fold交叉验证代替随机划分")
    print("4. 分析哪些电池难预测,检查是否有异常电池")


if __name__ == "__main__":
    main()
