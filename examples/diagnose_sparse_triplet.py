"""
诊断稀疏采样场景下三元组采样失败的原因

目的：检查稀疏采样后的数据是否满足三元组采样的条件
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def simulate_sparse_sampling_impact():
    """模拟稀疏采样对三元组采样的影响"""
    print("="*70)
    print("诊断：场景二稀疏采样 + 三元组采样兼容性")
    print("="*70)

    # 模拟单个电池的数据
    print("\n假设：单个电池原始有 1000 个周期")
    n_cycles_original = 1000
    step_k = 1  # 三元组步长

    # 原始数据
    print(f"\n场景：密集数据（无采样）")
    print(f"  周期索引: [0, 1, 2, 3, ..., 999]")
    print(f"  样本数: {n_cycles_original}")

    # 三元组数量计算
    n_triplets_original = n_cycles_original - 2 * step_k
    print(f"  可用三元组数: {n_triplets_original}")
    print(f"  示例三元组: (0,1,2), (1,2,3), ..., (997,998,999)")

    # 稀疏采样后
    sampling_intervals = {
        'moderate (间隔5)': 5,
        'sparse (间隔10)': 10,
        'very_sparse (间隔20)': 20
    }

    print("\n" + "="*70)
    print("稀疏采样后的影响：")
    print("="*70)

    for level_name, interval in sampling_intervals.items():
        print(f"\n{level_name}:")

        # 稀疏采样后的周期索引
        sampled_cycles = np.arange(0, n_cycles_original, interval)
        n_sampled = len(sampled_cycles)

        print(f"  周期索引: {sampled_cycles[:5].tolist()}...{sampled_cycles[-3:].tolist()}")
        print(f"  样本数: {n_sampled} ({n_sampled/n_cycles_original*100:.1f}%)")

        # 检查三元组条件
        # 三元组要求：t, t+step_k, t+2*step_k 都存在
        # 在稀疏采样后：索引间隔变为 interval

        # 使用我们修复后的逻辑
        valid_triplets = 0
        for i in range(len(sampled_cycles) - 2 * step_k):
            cycle_diff_1 = sampled_cycles[i + step_k] - sampled_cycles[i]
            cycle_diff_2 = sampled_cycles[i + 2 * step_k] - sampled_cycles[i + step_k]

            # 新逻辑：允许等间隔
            if cycle_diff_1 > 0 and cycle_diff_1 == cycle_diff_2:
                valid_triplets += 1

        print(f"  可用三元组数: {valid_triplets}")

        if valid_triplets == 0:
            print(f"  [FAIL] 无可用三元组！三元组采样会失败！")
        else:
            expected = n_sampled - 2 * step_k
            print(f"  [OK] 三元组可用 ({valid_triplets}/{expected})")
            print(f"  示例三元组: cycles[{sampled_cycles[0]}, {sampled_cycles[1]}, {sampled_cycles[2]}]")


def check_按电池采样的影响():
    """检查按电池稀疏采样对三元组的影响"""
    print("\n" + "="*70)
    print("跨电池训练场景（46个电池）")
    print("="*70)

    n_batteries = 46
    avg_cycles = 800  # 平均周期数

    scenarios = {
        '密集数据': {'interval': 1, 'samples_per_battery': 800},
        '场景二 moderate': {'interval': 5, 'samples_per_battery': 160},
        '场景二 sparse': {'interval': 10, 'samples_per_battery': 80},
    }

    print(f"\n假设：{n_batteries}个电池，平均每个{avg_cycles}周期")
    print(f"{'场景':<20} {'每电池样本':<12} {'每电池三元组':<15} {'总三元组数':<12}")
    print("-"*70)

    for scenario_name, params in scenarios.items():
        samples = params['samples_per_battery']
        triplets_per_battery = max(0, samples - 2)  # step_k=1
        total_triplets = triplets_per_battery * n_batteries

        print(f"{scenario_name:<20} {samples:<12} {triplets_per_battery:<15} {total_triplets:<12,}")


def diagnose_实际问题():
    """诊断实际问题的可能原因"""
    print("\n" + "="*70)
    print("问题诊断：为什么场景二不显示Mask激活比例？")
    print("="*70)

    print("\n可能原因分析：")

    print("\n1. Dataset长度为0")
    print("   - 稀疏采样后，如果三元组验证逻辑未修复")
    print("   - 所有样本被过滤，valid_indices为空")
    print("   - Dataset.__len__() 返回 0")
    print("   - 训练时无数据，曲率损失=0")

    print("\n2. 修复后的逻辑未生效")
    print("   - 检查 data_loaders/data_loader_hust.py 是否已修改")
    print("   - 确认修改的是正确的文件路径")
    print("   - 可能需要重新导入模块")

    print("\n3. cycle_indices 重建问题")
    print("   - 稀疏采样后，cycle_indices 仍保留原始值 (0,5,10,...)")
    print("   - 但数组索引变为 (0,1,2,...)")
    print("   - 验证逻辑使用的是哪个？")


def solution_checklist():
    """解决方案检查清单"""
    print("\n" + "="*70)
    print("解决方案检查清单")
    print("="*70)

    checklist = [
        {
            "步骤": "1. 确认修复已应用",
            "检查": "data_loaders/data_loader_hust.py 第429行",
            "期望": "if cycle_diff_1 > 0 and cycle_diff_1 == cycle_diff_2:"
        },
        {
            "步骤": "2. 重新运行训练",
            "检查": "python train_cross_battery.py",
            "期望": "看到 '场景二: 按电池稀疏采样' 的输出"
        },
        {
            "步骤": "3. 检查Dataset长度",
            "检查": "训练输出中查找 'Dataset 长度'",
            "期望": "> 0，不是0"
        },
        {
            "步骤": "4. 确认三元组模式",
            "检查": "训练输出中查找 '三元组采样: 启用'",
            "期望": "显示 'step_k=1'"
        },
        {
            "步骤": "5. 验证Mask激活",
            "检查": "训练输出中查找 'Mask激活比例'",
            "期望": "> 0%，如果有数据的话"
        }
    ]

    for item in checklist:
        print(f"\n{item['步骤']}")
        print(f"  检查项: {item['检查']}")
        print(f"  期望结果: {item['期望']}")


if __name__ == "__main__":
    print("\n稀疏采样 + 三元组采样诊断工具\n")

    # 1. 模拟影响
    simulate_sparse_sampling_impact()

    # 2. 跨电池场景
    check_按电池采样的影响()

    # 3. 问题诊断
    diagnose_实际问题()

    # 4. 解决方案
    solution_checklist()

    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print("\n核心问题：稀疏采样后，如果三元组验证逻辑未修复，")
    print("所有三元组会被过滤，导致 Dataset 长度=0。")
    print("\n解决方案：")
    print("1. 确认 data_loaders/data_loader_hust.py 已修改")
    print("2. 重新运行训练并检查输出")
    print("3. 如果仍然失败，提供完整的训练输出进行诊断")
    print("="*70)
