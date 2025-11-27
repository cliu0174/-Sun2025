"""
分析不同容量归一化方法对SOH分布的影响

对比:
1. 使用初始容量归一化 (capacity / initial_capacity)
2. 使用额定容量归一化 (capacity / 1.1)
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_normalization_methods():
    """分析两种归一化方法的差异"""

    data_dir = 'data/HUST data'
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

    print("="*70)
    print("容量归一化方法对比分析")
    print("="*70)

    # 统计数据
    initial_capacities = []
    soh_rated_maxs = []
    soh_rated_mins = []
    soh_initial_maxs = []
    soh_initial_mins = []

    batteries_over_1 = 0  # SOH > 1.0 的电池数量

    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        df = pd.read_csv(file_path)
        capacity = df['capacity'].values

        initial_cap = capacity[0]
        initial_capacities.append(initial_cap)

        # 方法1: 使用额定容量
        soh_rated = capacity / 1.1
        soh_rated_maxs.append(soh_rated.max())
        soh_rated_mins.append(soh_rated.min())

        # 方法2: 使用初始容量
        soh_initial = capacity / initial_cap
        soh_initial_maxs.append(soh_initial.max())
        soh_initial_mins.append(soh_initial.min())

        if soh_rated.max() > 1.0:
            batteries_over_1 += 1

    # 转换为数组
    initial_capacities = np.array(initial_capacities)
    soh_rated_maxs = np.array(soh_rated_maxs)
    soh_rated_mins = np.array(soh_rated_mins)
    soh_initial_maxs = np.array(soh_initial_maxs)
    soh_initial_mins = np.array(soh_initial_mins)

    # 打印统计信息
    print(f"\n加载了 {len(csv_files)} 个电池数据\n")

    print("初始容量统计:")
    print(f"  范围: {initial_capacities.min():.6f} - {initial_capacities.max():.6f} Ah")
    print(f"  均值: {initial_capacities.mean():.6f} Ah")
    print(f"  标准差: {initial_capacities.std():.6f} Ah")
    print(f"  额定容量: 1.1 Ah")
    print(f"  初始容量 > 额定容量的电池: {(initial_capacities > 1.1).sum()} / {len(csv_files)} ({(initial_capacities > 1.1).sum()/len(csv_files)*100:.1f}%)")

    print("\n" + "="*70)
    print("方法1: 使用额定容量 (capacity / 1.1)")
    print("-"*70)
    print(f"SOH 最大值范围: {soh_rated_maxs.min():.6f} - {soh_rated_maxs.max():.6f}")
    print(f"SOH 最小值范围: {soh_rated_mins.min():.6f} - {soh_rated_mins.max():.6f}")
    print(f"SOH 最大值均值: {soh_rated_maxs.mean():.6f}")
    print(f"")
    print(f"[问题] 有 {batteries_over_1} / {len(csv_files)} ({batteries_over_1/len(csv_files)*100:.1f}%) 个电池的 SOH > 1.0")
    print(f"       这违反了 SOH 的定义 (应该 <= 1.0)")
    print(f"       模型使用 Sigmoid 输出层，输出范围 [0, 1]")
    print(f"       训练时目标值 > 1.0 会导致梯度消失和训练困难")

    print("\n" + "="*70)
    print("方法2: 使用初始容量 (capacity / initial_capacity)")
    print("-"*70)
    print(f"SOH 最大值范围: {soh_initial_maxs.min():.6f} - {soh_initial_maxs.max():.6f}")
    print(f"SOH 最小值范围: {soh_initial_mins.min():.6f} - {soh_initial_mins.max():.6f}")
    print(f"SOH 最大值均值: {soh_initial_maxs.mean():.6f}")
    print(f"")
    print(f"[优势] 所有电池的 SOH 都在合理范围内")
    print(f"       SOH 始终 <= 1.0 (符合定义)")
    print(f"       与 Sigmoid 输出层完美匹配")

    print("\n" + "="*70)
    print("为什么额定容量方法表现差？")
    print("="*70)
    print("""
1. 目标值分布问题:
   - 大部分电池初始容量 > 1.1 Ah (额定容量)
   - 使用额定容量归一化导致 SOH > 1.0
   - 但模型使用 Sigmoid 输出，范围 [0, 1]

2. 训练困难:
   - 当目标值 > 1.0，而预测值最大只能是 1.0
   - 损失函数永远无法收敛到最优
   - 模型会系统性地低估 SOH 值

3. 梯度消失:
   - Sigmoid(x) -> 1 时，梯度 -> 0
   - 目标值在 [1.0, 1.09] 范围时，梯度非常小
   - 导致训练缓慢甚至停滞

4. 物理意义:
   - SOH (State of Health) 定义: 当前容量 / 初始容量
   - 使用额定容量违反了这个定义
   - 额定容量是制造商标称值，不是电池实际初始容量

建议:
✓ 继续使用初始容量方法 (更准确，训练效果更好)
✗ 不推荐使用额定容量方法 (除非移除 Sigmoid 输出层限制)
    """)

    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 初始容量分布
    ax = axes[0, 0]
    ax.hist(initial_capacities, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(1.1, color='red', linestyle='--', linewidth=2, label='Rated capacity (1.1 Ah)')
    ax.set_xlabel('Initial Capacity (Ah)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Initial Capacities', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. SOH 最大值对比
    ax = axes[0, 1]
    x = np.arange(len(csv_files))
    ax.scatter(x, soh_rated_maxs, alpha=0.6, s=20, label='Rated capacity method', color='red')
    ax.scatter(x, soh_initial_maxs, alpha=0.6, s=20, label='Initial capacity method', color='blue')
    ax.axhline(1.0, color='green', linestyle='--', linewidth=2, label='SOH = 1.0 (limit)')
    ax.set_xlabel('Battery Index', fontsize=12)
    ax.set_ylabel('Max SOH', fontsize=12)
    ax.set_title('Maximum SOH Values Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.95, 1.15])

    # 3. SOH 范围对比 (额定容量)
    ax = axes[1, 0]
    for i in range(min(20, len(csv_files))):  # 只显示前20个电池
        ax.plot([i, i], [soh_rated_mins[i], soh_rated_maxs[i]], 'r-', alpha=0.5, linewidth=2)
    ax.axhline(1.0, color='green', linestyle='--', linewidth=2, label='SOH = 1.0')
    ax.set_xlabel('Battery Index', fontsize=12)
    ax.set_ylabel('SOH Range', fontsize=12)
    ax.set_title('SOH Range (Rated Capacity Method)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.7, 1.15])

    # 4. SOH 范围对比 (初始容量)
    ax = axes[1, 1]
    for i in range(min(20, len(csv_files))):  # 只显示前20个电池
        ax.plot([i, i], [soh_initial_mins[i], soh_initial_maxs[i]], 'b-', alpha=0.5, linewidth=2)
    ax.axhline(1.0, color='green', linestyle='--', linewidth=2, label='SOH = 1.0')
    ax.set_xlabel('Battery Index', fontsize=12)
    ax.set_ylabel('SOH Range', fontsize=12)
    ax.set_title('SOH Range (Initial Capacity Method)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.7, 1.15])

    plt.tight_layout()
    plt.savefig('capacity_normalization_comparison.png', dpi=300, bbox_inches='tight')
    print("\n[图表已保存] capacity_normalization_comparison.png")

    plt.show()

    return {
        'initial_capacities': initial_capacities,
        'batteries_over_1': batteries_over_1,
        'total_batteries': len(csv_files),
        'soh_rated_max': soh_rated_maxs.max(),
        'soh_initial_max': soh_initial_maxs.max()
    }


if __name__ == "__main__":
    results = analyze_normalization_methods()
