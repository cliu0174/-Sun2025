"""
对比单电池训练 vs 跨电池训练的性能。

帮助理解两种训练方式的差异。
"""

import pandas as pd
import matplotlib.pyplot as plt


def compare_results():
    """对比两种训练模式的结果"""

    print("\n" + "="*70)
    print("单电池训练 vs 跨电池训练 - 性能对比")
    print("="*70)

    # 对比说明
    comparison = {
        '特性': [
            '数据来源',
            '训练样本数',
            '测试方式',
            '测试对象',
            '泛化能力',
            '训练时间',
            '预期MAE',
            '应用场景'
        ],
        '单电池训练': [
            '1个电池 (如1-1)',
            '~1,100',
            '同一电池的后续循环',
            '同一电池的未来数据',
            '低 (特定电池)',
            '2-5分钟',
            '2-3%',
            '特定电池监控'
        ],
        '跨电池训练': [
            '77个电池 (全部HUST)',
            '~85,000',
            '完全未见过的电池',
            '新电池',
            '高 (通用模型)',
            '10-30分钟',
            '3-5%',
            '通用SOH估计'
        ]
    }

    df = pd.DataFrame(comparison)

    print("\n对比表格:")
    print(df.to_string(index=False))

    print("\n" + "="*70)
    print("关键差异")
    print("="*70)

    print("\n📊 单电池训练 (train_single_model.py):")
    print("  优点:")
    print("    ✅ 训练快速")
    print("    ✅ 在同一电池上准确度高")
    print("    ✅ 适合快速原型")
    print("  缺点:")
    print("    ❌ 无法泛化到新电池")
    print("    ❌ 需要每个电池单独训练")

    print("\n🌍 跨电池训练 (train_cross_battery.py):")
    print("  优点:")
    print("    ✅ 可以泛化到新电池")
    print("    ✅ 训练一次，应用到任何电池")
    print("    ✅ 更符合实际应用场景")
    print("  缺点:")
    print("    ❌ 训练时间较长")
    print("    ❌ 个体电池上可能略低于单电池训练")

    print("\n" + "="*70)
    print("推荐使用")
    print("="*70)

    print("\n🎯 使用单电池训练，当你需要:")
    print("  - 快速测试算法")
    print("  - 调试代码")
    print("  - 监控特定电池")

    print("\n🎯 使用跨电池训练，当你需要:")
    print("  - 评估真实泛化能力")
    print("  - 训练通用模型")
    print("  - 论文实验")
    print("  - 实际部署")

    print("\n" + "="*70)
    print("建议工作流程")
    print("="*70)

    print("\n第1步: 单电池快速测试")
    print("  python train_single_model.py")
    print("  → 确保代码能正常运行，模型能收敛")

    print("\n第2步: 跨电池完整训练")
    print("  python train_cross_battery.py")
    print("  → 评估真实泛化能力")

    print("\n第3步: 对比分析")
    print("  python compare_training_modes.py")
    print("  → 理解两种方式的性能差异")

    print("\n" + "="*70)


def plot_expected_performance():
    """绘制预期性能对比图"""

    # 模拟数据
    models = ['FNN', 'CNN', 'LSTM', 'BPINN']
    single_battery_mae = [2.5, 2.3, 2.7, 2.1]  # 单电池训练预期MAE
    cross_battery_mae = [4.2, 3.8, 4.5, 3.5]   # 跨电池训练预期MAE

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(models))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], single_battery_mae, width,
                   label='单电池训练', color='#2ca02c', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], cross_battery_mae, width,
                   label='跨电池训练', color='#1f77b4', alpha=0.8)

    ax.set_ylabel('MAE (%)')
    ax.set_title('预期性能对比：单电池 vs 跨电池训练')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('results/training_mode_comparison.png', dpi=300)
    print("\n图表已保存: results/training_mode_comparison.png")


if __name__ == "__main__":
    import os
    os.makedirs('results', exist_ok=True)

    compare_results()
    plot_expected_performance()

    print("\n" + "="*70)
    print("对比分析完成")
    print("="*70)
    print("\n💡 建议:")
    print("  1. 先用单电池训练测试代码")
    print("  2. 再用跨电池训练评估泛化能力")
    print("  3. 跨电池的测试集MAE才是真实性能！")
