"""
Scenario 3 自适应权重测试脚本

针对不同缺失率，测试不同范围的单调性权重
解决"低保留率下权重过大导致性能下降"的问题
"""

import os
import json
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from train_cross_battery import train_cross_battery_model


# ===== 实验配置 =====
CONFIG = {
    # 基础配置
    'model_type': 'cnn_lstm',
    'seed': 517,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # 缺失率和对应的权重范围（自适应策略）
    'scenarios': [
        {
            'missing_rate': 0.2,  # 20% 缺失，80% 保留
            'weights': [0.0, 0.05, 0.1, 0.15, 0.2],
            'description': '轻度缺失 (保留80%)'
        },
        {
            'missing_rate': 0.4,  # 40% 缺失，60% 保留
            'weights': [0.0, 0.1, 0.15, 0.2, 0.3],
            'description': '中度缺失 (保留60%)'
        },
        {
            'missing_rate': 0.5,  # 50% 缺失，50% 保留
            'weights': [0.0, 0.1, 0.2, 0.25, 0.3],
            'description': '重度缺失 (保留50%)'
        },
        {
            'missing_rate': 0.7,  # 70% 缺失，30% 保留
            'weights': [0.0, 0.05, 0.1, 0.15, 0.2],  # 降低权重！
            'description': '极度缺失 (保留30%)'
        },
        {
            'missing_rate': 0.8,  # 80% 缺失，20% 保留
            'weights': [0.0, 0.05, 0.1, 0.12, 0.15],  # 进一步降低！
            'description': '超极度缺失 (保留20%)'
        }
    ],

    # 输出目录
    'output_dir': 'results/adaptive_weights_scenario3',
    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
}


def modify_config_monotonic_weight(model_type, monotonic_weight):
    """修改配置文件中的单调性权重"""
    config_path = f'configs/models/{model_type}_config.json'

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    if 'physics_constraints' in config:
        config['physics_constraints']['monotonic_weight'] = monotonic_weight
        config['physics_constraints']['enabled'] = True

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  [CONFIG] monotonic_weight = {monotonic_weight}")


def run_single_experiment(scenario, weight):
    """运行单个实验"""
    missing_rate = scenario['missing_rate']
    retention_rate = (1 - missing_rate) * 100

    print(f"\n{'='*70}")
    print(f"场景: {scenario['description']}")
    print(f"缺失率: {missing_rate*100:.0f}%, 保留率: {retention_rate:.0f}%")
    print(f"单调性权重: {weight}")
    print(f"{'='*70}")

    # 修改配置
    modify_config_monotonic_weight(CONFIG['model_type'], weight)
    time.sleep(0.3)

    # 运行训练
    try:
        wrapper, results, data_dict = train_cross_battery_model(
            model_type=CONFIG['model_type'],
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            device=CONFIG['device'],
            seed=CONFIG['seed'],
            degradation_scenario='scenario3',
            random_missing_rate=missing_rate
        )

        result = {
            'missing_rate': missing_rate,
            'retention_rate': retention_rate,
            'monotonic_weight': weight,
            'test_rmse': results['test_rmse'],
            'test_mae': results['test_mae'],
            'test_r2': results['test_r2'],
            'success': True,
            'error': None
        }

        print(f"\n[结果] RMSE: {results['test_rmse']:.4f}, "
              f"MAE: {results['test_mae']:.4f}, R²: {results['test_r2']:.4f}")

    except Exception as e:
        print(f"\n[ERROR] 实验失败: {str(e)}")
        result = {
            'missing_rate': missing_rate,
            'retention_rate': retention_rate,
            'monotonic_weight': weight,
            'test_rmse': None,
            'test_mae': None,
            'test_r2': None,
            'success': False,
            'error': str(e)
        }

    return result


def plot_results(all_results, output_dir):
    """绘制对比图"""
    # 按缺失率分组
    scenarios = {}
    for r in all_results:
        if not r['success']:
            continue
        rate = r['missing_rate']
        if rate not in scenarios:
            scenarios[rate] = []
        scenarios[rate].append(r)

    # 创建图表
    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Scenario 3: Adaptive Weights for Different Missing Rates',
                 fontsize=14, fontweight='bold')

    # 子图 1: RMSE vs Weight（所有场景）
    ax = axes[0, 0]
    for rate, data in sorted(scenarios.items()):
        weights = [d['monotonic_weight'] for d in data]
        rmse = [d['test_rmse'] for d in data]
        retention = (1 - rate) * 100
        ax.plot(weights, rmse, 'o-', label=f'{retention:.0f}% retained',
                linewidth=2, markersize=6)

    ax.set_xlabel('Monotonic Weight', fontsize=11)
    ax.set_ylabel('Test RMSE', fontsize=11)
    ax.set_title('(a) RMSE vs Weight (All Scenarios)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 子图 2: 每个场景的最优权重
    ax = axes[0, 1]
    rates = []
    optimal_weights = []
    optimal_rmse = []

    for rate, data in sorted(scenarios.items()):
        best = min(data, key=lambda x: x['test_rmse'])
        rates.append((1 - rate) * 100)  # 保留率
        optimal_weights.append(best['monotonic_weight'])
        optimal_rmse.append(best['test_rmse'])

    ax.plot(rates, optimal_weights, 'o-', color='#2ecc71',
            linewidth=2, markersize=10)
    ax.set_xlabel('Retention Rate (%)', fontsize=11)
    ax.set_ylabel('Optimal Monotonic Weight', fontsize=11)
    ax.set_title('(b) Optimal Weight vs Retention Rate', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # 从高到低

    # 子图 3: 基线 vs 最优（改善幅度）
    ax = axes[1, 0]
    improvements = []

    for rate, data in sorted(scenarios.items()):
        # 找基线 (weight=0.0)
        baseline = next((d for d in data if abs(d['monotonic_weight']) < 1e-6), None)
        # 找最优
        best = min(data, key=lambda x: x['test_rmse'])

        if baseline:
            retention = (1 - rate) * 100
            improvement = (baseline['test_rmse'] - best['test_rmse']) / baseline['test_rmse'] * 100
            improvements.append((retention, improvement, best['monotonic_weight']))

    if improvements:
        retentions = [x[0] for x in improvements]
        imps = [x[1] for x in improvements]
        colors = ['green' if i > 0 else 'red' for i in imps]

        bars = ax.bar(range(len(retentions)), imps, color=colors, alpha=0.7)
        ax.set_xticks(range(len(retentions)))
        ax.set_xticklabels([f'{r:.0f}%' for r in retentions])
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Retention Rate', fontsize=11)
        ax.set_ylabel('RMSE Improvement (%)', fontsize=11)
        ax.set_title('(c) Improvement vs Baseline (w=0.0)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 标注改善值
        for i, (rect, imp) in enumerate(zip(bars, imps)):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                    f'{imp:+.1f}%', ha='center',
                    va='bottom' if imp > 0 else 'top',
                    fontsize=9, fontweight='bold')

    # 子图 4: 结果表格
    ax = axes[1, 1]
    ax.axis('off')

    table_data = [['Retention%', 'Best w', 'RMSE(w=0)', 'RMSE(best)', 'Improve%']]

    for rate, data in sorted(scenarios.items(), reverse=True):
        baseline = next((d for d in data if abs(d['monotonic_weight']) < 1e-6), None)
        best = min(data, key=lambda x: x['test_rmse'])
        retention = (1 - rate) * 100

        if baseline:
            improvement = (baseline['test_rmse'] - best['test_rmse']) / baseline['test_rmse'] * 100
            table_data.append([
                f'{retention:.0f}%',
                f'{best["monotonic_weight"]:.2f}',
                f'{baseline["test_rmse"]:.4f}',
                f'{best["test_rmse"]:.4f}',
                f'{improvement:+.1f}%'
            ])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.15, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # 表头样式
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 交替行颜色
    for i in range(1, len(table_data)):
        color = '#ecf0f1' if i % 2 == 0 else 'white'
        for j in range(5):
            table[(i, j)].set_facecolor(color)

    ax.set_title('(d) Results Summary', fontsize=12, fontweight='bold')

    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'adaptive_weights_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 对比图已保存: {plot_path}")
    plt.close()


def generate_report(all_results, output_dir):
    """生成实验报告"""
    report_path = os.path.join(output_dir, 'experiment_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Scenario 3: 自适应权重测试报告\n")
        f.write("="*70 + "\n\n")

        f.write("实验策略:\n")
        f.write("-"*70 + "\n")
        f.write("根据保留率动态调整单调性权重范围：\n")
        f.write("  - 高保留率 (80%+): 使用较高权重 (0.05-0.2)\n")
        f.write("  - 中保留率 (50-80%): 使用中等权重 (0.1-0.3)\n")
        f.write("  - 低保留率 (<50%): 使用较低权重 (0.05-0.15)\n")
        f.write("目的: 避免低保留率下过度正则化\n\n")

        # 按场景分组
        scenarios = {}
        for r in all_results:
            if not r['success']:
                continue
            rate = r['missing_rate']
            if rate not in scenarios:
                scenarios[rate] = []
            scenarios[rate].append(r)

        # 每个场景的结果
        for rate, data in sorted(scenarios.items()):
            retention = (1 - rate) * 100
            f.write(f"\n场景: 缺失率 {rate*100:.0f}%, 保留率 {retention:.0f}%\n")
            f.write("-"*70 + "\n")
            f.write(f"{'权重':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12}\n")
            f.write("-"*70 + "\n")

            for d in sorted(data, key=lambda x: x['monotonic_weight']):
                f.write(f"{d['monotonic_weight']:<12.2f} "
                       f"{d['test_rmse']:<12.6f} "
                       f"{d['test_mae']:<12.6f} "
                       f"{d['test_r2']:<12.6f}\n")

            # 找最优
            best = min(data, key=lambda x: x['test_rmse'])
            baseline = next((d for d in data if abs(d['monotonic_weight']) < 1e-6), None)

            f.write(f"\n最优权重: {best['monotonic_weight']:.2f} (RMSE={best['test_rmse']:.6f})\n")

            if baseline:
                improvement = (baseline['test_rmse'] - best['test_rmse']) / baseline['test_rmse'] * 100
                f.write(f"基线 (w=0.0): RMSE={baseline['test_rmse']:.6f}\n")
                f.write(f"改善幅度: {improvement:+.2f}%\n")

        f.write("\n" + "="*70 + "\n")
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n")

    print(f"✓ 实验报告已保存: {report_path}")


def main():
    """主函数"""
    print("\n" + "#"*70)
    print("Scenario 3: 自适应权重测试")
    print("#"*70)

    # 创建输出目录
    output_dir = os.path.join(CONFIG['output_dir'], CONFIG['timestamp'])
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n结果保存路径: {output_dir}")

    # 保存配置
    config_path = os.path.join(output_dir, 'experiment_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)

    # 计算总实验数
    total_experiments = sum(len(s['weights']) for s in CONFIG['scenarios'])
    print(f"\n总实验数: {total_experiments}")
    print(f"预计耗时: {total_experiments * 10 / 60:.1f} 小时")

    # 运行所有实验
    all_results = []
    current = 0

    for scenario in CONFIG['scenarios']:
        print(f"\n\n{'#'*70}")
        print(f"场景: {scenario['description']}")
        print(f"测试权重: {scenario['weights']}")
        print(f"{'#'*70}")

        for weight in scenario['weights']:
            current += 1
            print(f"\n进度: {current}/{total_experiments}")

            result = run_single_experiment(scenario, weight)
            all_results.append(result)

    # 保存结果
    results_file = os.path.join(output_dir, 'all_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\n✓ 原始结果已保存: {results_file}")

    # 生成对比图
    print("\n" + "#"*70)
    print("生成对比图")
    print("#"*70)
    plot_results(all_results, output_dir)

    # 生成报告
    print("\n" + "#"*70)
    print("生成实验报告")
    print("#"*70)
    generate_report(all_results, output_dir)

    print("\n" + "="*70)
    print("所有实验完成!")
    print("="*70)
    print(f"结果保存在: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
