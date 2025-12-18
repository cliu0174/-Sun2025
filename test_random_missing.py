"""
Random Missing 场景测试脚本

测试三种缺失级别 (20%, 40%, 60%) 下的模型性能，
并与场景二 (Uniform Subsampling) 进行对比。
"""

import os
import json
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from train_cross_battery import train_cross_battery_model


# ============================================================================
# 配置区域 - 可以修改这些参数
# ============================================================================

CONFIG = {
    # 基础配置
    'model_type': 'cnn_lstm',
    'seed': 517,

    # 测试的缺失率
    'missing_rates': [0.2, 0.4, 0.5, 0.7, 0.8],  # 20%, 40%, 60% 缺失

    # 是否同时测试 Uniform Subsampling 对比
    'test_uniform_comparison': False,

    # 结果保存路径
    'results_dir': 'results/random_missing_test',
}


# ============================================================================
# 辅助函数
# ============================================================================

def create_results_directory():
    """创建结果保存目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(CONFIG['results_dir'], timestamp)
    os.makedirs(results_path, exist_ok=True)
    return results_path


def run_single_experiment(missing_rate, experiment_name):
    """
    运行单次实验

    Args:
        missing_rate: 缺失率 (0-1)
        experiment_name: 实验名称

    Returns:
        results: 实验结果字典
    """
    print("\n" + "="*70)
    print(f"实验: {experiment_name}")
    print(f"缺失率: {missing_rate*100:.0f}% (保留率: {(1-missing_rate)*100:.0f}%)")
    print("="*70)

    start_time = time.time()
    success = True
    error_msg = None

    try:
        wrapper, results, data_dict = train_cross_battery_model(
            model_type=CONFIG['model_type'],
            degradation_scenario='scenario3',
            random_missing_rate=missing_rate,
            seed=CONFIG['seed']
        )

        elapsed_time = time.time() - start_time

        print("\n" + "-"*70)
        print(f"✓ 实验完成! 耗时: {elapsed_time/60:.1f} 分钟")
        print(f"  RMSE: {results['test_rmse']:.6f}")
        print(f"  MAE:  {results['test_mae']:.6f}")
        print(f"  R²:   {results['test_r2']:.6f}")
        print("-"*70)

    except Exception as e:
        success = False
        error_msg = str(e)
        elapsed_time = time.time() - start_time
        results = None
        wrapper = None
        data_dict = None

        print("\n" + "-"*70)
        print(f"✗ 实验失败! 错误: {error_msg}")
        print("-"*70)

    return {
        'experiment_name': experiment_name,
        'missing_rate': missing_rate,
        'retention_rate': (1 - missing_rate) * 100,
        'success': success,
        'error_msg': error_msg,
        'elapsed_time': elapsed_time,
        'results': results,
        'wrapper': wrapper,
        'data_dict': data_dict
    }


def run_uniform_comparison(retention_rate_target):
    """
    运行 Uniform Subsampling 对比实验

    Args:
        retention_rate_target: 目标保留率 (百分比)

    Returns:
        results: 实验结果字典
    """
    # 计算最接近的 interval
    # retention_rate ≈ 100 / interval
    interval = max(1, round(100 / retention_rate_target))
    actual_retention = 100 / interval

    print("\n" + "="*70)
    print(f"对比实验: Uniform Subsampling")
    print(f"目标保留率: {retention_rate_target:.0f}%, 实际保留率: {actual_retention:.1f}%")
    print(f"采样间隔: {interval}")
    print("="*70)

    start_time = time.time()
    success = True
    error_msg = None

    try:
        wrapper, results, data_dict = train_cross_battery_model(
            model_type=CONFIG['model_type'],
            degradation_scenario='scenario2',
            sparse_sampling_interval=interval,
            seed=CONFIG['seed']
        )

        elapsed_time = time.time() - start_time

        print("\n" + "-"*70)
        print(f"✓ 实验完成! 耗时: {elapsed_time/60:.1f} 分钟")
        print(f"  RMSE: {results['test_rmse']:.6f}")
        print(f"  MAE:  {results['test_mae']:.6f}")
        print(f"  R²:   {results['test_r2']:.6f}")
        print("-"*70)

    except Exception as e:
        success = False
        error_msg = str(e)
        elapsed_time = time.time() - start_time
        results = None
        wrapper = None
        data_dict = None

        print("\n" + "-"*70)
        print(f"✗ 实验失败! 错误: {error_msg}")
        print("-"*70)

    return {
        'experiment_name': f'Uniform (interval={interval})',
        'retention_rate': actual_retention,
        'interval': interval,
        'success': success,
        'error_msg': error_msg,
        'elapsed_time': elapsed_time,
        'results': results,
        'wrapper': wrapper,
        'data_dict': data_dict
    }


def plot_comparison(all_results, save_path):
    """
    绘制对比图

    Args:
        all_results: 所有实验结果列表
        save_path: 保存路径
    """
    # 过滤成功的实验
    success_results = [r for r in all_results if r['success']]

    if len(success_results) == 0:
        print("\n[WARNING] 没有成功的实验，无法绘制对比图")
        return

    # 分离 Random 和 Uniform 实验
    random_results = [r for r in success_results if 'missing_rate' in r]
    uniform_results = [r for r in success_results if 'interval' in r]

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Random Missing vs Uniform Subsampling', fontsize=16, fontweight='bold')

    # 提取数据
    random_retention = [r['retention_rate'] for r in random_results]
    random_rmse = [r['results']['test_rmse'] for r in random_results]
    random_mae = [r['results']['test_mae'] for r in random_results]
    random_r2 = [r['results']['test_r2'] for r in random_results]

    # 子图 1: RMSE 对比
    ax = axes[0, 0]
    ax.plot(random_retention, random_rmse, 'o-', label='Random Missing',
            color='#e74c3c', linewidth=2, markersize=8)
    if uniform_results:
        uniform_retention = [r['retention_rate'] for r in uniform_results]
        uniform_rmse = [r['results']['test_rmse'] for r in uniform_results]
        ax.plot(uniform_retention, uniform_rmse, 's-', label='Uniform Subsampling',
                color='#3498db', linewidth=2, markersize=8)
    ax.set_xlabel('Retention Rate (%)', fontsize=11)
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title('(a) RMSE vs Retention Rate', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # 保留率从高到低

    # 子图 2: MAE 对比
    ax = axes[0, 1]
    ax.plot(random_retention, random_mae, 'o-', label='Random Missing',
            color='#e74c3c', linewidth=2, markersize=8)
    if uniform_results:
        uniform_mae = [r['results']['test_mae'] for r in uniform_results]
        ax.plot(uniform_retention, uniform_mae, 's-', label='Uniform Subsampling',
                color='#3498db', linewidth=2, markersize=8)
    ax.set_xlabel('Retention Rate (%)', fontsize=11)
    ax.set_ylabel('MAE', fontsize=11)
    ax.set_title('(b) MAE vs Retention Rate', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # 子图 3: R² 对比
    ax = axes[1, 0]
    ax.plot(random_retention, random_r2, 'o-', label='Random Missing',
            color='#e74c3c', linewidth=2, markersize=8)
    if uniform_results:
        uniform_r2 = [r['results']['test_r2'] for r in uniform_results]
        ax.plot(uniform_retention, uniform_r2, 's-', label='Uniform Subsampling',
                color='#3498db', linewidth=2, markersize=8)
    ax.set_xlabel('Retention Rate (%)', fontsize=11)
    ax.set_ylabel('R²', fontsize=11)
    ax.set_title('(c) R² vs Retention Rate', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # 子图 4: 结果表格
    ax = axes[1, 1]
    ax.axis('off')

    # 构造表格数据
    table_data = []
    table_data.append(['Method', 'Retention%', 'RMSE', 'MAE', 'R²'])

    for r in random_results:
        table_data.append([
            f"Random ({100-r['retention_rate']:.0f}% missing)",
            f"{r['retention_rate']:.1f}",
            f"{r['results']['test_rmse']:.4f}",
            f"{r['results']['test_mae']:.4f}",
            f"{r['results']['test_r2']:.4f}"
        ])

    for r in uniform_results:
        table_data.append([
            f"Uniform (interval={r['interval']})",
            f"{r['retention_rate']:.1f}",
            f"{r['results']['test_rmse']:.4f}",
            f"{r['results']['test_mae']:.4f}",
            f"{r['results']['test_r2']:.4f}"
        ])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # 设置表头样式
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

    # 保存
    plot_path = os.path.join(save_path, 'random_missing_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 对比图已保存: {plot_path}")

    plt.close()


def generate_report(all_results, save_path):
    """
    生成实验报告

    Args:
        all_results: 所有实验结果列表
        save_path: 保存路径
    """
    report_path = os.path.join(save_path, 'experiment_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Random Missing 场景测试报告\n")
        f.write("="*70 + "\n\n")

        # 实验配置
        f.write("实验配置:\n")
        f.write("-"*70 + "\n")
        f.write(f"  模型类型: {CONFIG['model_type']}\n")
        f.write(f"  随机种子: {CONFIG['seed']}\n")
        f.write(f"  测试缺失率: {CONFIG['missing_rates']}\n")
        f.write(f"  总实验数: {len(all_results)}\n")
        f.write(f"  成功实验数: {sum(1 for r in all_results if r['success'])}\n")
        f.write(f"  失败实验数: {sum(1 for r in all_results if not r['success'])}\n\n")

        # Random Missing 结果
        random_results = [r for r in all_results if r['success'] and 'missing_rate' in r]
        if random_results:
            f.write("Random Missing 结果:\n")
            f.write("-"*70 + "\n")
            f.write(f"{'缺失率':<12} {'保留率':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12}\n")
            f.write("-"*70 + "\n")
            for r in random_results:
                f.write(f"{r['missing_rate']*100:.0f}%{'':<9} "
                       f"{r['retention_rate']:.1f}%{'':<8} "
                       f"{r['results']['test_rmse']:.6f}   "
                       f"{r['results']['test_mae']:.6f}   "
                       f"{r['results']['test_r2']:.6f}\n")
            f.write("\n")

        # Uniform Subsampling 对比
        uniform_results = [r for r in all_results if r['success'] and 'interval' in r]
        if uniform_results:
            f.write("Uniform Subsampling 对比:\n")
            f.write("-"*70 + "\n")
            f.write(f"{'间隔':<12} {'保留率':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12}\n")
            f.write("-"*70 + "\n")
            for r in uniform_results:
                f.write(f"{r['interval']:<12} "
                       f"{r['retention_rate']:.1f}%{'':<8} "
                       f"{r['results']['test_rmse']:.6f}   "
                       f"{r['results']['test_mae']:.6f}   "
                       f"{r['results']['test_r2']:.6f}\n")
            f.write("\n")

        # 失败的实验
        failed_results = [r for r in all_results if not r['success']]
        if failed_results:
            f.write("失败的实验:\n")
            f.write("-"*70 + "\n")
            for r in failed_results:
                f.write(f"  {r['experiment_name']}: {r['error_msg']}\n")
            f.write("\n")

        # 关键发现
        if random_results:
            f.write("关键发现:\n")
            f.write("-"*70 + "\n")

            # 找到最佳/最差结果
            best_rmse = min(random_results, key=lambda x: x['results']['test_rmse'])
            worst_rmse = max(random_results, key=lambda x: x['results']['test_rmse'])

            f.write(f"  最佳结果: 缺失率 {best_rmse['missing_rate']*100:.0f}%, "
                   f"RMSE={best_rmse['results']['test_rmse']:.6f}\n")
            f.write(f"  最差结果: 缺失率 {worst_rmse['missing_rate']*100:.0f}%, "
                   f"RMSE={worst_rmse['results']['test_rmse']:.6f}\n")

            # 计算性能下降
            if len(random_results) > 1:
                rmse_range = worst_rmse['results']['test_rmse'] - best_rmse['results']['test_rmse']
                f.write(f"  RMSE 变化范围: {rmse_range:.6f} "
                       f"({rmse_range/best_rmse['results']['test_rmse']*100:.1f}% 相对增长)\n")

            f.write("\n")

        f.write("="*70 + "\n")
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n")

    print(f"✓ 实验报告已保存: {report_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("\n" + "#"*70)
    print("Random Missing 场景测试")
    print("#"*70)
    print(f"\n模型: {CONFIG['model_type']}")
    print(f"测试缺失率: {CONFIG['missing_rates']}")
    print(f"种子: {CONFIG['seed']}")

    # 估算时间
    n_experiments = len(CONFIG['missing_rates'])
    if CONFIG['test_uniform_comparison']:
        n_experiments += len(CONFIG['missing_rates'])
    estimated_time = n_experiments * 10  # 假设每个实验10分钟
    print(f"\n预计实验数: {n_experiments}")
    print(f"预计总耗时: {estimated_time} 分钟 ({estimated_time/60:.1f} 小时)")

    # 创建结果目录
    results_path = create_results_directory()
    print(f"\n结果保存路径: {results_path}")

    # 开始实验
    print("\n" + "#"*70)
    print("开始实验")
    print("#"*70)

    all_results = []

    # Random Missing 实验
    for i, rate in enumerate(CONFIG['missing_rates'], 1):
        print(f"\n进度: {i}/{len(CONFIG['missing_rates'])} (Random Missing)")
        result = run_single_experiment(
            missing_rate=rate,
            experiment_name=f"Random Missing (rate={rate*100:.0f}%)"
        )
        all_results.append(result)

    # Uniform Subsampling 对比实验
    if CONFIG['test_uniform_comparison']:
        print("\n" + "#"*70)
        print("Uniform Subsampling 对比实验")
        print("#"*70)

        for i, rate in enumerate(CONFIG['missing_rates'], 1):
            retention_rate = (1 - rate) * 100
            print(f"\n进度: {i}/{len(CONFIG['missing_rates'])} (Uniform)")
            result = run_uniform_comparison(retention_rate_target=retention_rate)
            all_results.append(result)

    # 保存原始结果
    results_file = os.path.join(results_path, 'all_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\n✓ 原始结果已保存: {results_file}")

    # 保存 JSON 摘要
    summary = []
    for r in all_results:
        if r['success']:
            summary.append({
                'experiment_name': r['experiment_name'],
                'retention_rate': r.get('retention_rate'),
                'rmse': r['results']['test_rmse'],
                'mae': r['results']['test_mae'],
                'r2': r['results']['test_r2']
            })

    summary_file = os.path.join(results_path, 'results_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✓ 结果摘要已保存: {summary_file}")

    # 生成对比图
    print("\n" + "#"*70)
    print("生成对比图")
    print("#"*70)
    plot_comparison(all_results, results_path)

    # 生成报告
    print("\n" + "#"*70)
    print("生成实验报告")
    print("#"*70)
    generate_report(all_results, results_path)

    print("\n" + "="*70)
    print("所有实验完成!")
    print("="*70)
    print(f"结果保存在: {results_path}")
    print(f"  - 对比图: random_missing_comparison.png")
    print(f"  - 报告: experiment_report.txt")
    print(f"  - 原始数据: all_results.pkl")
    print(f"  - 摘要: results_summary.json")
    print("="*70)


if __name__ == "__main__":
    main()
