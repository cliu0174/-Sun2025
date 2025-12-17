"""
一键测试不同单调性权重下物理约束的效果

功能：
1. 自动测试多个单调性权重值
2. 对比有/无物理约束的效果
3. 生成对比图表和结果报告
4. 保存所有实验数据

使用方法：
    python test_monotonic_weights.py
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from train_cross_battery import train_cross_battery_model

# ===== 实验配置 =====
CONFIG = {
    # 基础配置
    'model_type': 'cnn_lstm',
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 517,

    # 数据退化场景
    'degradation_scenario': 'scenario2',  # 'none', 'scenario1', 'scenario2'
    'sparse_sampling_interval': 4,        # 稀疏采样间隔

    # 单调性权重范围
    'monotonic_weights': [0.2, 0.3, 0.5, 0.6, 0.8],

    # 输出目录
    'output_dir': 'results/monotonic_weight_test',
    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
}


def modify_config_monotonic_weight(model_type, monotonic_weight):
    """
    修改模型配置文件中的单调性权重

    Args:
        model_type: 模型类型
        monotonic_weight: 单调性权重值
    """
    config_path = f'configs/models/{model_type}_config.json'

    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 修改单调性权重
    if 'physics_constraints' in config:
        config['physics_constraints']['monotonic_weight'] = monotonic_weight
        # ⭐ 关键：确保物理约束启用！
        config['physics_constraints']['enabled'] = True

    # 保存配置
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  [OK] 已修改配置: monotonic_weight = {monotonic_weight}, enabled = True")


def run_single_experiment(weight, config):
    """
    运行单个实验

    Args:
        weight: 单调性权重
        config: 实验配置

    Returns:
        results: 实验结果字典
    """
    print(f"\n{'='*70}")
    print(f"运行实验: monotonic_weight = {weight}")
    print(f"{'='*70}")

    # 修改配置文件
    modify_config_monotonic_weight(config['model_type'], weight)

    # 等待文件系统同步（重要！）
    import time
    time.sleep(0.5)

    # 验证配置是否修改成功
    config_path = f'configs/models/{config["model_type"]}_config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        verify_config = json.load(f)
    actual_weight = verify_config.get('physics_constraints', {}).get('monotonic_weight', None)
    print(f"  [VERIFY] 配置文件中的权重: {actual_weight}")

    if actual_weight != weight:
        print(f"  [ERROR] 权重验证失败！预期 {weight}，实际 {actual_weight}")
        raise ValueError(f"配置修改失败")

    # 运行训练
    try:
        wrapper, results, data_dict = train_cross_battery_model(
            model_type=config['model_type'],
            train_ratio=config['train_ratio'],
            val_ratio=config['val_ratio'],
            test_ratio=config['test_ratio'],
            device=config['device'],
            seed=config['seed'],
            apply_cleaning=False,
            color_by_battery=True,
            highlight_anomalies=False,
            degradation_scenario=config['degradation_scenario'],
            sparse_sampling_interval=config['sparse_sampling_interval']
        )

        # 验证损失函数是否使用了正确的权重
        if hasattr(wrapper, 'criterion') and hasattr(wrapper.criterion, 'monotonic_weight'):
            actual_used_weight = wrapper.criterion.monotonic_weight
            print(f"  [VERIFY] 训练使用的权重: {actual_used_weight}")
            if abs(actual_used_weight - weight) > 1e-6:
                print(f"  [WARNING] 权重不匹配！预期 {weight}，实际使用 {actual_used_weight}")

        # 提取关键指标
        experiment_results = {
            'monotonic_weight': weight,
            'test_rmse': results['test_rmse'],
            'test_mae': results['test_mae'],
            'test_r2': results['test_r2'],
            'train_loss': results['train_loss'],
            'val_loss': results['val_loss'],
            'training_time': results.get('training_time', None),
            'success': True,
            'error': None
        }

        print(f"\n[结果] Test RMSE: {results['test_rmse']:.4f}, MAE: {results['test_mae']:.4f}, R²: {results['test_r2']:.4f}")

        # 如果结果与上次完全相同，发出警告
        if hasattr(run_single_experiment, '_last_rmse'):
            if abs(results['test_rmse'] - run_single_experiment._last_rmse) < 1e-8:
                print(f"  [WARNING] RMSE与上次实验完全相同，可能权重未生效！")
        run_single_experiment._last_rmse = results['test_rmse']

    except Exception as e:
        print(f"\n[ERROR] 实验失败: {str(e)}")
        experiment_results = {
            'monotonic_weight': weight,
            'test_rmse': None,
            'test_mae': None,
            'test_r2': None,
            'train_loss': None,
            'val_loss': None,
            'training_time': None,
            'success': False,
            'error': str(e)
        }

    return experiment_results


def plot_results(all_results, config):
    """
    绘制实验结果对比图

    Args:
        all_results: 所有实验结果列表
        config: 实验配置
    """
    # 过滤成功的实验
    successful_results = [r for r in all_results if r['success']]

    if len(successful_results) == 0:
        print("\n[WARN] 没有成功的实验，无法绘图")
        return

    weights = [r['monotonic_weight'] for r in successful_results]
    rmse = [r['test_rmse'] * 100 for r in successful_results]  # 转换为百分比
    mae = [r['test_mae'] * 100 for r in successful_results]
    r2 = [r['test_r2'] for r in successful_results]

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Monotonic Weight Impact Analysis\n'
                 f'Scenario: {config["degradation_scenario"]}, '
                 f'Interval: {config.get("sparse_sampling_interval", "N/A")}',
                 fontsize=14, fontweight='bold')

    # 1. RMSE vs Weight
    axes[0, 0].plot(weights, rmse, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    axes[0, 0].set_xlabel('Monotonic Weight', fontsize=11)
    axes[0, 0].set_ylabel('Test RMSE (%)', fontsize=11)
    axes[0, 0].set_title('RMSE vs. Monotonic Weight', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='No Physics')

    # 标注最优点
    min_idx = np.argmin(rmse)
    axes[0, 0].scatter([weights[min_idx]], [rmse[min_idx]],
                       color='red', s=150, marker='*', zorder=5,
                       label=f'Best: {weights[min_idx]} (RMSE={rmse[min_idx]:.2f}%)')
    axes[0, 0].legend()

    # 2. MAE vs Weight
    axes[0, 1].plot(weights, mae, 'o-', linewidth=2, markersize=8, color='#A23B72')
    axes[0, 1].set_xlabel('Monotonic Weight', fontsize=11)
    axes[0, 1].set_ylabel('Test MAE (%)', fontsize=11)
    axes[0, 1].set_title('MAE vs. Monotonic Weight', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='No Physics')

    # 标注最优点
    min_idx_mae = np.argmin(mae)
    axes[0, 1].scatter([weights[min_idx_mae]], [mae[min_idx_mae]],
                       color='red', s=150, marker='*', zorder=5,
                       label=f'Best: {weights[min_idx_mae]} (MAE={mae[min_idx_mae]:.2f}%)')
    axes[0, 1].legend()

    # 3. R² vs Weight
    axes[1, 0].plot(weights, r2, 'o-', linewidth=2, markersize=8, color='#F18F01')
    axes[1, 0].set_xlabel('Monotonic Weight', fontsize=11)
    axes[1, 0].set_ylabel('Test R²', fontsize=11)
    axes[1, 0].set_title('R² vs. Monotonic Weight', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='No Physics')

    # 标注最优点
    max_idx_r2 = np.argmax(r2)
    axes[1, 0].scatter([weights[max_idx_r2]], [r2[max_idx_r2]],
                       color='red', s=150, marker='*', zorder=5,
                       label=f'Best: {weights[max_idx_r2]} (R²={r2[max_idx_r2]:.4f})')
    axes[1, 0].legend()

    # 4. 改善百分比（相对于无物理约束）
    if weights[0] == 0.0:
        baseline_rmse = rmse[0]
        improvements = [(baseline_rmse - r) / baseline_rmse * 100 for r in rmse[1:]]
        weights_with_physics = weights[1:]

        axes[1, 1].bar(range(len(improvements)), improvements,
                       color=['green' if i > 0 else 'red' for i in improvements],
                       alpha=0.7)
        axes[1, 1].set_xticks(range(len(improvements)))
        axes[1, 1].set_xticklabels([f'{w}' for w in weights_with_physics], rotation=45)
        axes[1, 1].set_xlabel('Monotonic Weight', fontsize=11)
        axes[1, 1].set_ylabel('RMSE Improvement (%)', fontsize=11)
        axes[1, 1].set_title('Improvement over Baseline (weight=0)', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        # 标注数值
        for i, imp in enumerate(improvements):
            axes[1, 1].text(i, imp + 0.5 if imp > 0 else imp - 0.5,
                           f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top',
                           fontsize=9, fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'No baseline (weight=0)\nto compare',
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12)
        axes[1, 1].set_xlabel('Monotonic Weight', fontsize=11)
        axes[1, 1].set_ylabel('RMSE Improvement (%)', fontsize=11)
        axes[1, 1].set_title('Improvement over Baseline', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # 保存图表
    output_dir = os.path.join(config['output_dir'], config['timestamp'])
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'monotonic_weight_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] 对比图已保存: {plot_path}")

    plt.show()


def save_results(all_results, config):
    """
    保存实验结果

    Args:
        all_results: 所有实验结果列表
        config: 实验配置
    """
    output_dir = os.path.join(config['output_dir'], config['timestamp'])
    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存原始数据（pickle）
    results_pkl_path = os.path.join(output_dir, 'all_results.pkl')
    with open(results_pkl_path, 'wb') as f:
        pickle.dump({
            'config': config,
            'results': all_results
        }, f)
    print(f"[OK] 原始数据已保存: {results_pkl_path}")

    # 2. 保存结果表格（JSON）
    results_json_path = os.path.join(output_dir, 'results_summary.json')
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': config,
            'results': all_results
        }, f, indent=2, ensure_ascii=False)
    print(f"[OK] 结果摘要已保存: {results_json_path}")

    # 3. 生成文本报告
    report_path = os.path.join(output_dir, 'experiment_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("单调性权重实验报告\n")
        f.write("="*70 + "\n\n")

        f.write("实验配置:\n")
        f.write("-"*70 + "\n")
        f.write(f"模型类型: {config['model_type']}\n")
        f.write(f"数据退化场景: {config['degradation_scenario']}\n")
        if config.get('sparse_sampling_interval'):
            f.write(f"稀疏采样间隔: {config['sparse_sampling_interval']}\n")
        f.write(f"随机种子: {config['seed']}\n")
        f.write(f"测试权重: {config['monotonic_weights']}\n\n")

        f.write("实验结果:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'权重':<12} {'RMSE(%)':<12} {'MAE(%)':<12} {'R²':<12} {'状态':<10}\n")
        f.write("-"*70 + "\n")

        for r in all_results:
            if r['success']:
                f.write(f"{r['monotonic_weight']:<12.2f} "
                       f"{r['test_rmse']*100:<12.4f} "
                       f"{r['test_mae']*100:<12.4f} "
                       f"{r['test_r2']:<12.4f} "
                       f"{'成功':<10}\n")
            else:
                f.write(f"{r['monotonic_weight']:<12.2f} "
                       f"{'N/A':<12} {'N/A':<12} {'N/A':<12} "
                       f"{'失败':<10}\n")

        f.write("\n")

        # 找出最优配置
        successful_results = [r for r in all_results if r['success']]
        if successful_results:
            best_rmse = min(successful_results, key=lambda x: x['test_rmse'])
            best_r2 = max(successful_results, key=lambda x: x['test_r2'])

            f.write("最优配置:\n")
            f.write("-"*70 + "\n")
            f.write(f"最低 RMSE: weight={best_rmse['monotonic_weight']}, "
                   f"RMSE={best_rmse['test_rmse']*100:.4f}%\n")
            f.write(f"最高 R²:   weight={best_r2['monotonic_weight']}, "
                   f"R²={best_r2['test_r2']:.4f}\n")

            # 计算改善
            if all_results[0]['monotonic_weight'] == 0.0 and all_results[0]['success']:
                baseline_rmse = all_results[0]['test_rmse'] * 100
                best_improvement = (baseline_rmse - best_rmse['test_rmse']*100) / baseline_rmse * 100
                f.write(f"\n相对基线改善: {best_improvement:+.2f}%\n")

    print(f"[OK] 实验报告已保存: {report_path}")


def print_summary(all_results):
    """
    打印实验摘要

    Args:
        all_results: 所有实验结果列表
    """
    print("\n" + "="*70)
    print("实验摘要")
    print("="*70)

    successful_results = [r for r in all_results if r['success']]
    failed_results = [r for r in all_results if not r['success']]

    print(f"\n总实验数: {len(all_results)}")
    print(f"成功: {len(successful_results)}")
    print(f"失败: {len(failed_results)}")

    if successful_results:
        print("\n结果汇总:")
        print("-"*70)
        print(f"{'权重':<12} {'RMSE(%)':<12} {'MAE(%)':<12} {'R²':<12}")
        print("-"*70)

        for r in successful_results:
            print(f"{r['monotonic_weight']:<12.2f} "
                  f"{r['test_rmse']*100:<12.4f} "
                  f"{r['test_mae']*100:<12.4f} "
                  f"{r['test_r2']:<12.4f}")

        # 找出最优配置
        best_rmse = min(successful_results, key=lambda x: x['test_rmse'])
        best_r2 = max(successful_results, key=lambda x: x['test_r2'])

        print("\n最优配置:")
        print("-"*70)
        print(f"最低 RMSE: weight={best_rmse['monotonic_weight']}, RMSE={best_rmse['test_rmse']*100:.4f}%")
        print(f"最高 R²:   weight={best_r2['monotonic_weight']}, R²={best_r2['test_r2']:.4f}")

        # 计算改善
        if all_results[0]['monotonic_weight'] == 0.0 and all_results[0]['success']:
            baseline_rmse = all_results[0]['test_rmse'] * 100
            best_improvement = (baseline_rmse - best_rmse['test_rmse']*100) / baseline_rmse * 100
            print(f"\n相对基线改善: {best_improvement:+.2f}%")


def main():
    """主函数"""
    print("="*70)
    print("单调性权重批量测试")
    print("="*70)
    print(f"\n实验配置:")
    print(f"  模型类型: {CONFIG['model_type']}")
    print(f"  数据场景: {CONFIG['degradation_scenario']}")
    if CONFIG.get('sparse_sampling_interval'):
        print(f"  稀疏间隔: {CONFIG['sparse_sampling_interval']}")
    print(f"  测试权重: {CONFIG['monotonic_weights']}")
    print(f"  输出目录: {CONFIG['output_dir']}/{CONFIG['timestamp']}")

    # 运行所有实验
    all_results = []

    for i, weight in enumerate(CONFIG['monotonic_weights'], 1):
        print(f"\n{'#'*70}")
        print(f"进度: {i}/{len(CONFIG['monotonic_weights'])}")
        print(f"{'#'*70}")

        result = run_single_experiment(weight, CONFIG)
        all_results.append(result)

    # 打印摘要
    print_summary(all_results)

    # 保存结果
    save_results(all_results, CONFIG)

    # 绘制对比图
    plot_results(all_results, CONFIG)

    print("\n" + "="*70)
    print("所有实验完成！")
    print("="*70)
    print(f"\n结果保存在: {CONFIG['output_dir']}/{CONFIG['timestamp']}/")


if __name__ == "__main__":
    main()
