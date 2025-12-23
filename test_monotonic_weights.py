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
    'seed': 999,

    # 数据退化场景
    'degradation_scenario': 'scenario4',  # 'none', 'scenario1', 'scenario2', 'scenario3', 'scenario4'

    # Scenario 2 参数 (规律稀疏采样)
    'sparse_sampling_interval': 5,        # 稀疏采样间隔

    # Scenario 3 参数 (随机缺失)
    'random_missing_rate': 0.7,          # 随机缺失率 (例如: 0.4 表示丢弃40%，保留60%)
                                          # None = 不使用 Scenario 3

    # Scenario 4 参数 (连续循环缺失) - 支持批量测试
    'cycle_drop_rates': [0.1, 0.2, 0.3],     # 循环丢弃率列表 (例如: [0.2, 0.3, 0.5])
    'cycle_drop_num_gaps_list': [50, 100],   # 缺失段数量列表 (例如: [1, 2, 3])
                                              # 将测试所有 (drop_rate, num_gaps) 组合

    # 单调性权重范围
    'monotonic_weights': [0.1, 0.3, 0.5],

    # 输出目录
    'output_dir': 'results/monotonic_weight_test',
    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
}


def modify_config_monotonic_weight(model_type, monotonic_weight):
    """
    修改模型配置文件中的单调性权重，并确保其他物理约束参数一致

    Args:
        model_type: 模型类型
        monotonic_weight: 单调性权重值
    """
    config_path = f'configs/models/{model_type}_config.json'

    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 修改物理约束参数，确保一致性
    if 'physics_constraints' in config:
        config['physics_constraints']['monotonic_weight'] = monotonic_weight
        # ⭐ 关键：确保物理约束启用！
        config['physics_constraints']['enabled'] = True

        # ⭐ 显式设置其他参数，确保批量测试一致性
        config['physics_constraints']['monotonic_tolerance'] = 0.01  # 软约束容差
        config['physics_constraints']['boundary_weight'] = 0.0       # 边界约束（不使用）
        config['physics_constraints']['smoothness_weight'] = 0.0     # 平滑性约束（不使用）
        config['physics_constraints']['curvature_weight'] = 0.0      # 曲率约束（不使用）

    # 保存配置
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  [OK] 已修改配置: monotonic_weight = {monotonic_weight}, tolerance = 0.01, enabled = True")


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

    # 显示场景信息
    if config['degradation_scenario'] == 'scenario2' and config.get('sparse_sampling_interval'):
        retention_rate = 100.0 / config['sparse_sampling_interval']
        print(f"场景: Scenario 2 (Uniform, interval={config['sparse_sampling_interval']}, 保留率≈{retention_rate:.1f}%)")
    elif config['degradation_scenario'] == 'scenario3' and config.get('random_missing_rate'):
        retention_rate = (1 - config['random_missing_rate']) * 100
        print(f"场景: Scenario 3 (Random Missing, 缺失率={config['random_missing_rate']*100:.0f}%, 保留率≈{retention_rate:.1f}%)")
    elif config['degradation_scenario'] == 'scenario4' and (config.get('cycle_drop_rate') or config.get('cycle_drop_num_gaps')):
        drop_rate = config.get('cycle_drop_rate', 0.3)
        num_gaps = config.get('cycle_drop_num_gaps', 2)
        retention_rate = (1 - drop_rate) * 100
        print(f"场景: Scenario 4 (Consecutive Cycle Drop, 丢弃{drop_rate*100:.0f}%, {num_gaps}段连续缺失, 保留率≈{retention_rate:.1f}%)")
    else:
        print(f"场景: {config['degradation_scenario']}")

    print(f"{'='*70}")

    # 修改配置文件
    modify_config_monotonic_weight(config['model_type'], weight)

    # 等待文件系统同步（重要！）
    import time
    time.sleep(0.5)

    # 验证配置是否修改成功（扩展验证所有关键参数）
    config_path = f'configs/models/{config["model_type"]}_config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        verify_config = json.load(f)

    physics = verify_config.get('physics_constraints', {})
    actual_weight = physics.get('monotonic_weight', None)

    print(f"  [VERIFY] 配置文件中的权重: {actual_weight}")
    print(f"  [VERIFY] 完整物理约束配置:")
    print(f"    enabled: {physics.get('enabled')}")
    print(f"    monotonic_tolerance: {physics.get('monotonic_tolerance')}")
    print(f"    boundary_weight: {physics.get('boundary_weight')}")
    print(f"    smoothness_weight: {physics.get('smoothness_weight')}")
    print(f"    curvature_weight: {physics.get('curvature_weight')}")

    # 验证关键参数
    if actual_weight != weight:
        print(f"  [ERROR] monotonic_weight 验证失败！预期 {weight}，实际 {actual_weight}")
        raise ValueError(f"配置修改失败")

    if physics.get('monotonic_tolerance') != 0.01:
        print(f"  [WARNING] monotonic_tolerance 不是预期值 0.01，实际: {physics.get('monotonic_tolerance')}")

    if physics.get('enabled') != True:
        print(f"  [WARNING] physics_constraints 未启用！")

    # 检查其他权重是否为 0（确保批量测试一致性）
    unexpected_params = []
    if physics.get('boundary_weight', 0.0) != 0.0:
        unexpected_params.append(f"boundary_weight={physics.get('boundary_weight')}")
    if physics.get('smoothness_weight', 0.0) != 0.0:
        unexpected_params.append(f"smoothness_weight={physics.get('smoothness_weight')}")
    if physics.get('curvature_weight', 0.0) != 0.0:
        unexpected_params.append(f"curvature_weight={physics.get('curvature_weight')}")

    if unexpected_params:
        print(f"  [WARNING] 检测到非零参数: {', '.join(unexpected_params)}")
        print(f"  [WARNING] 这可能影响批量测试一致性！")
    else:
        print(f"  [OK] 所有约束参数已正确设置")

    # 运行训练
    try:
        # 根据场景选择参数
        train_params = {
            'model_type': config['model_type'],
            'train_ratio': config['train_ratio'],
            'val_ratio': config['val_ratio'],
            'test_ratio': config['test_ratio'],
            'device': config['device'],
            'seed': config['seed'],
            'apply_cleaning': False,
            'color_by_battery': True,
            'highlight_anomalies': False,
            'degradation_scenario': config['degradation_scenario']
        }

        # 添加 Scenario 2 参数
        if 'sparse_sampling_interval' in config and config['sparse_sampling_interval'] is not None:
            train_params['sparse_sampling_interval'] = config['sparse_sampling_interval']

        # 添加 Scenario 3 参数
        if 'random_missing_rate' in config and config['random_missing_rate'] is not None:
            train_params['random_missing_rate'] = config['random_missing_rate']

        # 添加 Scenario 4 参数
        if 'cycle_drop_rate' in config and config['cycle_drop_rate'] is not None:
            train_params['cycle_drop_rate'] = config['cycle_drop_rate']
        if 'cycle_drop_num_gaps' in config and config['cycle_drop_num_gaps'] is not None:
            train_params['cycle_drop_num_gaps'] = config['cycle_drop_num_gaps']

        wrapper, results, data_dict = train_cross_battery_model(**train_params)

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

    # 检查是否是 Scenario 4 多参数测试
    has_scenario4_params = 'cycle_drop_rate' in successful_results[0]

    if has_scenario4_params and config['degradation_scenario'] == 'scenario4':
        # Scenario 4: 绘制 3D 参数组合对比图
        plot_scenario4_results(successful_results, config)
    else:
        # 其他场景: 绘制标准的单调性权重对比图
        plot_standard_results(successful_results, config)


def plot_standard_results(successful_results, config):
    """
    绘制标准的单调性权重对比图（非 Scenario 4）

    Args:
        successful_results: 成功的实验结果列表
        config: 实验配置
    """
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


def plot_scenario4_results(successful_results, config):
    """
    绘制 Scenario 4 的多参数对比图（drop_rate × num_gaps × weight）

    Args:
        successful_results: 成功的实验结果列表
        config: 实验配置
    """
    import pandas as pd
    # Import needed for 3D projection
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # 转换为 DataFrame 便于分组
    df = pd.DataFrame(successful_results)
    df['rmse_pct'] = df['test_rmse'] * 100
    df['mae_pct'] = df['test_mae'] * 100

    # 获取所有参数组合
    drop_rates = sorted(df['cycle_drop_rate'].unique())
    num_gaps_list = sorted(df['cycle_drop_num_gaps'].unique())
    weights = sorted(df['monotonic_weight'].unique())

    # 创建多子图布局
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'Scenario 4: Consecutive Cycle Drop - Multi-Parameter Analysis\n'
                 f'{len(drop_rates)} Drop Rates × {len(num_gaps_list)} Num Gaps × {len(weights)} Weights = {len(successful_results)} Experiments',
                 fontsize=14, fontweight='bold')

    # 1. 热力图: Drop Rate vs Num Gaps (固定最优权重)
    ax1 = plt.subplot(2, 3, 1)
    # 找出每个 (drop_rate, num_gaps) 组合的最优权重和对应RMSE
    heatmap_data = np.zeros((len(num_gaps_list), len(drop_rates)))
    for i, num_gaps in enumerate(num_gaps_list):
        for j, drop_rate in enumerate(drop_rates):
            subset = df[(df['cycle_drop_rate'] == drop_rate) & (df['cycle_drop_num_gaps'] == num_gaps)]
            if not subset.empty:
                best_rmse = subset['rmse_pct'].min()
                heatmap_data[i, j] = best_rmse
            else:
                heatmap_data[i, j] = np.nan

    im1 = ax1.imshow(heatmap_data, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
    ax1.set_xticks(range(len(drop_rates)))
    ax1.set_yticks(range(len(num_gaps_list)))
    ax1.set_xticklabels([f'{dr:.1f}' for dr in drop_rates])
    ax1.set_yticklabels([f'{ng}' for ng in num_gaps_list])
    ax1.set_xlabel('Drop Rate', fontsize=10)
    ax1.set_ylabel('Num Gaps', fontsize=10)
    ax1.set_title('Best RMSE (%) Heatmap\n(Optimal Weight for Each Config)', fontsize=11, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='RMSE (%)')

    # 标注数值
    for i in range(len(num_gaps_list)):
        for j in range(len(drop_rates)):
            if not np.isnan(heatmap_data[i, j]):
                ax1.text(j, i, f'{heatmap_data[i, j]:.2f}',
                        ha='center', va='center', color='black', fontsize=9)

    # 2. 折线图: Weight vs RMSE (不同 drop_rate，固定 num_gaps)
    ax2 = plt.subplot(2, 3, 2)
    if len(num_gaps_list) > 0:
        mid_num_gaps = num_gaps_list[len(num_gaps_list)//2]
        for drop_rate in drop_rates:
            subset = df[(df['cycle_drop_rate'] == drop_rate) & (df['cycle_drop_num_gaps'] == mid_num_gaps)]
            subset = subset.sort_values('monotonic_weight')
            if not subset.empty:
                ax2.plot(subset['monotonic_weight'], subset['rmse_pct'],
                        'o-', linewidth=2, markersize=6, label=f'drop={drop_rate:.1f}')
    ax2.set_xlabel('Monotonic Weight', fontsize=10)
    ax2.set_ylabel('RMSE (%)', fontsize=10)
    ax2.set_title(f'RMSE vs Weight\n(Fixed num_gaps={mid_num_gaps})', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. 折线图: Weight vs RMSE (不同 num_gaps，固定 drop_rate)
    ax3 = plt.subplot(2, 3, 3)
    if len(drop_rates) > 0:
        mid_drop_rate = drop_rates[len(drop_rates)//2]
        for num_gaps in num_gaps_list:
            subset = df[(df['cycle_drop_rate'] == mid_drop_rate) & (df['cycle_drop_num_gaps'] == num_gaps)]
            subset = subset.sort_values('monotonic_weight')
            if not subset.empty:
                ax3.plot(subset['monotonic_weight'], subset['rmse_pct'],
                        'o-', linewidth=2, markersize=6, label=f'gaps={num_gaps}')
    ax3.set_xlabel('Monotonic Weight', fontsize=10)
    ax3.set_ylabel('RMSE (%)', fontsize=10)
    ax3.set_title(f'RMSE vs Weight\n(Fixed drop_rate={mid_drop_rate:.1f})', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. 3D 散点图: Drop Rate × Num Gaps × RMSE (使用权重着色)
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    scatter = ax4.scatter(df['cycle_drop_rate'].values, df['cycle_drop_num_gaps'].values,
                         df['rmse_pct'].values,
                         c=df['monotonic_weight'].values, cmap='viridis', s=50, alpha=0.7)
    ax4.set_xlabel('Drop Rate', fontsize=9)
    ax4.set_ylabel('Num Gaps', fontsize=9)
    ax4.set_zlabel('RMSE (%)', fontsize=9)
    ax4.set_title('3D: Drop Rate × Num Gaps × RMSE\n(Color = Weight)', fontsize=11, fontweight='bold')
    plt.colorbar(scatter, ax=ax4, label='Weight', shrink=0.6)

    # 5. 柱状图: 每个参数组合的最优权重分布
    ax5 = plt.subplot(2, 3, 5)
    best_weights = []
    config_labels = []
    for drop_rate in drop_rates:
        for num_gaps in num_gaps_list:
            subset = df[(df['cycle_drop_rate'] == drop_rate) & (df['cycle_drop_num_gaps'] == num_gaps)]
            if not subset.empty:
                best_idx = subset['rmse_pct'].idxmin()
                best_weight = subset.loc[best_idx, 'monotonic_weight']
                best_weights.append(best_weight)
                config_labels.append(f'dr={drop_rate:.1f}\nng={num_gaps}')

    bars = ax5.bar(range(len(best_weights)), best_weights, alpha=0.7, color='steelblue')
    ax5.set_xticks(range(len(best_weights)))
    ax5.set_xticklabels(config_labels, rotation=45, ha='right', fontsize=8)
    ax5.set_ylabel('Optimal Weight', fontsize=10)
    ax5.set_title('Optimal Weight for Each Config', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # 标注数值
    for i, (bar, weight) in enumerate(zip(bars, best_weights)):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{weight:.2f}', ha='center', va='bottom', fontsize=8)

    # 6. 表格: Top 5 最优配置
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # 找出 RMSE 最低的前5个配置
    top5 = df.nsmallest(5, 'rmse_pct')[['cycle_drop_rate', 'cycle_drop_num_gaps',
                                          'monotonic_weight', 'rmse_pct', 'test_r2']]

    table_data = []
    for _, row in top5.iterrows():
        table_data.append([
            f"{row['cycle_drop_rate']:.2f}",
            f"{row['cycle_drop_num_gaps']}",
            f"{row['monotonic_weight']:.2f}",
            f"{row['rmse_pct']:.3f}%",
            f"{row['test_r2']:.4f}"
        ])

    from matplotlib.transforms import Bbox as BboxClass
    table = ax6.table(cellText=table_data,
                     colLabels=['Drop\nRate', 'Num\nGaps', 'Weight', 'RMSE', 'R²'],
                     cellLoc='center',
                     loc='center',
                     bbox=BboxClass.from_bounds(0, 0, 1, 1))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # 设置表头样式
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax6.set_title('Top 5 Best Configurations', fontsize=11, fontweight='bold', pad=20)

    plt.tight_layout()

    # 保存图表
    output_dir = os.path.join(config['output_dir'], config['timestamp'])
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'scenario4_parameter_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Scenario 4 参数分析图已保存: {plot_path}")

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
        if config.get('random_missing_rate'):
            f.write(f"随机缺失率: {config['random_missing_rate']}\n")
        if config.get('cycle_drop_rates'):
            f.write(f"循环丢弃率列表: {config['cycle_drop_rates']}\n")
            f.write(f"缺失段数量列表: {config.get('cycle_drop_num_gaps_list', 'N/A')}\n")
        f.write(f"随机种子: {config['seed']}\n")
        f.write(f"测试权重: {config['monotonic_weights']}\n\n")

        # 检查是否是 Scenario 4 多参数测试
        has_scenario4_params = all_results and 'cycle_drop_rate' in all_results[0]

        f.write("实验结果:\n")
        f.write("-"*70 + "\n")
        if has_scenario4_params:
            f.write(f"{'丢弃率':<10} {'缺失段':<10} {'权重':<10} {'RMSE(%)':<12} {'MAE(%)':<12} {'R²':<12} {'状态':<10}\n")
        else:
            f.write(f"{'权重':<12} {'RMSE(%)':<12} {'MAE(%)':<12} {'R²':<12} {'状态':<10}\n")
        f.write("-"*70 + "\n")

        for r in all_results:
            if r['success']:
                if has_scenario4_params:
                    f.write(f"{r['cycle_drop_rate']:<10.2f} "
                           f"{r['cycle_drop_num_gaps']:<10} "
                           f"{r['monotonic_weight']:<10.2f} "
                           f"{r['test_rmse']*100:<12.4f} "
                           f"{r['test_mae']*100:<12.4f} "
                           f"{r['test_r2']:<12.4f} "
                           f"{'成功':<10}\n")
                else:
                    f.write(f"{r['monotonic_weight']:<12.2f} "
                           f"{r['test_rmse']*100:<12.4f} "
                           f"{r['test_mae']*100:<12.4f} "
                           f"{r['test_r2']:<12.4f} "
                           f"{'成功':<10}\n")
            else:
                if has_scenario4_params:
                    f.write(f"{r.get('cycle_drop_rate', 'N/A'):<10} "
                           f"{r.get('cycle_drop_num_gaps', 'N/A'):<10} "
                           f"{r['monotonic_weight']:<10.2f} "
                           f"{'N/A':<12} {'N/A':<12} {'N/A':<12} "
                           f"{'失败':<10}\n")
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
            if has_scenario4_params:
                f.write(f"最低 RMSE: drop_rate={best_rmse['cycle_drop_rate']}, "
                       f"num_gaps={best_rmse['cycle_drop_num_gaps']}, "
                       f"weight={best_rmse['monotonic_weight']}, "
                       f"RMSE={best_rmse['test_rmse']*100:.4f}%\n")
                f.write(f"最高 R²:   drop_rate={best_r2['cycle_drop_rate']}, "
                       f"num_gaps={best_r2['cycle_drop_num_gaps']}, "
                       f"weight={best_r2['monotonic_weight']}, "
                       f"R²={best_r2['test_r2']:.4f}\n")
            else:
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
        # 检查是否是 Scenario 4 多参数测试
        has_scenario4_params = 'cycle_drop_rate' in successful_results[0]

        print("\n结果汇总:")
        print("-"*70)
        if has_scenario4_params:
            print(f"{'丢弃率':<10} {'缺失段':<10} {'权重':<10} {'RMSE(%)':<12} {'MAE(%)':<12} {'R²':<12}")
        else:
            print(f"{'权重':<12} {'RMSE(%)':<12} {'MAE(%)':<12} {'R²':<12}")
        print("-"*70)

        for r in successful_results:
            if has_scenario4_params:
                print(f"{r['cycle_drop_rate']:<10.2f} "
                      f"{r['cycle_drop_num_gaps']:<10} "
                      f"{r['monotonic_weight']:<10.2f} "
                      f"{r['test_rmse']*100:<12.4f} "
                      f"{r['test_mae']*100:<12.4f} "
                      f"{r['test_r2']:<12.4f}")
            else:
                print(f"{r['monotonic_weight']:<12.2f} "
                      f"{r['test_rmse']*100:<12.4f} "
                      f"{r['test_mae']*100:<12.4f} "
                      f"{r['test_r2']:<12.4f}")

        # 找出最优配置
        best_rmse = min(successful_results, key=lambda x: x['test_rmse'])
        best_r2 = max(successful_results, key=lambda x: x['test_r2'])

        print("\n最优配置:")
        print("-"*70)
        if has_scenario4_params:
            print(f"最低 RMSE: drop_rate={best_rmse['cycle_drop_rate']}, "
                  f"num_gaps={best_rmse['cycle_drop_num_gaps']}, "
                  f"weight={best_rmse['monotonic_weight']}, "
                  f"RMSE={best_rmse['test_rmse']*100:.4f}%")
            print(f"最高 R²:   drop_rate={best_r2['cycle_drop_rate']}, "
                  f"num_gaps={best_r2['cycle_drop_num_gaps']}, "
                  f"weight={best_r2['monotonic_weight']}, "
                  f"R²={best_r2['test_r2']:.4f}")
        else:
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

    # 显示场景参数
    if CONFIG['degradation_scenario'] == 'scenario2' and CONFIG.get('sparse_sampling_interval'):
        retention_rate = 100.0 / CONFIG['sparse_sampling_interval']
        print(f"  稀疏间隔: {CONFIG['sparse_sampling_interval']} (保留率≈{retention_rate:.1f}%)")
    elif CONFIG['degradation_scenario'] == 'scenario3' and CONFIG.get('random_missing_rate'):
        retention_rate = (1 - CONFIG['random_missing_rate']) * 100
        print(f"  缺失率: {CONFIG['random_missing_rate']*100:.0f}% (保留率≈{retention_rate:.1f}%)")
    elif CONFIG['degradation_scenario'] == 'scenario4':
        drop_rates = CONFIG.get('cycle_drop_rates', [0.3])
        num_gaps_list = CONFIG.get('cycle_drop_num_gaps_list', [2])
        print(f"  丢弃率列表: {drop_rates}")
        print(f"  缺失段列表: {num_gaps_list}")
        print(f"  将测试 {len(drop_rates)} x {len(num_gaps_list)} = {len(drop_rates)*len(num_gaps_list)} 种 Scenario 4 参数组合")

    print(f"  测试权重: {CONFIG['monotonic_weights']}")
    print(f"  输出目录: {CONFIG['output_dir']}/{CONFIG['timestamp']}")

    # 生成参数组合
    if CONFIG['degradation_scenario'] == 'scenario4':
        # 对于 Scenario 4，生成 (drop_rate, num_gaps, weight) 三维组合
        drop_rates = CONFIG.get('cycle_drop_rates', [0.3])
        num_gaps_list = CONFIG.get('cycle_drop_num_gaps_list', [2])

        param_combinations = []
        for drop_rate in drop_rates:
            for num_gaps in num_gaps_list:
                for weight in CONFIG['monotonic_weights']:
                    param_combinations.append({
                        'cycle_drop_rate': drop_rate,
                        'cycle_drop_num_gaps': num_gaps,
                        'monotonic_weight': weight
                    })

        total_experiments = len(param_combinations)
        print(f"\n总实验数: {total_experiments} "
              f"({len(drop_rates)} drop_rates × {len(num_gaps_list)} num_gaps × {len(CONFIG['monotonic_weights'])} weights)")
    else:
        # 对于其他场景，只遍历权重
        param_combinations = [{'monotonic_weight': w} for w in CONFIG['monotonic_weights']]
        total_experiments = len(param_combinations)
        print(f"\n总实验数: {total_experiments}")

    # 运行所有实验
    all_results = []

    for i, params in enumerate(param_combinations, 1):
        print(f"\n{'#'*70}")
        print(f"进度: {i}/{total_experiments}")
        print(f"{'#'*70}")

        # 创建实验配置（合并全局配置和当前参数）
        experiment_config = CONFIG.copy()
        experiment_config.update(params)

        result = run_single_experiment(params['monotonic_weight'], experiment_config)

        # 记录额外参数到结果中
        if 'cycle_drop_rate' in params:
            result['cycle_drop_rate'] = params['cycle_drop_rate']
        if 'cycle_drop_num_gaps' in params:
            result['cycle_drop_num_gaps'] = params['cycle_drop_num_gaps']

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
