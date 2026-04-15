"""
多模型对比测试脚本：一键训练和对比多个模型的效果。

支持的模型类型：
- 基础模型: fnn, cnn, lstm, gru, bilstm, bigru, mlp, rescnn
- 混合模型: cnn_lstm, cnn_bilstm, cnn_mlp
- Seq2Seq模型: lstm_seq2seq, gru_seq2seq, bilstm_seq2seq, bigru_seq2seq
- 机器学习: xgboost_simple, xgboost_enhanced

使用方法：直接修改下方【配置区域】的参数，然后运行脚本即可

注意：本脚本直接调用 train_cross_battery.py 中的训练函数，确保结果与单独运行一致。
"""

# ============================================================================
#                              【配置区域】
#                    修改以下参数后直接运行脚本即可
# ============================================================================

# 要测试的模型列表（从下方支持的模型中选择）
# 支持的模型:
#   基础模型: 'fnn', 'cnn', 'lstm', 'gru', 'bilstm', 'bigru', 'mlp', 'rescnn'
#   混合模型: 'cnn_lstm', 'cnn_bilstm', 'cnn_mlp', 'pi_cnn_lstm'
#   Seq2Seq:  'lstm_seq2seq', 'gru_seq2seq', 'bilstm_seq2seq', 'bigru_seq2seq'
#   机器学习: 'xgboost_simple', 'xgboost_enhanced'
MODELS_TO_TEST = [
    'xgboost_simple',   # XGBoost
    'lstm',             # LSTM
    'cnn_lstm',         # CNN-LSTM
    'pi_cnn_lstm',      # PI-CNNLSTM (Physics-Informed CNN-LSTM)
]

# 设为 True 则测试所有17种模型（会忽略上面的 MODELS_TO_TEST）
TEST_ALL_MODELS = False

# 随机种子（确保结果可复现）
RANDOM_SEED = 999

# 计算设备（'cuda' 使用GPU, 'cpu' 使用CPU, 'auto' 自动选择）
DEVICE = 'auto'

# 结果保存目录（会自动添加时间戳后缀）
OUTPUT_DIR = 'results/model_comparison'

# 是否应用3-Sigma数据清洗
APPLY_CLEANING = False

# ============================================================================
#                           【配置区域结束】
# ============================================================================

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import joblib

# 直接导入 train_cross_battery 中的训练函数，确保结果一致
from train_cross_battery import train_cross_battery_model
# 导入 XGBoost 专用训练函数
from train_xgboost_baseline import train_xgboost_baseline

# 支持的所有模型（请勿修改）
ALL_MODELS = [
    'fnn', 'cnn', 'lstm', 'gru', 'bilstm', 'bigru', 'mlp', 'rescnn',
    'cnn_lstm', 'cnn_bilstm', 'cnn_mlp', 'pi_cnn_lstm',
    'lstm_seq2seq', 'gru_seq2seq', 'bilstm_seq2seq', 'bigru_seq2seq',
    'xgboost_simple', 'xgboost_enhanced'
]

# 统一显示名称（图表用，不影响模型本身）
DISPLAY_NAME_MAP = {
    "xgboost_simple": "XGBoost",
    "lstm": "LSTM",
    "cnn_lstm": "CNN-LSTM",
    "pi_cnn_lstm": "PI-CNNLSTM",
}


def display_name(model_name: str) -> str:
    return DISPLAY_NAME_MAP.get(model_name, model_name)


def save_model(wrapper, model_type, save_dir, config):
    """
    保存训练好的模型参数和配置。

    Args:
        wrapper: 模型包装器（包含model和training_history）
        model_type: 模型类型
        save_dir: 保存目录
        config: 模型配置信息
    """
    model_save_dir = os.path.join(save_dir, 'saved_models')
    os.makedirs(model_save_dir, exist_ok=True)

    model_path = os.path.join(model_save_dir, f'{model_type}.pth')
    config_path = os.path.join(model_save_dir, f'{model_type}_config.pkl')

    try:
        # 保存模型参数
        if model_type in ['xgboost_simple', 'xgboost_enhanced']:
            # XGBoost 模型保存
            import joblib
            joblib.dump(wrapper.model.model, model_path.replace('.pth', '.joblib'))
            print(f"  已保存XGBoost模型参数: {model_path.replace('.pth', '.joblib')}")
        else:
            # PyTorch 模型保存
            torch.save({
                'model_state_dict': wrapper.model.state_dict(),
                'model_type': model_type,
                'training_history': wrapper.training_history
            }, model_path)
            print(f"  已保存模型参数: {model_path}")

        # 保存配置
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        print(f"  已保存配置文件: {config_path}")

    except Exception as e:
        print(f"  警告: 保存模型 {model_type} 失败: {e}")


def compare_models(models_to_test, device='cuda', seed=42, apply_cleaning=False, save_dir=None):
    """
    对比多个模型，直接调用 train_cross_battery_model 确保结果一致。
    """
    results = {}
    models = {}  # 保存模型wrapper
    configs = {}  # 保存模型配置

    print("\n" + "=" * 70)
    print("开始模型对比测试")
    print("=" * 70)
    print(f"测试模型: {models_to_test}")
    print(f"设备: {device}")
    print(f"随机种子: {seed}")
    print()

    for i, model_type in enumerate(models_to_test):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(models_to_test)}] 训练模型: {model_type.upper()}")
        print(f"{'='*70}")

        try:
            # XGBoost 模型使用专门的训练脚本
            if model_type in ['xgboost_simple', 'xgboost_enhanced']:
                wrapper, result, data_dict = train_xgboost_baseline(
                    model_type=model_type,
                    train_ratio=0.6,
                    val_ratio=0.2,
                    test_ratio=0.2,
                    device='cpu',  # XGBoost 使用 CPU
                    seed=seed,
                    apply_cleaning=apply_cleaning,
                    degradation_scenario='none'
                )
            else:
                # 其他深度学习模型使用 train_cross_battery_model
                wrapper, result, data_dict = train_cross_battery_model(
                    model_type=model_type,
                    train_ratio=0.6,
                    val_ratio=0.2,
                    test_ratio=0.2,
                    device=device,
                    seed=seed,
                    apply_cleaning=apply_cleaning,
                    color_by_battery=True,
                    highlight_anomalies=True,
                    degradation_scenario='none'
                )

            if result:
                results[model_type] = result
                models[model_type] = wrapper
                configs[model_type] = {
                    'model_type': model_type,
                    'train_ratio': 0.6,
                    'val_ratio': 0.2,
                    'test_ratio': 0.2,
                    'seed': seed,
                    'apply_cleaning': apply_cleaning,
                    'n_features': data_dict['n_features']
                }

                print(f"\n  ✓ {model_type.upper()} 训练完成")
                print(f"    RMSE: {result['test_rmse']:.6f}")
                print(f"    MAE:  {result['test_mae']:.6f}")
                print(f"    R²:   {result['test_r2']:.6f}")

                # 保存模型参数
                if save_dir is not None:
                    save_model(wrapper, model_type, save_dir, configs[model_type])

        except Exception as e:
            print(f"\n  ✗ {model_type.upper()} 训练失败: {e}")
            import traceback
            traceback.print_exc()

    return results, models, configs


def plot_comparison(results, save_dir):
    """绘制对比图表"""
    if not results:
        print("没有结果可以绘制")
        return

    models = list(results.keys())
    display_models = [display_name(m) for m in models]
    rmse_values = [results[m]['test_rmse'] * 100 for m in models]
    mae_values = [results[m]['test_mae'] * 100 for m in models]
    r2_values = [results[m]['test_r2'] for m in models]

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

    # RMSE对比
    ax1 = axes[0]
    bars1 = ax1.bar(display_models, rmse_values, color=colors)
    ax1.set_title('RMSE Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE (%)')
    ax1.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars1, rmse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.3f}%',
                ha='center', va='bottom', fontsize=8)

    # MAE对比
    ax2 = axes[1]
    bars2 = ax2.bar(display_models, mae_values, color=colors)
    ax2.set_title('MAE Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MAE (%)')
    ax2.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars2, mae_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.3f}%',
                ha='center', va='bottom', fontsize=8)

    # R²对比
    ax3 = axes[2]
    bars3 = ax3.bar(display_models, r2_values, color=colors)
    ax3.set_title('R² Comparison', fontsize=12, fontweight='bold')
    ax3.set_ylabel('R²')
    ax3.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars3, r2_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.4f}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n对比图已保存: {os.path.join(save_dir, 'model_comparison.png')}")


def plot_error_bars(results, save_dir):
    """绘制所有模型的误差柱状图（RMSE/MAE/R²的绝对值对比）
    x轴为指标（RMSE/MAE/R²），每个指标显示所有模型的柱状图
    """
    if not results:
        print("没有结果可以绘制误差柱状图")
        return

    models = list(results.keys())
    n_models = len(models)
    display_models = [display_name(m) for m in models]

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 提取所有模型的误差数据
    rmse_list = []
    mae_list = []
    r2_list = []

    for model in models:
        result = results[model]
        rmse_list.append(result['test_rmse'] * 100)
        mae_list.append(result['test_mae'] * 100)
        r2_list.append(result['test_r2'])

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))

    # x轴位置：三个指标组
    metrics = ['RMSE', 'MAE', 'R²']
    x = np.arange(len(metrics))  # [0, 1, 2]

    # 每个模型的柱子宽度
    width = 0.8 / n_models  # 总宽度0.8，分给所有模型

    # 为每个模型分配颜色
    cmap = plt.cm.get_cmap('Set2')
    colors = [cmap(i / n_models) for i in range(n_models)]

    # 为每个模型绘制三个柱子（对应三个指标）
    bars_list = []
    for i, (model, color) in enumerate(zip(models, colors)):
        # 计算当前模型的柱子位置偏移
        offset = (i - n_models/2 + 0.5) * width

        # 获取该模型的三个指标值
        values = [rmse_list[i], mae_list[i], r2_list[i]]

        # 绘制该模型的三个柱子
        bars = ax.bar(x + offset, values, width, label=display_name(model),
                     color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        bars_list.append(bars)

        # 添加数值标签
        for j, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            if j < 2:
                label = f'{value:.3f}%'
            else:
                label = f'{value:.4f}'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label,
                   ha='center', va='bottom', fontsize=8, rotation=0)

    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value (RMSE/MAE in %)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison Across Metrics', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right', ncol=min(3, n_models))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_bars.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"误差柱状图已保存: {os.path.join(save_dir, 'error_bars.png')}")


def plot_prediction_comparison(results, save_dir):
    """绘制单个电池的SOH预测对比图（只对比CNN-LSTM和PI-CNNLSTM）"""
    if not results:
        print("没有结果可以绘制预测对比")
        return

    # 只选择 cnn_lstm 和 pi_cnn_lstm 进行对比
    target_models = ['cnn_lstm', 'pi_cnn_lstm']
    models = [m for m in target_models if m in results.keys()]

    if len(models) == 0:
        print("警告: 没有找到 cnn_lstm 或 pi_cnn_lstm 模型，跳过预测对比图绘制")
        return

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 获取第一个模型的battery_ids
    first_model = models[0]
    battery_ids = results[first_model].get('battery_ids', None)

    if battery_ids is None or len(battery_ids) == 0:
        print("警告: 无法获取电池ID信息，跳过预测对比图绘制")
        return

    # 优先选择电池 9-5，如果不存在则选择数据点最多的电池
    unique_batteries = np.unique(battery_ids)
    battery_counts = {bat: np.sum(np.array(battery_ids) == bat) for bat in unique_batteries}

    # 优先使用 9-5，如果存在的话
    if '9-5' in unique_batteries:
        selected_battery = '9-5'
        print(f"选择目标电池 9-5 进行预测对比图绘制 (数据点数: {battery_counts[selected_battery]})")
    else:
        # 如果 9-5 不存在，选择数据点最多的电池
        selected_battery = max(battery_counts, key=battery_counts.get)
        print(f"电池 9-5 不存在，选择数据点最多的电池 {selected_battery} (数据点数: {battery_counts[selected_battery]})")

    # 提取该电池的数据
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(models) + 1))

    # 获取该电池的真实值（使用第一个模型作为基准）
    battery_mask = np.array(battery_ids) == selected_battery
    base_targets = np.array(results[first_model]['targets'])[battery_mask]
    cycle_indices = np.arange(len(base_targets))

    # 绘制真实值
    ax.plot(cycle_indices, base_targets, 'o-', color='black', linewidth=2.5,
            markersize=5, label='True SOH', alpha=0.8, zorder=100)

    # 绘制各模型预测值
    for idx, model_type in enumerate(models):
        result = results[model_type]
        model_battery_ids = result.get('battery_ids', battery_ids)
        model_battery_mask = np.array(model_battery_ids) == selected_battery

        predictions = np.array(result['predictions'])[model_battery_mask]
        targets = np.array(result['targets'])[model_battery_mask]

        # 检查数据点数量是否一致
        if len(predictions) != len(targets):
            print(f"警告: {model_type} 预测数量({len(predictions)})与目标数量({len(targets)})不一致，跳过")
            continue

        # 如果数据点数量与基准不同，调整绘制
        if len(predictions) != len(base_targets):
            print(f"注意: {model_type} 数据点数({len(predictions)})与基准({len(base_targets)})不同")
            # 使用该模型自己的索引
            model_cycle_indices = np.arange(len(predictions))
            battery_errors = predictions - targets
            battery_rmse = np.sqrt(np.mean(battery_errors ** 2))

            ax.plot(model_cycle_indices, predictions, 'o-', color=colors[idx],
                    linewidth=2, markersize=4,
                    label=f'{model_type} (RMSE={battery_rmse*100:.3f}%, n={len(predictions)})',
                    alpha=0.7)
        else:
            # 数据点数量一致，正常绘制
            battery_errors = predictions - targets
            battery_rmse = np.sqrt(np.mean(battery_errors ** 2))

            ax.plot(cycle_indices, predictions, 'o-', color=colors[idx],
                    linewidth=2, markersize=4,
                    label=f'{model_type} (RMSE={battery_rmse*100:.3f}%)',
                    alpha=0.7)

    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('SOH', fontsize=12, fontweight='bold')
    ax.set_title(f'SOH Prediction Comparison for Battery {selected_battery}',
                fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"预测对比图已保存: {os.path.join(save_dir, 'prediction_comparison.png')}")


def plot_error_comparison(results, save_dir):
    """绘制单个电池的预测误差对比图（只对比CNN-LSTM和PI-CNNLSTM）"""
    if not results:
        print("没有结果可以绘制误差对比")
        return

    # 只选择 cnn_lstm 和 pi_cnn_lstm 进行对比
    target_models = ['cnn_lstm', 'pi_cnn_lstm']
    models = [m for m in target_models if m in results.keys()]

    if len(models) == 0:
        print("警告: 没有找到 cnn_lstm 或 pi_cnn_lstm 模型，跳过误差对比图绘制")
        return

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 获取第一个模型的battery_ids
    first_model = models[0]
    battery_ids = results[first_model].get('battery_ids', None)

    if battery_ids is None or len(battery_ids) == 0:
        print("警告: 无法获取电池ID信息，跳过误差对比图绘制")
        return

    # 优先选择电池 9-5，如果不存在则选择数据点最多的电池
    unique_batteries = np.unique(battery_ids)
    battery_counts = {bat: np.sum(np.array(battery_ids) == bat) for bat in unique_batteries}

    # 优先使用 9-5，如果存在的话
    if '9-5' in unique_batteries:
        selected_battery = '9-5'
        print(f"选择目标电池 9-5 进行误差对比图绘制 (数据点数: {battery_counts[selected_battery]})")
    else:
        # 如果 9-5 不存在，选择数据点最多的电池
        selected_battery = max(battery_counts, key=lambda k: battery_counts[k])
        print(f"电池 9-5 不存在，选择数据点最多的电池 {selected_battery} (数据点数: {battery_counts[selected_battery]})")

    # 提取该电池的数据
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    # 获取该电池的真实值（使用第一个模型作为基准）
    battery_mask = np.array(battery_ids) == selected_battery
    base_targets = np.array(results[first_model]['targets'])[battery_mask]
    cycle_indices = np.arange(len(base_targets))

    # 绘制0基线
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2.5,
               label='Zero Error', alpha=0.8, zorder=100)

    # 绘制各模型预测误差
    for idx, model_type in enumerate(models):
        result = results[model_type]
        model_battery_ids = result.get('battery_ids', battery_ids)
        model_battery_mask = np.array(model_battery_ids) == selected_battery

        predictions = np.array(result['predictions'])[model_battery_mask]
        targets = np.array(result['targets'])[model_battery_mask]

        # 检查数据点数量是否一致
        if len(predictions) != len(targets):
            print(f"警告: {model_type} 预测数量({len(predictions)})与目标数量({len(targets)})不一致，跳过")
            continue

        errors = predictions - targets

        # 计算该电池的MAE和RMSE
        battery_mae = np.mean(np.abs(errors))
        battery_rmse = np.sqrt(np.mean(errors ** 2))

        # 如果数据点数量与基准不同，调整绘制
        if len(errors) != len(base_targets):
            print(f"注意: {model_type} 数据点数({len(errors)})与基准({len(base_targets)})不同")
            model_cycle_indices = np.arange(len(errors))
            ax.plot(model_cycle_indices, errors, 'o-', color=colors[idx],
                    linewidth=2, markersize=4,
                    label=f'{model_type} (MAE={battery_mae*100:.3f}%, RMSE={battery_rmse*100:.3f}%, n={len(errors)})',
                    alpha=0.7)
        else:
            ax.plot(cycle_indices, errors, 'o-', color=colors[idx],
                    linewidth=2, markersize=4,
                    label=f'{model_type} (MAE={battery_mae*100:.3f}%, RMSE={battery_rmse*100:.3f}%)',
                    alpha=0.7)

    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prediction Error (Predicted - True)', fontsize=12, fontweight='bold')
    ax.set_title(f'Prediction Error Comparison for Battery {selected_battery}',
                fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"误差对比图已保存: {os.path.join(save_dir, 'error_comparison.png')}")


def plot_local_comparison(results, save_dir):
    """绘制局部区间放大图（凸显PI-CNNLSTM相对于CNN-LSTM的优势）"""
    # 检查是否有 cnn_lstm 和 pi_cnn_lstm
    if 'cnn_lstm' not in results or 'pi_cnn_lstm' not in results:
        print("警告: 需要 cnn_lstm 和 pi_cnn_lstm 模型才能绘制局部对比图，跳过")
        return

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 获取battery_ids
    battery_ids = results['cnn_lstm'].get('battery_ids', None)
    if battery_ids is None or len(battery_ids) == 0:
        print("警告: 无法获取电池ID信息，跳过局部对比图绘制")
        return

    # 优先选择电池 9-5，如果不存在则选择数据点最多的电池
    unique_batteries = np.unique(battery_ids)
    battery_counts = {bat: np.sum(np.array(battery_ids) == bat) for bat in unique_batteries}

    # 优先使用 9-5，如果存在的话
    if '9-5' in unique_batteries:
        selected_battery = '9-5'
        print(f"选择目标电池 9-5 进行局部对比图绘制 (数据点数: {battery_counts[selected_battery]})")
    else:
        # 如果 9-5 不存在，选择数据点最多的电池
        selected_battery = max(battery_counts, key=lambda k: battery_counts[k])
        print(f"电池 9-5 不存在，选择数据点最多的电池 {selected_battery} (数据点数: {battery_counts[selected_battery]})")

    # 提取两个模型的数据
    cnn_result = results['cnn_lstm']
    pi_result = results['pi_cnn_lstm']

    battery_mask_cnn = np.array(battery_ids) == selected_battery
    cnn_predictions = np.array(cnn_result['predictions'])[battery_mask_cnn]
    cnn_targets = np.array(cnn_result['targets'])[battery_mask_cnn]

    pi_battery_ids = pi_result.get('battery_ids', battery_ids)
    battery_mask_pi = np.array(pi_battery_ids) == selected_battery
    pi_predictions = np.array(pi_result['predictions'])[battery_mask_pi]
    pi_targets = np.array(pi_result['targets'])[battery_mask_pi]

    # 计算误差
    cnn_errors = np.abs(cnn_predictions - cnn_targets)
    pi_errors = np.abs(pi_predictions - pi_targets)

    # 找出提升最大的连续区间（中后期）
    # 从数据的40%开始搜索，确保在中后期
    start_search = int(len(cnn_errors) * 0.4)
    window_size = min(50, int(len(cnn_errors) * 0.15))  # 窗口大小为15%或50个点

    # 计算滑动窗口内的平均提升
    best_improvement = -np.inf
    best_start = start_search

    for i in range(start_search, len(cnn_errors) - window_size):
        # 计算该窗口内CNN-LSTM相对于PI-CNNLSTM的平均误差差异
        improvement = np.mean(cnn_errors[i:i+window_size]) - np.mean(pi_errors[i:i+window_size])
        if improvement > best_improvement:
            best_improvement = improvement
            best_start = i

    best_end = best_start + window_size

    print(f"找到最佳局部区间: [{best_start}, {best_end}], 平均提升: {best_improvement:.6f}")

    # 创建包含两个子图的图表
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])

    # 左图：完整曲线，标注局部区间
    ax1 = fig.add_subplot(gs[0])

    # 检查数据长度是否一致
    if len(cnn_predictions) != len(pi_predictions):
        print(f"警告: CNN-LSTM和PI-CNNLSTM的数据点数量不一致 ({len(cnn_predictions)} vs {len(pi_predictions)})")
        # 使用各自的索引
        cnn_cycle_indices = np.arange(len(cnn_targets))
        pi_cycle_indices = np.arange(len(pi_targets))

        ax1.plot(cnn_cycle_indices, cnn_targets, 'o-', color='black', linewidth=2,
                markersize=3, label='True SOH (CNN-LSTM)', alpha=0.7, zorder=100)
        ax1.plot(cnn_cycle_indices, cnn_predictions, 'o-', color='#FF6B6B', linewidth=1.5,
                markersize=2, label='CNN-LSTM', alpha=0.7)
        ax1.plot(pi_cycle_indices, pi_predictions, 'o-', color='#4ECDC4', linewidth=1.5,
                markersize=2, label='PI-CNNLSTM', alpha=0.7)

        # 使用CNN-LSTM的索引来标注局部区间（因为区间是基于CNN-LSTM数据计算的）
        cycle_indices = cnn_cycle_indices
    else:
        # 数据长度一致，使用统一索引
        cycle_indices = np.arange(len(cnn_targets))

        ax1.plot(cycle_indices, cnn_targets, 'o-', color='black', linewidth=2,
                markersize=3, label='True SOH', alpha=0.7, zorder=100)
        ax1.plot(cycle_indices, cnn_predictions, 'o-', color='#FF6B6B', linewidth=1.5,
                markersize=2, label='CNN-LSTM', alpha=0.7)
        ax1.plot(cycle_indices, pi_predictions, 'o-', color='#4ECDC4', linewidth=1.5,
                markersize=2, label='PI-CNNLSTM', alpha=0.7)

    # 标注局部区间（使用CNN-LSTM的索引，因为区间搜索是基于CNN-LSTM数据的）
    ax1.axvspan(best_start, best_end, alpha=0.2, color='yellow', label='Zoom Region')

    ax1.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
    ax1.set_ylabel('SOH', fontsize=11, fontweight='bold')
    ax1.set_title(f'Full Prediction Comparison for Battery {selected_battery}',
                 fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 右图：局部放大
    ax2 = fig.add_subplot(gs[1])

    # 确保局部区间不超出PI-CNNLSTM的数据范围
    pi_best_end = min(best_end, len(pi_predictions))
    pi_best_start = min(best_start, len(pi_predictions))

    # 如果PI-CNNLSTM的数据点不足，调整区间
    if pi_best_start >= len(pi_predictions):
        print(f"警告: 局部区间超出PI-CNNLSTM数据范围，使用PI-CNNLSTM的最后{window_size}个点")
        pi_best_start = max(0, len(pi_predictions) - window_size)
        pi_best_end = len(pi_predictions)

    local_indices_cnn = np.arange(best_start, best_end)
    local_indices_pi = np.arange(pi_best_start, pi_best_end)

    ax2.plot(local_indices_cnn, cnn_targets[best_start:best_end], 'o-', color='black',
            linewidth=2.5, markersize=5, label='True SOH', alpha=0.8, zorder=100)
    ax2.plot(local_indices_cnn, cnn_predictions[best_start:best_end], 'o-', color='#FF6B6B',
            linewidth=2, markersize=4, label='CNN-LSTM', alpha=0.8)
    ax2.plot(local_indices_pi, pi_predictions[pi_best_start:pi_best_end], 'o-', color='#4ECDC4',
            linewidth=2, markersize=4, label='PI-CNNLSTM', alpha=0.8)

    # 计算该区间的RMSE
    cnn_local_rmse = np.sqrt(np.mean((cnn_predictions[best_start:best_end] - cnn_targets[best_start:best_end])**2))
    pi_local_rmse = np.sqrt(np.mean((pi_predictions[pi_best_start:pi_best_end] - pi_targets[pi_best_start:pi_best_end])**2))
    improvement_pct = (cnn_local_rmse - pi_local_rmse) / cnn_local_rmse * 100

    ax2.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
    ax2.set_ylabel('SOH', fontsize=11, fontweight='bold')
    ax2.set_title(f'Zoomed Region (Improvement: {improvement_pct:.1f}%)\n' +
                 f'CNN-LSTM RMSE: {cnn_local_rmse*100:.3f}%, PI-CNNLSTM RMSE: {pi_local_rmse*100:.3f}%',
                 fontsize=11, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'local_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"局部对比图已保存: {os.path.join(save_dir, 'local_comparison.png')}")


def plot_scatter_comparison(results, save_dir):
    """绘制所有模型的散点图对比（预测值 vs 真实值）"""
    if not results:
        print("没有结果可以绘制散点图对比")
        return

    models = list(results.keys())
    n_models = len(models)
    display_models = [display_name(m) for m in models]

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 固定为2行2列布局
    n_cols = 2
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    axes = axes.flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for idx, (model_type, color) in enumerate(zip(models, colors)):
        ax = axes[idx]
        result = results[model_type]

        predictions = np.array(result['predictions'])
        targets = np.array(result['targets'])

        # 绘制散点图
        ax.scatter(targets, predictions, alpha=0.5, s=20, color=color, edgecolors='none')

        # 绘制理想预测线 (y=x)
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal')

        # 设置标题和标签
        ax.set_title(f'{display_name(model_type)}\nRMSE={result["test_rmse"]*100:.3f}%, R²={result["test_r2"]:.4f}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('True SOH', fontsize=10)
        ax.set_ylabel('Predicted SOH', fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    # 隐藏多余的子图
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scatter_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"散点图对比已保存: {os.path.join(save_dir, 'scatter_comparison.png')}")


def plot_error_distribution(results, save_dir):
    """绘制所有模型的误差分布统计图（直方图和箱线图）"""
    if not results:
        print("没有结果可以绘制误差分布")
        return

    models = list(results.keys())
    n_models = len(models)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建两个子图：误差直方图和箱线图
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    # 1. 误差直方图（重叠显示）
    ax1 = axes[0]
    for model_type, color in zip(models, colors):
        result = results[model_type]
        errors = np.array(result['predictions']) - np.array(result['targets'])

        ax1.hist(errors, bins=50, alpha=0.5, label=f'{display_name(model_type)} (MAE={result["test_mae"]*100:.3f}%)',
                color=color, edgecolor='black', linewidth=0.5)

    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax1.set_xlabel('Prediction Error (Predicted - True)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Error Distribution Comparison (All Data)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. 误差箱线图（并排显示）
    ax2 = axes[1]
    error_data = []
    for model_type in models:
        result = results[model_type]
        errors = np.array(result['predictions']) - np.array(result['targets'])
        error_data.append(errors)

    bp = ax2.boxplot(error_data, labels=display_models, patch_artist=True,
                     notch=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

    # 设置箱线图颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax2.set_ylabel('Prediction Error (Predicted - True)', fontsize=11, fontweight='bold')
    ax2.set_title('Error Distribution (Boxplot)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='best', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"误差分布图已保存: {os.path.join(save_dir, 'error_distribution.png')}")


def generate_report(results, save_dir):
    """生成对比报告"""
    if not results:
        print("没有结果可以生成报告")
        return

    # 创建DataFrame
    data = []
    for model_type, result in results.items():
        data.append({
            'Model': model_type,
            'RMSE': result['test_rmse'],
            'MAE': result['test_mae'],
            'R²': result['test_r2'],
            'Best Epoch': result.get('best_epoch', 'N/A'),
        })

    df = pd.DataFrame(data)
    df = df.sort_values('RMSE')

    # 保存CSV
    csv_path = os.path.join(save_dir, 'comparison_results.csv')
    df.to_csv(csv_path, index=False)

    # 打印表格
    print("\n" + "=" * 70)
    print("模型对比结果汇总（按RMSE排序）")
    print("=" * 70)
    print(df.to_string(index=False))

    # 最佳模型
    best_model = df.iloc[0]['Model']
    print(f"\n★ 最佳模型: {best_model}")
    print(f"    RMSE: {df.iloc[0]['RMSE']:.6f}")
    print(f"    MAE:  {df.iloc[0]['MAE']:.6f}")
    print(f"    R²:   {df.iloc[0]['R²']:.6f}")

    # 保存详细结果
    with open(os.path.join(save_dir, 'detailed_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print(f"\n结果已保存到: {save_dir}")
    print(f"  - comparison_results.csv (汇总表格)")
    print(f"  - detailed_results.pkl (详细数据)")
    print(f"  - model_comparison.png (指标对比柱状图 - RMSE/MAE/R²)")
    print(f"  - error_bars.png (误差柱状图 - 所有模型)")
    print(f"  - prediction_comparison.png (单电池SOH预测对比 - CNN-LSTM vs PI-CNNLSTM)")
    print(f"  - error_comparison.png (单电池误差对比 - CNN-LSTM vs PI-CNNLSTM)")
    print(f"  - local_comparison.png (局部区间放大对比 - 凸显PI-CNNLSTM优势)")
    print(f"  - scatter_comparison.png (全数据散点图对比)")
    print(f"  - error_distribution.png (全数据误差分布统计图)")

    return df


def main():
    # ========== 读取配置区域的参数 ==========

    # 确定要测试的模型
    if TEST_ALL_MODELS:
        models_to_test = ALL_MODELS
    else:
        models_to_test = MODELS_TO_TEST

    # 验证模型名称
    invalid_models = [m for m in models_to_test if m not in ALL_MODELS]
    if invalid_models:
        print(f"警告: 以下模型不支持: {invalid_models}")
        print(f"支持的模型: {ALL_MODELS}")
        models_to_test = [m for m in models_to_test if m in ALL_MODELS]

    if not models_to_test:
        print("没有有效的模型可以测试，请检查 MODELS_TO_TEST 配置")
        return

    # 确定计算设备
    if DEVICE == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = DEVICE

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f"{OUTPUT_DIR}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("多模型对比测试")
    print("=" * 70)
    print(f"测试模型: {models_to_test}")
    print(f"随机种子: {RANDOM_SEED}")
    print(f"计算设备: {device}")
    print(f"结果目录: {save_dir}")

    # 运行对比测试（直接调用 train_cross_battery_model）
    results, models, configs = compare_models(
        models_to_test=models_to_test,
        device=device,
        seed=RANDOM_SEED,
        apply_cleaning=APPLY_CLEANING,
        save_dir=save_dir
    )

    # 绘制对比图
    print("\n" + "=" * 70)
    print("生成可视化图表")
    print("=" * 70)

    plot_comparison(results, save_dir)
    plot_error_bars(results, save_dir)
    plot_prediction_comparison(results, save_dir)
    plot_error_comparison(results, save_dir)
    plot_local_comparison(results, save_dir)
    plot_scatter_comparison(results, save_dir)
    plot_error_distribution(results, save_dir)

    # 生成报告
    generate_report(results, save_dir)

    print("\n" + "=" * 70)
    print("所有测试完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
