"""
统一推理脚本 - 配置式电池SOH预测

只需修改下面的配置部分，然后运行：
    python inference.py

无需命令行参数！
"""

import sys
import io

# 设置stdout编码为UTF-8，避免Windows控制台emoji显示问题
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# 📝 配置区域 - 只需修改这里！
# ============================================================================

CONFIG = {
    # 模型设置
    'model_path': 'results/cross_battery/CNN_LSTM/model_checkpoint.pth',  # 模型路径
    'data_dir': 'data/HUST data',                                          # 数据目录
    'output_dir': 'inference_results',                                     # 结果保存目录

    # 推理模式选择（三选一）
    'mode': 'single',  # 'single' = 单电池, 'batch' = 批量, 'test_set' = 测试集

    # 单电池模式配置（mode='single'时使用）
    'single_battery_id': '4-3',

    # 批量模式配置（mode='batch'时使用）
    'batch_batteries': ['4-3', '6-2', '10-1', '3-8'],  # 要预测的电池列表
}

# ============================================================================
# 代码实现部分 - 不需要修改
# ============================================================================

from models import UnifiedModelWrapper
from data_loaders.data_loader_hust import load_all_hust_batteries, HUSTBatteryDatasetWithMetadata
from torch.utils.data import DataLoader


def load_model(checkpoint_path):
    """加载已训练的模型"""
    print(f"\n{'='*70}")
    print("📦 加载模型")
    print(f"{'='*70}")
    print(f"路径: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ 模型文件不存在: {checkpoint_path}")

    # 使用 weights_only=False 以兼容旧版本模型
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', None)

    if config is None:
        raise ValueError("❌ Checkpoint中缺少配置信息")

    print(f"✓ 模型类型: {config['model_type']}")
    print(f"✓ 训练轮数: {checkpoint.get('epoch', 'Unknown')}")
    print(f"✓ 最佳MAE: {checkpoint.get('best_mae', 'Unknown')}")

    # 使用静态方法加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wrapper = UnifiedModelWrapper.load_checkpoint(checkpoint_path, device=device)
    wrapper.model.eval()

    print("✓ 模型加载成功！")
    return wrapper, config


def load_battery_data(battery_id, data_dir, window_size):
    """加载单个电池的数据"""
    # 加载所有电池数据（train_ratio=1.0 确保获取全部数据）
    all_batteries = load_all_hust_batteries(data_dir=data_dir, train_ratio=1.0)

    if battery_id not in all_batteries:
        available = ', '.join(sorted(all_batteries.keys())[:10])
        raise ValueError(f"❌ 电池 {battery_id} 不存在。可用电池示例: {available}...")

    battery_data = all_batteries[battery_id]

    # 使用训练集特征（train_ratio=1.0时包含所有数据）
    features = battery_data['train_features']
    soh = battery_data['train_capacity']

    print(f"✓ 成功加载电池 {battery_id}")
    print(f"  总循环数: {len(soh)}")
    print(f"  特征维度: {features.shape[1]}")
    print(f"  SOH范围: {soh.min():.4f} - {soh.max():.4f}")

    # 创建滑动窗口
    X_windows = []
    y_soh = []
    cycle_indices = []

    for i in range(window_size, len(features)):
        window = features[i-window_size:i]
        X_windows.append(window)
        y_soh.append(soh[i])
        cycle_indices.append(i)

    X = np.array(X_windows)
    y = np.array(y_soh).reshape(-1, 1)
    cycle_indices = np.array(cycle_indices)

    return X, y, cycle_indices, battery_id


def predict(wrapper, X, config):
    """使用模型进行预测"""
    battery_ids = ['inference'] * len(X)
    cycle_indices = np.arange(len(X))

    dataset = HUSTBatteryDatasetWithMetadata(
        X, X[:, 0:1, :],
        battery_ids, cycle_indices,
        siamese_mode=False,
        triplet_mode=False,
        step_k=1,
        mode='test'
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    # 模型已经在load_model时移到正确设备
    device = wrapper.device

    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                x = batch['window'].to(device)
            else:
                x = batch[0].to(device)

            pred = wrapper.model(x)
            predictions.append(pred.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0).flatten()
    return predictions


def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    max_error = np.max(np.abs(y_true - y_pred))

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Max_Error': max_error
    }


def visualize_single(y_true, y_pred, cycle_indices, battery_id, save_dir):
    """可视化单电池结果"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Battery {battery_id} - SOH Prediction Results',
                 fontsize=16, fontweight='bold')

    # 1. 时间序列
    ax1 = axes[0, 0]
    ax1.plot(cycle_indices, y_true, 'b-', label='True SOH', linewidth=2, alpha=0.7)
    ax1.plot(cycle_indices, y_pred, 'r--', label='Predicted SOH', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Cycle Index', fontsize=12)
    ax1.set_ylabel('SOH', fontsize=12)
    ax1.set_title('SOH Prediction vs True Values', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 2. 散点图
    ax2 = axes[0, 1]
    ax2.scatter(y_true, y_pred, alpha=0.5, s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    ax2.set_xlabel('True SOH', fontsize=12)
    ax2.set_ylabel('Predicted SOH', fontsize=12)
    ax2.set_title('Prediction Scatter Plot', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # 3. 误差曲线
    ax3 = axes[1, 0]
    errors = y_pred - y_true
    ax3.plot(cycle_indices, errors, 'g-', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.fill_between(cycle_indices, errors, 0, alpha=0.3, color='g')
    ax3.set_xlabel('Cycle Index', fontsize=12)
    ax3.set_ylabel('Prediction Error', fontsize=12)
    ax3.set_title('Prediction Error Over Time', fontsize=14)
    ax3.grid(True, alpha=0.3)

    # 4. 误差分布
    ax4 = axes[1, 1]
    ax4.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax4.axvline(x=np.mean(errors), color='g', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(errors):.4f}')
    ax4.set_xlabel('Prediction Error', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Error Distribution', fontsize=14)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{battery_id}_predictions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 图表已保存: {save_path}")
    # plt.show()  # 已禁用弹窗显示


def visualize_batch(all_results, output_dir):
    """可视化批量结果"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Batch Inference Results ({len(all_results)} Batteries)',
                 fontsize=16, fontweight='bold')

    battery_ids = list(all_results.keys())

    # 1. MAE对比
    ax1 = axes[0, 0]
    maes = [all_results[bid]['metrics']['MAE'] * 100 for bid in battery_ids]
    ax1.bar(range(len(battery_ids)), maes, alpha=0.7, color='blue')
    ax1.axhline(y=np.mean(maes), color='r', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(maes):.4f}%')
    ax1.set_xlabel('Battery Index', fontsize=12)
    ax1.set_ylabel('MAE (%)', fontsize=12)
    ax1.set_title('MAE per Battery', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. R²对比
    ax2 = axes[0, 1]
    r2s = [all_results[bid]['metrics']['R2'] for bid in battery_ids]
    ax2.bar(range(len(battery_ids)), r2s, alpha=0.7, color='green')
    ax2.axhline(y=np.mean(r2s), color='r', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(r2s):.6f}')
    ax2.set_xlabel('Battery Index', fontsize=12)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('R² per Battery', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 所有电池散点图
    ax3 = axes[1, 0]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(battery_ids)))

    for idx, (bid, color) in enumerate(zip(battery_ids, colors)):
        y_true = all_results[bid]['y_true']
        y_pred = all_results[bid]['y_pred']
        ax3.scatter(y_true, y_pred, alpha=0.3, s=10, color=color,
                   label=bid if idx < 10 else '')

    all_true = np.concatenate([all_results[bid]['y_true'] for bid in battery_ids])
    all_pred = np.concatenate([all_results[bid]['y_pred'] for bid in battery_ids])
    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    ax3.set_xlabel('True SOH', fontsize=12)
    ax3.set_ylabel('Predicted SOH', fontsize=12)
    ax3.set_title('All Batteries: Prediction vs True', fontsize=14)
    if len(battery_ids) <= 10:
        ax3.legend(fontsize=8, loc='best')
    ax3.grid(True, alpha=0.3)

    # 4. 总体误差分布
    ax4 = axes[1, 1]
    all_errors = []
    for bid in battery_ids:
        errors = all_results[bid]['y_pred'] - all_results[bid]['y_true']
        all_errors.extend(errors)

    ax4.hist(all_errors, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax4.axvline(x=np.mean(all_errors), color='g', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(all_errors):.4f}')
    ax4.set_xlabel('Prediction Error', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Error Distribution (All Batteries)', fontsize=14)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'batch_results_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 批量图表已保存: {save_path}")
    # plt.show()  # 已禁用弹窗显示


def save_single_results(y_true, y_pred, cycle_indices, metrics, battery_id, save_dir):
    """保存单电池结果"""
    os.makedirs(save_dir, exist_ok=True)

    results = {
        'battery_id': battery_id,
        'cycle_indices': cycle_indices.tolist(),
        'true_soh': y_true.tolist(),
        'predicted_soh': y_pred.tolist(),
        'metrics': metrics
    }

    save_path = os.path.join(save_dir, f'{battery_id}_results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ 结果已保存: {save_path}")


def save_batch_summary(all_results, output_dir):
    """保存批量汇总"""
    all_mae = [r['metrics']['MAE'] for r in all_results.values()]
    all_rmse = [r['metrics']['RMSE'] for r in all_results.values()]
    all_r2 = [r['metrics']['R2'] for r in all_results.values()]

    summary = {
        'total_batteries': len(all_results),
        'metrics_summary': {
            'MAE': {
                'mean': np.mean(all_mae) * 100,
                'std': np.std(all_mae) * 100,
                'min': np.min(all_mae) * 100,
                'max': np.max(all_mae) * 100
            },
            'RMSE': {
                'mean': np.mean(all_rmse) * 100,
                'std': np.std(all_rmse) * 100,
                'min': np.min(all_rmse) * 100,
                'max': np.max(all_rmse) * 100
            },
            'R2': {
                'mean': np.mean(all_r2),
                'std': np.std(all_r2),
                'min': np.min(all_r2),
                'max': np.max(all_r2)
            }
        },
        'per_battery': {
            bid: {
                'MAE': r['metrics']['MAE'] * 100,
                'RMSE': r['metrics']['RMSE'] * 100,
                'R2': r['metrics']['R2']
            }
            for bid, r in all_results.items()
        }
    }

    summary_file = os.path.join(output_dir, 'summary_report.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n📊 批量统计:")
    print(f"  总电池数: {summary['total_batteries']}")
    print(f"  平均MAE: {summary['metrics_summary']['MAE']['mean']:.4f}%")
    print(f"  平均RMSE: {summary['metrics_summary']['RMSE']['mean']:.4f}%")
    print(f"  平均R²: {summary['metrics_summary']['R2']['mean']:.6f}")
    print(f"\n✓ 汇总报告已保存: {summary_file}")


def run_single_inference(config, wrapper, model_config):
    """执行单电池推理"""
    battery_id = config['single_battery_id']
    output_dir = os.path.join(config['output_dir'], battery_id)

    print(f"\n{'='*70}")
    print(f"🔍 单电池推理: {battery_id}")
    print(f"{'='*70}")

    # 加载数据
    X, y_true, cycle_indices, _ = load_battery_data(
        battery_id, config['data_dir'], model_config['data']['window_size']
    )
    print(f"✓ 数据加载完成: {len(X)}个样本")

    # 预测
    print("⏳ 正在预测...")
    y_pred = predict(wrapper, X, model_config)
    print(f"✓ 预测完成")

    # 计算指标
    y_true_flat = y_true.flatten()
    metrics = calculate_metrics(y_true_flat, y_pred)

    print(f"\n📈 评估指标:")
    print(f"  MAE:        {metrics['MAE']*100:.4f}%")
    print(f"  RMSE:       {metrics['RMSE']*100:.4f}%")
    print(f"  MAPE:       {metrics['MAPE']:.4f}%")
    print(f"  R²:         {metrics['R2']:.6f}")
    print(f"  最大误差:   {metrics['Max_Error']*100:.4f}%")

    # 可视化
    print("\n📊 生成图表...")
    visualize_single(y_true_flat, y_pred, cycle_indices, battery_id, output_dir)

    # 保存结果
    save_single_results(y_true_flat, y_pred, cycle_indices, metrics, battery_id, output_dir)

    print(f"\n✅ 单电池推理完成！结果保存在: {output_dir}")


def run_batch_inference(config, wrapper, model_config, battery_ids):
    """执行批量推理"""
    output_dir = os.path.join(config['output_dir'], 'batch')

    print(f"\n{'='*70}")
    print(f"🔍 批量推理: {len(battery_ids)}个电池")
    print(f"{'='*70}")
    print(f"电池列表: {battery_ids}")

    all_results = {}
    window_size = model_config['data']['window_size']

    for battery_id in tqdm(battery_ids, desc="推理进度"):
        try:
            # 加载数据
            X, y_true, cycle_indices, _ = load_battery_data(
                battery_id, config['data_dir'], window_size
            )

            # 预测
            y_pred = predict(wrapper, X, model_config)

            # 计算指标
            y_true_flat = y_true.flatten()
            metrics = calculate_metrics(y_true_flat, y_pred)

            # 保存结果
            all_results[battery_id] = {
                'y_true': y_true_flat,
                'y_pred': y_pred,
                'cycle_indices': cycle_indices,
                'metrics': metrics
            }

            # 保存单个电池结果
            battery_dir = os.path.join(output_dir, 'individual', battery_id)
            os.makedirs(battery_dir, exist_ok=True)

            # 保存JSON结果
            result_file = os.path.join(battery_dir, 'results.json')
            with open(result_file, 'w') as f:
                json.dump({
                    'battery_id': battery_id,
                    'cycle_indices': cycle_indices.tolist(),
                    'true_soh': y_true_flat.tolist(),
                    'predicted_soh': y_pred.tolist(),
                    'metrics': metrics
                }, f, indent=2)

            # 生成单个电池的可视化图表
            visualize_single(y_true_flat, y_pred, cycle_indices, battery_id, battery_dir)

        except Exception as e:
            print(f"\n⚠️  电池 {battery_id} 推理失败: {str(e)}")
            continue

    print(f"\n✓ 成功推理 {len(all_results)}/{len(battery_ids)} 个电池")

    # 生成汇总
    save_batch_summary(all_results, output_dir)

    # 可视化
    print("\n📊 生成批量图表...")
    visualize_batch(all_results, output_dir)

    print(f"\n✅ 批量推理完成！结果保存在: {output_dir}")


def main():
    """主函数"""
    print("="*70)
    print("🚀 电池SOH预测 - 统一推理脚本")
    print("="*70)

    # 显示配置
    print(f"\n📝 当前配置:")
    print(f"  模式: {CONFIG['mode']}")
    print(f"  模型: {CONFIG['model_path']}")
    print(f"  数据目录: {CONFIG['data_dir']}")

    if CONFIG['mode'] == 'single':
        print(f"  预测电池: {CONFIG['single_battery_id']}")
    elif CONFIG['mode'] == 'batch':
        print(f"  电池列表: {CONFIG['batch_batteries']}")

    # 加载模型
    wrapper, model_config = load_model(CONFIG['model_path'])

    # 执行推理
    if CONFIG['mode'] == 'single':
        run_single_inference(CONFIG, wrapper, model_config)

    elif CONFIG['mode'] == 'batch':
        run_batch_inference(CONFIG, wrapper, model_config, CONFIG['batch_batteries'])

    elif CONFIG['mode'] == 'test_set':
        # 加载测试集电池列表
        model_dir = os.path.dirname(CONFIG['model_path'])
        split_file = os.path.join(model_dir, 'battery_split.json')

        if not os.path.exists(split_file):
            print(f"❌ 找不到电池划分文件: {split_file}")
            print("   请使用 'batch' 模式并手动指定电池列表")
            return

        with open(split_file, 'r') as f:
            split_info = json.load(f)

        test_batteries = split_info['test_batteries']
        print(f"\n从 {split_file} 加载测试集: {len(test_batteries)}个电池")

        run_batch_inference(CONFIG, wrapper, model_config, test_batteries)

    else:
        print(f"❌ 未知模式: {CONFIG['mode']}")
        print("   支持的模式: 'single', 'batch', 'test_set'")

    print(f"\n{'='*70}")
    print("✨ 推理完成！")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
