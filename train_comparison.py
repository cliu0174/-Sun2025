"""
批量训练和对比多个模型的脚本。

支持：
1. 在同一数据集上训练多个模型
2. 自动保存结果和对比图表
3. 生成详细的对比报告
"""

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from train_single_model import train_model


def train_and_compare_models(
    model_types=['fnn', 'cnn', 'lstm', 'bpinn'],
    battery_id='1-1',
    device='cuda'
):
    """
    训练多个模型并生成对比报告。

    Args:
        model_types: 要训练的模型类型列表
        battery_id: 电池ID
        device: 计算设备
    """

    print("\n" + "=" * 70)
    print("批量模型训练与对比")
    print("=" * 70)
    print(f"数据集: HUST Battery {battery_id}")
    print(f"模型: {', '.join([m.upper() for m in model_types])}")
    print(f"设备: {device}")
    print("=" * 70)

    # 存储所有模型的结果
    all_results = {}
    all_wrappers = {}

    # 训练每个模型
    for i, model_type in enumerate(model_types, 1):
        print(f"\n\n{'='*70}")
        print(f"模型 {i}/{len(model_types)}: {model_type.upper()}")
        print('='*70)

        try:
            wrapper, results = train_model(
                model_type=model_type,
                battery_id=battery_id,
                device=device
            )

            all_results[model_type] = results
            all_wrappers[model_type] = wrapper

            print(f"✅ {model_type.upper()} 训练完成")

        except Exception as e:
            print(f"❌ {model_type.upper()} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 生成对比结果
    if len(all_results) > 0:
        print("\n\n" + "=" * 70)
        print("生成对比报告")
        print("=" * 70)

        comparison_dir = f'results/comparison_hust/{battery_id}'
        os.makedirs(comparison_dir, exist_ok=True)

        # 1. 生成对比表格
        generate_comparison_table(all_results, comparison_dir)

        # 2. 绘制对比图表
        plot_comparison_charts(all_results, comparison_dir)

        # 3. 保存详细报告
        save_comparison_report(all_results, all_wrappers, battery_id, comparison_dir)

        print(f"\n对比结果已保存到: {comparison_dir}/")

    return all_results, all_wrappers


def generate_comparison_table(all_results, save_dir):
    """生成对比表格"""
    print("\n生成对比表格...")

    # 创建DataFrame
    data = {
        'Model': [],
        'MAE (%)': [],
        'RMSE (%)': [],
        'MAPE (%)': [],
        'Best Epoch': []
    }

    for model_type, results in all_results.items():
        data['Model'].append(model_type.upper())
        data['MAE (%)'].append(results['mae'] * 100)
        data['RMSE (%)'].append(results['rmse'] * 100)
        data['MAPE (%)'].append(results['mape'])
        data['Best Epoch'].append(results['best_epoch'])

    df = pd.DataFrame(data)

    # 添加排名
    df['MAE Rank'] = df['MAE (%)'].rank()
    df['RMSE Rank'] = df['RMSE (%)'].rank()

    # 保存CSV
    csv_path = os.path.join(save_dir, 'comparison_table.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')

    # 打印表格
    print("\n对比结果:")
    print(df.to_string(index=False))

    # 找出最佳模型
    best_mae_model = df.loc[df['MAE (%)'].idxmin(), 'Model']
    best_rmse_model = df.loc[df['RMSE (%)'].idxmin(), 'Model']

    print(f"\n🏆 最佳模型 (MAE):  {best_mae_model}")
    print(f"🏆 最佳模型 (RMSE): {best_rmse_model}")

    return df


def plot_comparison_charts(all_results, save_dir):
    """绘制对比图表"""
    print("\n绘制对比图表...")

    # 提取数据
    models = list(all_results.keys())
    mae_values = [all_results[m]['mae'] * 100 for m in models]
    rmse_values = [all_results[m]['rmse'] * 100 for m in models]

    # 1. 指标对比柱状图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MAE对比
    bars1 = axes[0].bar(models, mae_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)])
    axes[0].set_ylabel('MAE (%)')
    axes[0].set_title('MAE Comparison')
    axes[0].grid(axis='y', alpha=0.3)

    # 在柱子上标注数值
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

    # RMSE对比
    bars2 = axes[1].bar(models, rmse_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)])
    axes[1].set_ylabel('RMSE (%)')
    axes[1].set_title('RMSE Comparison')
    axes[1].grid(axis='y', alpha=0.3)

    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300)
    plt.close()

    # 2. 训练曲线对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (model, results) in enumerate(all_results.items()):
        history = results['history']
        color = colors[i % len(colors)]

        # MAE曲线
        mae_curve = [mae * 100 for mae in history['test_mae']]
        axes[0].plot(mae_curve, label=model.upper(), color=color, linewidth=2)

        # RMSE曲线
        rmse_curve = [rmse * 100 for rmse in history['test_rmse']]
        axes[1].plot(rmse_curve, label=model.upper(), color=color, linewidth=2)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MAE (%)')
    axes[0].set_title('MAE Training Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('RMSE (%)')
    axes[1].set_title('RMSE Training Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves_comparison.png'), dpi=300)
    plt.close()

    # 3. 预测对比散点图
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for i, (model, results) in enumerate(all_results.items()):
        if i >= 4:
            break

        ax = axes[i]
        predictions = results['predictions']
        targets = results['targets']

        ax.scatter(targets, predictions, alpha=0.5, s=20)
        ax.plot([min(targets), max(targets)],
                [min(targets), max(targets)],
                'r--', lw=2, label='Perfect')

        ax.set_xlabel('True SOH')
        ax.set_ylabel('Predicted SOH')
        ax.set_title(f'{model.upper()} (MAE: {results["mae"]*100:.3f}%)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'predictions_comparison.png'), dpi=300)
    plt.close()

    print("图表已保存")


def save_comparison_report(all_results, all_wrappers, battery_id, save_dir):
    """保存详细的对比报告"""
    print("\n生成详细报告...")

    from models import ModelFactory

    report_path = os.path.join(save_dir, 'comparison_report.md')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 模型对比报告\n\n")
        f.write(f"**数据集**: HUST Battery {battery_id}\n\n")
        f.write(f"**训练时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"---\n\n")

        # 1. 总体结果
        f.write(f"## 1. 总体结果\n\n")
        f.write(f"| 模型 | MAE (%) | RMSE (%) | MAPE (%) | 最佳Epoch |\n")
        f.write(f"|------|---------|----------|----------|----------|\n")

        for model_type, results in all_results.items():
            f.write(f"| {model_type.upper()} | "
                   f"{results['mae']*100:.4f} | "
                   f"{results['rmse']*100:.4f} | "
                   f"{results['mape']:.4f} | "
                   f"{results['best_epoch']} |\n")

        # 2. 模型详情
        f.write(f"\n## 2. 模型详细信息\n\n")

        for model_type, wrapper in all_wrappers.items():
            f.write(f"### {model_type.upper()}\n\n")

            # 模型信息
            info = ModelFactory.get_model_info(wrapper.model)
            f.write(f"- **总参数量**: {info['total_parameters']:,}\n")
            f.write(f"- **模型大小**: {info['model_size_mb']:.2f} MB\n")

            # 配置信息
            config = wrapper.config
            f.write(f"- **隐藏层配置**: {config['architecture'].get('hidden_sizes', 'N/A')}\n")
            f.write(f"- **学习率**: {config['training']['learning_rate']}\n")
            f.write(f"- **批次大小**: {config['training']['batch_size']}\n")
            f.write(f"- **训练轮数**: {config['training']['num_epochs']}\n\n")

        # 3. 结论
        f.write(f"## 3. 结论\n\n")

        # 找出最佳模型
        best_mae = min(all_results.items(), key=lambda x: x[1]['mae'])
        best_rmse = min(all_results.items(), key=lambda x: x[1]['rmse'])

        f.write(f"- **MAE最佳模型**: {best_mae[0].upper()} "
               f"({best_mae[1]['mae']*100:.4f}%)\n")
        f.write(f"- **RMSE最佳模型**: {best_rmse[0].upper()} "
               f"({best_rmse[1]['rmse']*100:.4f}%)\n\n")

        # 性能排名
        f.write(f"### 性能排名 (按MAE)\n\n")
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['mae'])
        for rank, (model, results) in enumerate(sorted_results, 1):
            f.write(f"{rank}. **{model.upper()}**: {results['mae']*100:.4f}%\n")

        f.write(f"\n---\n\n")
        f.write(f"*报告自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    print(f"详细报告已保存: {report_path}")


if __name__ == "__main__":
    """
    使用示例：

    1. 对比所有模型:
        python train_comparison.py

    2. 对比特定模型:
        修改下面的MODEL_TYPES列表
    """

    # ===== 配置参数 =====
    MODEL_TYPES = ['fnn', 'cnn', 'lstm', 'bpinn']  # 要对比的模型
    BATTERY_ID = '1-1'                              # 电池ID
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ===== 开始训练 =====
    print(f"\n使用设备: {DEVICE}\n")

    all_results, all_wrappers = train_and_compare_models(
        model_types=MODEL_TYPES,
        battery_id=BATTERY_ID,
        device=DEVICE
    )

    print("\n" + "=" * 70)
    print("所有训练和对比已完成！")
    print("=" * 70)

    # 打印最终总结
    if all_results:
        print("\n最终结果总结:")
        for model, results in all_results.items():
            print(f"  {model.upper():<8} - MAE: {results['mae']*100:.4f}%, "
                  f"RMSE: {results['rmse']*100:.4f}%")
