"""
基准模型对比实验：FNN, CNN, LSTM

目的：找出性能最好的基准模型，为后续PINN实验做准备
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from train_cross_battery import train_cross_battery_model


def train_baseline_models(
    model_types=['fnn', 'cnn', 'lstm'],
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    device='cuda',
    seed=42
):
    """
    训练所有基准模型并对比。

    Args:
        model_types: 要对比的模型类型列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        device: 计算设备
        seed: 随机种子
    """
    print("\n" + "="*70)
    print("基准模型对比实验")
    print("="*70)
    print(f"模型: {', '.join([m.upper() for m in model_types])}")
    print(f"数据划分: {train_ratio*100:.0f}%/{val_ratio*100:.0f}%/{test_ratio*100:.0f}%")
    print(f"设备: {device}")
    print("="*70)

    results_summary = {}

    # 训练每个模型
    for i, model_type in enumerate(model_types, 1):
        print(f"\n\n{'='*70}")
        print(f"训练模型 {i}/{len(model_types)}: {model_type.upper()}")
        print('='*70)

        try:
            wrapper, results, data_dict = train_cross_battery_model(
                model_type=model_type,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                device=device,
                seed=seed
            )

            results_summary[model_type] = {
                'test_mae': results['test_mae'],
                'test_rmse': results['test_rmse'],
                'test_mape': results['test_mape'],
                'best_val_mae': results['best_val_mae'],
                'best_epoch': results['best_epoch']
            }

            print(f"\n✅ {model_type.upper()} 训练完成")
            print(f"   测试集MAE: {results['test_mae']*100:.4f}%")

        except Exception as e:
            print(f"\n❌ {model_type.upper()} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 生成对比报告
    if len(results_summary) > 0:
        generate_comparison_report(results_summary)

    return results_summary


def generate_comparison_report(results_summary):
    """生成详细的对比报告"""

    print("\n\n" + "="*70)
    print("基准模型对比报告")
    print("="*70)

    # 1. 创建对比表格
    data = {
        'Model': [],
        'Test MAE (%)': [],
        'Test RMSE (%)': [],
        'Test MAPE (%)': [],
        'Best Val MAE (%)': [],
        'Best Epoch': []
    }

    for model, results in results_summary.items():
        data['Model'].append(model.upper())
        data['Test MAE (%)'].append(results['test_mae'] * 100)
        data['Test RMSE (%)'].append(results['test_rmse'] * 100)
        data['Test MAPE (%)'].append(results['test_mape'])
        data['Best Val MAE (%)'].append(results['best_val_mae'] * 100)
        data['Best Epoch'].append(results['best_epoch'])

    df = pd.DataFrame(data)

    # 添加排名
    df['MAE Rank'] = df['Test MAE (%)'].rank()
    df['RMSE Rank'] = df['Test RMSE (%)'].rank()

    # 保存CSV
    save_dir = 'results/baseline_comparison'
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'baseline_comparison.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')

    # 打印表格
    print("\n对比结果表格:")
    print(df.to_string(index=False))

    # 找出最佳模型
    best_mae_idx = df['Test MAE (%)'].idxmin()
    best_rmse_idx = df['Test RMSE (%)'].idxmin()

    best_mae_model = df.loc[best_mae_idx, 'Model']
    best_mae_value = df.loc[best_mae_idx, 'Test MAE (%)']

    best_rmse_model = df.loc[best_rmse_idx, 'Model']
    best_rmse_value = df.loc[best_rmse_idx, 'Test RMSE (%)']

    print("\n" + "="*70)
    print("最佳模型")
    print("="*70)
    print(f"🏆 MAE最佳:  {best_mae_model} ({best_mae_value:.4f}%)")
    print(f"🏆 RMSE最佳: {best_rmse_model} ({best_rmse_value:.4f}%)")

    # 绘制对比图表
    plot_baseline_comparison(df, save_dir)

    # 生成研究建议
    print("\n" + "="*70)
    print("研究建议")
    print("="*70)
    print(f"\n📊 基准模型性能排名 (按MAE):")
    sorted_df = df.sort_values('Test MAE (%)')
    for i, (idx, row) in enumerate(sorted_df.iterrows(), 1):
        print(f"   {i}. {row['Model']}: {row['Test MAE (%)']:.4f}%")

    print(f"\n💡 建议:")
    print(f"   1. 选择 {best_mae_model} 作为基准模型")
    print(f"      → 测试集MAE: {best_mae_value:.4f}%")
    print(f"\n   2. 下一步: 在 {best_mae_model} 基础上添加物理约束")
    print(f"      → 创建 {best_mae_model}-PINN 模型")
    print(f"      → 对比是否有性能提升")

    print(f"\n   3. 如果 PINN 版本性能提升:")
    print(f"      → 证明物理约束的有效性 ✅")
    print(f"      → 可作为论文贡献点")

    # 保存报告
    save_markdown_report(df, results_summary, save_dir)

    print(f"\n📁 所有结果已保存到: {save_dir}/")
    print("="*70)


def plot_baseline_comparison(df, save_dir):
    """绘制对比图表"""

    # 设置样式
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # 1. 指标对比柱状图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['Test MAE (%)', 'Test RMSE (%)', 'Test MAPE (%)']
    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(df['Model'], df[metric], color=colors[:len(df)], alpha=0.7)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.grid(axis='y', alpha=0.3)

        # 标记最佳
        min_idx = df[metric].idxmin()
        ax.get_children()[min_idx].set_color('#2ca02c')
        ax.get_children()[min_idx].set_alpha(1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 雷达图对比
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    # 归一化指标（越小越好，所以用1-normalize）
    metrics_data = df[['Test MAE (%)', 'Test RMSE (%)', 'Test MAPE (%)']].values
    metrics_norm = 1 - (metrics_data - metrics_data.min(axis=0)) / (metrics_data.max(axis=0) - metrics_data.min(axis=0) + 1e-8)

    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
    angles += angles[:1]

    for i, model in enumerate(df['Model']):
        values = metrics_norm[i].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['MAE', 'RMSE', 'MAPE'])
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('Baseline Models Performance Comparison\n(Larger area = Better)', pad=20)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'radar_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("\n📊 对比图表已生成:")
    print(f"   - {save_dir}/metrics_comparison.png")
    print(f"   - {save_dir}/radar_comparison.png")


def save_markdown_report(df, results_summary, save_dir):
    """保存Markdown格式的报告"""

    report_path = os.path.join(save_dir, 'BASELINE_COMPARISON_REPORT.md')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 基准模型对比实验报告\n\n")
        f.write("## 1. 实验目的\n\n")
        f.write("对比不同基准模型（FNN, CNN, LSTM）在HUST电池数据集上的性能，")
        f.write("为后续Physics-Informed Neural Network (PINN)实验选择最佳基准模型。\n\n")

        f.write("## 2. 实验设置\n\n")
        f.write("- **数据集**: HUST 77组电池数据\n")
        f.write("- **数据划分**: Train/Val/Test = 6:2:2\n")
        f.write("- **评估方式**: 跨电池泛化（测试集为完全未见过的电池）\n\n")

        f.write("## 3. 实验结果\n\n")
        f.write("### 3.1 性能对比表格\n\n")
        f.write(df.to_markdown(index=False, floatfmt='.4f'))
        f.write("\n\n")

        f.write("### 3.2 最佳模型\n\n")
        best_model = df.loc[df['Test MAE (%)'].idxmin(), 'Model']
        best_mae = df.loc[df['Test MAE (%)'].idxmin(), 'Test MAE (%)']

        f.write(f"🏆 **最佳模型**: {best_model}\n\n")
        f.write(f"- **测试集MAE**: {best_mae:.4f}%\n")
        f.write(f"- **测试集RMSE**: {df.loc[df['Test MAE (%)'].idxmin(), 'Test RMSE (%)']:.4f}%\n")
        f.write(f"- **测试集MAPE**: {df.loc[df['Test MAE (%)'].idxmin(), 'Test MAPE (%)']:.4f}%\n\n")

        f.write("### 3.3 性能排名\n\n")
        sorted_df = df.sort_values('Test MAE (%)')
        for i, (idx, row) in enumerate(sorted_df.iterrows(), 1):
            f.write(f"{i}. **{row['Model']}**: MAE = {row['Test MAE (%)']:.4f}%\n")

        f.write("\n## 4. 下一步研究计划\n\n")
        f.write(f"### 阶段1: 基准模型训练 ✅ (已完成)\n\n")
        f.write(f"- 已对比 {len(df)} 个基准模型\n")
        f.write(f"- 最佳模型: {best_model}\n\n")

        f.write(f"### 阶段2: 添加物理约束 (下一步)\n\n")
        f.write(f"1. 在 {best_model} 基础上添加物理约束\n")
        f.write(f"   - 实现 {best_model}-PINN\n")
        f.write(f"   - 加入单调性约束\n\n")

        f.write(f"2. 训练 {best_model}-PINN\n")
        f.write(f"   - 使用相同的数据划分（seed=42）\n")
        f.write(f"   - 相同的训练配置\n\n")

        f.write(f"3. 对比分析\n")
        f.write(f"   - {best_model} (基准) vs {best_model}-PINN\n")
        f.write(f"   - 分析物理约束的作用\n\n")

        f.write("### 阶段3: 论文撰写\n\n")
        f.write("如果PINN版本性能提升，可作为论文贡献点：\n\n")
        f.write("- 提出了针对电池SOH估计的物理约束方法\n")
        f.write("- 实验证明物理约束提升了模型性能\n")
        f.write("- 提高了模型的可解释性\n\n")

        f.write("## 5. 附图\n\n")
        f.write("- [指标对比柱状图](metrics_comparison.png)\n")
        f.write("- [雷达图对比](radar_comparison.png)\n\n")

        f.write("---\n\n")
        f.write("*报告生成时间: 自动生成*\n")

    print(f"\n📄 详细报告已保存: {report_path}")


if __name__ == "__main__":
    """
    基准模型对比实验

    运行此脚本将：
    1. 训练 FNN, CNN, LSTM 三个基准模型
    2. 生成详细的对比报告
    3. 推荐最佳模型用于后续PINN实验
    """

    import numpy as np
    import torch

    # ===== 配置参数 =====
    MODEL_TYPES = ['fnn', 'cnn', 'lstm']  # 基准模型列表
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 42  # 固定种子确保可复现

    # ===== 开始实验 =====
    print(f"\n使用设备: {DEVICE}\n")

    results = train_baseline_models(
        model_types=MODEL_TYPES,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        device=DEVICE,
        seed=SEED
    )

    print("\n" + "="*70)
    print("基准模型对比实验完成！")
    print("="*70)
    print("\n下一步: 根据报告选择最佳模型，添加物理约束构建PINN")
