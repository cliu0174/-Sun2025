"""
数据增强训练示例: 验证物理约束在噪声数据上的鲁棒性

这个脚本展示了如何使用数据增强功能来验证物理约束的作用。

实验设计:
1. 干净数据 + 无物理约束 (基线)
2. 干净数据 + 有物理约束 (物理约束的作用)
3. 噪声数据 + 无物理约束 (噪声数据基线)
4. 噪声数据 + 有物理约束 (物理约束在噪声数据上的作用)
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
from train_cross_battery import train_cross_battery_model


def experiment_1_clean_no_physics():
    """实验1: 干净数据 + 无物理约束 (基线)"""
    print("\n" + "="*80)
    print("实验1: 干净数据 + 无物理约束 (基线)")
    print("="*80)

    # 注意: 需要在配置文件中关闭物理约束
    # configs/models/cnn_lstm_config.json: "physics_constraints": {"enabled": false}

    wrapper, results, data_dict = train_cross_battery_model(
        model_type='cnn_lstm',
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42,
        add_noise=False,           # 干净数据
        noise_level='medium',
        apply_cleaning=False,
        color_by_battery=True,
        highlight_anomalies=False
    )

    print(f"\n[实验1结果] 测试集MAE: {results['test_mae']*100:.4f}%")
    return results


def experiment_2_clean_with_physics():
    """实验2: 干净数据 + 有物理约束"""
    print("\n" + "="*80)
    print("实验2: 干净数据 + 有物理约束")
    print("="*80)

    # 注意: 需要在配置文件中启用物理约束
    # configs/models/cnn_lstm_config.json: "physics_constraints": {"enabled": true}

    wrapper, results, data_dict = train_cross_battery_model(
        model_type='cnn_lstm',
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42,
        add_noise=False,           # 干净数据
        noise_level='medium',
        apply_cleaning=False,
        color_by_battery=True,
        highlight_anomalies=False
    )

    print(f"\n[实验2结果] 测试集MAE: {results['test_mae']*100:.4f}%")
    return results


def experiment_3_noisy_no_physics():
    """实验3: 噪声数据 + 无物理约束"""
    print("\n" + "="*80)
    print("实验3: 噪声数据 + 无物理约束")
    print("="*80)

    # 注意: 需要在配置文件中关闭物理约束

    wrapper, results, data_dict = train_cross_battery_model(
        model_type='cnn_lstm',
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42,
        add_noise=True,            # 噪声数据
        noise_level='medium',      # 中度噪声
        noise_seed=42,
        apply_cleaning=False,
        color_by_battery=True,
        highlight_anomalies=False
    )

    print(f"\n[实验3结果] 测试集MAE: {results['test_mae']*100:.4f}%")
    return results


def experiment_4_noisy_with_physics():
    """实验4: 噪声数据 + 有物理约束"""
    print("\n" + "="*80)
    print("实验4: 噪声数据 + 有物理约束")
    print("="*80)

    # 注意: 需要在配置文件中启用物理约束

    wrapper, results, data_dict = train_cross_battery_model(
        model_type='cnn_lstm',
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42,
        add_noise=True,            # 噪声数据
        noise_level='medium',      # 中度噪声
        noise_seed=42,
        apply_cleaning=False,
        color_by_battery=True,
        highlight_anomalies=False
    )

    print(f"\n[实验4结果] 测试集MAE: {results['test_mae']*100:.4f}%")
    return results


def compare_results(exp1, exp2, exp3, exp4):
    """对比实验结果"""
    print("\n" + "="*80)
    print("实验结果对比")
    print("="*80)

    mae1 = exp1['test_mae'] * 100
    mae2 = exp2['test_mae'] * 100
    mae3 = exp3['test_mae'] * 100
    mae4 = exp4['test_mae'] * 100

    print(f"\n干净数据:")
    print(f"  实验1 (无物理约束): {mae1:.4f}%")
    print(f"  实验2 (有物理约束): {mae2:.4f}%")
    print(f"  改善: {mae1 - mae2:.4f}%")

    print(f"\n噪声数据:")
    print(f"  实验3 (无物理约束): {mae3:.4f}%")
    print(f"  实验4 (有物理约束): {mae4:.4f}%")
    print(f"  改善: {mae3 - mae4:.4f}%")

    clean_improvement = mae1 - mae2
    noisy_improvement = mae3 - mae4

    print("\n" + "="*80)
    print("关键发现:")
    print("="*80)
    print(f"物理约束在干净数据上的改善: {clean_improvement:.4f}%")
    print(f"物理约束在噪声数据上的改善: {noisy_improvement:.4f}%")

    if noisy_improvement > clean_improvement:
        print(f"\n结论: 物理约束在噪声环境下更有价值!")
        print(f"       (噪声改善 {noisy_improvement:.4f}% > 干净改善 {clean_improvement:.4f}%)")
    else:
        print(f"\n结论: 物理约束在干净数据上更有效")
        print(f"       (干净改善 {clean_improvement:.4f}% > 噪声改善 {noisy_improvement:.4f}%)")

    print("="*80)


if __name__ == "__main__":
    """
    运行完整的对比实验

    注意:
    1. 需要手动切换配置文件中的 physics_constraints.enabled
    2. 每个实验约需30-60分钟 (取决于硬件)
    3. 可以单独运行某个实验函数
    """

    # 选择要运行的实验
    RUN_ALL = False  # 设为 True 运行全部4个实验

    if RUN_ALL:
        # 运行全部实验 (需要手动切换物理约束配置)
        print("警告: 需要在实验1/3和实验2/4之间手动切换physics_constraints配置!")
        input("按Enter继续...")

        exp1 = experiment_1_clean_no_physics()
        exp2 = experiment_2_clean_with_physics()
        exp3 = experiment_3_noisy_no_physics()
        exp4 = experiment_4_noisy_with_physics()

        compare_results(exp1, exp2, exp3, exp4)
    else:
        # 单独运行某个实验 (修改这里选择实验)
        print("单独实验模式 (修改代码选择实验)")
        print("\n可选实验:")
        print("  experiment_1_clean_no_physics()     - 干净数据 + 无物理约束")
        print("  experiment_2_clean_with_physics()   - 干净数据 + 有物理约束")
        print("  experiment_3_noisy_no_physics()     - 噪声数据 + 无物理约束")
        print("  experiment_4_noisy_with_physics()   - 噪声数据 + 有物理约束")

        # 运行实验3作为示例
        results = experiment_3_noisy_no_physics()
