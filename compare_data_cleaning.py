"""
对比数据清洗效果 - 训练对比实验

对比以下场景:
1. 原始数据（不清洗）
2. 3-Sigma 清洗后的数据
"""

import json
import numpy as np
from pathlib import Path


def run_comparison_experiment():
    """
    运行数据清洗对比实验
    """
    print("="*70)
    print("数据清洗效果对比实验")
    print("="*70)

    # 配置
    model_type = 'gru'
    num_epochs_test = 50  # 快速测试用较少轮数
    device = 'cuda'

    results = {}

    # ===== 实验1: 原始数据（不清洗） =====
    print("\n" + "="*70)
    print("实验 1: 使用原始数据（不清洗）")
    print("="*70)

    from train_cross_battery import train_cross_battery_model

    # 临时修改配置文件减少训练轮数
    config_path = f'configs/models/{model_type}_config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    original_epochs = config['training']['num_epochs']
    config['training']['num_epochs'] = num_epochs_test

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    try:
        # 训练原始数据模型
        wrapper_original, results_original, data_dict_original = train_cross_battery_model(
            model_type=model_type,
            device=device
        )

        results['original'] = {
            'test_mae': results_original['test_mae'],
            'test_rmse': results_original['test_rmse'],
            'test_mape': results_original['test_mape'],
            'train_samples': len(data_dict_original['train_features']),
            'val_samples': len(data_dict_original['val_features']),
            'test_samples': len(data_dict_original['test_features'])
        }

        print("\n[实验1完成] 原始数据结果:")
        print(f"  Test MAE:  {results['original']['test_mae']*100:.4f}%")
        print(f"  Test RMSE: {results['original']['test_rmse']*100:.4f}%")
        print(f"  Test MAPE: {results['original']['test_mape']:.4f}%")
        print(f"  训练样本数: {results['original']['train_samples']}")

    except Exception as e:
        print(f"\n[实验1失败] 错误: {e}")
        results['original'] = None

    # ===== 实验2: 3-Sigma 清洗数据 =====
    print("\n" + "="*70)
    print("实验 2: 使用 3-Sigma 清洗数据")
    print("="*70)

    # TODO: 这里需要修改 data_loader_hust.py 以支持数据清洗
    # 暂时提示用户手动修改
    print("\n[注意] 需要在数据加载器中启用 3-Sigma 清洗")
    print("请修改 data_loaders/data_loader_hust.py 添加清洗功能")

    # 恢复原始配置
    config['training']['num_epochs'] = original_epochs
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ===== 结果对比 =====
    print("\n" + "="*70)
    print("对比结果总结")
    print("="*70)

    if results.get('original'):
        print("\n原始数据（不清洗）:")
        print(f"  Test MAE:  {results['original']['test_mae']*100:.4f}%")
        print(f"  Test RMSE: {results['original']['test_rmse']*100:.4f}%")
        print(f"  Test MAPE: {results['original']['test_mape']:.4f}%")

    if results.get('cleaned'):
        print("\n清洗数据（3-Sigma）:")
        print(f"  Test MAE:  {results['cleaned']['test_mae']*100:.4f}%")
        print(f"  Test RMSE: {results['cleaned']['test_rmse']*100:.4f}%")
        print(f"  Test MAPE: {results['cleaned']['test_mape']:.4f}%")

        # 计算改进
        mae_improvement = (results['original']['test_mae'] - results['cleaned']['test_mae']) / results['original']['test_mae'] * 100
        rmse_improvement = (results['original']['test_rmse'] - results['cleaned']['test_rmse']) / results['original']['test_rmse'] * 100

        print("\n改进:")
        print(f"  MAE:  {mae_improvement:+.2f}%")
        print(f"  RMSE: {rmse_improvement:+.2f}%")

    print("="*70)

    return results


if __name__ == "__main__":
    results = run_comparison_experiment()
