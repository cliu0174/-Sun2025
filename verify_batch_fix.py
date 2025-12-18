"""
快速验证批量测试一致性修复是否生效

运行两次相同配置的训练，检查结果是否一致
"""

import json
import torch
from train_cross_battery import train_cross_battery_model


def test_batch_consistency():
    """测试批量训练的一致性"""
    print("="*70)
    print("批量测试一致性验证")
    print("="*70)

    # 测试配置
    test_weight = 0.3
    config = {
        'model_type': 'cnn_lstm',
        'train_ratio': 0.6,
        'val_ratio': 0.2,
        'test_ratio': 0.2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 999,
        'degradation_scenario': 'scenario2',
        'sparse_sampling_interval': 2,
    }

    print(f"\n测试配置:")
    print(f"  模型: {config['model_type']}")
    print(f"  场景: {config['degradation_scenario']}")
    print(f"  稀疏间隔: {config['sparse_sampling_interval']}")
    print(f"  单调性权重: {test_weight}")
    print(f"  随机种子: {config['seed']} (固定)")

    # 修改配置文件
    print(f"\n{'='*70}")
    print("第1步: 修改配置文件")
    print(f"{'='*70}")

    config_path = f'configs/models/{config["model_type"]}_config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        model_config = json.load(f)

    # 显式设置所有物理约束参数（应用修复）
    model_config['physics_constraints']['enabled'] = True
    model_config['physics_constraints']['monotonic_weight'] = test_weight
    model_config['physics_constraints']['monotonic_tolerance'] = 0.01
    model_config['physics_constraints']['boundary_weight'] = 0.0
    model_config['physics_constraints']['smoothness_weight'] = 0.0
    model_config['physics_constraints']['curvature_weight'] = 0.0

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)

    print(f"  ✓ 已设置 monotonic_weight = {test_weight}")
    print(f"  ✓ 已设置 monotonic_tolerance = 0.01")
    print(f"  ✓ 已设置 boundary_weight = 0.0")
    print(f"  ✓ 已设置 smoothness_weight = 0.0")
    print(f"  ✓ 已设置 curvature_weight = 0.0")

    # 验证配置
    print(f"\n{'='*70}")
    print("第2步: 验证配置文件")
    print(f"{'='*70}")

    import time
    time.sleep(0.5)

    with open(config_path, 'r', encoding='utf-8') as f:
        verify_config = json.load(f)

    physics = verify_config['physics_constraints']
    print(f"  enabled: {physics['enabled']}")
    print(f"  monotonic_weight: {physics['monotonic_weight']}")
    print(f"  monotonic_tolerance: {physics['monotonic_tolerance']}")
    print(f"  boundary_weight: {physics['boundary_weight']}")
    print(f"  smoothness_weight: {physics['smoothness_weight']}")
    print(f"  curvature_weight: {physics['curvature_weight']}")

    # 运行两次训练
    results = []

    for run in range(2):
        print(f"\n{'='*70}")
        print(f"第{run+1}次训练")
        print(f"{'='*70}")

        _, res, _ = train_cross_battery_model(
            model_type=config['model_type'],
            train_ratio=config['train_ratio'],
            val_ratio=config['val_ratio'],
            test_ratio=config['test_ratio'],
            device=config['device'],
            seed=config['seed'],  # 相同种子
            degradation_scenario=config['degradation_scenario'],
            sparse_sampling_interval=config['sparse_sampling_interval']
        )

        results.append({
            'rmse': res['test_rmse'],
            'mae': res['test_mae'],
            'r2': res['test_r2']
        })

        print(f"\n  结果:")
        print(f"    RMSE: {res['test_rmse']:.6f}")
        print(f"    MAE:  {res['test_mae']:.6f}")
        print(f"    R²:   {res['test_r2']:.6f}")

    # 对比结果
    print(f"\n{'='*70}")
    print("第3步: 对比两次训练结果")
    print(f"{'='*70}")

    rmse_diff = abs(results[0]['rmse'] - results[1]['rmse'])
    mae_diff = abs(results[0]['mae'] - results[1]['mae'])
    r2_diff = abs(results[0]['r2'] - results[1]['r2'])

    print(f"\n  第1次 RMSE: {results[0]['rmse']:.6f}")
    print(f"  第2次 RMSE: {results[1]['rmse']:.6f}")
    print(f"  差异:       {rmse_diff:.8f}")

    print(f"\n  第1次 MAE:  {results[0]['mae']:.6f}")
    print(f"  第2次 MAE:  {results[1]['mae']:.6f}")
    print(f"  差异:       {mae_diff:.8f}")

    print(f"\n  第1次 R²:   {results[0]['r2']:.6f}")
    print(f"  第2次 R²:   {results[1]['r2']:.6f}")
    print(f"  差异:       {r2_diff:.8f}")

    # 判断是否一致
    print(f"\n{'='*70}")
    print("结论")
    print(f"{'='*70}")

    if rmse_diff < 1e-6 and mae_diff < 1e-6 and r2_diff < 1e-6:
        print("\n✅ 批量测试一致性修复成功！")
        print("   两次训练结果完全一致（差异 < 1e-6）")
        print("   说明配置文件参数设置正确，无残留值影响")
    elif rmse_diff < 1e-4 and mae_diff < 1e-4 and r2_diff < 1e-4:
        print("\n⚠️  两次训练结果基本一致（差异 < 1e-4）")
        print("   可能存在轻微的数值误差或GPU非确定性操作")
        print("   这在可接受范围内")
    else:
        print("\n❌ 批量测试一致性仍有问题！")
        print("   两次训练结果差异较大（差异 > 1e-4）")
        print("\n可能原因:")
        print("  1. GPU 非确定性操作 (CUDA)")
        print("  2. 多线程数据加载")
        print("  3. 配置文件被其他进程修改")
        print("  4. 随机数生成器未正确固定")
        print("\n建议:")
        print("  - 检查是否有其他脚本在同时运行")
        print("  - 尝试使用 CPU 进行测试 (device='cpu')")
        print("  - 运行 diagnose_batch_vs_single.py 进行详细诊断")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    test_batch_consistency()
