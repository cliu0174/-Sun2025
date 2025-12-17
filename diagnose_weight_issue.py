"""
诊断单调性权重为什么不生效

检查项：
1. 配置文件是否正确修改
2. 配置是否被正确读取
3. 损失函数是否使用了正确的权重
4. 不同权重下的训练损失是否不同
"""

import json
import torch
from train_cross_battery import train_cross_battery_model

def diagnose_weight_modification():
    """诊断配置修改流程"""
    print("="*70)
    print("诊断1: 配置文件修改测试")
    print("="*70)

    config_path = 'configs/models/cnn_lstm_config.json'

    # 读取原始配置
    with open(config_path, 'r', encoding='utf-8') as f:
        original_config = json.load(f)
    original_weight = original_config.get('physics_constraints', {}).get('monotonic_weight', None)
    print(f"\n原始权重: {original_weight}")

    # 修改为新权重
    test_weight = 0.999
    original_config['physics_constraints']['monotonic_weight'] = test_weight

    # 保存
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(original_config, f, indent=2, ensure_ascii=False)
    print(f"已修改为: {test_weight}")

    # 重新读取验证
    with open(config_path, 'r', encoding='utf-8') as f:
        verify_config = json.load(f)
    verify_weight = verify_config.get('physics_constraints', {}).get('monotonic_weight', None)
    print(f"验证读取: {verify_weight}")

    if verify_weight == test_weight:
        print("✓ 配置文件修改和读取正常")
    else:
        print("✗ 配置文件修改失败！")

    # 恢复原始权重
    original_config['physics_constraints']['monotonic_weight'] = original_weight
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(original_config, f, indent=2, ensure_ascii=False)
    print(f"\n已恢复原始权重: {original_weight}")


def diagnose_training_with_different_weights():
    """诊断不同权重下的训练是否真的不同"""
    print("\n" + "="*70)
    print("诊断2: 训练两次，使用不同权重")
    print("="*70)

    config_path = 'configs/models/cnn_lstm_config.json'

    # 保存原始配置
    with open(config_path, 'r', encoding='utf-8') as f:
        backup_config = json.load(f)

    try:
        results_dict = {}

        for weight in [0.0, 0.5]:
            print(f"\n{'#'*70}")
            print(f"测试权重: {weight}")
            print(f"{'#'*70}")

            # 修改配置
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            config['physics_constraints']['monotonic_weight'] = weight
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            # 等待文件系统同步
            import time
            time.sleep(0.5)

            # 验证修改
            with open(config_path, 'r', encoding='utf-8') as f:
                verify = json.load(f)
            print(f"配置文件中的权重: {verify['physics_constraints']['monotonic_weight']}")

            # 运行训练（只训练5个epoch用于快速测试）
            # 先备份原始epoch数
            original_epochs = config['training']['num_epochs']
            config['training']['num_epochs'] = 5  # 只训练5个epoch
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            wrapper, results, data_dict = train_cross_battery_model(
                model_type='cnn_lstm',
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                seed=42,  # 固定种子确保数据划分相同
                degradation_scenario='scenario2',
                sparse_sampling_interval=5
            )

            # 验证损失函数权重
            if hasattr(wrapper, 'criterion') and hasattr(wrapper.criterion, 'monotonic_weight'):
                actual_weight = wrapper.criterion.monotonic_weight
                print(f"\n训练使用的权重: {actual_weight}")
            else:
                print(f"\n[WARNING] 无法获取损失函数的权重")

            # 记录结果
            results_dict[weight] = {
                'test_rmse': results['test_rmse'],
                'test_mae': results['test_mae'],
                'train_loss': results['train_loss'][-1] if results['train_loss'] else None
            }

            print(f"\nRMSE: {results['test_rmse']:.6f}")
            print(f"MAE: {results['test_mae']:.6f}")
            if results['train_loss']:
                print(f"最终训练损失: {results['train_loss'][-1]:.6f}")

        # 对比结果
        print("\n" + "="*70)
        print("结果对比")
        print("="*70)
        print(f"{'权重':<10} {'RMSE':<15} {'MAE':<15} {'训练损失':<15}")
        print("-"*70)
        for weight, res in results_dict.items():
            print(f"{weight:<10} {res['test_rmse']:<15.6f} {res['test_mae']:<15.6f} {res['train_loss']:<15.6f}")

        # 判断是否有差异
        weights = list(results_dict.keys())
        rmse_diff = abs(results_dict[weights[0]]['test_rmse'] - results_dict[weights[1]]['test_rmse'])
        mae_diff = abs(results_dict[weights[0]]['test_mae'] - results_dict[weights[1]]['test_mae'])

        print(f"\nRMSE 差异: {rmse_diff:.6f}")
        print(f"MAE 差异: {mae_diff:.6f}")

        if rmse_diff < 1e-6 and mae_diff < 1e-6:
            print("\n✗ 结果完全相同，权重可能未生效！")
            print("\n可能的原因：")
            print("1. 配置文件被缓存，修改未生效")
            print("2. 损失函数在其他地方被初始化")
            print("3. 随机种子导致结果相同（概率极低）")
        else:
            print("\n✓ 结果有差异，权重正常生效")

    finally:
        # 恢复原始配置
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(backup_config, f, indent=2, ensure_ascii=False)
        print(f"\n已恢复原始配置")


def quick_check_criterion_initialization():
    """快速检查损失函数初始化位置"""
    print("\n" + "="*70)
    print("诊断3: 检查损失函数初始化流程")
    print("="*70)

    import inspect
    from models import PhysicsConstrainedLoss

    # 查看损失函数的初始化参数
    sig = inspect.signature(PhysicsConstrainedLoss.__init__)
    print(f"\nPhysicsConstrainedLoss 初始化参数:")
    for param_name, param in sig.parameters.items():
        if param_name != 'self':
            default = param.default if param.default != inspect.Parameter.empty else 'Required'
            print(f"  {param_name}: {default}")

    print("\n建议检查 train_cross_battery.py 中：")
    print("  - 搜索 'PhysicsConstrainedLoss(' 找到初始化位置")
    print("  - 确认 monotonic_weight 参数从哪里读取")
    print("  - 检查是否有缓存机制")


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("单调性权重问题诊断工具")
    print("#"*70)

    # 诊断1: 配置修改
    diagnose_weight_modification()

    # 诊断2: 实际训练对比（警告：会运行真实训练，耗时约2-5分钟）
    print("\n\n")
    response = input("是否进行实际训练测试？(需要 2-5 分钟) [y/N]: ")
    if response.lower() == 'y':
        diagnose_training_with_different_weights()
    else:
        print("跳过实际训练测试")

    # 诊断3: 检查初始化
    quick_check_criterion_initialization()

    print("\n" + "="*70)
    print("诊断完成")
    print("="*70)
