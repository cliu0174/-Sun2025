"""
诊断批量训练 vs 单独训练的差异

对比 test_monotonic_weights.py 和 train_cross_battery.py 的训练结果
找出可能导致结果不一致的原因
"""

import json
import torch
import numpy as np
from train_cross_battery import train_cross_battery_model
import time


def compare_training_configurations():
    """对比两种训练方式的配置"""
    print("="*70)
    print("配置对比诊断")
    print("="*70)

    # 配置1: test_monotonic_weights.py 的配置
    config_batch = {
        'model_type': 'cnn_lstm',
        'train_ratio': 0.6,
        'val_ratio': 0.2,
        'test_ratio': 0.2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 999,
        'degradation_scenario': 'scenario2',
        'sparse_sampling_interval': 2,
    }

    # 配置2: train_cross_battery.py 的配置
    config_single = {
        'model_type': 'cnn_lstm',
        'train_ratio': 0.6,
        'val_ratio': 0.2,
        'test_ratio': 0.2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 999,
        'degradation_scenario': 'scenario2',
        'sparse_sampling_interval': 2,
    }

    print("\n配置1 (批量训练):")
    for k, v in config_batch.items():
        print(f"  {k}: {v}")

    print("\n配置2 (单独训练):")
    for k, v in config_single.items():
        print(f"  {k}: {v}")

    print("\n差异:")
    diff_found = False
    for key in config_batch:
        if config_batch[key] != config_single.get(key):
            print(f"  ⚠️  {key}: {config_batch[key]} vs {config_single.get(key)}")
            diff_found = True

    if not diff_found:
        print("  ✓ 训练参数完全一致")

    return config_batch, config_single


def test_same_weight_twice(weight, config):
    """测试相同权重两次训练是否结果一致"""
    print(f"\n{'='*70}")
    print(f"测试: 相同权重 ({weight}) 两次训练的一致性")
    print(f"{'='*70}")

    # 修改配置文件
    config_path = f'configs/models/{config["model_type"]}_config.json'

    with open(config_path, 'r', encoding='utf-8') as f:
        model_config = json.load(f)

    model_config['physics_constraints']['monotonic_weight'] = weight
    model_config['physics_constraints']['enabled'] = True

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)

    time.sleep(0.5)

    results = []

    for run in range(2):
        print(f"\n第 {run+1} 次训练...")

        _, res, _ = train_cross_battery_model(
            model_type=config['model_type'],
            train_ratio=config['train_ratio'],
            val_ratio=config['val_ratio'],
            test_ratio=config['test_ratio'],
            device=config['device'],
            seed=config['seed'],  # ← 相同种子
            degradation_scenario=config['degradation_scenario'],
            sparse_sampling_interval=config['sparse_sampling_interval']
        )

        results.append({
            'rmse': res['test_rmse'],
            'mae': res['test_mae'],
            'r2': res['test_r2']
        })

        print(f"  RMSE: {res['test_rmse']:.6f}")
        print(f"  MAE:  {res['test_mae']:.6f}")
        print(f"  R²:   {res['test_r2']:.6f}")

    # 对比两次结果
    print(f"\n{'='*70}")
    print("结果对比:")
    print(f"{'='*70}")
    print(f"第1次 RMSE: {results[0]['rmse']:.6f}")
    print(f"第2次 RMSE: {results[1]['rmse']:.6f}")
    print(f"差异:       {abs(results[0]['rmse'] - results[1]['rmse']):.6f}")

    if abs(results[0]['rmse'] - results[1]['rmse']) < 1e-6:
        print("\n✓ 两次训练结果完全一致 (seed 生效)")
    elif abs(results[0]['rmse'] - results[1]['rmse']) < 1e-4:
        print("\n⚠️  两次训练结果略有差异 (可能是随机性或数值误差)")
    else:
        print("\n✗ 两次训练结果差异较大！(seed 可能未生效)")
        print("\n可能原因:")
        print("  1. GPU 非确定性操作 (CUDA)")
        print("  2. 多线程数据加载")
        print("  3. 代码中存在未固定的随机操作")

    return results


def check_config_file_before_after(weight):
    """检查配置文件修改前后的状态"""
    print(f"\n{'='*70}")
    print("检查配置文件修改机制")
    print(f"{'='*70}")

    config_path = 'configs/models/cnn_lstm_config.json'

    # 读取原始配置
    with open(config_path, 'r', encoding='utf-8') as f:
        before = json.load(f)

    print(f"\n修改前:")
    print(f"  monotonic_weight: {before['physics_constraints']['monotonic_weight']}")
    print(f"  enabled: {before['physics_constraints']['enabled']}")

    # 修改配置
    before['physics_constraints']['monotonic_weight'] = weight
    before['physics_constraints']['enabled'] = True

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(before, f, indent=2, ensure_ascii=False)

    time.sleep(0.5)

    # 读取修改后的配置
    with open(config_path, 'r', encoding='utf-8') as f:
        after = json.load(f)

    print(f"\n修改后:")
    print(f"  monotonic_weight: {after['physics_constraints']['monotonic_weight']}")
    print(f"  enabled: {after['physics_constraints']['enabled']}")

    if after['physics_constraints']['monotonic_weight'] == weight:
        print("\n✓ 配置文件修改成功")
    else:
        print(f"\n✗ 配置文件修改失败！预期 {weight}，实际 {after['physics_constraints']['monotonic_weight']}")


def check_random_state():
    """检查随机数生成器状态"""
    print(f"\n{'='*70}")
    print("检查随机数生成器")
    print(f"{'='*70}")

    import random

    # 固定种子
    seed = 999
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 生成一些随机数
    print(f"\n种子 {seed} 下的随机数序列:")
    print(f"  random.random(): {random.random():.10f}")
    print(f"  np.random.rand(): {np.random.rand():.10f}")
    print(f"  torch.rand(1): {torch.rand(1).item():.10f}")

    # 再次固定相同种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 应该生成相同的随机数
    print(f"\n重新设置种子后:")
    print(f"  random.random(): {random.random():.10f}")
    print(f"  np.random.rand(): {np.random.rand():.10f}")
    print(f"  torch.rand(1): {torch.rand(1).item():.10f}")

    print("\n如果两次输出相同，说明 seed 机制正常")


def check_data_loading_consistency():
    """检查数据加载的一致性"""
    print(f"\n{'='*70}")
    print("检查数据加载一致性")
    print(f"{'='*70}")

    from utils.data_augmentation import apply_sparse_sampling_by_battery
    import random

    # 模拟数据
    features = np.random.randn(100, 11)
    targets = np.random.rand(100)
    battery_ids = np.array(['B1']*50 + ['B2']*50)

    results = []

    for run in range(2):
        # 固定种子
        random.seed(999)
        np.random.seed(999)

        # 稀疏采样
        sampled_f, sampled_t, sampled_b = apply_sparse_sampling_by_battery(
            features.copy(), targets.copy(), battery_ids.copy(),
            sampling_interval=2,
            offset=0,
            verbose=False
        )

        results.append({
            'n_samples': len(sampled_t),
            'first_5_targets': sampled_t[:5].tolist()
        })

        print(f"\n第 {run+1} 次采样:")
        print(f"  样本数: {len(sampled_t)}")
        print(f"  前5个目标: {sampled_t[:5]}")

    if results[0]['n_samples'] == results[1]['n_samples']:
        print("\n✓ 稀疏采样样本数一致")
    else:
        print(f"\n✗ 稀疏采样样本数不一致！{results[0]['n_samples']} vs {results[1]['n_samples']}")

    if np.allclose(results[0]['first_5_targets'], results[1]['first_5_targets']):
        print("✓ 稀疏采样结果一致")
    else:
        print("✗ 稀疏采样结果不一致！")


def main():
    """主函数"""
    print("\n" + "#"*70)
    print("批量训练 vs 单独训练 差异诊断")
    print("#"*70)

    # 1. 对比配置
    config_batch, config_single = compare_training_configurations()

    # 2. 检查配置文件修改机制
    check_config_file_before_after(0.3)

    # 3. 检查随机数生成器
    check_random_state()

    # 4. 检查数据加载一致性
    check_data_loading_consistency()

    # 5. 测试相同权重两次训练
    print("\n\n是否进行实际训练测试？(需要约20分钟)")
    response = input("输入 'y' 继续，或任意其他键跳过: ")

    if response.lower() == 'y':
        test_same_weight_twice(0.3, config_batch)
    else:
        print("\n跳过实际训练测试")

    print("\n" + "="*70)
    print("诊断完成")
    print("="*70)
    print("\n可能的差异原因:")
    print("1. 随机种子设置不同")
    print("2. 配置文件中其他参数不同 (如 batch_size, learning_rate)")
    print("3. 数据加载顺序不同 (shuffle)")
    print("4. GPU 非确定性操作")
    print("5. Early stopping 触发时机不同")
    print("6. 配置文件被其他进程修改")
    print("\n建议:")
    print("- 确保两种训练方式使用完全相同的配置文件")
    print("- 检查是否有多个脚本同时修改配置文件")
    print("- 使用 test_adaptive_weights_scenario3.py (每次训练前验证配置)")


if __name__ == "__main__":
    main()
