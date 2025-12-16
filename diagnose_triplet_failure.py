"""
快速诊断 Triplet 二阶曲率约束为什么不起作用

检查项：
1. Triplet 采样是否生成了有效样本
2. Split threshold 是否合理
3. Curvature loss 是否真的为 0
4. 不同场景下的表现
"""

import json
import numpy as np
import torch
from data_loaders import load_single_hust_battery
from data_loaders.data_loader_hust import apply_windowing_with_metadata, HUSTBatteryDatasetWithMetadata

def diagnose_triplet_sampling():
    """诊断三元组采样是否正常"""

    print("="*70)
    print("诊断 Triplet 三元组采样")
    print("="*70)

    # 加载配置
    config_path = 'configs/models/cnn_lstm_config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 加载一个电池数据
    battery_file = 'data/HUST data/10-1.csv'
    import pandas as pd
    data = pd.read_csv(battery_file)
    battery_name = '10-1'

    print(f"\n[1] 加载电池 {battery_name}")
    print(f"    原始循环数: {data['cycle'].max()}")
    print(f"    原始样本数: {len(data)}")

    # 提取特征和目标
    feature_names = config['features']
    X = data[feature_names].values
    y = data['SOH'].values
    cycles = data['cycle'].values

    # 应用滑动窗口
    sequence_length = config['sequence_length']
    features_seq, targets, battery_ids, cycle_indices = apply_windowing_with_metadata(
        X, y, cycles, battery_name, sequence_length
    )

    print(f"\n[2] 滑动窗口后")
    print(f"    样本数: {len(targets)}")
    print(f"    Cycle 范围: [{cycle_indices.min()}, {cycle_indices.max()}]")

    # 测试不同场景
    scenarios = [
        ('干净数据', features_seq, targets, battery_ids, cycle_indices),
    ]

    # 场景 2: 稀疏采样
    print(f"\n[3] 应用稀疏采样 (interval=5)")
    from utils.data_augmentation import sparse_sampling
    features_sparse, targets_sparse, battery_ids_sparse = sparse_sampling(
        features_seq, targets, battery_ids, sampling_interval=5, offset=0, verbose=False
    )
    # 计算稀疏采样后的 cycle_indices
    keep_indices = np.arange(0, len(cycle_indices), 5)
    cycle_indices_sparse = cycle_indices[keep_indices]

    print(f"    稀疏采样后样本数: {len(targets_sparse)}")
    print(f"    Cycle 范围: [{cycle_indices_sparse.min()}, {cycle_indices_sparse.max()}]")

    scenarios.append(('稀疏采样', features_sparse, targets_sparse, battery_ids_sparse, cycle_indices_sparse))

    # 对每个场景测试不同的 split_threshold
    thresholds = [50, 100, 150, 200, 300]

    print("\n" + "="*70)
    print("测试不同 Split Threshold 下的三元组采样")
    print("="*70)

    for scenario_name, feats, targs, bids, cycs in scenarios:
        print(f"\n{'='*70}")
        print(f"场景: {scenario_name}")
        print(f"{'='*70}")

        for threshold in thresholds:
            # 创建 Triplet Dataset
            triplet_config = config['physics_constraints']['triplet_sampling']

            dataset = HUSTBatteryDatasetWithMetadata(
                feats, targs, bids, cycs,
                sampling_mode='triplet',
                triplet_config={
                    'enabled': True,
                    'split_threshold': threshold,
                    'step_k': 1
                }
            )

            # 统计信息
            n_samples = len(dataset)
            if n_samples > 0:
                # 计算有多少样本的 cycle >= threshold
                n_above_threshold = np.sum(cycs >= threshold)
                ratio = n_above_threshold / len(cycs) * 100
            else:
                n_above_threshold = 0
                ratio = 0

            print(f"\n  Split Threshold = {threshold}")
            print(f"    三元组数量: {n_samples}")
            print(f"    Cycle >= {threshold} 的样本数: {n_above_threshold}/{len(cycs)} ({ratio:.1f}%)")

            if n_samples == 0:
                print(f"    [FAIL] 无有效三元组！")
            elif ratio < 10:
                print(f"    [WARN] 激活样本太少 (<10%)，曲率损失几乎无效")
            else:
                print(f"    [OK] 有足够的样本用于曲率约束")

def test_curvature_computation():
    """测试曲率计算是否正确"""

    print("\n" + "="*70)
    print("测试曲率计算")
    print("="*70)

    # 模拟三种衰减曲线
    test_cases = [
        ("线性衰减（理想）", np.array([1.0, 0.95, 0.90, 0.85, 0.80])),
        ("带锯齿波动", np.array([1.0, 0.93, 0.96, 0.82, 0.85])),
        ("平滑但非线性", np.array([1.0, 0.96, 0.91, 0.85, 0.78])),
    ]

    for name, y in test_cases:
        # 计算二阶曲率
        curvatures = []
        for i in range(len(y) - 2):
            curv = y[i+2] - 2*y[i+1] + y[i]
            curvatures.append(curv)

        # 曲率损失
        curv_loss = np.mean([c**2 for c in curvatures])

        print(f"\n  {name}")
        print(f"    SOH 序列: {y}")
        print(f"    曲率序列: {[f'{c:.4f}' for c in curvatures]}")
        print(f"    曲率损失: {curv_loss:.6f}")

        if curv_loss < 1e-4:
            print(f"    [OK] 曲线平滑，曲率接近 0")
        else:
            print(f"    [WARN] 曲线不平滑，需要曲率约束")

def check_real_predictions():
    """检查真实预测是否有锯齿"""

    print("\n" + "="*70)
    print("检查真实预测曲线是否有锯齿")
    print("="*70)

    import os
    import pickle

    results_path = 'results/cross_battery/cnn_lstm/results.pkl'

    if not os.path.exists(results_path):
        print(f"\n  [SKIP] 未找到结果文件: {results_path}")
        print(f"  请先运行训练生成预测结果")
        return

    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    predictions = results.get('test_predictions', None)
    targets = results.get('test_targets', None)
    battery_ids = results.get('test_battery_ids', None)

    if predictions is None:
        print(f"\n  [SKIP] 结果文件中没有预测数据")
        return

    print(f"\n  加载预测结果: {len(predictions)} 个样本")

    # 找一个电池，计算其曲率
    if battery_ids is not None:
        unique_batteries = sorted(set(battery_ids))[:3]  # 取前3个电池

        for battery_id in unique_batteries:
            indices = [i for i, bid in enumerate(battery_ids) if bid == battery_id]

            if len(indices) < 10:
                continue

            battery_preds = np.array([predictions[i] for i in indices[:20]])  # 取前20个点

            # 计算曲率
            curvatures = []
            for i in range(len(battery_preds) - 2):
                curv = battery_preds[i+2] - 2*battery_preds[i+1] + battery_preds[i]
                curvatures.append(curv)

            curv_loss = np.mean([c**2 for c in curvatures])
            max_curv = np.max(np.abs(curvatures))

            print(f"\n  电池 {battery_id} (前20点)")
            print(f"    预测序列: {battery_preds[:10]}")
            print(f"    平均曲率: {np.mean(np.abs(curvatures)):.6f}")
            print(f"    最大曲率: {max_curv:.6f}")
            print(f"    曲率损失: {curv_loss:.6f}")

            if max_curv < 0.01:
                print(f"    [结论] 预测曲线已经很平滑，不需要曲率约束")
            else:
                print(f"    [结论] 预测曲线有明显波动，曲率约束有潜在价值")

if __name__ == "__main__":
    # 1. 诊断采样
    diagnose_triplet_sampling()

    # 2. 测试曲率计算
    test_curvature_computation()

    # 3. 检查真实预测
    check_real_predictions()

    print("\n" + "="*70)
    print("诊断完成")
    print("="*70)
    print("\n根据诊断结果：")
    print("1. 如果'三元组数量=0' → split_threshold 太高，需要降低")
    print("2. 如果'激活样本太少' → 物理约束覆盖范围不足")
    print("3. 如果'预测曲线已经很平滑' → Triplet 本质上不需要（模型已经学到了）")
    print("4. 如果'预测曲线有明显波动' → Triplet 应该有用，检查权重设置")
