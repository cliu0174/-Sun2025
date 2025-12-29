"""
XGBoost Baseline Training Script for HUST Dataset

使用与其他模型相同的数据处理流程,但采用XGBoost专用的训练方式。
"""

import os
import numpy as np
import torch
from models.model_factory import ModelFactory, ConfigLoader
from train_cross_battery import (
    load_all_batteries,
    split_batteries,
    prepare_cross_battery_data,
    create_dataloaders,
    set_seed
)
import time


def train_xgboost_baseline(
    model_type='xgboost_simple',  # 'xgboost_simple' or 'xgboost_enhanced'
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    device='cpu',  # XGBoost uses CPU by default
    seed=42,
    apply_cleaning=False,
    degradation_scenario='none',
    noise_level='medium',
    sparse_sampling_interval=None,
    random_missing_rate=None,
    cycle_drop_rate=None,
    cycle_drop_num_gaps=None
):
    """
    训练XGBoost baseline模型。

    Args:
        model_type: 'xgboost_simple' or 'xgboost_enhanced'
        (其他参数与train_cross_battery_model相同)

    Returns:
        wrapper: 模型包装器
        results: 评估结果
        data_dict: 数据字典
    """
    set_seed(seed)

    print("\n" + "="*70)
    print(f"XGBoost Baseline Training: {model_type.upper()}")
    print(f"Data Split: Train/Val/Test = {train_ratio*100:.0f}%/{val_ratio*100:.0f}%/{test_ratio*100:.0f}%")
    if degradation_scenario != 'none':
        print(f"Degradation: {degradation_scenario}")
    print("="*70)

    # 1. 加载所有电池数据
    battery_names, all_data = load_all_batteries(apply_cleaning=apply_cleaning)

    # 2. 划分数据集
    train_batteries, val_batteries, test_batteries = split_batteries(
        battery_names,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    # 3. 准备数据
    data_dict = prepare_cross_battery_data(
        all_data, train_batteries, val_batteries, test_batteries,
        degradation_scenario=degradation_scenario,
        noise_level=noise_level,
        sparse_sampling_interval=sparse_sampling_interval,
        random_missing_rate=random_missing_rate,
        cycle_drop_rate=cycle_drop_rate,
        cycle_drop_num_gaps=cycle_drop_num_gaps,
        seed=seed
    )

    # 4. 加载配置
    config = ConfigLoader.load_model_config(model_type)

    print("\n" + "="*70)
    print("Model Configuration")
    print("="*70)
    print(f"Model: {model_type}")
    print(f"Architecture: {config['architecture']}")
    print("="*70)

    # 5. 创建数据加载器
    window_size = config['data'].get('window_size', 40)

    train_loader, val_loader, test_loader, test_battery_ids = create_dataloaders(
        data_dict,
        batch_size=config['training']['batch_size'],
        window_size=window_size,
        seq2seq=False,
        use_physics=False,
        siamese_mode=False,
        triplet_mode=False
    )

    # 6. 创建XGBoost模型
    print("\nCreating XGBoost model...")
    model = ModelFactory.create_model(
        model_type=model_type,
        input_size=data_dict['n_features']
    )

    print(f"Model created: {model.__class__.__name__}")
    if model_type == 'xgboost_simple':
        print(f"  Flattened feature dim: {model.feature_dim}")
    else:
        print(f"  Engineered feature dim: {model.feature_dim}")
        print(f"  N lags: {model.n_lags}")
        print(f"  Rolling windows: {model.rolling_windows}")

    # 7. 准备训练数据 (XGBoost需要完整的NumPy数组)
    print("\nPreparing training data...")

    X_train_list = []
    y_train_list = []
    for batch in train_loader:
        if isinstance(batch, dict):
            features = batch['window'].cpu().numpy()
            targets = batch['target_soh'].cpu().numpy().flatten()
        else:
            features, targets = batch
            features = features.cpu().numpy()
            targets = targets.cpu().numpy().flatten()
        X_train_list.append(features)
        y_train_list.append(targets)

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)

    # 准备验证数据
    X_val_list = []
    y_val_list = []
    for batch in val_loader:
        if isinstance(batch, dict):
            features = batch['window'].cpu().numpy()
            targets = batch['target_soh'].cpu().numpy().flatten()
        else:
            features, targets = batch
            features = features.cpu().numpy()
            targets = targets.cpu().numpy().flatten()
        X_val_list.append(features)
        y_val_list.append(targets)

    X_val = np.vstack(X_val_list)
    y_val = np.concatenate(y_val_list)

    print(f"Train set: {X_train.shape}, Val set: {X_val.shape}")

    # 8. 训练XGBoost模型
    print("\n" + "="*70)
    print("Training XGBoost Model")
    print("="*70)

    start_time = time.time()
    model.fit(X_train, y_train, X_val, y_val, verbose=True)
    training_time = time.time() - start_time

    print(f"\n[OK] XGBoost training completed! Time: {training_time:.2f}s")

    # 9. 测试集评估
    print("\n" + "="*70)
    print("Evaluating on Test Set")
    print("="*70)

    model.eval()  # 虽然XGBoost没有eval模式,但保持接口一致

    all_predictions = []
    all_targets = []
    all_battery_ids_from_batch = []

    for batch in test_loader:
        if isinstance(batch, dict):
            features = batch['window']
            targets = batch['target_soh'].cpu().numpy().flatten()
            battery_ids = batch['battery_id']
        else:
            features, targets = batch
            targets = targets.cpu().numpy().flatten()
            battery_ids = None

        # XGBoost预测
        predictions = model(features)  # 返回torch.Tensor
        predictions_np = predictions.cpu().numpy().flatten()

        all_predictions.append(predictions_np)
        all_targets.append(targets)

        if battery_ids is not None:
            all_battery_ids_from_batch.extend(battery_ids)

    # 合并结果
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    # 合并电池ID
    if len(all_battery_ids_from_batch) > 0:
        final_battery_ids = all_battery_ids_from_batch
    elif test_battery_ids is not None:
        final_battery_ids = test_battery_ids
    else:
        final_battery_ids = None

    # 计算指标
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    # 计算MAPE
    mask = targets > 0
    mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) if mask.any() else 0

    print(f"\nTest Results:")
    print(f"  RMSE: {rmse*100:.4f}%")
    print(f"  MAE:  {mae*100:.4f}%")
    print(f"  MAPE: {mape*100:.4f}%")
    print(f"  R2:   {r2:.6f}")

    # 10. 保存结果
    results = {
        'test_rmse': rmse,
        'test_mae': mae,
        'test_mape': mape,
        'test_r2': r2,
        'predictions': predictions,
        'targets': targets,
        'battery_ids': final_battery_ids,
        'training_time': training_time,
        'history': {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': []
        }
    }

    # 创建简单的wrapper用于兼容性
    class XGBoostWrapper:
        def __init__(self, model, results):
            self.model = model
            self.results = results
            self.training_history = results['history']

    wrapper = XGBoostWrapper(model, results)

    print("\n" + "="*70)
    print("XGBoost Baseline Training Completed!")
    print("="*70)

    return wrapper, results, data_dict


if __name__ == "__main__":
    # 测试XGBoost_Simple
    print("\n" + "="*70)
    print("Test 1: XGBoost_Simple Baseline")
    print("="*70)

    wrapper_simple, results_simple, data_dict_simple = train_xgboost_baseline(
        model_type='xgboost_simple',
        seed=42
    )

    print("\n\n" + "="*70)
    print("Test 2: XGBoost_Enhanced Baseline")
    print("="*70)

    wrapper_enhanced, results_enhanced, data_dict_enhanced = train_xgboost_baseline(
        model_type='xgboost_enhanced',
        seed=42
    )

    # 对比结果
    print("\n\n" + "="*70)
    print("Comparison Summary")
    print("="*70)
    print(f"\nXGBoost_Simple:")
    print(f"  RMSE: {results_simple['test_rmse']*100:.4f}%")
    print(f"  MAE:  {results_simple['test_mae']*100:.4f}%")
    print(f"  R2:   {results_simple['test_r2']:.6f}")
    print(f"  Time: {results_simple['training_time']:.2f}s")

    print(f"\nXGBoost_Enhanced:")
    print(f"  RMSE: {results_enhanced['test_rmse']*100:.4f}%")
    print(f"  MAE:  {results_enhanced['test_mae']*100:.4f}%")
    print(f"  R2:   {results_enhanced['test_r2']:.6f}")
    print(f"  Time: {results_enhanced['training_time']:.2f}s")

    improvement = (results_simple['test_rmse'] - results_enhanced['test_rmse']) / results_simple['test_rmse'] * 100
    print(f"\nEnhanced vs Simple RMSE Improvement: {improvement:+.2f}%")
