"""
MIT数据集训练脚本

使用方式（和train_cross_battery.py完全一样，只是数据集换成MIT）:
    python train_mit.py

这个脚本会:
1. 加载MIT数据集 (125个电池)
2. 使用相同的数据划分方式 (60%/20%/20%)
3. 使用相同的训练流程
4. 输出相同格式的结果
"""

import os
import sys

# 复用train_cross_battery的所有函数
from train_cross_battery import (
    train_cross_battery_model,
    set_seed,
    split_batteries,
    prepare_cross_battery_data,
    create_dataloaders,
    plot_cross_battery_results,
    plot_top_batteries_capacity_curves
)

# 导入MIT数据加载器
from data_loaders import load_all_mit_batteries


def train_mit_model(
    model_type='cnn_lstm',
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    device='cuda',
    seed=42,
    apply_cleaning=False,
    degradation_scenario='none',
    **kwargs  # 其他参数传递给train_cross_battery_model
):
    """
    在MIT数据集上训练模型

    所有参数与train_cross_battery_model相同
    """
    set_seed(seed)

    print("\n" + "="*70)
    print(f"MIT数据集训练: {model_type.upper()}")
    print(f"数据划分: Train/Val/Test = {train_ratio*100:.0f}%/{val_ratio*100:.0f}%/{test_ratio*100:.0f}%")
    if apply_cleaning:
        print("数据清洗: 启用 (3-Sigma)")
    print("="*70)

    # 1. 加载MIT数据集
    battery_names, all_data = load_all_mit_batteries(
        data_dir='data/MIT data',
        apply_cleaning=apply_cleaning
    )

    # 2. 划分数据集 (与HUST完全相同的方式)
    train_batteries, val_batteries, test_batteries = split_batteries(
        battery_names,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    # 3. 准备数据 (与HUST完全相同的方式)
    data_dict = prepare_cross_battery_data(
        all_data,
        train_batteries,
        val_batteries,
        test_batteries,
        degradation_scenario=degradation_scenario,
        seed=seed,
        **kwargs
    )

    # 4. 加载模型配置
    from models import ConfigLoader
    config = ConfigLoader.load_model_config(model_type)

    print("\n" + "="*70)
    print("模型配置")
    print("="*70)
    print(f"模型类型: {config['model_type']}")
    print(f"学习率: {config['training']['learning_rate']}")
    print(f"批次大小: {config['training']['batch_size']}")
    print(f"训练轮数: {config['training']['num_epochs']}")

    # 5. 创建DataLoader
    batch_size = config['training']['batch_size']
    # 强制使用window_size=40，与HUST保持一致（CNN-LSTM需要序列维度）
    window_size = 40

    physics_config = config.get('physics_constraints', {})
    use_physics = physics_config.get('enabled', False)

    siamese_mode = False
    triplet_mode = False
    if use_physics:
        siamese_mode = physics_config.get('siamese_sampling', {}).get('enabled', False)
        triplet_mode = physics_config.get('triplet_sampling', {}).get('enabled', False)

    train_loader, val_loader, test_loader, test_battery_ids = create_dataloaders(
        data_dict,
        batch_size=batch_size,
        window_size=window_size,
        seq2seq=False,
        use_physics=use_physics,
        siamese_mode=siamese_mode,
        triplet_mode=triplet_mode
    )

    # 6. 创建模型
    from models import ModelFactory, UnifiedModelWrapper
    import torch
    import torch.nn as nn

    input_size = data_dict['n_features']

    # 使用UnifiedModelWrapper创建模型
    wrapper = UnifiedModelWrapper(
        model_type=model_type,
        input_size=input_size,
        config=config,
        device=device
    )

    model = wrapper.model

    # 7. 训练模型
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # 准备损失函数
    if use_physics:
        from models import PhysicsConstrainedLoss, SiamesePhysicsLoss, TripletPhysicsLoss

        # 提取时间衰减配置（兼容新旧两种格式）
        temporal_decay = physics_config.get('temporal_decay', {})

        if triplet_mode:
            criterion = TripletPhysicsLoss(
                base_loss_weight=physics_config.get('base_loss_weight', 1.0),
                monotonic_weight=physics_config.get('monotonic_weight', 0.1),
                boundary_weight=physics_config.get('boundary_weight', 0.05),
                smoothness_weight=physics_config.get('smoothness_weight', 0.0),
                monotonic_tolerance=physics_config.get('monotonic_tolerance', 0.01),
                temporal_decay_enabled=temporal_decay.get('enabled', True),
                temporal_max_step=temporal_decay.get('max_step', 20),
                temporal_decay_type=temporal_decay.get('decay_type', 'exp'),
                temporal_decay_alpha=temporal_decay.get('decay_alpha', 0.2)
            )
        elif siamese_mode:
            criterion = SiamesePhysicsLoss(
                base_loss_weight=physics_config.get('base_loss_weight', 1.0),
                monotonic_weight=physics_config.get('monotonic_weight', 0.1),
                boundary_weight=physics_config.get('boundary_weight', 0.05),
                smoothness_weight=physics_config.get('smoothness_weight', 0.0),
                monotonic_tolerance=physics_config.get('monotonic_tolerance', 0.01),
                temporal_decay_enabled=temporal_decay.get('enabled', True),
                temporal_max_step=temporal_decay.get('max_step', 20),
                temporal_decay_type=temporal_decay.get('decay_type', 'exp'),
                temporal_decay_alpha=temporal_decay.get('decay_alpha', 0.2)
            )
        else:
            criterion = PhysicsConstrainedLoss(
                base_loss_weight=physics_config.get('base_loss_weight', 1.0),
                monotonic_weight=physics_config.get('monotonic_weight', 0.1),
                boundary_weight=physics_config.get('boundary_weight', 0.05),
                smoothness_weight=physics_config.get('smoothness_weight', 0.0),
                monotonic_tolerance=physics_config.get('monotonic_tolerance', 0.01),
                temporal_decay_enabled=temporal_decay.get('enabled', True),
                temporal_max_step=temporal_decay.get('max_step', 20),
                temporal_decay_type=temporal_decay.get('decay_type', 'exp'),
                temporal_decay_alpha=temporal_decay.get('decay_alpha', 0.2)
            )
    else:
        criterion = nn.MSELoss()

    # 简化的训练循环（使用train_cross_battery的逻辑）
    print("\n" + "="*70)
    print("开始训练")
    print("="*70)

    from tqdm import tqdm
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': []
    }

    best_val_mae = float('inf')
    best_model_state = None
    patience = config['training']['early_stopping'].get('patience', 10)
    patience_counter = 0

    num_epochs = config['training']['num_epochs']

    for epoch in tqdm(range(num_epochs), desc="Training"):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            # 处理不同类型的batch（dict或tuple）
            if isinstance(batch, dict):
                features = batch['window'].to(device)
                targets = batch['target_soh'].to(device)
                battery_ids = batch.get('battery_id')
                cycle_indices = batch.get('cycle_idx')
            else:
                # window_size=1时返回tuple
                features, targets = batch
                features = features.to(device)
                targets = targets.to(device)
                targets = targets.unsqueeze(1)  # (batch,) -> (batch, 1)
                battery_ids = None
                cycle_indices = None

            optimizer.zero_grad()
            predictions = model(features)

            if use_physics and battery_ids is not None:
                loss = criterion(predictions, targets, battery_ids, cycle_indices)
            else:
                loss = criterion(predictions, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_rmse = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    features = batch['window'].to(device)
                    targets = batch['target_soh'].to(device)
                else:
                    features, targets = batch
                    features = features.to(device)
                    targets = targets.to(device)
                    targets = targets.unsqueeze(1)

                predictions = model(features)
                loss = nn.MSELoss()(predictions, targets)

                val_loss += loss.item() * features.size(0)
                val_mae += torch.mean(torch.abs(predictions - targets)).item() * features.size(0)
                val_rmse += torch.sqrt(loss).item() * features.size(0)

        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        val_rmse /= len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)

        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 8. 测试评估
    model.eval()
    predictions = []
    targets_list = []
    battery_ids_list = []

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                features = batch['window'].to(device)
                targets = batch['target_soh'].to(device)
                battery_ids = batch.get('battery_id')
            else:
                features, targets = batch
                features = features.to(device)
                targets = targets.to(device)
                targets = targets.unsqueeze(1)
                battery_ids = None

            preds = model(features)
            predictions.append(preds.cpu())
            targets_list.append(targets.cpu())
            if battery_ids:
                battery_ids_list.extend(battery_ids)

    predictions = torch.cat(predictions, dim=0).numpy()
    targets = torch.cat(targets_list, dim=0).numpy()

    # 计算指标
    from sklearn.metrics import mean_absolute_error, r2_score
    import numpy as np

    rmse = np.sqrt(np.mean((targets - predictions) ** 2))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    print("\n" + "="*70)
    print("测试结果 (MIT数据集)")
    print("="*70)
    print(f"RMSE: {rmse*100:.4f}%")
    print(f"MAE:  {mae*100:.4f}%")
    print(f"R2:   {r2:.6f}")
    print("="*70)

    # 9. 保存结果
    results = {
        'test_rmse': rmse,
        'test_mae': mae,
        'test_r2': r2,
        'predictions': predictions,
        'targets': targets,
        'battery_ids': battery_ids_list,
        'history': history,
        'best_epoch': best_epoch
    }

    save_dir = f'results/mit_{model_type}'
    os.makedirs(save_dir, exist_ok=True)

    import pickle
    with open(os.path.join(save_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print(f"\n[OK] Results saved to {save_dir}/")

    return wrapper, results, data_dict


if __name__ == '__main__':
    # 训练CNN-LSTM模型
    wrapper, results, data_dict = train_mit_model(
        model_type='cnn_lstm',
        seed=999,
        device='cuda' if __import__('torch').cuda.is_available() else 'cpu'
    )
