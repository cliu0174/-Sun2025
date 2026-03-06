"""
带物理约束的跨电池训练脚本

支持 Many-to-One 模型（LSTM, GRU, BiLSTM, BiGRU 等）
通过 battery_id 和 cycle_idx 在 batch 内重建序列关系，应用物理约束
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import json
from tqdm import tqdm

# 导入模型和数据加载器
from models import PhysicsConstrainedLoss, SiamesePhysicsLoss, ConfigLoader, ModelFactory
from data_loaders.data_loader_hust import (
    load_single_hust_battery,
    apply_windowing_with_metadata,
    HUSTBatteryDatasetWithMetadata
)


def load_batteries_with_windowing(data_dir='data/HUST data',
                                  train_ratio=0.6,
                                  window_size=40,
                                  max_batteries=None,
                                  apply_cleaning=False,
                                  siamese_mode=False,
                                  step_k=1):
    """
    加载多个电池数据并应用窗口，附加 battery_id 和 cycle_idx

    Args:
        data_dir: 数据目录
        train_ratio: 训练集比例
        window_size: 窗口大小
        max_batteries: 最多加载多少个电池
        apply_cleaning: 是否应用数据清洗
        siamese_mode: 是否使用孪生采样模式 (default=False保持兼容)
        step_k: 孪生采样的步长 (default=1, 相邻样本)

    Returns:
        train_datasets: 训练集 Dataset 列表
        test_datasets: 测试集 Dataset 列表
        battery_names: 电池名称列表
    """
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

    if max_batteries is not None:
        csv_files = csv_files[:max_batteries]

    mode_str = "孪生采样" if siamese_mode else "标准"
    print(f"加载 {len(csv_files)} 个电池数据 ({mode_str}模式)...")

    train_datasets = []
    test_datasets = []
    battery_names = []

    for csv_file in tqdm(csv_files, desc="加载数据"):
        file_path = os.path.join(data_dir, csv_file)
        battery_id = csv_file.replace('.csv', '')

        try:
            # 加载原始数据
            data = load_single_hust_battery(
                file_path,
                train_ratio=train_ratio,
                normalize_target=True,
                apply_cleaning=apply_cleaning
            )

            # 应用窗口并附加元数据
            X_train, y_train, bid_train, cyc_train = apply_windowing_with_metadata(
                data['train_features'],
                data['train_capacity'],
                window_size=window_size,
                battery_id=battery_id,
                mode='many_to_one'
            )

            X_test, y_test, bid_test, cyc_test = apply_windowing_with_metadata(
                data['test_features'],
                data['test_capacity'],
                window_size=window_size,
                battery_id=battery_id,
                mode='many_to_one'
            )

            # 创建 Dataset (支持孪生模式切换)
            train_dataset = HUSTBatteryDatasetWithMetadata(
                X_train, y_train, bid_train, cyc_train,
                siamese_mode=siamese_mode,
                step_k=step_k
            )
            test_dataset = HUSTBatteryDatasetWithMetadata(
                X_test, y_test, bid_test, cyc_test,
                siamese_mode=siamese_mode,
                step_k=step_k
            )

            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)
            battery_names.append(battery_id)

        except Exception as e:
            print(f"[ERROR] 加载 {csv_file} 失败: {e}")
            continue

    print(f"成功加载 {len(battery_names)} 个电池")

    return train_datasets, test_datasets, battery_names


def custom_collate_fn(batch):
    """
    自定义 collate function，处理带元数据的 batch

    Supports both standard and Siamese modes:
    - Standard mode: returns single windows
    - Siamese mode: returns paired windows (x_t, x_next)

    Args:
        batch: list of dicts

    Returns:
        dict with batched tensors and lists
    """
    # Check mode by examining first item's keys
    if 'window' in batch[0]:
        # Standard mode
        windows = torch.stack([item['window'] for item in batch])
        targets = torch.stack([item['target_soh'] for item in batch])
        battery_ids = [item['battery_id'] for item in batch]
        cycle_indices = torch.stack([item['cycle_idx'] for item in batch])

        return {
            'window': windows,
            'target_soh': targets,
            'battery_id': battery_ids,
            'cycle_idx': cycle_indices
        }
    else:
        # Siamese mode
        x_t = torch.stack([item['x_t'] for item in batch])
        x_next = torch.stack([item['x_next'] for item in batch])
        y_t = torch.stack([item['y_t'] for item in batch])
        y_next = torch.stack([item['y_next'] for item in batch])
        cycle_indices = torch.stack([item['cycle_index'] for item in batch])
        battery_ids = [item['battery_id'] for item in batch]

        return {
            'x_t': x_t,
            'x_next': x_next,
            'y_t': y_t,
            'y_next': y_next,
            'cycle_index': cycle_indices,
            'battery_id': battery_ids
        }


def train_with_physics_constraints(
    model_type='lstm',
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    device='cuda',
    seed=42,
    max_batteries=None,
    apply_cleaning=False,
    use_physics=True,
    save_results=True,
    siamese_mode=False,
    split_threshold=300,
    step_k=1
):
    """
    使用物理约束训练 Many-to-One 模型

    Supports both standard and Siamese/Pairwise sampling modes.

    Args:
        model_type: 模型类型 ('lstm', 'gru', 'bilstm', 'bigru')
        train_ratio, val_ratio, test_ratio: 数据集划分比例
        device: 计算设备
        seed: 随机种子
        max_batteries: 最多使用多少个电池（None=全部）
        apply_cleaning: 是否应用数据清洗
        use_physics: 是否使用物理约束
        save_results: 是否保存结果
        siamese_mode: 是否使用孪生采样 (default=False保持兼容)
        split_threshold: 分段约束阈值 (仅Siamese模式, default=300)
        step_k: 配对步长 (仅Siamese模式, default=1)
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载配置
    config = ConfigLoader.load_model_config(model_type)
    window_size = config['data']['window_size']
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']

    # 读取孪生采样配置（如果命令行未指定，则从配置文件读取）
    if siamese_mode is False and split_threshold == 300 and step_k == 1:
        # 使用默认值，检查配置文件
        siamese_config = config.get('physics_constraints', {}).get('siamese_sampling', {})
        siamese_mode = siamese_config.get('enabled', False)
        split_threshold = siamese_config.get('split_threshold', 300)
        step_k = siamese_config.get('step_k', 1)

    # 加载数据 (支持孪生模式)
    train_datasets, test_datasets, battery_names = load_batteries_with_windowing(
        data_dir='data/HUST data',
        train_ratio=train_ratio + val_ratio,  # 先合并训练+验证
        window_size=window_size,
        max_batteries=max_batteries,
        apply_cleaning=apply_cleaning,
        siamese_mode=siamese_mode,
        step_k=step_k
    )

    # 合并所有电池的数据
    full_train_dataset = ConcatDataset(train_datasets)
    full_test_dataset = ConcatDataset(test_datasets)

    # 从训练集中分出验证集
    train_size = int(len(full_train_dataset) * train_ratio / (train_ratio + val_ratio))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size]
    )

    print(f"\n数据集大小:")
    print(f"  训练集: {len(train_dataset)}")
    print(f"  验证集: {len(val_dataset)}")
    print(f"  测试集: {len(full_test_dataset)}")

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 关键！可以打乱
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    test_loader = DataLoader(
        full_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    # 创建模型
    input_size = 16  # HUST 数据集特征数
    model = ModelFactory.create_model(model_type, input_size, config)
    model = model.to(device)

    print(f"\n模型: {model_type.upper()}")
    print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建损失函数
    if use_physics and config.get('physics_constraints', {}).get('enabled', False):
        physics_config = config['physics_constraints']

        if siamese_mode:
            # Siamese mode: use SiamesePhysicsLoss with split constraint
            criterion = SiamesePhysicsLoss(
                base_loss_weight=physics_config.get('base_loss_weight', 1.0),
                monotonic_weight=physics_config.get('monotonic_weight', 0.1),
                smoothness_weight=physics_config.get('smoothness_weight', 0.01),
                boundary_weight=physics_config.get('boundary_weight', 0.05),
                split_threshold=split_threshold,
                monotonic_tolerance=physics_config.get('monotonic_tolerance', 0.0),
                verbose=False
            ).to(device)
            print(f"\n损失函数: SiamesePhysicsLoss (孪生采样 + 分段约束)")
            print(f"  分段阈值: {split_threshold} cycles")
            print(f"  单调性权重: {physics_config.get('monotonic_weight', 0.1)}")
            print(f"  平滑性权重: {physics_config.get('smoothness_weight', 0.01)}")
            print(f"  容忍度: {physics_config.get('monotonic_tolerance', 0.0)}")
        else:
            # Standard mode: use PhysicsConstrainedLoss
            criterion = PhysicsConstrainedLoss(
                base_loss_weight=physics_config.get('base_loss_weight', 1.0),
                monotonic_weight=physics_config.get('monotonic_weight', 0.1),
                boundary_weight=physics_config.get('boundary_weight', 0.05),
                smoothness_weight=physics_config.get('smoothness_weight', 0.0),
                monotonic_tolerance=physics_config.get('monotonic_tolerance', 0.01),
                temporal_decay_enabled=physics_config.get('temporal_decay', {}).get('enabled', True),
                temporal_max_step=physics_config.get('temporal_decay', {}).get('max_step', 20),
                temporal_decay_type=physics_config.get('temporal_decay', {}).get('decay_type', 'exp'),
                temporal_decay_alpha=physics_config.get('temporal_decay', {}).get('decay_alpha', 0.2),
                verbose=False
            ).to(device)
            print(f"\n损失函数: PhysicsConstrainedLoss (标准模式)")
            print(f"  单调性权重: {physics_config.get('monotonic_weight', 0.1)}")
            print(f"  容忍度: {physics_config.get('monotonic_tolerance', 0.01)}")
            print(f"  时间衰减: {physics_config.get('temporal_decay', {}).get('enabled', True)}")
    else:
        criterion = nn.MSELoss()
        print(f"\n损失函数: MSELoss (无物理约束)")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练
    print(f"\n开始训练...")
    print("="*70)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()

            if siamese_mode:
                # Siamese mode: paired samples
                x_t = batch['x_t'].to(device)
                x_next = batch['x_next'].to(device)
                y_t = batch['y_t'].to(device)
                y_next = batch['y_next'].to(device)
                cycle_indices = batch['cycle_index'].to(device)

                # Two forward passes
                pred_t = model(x_t)
                pred_next = model(x_next)

                # Compute loss with Siamese physics loss
                loss = criterion(pred_t, pred_next, y_t, y_next, cycle_indices)
            else:
                # Standard mode: single samples
                windows = batch['window'].to(device)
                targets = batch['target_soh'].to(device)
                battery_ids = batch['battery_id']
                cycle_indices = batch['cycle_idx']

                # Forward pass
                outputs = model(windows)

                # Compute loss
                if isinstance(criterion, PhysicsConstrainedLoss):
                    loss = criterion(outputs, targets, battery_ids, cycle_indices)
                else:
                    loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if siamese_mode:
                    # Siamese mode: paired samples
                    x_t = batch['x_t'].to(device)
                    x_next = batch['x_next'].to(device)
                    y_t = batch['y_t'].to(device)
                    y_next = batch['y_next'].to(device)
                    cycle_indices = batch['cycle_index'].to(device)

                    pred_t = model(x_t)
                    pred_next = model(x_next)

                    loss = criterion(pred_t, pred_next, y_t, y_next, cycle_indices)
                else:
                    # Standard mode
                    windows = batch['window'].to(device)
                    targets = batch['target_soh'].to(device)
                    battery_ids = batch['battery_id']
                    cycle_indices = batch['cycle_idx']

                    outputs = model(windows)

                    if isinstance(criterion, PhysicsConstrainedLoss):
                        loss = criterion(outputs, targets, battery_ids, cycle_indices)
                    else:
                        loss = criterion(outputs, targets)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)

        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

            # 如果使用物理约束，打印详细损失
            if isinstance(criterion, (PhysicsConstrainedLoss, SiamesePhysicsLoss)):
                details = criterion.get_loss_details()
                if siamese_mode:
                    print(f"  [详细] Base: {details['base']:.6f}, Mono: {details['monotonic']:.6f}, "
                          f"Smooth: {details['smoothness']:.6f}, Bound: {details['boundary']:.6f}, "
                          f"Mask: {details['mask_active_ratio']*100:.1f}%")
                else:
                    print(f"  [详细] Base: {details['base']:.6f}, Mono: {details['monotonic']:.6f}, Bound: {details['boundary']:.6f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if save_results:
                torch.save(model.state_dict(), f'best_model_{model_type}_physics.pth')

    print("\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.6f}")

    # 测试阶段
    print("\n测试阶段...")
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="测试"):
            windows = batch['window'].to(device)
            targets = batch['target_soh'].to(device)

            outputs = model(windows)

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # 计算指标
    mae = np.mean(np.abs(predictions - targets)) * 100
    rmse = np.sqrt(np.mean((predictions - targets)**2)) * 100
    mape = np.mean(np.abs((predictions - targets) / (targets + 1e-8))) * 100

    print("\n测试结果:")
    print(f"  MAE:  {mae:.4f}%")
    print(f"  RMSE: {rmse:.4f}%")
    print(f"  MAPE: {mape:.4f}%")

    # 保存结果
    if save_results:
        results = {
            'model_type': model_type,
            'use_physics': use_physics,
            'test_mae': mae,
            'test_rmse': rmse,
            'test_mape': mape,
            'train_losses': train_losses,
            'val_losses': val_losses
        }

        os.makedirs('results/physics_constraints', exist_ok=True)
        result_file = f'results/physics_constraints/{model_type}_{"with" if use_physics else "without"}_physics.json'

        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n结果已保存到: {result_file}")

    return mae, rmse, mape


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train with physics constraints')
    parser.add_argument('--model', type=str, default='lstm', help='Model type (lstm, gru, bilstm, bigru)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--max_batteries', type=int, default=None, help='Max number of batteries')
    parser.add_argument('--no_physics', action='store_true', help='Disable physics constraints')
    parser.add_argument('--cleaning', action='store_true', help='Enable data cleaning')

    # New: Siamese mode arguments
    parser.add_argument('--siamese', action='store_true', help='Enable Siamese/Pairwise sampling mode')
    parser.add_argument('--split_threshold', type=int, default=300,
                       help='Split threshold for masking physics loss (cycles, default=300)')
    parser.add_argument('--step_k', type=int, default=1,
                       help='Step size for pairing (default=1 for consecutive samples)')

    args = parser.parse_args()

    # 训练
    mae, rmse, mape = train_with_physics_constraints(
        model_type=args.model,
        device=args.device,
        max_batteries=args.max_batteries,
        apply_cleaning=args.cleaning,
        use_physics=not args.no_physics,
        siamese_mode=args.siamese,
        split_threshold=args.split_threshold,
        step_k=args.step_k
    )

    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)
