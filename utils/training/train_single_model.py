"""
使用统一模型工厂系统训练单个模型的示例脚本。

示例：使用CNN在HUST 1-1数据上进行训练
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# 导入新的统一模型工厂系统
from models import ModelFactory, ConfigLoader, UnifiedModelWrapper
from data_loaders import load_single_hust_battery, create_hust_dataloaders
from src.feature_selector import (
    get_top_correlated_features,
    create_filtered_data_dict,
    print_feature_selection_report
)


def load_hust_battery_data(battery_id, data_dir='data/HUST data'):
    """
    加载HUST电池数据的包装函数。

    Args:
        battery_id: 电池ID (例如 '1-1')
        data_dir: 数据目录

    Returns:
        data_dict: 数据字典
    """
    file_path = os.path.join(data_dir, f'{battery_id}.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Battery data file not found: {file_path}")

    return load_single_hust_battery(file_path)


def train_model(model_type='cnn', battery_id='1-1', device='cuda'):
    """
    训练单个模型的完整流程。

    Args:
        model_type: 模型类型 ('fnn', 'cnn', 'lstm', 'bpinn')
        battery_id: 电池ID
        device: 计算设备
    """

    print("=" * 70)
    print(f"训练模型: {model_type.upper()} on HUST Battery {battery_id}")
    print("=" * 70)

    # ==================== 1. 加载配置 ====================
    print("\n[1/6] 加载配置...")
    config = ConfigLoader.load_model_config(model_type)
    ConfigLoader.print_config(config)

    # ==================== 2. 加载数据 ====================
    print("\n[2/6] 加载数据...")
    data_dict = load_hust_battery_data(battery_id)

    print(f"原始特征数: {data_dict['train_features'].shape[1]}")
    print(f"训练样本数: {data_dict['train_features'].shape[0]}")
    print(f"测试样本数: {data_dict['test_features'].shape[0]}")

    # ==================== 3. 特征选择（可选）====================
    if config['feature_selection']['enabled']:
        print("\n[3/6] 特征选择...")
        threshold = config['feature_selection']['correlation_threshold']
        top_k = config['feature_selection'].get('top_k', None)

        selection_result = get_top_correlated_features(
            data_dict,
            threshold,
            top_k=top_k
        )

        print_feature_selection_report(selection_result, threshold)

        # 应用特征筛选
        data_dict = create_filtered_data_dict(
            data_dict,
            selection_result['selected_indices']
        )

        print(f"\n筛选后特征数: {data_dict['selected_feature_count']}")
    else:
        print("\n[3/6] 跳过特征选择")

    # ==================== 4. 创建数据加载器 ====================
    print("\n[4/6] 创建数据加载器...")

    # 根据模型类型决定是否使用窗口化数据
    # LSTM、GRU、BiLSTM、BiGRU 需要窗口化的时序数据，其他模型需要平坦的特征输入
    if model_type.lower() in ['lstm', 'gru', 'bilstm', 'bigru']:
        window_size = config.get('data', {}).get('window_size', 10)
        print(f"  模型 {model_type.upper()} 使用窗口化数据，窗口大小: {window_size}")
    else:
        window_size = 1  # FNN, CNN, MLP, ResCNN 不需要窗口，设置为1以兼容
        print(f"  模型 {model_type.upper()} 使用平坦特征（window_size=1）")

    train_loader, test_loader = create_hust_dataloaders(
        data_dict,
        batch_size=config['training']['batch_size'],
        window_size=window_size
    )

    # ==================== 5. 创建模型 ====================
    print("\n[5/6] 创建模型...")
    input_size = data_dict['train_features'].shape[1]

    # 方式1: 使用UnifiedModelWrapper（推荐）
    wrapper = UnifiedModelWrapper(
        model_type=model_type,
        input_size=input_size,
        config=config,
        device=device
    )

    model = wrapper.model
    ModelFactory.print_model_info(model)

    # 创建优化器和损失函数
    optimizer = wrapper.get_optimizer()
    criterion = wrapper.criterion

    # 学习率调度器（如果配置中启用）
    scheduler = None
    if config['training'].get('scheduler', {}).get('enabled', False):
        scheduler_config = config['training']['scheduler']
        if scheduler_config['type'] == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 500),
                gamma=scheduler_config.get('gamma', 0.9)
            )

    # ==================== 6. 训练模型 ====================
    print("\n[6/6] 开始训练...")
    print(f"训练配置:")
    print(f"  - Epochs: {config['training']['num_epochs']}")
    print(f"  - Batch Size: {config['training']['batch_size']}")
    print(f"  - Learning Rate: {config['training']['learning_rate']}")
    print(f"  - Optimizer: {config['training']['optimizer']}")
    print(f"  - Device: {device}")

    history = {
        'train_loss': [],
        'test_mae': [],
        'test_rmse': []
    }

    best_mae = float('inf')
    best_epoch = 0
    best_model_state = None

    num_epochs = config['training']['num_epochs']

    for epoch in tqdm(range(num_epochs), desc="Training"):
        # ===== 训练阶段 =====
        model.train()
        train_loss = 0.0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device).unsqueeze(1)

            optimizer.zero_grad()

            # 前向传播
            predictions = model(features)
            loss = criterion(predictions, targets)

            # 反向传播
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)

        train_loss /= len(train_loader.dataset)

        # ===== 评估阶段 =====
        model.eval()
        test_mae = 0.0
        test_rmse = 0.0

        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(device)
                targets = targets.to(device).unsqueeze(1)

                predictions = model(features)

                # 计算指标
                mae = torch.mean(torch.abs(predictions - targets)).item()
                rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()

                test_mae += mae * features.size(0)
                test_rmse += rmse * features.size(0)

        test_mae /= len(test_loader.dataset)
        test_rmse /= len(test_loader.dataset)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['test_mae'].append(test_mae)
        history['test_rmse'].append(test_rmse)

        # 保存最佳模型
        if test_mae < best_mae:
            best_mae = test_mae
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        # 每100个epoch打印一次
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Test MAE:   {test_mae*100:.4f}%")
            print(f"  Test RMSE:  {test_rmse*100:.4f}%")
            print(f"  Best MAE:   {best_mae*100:.4f}% (Epoch {best_epoch})")

    # ==================== 7. 恢复最佳模型 ====================
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n已恢复最佳模型 (Epoch {best_epoch})")

    # 更新wrapper的训练状态
    wrapper.best_mae = best_mae
    wrapper.best_epoch = best_epoch
    wrapper.training_history = history

    # ==================== 8. 最终评估 ====================
    print("\n最终评估:")
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            targets = targets.to(device)

            predictions = model(features).squeeze()

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    import numpy as np
    predictions_np = np.array(all_predictions)
    targets_np = np.array(all_targets)

    mae = np.mean(np.abs(predictions_np - targets_np))
    rmse = np.sqrt(np.mean((predictions_np - targets_np) ** 2))
    mape = np.mean(np.abs((predictions_np - targets_np) / targets_np)) * 100

    print(f"  MAE:  {mae*100:.4f}%")
    print(f"  RMSE: {rmse*100:.4f}%")
    print(f"  MAPE: {mape:.4f}%")

    # ==================== 9. 保存结果 ====================
    print("\n保存结果...")
    results_dir = f'results/results_hust/{battery_id}/{model_type}'
    os.makedirs(results_dir, exist_ok=True)

    # 保存模型检查点
    checkpoint_path = os.path.join(results_dir, 'model_checkpoint.pth')
    wrapper.save_checkpoint(
        checkpoint_path,
        epoch=best_epoch,
        optimizer_state=optimizer.state_dict()
    )

    # 保存配置
    config_path = os.path.join(results_dir, 'config.json')
    ConfigLoader.save_config(config, config_path)

    # 保存结果
    results = {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'best_epoch': best_epoch,
        'predictions': predictions_np.tolist(),
        'targets': targets_np.tolist(),
        'history': history
    }

    import pickle
    with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # ==================== 10. 绘制结果 ====================
    print("\n绘制训练曲线...")
    plot_training_results(history, results_dir)

    print("\n绘制预测对比...")
    plot_predictions(predictions_np, targets_np, results_dir)

    print(f"\n所有结果已保存到: {results_dir}/")
    print("=" * 70)

    return wrapper, results


def plot_training_results(history, save_dir):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 训练损失
    axes[0].plot(history['train_loss'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True)

    # MAE
    axes[1].plot([mae * 100 for mae in history['test_mae']])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (%)')
    axes[1].set_title('Test MAE')
    axes[1].grid(True)

    # RMSE
    axes[2].plot([rmse * 100 for rmse in history['test_rmse']])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('RMSE (%)')
    axes[2].set_title('Test RMSE')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300)
    plt.close()


def plot_predictions(predictions, targets, save_dir):
    """绘制预测对比"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 散点图
    axes[0].scatter(targets, predictions, alpha=0.5)
    axes[0].plot([targets.min(), targets.max()],
                 [targets.min(), targets.max()],
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('True SOH')
    axes[0].set_ylabel('Predicted SOH')
    axes[0].set_title('Prediction vs True')
    axes[0].legend()
    axes[0].grid(True)

    # 误差分布
    errors = predictions - targets
    axes[1].hist(errors, bins=50, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Error')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Error Distribution')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'predictions.png'), dpi=300)
    plt.close()


if __name__ == "__main__":
    """
    使用示例：

    1. 训练CNN在HUST 1-1数据上:
        python train_single_model.py

    2. 修改下面的参数来训练不同的配置
    """

    # ===== 配置参数 =====
    MODEL_TYPE = 'gru'      # 选择: 'fnn', 'cnn', 'lstm', 'gru', 'mlp', 'rescnn'
    BATTERY_ID = '1-1'      # HUST电池ID
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ===== 开始训练 =====
    print(f"\n使用设备: {DEVICE}")

    wrapper, results = train_model(
        model_type=MODEL_TYPE,
        battery_id=BATTERY_ID,
        device=DEVICE
    )

    print("\n训练完成！")
    print(f"最终 MAE: {results['mae']*100:.4f}%")
    print(f"最终 RMSE: {results['rmse']*100:.4f}%")
