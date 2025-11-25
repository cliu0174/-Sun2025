"""
跨电池训练脚本：使用所有77组HUST电池数据。

数据划分：train/val/test = 6:2:2
- 训练集: 46个电池 (60%)
- 验证集: 16个电池 (20%)
- 测试集: 15个电池 (20%)
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 导入模型工厂
from models import ModelFactory, ConfigLoader, UnifiedModelWrapper
from data_loaders import load_single_hust_battery


def set_seed(seed=42):
    """设置随机种子确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_all_batteries(data_dir='data/HUST data'):
    """
    加载所有77组HUST电池数据。

    Returns:
        battery_names: 电池名称列表
        all_data: 所有电池的数据字典
    """
    print("="*70)
    print("加载所有HUST电池数据")
    print("="*70)

    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    battery_names = [f.replace('.csv', '') for f in csv_files]

    print(f"找到 {len(battery_names)} 组电池数据")

    all_data = {}
    for battery_name in tqdm(battery_names, desc="加载数据"):
        file_path = os.path.join(data_dir, f'{battery_name}.csv')
        try:
            data = load_single_hust_battery(file_path, train_ratio=1.0, normalize_target=True)
            all_data[battery_name] = data
        except Exception as e:
            print(f"警告: 加载 {battery_name} 失败: {e}")

    print(f"成功加载 {len(all_data)} 组电池")
    return battery_names, all_data


def split_batteries(battery_names, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    将电池划分为 train/val/test。

    Args:
        battery_names: 电池名称列表
        train_ratio: 训练集比例 (默认0.6)
        val_ratio: 验证集比例 (默认0.2)
        test_ratio: 测试集比例 (默认0.2)
        seed: 随机种子

    Returns:
        train_batteries, val_batteries, test_batteries: 三个电池名称列表
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 打乱电池顺序
    shuffled_batteries = battery_names.copy()
    random.shuffle(shuffled_batteries)

    n_total = len(shuffled_batteries)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_batteries = shuffled_batteries[:n_train]
    val_batteries = shuffled_batteries[n_train:n_train + n_val]
    test_batteries = shuffled_batteries[n_train + n_val:]

    print("\n" + "="*70)
    print("数据集划分")
    print("="*70)
    print(f"总电池数: {n_total}")
    print(f"训练集: {len(train_batteries)} 个电池 ({len(train_batteries)/n_total*100:.1f}%)")
    print(f"验证集: {len(val_batteries)} 个电池 ({len(val_batteries)/n_total*100:.1f}%)")
    print(f"测试集: {len(test_batteries)} 个电池 ({len(test_batteries)/n_total*100:.1f}%)")

    print(f"\n训练集电池: {train_batteries[:5]}... (显示前5个)")
    print(f"验证集电池: {val_batteries[:5]}... (显示前5个)")
    print(f"测试集电池: {test_batteries[:5]}... (显示前5个)")

    return train_batteries, val_batteries, test_batteries


def prepare_cross_battery_data(all_data, train_batteries, val_batteries, test_batteries):
    """
    准备跨电池的训练/验证/测试数据。

    Args:
        all_data: 所有电池的数据字典
        train_batteries: 训练集电池列表
        val_batteries: 验证集电池列表
        test_batteries: 测试集电池列表

    Returns:
        data_dict: 包含train/val/test的数据字典
    """
    print("\n" + "="*70)
    print("合并数据集")
    print("="*70)

    def merge_batteries(battery_list):
        """合并多个电池的数据"""
        all_features = []
        all_targets = []

        for battery_name in battery_list:
            data = all_data[battery_name]
            all_features.append(data['train_features'])
            all_targets.append(data['train_capacity'])

        features = np.vstack(all_features)
        targets = np.concatenate(all_targets)

        return features, targets

    # 合并各个集合
    train_features, train_targets = merge_batteries(train_batteries)
    val_features, val_targets = merge_batteries(val_batteries)
    test_features, test_targets = merge_batteries(test_batteries)

    print(f"训练集: {train_features.shape[0]} 个样本")
    print(f"验证集: {val_features.shape[0]} 个样本")
    print(f"测试集: {test_features.shape[0]} 个样本")
    print(f"特征维度: {train_features.shape[1]}")

    # 使用训练集的统计量进行标准化
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    test_features_scaled = scaler.transform(test_features)

    data_dict = {
        'train_features': train_features_scaled,
        'train_targets': train_targets,
        'val_features': val_features_scaled,
        'val_targets': val_targets,
        'test_features': test_features_scaled,
        'test_targets': test_targets,
        'scaler': scaler,
        'n_features': train_features.shape[1],
        'train_batteries': train_batteries,
        'val_batteries': val_batteries,
        'test_batteries': test_batteries
    }

    return data_dict


def create_dataloaders(data_dict, batch_size=64, window_size=1, seq2seq=False):
    """
    创建PyTorch DataLoader，支持窗口化数据。

    Args:
        data_dict: 数据字典
        batch_size: 批次大小
        window_size: 窗口大小（LSTM/GRU/BiLSTM/BiGRU使用>1的值，其他模型使用1）
        seq2seq: 是否使用Seq2Seq模式（Many-to-Many）
                 注意：LSTM/GRU/BiLSTM/BiGRU现在都是Seq2Seq架构
    """
    from torch.utils.data import TensorDataset, DataLoader

    def apply_windowing(features, targets, window_size, seq2seq=False):
        """
        对特征进行滑动窗口处理

        Args:
            seq2seq: 如果为True，返回完整序列目标（Many-to-Many）
                    如果为False，只返回最后一个时间步目标（Many-to-One）
        """
        if window_size <= 1:
            # 不需要窗口化，直接返回
            return features, targets

        # 创建滑动窗口数据
        windowed_features = []
        windowed_targets = []

        for i in range(len(features) - window_size):
            window_feat = features[i:i+window_size]  # (window_size, n_features)
            windowed_features.append(window_feat)

            if seq2seq:
                # Seq2Seq模式：输入X[i:i+window_size]，预测Y[i+1:i+window_size+1]
                # 例如：输入cycle[0-19]特征 → 预测cycle[1-20]的SOH
                window_targ = targets[i+1:i+window_size+1]  # (window_size,)
                windowed_targets.append(window_targ)
            else:
                # Many-to-One模式：只返回最后一个时间步的目标
                windowed_targets.append(targets[i+window_size-1])  # scalar

        return np.array(windowed_features), np.array(windowed_targets)

    # 对训练集、验证集、测试集分别进行窗口化处理
    train_feat, train_targ = apply_windowing(
        data_dict['train_features'],
        data_dict['train_targets'],
        window_size,
        seq2seq
    )
    val_feat, val_targ = apply_windowing(
        data_dict['val_features'],
        data_dict['val_targets'],
        window_size,
        seq2seq
    )
    test_feat, test_targ = apply_windowing(
        data_dict['test_features'],
        data_dict['test_targets'],
        window_size,
        seq2seq
    )

    # 处理目标张量形状
    if seq2seq:
        # Seq2Seq模式：目标是 (N, seq_len, 1)
        train_targ_tensor = torch.FloatTensor(train_targ).unsqueeze(-1)
        val_targ_tensor = torch.FloatTensor(val_targ).unsqueeze(-1)
        test_targ_tensor = torch.FloatTensor(test_targ).unsqueeze(-1)
        # Seq2Seq使用shuffle=False保持时序
        shuffle_train = False
    else:
        # Many-to-One模式：目标是 (N,)
        train_targ_tensor = torch.FloatTensor(train_targ)
        val_targ_tensor = torch.FloatTensor(val_targ)
        test_targ_tensor = torch.FloatTensor(test_targ)
        shuffle_train = True

    # 训练集
    train_dataset = TensorDataset(
        torch.FloatTensor(train_feat),
        train_targ_tensor
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)

    # 验证集
    val_dataset = TensorDataset(
        torch.FloatTensor(val_feat),
        val_targ_tensor
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 测试集
    test_dataset = TensorDataset(
        torch.FloatTensor(test_feat),
        test_targ_tensor
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_cross_battery_model(
    model_type='cnn',
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    device='cuda',
    seed=42
):
    """
    跨电池训练模型。

    Args:
        model_type: 模型类型 ('fnn', 'cnn', 'lstm', 'gru', 'mlp', 'rescnn')
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        device: 计算设备
        seed: 随机种子
    """
    set_seed(seed)

    print("\n" + "="*70)
    print(f"跨电池训练: {model_type.upper()}")
    print(f"数据划分: Train/Val/Test = {train_ratio*100:.0f}%/{val_ratio*100:.0f}%/{test_ratio*100:.0f}%")
    print("="*70)

    # 1. 加载所有电池数据
    battery_names, all_data = load_all_batteries()

    # 2. 划分数据集
    train_batteries, val_batteries, test_batteries = split_batteries(
        battery_names,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    # 3. 准备数据
    data_dict = prepare_cross_battery_data(all_data, train_batteries, val_batteries, test_batteries)

    # 4. 加载模型配置
    config = ConfigLoader.load_model_config(model_type)
    print("\n" + "="*70)
    print("模型配置")
    print("="*70)
    print(f"模型类型: {config['model_type']}")
    print(f"隐藏层配置: {config['architecture'].get('hidden_sizes', config['architecture'])}")
    print(f"学习率: {config['training']['learning_rate']}")
    print(f"批次大小: {config['training']['batch_size']}")
    print(f"训练轮数: {config['training']['num_epochs']}")

    # 5. 创建数据加载器（根据模型类型设置窗口大小和Seq2Seq模式）
    config_model_type = config['model_type'].lower()

    # 检测是否为 Many-to-Many 模型（输出整个序列）
    is_seq2seq = any(keyword in config_model_type for keyword in ['manytomany', 'seq2seq'])

    # 检测是否为需要窗口化的模型（即使是Many-to-One的LSTM/GRU）
    needs_window = any(model_name in config_model_type for model_name in ['lstm', 'gru'])

    # 设置窗口大小
    if is_seq2seq or needs_window:
        window_size = config.get('data', {}).get('window_size', 40)
        if is_seq2seq:
            print(f"\n模型 {config_model_type.upper()} 使用 Many-to-Many，窗口大小: {window_size}")
        else:
            print(f"\n模型 {config_model_type.upper()} 使用 Many-to-One，窗口大小: {window_size}")
    else:
        window_size = 1
        print(f"\n模型 {config_model_type.upper()} 使用平坦特征（window_size=1）")

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dict,
        batch_size=config['training']['batch_size'],
        window_size=window_size,
        seq2seq=is_seq2seq
    )

    # 6. 创建模型
    print("\n创建模型...")
    wrapper = UnifiedModelWrapper(
        model_type=model_type,
        input_size=data_dict['n_features'],
        config=config,
        device=device
    )

    model = wrapper.model
    ModelFactory.print_model_info(model)

    optimizer = wrapper.get_optimizer()
    criterion = wrapper.criterion

    # 6.5 创建学习率调度器 (Warmup + Cosine Decay)
    scheduler = None
    if config['training'].get('scheduler', {}).get('enabled', False):
        print("\n创建学习率调度器...")
        scheduler_config = config['training']['scheduler']
        warmup_epochs = scheduler_config.get('warmup_epochs', 30)
        warmup_lr = scheduler_config.get('warmup_lr', 2e-3)
        base_lr = scheduler_config.get('base_lr', 1e-2)
        final_lr = scheduler_config.get('final_lr', 2e-4)

        print(f"  Warmup epochs: {warmup_epochs}")
        print(f"  Warmup LR: {warmup_lr:.6f}")
        print(f"  Base LR: {base_lr:.6f}")
        print(f"  Final LR: {final_lr:.6f}")

        def lr_lambda(epoch):
            """
            学习率调度函数: Warmup + Cosine Decay
            返回相对于optimizer中base_lr的倍数因子
            """
            if epoch < warmup_epochs:
                # Warmup阶段: 线性增长 (warmup_lr → base_lr)
                current_lr = warmup_lr + (base_lr - warmup_lr) * epoch / warmup_epochs
            else:
                # Cosine Decay阶段 (base_lr → final_lr)
                progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                current_lr = final_lr + (base_lr - final_lr) * 0.5 * (1 + np.cos(np.pi * progress))

            # 返回相对于optimizer base_lr的倍数
            # optimizer的base_lr是config中的learning_rate (0.002)
            optimizer_base_lr = config['training']['learning_rate']
            return current_lr / optimizer_base_lr

        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lr_lambda)
        print("  调度器创建成功 [OK]")

    # 7. 训练模型
    print("\n" + "="*70)
    print("开始训练")
    print("="*70)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': []
    }

    # 使用val_mae判断最佳模型
    best_val_mae = float('inf')
    best_epoch = 0
    best_model_state = None

    # Early stopping配置
    patience = config['training']['early_stopping'].get('patience', 10)
    patience_counter = 0

    num_epochs = config['training']['num_epochs']

    for epoch in tqdm(range(num_epochs), desc="Training"):
        # ===== 训练阶段 =====
        model.train()
        train_loss = 0.0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            # 只有Many-to-One模型需要unsqueeze
            if not is_seq2seq:
                targets = targets.unsqueeze(1)

            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)

        train_loss /= len(train_loader.dataset)

        # ===== 更新学习率 (在validation之前，与原始代码一致) =====
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()

        # ===== 验证阶段 =====
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_rmse = 0.0

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)

                # 只有Many-to-One模型需要unsqueeze
                if not is_seq2seq:
                    targets = targets.unsqueeze(1)

                predictions = model(features)
                loss = criterion(predictions, targets)

                val_loss += loss.item() * features.size(0)
                val_mae += torch.mean(torch.abs(predictions - targets)).item() * features.size(0)
                val_rmse += torch.sqrt(torch.mean((predictions - targets) ** 2)).item() * features.size(0)

        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        val_rmse /= len(val_loader.dataset)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)

        # ===== 保存最佳模型和Early Stopping =====
        patience_counter += 1  # 每个epoch递增

        if val_mae < best_val_mae:  # 基于val_mae判断
            best_val_mae = val_mae
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0  # 重置early stopping计数器

        # 每10个epoch打印一次
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            if scheduler is not None:
                print(f"  LR:         {current_lr:.6f}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Val MAE:    {val_mae*100:.4f}% (Best: {best_val_mae*100:.4f}%)")
            print(f"  Val RMSE:   {val_rmse*100:.4f}%")
            print(f"  Best Epoch: {best_epoch}")
            if config['training']['early_stopping'].get('enabled', False):
                print(f"  Patience:   {patience_counter}/{patience}")

        # Early stopping检查
        if config['training']['early_stopping'].get('enabled', False):
            if patience_counter > patience:
                print(f"\n早停触发！验证集MAE连续{patience}个epoch未改善")
                print(f"最佳epoch: {best_epoch}, 最佳val_mae: {best_val_mae*100:.4f}%")
                break

    # 8. 恢复最佳模型
    print("\n" + "="*70)
    print("训练完成")
    print("="*70)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"已恢复最佳模型 (Epoch {best_epoch}, Val MAE: {best_val_mae*100:.4f}%)")

    # 9. 测试集评估
    print("\n在测试集上评估...")
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            predictions = model(features).cpu().numpy()
            targets = targets.cpu().numpy()

            # 处理输出形状
            if is_seq2seq:
                # Seq2Seq: (batch, seq_len, 1) -> 展平为1D
                predictions = predictions.reshape(-1)
                targets = targets.reshape(-1)
            else:
                # Many-to-One: (batch, 1) -> squeeze
                predictions = predictions.squeeze()

            all_predictions.extend(predictions)
            all_targets.extend(targets)

    predictions_np = np.array(all_predictions)
    targets_np = np.array(all_targets)

    test_mae = np.mean(np.abs(predictions_np - targets_np))
    test_rmse = np.sqrt(np.mean((predictions_np - targets_np) ** 2))
    test_mape = np.mean(np.abs((predictions_np - targets_np) / targets_np)) * 100

    print("\n" + "="*70)
    print("测试集结果")
    print("="*70)
    print(f"MAE:  {test_mae*100:.4f}%")
    print(f"RMSE: {test_rmse*100:.4f}%")
    print(f"MAPE: {test_mape:.4f}%")

    # 10. 保存结果
    results_dir = f'results/cross_battery/{model_type}'
    os.makedirs(results_dir, exist_ok=True)

    # 保存模型
    checkpoint_path = os.path.join(results_dir, 'model_checkpoint.pth')
    wrapper.best_mae = best_val_mae
    wrapper.best_epoch = best_epoch
    wrapper.training_history = history
    wrapper.save_checkpoint(checkpoint_path, epoch=best_epoch)

    # 保存配置和数据划分信息
    import json
    split_info = {
        'train_batteries': train_batteries,
        'val_batteries': val_batteries,
        'test_batteries': test_batteries,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'seed': seed
    }
    with open(os.path.join(results_dir, 'battery_split.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    # 保存结果
    results = {
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_mape': float(test_mape),
        'best_val_mae': float(best_val_mae),
        'best_epoch': best_epoch,
        'predictions': predictions_np.tolist(),
        'targets': targets_np.tolist(),
        'history': history
    }

    import pickle
    with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # 绘制图表
    plot_cross_battery_results(history, predictions_np, targets_np, results_dir)

    print(f"\n所有结果已保存到: {results_dir}/")
    print("="*70)

    return wrapper, results, data_dict


def plot_cross_battery_results(history, predictions, targets, save_dir):
    """绘制训练结果"""

    # 1. 训练曲线
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot([mae * 100 for mae in history['val_mae']])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (%)')
    axes[1].set_title('Validation MAE')
    axes[1].grid(True)

    axes[2].plot([rmse * 100 for rmse in history['val_rmse']])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('RMSE (%)')
    axes[2].set_title('Validation RMSE')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300)
    plt.close()

    # 2. 预测对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(targets, predictions, alpha=0.5, s=10)
    axes[0].plot([targets.min(), targets.max()],
                 [targets.min(), targets.max()],
                 'r--', lw=2, label='Perfect')
    axes[0].set_xlabel('True SOH')
    axes[0].set_ylabel('Predicted SOH')
    axes[0].set_title('Test Set Predictions')
    axes[0].legend()
    axes[0].grid(True)

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

    跨电池训练，使用77组HUST数据
    train/val/test = 6:2:2 划分
    """

    # ===== 配置参数 =====
    MODEL_TYPE = 'lstm_manytomany'         # 模型类型: 'fnn', 'cnn', 'lstm', 'gru', 'bilstm', 'bigru', 'mlp', 'rescnn'
    TRAIN_RATIO = 0.6           # 训练集比例 (60%)
    VAL_RATIO = 0.2             # 验证集比例 (20%)
    TEST_RATIO = 0.2            # 测试集比例 (20%)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 42                   # 随机种子（确保可复现）

    # ===== 开始训练 =====
    print(f"\n使用设备: {DEVICE}\n")

    wrapper, results, data_dict = train_cross_battery_model(
        model_type=MODEL_TYPE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        device=DEVICE,
        seed=SEED
    )

    print("\n" + "="*70)
    print("跨电池训练完成！")
    print("="*70)
    print(f"\n最终测试集结果:")
    print(f"  MAE:  {results['test_mae']*100:.4f}%")
    print(f"  RMSE: {results['test_rmse']*100:.4f}%")
    print(f"  MAPE: {results['test_mape']:.4f}%")
    print(f"\n训练集: {len(data_dict['train_batteries'])} 个电池")
    print(f"验证集: {len(data_dict['val_batteries'])} 个电池")
    print(f"测试集: {len(data_dict['test_batteries'])} 个电池")
