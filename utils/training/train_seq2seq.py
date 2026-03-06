"""
Seq2Seq (Many-to-Many) 模型训练脚本

与标准训练的区别：
1. 模型输出整个序列 (batch, seq_len, 1)
2. 目标也是序列 (batch, seq_len, 1)
3. 物理约束在序列维度上计算（终于可以生效了！）
4. shuffle=False 保持时序完整性
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# 导入Seq2Seq模型
from models import (
    LSTMSeq2Seq, GRUSeq2Seq, BiLSTMSeq2Seq, BiGRUSeq2Seq,
    ConfigLoader, PhysicsConstrainedLoss
)
from data_loaders import load_single_hust_battery


def create_seq2seq_dataset(data_dict, window_size=40):
    """
    为Seq2Seq模型创建数据集

    Args:
        data_dict: 原始数据字典
        window_size: 输入窗口大小

    Returns:
        X: (N, window_size, features)
        y: (N, window_size, 1)  - 序列目标（与输入窗口对应的SOH值）
    """
    features = data_dict['train_features']
    capacity = data_dict['train_capacity']

    X_seq, y_seq = [], []

    # 创建滑动窗口
    # 输入：[t-window_size+1, ..., t] 的特征
    # 输出：[t-window_size+1, ..., t] 的容量（SOH）
    for i in range(window_size, len(features) + 1):
        # 输入窗口
        X_seq.append(features[i-window_size:i])

        # 输出序列（完整窗口的SOH值）
        y_seq.append(capacity[i-window_size:i])

    X = np.array(X_seq)
    y = np.array(y_seq)

    return torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(-1)


def train_seq2seq_model(
    model_type='lstm_seq2seq',
    battery_id='1-1',
    device='cuda'
):
    """
    训练Seq2Seq模型

    Args:
        model_type: 模型类型 ('lstm_seq2seq', 'gru_seq2seq', 'bilstm_seq2seq', 'bigru_seq2seq')
        battery_id: 电池ID
        device: 计算设备
    """
    print("="*70)
    print(f"训练 Seq2Seq 模型: {model_type.upper()} on Battery {battery_id}")
    print("="*70)

    # 1. 加载配置
    print("\n[1/6] 加载配置...")
    config = ConfigLoader.load_model_config(model_type)
    print(f"模型: {config['model_type']}")
    print(f"Window size: {config['data']['window_size']}")
    print(f"输出方式: Many-to-Many (完整序列)")

    # 2. 加载数据
    print("\n[2/6] 加载数据...")
    data_path = f'data/HUST data/{battery_id}.csv'
    data_dict = load_single_hust_battery(data_path, train_ratio=0.75)

    print(f"原始数据: {data_dict['train_features'].shape[0]} 个循环")

    # 3. 创建Seq2Seq数据集
    print("\n[3/6] 创建Seq2Seq数据集...")
    window_size = config['data']['window_size']

    X_train, y_train = create_seq2seq_dataset(
        {'train_features': data_dict['train_features'],
         'train_capacity': data_dict['train_capacity']},
        window_size=window_size
    )

    X_test, y_test = create_seq2seq_dataset(
        {'train_features': data_dict['test_features'],
         'train_capacity': data_dict['test_capacity']},
        window_size=window_size
    )

    print(f"训练集: X{X_train.shape}, y{y_train.shape}")
    print(f"测试集: X{X_test.shape}, y{y_test.shape}")

    # 4. 创建DataLoader (shuffle=False for time series!)
    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Batch size: {batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # 5. 创建模型
    print("\n[4/6] 创建模型...")
    input_size = X_train.size(2)

    # 根据配置中的model_type选择模型类
    model_type_map = {
        'lstmseq2seq': LSTMSeq2Seq,
        'gruseq2seq': GRUSeq2Seq,
        'bilstmseq2seq': BiLSTMSeq2Seq,
        'bigruseq2seq': BiGRUSeq2Seq
    }

    # 从配置文件中获取实际的模型类型
    config_model_type = config['model_type'].lower().replace('_', '').replace('-', '')
    model_class = model_type_map.get(config_model_type)

    if model_class is None:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    model = model_class(
        input_size=input_size,
        hidden_size=config['architecture']['hidden_size'],
        num_layers=config['architecture']['num_layers'],
        dropout_rate=config['architecture']['dropout_rate']
    ).to(device)

    print(f"模型: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")

    # 6. 创建损失函数（带物理约束）
    phys_config = config.get('physics_constraints', {})
    if phys_config.get('enabled', False):
        print("\n[物理约束] 启用")
        print(f"  单调性权重: {phys_config['monotonic_weight']}")
        print(f"  边界权重: {phys_config['boundary_weight']}")
        print(f"  平滑性权重: {phys_config['smoothness_weight']}")

        criterion = PhysicsConstrainedLoss(
            base_loss='mse',
            monotonic_weight=phys_config['monotonic_weight'],
            boundary_weight=phys_config['boundary_weight'],
            smoothness_weight=phys_config['smoothness_weight']
        )
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # 7. 训练
    print("\n[5/6] 开始训练...")
    num_epochs = config['training']['num_epochs']
    history = {'train_loss': [], 'test_mae': [], 'test_rmse': []}

    best_mae = float('inf')
    best_epoch = 0

    for epoch in tqdm(range(num_epochs), desc="Training"):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)  # (batch, seq_len, 1)
            loss = criterion(predictions, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # 评估阶段
        model.eval()
        test_mae = 0.0
        test_rmse = 0.0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(X_batch)

                # 只在最后一个时间步计算误差（或者所有时间步的平均）
                mae = torch.mean(torch.abs(predictions - y_batch)).item()
                rmse = torch.sqrt(torch.mean((predictions - y_batch) ** 2)).item()

                test_mae += mae * X_batch.size(0)
                test_rmse += rmse * X_batch.size(0)

        test_mae /= len(test_loader.dataset)
        test_rmse /= len(test_loader.dataset)

        history['train_loss'].append(train_loss)
        history['test_mae'].append(test_mae)
        history['test_rmse'].append(test_rmse)

        if test_mae < best_mae:
            best_mae = test_mae
            best_epoch = epoch + 1

        if (epoch + 1) % 20 == 0:
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Test MAE:   {test_mae*100:.4f}%")
            print(f"  Test RMSE:  {test_rmse*100:.4f}%")
            print(f"  Best MAE:   {best_mae*100:.4f}% (Epoch {best_epoch})")

    # 8. 保存结果
    print("\n[6/6] 保存结果...")
    results_dir = f'results/seq2seq/{battery_id}/{model_type}'
    os.makedirs(results_dir, exist_ok=True)

    # 绘制训练曲线
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True)

    axes[1].plot([mae * 100 for mae in history['test_mae']])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (%)')
    axes[1].set_title('Test MAE')
    axes[1].grid(True)

    axes[2].plot([rmse * 100 for rmse in history['test_rmse']])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('RMSE (%)')
    axes[2].set_title('Test RMSE')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_history.png'), dpi=300)
    plt.close()

    print(f"结果已保存到: {results_dir}/")
    print("="*70)
    print(f"最佳测试 MAE: {best_mae*100:.4f}% (Epoch {best_epoch})")
    print("="*70)

    return model, history


if __name__ == "__main__":
    MODEL_TYPE = 'lstm_seq2seq'  # 'lstm_seq2seq', 'gru_seq2seq', 'bilstm_seq2seq', 'bigru_seq2seq'
    BATTERY_ID = '1-1'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n使用设备: {DEVICE}\n")

    model, history = train_seq2seq_model(
        model_type=MODEL_TYPE,
        battery_id=BATTERY_ID,
        device=DEVICE
    )

    print("\n训练完成！")
