"""
对比训练脚本：FNN vs CNN vs LSTM for HUST Dataset (Per-Battery模式)

训练三个基准模型并对比HUST数据集结果。
"""

import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import argparse
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader_hust import load_single_hust_battery, create_hust_dataloaders
from src.baseline_models import FNN, CNN, LSTM
from src.utils import ensure_dir
from src.train import evaluate


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_baseline_model(model, train_loader, test_loader, num_epochs, learning_rate, device):
    """
    训练基准模型（FNN/CNN/LSTM）。

    Args:
        model: 模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 设备

    Returns:
        训练历史和最终结果
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_mae = float('inf')
    best_epoch = 0
    best_model_state = None

    history = {
        'train_loss': [],
        'test_mae': [],
        'test_rmse': []
    }

    # Training loop
    iterator = tqdm(range(num_epochs), desc="Training")
    for epoch in iterator:
        # Training
        model.train()
        train_loss = 0.0

        for features, capacity in train_loader:
            features = features.to(device)
            capacity = capacity.to(device).unsqueeze(1)

            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, capacity)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)

        train_loss /= len(train_loader.dataset)

        # Evaluation
        model.eval()
        test_mae_total = 0.0
        test_rmse_total = 0.0

        with torch.no_grad():
            for features, capacity in test_loader:
                features = features.to(device)
                capacity = capacity.to(device).unsqueeze(1)

                predictions = model(features)

                # Calculate metrics
                mae = torch.mean(torch.abs(predictions - capacity)).item()
                rmse = torch.sqrt(torch.mean((predictions - capacity) ** 2)).item()

                test_mae_total += mae * features.size(0)
                test_rmse_total += rmse * features.size(0)

        mae = test_mae_total / len(test_loader.dataset)
        rmse = test_rmse_total / len(test_loader.dataset)

        history['train_loss'].append(train_loss)
        history['test_mae'].append(mae)
        history['test_rmse'].append(rmse)

        # Save best model
        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Update progress bar
        if (epoch + 1) % 100 == 0:
            iterator.set_postfix({
                'Loss': f'{train_loss:.6f}',
                'MAE': f'{mae:.6f}',
                'Best': f'{best_mae:.6f}'
            })

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n=== Restored best model from epoch {best_epoch} ===")

    # Final evaluation - compute final metrics on test set
    model.eval()
    final_mae = 0.0
    final_rmse = 0.0

    with torch.no_grad():
        for features, capacity in test_loader:
            features = features.to(device)
            capacity = capacity.to(device).unsqueeze(1)

            predictions = model(features)

            mae = torch.mean(torch.abs(predictions - capacity)).item()
            rmse = torch.sqrt(torch.mean((predictions - capacity) ** 2)).item()

            final_mae += mae * features.size(0)
            final_rmse += rmse * features.size(0)

    final_mae /= len(test_loader.dataset)
    final_rmse /= len(test_loader.dataset)

    results = {
        'mae': final_mae,
        'rmse': final_rmse,
        'best_epoch': best_epoch,
        'best_mae': best_mae
    }

    return history, results


def main():
    """主函数：训练并对比所有基准模型。"""
    parser = argparse.ArgumentParser(description='Compare FNN, CNN, and LSTM baseline models on HUST dataset')
    parser.add_argument('--battery', type=str, default='1-1',
                        help='Battery to train (e.g., 1-1, 1-2, ..., 10-8)')
    parser.add_argument('--max-batteries', type=int, default=None,
                        help='Train on first N batteries (for batch testing)')
    args = parser.parse_args()

    # Set seed
    set_seed(42)

    # Configuration
    config = {
        # Model architecture
        'hidden_sizes': [64, 32, 16],
        'dropout_rate': 0.2,

        # Training parameters
        'num_epochs': 2500,
        'learning_rate': 0.001,
        'batch_size': 64,

        # Data
        'train_ratio': 0.75,

        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        # Paths
        'data_dir': 'data/HUST data',
        'results_dir': f'results/results_hust/{args.battery}' if args.battery else 'results/results_hust',
    }

    ensure_dir(config['results_dir'])

    device = torch.device(config['device'])
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print("\n" + "=" * 70)
    print(f"Loading HUST Battery Data: {args.battery}")
    print("=" * 70)

    battery_file = os.path.join(config['data_dir'], f'{args.battery}.csv')
    if not os.path.exists(battery_file):
        print(f"Error: Battery file not found: {battery_file}")
        return

    data_dict = load_single_hust_battery(
        battery_file,
        train_ratio=config['train_ratio'],
        normalize_target=True
    )

    print(f"\nLoaded {args.battery}:")
    print(f"  Train: {data_dict['n_train']} samples")
    print(f"  Test:  {data_dict['n_test']} samples")
    print(f"  Features: {len(data_dict['feature_names'])}")

    train_loader, test_loader = create_hust_dataloaders(
        data_dict, batch_size=config['batch_size']
    )

    # Store all results
    all_results = {}

    # ======================================================================
    # 1. Train FNN
    # ======================================================================
    print("\n" + "=" * 70)
    print("Training FNN")
    print("=" * 70)

    fnn = FNN(input_size=16, hidden_sizes=config['hidden_sizes'], dropout_rate=config['dropout_rate'])
    print(f"FNN Parameters: {sum(p.numel() for p in fnn.parameters())}")

    fnn_history, fnn_results = train_baseline_model(
        fnn, train_loader, test_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        device=device
    )

    all_results['FNN'] = fnn_results
    print(f"\nFNN Results: MAE={fnn_results['mae']*100:.4f}%, RMSE={fnn_results['rmse']*100:.4f}%")

    # Save model
    torch.save(fnn.state_dict(), os.path.join(config['results_dir'], 'fnn_model.pth'))

    # ======================================================================
    # 2. Train CNN
    # ======================================================================
    print("\n" + "=" * 70)
    print("Training CNN")
    print("=" * 70)

    cnn = CNN(input_size=16, num_filters=64, fc_hidden_sizes=[32, 16], dropout_rate=config['dropout_rate'])
    print(f"CNN Parameters: {sum(p.numel() for p in cnn.parameters())}")

    cnn_history, cnn_results = train_baseline_model(
        cnn, train_loader, test_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        device=device
    )

    all_results['CNN'] = cnn_results
    print(f"\nCNN Results: MAE={cnn_results['mae']*100:.4f}%, RMSE={cnn_results['rmse']*100:.4f}%")

    # Save model
    torch.save(cnn.state_dict(), os.path.join(config['results_dir'], 'cnn_model.pth'))

    # ======================================================================
    # 3. Train LSTM
    # ======================================================================
    print("\n" + "=" * 70)
    print("Training LSTM")
    print("=" * 70)

    lstm = LSTM(input_size=16, hidden_size=64, num_layers=2, fc_hidden_sizes=[32, 16], dropout_rate=config['dropout_rate'])
    print(f"LSTM Parameters: {sum(p.numel() for p in lstm.parameters())}")

    lstm_history, lstm_results = train_baseline_model(
        lstm, train_loader, test_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        device=device
    )

    all_results['LSTM'] = lstm_results
    print(f"\nLSTM Results: MAE={lstm_results['mae']*100:.4f}%, RMSE={lstm_results['rmse']*100:.4f}%")

    # Save model
    torch.save(lstm.state_dict(), os.path.join(config['results_dir'], 'lstm_model.pth'))

    # ======================================================================
    # Summary
    # ======================================================================
    print("\n" + "=" * 70)
    print(f"COMPARISON SUMMARY - HUST Battery {args.battery}")
    print("=" * 70)

    print(f"\n{'Model':<12} {'MAE (%)':<12} {'RMSE (%)':<12} {'Best Epoch':<15}")
    print("-" * 70)

    for model_name, results in all_results.items():
        mae = results['mae'] * 100
        rmse = results['rmse'] * 100
        best_info = results.get('best_epoch', 'N/A')

        print(f"{model_name:<12} {mae:<12.4f} {rmse:<12.4f} {best_info:<15}")

    print("-" * 70)

    # Find best model
    best_model = min(all_results.items(), key=lambda x: x[1]['mae'])
    print(f"\nBest Model: {best_model[0]} (MAE={best_model[1]['mae']*100:.4f}%)")

    # Save results
    import pickle
    with open(os.path.join(config['results_dir'], 'comparison_results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\nAll results saved to: {config['results_dir']}/")

    return all_results


if __name__ == "__main__":
    results = main()
