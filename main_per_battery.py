"""
Per-battery training script for BPINN (按论文方法).

每个电池单独训练一个模型：
- B05: 训练一个BPINN
- B06: 训练另一个BPINN
- B07: 训练第三个BPINN

最后对比各个电池上的结果。
"""

import os
import sys
import torch
import torch.optim as optim
import numpy as np
import random

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader_per_battery import create_data_loaders_for_battery
from src.model import BPINN, BPINNLoss, SecondaryTrainingLoss
from src.train import train_model, secondary_training, evaluate, save_model
from src.utils import ensure_dir, print_metrics


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_single_battery(battery_name, data_dict, config, device):
    """
    训练单个电池的BPINN模型。

    Args:
        battery_name: 电池名称（如'B05'）
        data_dict: 电池数据字典
        config: 配置参数
        device: 设备

    Returns:
        Dictionary包含所有结果
    """
    print("\n" + "=" * 70)
    print(f"Training {battery_name}")
    print("=" * 70)

    # 创建DataLoaders
    train_loader, test_loader = create_data_loaders_for_battery(
        data_dict, batch_size=config['batch_size']
    )

    # 创建模型
    model = BPINN(
        input_size=6,
        hidden_sizes=config['hidden_sizes'],
        dropout_rate=config['dropout_rate']
    ).to(device)

    print(f"\nModel architecture:")
    print(f"  Input: 6 features")
    print(f"  Hidden layers: {config['hidden_sizes']}")
    print(f"  Output: 1 (SOH)")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")

    # ======================================================================
    # Phase 1: Primary Training (BPINN-1)
    # ======================================================================
    print("\n" + "-" * 70)
    print(f"Phase 1: Primary Training (BPINN-1) for {battery_name}")
    print("-" * 70)

    criterion = BPINNLoss(lambda_physics=config['lambda_physics'])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)

    history_phase1 = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config['num_epochs'],
        device=device,
        verbose=True
    )

    # Evaluate BPINN-1
    results_phase1 = evaluate(model, test_loader, device)
    print(f"\n{battery_name} - BPINN-1 Results:")
    print_metrics(results_phase1, f"{battery_name} BPINN-1")

    # Save Phase 1 model
    phase1_path = os.path.join(
        config['results_dir'],
        f'{battery_name}_bpinn_phase1.pth'
    )
    save_model(model, phase1_path)

    # ======================================================================
    # Phase 2: Secondary Training (BPINN-2)
    # ======================================================================
    print("\n" + "-" * 70)
    print(f"Phase 2: Secondary Training (BPINN-2) for {battery_name}")
    print("-" * 70)

    criterion_secondary = SecondaryTrainingLoss(
        lambda_train_physics=config['lambda_physics'],
        lambda_test_physics=config['lambda_test_physics']
    )

    optimizer_secondary = optim.Adam(
        model.parameters(),
        lr=config['secondary_learning_rate']
    )

    history_phase2 = secondary_training(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion_secondary=criterion_secondary,
        optimizer=optimizer_secondary,
        num_iterations=config['num_secondary_iterations'],
        device=device,
        verbose=True
    )

    # Evaluate BPINN-2
    results_phase2 = evaluate(model, test_loader, device)
    print(f"\n{battery_name} - BPINN-2 Results:")
    print_metrics(results_phase2, f"{battery_name} BPINN-2")

    # Save final model
    final_path = os.path.join(
        config['results_dir'],
        f'{battery_name}_bpinn_final.pth'
    )
    save_model(model, final_path)

    # Return all results
    return {
        'battery_name': battery_name,
        'model': model,
        'history_phase1': history_phase1,
        'history_phase2': history_phase2,
        'results_phase1': results_phase1,
        'results_phase2': results_phase2,
        'test_cycles': data_dict['test_cycles'],
        'train_size': data_dict['n_train'],
        'test_size': data_dict['n_test']
    }


def main():
    """Main training function."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train BPINN for a single battery')
    parser.add_argument('--battery', type=str, default='B05',
                        choices=['B05', 'B06', 'B07'],
                        help='Battery to train (default: B05)')
    args = parser.parse_args()

    # Set random seed
    set_seed(42)

    # Configuration
    config = {
        # Model architecture
        'hidden_sizes': [10, 10, 10],
        'dropout_rate': 0.0,

        # Training parameters
        'num_epochs': 2500,
        'learning_rate': 0.001,
        'lambda_physics': 0.01,
        'batch_size': 64,

        # Secondary training
        'num_secondary_iterations': 800,
        'secondary_learning_rate': 0.001,
        'lambda_test_physics': 0.01,

        # Data
        'train_ratio': 0.75,  # 75% for training, 25% for testing

        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        # Paths
        'results_dir': 'results_per_battery',
    }

    # Create results directory
    ensure_dir(config['results_dir'])

    # Device
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ======================================================================
    # Load Data for Selected Battery
    # ======================================================================
    print("\n" + "=" * 70)
    print(f"Loading Battery Data: {args.battery}")
    print("=" * 70)

    # Map battery name to file path
    battery_file_map = {
        'B05': 'data/B05_IC.csv',
        'B06': 'data/B06_IC.csv',
        'B07': 'data/B07_IC.csv'
    }

    file_path = battery_file_map[args.battery]

    # Load single battery data
    from src.data_loader_per_battery import load_single_battery_data
    data_dict = load_single_battery_data(file_path, train_ratio=config['train_ratio'])

    print(f"\nLoaded {args.battery}:")
    print(f"  Train: {data_dict['n_train']} samples")
    print(f"  Test:  {data_dict['n_test']} samples")
    print(f"  Train SOH range: {data_dict['train_soh'].min():.3f} - {data_dict['train_soh'].max():.3f}")
    print(f"  Test SOH range:  {data_dict['test_soh'].min():.3f} - {data_dict['test_soh'].max():.3f}")

    # ======================================================================
    # Train the Selected Battery
    # ======================================================================
    results = train_single_battery(
        args.battery,
        data_dict,
        config,
        device
    )

    # ======================================================================
    # Summary
    # ======================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - Summary")
    print("=" * 70)

    mae1 = results['results_phase1']['mae']
    rmse1 = results['results_phase1']['rmse']
    mae2 = results['results_phase2']['mae']
    rmse2 = results['results_phase2']['rmse']

    print(f"\n{args.battery} Performance:")
    print("-" * 70)
    print(f"BPINN-1: MAE={mae1*100:.4f}%, RMSE={rmse1*100:.4f}%")
    print(f"BPINN-2: MAE={mae2*100:.4f}%, RMSE={rmse2*100:.4f}%")

    print("\n" + "=" * 70)
    print(f"Results saved to: {config['results_dir']}/")
    print(f"  - {args.battery}_bpinn_phase1.pth")
    print(f"  - {args.battery}_bpinn_final.pth")
    print("=" * 70)

    # Save single battery results
    import pickle
    result_file = os.path.join(config['results_dir'], f'{args.battery}_results.pkl')
    with open(result_file, 'wb') as f:
        pickle.dump({args.battery: results}, f)
    print(f"\nResults saved to: {result_file}")

    return results


if __name__ == "__main__":
    results = main()
