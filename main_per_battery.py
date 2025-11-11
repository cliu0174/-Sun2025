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

from src.data_loader_per_battery import load_all_batteries, create_data_loaders_for_battery
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
    # Set random seed
    set_seed(42)

    # Configuration
    config = {
        # Model architecture
        'hidden_sizes': [10, 10, 10],
        'dropout_rate': 0.0,

        # Training parameters
        'num_epochs': 2000,
        'learning_rate': 0.001,
        'lambda_physics': 0.01,
        'batch_size': 64,

        # Secondary training
        'num_secondary_iterations': 400,
        'secondary_learning_rate': 0.002,
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
    # Load Data for All Batteries
    # ======================================================================
    print("\n" + "=" * 70)
    print("Loading Battery Data (Per-Battery Mode)")
    print("=" * 70)

    file_paths = [
        'data/B05_IC.csv',
        'data/B06_IC.csv',
        'data/B07_IC.csv'
    ]

    all_battery_data = load_all_batteries(
        file_paths,
        train_ratio=config['train_ratio']
    )

    # ======================================================================
    # Train Each Battery Separately
    # ======================================================================
    all_results = {}

    for battery_name, data_dict in all_battery_data.items():
        results = train_single_battery(
            battery_name,
            data_dict,
            config,
            device
        )
        all_results[battery_name] = results

    # ======================================================================
    # Summary
    # ======================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - Summary")
    print("=" * 70)

    print("\n{:<10} {:<15} {:<15} {:<15} {:<15}".format(
        "Battery", "BPINN-1 MAE", "BPINN-1 RMSE", "BPINN-2 MAE", "BPINN-2 RMSE"
    ))
    print("-" * 70)

    for battery_name, results in all_results.items():
        mae1 = results['results_phase1']['mae']
        rmse1 = results['results_phase1']['rmse']
        mae2 = results['results_phase2']['mae']
        rmse2 = results['results_phase2']['rmse']

        print("{:<10} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f}".format(
            battery_name,
            mae1 * 100,  # Convert to percentage
            rmse1 * 100,
            mae2 * 100,
            rmse2 * 100
        ))

    # Calculate average
    avg_mae1 = np.mean([r['results_phase1']['mae'] for r in all_results.values()])
    avg_rmse1 = np.mean([r['results_phase1']['rmse'] for r in all_results.values()])
    avg_mae2 = np.mean([r['results_phase2']['mae'] for r in all_results.values()])
    avg_rmse2 = np.mean([r['results_phase2']['rmse'] for r in all_results.values()])

    print("-" * 70)
    print("{:<10} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f}".format(
        "Average",
        avg_mae1 * 100,
        avg_rmse1 * 100,
        avg_mae2 * 100,
        avg_rmse2 * 100
    ))

    print("\n" + "=" * 70)
    print(f"All results saved to: {config['results_dir']}/")
    print("=" * 70)

    # Save results summary
    import pickle
    with open(os.path.join(config['results_dir'], 'all_results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults summary saved to: {config['results_dir']}/all_results.pkl")

    return all_results


if __name__ == "__main__":
    all_results = main()
