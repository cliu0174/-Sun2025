"""
Main training script for BPINN model.
Implements the complete training pipeline including primary training and secondary training.
"""

import os
import sys
import torch
import torch.optim as optim
import numpy as np
import random

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_battery_data, create_data_loaders
from src.model import BPINN, BPINNLoss, SecondaryTrainingLoss
from src.train import train_model, secondary_training, evaluate, save_model, load_model
from src.utils import (plot_training_history, plot_predictions,
                   plot_secondary_training_history, plot_comparison,
                   plot_soh_predictions_over_cycles, ensure_dir, print_metrics)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training pipeline."""

    print("="*70)
    print("BPINN for Battery SOH Estimation")
    print("Based on: Sun et al. (2025)")
    print("="*70)

    # Set random seed
    set_seed(42)

    # Configuration
    config = {
        # Data
        'data_files': [
            'data/B05_IC.csv',
            'data/B06_IC.csv',
            'data/B07_IC.csv'
        ],
        'train_ratio': 0.6,  # As per paper
        'batch_size': 32,

        # Model
        'input_size': 6,
        'hidden_sizes': [10, 10, 10],  # Paper uses 3 hidden layers with 10 neurons each
        'dropout_rate': 0.0,

        # Training
        'num_epochs': 2000,  # Paper uses 2000 epochs
        'learning_rate': 0.001,
        'lambda_physics': 0.01,  # ω_1 in paper

        # Secondary training
        'num_secondary_iterations': 400,  # Paper uses 400 iterations
        'secondary_learning_rate': 0.002,
        'lambda_test_physics': 0.01,  # ω_2 in paper

        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        # Paths
        'results_dir': 'results',
        'model_save_path': 'results/bpinn_model.pth'
    }

    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Create results directory
    ensure_dir(config['results_dir'])

    # Load data
    print("\n" + "="*70)
    print("Loading Data...")
    print("="*70)

    data = load_battery_data(config['data_files'], train_ratio=config['train_ratio'])

    print(f"Training samples: {len(data['train_features'])}")
    print(f"Testing samples: {len(data['test_features'])}")
    print(f"Features: {data['feature_columns']}")

    # Create data loaders
    train_loader, test_loader = create_data_loaders(data, batch_size=config['batch_size'])

    # Initialize model
    print("\n" + "="*70)
    print("Initializing Model...")
    print("="*70)

    device = torch.device(config['device'])
    print(f"Using device: {device}")

    model = BPINN(
        input_size=config['input_size'],
        hidden_sizes=config['hidden_sizes'],
        dropout_rate=config['dropout_rate']
    ).to(device)

    print(f"\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # ======================================================================
    # Phase 1: Primary Training (BPINN-1)
    # ======================================================================
    print("\n" + "="*70)
    print("Phase 1: Primary Training (BPINN-1)")
    print("="*70)

    criterion = BPINNLoss(
        lambda_physics=config['lambda_physics']
    )
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
    print("\n" + "="*70)
    print("Evaluating BPINN-1 (After Primary Training)")
    print("="*70)
    print(f"Best model was from epoch {history_phase1['best_epoch']}")
    print(f"Best MAE: {history_phase1['best_mae']:.6f}, Best RMSE: {history_phase1['best_rmse']:.6f}")

    results_phase1 = evaluate(model, test_loader, device)
    print_metrics(results_phase1, "BPINN-1")

    # Plot training history
    plot_training_history(
        history_phase1,
        save_path=os.path.join(config['results_dir'], 'training_history_phase1.png')
    )

    # Plot predictions
    plot_predictions(
        results_phase1['predictions'],
        results_phase1['targets'],
        title="BPINN-1 (Primary Training)",
        save_path=os.path.join(config['results_dir'], 'predictions_phase1.png')
    )

    # Save model after phase 1
    save_model(model, config['model_save_path'].replace('.pth', '_phase1.pth'))

    # ======================================================================
    # Phase 2: Secondary Training (BPINN-2)
    # ======================================================================
    print("\n" + "="*70)
    print("Phase 2: Secondary Training (BPINN-2)")
    print("="*70)
    print("Performing online optimization on test data...")

    criterion_secondary = SecondaryTrainingLoss(
        lambda_train_physics=config['lambda_physics'],
        lambda_test_physics=config['lambda_test_physics']
    )

    # Use a fresh optimizer for secondary training
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
    print("\n" + "="*70)
    print("Evaluating BPINN-2 (After Secondary Training)")
    print("="*70)
    print(f"Best model was from iteration {history_phase2['best_iteration']}")
    print(f"Best MAE: {history_phase2['best_mae']:.6f}, Best RMSE: {history_phase2['best_rmse']:.6f}")

    results_phase2 = evaluate(model, test_loader, device)
    print_metrics(results_phase2, "BPINN-2")

    # Plot secondary training history
    plot_secondary_training_history(
        history_phase2,
        save_path=os.path.join(config['results_dir'], 'secondary_training_history.png')
    )

    # Plot predictions
    plot_predictions(
        results_phase2['predictions'],
        results_phase2['targets'],
        title="BPINN-2 (After Secondary Training)",
        save_path=os.path.join(config['results_dir'], 'predictions_phase2.png')
    )

    # Save final model
    save_model(model, config['model_save_path'])

    # ======================================================================
    # Comparison
    # ======================================================================
    print("\n" + "="*70)
    print("Model Comparison")
    print("="*70)

    comparison_results = {
        'BPINN-1': (results_phase1['predictions'], results_phase1['targets']),
        'BPINN-2': (results_phase2['predictions'], results_phase2['targets'])
    }

    plot_comparison(
        comparison_results,
        save_path=os.path.join(config['results_dir'], 'model_comparison.png')
    )

    # Plot SOH predictions over cycles (similar to paper figure)
    print("\nGenerating SOH prediction curves...")
    plot_soh_predictions_over_cycles(
        results_phase1,
        results_phase2,
        save_path=os.path.join(config['results_dir'], 'soh_prediction_curves.png')
    )

    # Print final summary
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print("\nFinal Results:")
    print(f"BPINN-1 - MAE: {results_phase1['mae']:.4f}, RMSE: {results_phase1['rmse']:.4f}")
    print(f"BPINN-2 - MAE: {results_phase2['mae']:.4f}, RMSE: {results_phase2['rmse']:.4f}")
    print(f"\nImprovement from secondary training:")
    print(f"  MAE:  {(results_phase1['mae'] - results_phase2['mae']):.4f} "
          f"({(1 - results_phase2['mae']/results_phase1['mae'])*100:.2f}% reduction)")
    print(f"  RMSE: {(results_phase1['rmse'] - results_phase2['rmse']):.4f} "
          f"({(1 - results_phase2['rmse']/results_phase1['rmse'])*100:.2f}% reduction)")

    print(f"\nResults and figures saved to: {config['results_dir']}/")
    print("="*70)


if __name__ == "__main__":
    main()
