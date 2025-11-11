"""
Evaluation script for trained BPINN model.
Can be used to evaluate a saved model on test data.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_battery_data, create_data_loaders
from model import BPINN
from train import evaluate, load_model
from utils import plot_predictions, print_metrics


def evaluate_model(model_path, data_files, train_ratio=0.6, batch_size=32):
    """
    Evaluate a trained BPINN model.

    Args:
        model_path: Path to saved model checkpoint
        data_files: List of data file paths
        train_ratio: Train/test split ratio
        batch_size: Batch size for evaluation
    """
    print("="*70)
    print("BPINN Model Evaluation")
    print("="*70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    data = load_battery_data(data_files, train_ratio=train_ratio)
    train_loader, test_loader = create_data_loaders(data, batch_size=batch_size)

    print(f"Training samples: {len(data['train_features'])}")
    print(f"Testing samples: {len(data['test_features'])}")

    # Load model
    print(f"\nLoading model from: {model_path}")
    model = BPINN(input_size=6, hidden_sizes=[10, 10, 10]).to(device)
    model = load_model(model, model_path, device)

    # Evaluate on training set
    print("\n" + "="*70)
    print("Evaluating on Training Set")
    print("="*70)

    train_results = evaluate(model, train_loader, device)
    print_metrics(train_results, "Training Set")

    # Evaluate on test set
    print("\n" + "="*70)
    print("Evaluating on Test Set")
    print("="*70)

    test_results = evaluate(model, test_loader, device)
    print_metrics(test_results, "Test Set")

    # Plot predictions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training predictions
    axes[0].scatter(train_results['targets'], train_results['predictions'],
                   alpha=0.6, s=30, label='Predictions')
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Ideal')
    axes[0].set_xlabel('Actual SOH')
    axes[0].set_ylabel('Predicted SOH')
    axes[0].set_title(f'Training Set (MAE: {train_results["mae"]:.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Test predictions
    axes[1].scatter(test_results['targets'], test_results['predictions'],
                   alpha=0.6, s=30, label='Predictions', color='orange')
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Ideal')
    axes[1].set_xlabel('Actual SOH')
    axes[1].set_ylabel('Predicted SOH')
    axes[1].set_title(f'Test Set (MAE: {test_results["mae"]:.4f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/evaluation_results.png', dpi=300, bbox_inches='tight')
    print("\nEvaluation plot saved to: results/evaluation_results.png")
    plt.show()

    # Detailed error analysis
    print("\n" + "="*70)
    print("Detailed Error Analysis")
    print("="*70)

    test_errors = test_results['predictions'] - test_results['targets']

    print("\nTest Set Error Statistics:")
    print(f"  Mean Error: {np.mean(test_errors):.6f}")
    print(f"  Std Error:  {np.std(test_errors):.6f}")
    print(f"  Max Error:  {np.max(np.abs(test_errors)):.6f}")
    print(f"  Min Error:  {np.min(np.abs(test_errors)):.6f}")
    print(f"  Median Error: {np.median(np.abs(test_errors)):.6f}")

    # Error distribution
    print(f"\nError Distribution:")
    print(f"  Errors < 1%:  {np.sum(np.abs(test_errors) < 0.01) / len(test_errors) * 100:.2f}%")
    print(f"  Errors < 2%:  {np.sum(np.abs(test_errors) < 0.02) / len(test_errors) * 100:.2f}%")
    print(f"  Errors < 5%:  {np.sum(np.abs(test_errors) < 0.05) / len(test_errors) * 100:.2f}%")
    print(f"  Errors < 10%: {np.sum(np.abs(test_errors) < 0.10) / len(test_errors) * 100:.2f}%")

    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)

    return {
        'train_results': train_results,
        'test_results': test_results
    }


def main():
    """Main evaluation function."""

    # Configuration
    model_path = 'results/bpinn_model.pth'
    data_files = [
        'data/B05_IC.csv',
        'data/B06_IC.csv',
        'data/B07_IC.csv'
    ]

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using: python main.py")
        return

    # Evaluate model
    results = evaluate_model(
        model_path=model_path,
        data_files=data_files,
        train_ratio=0.6,
        batch_size=32
    )


if __name__ == "__main__":
    main()
