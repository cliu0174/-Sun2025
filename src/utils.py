"""
Utility functions for BPINN training and evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def plot_training_history(history, save_path=None):
    """
    Plot training history with best epoch marked.

    Args:
        history: Dictionary with training history (includes best_epoch, best_mae, best_rmse)
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get best epoch index (convert from 1-indexed to 0-indexed)
    best_epoch_idx = history.get('best_epoch', 0) - 1

    # Plot 1: Total training loss
    axes[0, 0].plot(history['train_loss'], label='Total Loss', linewidth=2)
    if best_epoch_idx >= 0:
        axes[0, 0].axvline(x=best_epoch_idx, color='red', linestyle='--', alpha=0.7,
                           label=f'Best Epoch ({history["best_epoch"]})')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Data loss and physics loss
    axes[0, 1].plot(history['train_data_loss'], label='Data Loss', linewidth=2)
    axes[0, 1].plot(history['train_physics_loss'], label='Physics Loss', linewidth=2)
    if best_epoch_idx >= 0:
        axes[0, 1].axvline(x=best_epoch_idx, color='red', linestyle='--', alpha=0.7,
                           label=f'Best Epoch ({history["best_epoch"]})')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss Components')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Test MAE
    axes[1, 0].plot(history['test_mae'], label='Test MAE', color='green', linewidth=2)
    if best_epoch_idx >= 0:
        axes[1, 0].scatter([best_epoch_idx], [history['best_mae']],
                          color='red', s=100, zorder=5, label=f'Best: {history["best_mae"]:.6f}')
        axes[1, 0].axvline(x=best_epoch_idx, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('Test MAE over Epochs')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Test RMSE
    axes[1, 1].plot(history['test_rmse'], label='Test RMSE', color='red', linewidth=2)
    if best_epoch_idx >= 0:
        axes[1, 1].scatter([best_epoch_idx], [history['best_rmse']],
                          color='darkred', s=100, zorder=5, label=f'Best: {history["best_rmse"]:.6f}')
        axes[1, 1].axvline(x=best_epoch_idx, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].set_title('Test RMSE over Epochs')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    plt.show()


def plot_predictions(predictions, targets, title="SOH Predictions", save_path=None):
    """
    Plot predicted vs actual SOH values.

    Args:
        predictions: Predicted SOH values
        targets: Actual SOH values
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Predicted vs Actual
    axes[0].scatter(targets, predictions, alpha=0.6, s=30)
    axes[0].plot([targets.min(), targets.max()],
                 [targets.min(), targets.max()],
                 'r--', linewidth=2, label='Ideal')
    axes[0].set_xlabel('Actual SOH')
    axes[0].set_ylabel('Predicted SOH')
    axes[0].set_title(f'{title}: Predicted vs Actual')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Prediction errors
    errors = predictions - targets
    axes[1].plot(errors, alpha=0.6, linewidth=1)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].fill_between(range(len(errors)), errors, alpha=0.3)
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Prediction Error')
    axes[1].set_title(f'{title}: Prediction Errors')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions plot saved to {save_path}")

    plt.show()


def plot_secondary_training_history(history, save_path=None):
    """
    Plot secondary training (online optimization) history with best iteration marked.

    Args:
        history: Dictionary with secondary training history (includes best_iteration, best_mae, best_rmse)
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Get best iteration index (convert from 1-indexed to 0-indexed)
    best_iter_idx = history.get('best_iteration', 0) - 1

    # Plot 1: Total loss
    axes[0, 0].plot(history['total_loss'], linewidth=2, color='blue')
    if best_iter_idx >= 0:
        axes[0, 0].axvline(x=best_iter_idx, color='red', linestyle='--', alpha=0.7,
                           label=f'Best Iter ({history["best_iteration"]})')
        axes[0, 0].legend()
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Secondary Training: Total Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Train MSE
    axes[0, 1].plot(history['train_mse'], linewidth=2, color='green')
    if best_iter_idx >= 0:
        axes[0, 1].axvline(x=best_iter_idx, color='red', linestyle='--', alpha=0.7,
                           label=f'Best Iter ({history["best_iteration"]})')
        axes[0, 1].legend()
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Secondary Training: Train MSE')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Test MAE
    if 'test_mae' in history and len(history['test_mae']) > 0:
        axes[0, 2].plot(history['test_mae'], linewidth=2, color='purple')
        if best_iter_idx >= 0:
            axes[0, 2].scatter([best_iter_idx], [history['best_mae']],
                              color='red', s=100, zorder=5, label=f'Best: {history["best_mae"]:.6f}')
            axes[0, 2].axvline(x=best_iter_idx, color='red', linestyle='--', alpha=0.7)
            axes[0, 2].legend()
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('MAE')
        axes[0, 2].set_title('Secondary Training: Test MAE')
        axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Train physics loss
    axes[1, 0].plot(history['train_physics_loss'], linewidth=2, color='orange')
    if best_iter_idx >= 0:
        axes[1, 0].axvline(x=best_iter_idx, color='red', linestyle='--', alpha=0.7,
                           label=f'Best Iter ({history["best_iteration"]})')
        axes[1, 0].legend()
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Secondary Training: Train Physics Loss')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Test physics loss
    axes[1, 1].plot(history['test_physics_loss'], linewidth=2, color='red')
    if best_iter_idx >= 0:
        axes[1, 1].axvline(x=best_iter_idx, color='red', linestyle='--', alpha=0.7,
                           label=f'Best Iter ({history["best_iteration"]})')
        axes[1, 1].legend()
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Secondary Training: Test Physics Loss')
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Test RMSE
    if 'test_rmse' in history and len(history['test_rmse']) > 0:
        axes[1, 2].plot(history['test_rmse'], linewidth=2, color='darkred')
        if best_iter_idx >= 0:
            axes[1, 2].scatter([best_iter_idx], [history['best_rmse']],
                              color='red', s=100, zorder=5, label=f'Best: {history["best_rmse"]:.6f}')
            axes[1, 2].axvline(x=best_iter_idx, color='red', linestyle='--', alpha=0.7)
            axes[1, 2].legend()
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('RMSE')
        axes[1, 2].set_title('Secondary Training: Test RMSE')
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Secondary training history plot saved to {save_path}")

    plt.show()


def plot_comparison(results_dict, save_path=None):
    """
    Plot comparison between different model variants (e.g., BPINN-1 vs BPINN-2 vs FNN).

    Args:
        results_dict: Dictionary with model names as keys and (predictions, targets) tuples as values
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['blue', 'red', 'green', 'purple', 'orange']

    # Plot 1: All predictions on same plot
    for i, (model_name, (predictions, targets)) in enumerate(results_dict.items()):
        axes[0].scatter(targets, predictions, alpha=0.5, s=20,
                       color=colors[i % len(colors)], label=model_name)

    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Ideal')
    axes[0].set_xlabel('Actual SOH')
    axes[0].set_ylabel('Predicted SOH')
    axes[0].set_title('Model Comparison: Predictions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: MAE and RMSE comparison
    model_names = list(results_dict.keys())
    mae_values = []
    rmse_values = []

    for model_name, (predictions, targets) in results_dict.items():
        mae = np.mean(np.abs(predictions - targets)) * 100
        rmse = np.sqrt(np.mean((predictions - targets) ** 2)) * 100
        mae_values.append(mae)
        rmse_values.append(rmse)

    x = np.arange(len(model_names))
    width = 0.35

    axes[1].bar(x - width/2, mae_values, width, label='MAE (%)', alpha=0.8)
    axes[1].bar(x + width/2, rmse_values, width, label='RMSE (%)', alpha=0.8)
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Error (%)')
    axes[1].set_title('Model Comparison: Errors')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")

    plt.show()


def plot_soh_predictions_over_cycles(results_bpinn1, results_bpinn2, save_path=None):
    """
    Plot SOH predictions over sample indices (test set).
    Shows BPINN-1, BPINN-2, and Real SOH values as curves.

    Note: Since test data is mixed from multiple batteries (B05, B06, B07),
    the x-axis represents sample index in the test set, not actual cycle numbers.
    For per-battery analysis, use separate evaluation.

    Args:
        results_bpinn1: Dictionary with 'predictions' and 'targets' from BPINN-1
        results_bpinn2: Dictionary with 'predictions' and 'targets' from BPINN-2
        save_path: Path to save the figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get data
    real_soh = results_bpinn1['targets']  # Same for both
    pred_bpinn1 = results_bpinn1['predictions']
    pred_bpinn2 = results_bpinn2['predictions']

    # Sort by real SOH to create a smooth curve (typical degradation pattern)
    sorted_indices = np.argsort(real_soh)[::-1]  # Sort descending (degradation)
    real_soh_sorted = real_soh[sorted_indices]
    pred_bpinn1_sorted = pred_bpinn1[sorted_indices]
    pred_bpinn2_sorted = pred_bpinn2[sorted_indices]

    # Create sample indices (x-axis)
    samples = np.arange(len(real_soh_sorted))

    # Plot lines with smoothing
    ax.plot(samples, real_soh_sorted, 'k-', linewidth=2.5, label='Real',
            marker='o', markersize=4, markevery=max(1, len(samples)//15), alpha=0.8)
    ax.plot(samples, pred_bpinn1_sorted, '-', color='#1f77b4', linewidth=2,
            label='BPINN-1', marker='s', markersize=4,
            markevery=max(1, len(samples)//15), alpha=0.8)
    ax.plot(samples, pred_bpinn2_sorted, '-', color='#ff7f0e', linewidth=2,
            label='BPINN-2', marker='^', markersize=4,
            markevery=max(1, len(samples)//15), alpha=0.8)

    # Formatting
    ax.set_xlabel('Test Sample Index (Sorted by SOH)', fontsize=12, fontweight='bold')
    ax.set_ylabel('SOH', fontsize=12, fontweight='bold')
    ax.set_title('SOH Prediction Comparison on Test Set', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set y-axis range similar to paper
    y_min = min(real_soh_sorted.min(), pred_bpinn1_sorted.min(), pred_bpinn2_sorted.min())
    y_max = max(real_soh_sorted.max(), pred_bpinn1_sorted.max(), pred_bpinn2_sorted.max())
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

    # Add annotation
    ax.text(0.02, 0.02,
            'Note: Data from 3 batteries (B05, B06, B07) test sets\nSorted by SOH for visualization',
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SOH prediction curves saved to {save_path}")
        plt.close()  # Close to avoid blocking
    else:
        plt.show()


def plot_soh_per_battery(test_loader, model_bpinn1, model_bpinn2, device, save_path=None):
    """
    Plot SOH predictions for each battery separately (more accurate visualization).

    Args:
        test_loader: Test data loader
        model_bpinn1: BPINN-1 model
        model_bpinn2: BPINN-2 model
        device: Device to use
        save_path: Path to save the figure (optional)
    """
    import torch
    from train import evaluate

    # This function would need battery IDs in the data to work properly
    # For now, we'll use the sorted version above
    # TODO: Add battery_id column to dataset for proper per-battery visualization
    print("Note: Per-battery visualization requires battery IDs in the dataset")
    print("Using sorted aggregate visualization instead")


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def print_metrics(metrics, model_name="Model"):
    """
    Print evaluation metrics in a formatted way.

    Args:
        metrics: Dictionary with 'mae' and 'rmse' keys
        model_name: Name of the model
    """
    print(f"\n{'='*50}")
    print(f"{model_name} Performance:")
    print(f"{'='*50}")
    print(f"MAE:  {metrics['mae']:.4f} ({metrics['mae']*100:.2f}%)")
    print(f"RMSE: {metrics['rmse']:.4f} ({metrics['rmse']*100:.2f}%)")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    print("Utility functions loaded successfully!")
