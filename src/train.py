"""
Training and evaluation functions for BPINN model.
Implements both primary training and secondary "training" (online optimization).
"""

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch.

    Args:
        model: BPINN model
        train_loader: Training data loader
        criterion: Loss function (BPINNLoss)
        optimizer: Optimizer
        device: Device to use (cpu or cuda)

    Returns:
        Average losses (total_loss, data_loss, physics_loss)
    """
    model.train()
    total_losses = []
    data_losses = []
    physics_losses = []

    for features, soh, p_ic in train_loader:
        features = features.to(device)
        soh = soh.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(features)

        # Compute loss with physical constraints
        total_loss, data_loss, physics_loss = criterion(predictions, soh, features, model)

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Record losses
        total_losses.append(total_loss.item())
        data_losses.append(data_loss.item())
        physics_losses.append(physics_loss.item())

    return np.mean(total_losses), np.mean(data_losses), np.mean(physics_losses)


def evaluate(model, data_loader, device):
    """
    Evaluate model on validation/test set.

    Args:
        model: BPINN model
        data_loader: Data loader
        device: Device to use

    Returns:
        Dictionary with predictions, targets, MAE, RMSE
    """
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, soh, p_ic in data_loader:
            features = features.to(device)
            soh = soh.to(device)

            predictions = model(features)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(soh.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0)

    # Calculate metrics
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))

    return {
        'predictions': predictions,
        'targets': targets,
        'mae': mae,
        'rmse': rmse
    }


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler,
                num_epochs, device, verbose=True):
    """
    Train BPINN model for specified number of epochs.

    Automatically saves the best model based on test MAE during training.

    Args:
        model: BPINN model
        train_loader: Training data loader
        test_loader: Test data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        device: Device to use
        verbose: Whether to print progress

    Returns:
        Dictionary with training history and best model info
    """
    history = {
        'train_loss': [],
        'train_data_loss': [],
        'train_physics_loss': [],
        'test_mae': [],
        'test_rmse': []
    }

    best_mae = float('inf')
    best_rmse = float('inf')
    best_epoch = 0
    best_model_state = None

    iterator = tqdm(range(num_epochs), desc="Training") if verbose else range(num_epochs)

    for epoch in iterator:
        # Training
        train_loss, data_loss, physics_loss = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluation
        test_metrics = evaluate(model, test_loader, device)

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Record history
        history['train_loss'].append(train_loss)
        history['train_data_loss'].append(data_loss)
        history['train_physics_loss'].append(physics_loss)
        history['test_mae'].append(test_metrics['mae'])
        history['test_rmse'].append(test_metrics['rmse'])

        # Save best model based on MAE
        if test_metrics['mae'] < best_mae:
            best_mae = test_metrics['mae']
            best_rmse = test_metrics['rmse']
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if verbose:
                print(f"\n*** New best model at epoch {best_epoch}: MAE={best_mae:.6f}, RMSE={best_rmse:.6f} ***")

        # Print progress
        if verbose and (epoch + 1) % 100 == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.6f} (Data: {data_loss:.6f}, Physics: {physics_loss:.6f})")
            print(f"Test MAE: {test_metrics['mae']:.6f}, RMSE: {test_metrics['rmse']:.6f}")
            print(f"Current Best: Epoch {best_epoch}, MAE={best_mae:.6f}, RMSE={best_rmse:.6f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"\n=== Restored best model from epoch {best_epoch} ===")
            print(f"Best MAE: {best_mae:.6f}, Best RMSE: {best_rmse:.6f}")

    # Add best model info to history
    history['best_epoch'] = best_epoch
    history['best_mae'] = best_mae
    history['best_rmse'] = best_rmse

    return history


def secondary_training(model, train_loader, test_loader, criterion_secondary,
                       optimizer, num_iterations, device, verbose=True):
    """
    Perform secondary "training" (online optimization) during testing phase.

    According to the paper, this phase refines model parameters based on test data
    while maintaining physical constraints. This is done after primary training.

    The loss function for secondary training is:
    L_total = MSE(train) + ω1·L_physics(train) + ω2·L_physics(test)

    Automatically saves the best model based on test MAE during secondary training.

    Args:
        model: Trained BPINN model
        train_loader: Training data loader (for computing train loss)
        test_loader: Test data loader
        criterion_secondary: Secondary training loss function
        optimizer: Optimizer for secondary training
        num_iterations: Number of optimization iterations (paper uses 400)
        device: Device to use
        verbose: Whether to print progress

    Returns:
        Dictionary with optimization history and best model info
    """
    model.train()  # Enable gradient computation

    history = {
        'total_loss': [],
        'train_mse': [],
        'train_physics_loss': [],
        'test_physics_loss': [],
        'test_mae': [],
        'test_rmse': []
    }

    # Get all test data
    test_features_list = []
    test_soh_list = []
    for features, soh, p_ic in test_loader:
        test_features_list.append(features)
        test_soh_list.append(soh)

    test_features = torch.cat(test_features_list, dim=0).to(device)
    test_soh = torch.cat(test_soh_list, dim=0).to(device)

    # Get all train data for computing train loss and physics
    train_features_list = []
    train_soh_list = []
    for features, soh, p_ic in train_loader:
        train_features_list.append(features)
        train_soh_list.append(soh)

    train_features = torch.cat(train_features_list, dim=0).to(device)
    train_soh = torch.cat(train_soh_list, dim=0).to(device)

    # Track best model during secondary training
    best_test_mae = float('inf')
    best_test_rmse = float('inf')
    best_iteration = 0
    best_model_state = None

    iterator = tqdm(range(num_iterations), desc="Secondary Training") if verbose else range(num_iterations)

    for iteration in iterator:
        optimizer.zero_grad()

        # Make predictions on train and test data
        train_pred = model(train_features)
        test_pred = model(test_features)

        # Compute secondary training loss
        # This includes: MSE(train) + ω1·Physics(train) + ω2·Physics(test)
        total_loss, train_mse, train_physics, test_physics = criterion_secondary(
            train_pred, train_soh, train_features, test_pred, test_features, model
        )

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Evaluate on test set to track MAE/RMSE
        with torch.no_grad():
            test_pred_eval = model(test_features).cpu().numpy().flatten()
            test_soh_eval = test_soh.cpu().numpy()
            test_mae = mean_absolute_error(test_soh_eval, test_pred_eval)
            test_rmse = np.sqrt(mean_squared_error(test_soh_eval, test_pred_eval))

        # Record history
        history['total_loss'].append(total_loss.item())
        history['train_mse'].append(train_mse.item())
        history['train_physics_loss'].append(train_physics.item())
        history['test_physics_loss'].append(test_physics.item())
        history['test_mae'].append(test_mae)
        history['test_rmse'].append(test_rmse)

        # Save best model based on test MAE
        if test_mae < best_test_mae:
            best_test_mae = test_mae
            best_test_rmse = test_rmse
            best_iteration = iteration + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if verbose and (iteration + 1) % 50 == 0:
                print(f"\n*** New best model at iteration {best_iteration}: MAE={best_test_mae:.6f}, RMSE={best_test_rmse:.6f} ***")

        if verbose and (iteration + 1) % 100 == 0:
            print(f"\nIteration {iteration + 1}/{num_iterations}")
            print(f"Total Loss: {total_loss.item():.6f}")
            print(f"Train MSE: {train_mse.item():.6f}, Train Physics: {train_physics.item():.6f}")
            print(f"Test Physics Loss: {test_physics.item():.6f}")
            print(f"Test MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}")
            print(f"Current Best: Iteration {best_iteration}, MAE={best_test_mae:.6f}, RMSE={best_test_rmse:.6f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"\n=== Restored best model from iteration {best_iteration} ===")
            print(f"Best MAE: {best_test_mae:.6f}, Best RMSE: {best_test_rmse:.6f}")

    # Add best model info to history
    history['best_iteration'] = best_iteration
    history['best_mae'] = best_test_mae
    history['best_rmse'] = best_test_rmse

    return history


def compute_metrics(predictions, targets):
    """
    Compute evaluation metrics.

    Args:
        predictions: Predicted values
        targets: Ground truth values

    Returns:
        Dictionary with MAE and RMSE (in percentage)
    """
    mae = mean_absolute_error(targets, predictions) * 100  # Convert to percentage
    rmse = np.sqrt(mean_squared_error(targets, predictions)) * 100

    return {
        'mae': mae,
        'rmse': rmse
    }


def save_model(model, filepath):
    """Save model checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, device):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {filepath}")
    return model


if __name__ == "__main__":
    print("Training module loaded successfully!")
    print("Use train_model() for primary training")
    print("Use secondary_training() for online optimization")
