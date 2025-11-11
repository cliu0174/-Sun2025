"""
BPINN (Battery Physics-Informed Neural Network) model definition.
Implements a feedforward neural network with physics constraints for SOH estimation.
"""

import torch
import torch.nn as nn


class BPINN(nn.Module):
    """
    Battery Physics-Informed Neural Network.

    This model is a feedforward neural network that incorporates physical constraints
    during training to ensure predictions follow physical laws (monotonic P-IC to SOH relationship).

    Architecture based on the paper:
    - Input layer: 6 features from IC curves
    - Hidden layers: 3 hidden layers with 10 neurons each (as mentioned in paper Section 3.1)
    - Output layer: 1 neuron for SOH prediction
    """

    def __init__(self, input_size=6, hidden_sizes=[10, 10, 10], dropout_rate=0.0):
        """
        Initialize BPINN model.

        Args:
            input_size: Number of input features (default: 6 for IC features)
            hidden_sizes: List of hidden layer sizes (default: [10, 10, 10] - 3 hidden layers as per paper)
            dropout_rate: Dropout rate for regularization (default: 0.0)
        """
        super(BPINN, self).__init__()

        layers = []
        prev_size = input_size

        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())  # Ensure output is in [0, 1] range for SOH

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            SOH predictions of shape (batch_size, 1)
        """
        return self.network(x)


class BPINNLoss(nn.Module):
    """
    Custom loss function for BPINN that combines data-driven loss and physical constraint loss.

    Loss = L_data + λ * L_physics

    where:
    - L_data: Mean squared error between predicted and actual SOH
    - L_physics: Physical constraint loss ensuring ∂SOH/∂P-IC > 0
    - λ: Weight parameter for physical loss (ω_1 in the paper, default 0.01)
    """

    def __init__(self, lambda_physics=0.01):
        """
        Initialize BPINN loss.

        Args:
            lambda_physics: Weight for physical constraint loss (ω_1 in paper, default: 0.01)
        """
        super(BPINNLoss, self).__init__()
        self.lambda_physics = lambda_physics
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets, features, model):
        """
        Compute combined loss with differential monotonicity constraint.

        Uses pairwise differential method within batch:
        For sorted samples by P-IC: δ_i = g(x_{i+1}) - g(x_i) > 0

        Args:
            predictions: Model predictions (batch_size, 1)
            targets: Ground truth SOH values (batch_size, 1)
            features: Input features (batch_size, n_features)
            model: BPINN model instance

        Returns:
            total_loss, data_loss, physics_loss
        """
        # Data-driven loss (MSE)
        data_loss = self.mse(predictions, targets.unsqueeze(1))

        # Physical constraint: pairwise monotonicity within batch
        # P-IC is the first feature (y_h)

        # Sort samples by P-IC value (first feature)
        p_ic_values = features[:, 0]
        sorted_indices = torch.argsort(p_ic_values)

        # Get predictions for sorted samples
        sorted_features = features[sorted_indices]
        sorted_predictions = model(sorted_features).squeeze()

        # Compute pairwise differences: δ_i = g(x_{i+1}) - g(x_i)
        # This should be >= 0 for monotonicity
        if len(sorted_predictions) > 1:
            pairwise_diffs = sorted_predictions[1:] - sorted_predictions[:-1]

            # Penalize negative differences (violations of monotonicity)
            physics_loss = torch.mean(torch.relu(-pairwise_diffs) ** 2)
        else:
            # If batch size is 1, no constraint can be computed
            physics_loss = torch.tensor(0.0, device=features.device)

        # Total loss
        total_loss = data_loss + self.lambda_physics * physics_loss

        return total_loss, data_loss, physics_loss


class SecondaryTrainingLoss(nn.Module):
    """
    Loss function for secondary training (online optimization) during testing phase.

    According to the paper, the secondary training objective should be:
    L_total = MSE(train) + ω1·L_physics(train) + ω2·L_physics(test)

    This helps refine the model predictions on test data while maintaining
    physical consistency on BOTH train and test sets.
    """

    def __init__(self, lambda_train_physics=0.01, lambda_test_physics=0.01):
        """
        Initialize secondary training loss.

        Args:
            lambda_train_physics: Weight for train physics loss (ω_1 in paper, default: 0.01)
            lambda_test_physics: Weight for test physics loss (ω_2 in paper, default: 0.01)
        """
        super(SecondaryTrainingLoss, self).__init__()
        self.lambda_train_physics = lambda_train_physics
        self.lambda_test_physics = lambda_test_physics
        self.mse = nn.MSELoss()

    def forward(self, train_predictions, train_targets, train_features,
                test_predictions, test_features, model):
        """
        Compute secondary training loss with pairwise monotonicity constraint.

        L_total = MSE(train) + ω1·L_physics(train) + ω2·L_physics(test)

        Uses pairwise difference method for physics constraint:
        For sorted samples: δ_i = g(x_{i+1}) - g(x_i) >= 0

        Args:
            train_predictions: Model predictions on train set
            train_targets: Ground truth SOH for train set
            train_features: Train input features
            test_predictions: Model predictions on test set (unused, recalculated internally)
            test_features: Test input features
            model: BPINN model instance

        Returns:
            total_loss, train_mse, train_physics_loss, test_physics_loss
        """
        # 1. Compute MSE on training data
        train_mse = self.mse(train_predictions.squeeze(), train_targets)

        # 2. Compute pairwise physics constraint on TRAINING data
        train_p_ic = train_features[:, 0]
        train_sorted_indices = torch.argsort(train_p_ic)
        train_sorted_features = train_features[train_sorted_indices]
        train_sorted_preds = model(train_sorted_features).squeeze()

        if len(train_sorted_preds) > 1:
            train_pairwise_diffs = train_sorted_preds[1:] - train_sorted_preds[:-1]
            train_physics_loss = torch.mean(torch.relu(-train_pairwise_diffs) ** 2)
        else:
            train_physics_loss = torch.tensor(0.0, device=train_features.device)

        # 3. Compute pairwise physics constraint on TEST data
        test_p_ic = test_features[:, 0]
        test_sorted_indices = torch.argsort(test_p_ic)
        test_sorted_features = test_features[test_sorted_indices]
        test_sorted_preds = model(test_sorted_features).squeeze()

        if len(test_sorted_preds) > 1:
            test_pairwise_diffs = test_sorted_preds[1:] - test_sorted_preds[:-1]
            test_physics_loss = torch.mean(torch.relu(-test_pairwise_diffs) ** 2)
        else:
            test_physics_loss = torch.tensor(0.0, device=test_features.device)

        # 4. Total loss: MSE(train) + ω1·Physics(train) + ω2·Physics(test)
        total_loss = (train_mse +
                     self.lambda_train_physics * train_physics_loss +
                     self.lambda_test_physics * test_physics_loss)

        return total_loss, train_mse, train_physics_loss, test_physics_loss


if __name__ == "__main__":
    # Test model instantiation
    model = BPINN(input_size=6, hidden_sizes=[10, 10, 10])
    print("BPINN Model:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 6)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Test loss function
    criterion = BPINNLoss(lambda_physics=0.01)
    targets = torch.rand(batch_size)
    loss, data_loss, physics_loss = criterion(y, targets, x, model)
    print(f"\nTotal loss: {loss.item():.6f}")
    print(f"Data loss: {data_loss.item():.6f}")
    print(f"Physics loss: {physics_loss.item():.6f}")
