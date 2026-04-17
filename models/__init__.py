"""Model architectures and training utilities."""

# Baseline models for SOH estimation (Many-to-One)
from .baseline_models import FNN, CNN, LSTM, GRU, BiLSTM, BiGRU, MLP, ResCNN

# Hybrid models
from .cnn_lstm import CNN_LSTM, CNN_BiLSTM, CNN_MLP

# Unified model factory system
from .model_factory import (
    ConfigLoader,
    ModelFactory,
    UnifiedModelWrapper
)

# Physics-constrained loss
from .physics_loss import PhysicsConstrainedLoss, SiamesePhysicsLoss, TripletPhysicsLoss

__all__ = [
    # Baseline Models (Many-to-One)
    'FNN',
    'CNN',
    'LSTM',
    'GRU',
    'BiLSTM',
    'BiGRU',
    'MLP',
    'ResCNN',

    # Hybrid Models
    'CNN_LSTM',
    'CNN_BiLSTM',
    'CNN_MLP',

    # Unified system
    'ConfigLoader',
    'ModelFactory',
    'UnifiedModelWrapper',

    # Physics constraints
    'PhysicsConstrainedLoss',
    'SiamesePhysicsLoss',
    'TripletPhysicsLoss',
]
