"""Model architectures and training utilities."""

# Baseline models for SOH estimation (Many-to-One)
from .baseline_models import FNN, CNN, LSTM, GRU, BiLSTM, BiGRU, MLP, ResCNN

# Seq2Seq models for SOH estimation (Many-to-Many)
from .seq2seq_models import LSTMSeq2Seq, GRUSeq2Seq, BiLSTMSeq2Seq, BiGRUSeq2Seq

# Unified model factory system
from .model_factory import (
    ConfigLoader,
    ModelFactory,
    UnifiedModelWrapper
)

# Physics-constrained loss
from .physics_loss import PhysicsConstrainedLoss

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

    # Seq2Seq Models (Many-to-Many)
    'LSTMSeq2Seq',
    'GRUSeq2Seq',
    'BiLSTMSeq2Seq',
    'BiGRUSeq2Seq',

    # Unified system
    'ConfigLoader',
    'ModelFactory',
    'UnifiedModelWrapper',

    # Physics constraints
    'PhysicsConstrainedLoss',
]
