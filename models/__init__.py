"""Model architectures and training utilities."""

# Baseline models for SOH estimation (Many-to-One)
from .baseline_models import FNN, CNN, LSTM, GRU, BiLSTM, BiGRU, MLP, ResCNN

# Many-to-Many models for SOH estimation with physics constraints
from .baseline_models import LSTMManyToMany, GRUManyToMany, BiLSTMManyToMany, BiGRUManyToMany

# Seq2Seq models for SOH estimation (Many-to-Many) - 保留兼容性
from .seq2seq_models import LSTMSeq2Seq, GRUSeq2Seq, BiLSTMSeq2Seq, BiGRUSeq2Seq

# Unified model factory system
from .model_factory import (
    ConfigLoader,
    ModelFactory,
    UnifiedModelWrapper
)

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

    # Many-to-Many Models (用于物理约束)
    'LSTMManyToMany',
    'GRUManyToMany',
    'BiLSTMManyToMany',
    'BiGRUManyToMany',

    # Seq2Seq Models (Many-to-Many) - 保留兼容性
    'LSTMSeq2Seq',
    'GRUSeq2Seq',
    'BiLSTMSeq2Seq',
    'BiGRUSeq2Seq',

    # Unified system
    'ConfigLoader',
    'ModelFactory',
    'UnifiedModelWrapper',
]
