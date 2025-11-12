"""Model architectures and training utilities."""

# Base models
from .baseline_models import FNN, CNN, LSTM
from .model import BPINN, BPINNLoss, SecondaryTrainingLoss

# Legacy trainer (for backward compatibility)
from .model_trainer import ConfigLoader as LegacyConfigLoader
from .model_trainer import ModelTrainer as LegacyModelTrainer
from .model_trainer import ModelFactory as LegacyModelFactory

# New unified model factory system (recommended)
from .model_factory import (
    ConfigLoader,
    ModelFactory,
    UnifiedModelWrapper
)

__all__ = [
    # Models
    'FNN',
    'CNN',
    'LSTM',
    'BPINN',
    'BPINNLoss',
    'SecondaryTrainingLoss',

    # New unified system (recommended)
    'ConfigLoader',
    'ModelFactory',
    'UnifiedModelWrapper',

    # Legacy system (backward compatibility)
    'LegacyConfigLoader',
    'LegacyModelTrainer',
    'LegacyModelFactory',
]
