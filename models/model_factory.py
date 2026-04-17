"""
统一的模型工厂系统。

支持创建和配置所有基准模型：FNN, CNN, LSTM, GRU。
提供统一的接口用于模型创建、配置加载和参数覆盖。
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .baseline_models import FNN, CNN, LSTM, GRU, BiLSTM, BiGRU, MLP, ResCNN, XGBoost_Simple, XGBoost_Enhanced
from .cnn_lstm import CNN_LSTM, CNN_BiLSTM, CNN_MLP


class ConfigLoader:
    """配置文件加载器。"""

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        加载JSON配置文件。

        Args:
            config_path: 配置文件路径

        Returns:
            配置字典
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config

    @staticmethod
    def load_model_config(model_type: str, config_dir: str = 'configs/models') -> Dict[str, Any]:
        """
        通过模型类型加载配置。

        Args:
            model_type: 模型类型 ('fnn', 'cnn', 'lstm', 'gru')
            config_dir: 配置目录

        Returns:
            配置字典
        """
        model_type = model_type.lower()
        config_path = Path(config_dir) / f'{model_type}_config.json'
        return ConfigLoader.load_config(str(config_path))

    @staticmethod
    def save_config(config: Dict[str, Any], save_path: str):
        """
        保存配置到文件。

        Args:
            config: 配置字典
            save_path: 保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"Configuration saved to: {save_path}")

    @staticmethod
    def print_config(config: Dict[str, Any]):
        """打印配置信息。"""
        print("\n" + "=" * 70)
        print("模型配置")
        print("=" * 70)
        print(f"\n型号: {config['model_type']} - {config['model_name']}")
        print(f"描述: {config.get('description', 'N/A')}")

        print(f"\n架构参数:")
        for key, value in config['architecture'].items():
            print(f"  {key}: {value}")

        print(f"\n训练参数:")
        for key, value in config['training'].items():
            if key not in ['early_stopping', 'scheduler']:
                print(f"  {key}: {value}")

        if 'data' in config:
            print(f"\n数据设置:")
            for key, value in config['data'].items():
                print(f"  {key}: {value}")

        if 'feature_selection' in config:
            print(f"\n特征筛选: {config['feature_selection']['enabled']}")
            if config['feature_selection']['enabled']:
                print(f"  阈值: {config['feature_selection'].get('correlation_threshold', 'N/A')}")
                print(f"  Top-K: {config['feature_selection'].get('top_k', 'N/A')}")

        print("=" * 70)


class ModelFactory:
    """
    模型工厂类。

    提供统一接口创建所有支持的基准模型。
    支持从配置文件加载或直接传入参数。
    """

    SUPPORTED_MODELS = [
        'fnn', 'cnn', 'lstm', 'gru', 'bilstm', 'bigru', 'mlp', 'rescnn',
        'lstm_seq2seq', 'gru_seq2seq', 'bilstm_seq2seq', 'bigru_seq2seq',
        'cnn_lstm', 'cnn_bilstm', 'cnn_mlp',
        'cnn_lstm_attention',         # Exp-01: CNN-LSTM + M1 循环级注意力
        'cnn_lstm_mc',               # Exp-02: CNN-LSTM + M2 MC Dropout
        'cnn_lstm_rate_smoothness',  # Exp-03: CNN-LSTM + M4 速率连续性约束
        'cnn_lstm_adaptive_weight',  # Exp-04/05: CNN-LSTM + M5 自适应损失权重（含 M4）
        'xgboost_simple', 'xgboost_enhanced'
    ]

    @staticmethod
    def create_model(
        model_type: str,
        input_size: int,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        **kwargs
    ) -> nn.Module:
        """
        创建模型实例。

        Args:
            model_type: 模型类型 ('fnn', 'cnn', 'lstm', 'gru')
            input_size: 输入特征维度
            config: 配置字典（可选）
            config_path: 配置文件路径（可选）
            **kwargs: 额外参数，用于覆盖配置中的值

        Returns:
            模型实例

        Example:
            # 方式1: 使用配置文件
            model = ModelFactory.create_model('fnn', input_size=6, config_path='configs/models/fnn_config.json')

            # 方式2: 使用配置字典
            config = ConfigLoader.load_model_config('fnn')
            model = ModelFactory.create_model('fnn', input_size=6, config=config)

            # 方式3: 覆盖配置参数
            model = ModelFactory.create_model('fnn', input_size=6, config=config, hidden_sizes=[128, 64, 32])
        """
        model_type = model_type.lower()

        if model_type not in ModelFactory.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported models: {ModelFactory.SUPPORTED_MODELS}")

        # 加载配置
        if config is None and config_path is None:
            config = ConfigLoader.load_model_config(model_type)
        elif config_path is not None:
            config = ConfigLoader.load_config(config_path)

        # 提取架构参数
        arch = config['architecture'].copy()
        arch['input_size'] = input_size  # 覆盖为实际输入维度

        # 应用kwargs覆盖
        arch.update(kwargs)

        # 创建模型
        if model_type == 'fnn':
            return ModelFactory._create_fnn(arch)
        elif model_type == 'cnn':
            return ModelFactory._create_cnn(arch)
        elif model_type == 'lstm':
            return ModelFactory._create_lstm(arch)
        elif model_type == 'gru':
            return ModelFactory._create_gru(arch)
        elif model_type == 'bilstm':
            return ModelFactory._create_bilstm(arch)
        elif model_type == 'bigru':
            return ModelFactory._create_bigru(arch)
        elif model_type == 'mlp':
            return ModelFactory._create_mlp(arch)
        elif model_type == 'rescnn':
            return ModelFactory._create_rescnn(arch)
        elif model_type == 'lstm_seq2seq':
            return ModelFactory._create_lstm_seq2seq(arch)
        elif model_type == 'gru_seq2seq':
            return ModelFactory._create_gru_seq2seq(arch)
        elif model_type == 'bilstm_seq2seq':
            return ModelFactory._create_bilstm_seq2seq(arch)
        elif model_type == 'bigru_seq2seq':
            return ModelFactory._create_bigru_seq2seq(arch)
        elif model_type in ('cnn_lstm', 'cnn_lstm_attention', 'cnn_lstm_mc',
                            'cnn_lstm_rate_smoothness', 'cnn_lstm_adaptive_weight'):
            # 三种 model_type 均实例化 CNN_LSTM，区别在 config 中的 attention/mc_dropout 开关
            config_copy = config.copy()
            config_copy['architecture'] = arch
            return ModelFactory._create_cnn_lstm(config_copy)
        elif model_type == 'cnn_bilstm':
            # 更新配置中的input_size
            config_copy = config.copy()
            config_copy['architecture'] = arch
            return ModelFactory._create_cnn_bilstm(config_copy)
        elif model_type == 'cnn_mlp':
            # 更新配置中的input_size
            config_copy = config.copy()
            config_copy['architecture'] = arch
            return ModelFactory._create_cnn_mlp(config_copy)
        elif model_type == 'xgboost_simple':
            return ModelFactory._create_xgboost_simple(arch)
        elif model_type == 'xgboost_enhanced':
            return ModelFactory._create_xgboost_enhanced(arch)

    @staticmethod
    def _create_fnn(arch: Dict[str, Any]) -> FNN:
        """创建FNN模型。"""
        return FNN(
            input_size=arch['input_size'],
            hidden_sizes=arch['hidden_sizes'],
            dropout_rate=arch.get('dropout_rate', 0.2)
        )

    @staticmethod
    def _create_cnn(arch: Dict[str, Any]) -> CNN:
        """创建CNN模型。"""
        return CNN(
            input_size=arch['input_size'],
            num_filters=arch.get('num_filters', 64),
            kernel_size=arch.get('kernel_size', 3),
            fc_hidden_sizes=arch.get('fc_hidden_sizes', [32, 16]),
            dropout_rate=arch.get('dropout_rate', 0.2)
        )

    @staticmethod
    def _create_lstm(arch: Dict[str, Any]) -> LSTM:
        """创建LSTM模型。"""
        return LSTM(
            input_size=arch['input_size'],
            hidden_size=arch.get('hidden_size', 64),
            num_layers=arch.get('num_layers', 2),
            fc_hidden_sizes=arch.get('fc_hidden_sizes', [32, 16]),
            dropout_rate=arch.get('dropout_rate', 0.2)
        )

    @staticmethod
    def _create_gru(arch: Dict[str, Any]) -> GRU:
        """创建GRU模型。"""
        return GRU(
            input_size=arch['input_size'],
            hidden_size=arch.get('hidden_size', 64),
            num_layers=arch.get('num_layers', 2),
            fc_hidden_sizes=arch.get('fc_hidden_sizes', [32, 16]),
            dropout_rate=arch.get('dropout_rate', 0.2)
        )

    @staticmethod
    def _create_bilstm(arch: Dict[str, Any]) -> BiLSTM:
        """创建BiLSTM模型。"""
        return BiLSTM(
            input_size=arch['input_size'],
            hidden_size=arch.get('hidden_size', 64),
            num_layers=arch.get('num_layers', 2),
            fc_hidden_sizes=arch.get('fc_hidden_sizes', [32, 16]),
            dropout_rate=arch.get('dropout_rate', 0.2)
        )

    @staticmethod
    def _create_bigru(arch: Dict[str, Any]) -> BiGRU:
        """创建BiGRU模型。"""
        return BiGRU(
            input_size=arch['input_size'],
            hidden_size=arch.get('hidden_size', 64),
            num_layers=arch.get('num_layers', 2),
            fc_hidden_sizes=arch.get('fc_hidden_sizes', [32, 16]),
            dropout_rate=arch.get('dropout_rate', 0.2)
        )

    @staticmethod
    def _create_mlp(arch: Dict[str, Any]) -> MLP:
        """创建MLP模型（多层感知机）。"""
        return MLP(
            input_size=arch['input_size'],
            hidden_sizes=arch.get('hidden_sizes', [60, 60, 60]),
            output_size=arch.get('output_size', 32),
            dropout_rate=arch.get('dropout_rate', 0.2)
        )

    @staticmethod
    def _create_rescnn(arch: Dict[str, Any]) -> ResCNN:
        """创建ResCNN模型（带残差连接的CNN）。"""
        return ResCNN(
            input_size=arch['input_size'],
            channel_config=arch.get('channel_config', [8, 16, 24, 16, 8]),
            stride_config=arch.get('stride_config', [1, 2, 2, 1, 1]),
            dropout_rate=arch.get('dropout_rate', 0.0)
        )

    @staticmethod
    def _create_lstm_seq2seq(arch: Dict[str, Any]):
        """创建LSTM Seq2Seq模型（Many-to-Many）。"""
        from .seq2seq_models import LSTMSeq2Seq
        return LSTMSeq2Seq(
            input_size=arch['input_size'],
            hidden_size=arch.get('hidden_size', 64),
            num_layers=arch.get('num_layers', 2),
            dropout_rate=arch.get('dropout_rate', 0.2),
            bidirectional=arch.get('bidirectional', False)
        )

    @staticmethod
    def _create_gru_seq2seq(arch: Dict[str, Any]):
        """创建GRU Seq2Seq模型（Many-to-Many）。"""
        from .seq2seq_models import GRUSeq2Seq
        return GRUSeq2Seq(
            input_size=arch['input_size'],
            hidden_size=arch.get('hidden_size', 64),
            num_layers=arch.get('num_layers', 2),
            dropout_rate=arch.get('dropout_rate', 0.2),
            bidirectional=arch.get('bidirectional', False)
        )

    @staticmethod
    def _create_bilstm_seq2seq(arch: Dict[str, Any]):
        """创建BiLSTM Seq2Seq模型（Many-to-Many）。"""
        from .seq2seq_models import BiLSTMSeq2Seq
        return BiLSTMSeq2Seq(
            input_size=arch['input_size'],
            hidden_size=arch.get('hidden_size', 64),
            num_layers=arch.get('num_layers', 2),
            dropout_rate=arch.get('dropout_rate', 0.2)
        )

    @staticmethod
    def _create_bigru_seq2seq(arch: Dict[str, Any]):
        """创建BiGRU Seq2Seq模型（Many-to-Many）。"""
        from .seq2seq_models import BiGRUSeq2Seq
        return BiGRUSeq2Seq(
            input_size=arch['input_size'],
            hidden_size=arch.get('hidden_size', 64),
            num_layers=arch.get('num_layers', 2),
            dropout_rate=arch.get('dropout_rate', 0.2)
        )

    @staticmethod
    def _create_cnn_lstm(config: Dict[str, Any]) -> CNN_LSTM:
        """创建CNN-LSTM混合模型。"""
        return CNN_LSTM(config)

    @staticmethod
    def _create_cnn_bilstm(config: Dict[str, Any]) -> CNN_BiLSTM:
        """创建CNN-BiLSTM混合模型。"""
        return CNN_BiLSTM(config)

    @staticmethod
    def _create_cnn_mlp(config: Dict[str, Any]) -> CNN_MLP:
        """创建CNN-MLP模型。"""
        return CNN_MLP(config)

    @staticmethod
    def _create_xgboost_simple(arch: Dict[str, Any]) -> XGBoost_Simple:
        """创建XGBoost_Simple模型。"""
        return XGBoost_Simple(
            input_size=arch['input_size'],
            window_size=arch.get('window_size', 40),
            n_estimators=arch.get('n_estimators', 300),
            max_depth=arch.get('max_depth', 6),
            learning_rate=arch.get('learning_rate', 0.05),
            subsample=arch.get('subsample', 0.8),
            colsample_bytree=arch.get('colsample_bytree', 0.8),
            reg_alpha=arch.get('reg_alpha', 0.1),
            reg_lambda=arch.get('reg_lambda', 1.0),
            random_state=arch.get('random_state', 42)
        )

    @staticmethod
    def _create_xgboost_enhanced(arch: Dict[str, Any]) -> XGBoost_Enhanced:
        """创建XGBoost_Enhanced模型。"""
        return XGBoost_Enhanced(
            input_size=arch['input_size'],
            window_size=arch.get('window_size', 40),
            n_lags=arch.get('n_lags', 10),
            rolling_windows=arch.get('rolling_windows', [5, 10, 20]),
            n_estimators=arch.get('n_estimators', 500),
            max_depth=arch.get('max_depth', 8),
            learning_rate=arch.get('learning_rate', 0.03),
            subsample=arch.get('subsample', 0.8),
            colsample_bytree=arch.get('colsample_bytree', 0.8),
            reg_alpha=arch.get('reg_alpha', 0.1),
            reg_lambda=arch.get('reg_lambda', 1.0),
            random_state=arch.get('random_state', 42)
        )

    @staticmethod
    def create_loss_function(
        model_type: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> nn.Module:
        """
        创建对应的损失函数。

        Args:
            model_type: 模型类型
            config: 配置字典（可选）
            **kwargs: 额外参数

        Returns:
            损失函数实例
        """
        # 直接返回标准MSE损失
        return nn.MSELoss()

    @staticmethod
    def get_model_info(model: nn.Module) -> Dict[str, Any]:
        """
        获取模型信息。

        Args:
            model: 模型实例

        Returns:
            模型信息字典
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        info = {
            'model_type': model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2),  # 假设float32
        }

        return info

    @staticmethod
    def print_model_info(model: nn.Module):
        """打印模型信息。"""
        info = ModelFactory.get_model_info(model)

        print("\n" + "=" * 70)
        print("模型信息")
        print("=" * 70)
        print(f"模型类型: {info['model_type']}")
        print(f"总参数量: {info['total_parameters']:,}")
        print(f"可训练参数: {info['trainable_parameters']:,}")
        print(f"模型大小: {info['model_size_mb']:.2f} MB")
        print("=" * 70)


class UnifiedModelWrapper:
    """
    统一的模型包装器。

    封装模型、配置和训练状态，提供一致的接口。
    """

    def __init__(
        self,
        model_type: str,
        input_size: int,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        device: str = 'cpu',
        **kwargs
    ):
        """
        初始化模型包装器。

        Args:
            model_type: 模型类型
            input_size: 输入维度
            config: 配置字典
            config_path: 配置文件路径
            device: 计算设备
            **kwargs: 额外参数
        """
        self.model_type = model_type.lower()
        self.input_size = input_size
        self.device = torch.device(device)

        # 加载配置
        if config is None and config_path is None:
            self.config = ConfigLoader.load_model_config(self.model_type)
        elif config_path is not None:
            self.config = ConfigLoader.load_config(config_path)
        else:
            self.config = config

        # 创建模型
        self.model = ModelFactory.create_model(
            model_type=self.model_type,
            input_size=input_size,
            config=self.config,
            **kwargs
        ).to(self.device)

        # 创建损失函数
        self.criterion = ModelFactory.create_loss_function(
            model_type=self.model_type,
            config=self.config
        )

        # 训练状态
        self.best_mae = float('inf')
        self.best_epoch = 0
        self.training_history = {
            'train_loss': [],
            'test_mae': [],
            'test_rmse': []
        }

    def get_optimizer(self, **kwargs):
        """
        获取优化器。

        Args:
            **kwargs: 覆盖配置中的优化器参数

        Returns:
            优化器实例
        """
        lr = kwargs.get('learning_rate', self.config['training'].get('learning_rate', 0.001))
        optimizer_type = kwargs.get('optimizer', self.config['training'].get('optimizer', 'Adam'))

        if optimizer_type.lower() == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_type.lower() == 'sgd':
            momentum = kwargs.get('momentum', 0.9)
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    def save_checkpoint(self, save_path: str, epoch: int, optimizer_state: Optional[Dict] = None):
        """保存检查点。"""
        checkpoint = {
            'model_type': self.model_type,
            'input_size': self.input_size,
            'config': self.config,
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
            'best_mae': self.best_mae,
            'best_epoch': self.best_epoch,
            'training_history': self.training_history
        }

        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state

        # 创建目录（如果路径包含目录）
        dir_path = os.path.dirname(save_path)
        if dir_path:  # 只有当路径包含目录时才创建
            os.makedirs(dir_path, exist_ok=True)

        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to: {save_path}")

    @staticmethod
    def load_checkpoint(checkpoint_path: str, device: str = 'cpu'):
        """加载检查点。"""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        wrapper = UnifiedModelWrapper(
            model_type=checkpoint['model_type'],
            input_size=checkpoint['input_size'],
            config=checkpoint['config'],
            device=device
        )

        wrapper.model.load_state_dict(checkpoint['model_state_dict'])
        wrapper.best_mae = checkpoint.get('best_mae', float('inf'))
        wrapper.best_epoch = checkpoint.get('best_epoch', 0)
        wrapper.training_history = checkpoint.get('training_history', {})

        print(f"Checkpoint loaded from: {checkpoint_path}")
        return wrapper

    def __repr__(self):
        return f"UnifiedModelWrapper(model_type={self.model_type}, input_size={self.input_size}, device={self.device})"


if __name__ == "__main__":
    """测试模型工厂功能。"""

    print("\n" + "=" * 70)
    print("Testing Model Factory")
    print("=" * 70)

    # 测试所有模型类型
    test_input_size = 6
    test_batch_size = 16
    test_input = torch.randn(test_batch_size, test_input_size)

    for model_type in ModelFactory.SUPPORTED_MODELS:
        print(f"\n{'='*70}")
        print(f"Testing: {model_type.upper()}")
        print('='*70)

        try:
            # 创建模型
            model = ModelFactory.create_model(
                model_type=model_type,
                input_size=test_input_size
            )

            # 前向传播测试
            output = model(test_input)

            # 打印信息
            ModelFactory.print_model_info(model)
            print(f"\nInput shape: {test_input.shape}")
            print(f"Output shape: {output.shape}")
            print(f"✅ {model_type.upper()} test passed!")

        except Exception as e:
            print(f"❌ {model_type.upper()} test failed: {e}")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
