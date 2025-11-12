"""
统一的模型训练器模块。

支持通过配置文件训练不同的模型（FNN、CNN、LSTM）。
"""

import os
import json
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

from src.baseline_models import FNN, CNN, LSTM
from src.feature_selector import (
    get_top_correlated_features,
    create_filtered_data_dict,
    print_feature_selection_report
)
from src.data_loader_hust import create_hust_dataloaders


class ConfigLoader:
    """配置文件加载器。"""
    
    @staticmethod
    def load_config(config_path):
        """
        加载JSON配置文件。
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    
    @staticmethod
    def load_model_config_by_name(model_name, config_dir='configs/models'):
        """
        通过模型名称加载配置。
        
        Args:
            model_name: 模型名称 ('FNN', 'CNN', 'LSTM')
            config_dir: 配置目录
            
        Returns:
            配置字典
        """
        config_path = os.path.join(config_dir, f'{model_name.lower()}_config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        return ConfigLoader.load_config(config_path)
    
    @staticmethod
    def print_config(config):
        """打印配置信息。"""
        print("\n" + "=" * 70)
        print("模型配置")
        print("=" * 70)
        print(f"\n型号: {config['model_type']} - {config['model_name']}")
        print(f"描述: {config['description']}")
        print(f"\n架构:")
        for key, value in config['architecture'].items():
            print(f"  {key}: {value}")
        print(f"\n训练参数:")
        for key, value in config['training'].items():
            if key != 'early_stopping':
                print(f"  {key}: {value}")
        print(f"\n数据设置:")
        for key, value in config['data'].items():
            print(f"  {key}: {value}")
        print(f"\n特征筛选: {config['feature_selection']['enabled']}")
        if config['feature_selection']['enabled']:
            print(f"  阈值: {config['feature_selection']['correlation_threshold']}")
        print("=" * 70)


class ModelFactory:
    """模型工厂类。"""
    
    @staticmethod
    def create_model(config, input_size):
        """
        根据配置创建模型。
        
        Args:
            config: 配置字典
            input_size: 输入特征维度
            
        Returns:
            模型对象
        """
        model_type = config['model_type'].upper()
        arch = config['architecture'].copy()
        arch['input_size'] = input_size  # 自动覆盖为实际的输入维度
        
        if model_type == 'FNN':
            return FNN(
                input_size=arch['input_size'],
                hidden_sizes=arch['hidden_sizes'],
                dropout_rate=arch['dropout_rate']
            )
        elif model_type == 'CNN':
            return CNN(
                input_size=arch['input_size'],
                num_filters=arch['num_filters'],
                kernel_size=arch.get('kernel_size', 3),
                fc_hidden_sizes=arch['fc_hidden_sizes'],
                dropout_rate=arch['dropout_rate']
            )
        elif model_type == 'LSTM':
            return LSTM(
                input_size=arch['input_size'],
                hidden_size=arch['hidden_size'],
                num_layers=arch['num_layers'],
                fc_hidden_sizes=arch['fc_hidden_sizes'],
                dropout_rate=arch['dropout_rate']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class ModelTrainer:
    """统一的模型训练器。"""
    
    def __init__(self, config, device='cpu'):
        """
        初始化训练器。
        
        Args:
            config: 配置字典
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.config = config
        self.device = torch.device(device)
        self.model = None
        self.best_mae = float('inf')
        self.best_epoch = 0
        
    def train(self, data_dict, verbose=True):
        """
        训练模型。
        
        Args:
            data_dict: 数据字典
            verbose: 是否打印详细信息
            
        Returns:
            训练历史和最终结果
        """
        # 1. 特征筛选
        if self.config['feature_selection']['enabled']:
            threshold = self.config['feature_selection']['correlation_threshold']
            selection_result = get_top_correlated_features(data_dict, threshold)
            
            if verbose:
                print_feature_selection_report(selection_result, threshold)
            
            data_dict = create_filtered_data_dict(
                data_dict,
                selection_result['selected_indices']
            )
            print(f"\nSelected features completed: {data_dict['selected_feature_count']} / "
                  f"{data_dict['original_feature_count']} features")
        
        # 2. 创建数据加载器
        train_loader, test_loader = create_hust_dataloaders(
            data_dict,
            batch_size=self.config['training']['batch_size']
        )
        
        # 3. 创建模型
        input_size = data_dict['train_features'].shape[1]
        self.model = ModelFactory.create_model(self.config, input_size)
        self.model = self.model.to(self.device)
        
        if verbose:
            print(f"\nModel parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        # 4. 设置优化器和损失函数
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate']
        )
        
        # 5. 训练循环
        history = {
            'train_loss': [],
            'test_mae': [],
            'test_rmse': []
        }
        
        num_epochs = self.config['training']['num_epochs']
        iterator = tqdm(range(num_epochs), desc="Training", disable=not verbose)
        
        for epoch in iterator:
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for features, capacity in train_loader:
                features = features.to(self.device)
                capacity = capacity.to(self.device).unsqueeze(1)
                
                optimizer.zero_grad()
                predictions = self.model(features)
                loss = criterion(predictions, capacity)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * features.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # 评估阶段
            self.model.eval()
            test_mae = 0.0
            test_rmse = 0.0
            
            with torch.no_grad():
                for features, capacity in test_loader:
                    features = features.to(self.device)
                    capacity = capacity.to(self.device).unsqueeze(1)
                    
                    predictions = self.model(features)
                    
                    mae = torch.mean(torch.abs(predictions - capacity)).item()
                    rmse = torch.sqrt(torch.mean((predictions - capacity) ** 2)).item()
                    
                    test_mae += mae * features.size(0)
                    test_rmse += rmse * features.size(0)
            
            mae = test_mae / len(test_loader.dataset)
            rmse = test_rmse / len(test_loader.dataset)
            
            history['train_loss'].append(train_loss)
            history['test_mae'].append(mae)
            history['test_rmse'].append(rmse)
            
            # 保存最佳模型
            if mae < self.best_mae:
                self.best_mae = mae
                self.best_epoch = epoch + 1
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            
            # 更新进度条
            if (epoch + 1) % 100 == 0:
                iterator.set_postfix({
                    'Loss': f'{train_loss:.6f}',
                    'MAE': f'{mae:.6f}',
                    'Best': f'{self.best_mae:.6f}'
                })
        
        # 6. 恢复最佳模型
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
            if verbose:
                print(f"\nRestored best model (Epoch {self.best_epoch})")
        
        # 7. 最终评估
        self.model.eval()
        final_mae = 0.0
        final_rmse = 0.0
        
        with torch.no_grad():
            for features, capacity in test_loader:
                features = features.to(self.device)
                capacity = capacity.to(self.device).unsqueeze(1)
                
                predictions = self.model(features)
                
                mae = torch.mean(torch.abs(predictions - capacity)).item()
                rmse = torch.sqrt(torch.mean((predictions - capacity) ** 2)).item()
                
                final_mae += mae * features.size(0)
                final_rmse += rmse * features.size(0)
        
        final_mae /= len(test_loader.dataset)
        final_rmse /= len(test_loader.dataset)
        
        results = {
            'mae': final_mae,
            'rmse': final_rmse,
            'best_epoch': self.best_epoch,
            'best_mae': self.best_mae,
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else 0
        }
        
        if verbose:
            print(f"\nFinal results:")
            print(f"  MAE: {final_mae*100:.4f}%")
            print(f"  RMSE: {final_rmse*100:.4f}%")
        
        return history, results
    
    def save_model(self, save_path):
        """Save model."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved: {save_path}")
    
    def load_model(self, load_path, input_size):
        """Load model."""
        self.model = ModelFactory.create_model(self.config, input_size)
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        self.model = self.model.to(self.device)
        print(f"Model loaded: {load_path}")


if __name__ == "__main__":
    # 测试配置加载
    config = ConfigLoader.load_model_config_by_name('FNN')
    ConfigLoader.print_config(config)
    print("\n✅ Configuration loader test passed!")
