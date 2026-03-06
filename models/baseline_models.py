"""
Baseline models for battery SOH estimation: FNN, CNN, LSTM, GRU, XGBoost.
用于电池SOH估计的基准模型。
"""

import torch
import torch.nn as nn
import numpy as np


class FNN(nn.Module):
    """
    Feedforward Neural Network (FNN).
    简单的前馈神经网络。
    """

    def __init__(self, input_size=6, hidden_sizes=[64, 32, 16], dropout_rate=0.2):
        """
        Args:
            input_size: 输入特征数量
            hidden_sizes: 隐藏层大小列表
            dropout_rate: Dropout比率
        """
        super(FNN, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())  # SOH in [0, 1]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, input_size)
               或 (batch_size, window_size, input_size) - 会被自动展平

        Returns:
            SOH predictions (batch_size, 1)
        """
        # 如果接收到窗口化的 3D 数据 (batch, window_size, features)，展平为 2D
        if x.dim() == 3:
            batch_size, window_size, num_features = x.shape
            x = x.reshape(batch_size * window_size, num_features)
            out = self.network(x)
            # 对所有窗口的预测求平均
            out = out.reshape(batch_size, window_size, 1).mean(dim=1)
            return out
        else:
            return self.network(x)


class CNN(nn.Module):
    """
    Convolutional Neural Network (CNN).
    将特征序列当作1D信号处理。
    """

    def __init__(self, input_size=6, num_filters=64, kernel_size=3,
                 fc_hidden_sizes=[32, 16], dropout_rate=0.2):
        """
        Args:
            input_size: 输入特征数量
            num_filters: 卷积核数量
            kernel_size: 卷积核大小
            fc_hidden_sizes: 全连接层大小
            dropout_rate: Dropout比率
        """
        super(CNN, self).__init__()

        # Reshape input to (batch, channels=1, length=input_size)
        # 1D Convolution
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AdaptiveMaxPool1d(1)  # Global max pooling

        # Fully connected layers
        fc_layers = []
        prev_size = num_filters

        for hidden_size in fc_hidden_sizes:
            fc_layers.append(nn.Linear(prev_size, hidden_size))
            fc_layers.append(nn.ReLU())
            if dropout_rate > 0:
                fc_layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer
        fc_layers.append(nn.Linear(prev_size, 1))
        fc_layers.append(nn.Sigmoid())

        self.fc_network = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, input_size) 期望 2D
               或 (batch_size, window_size, input_size) - 会被展平处理

        Returns:
            SOH predictions (batch_size, 1)
        """
        # 如果接收到窗口化的 3D 数据，展平后逐个处理，再求平均
        if x.dim() == 3:
            batch_size, window_size, num_features = x.shape
            x = x.reshape(batch_size * window_size, num_features)
            x = x.unsqueeze(1)  # (batch*window_size, 1, num_features)
        else:
            x = x.unsqueeze(1)  # (batch, 1, input_size)
            batch_size, window_size = x.shape[0], 1

        # Convolution
        x = self.conv1(x)   # (batch*window_size, num_filters, input_size)
        x = self.relu1(x)
        x = self.pool1(x)   # (batch*window_size, num_filters, 1)

        # Flatten
        x = x.squeeze(-1)   # (batch*window_size, num_filters)

        # Fully connected
        x = self.fc_network(x)  # (batch*window_size, 1)
        
        # 如果输入是窗口化数据，对窗口内的预测求平均
        if window_size > 1:
            x = x.reshape(batch_size, window_size, 1).mean(dim=1)

        return x


class LSTM(nn.Module):
    """
    Long Short-Term Memory (LSTM).
    将特征序列当作时序数据处理。
    """

    def __init__(self, input_size=6, hidden_size=64, num_layers=2,
                 fc_hidden_sizes=[32, 16], dropout_rate=0.2):
        """
        Args:
            input_size: 输入特征数量（每个时间步）
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            fc_hidden_sizes: 全连接层大小
            dropout_rate: Dropout比率
        """
        super(LSTM, self).__init__()

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        # Fully connected layers
        fc_layers = []
        prev_size = hidden_size

        for fc_hidden_size in fc_hidden_sizes:
            fc_layers.append(nn.Linear(prev_size, fc_hidden_size))
            fc_layers.append(nn.ReLU())
            if dropout_rate > 0:
                fc_layers.append(nn.Dropout(dropout_rate))
            prev_size = fc_hidden_size

        # Output layer
        fc_layers.append(nn.Linear(prev_size, 1))
        fc_layers.append(nn.Sigmoid())

        self.fc_network = nn.Sequential(*fc_layers)

    # def forward(self, x):
    #     """
    #     Args:
    #         x: Input features (batch_size, input_size)

    #     Returns:
    #         SOH predictions (batch_size, 1)
    #     """
    #     # Reshape to (batch, seq_len=1, input_size) for LSTM
    #     # Treat all features as a single time step
    #     x = x.unsqueeze(1)  # (batch, 1, input_size)

    #     # LSTM
    #     lstm_out, (h_n, c_n) = self.lstm(x)

    #     # Use last hidden state
    #     x = h_n[-1]  # (batch, hidden_size)

    #     # Fully connected
    #     x = self.fc_network(x)

    #     return x


    def forward(self, x):
            """
            Args:
                x: Input features (batch_size, window_size, input_size)
                注意：这里假设 x 已经是 3D 张量
            """
            
            # 1. 如果数据加载器还没改好（输入是2D），为了兼容可以保留这行判断
            # 但强烈建议直接改数据加载器，去掉这个 if
            if x.dim() == 2:
                x = x.unsqueeze(1) 
                
            # 2. LSTM 前向传播
            # 输入形状: (batch, window_size, input_size)
            # 输出形状: lstm_out -> (batch, window_size, hidden_size)
            #          h_n      -> (num_layers, batch, hidden_size)
            lstm_out, (h_n, c_n) = self.lstm(x)

            # 3. 取最后一个时间步的隐藏状态
            # 注意：h_n 包含所有层的状态，我们取最后一层 (h_n[-1])
            # 形状: (batch, hidden_size)
            out = h_n[-1]

            # 4. 全连接层预测 SOH
            out = self.fc_network(out)

            return out

class GRU(nn.Module):
    """
    Gated Recurrent Unit (GRU).
    将特征序列当作时序数据处理，相比LSTM更简单高效。
    """

    def __init__(self, input_size=6, hidden_size=64, num_layers=2,
                 fc_hidden_sizes=[32, 16], dropout_rate=0.2):
        """
        Args:
            input_size: 输入特征数量（每个时间步）
            hidden_size: GRU隐藏层大小
            num_layers: GRU层数
            fc_hidden_sizes: 全连接层大小
            dropout_rate: Dropout比率
        """
        super(GRU, self).__init__()

        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        # Fully connected layers
        fc_layers = []
        prev_size = hidden_size

        for fc_hidden_size in fc_hidden_sizes:
            fc_layers.append(nn.Linear(prev_size, fc_hidden_size))
            fc_layers.append(nn.ReLU())
            if dropout_rate > 0:
                fc_layers.append(nn.Dropout(dropout_rate))
            prev_size = fc_hidden_size

        # Output layer
        fc_layers.append(nn.Linear(prev_size, 1))
        fc_layers.append(nn.Sigmoid())

        self.fc_network = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, window_size, input_size) - 推荐用于GRU
               或 (batch_size, input_size) - 会被转换为单时间步

        Returns:
            SOH predictions (batch_size, 1)
        """
        # 如果输入是 2D，转换为 3D (单时间步)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_size)

        # GRU
        gru_out, h_n = self.gru(x)  # h_n: (num_layers, batch, hidden_size)

        # Use last hidden state from last layer
        x = h_n[-1]  # (batch, hidden_size)

        # Fully connected
        x = self.fc_network(x)

        return x


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM (BiLSTM).
    双向LSTM，同时从前向和后向处理时序数据。
    """

    def __init__(self, input_size=6, hidden_size=64, num_layers=2,
                 fc_hidden_sizes=[32, 16], dropout_rate=0.2):
        """
        Args:
            input_size: 输入特征数量（每个时间步）
            hidden_size: LSTM隐藏层大小（单向）
            num_layers: LSTM层数
            fc_hidden_sizes: 全连接层大小
            dropout_rate: Dropout比率
        """
        super(BiLSTM, self).__init__()

        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True  # 双向
        )

        # Fully connected layers
        fc_layers = []
        prev_size = hidden_size * 2  # 双向输出，维度翻倍

        for fc_hidden_size in fc_hidden_sizes:
            fc_layers.append(nn.Linear(prev_size, fc_hidden_size))
            fc_layers.append(nn.ReLU())
            if dropout_rate > 0:
                fc_layers.append(nn.Dropout(dropout_rate))
            prev_size = fc_hidden_size

        # Output layer
        fc_layers.append(nn.Linear(prev_size, 1))
        fc_layers.append(nn.Sigmoid())

        self.fc_network = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, window_size, input_size)
               或 (batch_size, input_size) - 会被转换为单时间步

        Returns:
            SOH predictions (batch_size, 1)
        """
        # 如果输入是 2D，转换为 3D (单时间步)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # BiLSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers * 2, batch, hidden_size)

        # 取最后一层的前向和后向隐藏状态并拼接
        forward_h = h_n[-2, :, :]  # 最后一层前向
        backward_h = h_n[-1, :, :] # 最后一层后向
        x = torch.cat([forward_h, backward_h], dim=1)  # (batch, hidden_size*2)

        # Fully connected
        x = self.fc_network(x)

        return x


class BiGRU(nn.Module):
    """
    Bidirectional GRU (BiGRU).
    双向GRU，同时从前向和后向处理时序数据。
    """

    def __init__(self, input_size=6, hidden_size=64, num_layers=2,
                 fc_hidden_sizes=[32, 16], dropout_rate=0.2):
        """
        Args:
            input_size: 输入特征数量（每个时间步）
            hidden_size: GRU隐藏层大小（单向）
            num_layers: GRU层数
            fc_hidden_sizes: 全连接层大小
            dropout_rate: Dropout比率
        """
        super(BiGRU, self).__init__()

        # BiGRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True  # 双向
        )

        # Fully connected layers
        fc_layers = []
        prev_size = hidden_size * 2  # 双向输出，维度翻倍

        for fc_hidden_size in fc_hidden_sizes:
            fc_layers.append(nn.Linear(prev_size, fc_hidden_size))
            fc_layers.append(nn.ReLU())
            if dropout_rate > 0:
                fc_layers.append(nn.Dropout(dropout_rate))
            prev_size = fc_hidden_size

        # Output layer
        fc_layers.append(nn.Linear(prev_size, 1))
        fc_layers.append(nn.Sigmoid())

        self.fc_network = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, window_size, input_size)
               或 (batch_size, input_size) - 会被转换为单时间步

        Returns:
            SOH predictions (batch_size, 1)
        """
        # 如果输入是 2D，转换为 3D (单时间步)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # BiGRU
        gru_out, h_n = self.gru(x)
        # h_n: (num_layers * 2, batch, hidden_size)

        # 取最后一层的前向和后向隐藏状态并拼接
        forward_h = h_n[-2, :, :]  # 最后一层前向
        backward_h = h_n[-1, :, :] # 最后一层后向
        x = torch.cat([forward_h, backward_h], dim=1)  # (batch, hidden_size*2)

        # Fully connected
        x = self.fc_network(x)

        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP).
    简单的多层感知机，无输出激活函数。
    """

    def __init__(self, input_size=17, hidden_sizes=[60, 60, 60], output_size=32, dropout_rate=0.2):
        """
        Args:
            input_size: 输入特征数量
            hidden_sizes: 隐藏层大小列表
            output_size: 编码器输出维度
            dropout_rate: Dropout比率
        """
        super(MLP, self).__init__()

        # Encoder部分
        encoder_layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            encoder_layers.append(nn.Linear(prev_size, hidden_size))
            encoder_layers.append(nn.ReLU())
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        encoder_layers.append(nn.Linear(prev_size, output_size))
        encoder_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_layers)

        # Predictor部分
        self.predictor = nn.Sequential(
            nn.Linear(output_size, 1),
            nn.Sigmoid()  # SOH in [0, 1]
        )

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, input_size)
               或 (batch_size, window_size, input_size) - 会被展平处理

        Returns:
            SOH predictions (batch_size, 1)
        """
        # 如果接收到窗口化的 3D 数据，展平为 2D
        if x.dim() == 3:
            batch_size, window_size, num_features = x.shape
            x = x.reshape(batch_size * window_size, num_features)
            x = self.encoder(x)
            x = self.predictor(x)
            # 对所有窗口的预测求平均
            x = x.reshape(batch_size, window_size, 1).mean(dim=1)
            return x
        else:
            x = self.encoder(x)
            x = self.predictor(x)
            return x


class ResBlock(nn.Module):
    """
    Residual Block for 1D CNN.
    带残差连接的卷积块。
    """
    def __init__(self, input_channel, output_channel, stride):
        """
        Args:
            input_channel: 输入通道数
            output_channel: 输出通道数
            stride: 步长
        """
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),
            nn.Conv1d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(output_channel)
        )

        self.skip_connection = nn.Sequential()
        if output_channel != input_channel:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_channel)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.skip_connection(x) + out
        out = self.relu(out)
        return out


class ResCNN(nn.Module):
    """
    Residual CNN for battery SOH estimation.
    基于残差块的卷积神经网络，适用于电池SOH估计。
    """

    def __init__(self, input_size=17, channel_config=[8, 16, 24, 16, 8],
                 stride_config=[1, 2, 2, 1, 1], dropout_rate=0.0):
        """
        Args:
            input_size: 输入特征数量
            channel_config: 各层通道数配置 [layer1_out, layer2_out, ...]
            stride_config: 各层步长配置 [stride1, stride2, ...]
            dropout_rate: Dropout比率
        """
        super(ResCNN, self).__init__()

        assert len(channel_config) == len(stride_config), \
            "channel_config和stride_config长度必须一致"

        self.input_size = input_size

        # 构建ResBlock层
        layers = []
        in_channels = 1
        current_length = input_size

        for out_channels, stride in zip(channel_config, stride_config):
            layers.append(ResBlock(in_channels, out_channels, stride))
            in_channels = out_channels
            current_length = (current_length + stride - 1) // stride  # 计算输出长度

        self.res_layers = nn.Sequential(*layers)

        # 计算flatten后的维度
        self.flatten_dim = channel_config[-1] * current_length

        # 输出层
        fc_layers = []
        if dropout_rate > 0:
            fc_layers.append(nn.Dropout(dropout_rate))
        fc_layers.append(nn.Linear(self.flatten_dim, 1))
        # 不添加Sigmoid，直接输出 (匹配原始模型)

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, input_size)
               或 (batch_size, window_size, input_size) - 会被展平处理

        Returns:
            SOH predictions (batch_size, 1)
        """
        # 如果接收到窗口化的 3D 数据，展平后逐个处理，再求平均
        if x.dim() == 3:
            batch_size, window_size, num_features = x.shape
            x = x.reshape(batch_size * window_size, num_features)
        else:
            batch_size, window_size = x.shape[0], 1

        N = x.shape[0]

        # Reshape to (batch, 1, input_size)
        x = x.view(N, 1, -1)

        # 通过ResBlock层
        x = self.res_layers(x)

        # Flatten
        x = x.view(N, -1)

        # 全连接输出
        x = self.fc(x)
        
        # 如果输入是窗口化数据，对窗口内的预测求平均
        if window_size > 1:
            x = x.reshape(batch_size, window_size, 1).mean(dim=1)

        return x


if __name__ == "__main__":
    # Test the models
    batch_size = 16
    input_size = 6
    x = torch.randn(batch_size, input_size)

    print("Testing baseline models...")
    print("=" * 50)

    # Test FNN
    fnn = FNN(input_size=input_size)
    fnn_out = fnn(x)
    print(f"FNN:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {fnn_out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in fnn.parameters())}")
    print()

    # Test CNN
    cnn = CNN(input_size=input_size)
    cnn_out = cnn(x)
    print(f"CNN:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {cnn_out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in cnn.parameters())}")
    print()

    # Test LSTM
    lstm = LSTM(input_size=input_size)
    lstm_out = lstm(x)
    print(f"LSTM:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {lstm_out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in lstm.parameters())}")
    print()

    # Test GRU
    gru = GRU(input_size=input_size)
    gru_out = gru(x)
    print(f"GRU:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {gru_out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in gru.parameters())}")
    print()

    # Test ResCNN
    rescnn = ResCNN(input_size=input_size)
    rescnn_out = rescnn(x)
    print(f"ResCNN:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {rescnn_out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in rescnn.parameters())}")
    print()

    print("All models tested successfully!")


# ============================================================================
# XGBoost Baseline Models (PyTorch-compatible wrapper)
# ============================================================================

class XGBoost_Simple(nn.Module):
    """
    XGBoost with simple flattened window input.

    Baseline 1: Demonstrates the limitation of treating temporal data as flat features.
    Input: (batch, window_size, features) → flatten to (batch, window*features)
    Expected RMSE: 3.5-4.5%

    Purpose: Prove the necessity of temporal structure modeling.
    """

    def __init__(self, input_size=16, window_size=40,
                 n_estimators=300, max_depth=6, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8,
                 reg_alpha=0.1, reg_lambda=1.0, random_state=42):
        """
        Args:
            input_size: Number of features per cycle
            window_size: Number of historical cycles
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of features
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            random_state: Random seed
        """
        super(XGBoost_Simple, self).__init__()

        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        self.input_size = input_size
        self.window_size = window_size
        self.feature_dim = window_size * input_size

        self.xgb_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'verbosity': 0
        }

        self.model = xgb.XGBRegressor(**self.xgb_params)
        self.is_fitted = False

    def flatten_input(self, x):
        """Flatten 3D window to 2D."""
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        if len(x.shape) == 3:
            batch_size = x.shape[0]
            return x.reshape(batch_size, -1)
        return x

    def forward(self, x):
        """
        Forward pass for inference.

        Note: Training uses fit() method instead of forward().
        """
        if not self.is_fitted:
            raise RuntimeError("XGBoost model must be fitted before forward(). Use fit() method.")

        x_flat = self.flatten_input(x)
        predictions = self.model.predict(x_flat)

        # Return as PyTorch tensor for compatibility
        return torch.tensor(predictions, dtype=torch.float32).unsqueeze(1)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=False):
        """
        Fit the XGBoost model.

        Args:
            X_train: Training features (N, window, features) or (N, flat_features)
            y_train: Training targets (N,)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            verbose: Print training progress
        """
        X_train_flat = self.flatten_input(X_train)

        if isinstance(y_train, torch.Tensor):
            y_train = y_train.cpu().numpy().flatten()

        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_flat = self.flatten_input(X_val)
            if isinstance(y_val, torch.Tensor):
                y_val = y_val.cpu().numpy().flatten()
            eval_set = [(X_val_flat, y_val)]

        self.model.fit(
            X_train_flat, y_train,
            eval_set=eval_set,
            verbose=verbose
        )

        self.is_fitted = True
        return self


class XGBoost_Enhanced(nn.Module):
    """
    XGBoost with engineered rolling/lag features.

    Baseline 2: Shows that even with careful feature engineering,
    end-to-end learning (CNN-LSTM) can still outperform.

    Features include:
    - Lag features (t-1, t-2, ..., t-k)
    - Rolling statistics (mean, std, min, max, slope)
    - Change rates (delta, acceleration)
    - Raw feature statistics

    Expected RMSE: 2.5-3.5%
    """

    def __init__(self, input_size=16, window_size=40,
                 n_lags=10, rolling_windows=[5, 10, 20],
                 n_estimators=500, max_depth=8, learning_rate=0.03,
                 subsample=0.8, colsample_bytree=0.8,
                 reg_alpha=0.1, reg_lambda=1.0, random_state=42):
        """
        Args:
            input_size: Number of features per cycle
            window_size: Number of historical cycles
            n_lags: Number of lag features to create
            rolling_windows: List of window sizes for rolling statistics
            (other params same as XGBoost_Simple)
        """
        super(XGBoost_Enhanced, self).__init__()

        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        self.input_size = input_size
        self.window_size = window_size
        self.n_lags = n_lags
        self.rolling_windows = rolling_windows

        # Calculate feature dimension
        lag_feats = n_lags * input_size
        rolling_feats = len(rolling_windows) * 5 * input_size  # 5 stats per window
        change_feats = 2 * input_size  # delta + acceleration
        raw_feats = input_size
        self.feature_dim = lag_feats + rolling_feats + change_feats + raw_feats

        self.xgb_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'verbosity': 0
        }

        self.model = xgb.XGBRegressor(**self.xgb_params)
        self.is_fitted = False

    def engineer_features(self, x):
        """
        Engineer temporal features from window.

        Args:
            x: (batch, window_size, input_size)

        Returns:
            Engineered features (batch, feature_dim)
        """
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        if len(x.shape) == 2:
            return x  # Already processed

        batch_size, window_size, input_size = x.shape
        features_list = []

        # 1. Lag features (t-1, t-2, ..., t-k)
        for lag in range(1, self.n_lags + 1):
            if lag < window_size:
                lag_feat = x[:, -lag, :]
                features_list.append(lag_feat)
            else:
                features_list.append(np.zeros((batch_size, input_size)))

        # 2. Rolling statistics
        for window in self.rolling_windows:
            if window <= window_size:
                recent = x[:, -window:, :]

                # Mean, Std, Min, Max
                features_list.append(np.mean(recent, axis=1))
                features_list.append(np.std(recent, axis=1))
                features_list.append(np.min(recent, axis=1))
                features_list.append(np.max(recent, axis=1))

                # Slope (linear trend)
                time_idx = np.arange(window)
                slopes = []
                for b in range(batch_size):
                    slope_per_feat = []
                    for f in range(input_size):
                        y = recent[b, :, f]
                        slope = np.polyfit(time_idx, y, 1)[0]
                        slope_per_feat.append(slope)
                    slopes.append(slope_per_feat)
                features_list.append(np.array(slopes))
            else:
                for _ in range(5):
                    features_list.append(np.zeros((batch_size, input_size)))

        # 3. Change rate features
        if window_size >= 2:
            delta = x[:, -1, :] - x[:, -2, :]
            features_list.append(delta)
        else:
            features_list.append(np.zeros((batch_size, input_size)))

        if window_size >= 3:
            delta_2 = x[:, -2, :] - x[:, -3, :]
            acceleration = delta - delta_2
            features_list.append(acceleration)
        else:
            features_list.append(np.zeros((batch_size, input_size)))

        # 4. Last cycle raw features
        features_list.append(x[:, -1, :])

        # Concatenate
        engineered = np.concatenate(features_list, axis=1)
        return engineered

    def forward(self, x):
        """Forward pass for inference."""
        if not self.is_fitted:
            raise RuntimeError("XGBoost model must be fitted before forward(). Use fit() method.")

        x_eng = self.engineer_features(x)
        predictions = self.model.predict(x_eng)

        return torch.tensor(predictions, dtype=torch.float32).unsqueeze(1)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=False):
        """Fit the XGBoost model with engineered features."""
        X_train_eng = self.engineer_features(X_train)

        if isinstance(y_train, torch.Tensor):
            y_train = y_train.cpu().numpy().flatten()

        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_eng = self.engineer_features(X_val)
            if isinstance(y_val, torch.Tensor):
                y_val = y_val.cpu().numpy().flatten()
            eval_set = [(X_val_eng, y_val)]

        self.model.fit(
            X_train_eng, y_train,
            eval_set=eval_set,
            verbose=verbose
        )

        self.is_fitted = True
        return self
