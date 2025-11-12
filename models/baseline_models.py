"""
Baseline models for comparison: LSTM, CNN, FNN.
用于与BPINN对比的基准模型。
"""

import torch
import torch.nn as nn


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

        Returns:
            SOH predictions (batch_size, 1)
        """
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
            x: Input features (batch_size, input_size)

        Returns:
            SOH predictions (batch_size, 1)
        """
        # Reshape to (batch, 1, input_size) for 1D conv
        x = x.unsqueeze(1)  # (batch, 1, input_size)

        # Convolution
        x = self.conv1(x)   # (batch, num_filters, input_size)
        x = self.relu1(x)
        x = self.pool1(x)   # (batch, num_filters, 1)

        # Flatten
        x = x.squeeze(-1)   # (batch, num_filters)

        # Fully connected
        x = self.fc_network(x)

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
            input_size=1,  # 每次输入一个特征
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
            x: Input features (batch_size, input_size)

        Returns:
            SOH predictions (batch_size, 1)
        """
        # Reshape to (batch, seq_len, input_dim) for LSTM
        # Treat each feature as a time step
        x = x.unsqueeze(-1)  # (batch, input_size, 1)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        x = h_n[-1]  # (batch, hidden_size)

        # Fully connected
        x = self.fc_network(x)

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

    print("All models tested successfully!")
