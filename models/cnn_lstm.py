"""
CNN混合模型：CNN-LSTM、CNN-BiLSTM、CNN-MLP
- CNN-LSTM: 结合卷积神经网络和长短期记忆网络
- CNN-BiLSTM: 结合CNN特征提取和双向LSTM时序建模
- CNN-MLP: 仅使用CNN+MLP，作为消融实验baseline
"""

import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    """
    CNN-LSTM混合模型

    架构：
    1. CNN层：提取时间窗口内的局部特征
    2. LSTM层：建模长期时序依赖关系
    3. 全连接层：输出SOH预测
    """

    def __init__(self, config):
        super(CNN_LSTM, self).__init__()

        # 从配置中获取参数
        input_size = config['architecture']['input_size']

        # input_size必须是正数
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}. "
                           "Please set input_size in the config or pass it as a parameter.")
        hidden_size = config['architecture']['hidden_size']
        num_layers = config['architecture']['num_layers']
        dropout_rate = config['architecture']['dropout_rate']
        cnn_channels = config['architecture'].get('cnn_channels', [32, 64])
        kernel_size = config['architecture'].get('kernel_size', 3)
        pool_size = config['architecture'].get('pool_size', 2)
        fc_hidden_sizes = config['architecture'].get('fc_hidden_sizes', [64])

        # 激活函数
        activation_name = config['architecture'].get('activation', 'ReLU')
        self.activation = getattr(nn, activation_name)()

        # 输出激活函数
        output_activation = config['architecture'].get('output_activation', 'Sigmoid')
        self.output_activation = getattr(nn, output_activation)() if output_activation else None

        # CNN部分：提取局部特征
        self.conv_layers = nn.ModuleList()
        in_channels = input_size

        for out_channels in cnn_channels:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(out_channels),
                    self.activation,
                    nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
                )
            )
            in_channels = out_channels

        # LSTM部分：建模时序依赖
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],  # CNN最后一层的输出通道数
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        # 全连接层
        fc_layers = []
        prev_size = hidden_size

        for fc_size in fc_hidden_sizes:
            fc_layers.extend([
                nn.Linear(prev_size, fc_size),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            prev_size = fc_size

        # 输出层
        fc_layers.append(nn.Linear(prev_size, 1))

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，shape: (batch_size, seq_len, input_size) 或 (batch_size, input_size)

        Returns:
            output: SOH预测，shape: (batch_size, 1)
        """
        # 处理2维输入（window_size=1的情况）
        if x.dim() == 2:
            # (batch_size, input_size) -> (batch_size, 1, input_size)
            x = x.unsqueeze(1)

        # 转换为CNN输入格式: (batch_size, input_size, seq_len)
        x = x.permute(0, 2, 1)

        # CNN特征提取
        # 注意：当seq_len太小时，MaxPool可能导致输出为0
        # 这种情况下，全局平均池化会更稳定
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            # 如果序列长度太短，跳过池化以避免输出维度为0
            if x.size(2) == 0:
                raise ValueError(f"CNN output sequence length is 0. Input sequence may be too short for the configured pooling.")

        # 转换为LSTM输入格式: (batch_size, seq_len, channels)
        x = x.permute(0, 2, 1)

        # LSTM时序建模
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]

        # 全连接层输出
        output = self.fc(last_output)

        # 应用输出激活函数
        if self.output_activation is not None:
            output = self.output_activation(output)

        return output


class CNN_BiLSTM(nn.Module):
    """
    CNN-BiLSTM混合模型

    架构：
    1. CNN层：提取时间窗口内的局部特征
    2. BiLSTM层：双向建模长期时序依赖关系
    3. 全连接层：输出SOH预测
    """

    def __init__(self, config):
        super(CNN_BiLSTM, self).__init__()

        # 从配置中获取参数
        input_size = config['architecture']['input_size']

        # input_size必须是正数
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}. "
                           "Please set input_size in the config or pass it as a parameter.")
        hidden_size = config['architecture']['hidden_size']
        num_layers = config['architecture']['num_layers']
        dropout_rate = config['architecture']['dropout_rate']
        cnn_channels = config['architecture'].get('cnn_channels', [32, 64])
        kernel_size = config['architecture'].get('kernel_size', 3)
        pool_size = config['architecture'].get('pool_size', 2)
        fc_hidden_sizes = config['architecture'].get('fc_hidden_sizes', [64])

        # 激活函数
        activation_name = config['architecture'].get('activation', 'ReLU')
        self.activation = getattr(nn, activation_name)()

        # 输出激活函数
        output_activation = config['architecture'].get('output_activation', 'Sigmoid')
        self.output_activation = getattr(nn, output_activation)() if output_activation else None

        # CNN部分：提取局部特征
        self.conv_layers = nn.ModuleList()
        in_channels = input_size

        for out_channels in cnn_channels:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(out_channels),
                    self.activation,
                    nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
                )
            )
            in_channels = out_channels

        # BiLSTM部分：双向建模时序依赖
        self.bilstm = nn.LSTM(
            input_size=cnn_channels[-1],  # CNN最后一层的输出通道数
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True  # 关键差异：双向LSTM
        )

        # 全连接层 - 注意BiLSTM输出是2*hidden_size
        fc_layers = []
        prev_size = hidden_size * 2  # 双向LSTM输出维度是hidden_size的2倍

        for fc_size in fc_hidden_sizes:
            fc_layers.extend([
                nn.Linear(prev_size, fc_size),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            prev_size = fc_size

        # 输出层
        fc_layers.append(nn.Linear(prev_size, 1))

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，shape: (batch_size, seq_len, input_size) 或 (batch_size, input_size)

        Returns:
            output: SOH预测，shape: (batch_size, 1)
        """
        # 处理2维输入（window_size=1的情况）
        if x.dim() == 2:
            # (batch_size, input_size) -> (batch_size, 1, input_size)
            x = x.unsqueeze(1)

        # 转换为CNN输入格式: (batch_size, input_size, seq_len)
        x = x.permute(0, 2, 1)

        # CNN特征提取
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # 转换为BiLSTM输入格式: (batch_size, seq_len, channels)
        x = x.permute(0, 2, 1)

        # BiLSTM时序建模（双向）
        lstm_out, (h_n, c_n) = self.bilstm(x)

        # 使用最后一个时间步的输出（包含前向和后向信息）
        last_output = lstm_out[:, -1, :]

        # 全连接层输出
        output = self.fc(last_output)

        # 应用输出激活函数
        if self.output_activation is not None:
            output = self.output_activation(output)

        return output


class CNN_MLP(nn.Module):
    """
    CNN-MLP模型（消融实验baseline）

    架构：
    1. CNN层：提取时间窗口内的局部特征
    2. 全局池化：将时序特征聚合
    3. MLP层：直接映射到SOH预测

    说明：不使用循环结构，验证CNN特征提取的有效性
    """

    def __init__(self, config):
        super(CNN_MLP, self).__init__()

        # 从配置中获取参数
        input_size = config['architecture']['input_size']

        # input_size必须是正数
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}. "
                           "Please set input_size in the config or pass it as a parameter.")
        dropout_rate = config['architecture']['dropout_rate']
        cnn_channels = config['architecture'].get('cnn_channels', [32, 64])
        kernel_size = config['architecture'].get('kernel_size', 3)
        pool_size = config['architecture'].get('pool_size', 2)
        fc_hidden_sizes = config['architecture'].get('fc_hidden_sizes', [128, 64, 32])

        # 激活函数
        activation_name = config['architecture'].get('activation', 'ReLU')
        self.activation = getattr(nn, activation_name)()

        # 输出激活函数
        output_activation = config['architecture'].get('output_activation', 'Sigmoid')
        self.output_activation = getattr(nn, output_activation)() if output_activation else None

        # CNN部分：提取局部特征
        self.conv_layers = nn.ModuleList()
        in_channels = input_size

        for out_channels in cnn_channels:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(out_channels),
                    self.activation,
                    nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
                )
            )
            in_channels = out_channels

        # 全局平均池化：将时序信息聚合为固定长度向量
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # MLP部分：直接映射特征到输出
        mlp_layers = []
        prev_size = cnn_channels[-1]  # CNN最后一层的输出通道数

        for fc_size in fc_hidden_sizes:
            mlp_layers.extend([
                nn.Linear(prev_size, fc_size),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            prev_size = fc_size

        # 输出层
        mlp_layers.append(nn.Linear(prev_size, 1))

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，shape: (batch_size, seq_len, input_size) 或 (batch_size, input_size)

        Returns:
            output: SOH预测，shape: (batch_size, 1)
        """
        # 处理2维输入（window_size=1的情况）
        if x.dim() == 2:
            # (batch_size, input_size) -> (batch_size, input_size, 1)
            x = x.unsqueeze(-1)
        else:
            # (batch_size, seq_len, input_size) -> (batch_size, input_size, seq_len)
            x = x.permute(0, 2, 1)

        # CNN特征提取
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # 全局平均池化: (batch_size, channels, seq_len) -> (batch_size, channels, 1)
        x = self.global_pool(x)

        # 展平: (batch_size, channels, 1) -> (batch_size, channels)
        x = x.squeeze(-1)

        # MLP映射
        output = self.mlp(x)

        # 应用输出激活函数
        if self.output_activation is not None:
            output = self.output_activation(output)

        return output
