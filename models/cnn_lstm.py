"""
CNN-LSTM模型：结合卷积神经网络和长短期记忆网络
- CNN用于提取局部特征模式（电压、电流、温度等）
- LSTM用于捕捉长期时序依赖关系（容量衰减趋势）
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
            x: 输入张量，shape: (batch_size, seq_len, input_size)

        Returns:
            output: SOH预测，shape: (batch_size, 1)
        """
        batch_size, seq_len, input_size = x.shape

        # 转换为CNN输入格式: (batch_size, input_size, seq_len)
        x = x.permute(0, 2, 1)

        # CNN特征提取
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

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
