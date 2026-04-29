"""
V2 架构实验模块 —— 独立支线，不影响原有任何代码

包含两个架构创新（均可独立开关）：
  M3: Per-Window LayerNorm —— 对每个输入窗口在特征维独立归一化，
       消除电池生命周期晚期特征漂移对模型的干扰。
  MS: 多尺度 CNN —— 3 路并行 CNN（kernel=3/7/15），分别捕捉
       短程、中程、长程的充电曲线模式，融合后送入 LSTM。

对照关系：
  ┌──────────────────────────────────────────────────────────────┐
  │ use_multiscale=False, per_window_norm=False  →  等价于旧版   │
  │ use_multiscale=False, per_window_norm=True   →  +M3（仅归一化）│
  │ use_multiscale=True,  per_window_norm=False  →  +MS（仅多尺度）│
  │ use_multiscale=True,  per_window_norm=True   →  M3 + MS 联合  │
  └──────────────────────────────────────────────────────────────┘

使用方式（ModelFactory）：
    model_type = 'ms_cnn_lstm_v2'
    config_override = {'architecture': {'use_multiscale': True, 'per_window_norm': True}}
"""

import torch
import torch.nn as nn


def _make_single_branch(in_ch: int, channels: list, kernel_size: int,
                        pool_size: int, activation: nn.Module) -> nn.Sequential:
    """
    构建单个 CNN 分支（多层 Conv-BN-Act-Pool）。

    Args:
        in_ch:      输入通道数（= input_size，特征维度）
        channels:   各层输出通道列表，如 [64, 64]
        kernel_size: 卷积核大小
        pool_size:  MaxPool 步长
        activation: 激活函数实例
    Returns:
        nn.Sequential 分支模块
    """
    layers = []
    for out_ch in channels:
        padding = kernel_size // 2          # same-padding 保持序列长度
        layers += [
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            activation,
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size),
        ]
        in_ch = out_ch
    return nn.Sequential(*layers)


class MultiScaleCNN_LSTM(nn.Module):
    """
    多尺度 CNN-LSTM V2

    配置参数（均来自 architecture 字典）：
      input_size          int,   输入特征维度（运行时自动注入）
      hidden_size         int,   LSTM 隐状态维度，默认 64
      num_layers          int,   LSTM 层数，默认 2
      dropout_rate        float, FC Dropout 比率，默认 0.4
      fc_hidden_sizes     list,  FC 隐层尺寸，默认 [64]
      activation          str,   激活函数名，默认 'ReLU'
      output_activation   str,   输出激活，默认 'Sigmoid'

      use_multiscale      bool,  是否启用多尺度 CNN，默认 True
      multiscale_kernels  list,  各分支的 kernel size，默认 [3, 7, 15]
      branch_channels     list,  每个分支的 channel 列表，默认 [64, 64]
      fused_channels      int,   融合后通道数（=LSTM 输入维度），默认 128
      pool_size           int,   MaxPool 步长，默认 2

      per_window_norm     bool,  是否启用 M3 LayerNorm，默认 False
      cnn_channels        list,  单路 CNN 通道（use_multiscale=False 时用），默认 [256, 128]
      kernel_size         int,   单路 CNN kernel（use_multiscale=False 时用），默认 7
    """

    def __init__(self, config: dict):
        super().__init__()
        arch = config['architecture']

        input_size   = arch['input_size']
        hidden_size  = arch.get('hidden_size', 64)
        num_layers   = arch.get('num_layers', 2)
        dropout_rate = arch.get('dropout_rate', 0.4)
        fc_sizes     = arch.get('fc_hidden_sizes', [64])
        pool_size    = arch.get('pool_size', 2)

        act_name = arch.get('activation', 'ReLU')
        self.activation = getattr(nn, act_name)()

        out_act_name = arch.get('output_activation', 'Sigmoid')
        self.output_activation = getattr(nn, out_act_name)() if out_act_name else None

        # ── M3: Per-Window LayerNorm ───────────────────────────────────────
        self.per_window_norm = arch.get('per_window_norm', False)
        if self.per_window_norm:
            # 归一化最后一维（特征维），每个时间步独立
            self.input_norm = nn.LayerNorm(input_size)

        # ── CNN 部分 ─────────────────────────────────────────────────────────
        self.use_multiscale = arch.get('use_multiscale', True)

        if self.use_multiscale:
            # 多尺度：3 路并行 CNN
            kernels        = arch.get('multiscale_kernels', [3, 7, 15])
            branch_chs     = arch.get('branch_channels', [64, 64])
            self.fused_ch  = arch.get('fused_channels', 128)
            n_branches     = len(kernels)

            self.branches = nn.ModuleList([
                _make_single_branch(input_size, branch_chs, k, pool_size, self.activation)
                for k in kernels
            ])
            # 1×1 Conv 将 n_branches×branch_chs[-1] 通道融合到 fused_ch
            concat_ch = n_branches * branch_chs[-1]
            self.fusion = nn.Sequential(
                nn.Conv1d(concat_ch, self.fused_ch, kernel_size=1),
                nn.BatchNorm1d(self.fused_ch),
                self.activation,
            )
            lstm_input_size = self.fused_ch
        else:
            # 单路 CNN（与原 CNN_LSTM 逻辑相同）
            cnn_channels = arch.get('cnn_channels', [256, 128])
            kernel_size  = arch.get('kernel_size', 7)
            self.conv_layers = nn.ModuleList()
            in_ch = input_size
            for out_ch in cnn_channels:
                padding = kernel_size // 2
                self.conv_layers.append(nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(out_ch),
                    self.activation,
                    nn.MaxPool1d(kernel_size=pool_size, stride=pool_size),
                ))
                in_ch = out_ch
            lstm_input_size = cnn_channels[-1]

        # ── LSTM ─────────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )

        # ── FC 头 ────────────────────────────────────────────────────────────
        fc_layers = []
        prev = hidden_size
        for fc_sz in fc_sizes:
            fc_layers += [
                nn.Linear(prev, fc_sz),
                self.activation,
                nn.Dropout(dropout_rate),
            ]
            prev = fc_sz
        fc_layers.append(nn.Linear(prev, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, window_size, input_size)  或  (B, input_size)
        Returns:
            (B, 1)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)   # → (B, 1, input_size)

        # M3: 对 (B, T, C) 最后一维归一化，再做 permute
        if self.per_window_norm:
            x = self.input_norm(x)   # (B, T, C)

        # permute → (B, C, T) 送 Conv1d
        x = x.permute(0, 2, 1)

        if self.use_multiscale:
            # 三路并联 CNN
            outs = [branch(x) for branch in self.branches]   # 每路 (B, branch_ch[-1], T')
            x = torch.cat(outs, dim=1)                        # (B, n*branch_ch[-1], T')
            x = self.fusion(x)                                # (B, fused_ch, T')
        else:
            for conv_layer in self.conv_layers:
                x = conv_layer(x)
                if x.size(2) == 0:
                    raise ValueError(
                        "CNN 输出序列长度为 0，请检查 window_size 与 pool_size 配置。"
                    )

        # permute → (B, T', C) 送 LSTM
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]    # (B, hidden_size)

        out = self.fc(last)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out
