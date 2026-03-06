"""
Sequence-to-Sequence (Many-to-Many) 时序模型

用于支持物理约束的序列预测版本。
与 baseline_models.py 中的 Many-to-One 模型不同，这些模型输出整个序列。

模型输出:
- Many-to-One: (batch, 1) - 单个SOH值
- Many-to-Many: (batch, seq_len, 1) - 整个序列的SOH值
"""

import torch
import torch.nn as nn


class LSTMSeq2Seq(nn.Module):
    """
    LSTM Sequence-to-Sequence 模型

    输入: (batch, seq_len, features)
    输出: (batch, seq_len, 1) - 预测整个序列的SOH
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # 全连接层：对每个时间步预测SOH
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # SOH在[0,1]范围
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            output: (batch, seq_len, 1) - 每个时间步的SOH预测
        """
        # LSTM处理
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * directions)

        # 对每个时间步应用全连接层
        # 方法1: 循环处理（清晰但慢）
        # outputs = []
        # for t in range(lstm_out.size(1)):
        #     out_t = self.fc(lstm_out[:, t, :])
        #     outputs.append(out_t)
        # output = torch.stack(outputs, dim=1)

        # 方法2: 批量处理（快速）
        batch_size, seq_len, hidden_dim = lstm_out.size()
        lstm_out_flat = lstm_out.reshape(-1, hidden_dim)  # (batch*seq_len, hidden)
        output_flat = self.fc(lstm_out_flat)  # (batch*seq_len, 1)
        output = output_flat.reshape(batch_size, seq_len, 1)  # (batch, seq_len, 1)

        return output


class GRUSeq2Seq(nn.Module):
    """
    GRU Sequence-to-Sequence 模型

    输入: (batch, seq_len, features)
    输出: (batch, seq_len, 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # GRU层
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # 全连接层
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            output: (batch, seq_len, 1)
        """
        gru_out, _ = self.gru(x)

        batch_size, seq_len, hidden_dim = gru_out.size()
        gru_out_flat = gru_out.reshape(-1, hidden_dim)
        output_flat = self.fc(gru_out_flat)
        output = output_flat.reshape(batch_size, seq_len, 1)

        return output


class BiLSTMSeq2Seq(nn.Module):
    """
    双向LSTM Sequence-to-Sequence 模型

    输入: (batch, seq_len, features)
    输出: (batch, seq_len, 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.2
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True  # 双向
        )

        # 全连接层（输入是 hidden_size*2）
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            output: (batch, seq_len, 1)
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)

        batch_size, seq_len, hidden_dim = lstm_out.size()
        lstm_out_flat = lstm_out.reshape(-1, hidden_dim)
        output_flat = self.fc(lstm_out_flat)
        output = output_flat.reshape(batch_size, seq_len, 1)

        return output


class BiGRUSeq2Seq(nn.Module):
    """
    双向GRU Sequence-to-Sequence 模型

    输入: (batch, seq_len, features)
    输出: (batch, seq_len, 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.2
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 双向GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            output: (batch, seq_len, 1)
        """
        gru_out, _ = self.gru(x)

        batch_size, seq_len, hidden_dim = gru_out.size()
        gru_out_flat = gru_out.reshape(-1, hidden_dim)
        output_flat = self.fc(gru_out_flat)
        output = output_flat.reshape(batch_size, seq_len, 1)

        return output


# ==================== 测试代码 ====================

def test_seq2seq_models():
    """测试Seq2Seq模型"""
    print("="*70)
    print("测试 Seq2Seq 模型")
    print("="*70)

    batch_size = 4
    seq_len = 20
    input_size = 16

    # 创建测试数据
    x = torch.randn(batch_size, seq_len, input_size)

    print(f"\n输入形状: {x.shape}")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  input_size: {input_size}")

    models = {
        'LSTMSeq2Seq': LSTMSeq2Seq(input_size, hidden_size=64, num_layers=2),
        'GRUSeq2Seq': GRUSeq2Seq(input_size, hidden_size=64, num_layers=2),
        'BiLSTMSeq2Seq': BiLSTMSeq2Seq(input_size, hidden_size=64, num_layers=2),
        'BiGRUSeq2Seq': BiGRUSeq2Seq(input_size, hidden_size=64, num_layers=2),
    }

    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"测试 {name}")
        print(f"{'='*70}")

        # 前向传播
        output = model(x)

        print(f"输出形状: {output.shape}")
        print(f"  预期: ({batch_size}, {seq_len}, 1)")
        print(f"  实际: {tuple(output.shape)}")

        # 验证输出范围（应该在[0,1]）
        print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")

        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {total_params:,}")

        # 测试单调性约束计算
        print(f"\n测试单调性约束:")
        diff = output[:, 1:, 0] - output[:, :-1, 0]  # (batch, seq_len-1)
        print(f"  diff形状: {diff.shape}")
        violations = torch.relu(diff)
        mono_loss = torch.mean(violations ** 2)
        print(f"  单调性损失: {mono_loss.item():.6f}")

        assert output.shape == (batch_size, seq_len, 1), f"{name} 输出形状错误"
        print(f"  [OK] {name} 测试通过")

    print("\n" + "="*70)
    print("所有模型测试通过！")
    print("="*70)


if __name__ == "__main__":
    test_seq2seq_models()
