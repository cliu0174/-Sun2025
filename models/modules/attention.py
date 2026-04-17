"""
M1：循环级注意力（Cycle-level Attention）
=========================================
在 CNN-LSTM 的 LSTM 输出层之后插入 Multi-Head Self-Attention，
让模型自适应聚焦对当前 SOH 估计贡献最大的历史循环区间。

结构：
    LSTM output (B, T, H)
        → Multi-Head Self-Attention (4 heads)
        → Residual + LayerNorm
        → Global Average Pool
        → (B, H)  ← 与原始 last_output 维度相同，FC 头无需修改

论文说辞：
    电池退化过程中存在信息不对称性——近期循环与历史特定退化节点对
    当前健康状态的贡献不尽相同。本文在 CNN-LSTM 的时序表征层引入
    循环级注意力机制，使模型能够自适应地聚焦退化敏感的历史区间。
"""

import torch
import torch.nn as nn


class CycleAttention(nn.Module):
    """
    循环级 Multi-Head Self-Attention 模块。

    Args:
        hidden_size: LSTM 隐藏层维度，必须能被 num_heads 整除
        num_heads:   注意力头数（默认 4）
        dropout:     注意力 dropout（默认 0.1）
    """

    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) 必须能被 num_heads ({num_heads}) 整除"
            )

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True   # 输入格式 (B, T, H)
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: LSTM 输出序列，shape (B, T, H)

        Returns:
            out:          全局聚合向量，shape (B, H)
            attn_weights: 注意力权重，shape (B, T, T)，可用于可视化
        """
        # Self-Attention：Q=K=V=x
        attn_out, attn_weights = self.attn(x, x, x)

        # 残差连接 + LayerNorm
        x = self.norm(x + self.dropout(attn_out))  # (B, T, H)

        # Global Average Pool：所有时间步等权平均 → (B, H)
        out = x.mean(dim=1)

        return out, attn_weights
