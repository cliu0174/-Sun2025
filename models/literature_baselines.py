"""Literature-derived neural baselines for fair Chapter 4 comparison.

These implementations reproduce the central network designs while keeping the
project's fixed split, 40-cycle input window, feature set, label mask, training
schedule, and validation protocol unchanged.  They deliberately do not copy
dataset-specific feature engineering or meta-heuristic tuning from the source
papers; those additions would make the comparison protocol unequal.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class TransformerSOH(nn.Module):
    """Transformer regressor following Shu et al. (JES, 2025).

    The input is a local sequence of cycle-level charging features.  A learned
    positional encoding and a Transformer encoder replace the source paper's
    voltage-segment-specific front end so that the model receives exactly the
    same 40-cycle, 16-feature windows as every neural comparator.
    """

    def __init__(self, config: dict):
        super().__init__()
        arch = config["architecture"]
        input_size = int(arch["input_size"])
        d_model = int(arch.get("d_model", 64))
        nhead = int(arch.get("nhead", 4))
        num_layers = int(arch.get("num_layers", 2))
        dim_feedforward = int(arch.get("dim_feedforward", 128))
        dropout = float(arch.get("dropout_rate", 0.1))
        max_sequence_length = int(arch.get("max_sequence_length", 64))

        if d_model % nhead:
            raise ValueError("Transformer d_model must be divisible by nhead")

        self.input_projection = nn.Linear(input_size, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_sequence_length, d_model)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, int(arch.get("head_hidden_size", 64))),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(arch.get("head_hidden_size", 64)), 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.size(1) > self.position_embedding.size(1):
            raise ValueError("input window exceeds Transformer maximum sequence length")

        scale = math.sqrt(self.input_projection.out_features)
        x = self.input_projection(x) * scale
        x = x + self.position_embedding[:, : x.size(1)]
        x = self.norm(self.encoder(x))
        return self.head(x[:, -1, :])


class CNNBiGRUAttention(nn.Module):
    """CNN--BiGRU--attention regressor following Wu et al. (WEVJ, 2025)."""

    def __init__(self, config: dict):
        super().__init__()
        arch = config["architecture"]
        input_size = int(arch["input_size"])
        cnn_channels = [int(value) for value in arch.get("cnn_channels", [64, 64])]
        kernel_size = int(arch.get("kernel_size", 3))
        hidden_size = int(arch.get("hidden_size", 64))
        num_layers = int(arch.get("num_layers", 2))
        dropout = float(arch.get("dropout_rate", 0.1))

        layers: list[nn.Module] = []
        channels_in = input_size
        for channels_out in cnn_channels:
            layers.extend(
                [
                    nn.Conv1d(
                        channels_in,
                        channels_out,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(channels_out),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            channels_in = channels_out
        self.cnn = nn.Sequential(*layers)
        self.bigru = nn.GRU(
            input_size=channels_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        representation_size = hidden_size * 2
        attention_size = int(arch.get("attention_size", 64))
        self.attention_score = nn.Sequential(
            nn.Linear(representation_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1, bias=False),
        )
        self.head = nn.Sequential(
            nn.Linear(representation_size, int(arch.get("head_hidden_size", 64))),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(arch.get("head_hidden_size", 64)), 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.bigru(x)
        weights = torch.softmax(self.attention_score(sequence).squeeze(-1), dim=1)
        context = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return self.head(context)
