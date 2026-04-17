"""
M2：MC Dropout 包装
===================
测试时保持 Dropout 激活，通过多次随机前向传播构建预测分布。

使用方式：
    1. 将模型 FC 头中的 nn.Dropout 替换为 MCDropout（通过 config 控制）
    2. 推理时调用 mc_predict(model, x, n_samples=50)
       → 返回 (mean, std)，mean 为点估计，std 为不确定性

论文说辞：
    本文通过测试时保留 Dropout 进行多次随机前向传播，构建 SOH 预测的
    置信区间，为 BMS 决策提供不确定性量化支持。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MCDropout(nn.Dropout):
    """
    始终激活的 Dropout 层（训练和推理阶段均有效）。

    继承 nn.Dropout，仅覆盖 forward()，强制 training=True。
    其余层（BatchNorm 等）可正常使用 model.eval() 的运行统计量。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # training=True 强制激活，与模型是否在 eval 模式无关
        return F.dropout(x, p=self.p, training=True, inplace=self.inplace)


def mc_predict(model: nn.Module, x: torch.Tensor, n_samples: int = 50):
    """
    MC Dropout 多次采样预测。

    Args:
        model:     已训练的模型（FC 头中含 MCDropout 层）
        x:         输入张量，shape (B, T, F) 或 (B, F)
        n_samples: 采样次数（默认 50）

    Returns:
        mean: 预测均值，shape (B, 1)
        std:  预测标准差（不确定性），shape (B, 1)
    """
    model.eval()   # BN 使用运行统计量；MCDropout 层始终激活

    with torch.no_grad():
        preds = torch.stack(
            [model(x) for _ in range(n_samples)],
            dim=0
        )  # (n_samples, B, 1)

    mean = preds.mean(dim=0)  # (B, 1)
    std  = preds.std(dim=0)   # (B, 1)

    return mean, std
