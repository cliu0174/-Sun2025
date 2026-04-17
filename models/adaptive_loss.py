"""
M5：自适应损失权重（Uncertainty-Weighted Multi-task Loss）
==========================================================
基于 Kendall et al. (2018) "Multi-Task Learning Using Uncertainty to Weigh Losses"

公式：
    L = Σ_i [ exp(-log_σᵢ) · Lᵢ + log_σᵢ ]

其中 log_σᵢ 为可学习参数，替代人工设定的固定权重，自动平衡各损失项。

损失项（4 项，对应索引 0-3）：
    0: base       — masked MSE 数据拟合
    1: monotonic  — 软单调性约束
    2: boundary   — 边界约束 SOH ∈ [0, 1]
    3: smoothness — 速率连续性约束（M4 的二阶差分项）

使用注意：
    log_vars 是可训练参数，必须加入优化器参数组：
        optimizer.add_param_group({
            'params': [criterion.log_vars],
            'lr': config['training']['learning_rate'],
        })

论文说辞：
    "在部分生命周期监督场景中，各损失项的相对重要性随训练进展动态变化——
    数据稀疏阶段物理约束贡献更大，数据充足时数据拟合项应占主导。本文采用
    基于不确定性的多任务加权框架，以可学习参数 log σᵢ 自适应调整各损失项权重，
    消除人工调参负担，并为损失项的贡献提供可解释的不确定性估计。"
"""

import torch
import torch.nn as nn

from .physics_loss import PhysicsConstrainedLoss


class AdaptivePhysicsLoss(nn.Module):
    """
    自适应权重物理约束损失函数。

    包装 PhysicsConstrainedLoss，用可学习 log σᵢ 替代固定权重。

    Args:
        physics_loss:  PhysicsConstrainedLoss 实例（负责计算各项损失）
        init_log_vars: 初始 log σᵢ 列表（长度 4）；None 则全为 0.0
                       建议根据各项损失的量级调整初始值
        l2_reg:        对 log_vars 施加的 L2 正则系数，防止某项权重无限趋近于 0
    """

    TASK_NAMES = ['base', 'monotonic', 'boundary', 'smoothness']

    def __init__(
        self,
        physics_loss: PhysicsConstrainedLoss,
        init_log_vars: list = None,
        l2_reg: float = 0.0,
    ):
        super().__init__()
        self.physics_loss = physics_loss
        self.l2_reg = l2_reg

        if init_log_vars is None:
            init_log_vars = [0.0, 0.0, 0.0, 0.0]

        assert len(init_log_vars) == 4, "init_log_vars 长度必须为 4"
        self.log_vars = nn.Parameter(
            torch.tensor(init_log_vars, dtype=torch.float32)
        )
        self.loss_details = {}

    def forward(self, predictions, targets, battery_ids=None, cycle_indices=None,
                supervision_mask=None):
        """
        计算自适应加权总损失。

        L = Σ_i [ exp(-log_σᵢ) · Lᵢ + log_σᵢ ] + l2_reg · Σ log_σᵢ²
        """
        # 1. 获取各损失项（未加权）
        comps = self.physics_loss.forward_components(
            predictions, targets, battery_ids, cycle_indices, supervision_mask
        )

        losses = torch.stack([
            comps['base'],
            comps['monotonic'],
            comps['boundary'],
            comps['smoothness'],
        ])  # shape (4,)

        # 2. 不确定性加权
        #    weight_i = exp(-log_σᵢ)；随 log_σᵢ 增大，权重减小（即对应项贡献降低）
        weights = torch.exp(-self.log_vars)   # (4,)
        total   = (weights * losses + self.log_vars).sum()

        # 3. L2 正则（可选，防止 log_vars 无约束增长而"关闭"某项约束）
        if self.l2_reg > 0.0:
            total = total + self.l2_reg * (self.log_vars ** 2).sum()

        # 4. 记录详情（与 PhysicsConstrainedLoss 接口保持一致）
        self.loss_details = {
            'total':            total.item(),
            'base':             comps['base'].item(),
            'monotonic':        comps['monotonic'].item(),
            'boundary':         comps['boundary'].item(),
            'smoothness':       comps['smoothness'].item(),
            'log_vars':         self.log_vars.detach().cpu().tolist(),
            'eff_weights':      weights.detach().cpu().tolist(),
            'n_labeled_in_batch': comps['n_labeled'],
        }

        return total

    def get_loss_details(self):
        """获取最近一次 forward 的详细损失（接口与 PhysicsConstrainedLoss 一致）"""
        return self.loss_details

    def get_effective_weights(self):
        """返回当前有效权重字典，方便日志打印"""
        weights = torch.exp(-self.log_vars).detach().cpu()
        return {name: float(weights[i]) for i, name in enumerate(self.TASK_NAMES)}

    def extra_repr(self):
        w = self.get_effective_weights()
        return (f"l2_reg={self.l2_reg}, "
                f"eff_weights={{{', '.join(f'{k}:{v:.3f}' for k, v in w.items())}}}")
