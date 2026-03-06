"""
学习率调度器工具集
支持多种学习率衰减策略，可在配置文件中灵活选择
"""

import torch
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    LambdaLR
)
import numpy as np


def create_scheduler(optimizer, config, num_epochs):
    """
    根据配置创建学习率调度器

    Args:
        optimizer: PyTorch优化器
        config: 配置字典，包含scheduler配置
        num_epochs: 总训练轮数

    Returns:
        scheduler: 学习率调度器对象
        scheduler_type: 调度器类型（用于判断是否需要验证损失）
    """

    scheduler_config = config['training'].get('scheduler', {})

    # 如果未启用调度器，返回None
    if not scheduler_config.get('enabled', False):
        return None, None

    scheduler_type = scheduler_config.get('type', 'CosineAnnealingLR')
    print(f"\n创建学习率调度器: {scheduler_type}")

    # ===== 1. CosineAnnealingLR - 余弦退火 (推荐) =====
    if scheduler_type == 'CosineAnnealingLR':
        """
        从初始学习率平滑下降到最小学习率
        公式: lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + cos(π * epoch / T_max))

        优点: 平滑下降，后期学习率非常小，适合精细调优
        适用: 大多数训练场景
        """
        T_max = scheduler_config.get('T_max', num_epochs)
        eta_min = scheduler_config.get('eta_min', 1e-6)

        print(f"  T_max (周期): {T_max}")
        print(f"  eta_min (最小学习率): {eta_min:.6f}")

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
        return scheduler, 'step'

    # ===== 2. CosineAnnealingWarmRestarts - 带重启的余弦退火 =====
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        """
        余弦退火 + 周期性重启
        每隔T_0个epoch重启学习率，有助于跳出局部最优

        优点: 周期性重启可以探索更多解空间，避免过早收敛
        适用: 复杂优化问题，容易陷入局部最优的情况
        """
        T_0 = scheduler_config.get('T_0', 50)  # 第一次重启的epoch数
        T_mult = scheduler_config.get('T_mult', 1)  # 每次重启后周期倍增
        eta_min = scheduler_config.get('eta_min', 1e-6)

        print(f"  T_0 (初始周期): {T_0}")
        print(f"  T_mult (周期倍数): {T_mult}")
        print(f"  eta_min (最小学习率): {eta_min:.6f}")

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )
        return scheduler, 'step'

    # ===== 3. StepLR - 阶梯衰减 =====
    elif scheduler_type == 'StepLR':
        """
        每隔step_size个epoch，学习率乘以gamma

        优点: 简单稳定，容易控制
        适用: 需要阶段性调整学习率的场景
        """
        step_size = scheduler_config.get('step_size', 30)
        gamma = scheduler_config.get('gamma', 0.5)

        print(f"  step_size (衰减间隔): {step_size}")
        print(f"  gamma (衰减因子): {gamma}")

        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
        return scheduler, 'step'

    # ===== 4. MultiStepLR - 多阶梯衰减 =====
    elif scheduler_type == 'MultiStepLR':
        """
        在指定的epoch处降低学习率

        优点: 可以精确控制每个衰减点
        适用: 已知训练过程中的关键节点
        """
        milestones = scheduler_config.get('milestones', [60, 120, 160])
        gamma = scheduler_config.get('gamma', 0.2)

        print(f"  milestones (衰减节点): {milestones}")
        print(f"  gamma (衰减因子): {gamma}")

        scheduler = MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma
        )
        return scheduler, 'step'

    # ===== 5. ExponentialLR - 指数衰减 =====
    elif scheduler_type == 'ExponentialLR':
        """
        每个epoch学习率乘以gamma

        优点: 连续平滑衰减
        适用: 需要快速降低学习率的场景
        """
        gamma = scheduler_config.get('gamma', 0.95)

        print(f"  gamma (衰减因子): {gamma}")

        scheduler = ExponentialLR(
            optimizer,
            gamma=gamma
        )
        return scheduler, 'step'

    # ===== 6. ReduceLROnPlateau - 自适应衰减 =====
    elif scheduler_type == 'ReduceLROnPlateau':
        """
        当验证指标停止改善时降低学习率

        优点: 自适应调整，不需要手动设置epoch
        适用: 验证集可用，希望根据性能自动调整
        注意: 需要在训练循环中传入验证损失
        """
        mode = scheduler_config.get('mode', 'min')  # 'min' or 'max'
        factor = scheduler_config.get('factor', 0.5)
        patience = scheduler_config.get('patience', 10)
        min_lr = scheduler_config.get('min_lr', 1e-6)

        print(f"  mode: {mode}")
        print(f"  factor (衰减因子): {factor}")
        print(f"  patience (容忍轮数): {patience}")
        print(f"  min_lr (最小学习率): {min_lr:.6f}")

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True
        )
        return scheduler, 'plateau'

    # ===== 7. WarmupCosineDecay - Warmup + 余弦衰减 (原有实现) =====
    elif scheduler_type == 'WarmupCosineDecay':
        """
        Warmup阶段 + 余弦衰减

        优点: Warmup可以稳定训练初期，余弦衰减平滑
        适用: 大模型训练，需要warmup阶段
        """
        warmup_epochs = scheduler_config.get('warmup_epochs', 30)
        warmup_lr = scheduler_config.get('warmup_lr', 2e-3)
        base_lr = scheduler_config.get('base_lr', 1e-2)
        final_lr = scheduler_config.get('final_lr', 2e-4)

        print(f"  warmup_epochs: {warmup_epochs}")
        print(f"  warmup_lr: {warmup_lr:.6f}")
        print(f"  base_lr: {base_lr:.6f}")
        print(f"  final_lr: {final_lr:.6f}")

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Warmup阶段: 线性增长
                return warmup_lr / config['training']['learning_rate'] + \
                       (base_lr - warmup_lr) / warmup_epochs * epoch / config['training']['learning_rate']
            else:
                # Cosine衰减阶段
                progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                current_lr = final_lr + (base_lr - final_lr) * 0.5 * (1 + np.cos(np.pi * progress))
                return current_lr / config['training']['learning_rate']

        scheduler = LambdaLR(optimizer, lr_lambda)
        return scheduler, 'step'

    # ===== 8. LinearDecay - 线性衰减 =====
    elif scheduler_type == 'LinearDecay':
        """
        从初始学习率线性衰减到最终学习率

        优点: 最简单的衰减策略
        适用: 快速实验
        """
        start_lr = config['training']['learning_rate']
        end_lr = scheduler_config.get('end_lr', 1e-5)

        print(f"  start_lr: {start_lr:.6f}")
        print(f"  end_lr: {end_lr:.6f}")

        def lr_lambda(epoch):
            return end_lr / start_lr + (1 - end_lr / start_lr) * (1 - epoch / num_epochs)

        scheduler = LambdaLR(optimizer, lr_lambda)
        return scheduler, 'step'

    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")


def get_current_lr(optimizer):
    """获取当前学习率"""
    return optimizer.param_groups[0]['lr']


def print_scheduler_info(epoch, optimizer, val_loss=None):
    """打印学习率信息"""
    current_lr = get_current_lr(optimizer)
    info = f"  LR: {current_lr:.6f}"
    if val_loss is not None:
        info += f" | Val Loss: {val_loss:.6f}"
    print(info)
