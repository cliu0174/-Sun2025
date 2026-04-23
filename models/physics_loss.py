"""
物理约束损失函数

针对 Many-to-One 模型的物理约束实现
支持 shuffle=True 的训练方式，通过 battery_id 和 cycle_idx 重建序列关系

核心特性：
1. 软单调性约束（允许 tolerance 范围内的轻微上升）
2. 时间衰减权重（所有配对，距离越远权重越小）
3. 边界约束（SOH ∈ [0, 1]）
4. 平滑性约束（可选）
"""

import torch
import torch.nn as nn
import numpy as np


class PhysicsConstrainedLoss(nn.Module):
    """
    物理约束损失函数

    适用于 Many-to-One 模型（如 LSTM, GRU, CNN 等）
    在 batch 内动态找出同一电池的样本，按 cycle_idx 排序后应用约束

    总损失：
    L = w_base * L_MSE + w_mono * L_monotonic + w_bound * L_boundary + w_smooth * L_smoothness
    """

    def __init__(
        self,
        base_loss_weight=1.0,
        monotonic_weight=0.1,
        boundary_weight=0.05,
        smoothness_weight=0.0,
        # 软约束参数
        monotonic_tolerance=0.01,
        # 时间衰减参数
        temporal_decay_enabled=True,
        temporal_max_step=20,
        temporal_decay_type='exp',
        temporal_decay_alpha=0.2,
        # 调试参数
        verbose=False
    ):
        """
        初始化物理约束损失函数

        Args:
            base_loss_weight: 基础 MSE 损失权重
            monotonic_weight: 单调性约束权重
            boundary_weight: 边界约束权重
            smoothness_weight: 平滑性约束权重
            monotonic_tolerance: 软约束容忍度（允许的最大上升幅度）
            temporal_decay_enabled: 是否启用时间衰减权重
            temporal_max_step: 最大考虑的时间步长
            temporal_decay_type: 衰减类型 ('exp', 'linear', 'inverse')
            temporal_decay_alpha: 衰减系数
            verbose: 是否输出详细调试信息
        """
        super().__init__()

        # 损失权重
        self.base_loss_weight = base_loss_weight
        self.monotonic_weight = monotonic_weight
        self.boundary_weight = boundary_weight
        self.smoothness_weight = smoothness_weight

        # 软约束参数
        self.monotonic_tolerance = monotonic_tolerance

        # 时间衰减参数
        self.temporal_decay_enabled = temporal_decay_enabled
        self.temporal_max_step = temporal_max_step
        self.temporal_decay_type = temporal_decay_type
        self.temporal_decay_alpha = temporal_decay_alpha

        # 调试参数
        self.verbose = verbose

        # 基础损失
        self.mse_loss = nn.MSELoss()

        # 用于记录详细损失（调试和可视化）
        self.loss_details = {}

    def compute_decay_weight(self, k):
        """
        计算步长 k 的时间衰减权重

        Args:
            k: 时间步长（cycle 间隔）

        Returns:
            权重值（相邻点权重大，远距离点权重小）
        """
        if isinstance(k, torch.Tensor):
            k = k.float()
        else:
            k = float(k)

        if self.temporal_decay_type == 'exp':
            # 指数衰减：w(k) = exp(-alpha * k)
            if isinstance(k, torch.Tensor):
                return torch.exp(-self.temporal_decay_alpha * k)
            else:
                return np.exp(-self.temporal_decay_alpha * k)

        elif self.temporal_decay_type == 'linear':
            # 线性衰减：w(k) = max(0, 1 - alpha * k)
            return max(0.0, 1.0 - self.temporal_decay_alpha * k)

        elif self.temporal_decay_type == 'inverse':
            # 倒数衰减：w(k) = 1 / k
            if isinstance(k, torch.Tensor):
                return 1.0 / (k + 1e-8)  # 避免除以0
            else:
                return 1.0 / k

        else:
            # 默认：无衰减
            return 1.0

    def _encode_battery_ids(self, battery_ids, device):
        """
        将字符串 battery_ids 一次性编码为整数 tensor，避免在 GPU 上反复做
        CPU→GPU 传输。返回 (battery_id_tensor, n_unique)。
        """
        unique_list = list(dict.fromkeys(battery_ids))   # 保序去重，O(N) Python
        id_map = {bid: i for i, bid in enumerate(unique_list)}
        int_ids = [id_map[bid] for bid in battery_ids]   # O(N) Python
        return torch.tensor(int_ids, device=device, dtype=torch.long), len(unique_list)

    def monotonic_loss(self, predictions, battery_ids, cycle_indices):
        """
        软单调性约束 + 时间衰减权重（所有配对）

        在 batch 内找出同一电池的样本，按 cycle_idx 排序后，
        对所有可能的配对应用约束，权重随距离衰减。

        Args:
            predictions: (batch_size, 1) 预测的 SOH
            battery_ids: list/tuple of str, 长度 batch_size
            cycle_indices: (batch_size,) cycle 索引

        Returns:
            单调性损失（保持梯度）
        """
        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)

        if not isinstance(cycle_indices, torch.Tensor):
            cycle_indices = torch.tensor(cycle_indices, device=predictions.device)
        elif cycle_indices.device != predictions.device:
            cycle_indices = cycle_indices.to(predictions.device)

        # 一次性编码：1 次 CPU→GPU 传输，替代原来每个电池一次传输
        bid_tensor, n_unique = self._encode_battery_ids(battery_ids, predictions.device)

        total_loss = torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)
        num_pairs = 0

        if self.verbose:
            print(f"\n[MonotonicLoss] Batch 内有 {n_unique} 个不同电池")

        # 用 GPU tensor 做分组，不再做 Python 字符串比较
        for uid in range(n_unique):
            indices = (bid_tensor == uid).nonzero(as_tuple=True)[0]

            if len(indices) < 2:
                continue

            battery_preds  = predictions[indices]
            battery_cycles = cycle_indices[indices]

            sorted_order   = torch.argsort(battery_cycles)
            sorted_preds   = battery_preds[sorted_order]
            sorted_cycles  = battery_cycles[sorted_order]

            if self.verbose:
                print(f"  电池 {uid}: {len(indices)} 个样本")

            n = len(sorted_preds)
            if n < 2:
                continue

            pred_diff_matrix  = sorted_preds.unsqueeze(1) - sorted_preds.unsqueeze(0)   # (n,n)
            cycle_diff_matrix = sorted_cycles.unsqueeze(1) - sorted_cycles.unsqueeze(0) # (n,n)

            pair_mask = (cycle_diff_matrix > 0) & (cycle_diff_matrix <= self.temporal_max_step)
            valid_pred_diffs  = pred_diff_matrix[pair_mask]
            valid_cycle_diffs = cycle_diff_matrix[pair_mask]

            if len(valid_pred_diffs) == 0:
                continue

            if self.temporal_decay_enabled:
                if self.temporal_decay_type == 'exp':
                    weights = torch.exp(-self.temporal_decay_alpha * valid_cycle_diffs.float())
                elif self.temporal_decay_type == 'linear':
                    weights = 1.0 - (valid_cycle_diffs.float() / self.temporal_max_step)
                elif self.temporal_decay_type == 'inverse':
                    weights = 1.0 / (valid_cycle_diffs.float() + 1e-8)
                else:
                    weights = torch.ones_like(valid_cycle_diffs, dtype=torch.float32)
            else:
                weights = torch.ones_like(valid_cycle_diffs, dtype=torch.float32)

            violations  = torch.nn.functional.relu(-valid_pred_diffs - self.monotonic_tolerance)
            battery_loss = torch.sum(weights * violations)
            total_loss  = total_loss + battery_loss
            num_pairs   += len(valid_pred_diffs)

            if self.verbose:
                print(f"    该电池 {len(valid_pred_diffs)} 个配对，损失={battery_loss.item():.6f}")

        if num_pairs > 0:
            normalized_loss = total_loss / num_pairs
            if self.verbose:
                print(f"  总配对数: {num_pairs}, 归一化损失: {normalized_loss.item():.6f}")
            return normalized_loss
        else:
            if self.verbose:
                print(f"  无有效配对")
            return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)

    def boundary_loss(self, predictions):
        """
        边界约束：SOH 应该在 [0, 1] 范围内

        Args:
            predictions: 模型预测值

        Returns:
            边界损失
        """
        # 惩罚 SOH < 0
        lower_violation = torch.relu(-predictions)

        # 惩罚 SOH > 1
        upper_violation = torch.relu(predictions - 1.0)

        return (lower_violation + upper_violation).mean()

    def smoothness_loss(self, predictions, battery_ids, cycle_indices):
        """
        平滑性约束：SOH 的变化应该平滑（二阶差分小）- 优化版

        仅在同一电池内部计算

        Args:
            predictions: (batch_size, 1) 预测值
            battery_ids: list/tuple of str
            cycle_indices: (batch_size,) cycle 索引

        Returns:
            平滑性损失（保持梯度）
        """
        # 确保 predictions 是一维的
        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)

        # 将 cycle_indices 转换为 tensor
        if not isinstance(cycle_indices, torch.Tensor):
            cycle_indices = torch.tensor(cycle_indices, device=predictions.device)
        elif cycle_indices.device != predictions.device:
            cycle_indices = cycle_indices.to(predictions.device)

        total_loss = torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)
        num_valid = 0

        # 一次性编码，复用 _encode_battery_ids 避免重复 CPU→GPU 传输
        bid_tensor, n_unique = self._encode_battery_ids(battery_ids, predictions.device)

        for uid in range(n_unique):
            idx_tensor = (bid_tensor == uid).nonzero(as_tuple=True)[0]
            if len(idx_tensor) < 3:
                continue
            indices = idx_tensor

            # 提取该电池的预测值和 cycle 索引（保持梯度）
            battery_preds = predictions[indices]
            battery_cycles = cycle_indices[indices]

            # 排序
            sorted_indices = torch.argsort(battery_cycles)
            sorted_preds = battery_preds[sorted_indices]

            # 计算一阶差分（保持梯度）
            first_diff = sorted_preds[1:] - sorted_preds[:-1]

            # 计算二阶差分（保持梯度）
            second_diff = first_diff[1:] - first_diff[:-1]

            # 二阶差分的平方和（保持梯度）
            total_loss = total_loss + torch.sum(second_diff ** 2)
            num_valid += len(second_diff)

        if num_valid > 0:
            return total_loss / num_valid
        else:
            return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)

    def forward(self, predictions, targets, battery_ids=None, cycle_indices=None,
                supervision_mask=None):
        """
        计算总损失

        Args:
            predictions: (batch_size, 1) 模型预测值
            targets: (batch_size, 1) 真实目标值
            battery_ids: list/tuple of str, 电池名称
            cycle_indices: (batch_size,) 或 list, cycle 索引
            supervision_mask: (batch_size,) 或 (batch_size, 1) bool tensor, optional
                True 表示该样本有监督标签（参与 MSE），False 表示无标签（跳过 MSE）
                None 表示所有样本都有标签（完全监督，行为与原来一致）
                物理约束（单调/边界/平滑）对所有样本都计算，不受此 mask 影响

        Returns:
            总损失
        """
        # 1. 基础 MSE 损失（数据拟合）—— 支持部分监督
        if supervision_mask is None:
            # 完全监督：原始行为
            base_loss = self.mse_loss(predictions, targets)
            n_labeled = predictions.shape[0]
        else:
            # 部分监督：只对有标签样本计算 MSE
            # 统一 mask 形状：(batch,) 或 (batch, 1) -> (batch, 1)
            mask = supervision_mask.to(predictions.device)
            if mask.dim() == 1:
                mask = mask.unsqueeze(1)
            mask = mask.to(predictions.dtype)  # float for multiplication

            n_labeled = mask.sum()
            if n_labeled > 0:
                # masked MSE: 只对有标签样本求均方误差
                sq_err = (predictions - targets) ** 2
                base_loss = (sq_err * mask).sum() / n_labeled.clamp(min=1.0)
            else:
                # 极端情况：整个 batch 都没标签
                base_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # 2. 物理约束（需要 battery_ids 和 cycle_indices）
        # 注意：物理约束对所有样本都计算，不受 supervision_mask 影响
        if battery_ids is not None and cycle_indices is not None:
            # 软单调性约束 + 时间衰减权重
            mono_loss = self.monotonic_loss(predictions, battery_ids, cycle_indices)

            # 边界约束
            bound_loss = self.boundary_loss(predictions)

            # 平滑性约束
            smooth_loss = self.smoothness_loss(predictions, battery_ids, cycle_indices)
        else:
            # 如果没有提供 battery_ids 和 cycle_indices，只能计算边界约束
            mono_loss = torch.tensor(0.0, device=predictions.device)
            bound_loss = self.boundary_loss(predictions)
            smooth_loss = torch.tensor(0.0, device=predictions.device)

        # 总损失
        total_loss = (
            self.base_loss_weight * base_loss +
            self.monotonic_weight * mono_loss +
            self.boundary_weight * bound_loss +
            self.smoothness_weight * smooth_loss
        )

        # 记录详细损失（用于调试和可视化）
        self.loss_details = {
            'total': total_loss.item(),
            'base': base_loss.item() if isinstance(base_loss, torch.Tensor) else base_loss,
            'monotonic': mono_loss.item() if isinstance(mono_loss, torch.Tensor) else mono_loss,
            'boundary': bound_loss.item(),
            'smoothness': smooth_loss.item() if isinstance(smooth_loss, torch.Tensor) else smooth_loss,
            'n_labeled_in_batch': int(n_labeled) if isinstance(n_labeled, (int, float)) else int(n_labeled.item())
        }

        return total_loss

    def forward_components(self, predictions, targets, battery_ids=None,
                           cycle_indices=None, supervision_mask=None):
        """
        返回各损失项的未加权张量，供 AdaptivePhysicsLoss (M5) 使用。

        Returns:
            dict with keys: 'base', 'monotonic', 'boundary', 'smoothness', 'n_labeled'
        """
        # --- base MSE（支持部分监督）---
        if supervision_mask is None:
            base_loss = self.mse_loss(predictions, targets)
            n_labeled = predictions.shape[0]
        else:
            mask = supervision_mask.to(predictions.device)
            if mask.dim() == 1:
                mask = mask.unsqueeze(1)
            mask = mask.to(predictions.dtype)
            n_labeled = mask.sum()
            if n_labeled > 0:
                sq_err = (predictions - targets) ** 2
                base_loss = (sq_err * mask).sum() / n_labeled.clamp(min=1.0)
            else:
                base_loss = torch.tensor(0.0, device=predictions.device,
                                         requires_grad=True)
            n_labeled = int(n_labeled.item())

        # --- 物理项 ---
        if battery_ids is not None and cycle_indices is not None:
            mono_loss   = self.monotonic_loss(predictions, battery_ids, cycle_indices)
            smooth_loss = self.smoothness_loss(predictions, battery_ids, cycle_indices)
        else:
            mono_loss   = torch.tensor(0.0, device=predictions.device)
            smooth_loss = torch.tensor(0.0, device=predictions.device)
        bound_loss = self.boundary_loss(predictions)

        return {
            'base':       base_loss,
            'monotonic':  mono_loss,
            'boundary':   bound_loss,
            'smoothness': smooth_loss,
            'n_labeled':  n_labeled,
        }

    def get_loss_details(self):
        """获取最近一次计算的详细损失"""
        return self.loss_details


class SiamesePhysicsLoss(nn.Module):
    """
    Siamese Physics-Informed Loss with Split Constraint

    Key Features:
    1. Works with paired samples (x_t, x_next) from Dataset
    2. MSE component applies to ALL samples
    3. Physics component (monotonicity + smoothness) only applies to samples
       where cycle_index >= split_threshold (using a mask)

    This allows early cycles (with capacity rise) to learn freely from MSE,
    while late cycles are constrained by physics.
    """

    def __init__(
        self,
        base_loss_weight=1.0,
        monotonic_weight=0.1,
        smoothness_weight=0.01,
        boundary_weight=0.05,
        split_threshold=300,
        monotonic_tolerance=0.0,
        verbose=False
    ):
        """
        Args:
            base_loss_weight: Weight for MSE loss
            monotonic_weight: Weight for monotonicity constraint
            smoothness_weight: Weight for smoothness constraint
            boundary_weight: Weight for boundary constraint (SOH in [0,1])
            split_threshold: Cycle threshold for masking physics loss
                            (e.g., 300 means physics only applies to cycles >= 300)
            monotonic_tolerance: Tolerance for monotonicity (0 = strict decrease)
            verbose: Whether to print debug info
        """
        super().__init__()

        self.base_loss_weight = base_loss_weight
        self.monotonic_weight = monotonic_weight
        self.smoothness_weight = smoothness_weight
        self.boundary_weight = boundary_weight
        self.split_threshold = split_threshold
        self.monotonic_tolerance = monotonic_tolerance
        self.verbose = verbose

        self.mse_loss = nn.MSELoss()
        self.loss_details = {}

    def forward(self, pred_t, pred_next, true_t, true_next, cycle_indices):
        """
        Compute total loss for Siamese samples.

        Args:
            pred_t: (batch, 1) predictions at time t
            pred_next: (batch, 1) predictions at time t+k
            true_t: (batch, 1) ground truth at time t
            true_next: (batch, 1) ground truth at time t+k
            cycle_indices: (batch,) cycle indices at time t

        Returns:
            total_loss: scalar tensor
        """
        # Ensure all tensors are on the same device
        device = pred_t.device

        # 1. MSE Component (applies to ALL samples)
        mse_t = self.mse_loss(pred_t, true_t)
        mse_next = self.mse_loss(pred_next, true_next)
        base_loss = (mse_t + mse_next) / 2

        # 2. Create mask based on cycle_indices
        # mask = 1 if cycle_index >= split_threshold, else 0
        if not isinstance(cycle_indices, torch.Tensor):
            cycle_indices = torch.tensor(cycle_indices, device=device)
        elif cycle_indices.device != device:
            cycle_indices = cycle_indices.to(device)

        mask = (cycle_indices >= self.split_threshold).float()  # (batch,)

        # 3. Monotonicity Constraint (masked)
        # pred_next should be <= pred_t (capacity decreases)
        # violation = max(0, pred_next - pred_t - tolerance)

        # Squeeze to ensure 1D
        pred_t_flat = pred_t.squeeze() if pred_t.dim() > 1 else pred_t
        pred_next_flat = pred_next.squeeze() if pred_next.dim() > 1 else pred_next

        # Calculate difference: diff = pred_next - pred_t
        diff = pred_next_flat - pred_t_flat

        # Violation: if diff > tolerance, penalize
        monotonic_violation = torch.relu(diff - self.monotonic_tolerance)

        # Apply mask: only penalize violations for cycles >= split_threshold
        masked_monotonic_loss = (monotonic_violation ** 2) * mask

        # Average over batch
        monotonic_loss = masked_monotonic_loss.mean()

        # 4. Smoothness Constraint (masked)
        # Penalize if diff is too large (in absolute value)
        # This prevents sudden jumps even if direction is correct
        smoothness_violation = diff ** 2  # Squared difference

        # Apply mask
        masked_smoothness_loss = smoothness_violation * mask

        # Average over batch
        smoothness_loss = masked_smoothness_loss.mean()

        # 5. Boundary Constraint (applies to ALL samples)
        # Penalize if SOH goes outside [0, 1]
        boundary_t = torch.relu(-pred_t) + torch.relu(pred_t - 1.0)
        boundary_next = torch.relu(-pred_next) + torch.relu(pred_next - 1.0)
        boundary_loss = (boundary_t.mean() + boundary_next.mean()) / 2

        # 6. Total Loss
        total_loss = (
            self.base_loss_weight * base_loss +
            self.monotonic_weight * monotonic_loss +
            self.smoothness_weight * smoothness_loss +
            self.boundary_weight * boundary_loss
        )

        # Record details
        self.loss_details = {
            'total': total_loss.item(),
            'base': base_loss.item(),
            'mse_t': mse_t.item(),
            'mse_next': mse_next.item(),
            'monotonic': monotonic_loss.item(),
            'smoothness': smoothness_loss.item(),
            'boundary': boundary_loss.item(),
            'mask_active_ratio': mask.mean().item()  # % of samples where physics applies
        }

        if self.verbose:
            print(f"[SiamesePhysicsLoss] Total: {total_loss.item():.6f}, "
                  f"Base: {base_loss.item():.6f}, "
                  f"Mono: {monotonic_loss.item():.6f}, "
                  f"Mask active: {mask.mean().item()*100:.1f}%")

        return total_loss

    def get_loss_details(self):
        """Get detailed loss breakdown from last forward pass."""
        return self.loss_details


class TripletPhysicsLoss(nn.Module):
    """
    Triplet Physics-Informed Loss with Second-Order Smoothness Constraint

    Key Features:
    1. Works with triplet samples (x_1, x_2, x_3) at times (t, t+k, t+2k)
    2. MSE component applies to ALL three samples
    3. First-order constraint: Monotonicity (pred_2 <= pred_1, pred_3 <= pred_2)
    4. Second-order constraint: Curvature = pred_3 - 2*pred_2 + pred_1
       - This is the discrete second derivative (Laplacian)
       - Minimizing curvature² eliminates high-frequency jitter (sawtooth patterns)
    5. Split constraint: Physics only applies to cycles >= split_threshold

    Advantages over Pairwise:
    - Pairwise only constrains "velocity" (first derivative)
    - Triplet also constrains "acceleration" (second derivative)
    - Can suppress oscillations that pairwise cannot catch
    """

    def __init__(
        self,
        base_loss_weight=1.0,
        monotonic_weight=0.1,
        curvature_weight=0.1,
        boundary_weight=0.05,
        split_threshold=300,
        monotonic_tolerance=0.0,
        verbose=False
    ):
        """
        Args:
            base_loss_weight: Weight for MSE loss
            monotonic_weight: Weight for monotonicity constraint (1st order)
            curvature_weight: Weight for curvature constraint (2nd order)
            boundary_weight: Weight for boundary constraint (SOH in [0,1])
            split_threshold: Cycle threshold for masking physics loss
            monotonic_tolerance: Tolerance for monotonicity (0 = strict decrease)
            verbose: Whether to print debug info
        """
        super().__init__()

        self.base_loss_weight = base_loss_weight
        self.monotonic_weight = monotonic_weight
        self.curvature_weight = curvature_weight
        self.boundary_weight = boundary_weight
        self.split_threshold = split_threshold
        self.monotonic_tolerance = monotonic_tolerance
        self.verbose = verbose

        self.mse_loss = nn.MSELoss()
        self.loss_details = {}

    def forward(self, pred_1, pred_2, pred_3, true_1, true_2, true_3, cycle_indices):
        """
        Compute total loss for Triplet samples.

        Args:
            pred_1: (batch, 1) predictions at time t
            pred_2: (batch, 1) predictions at time t+k
            pred_3: (batch, 1) predictions at time t+2k
            true_1: (batch, 1) ground truth at time t
            true_2: (batch, 1) ground truth at time t+k
            true_3: (batch, 1) ground truth at time t+2k
            cycle_indices: (batch,) cycle indices at time t

        Returns:
            total_loss: scalar tensor
        """
        # Ensure all tensors are on the same device
        device = pred_1.device

        # 1. MSE Component (applies to ALL three samples)
        mse_1 = self.mse_loss(pred_1, true_1)
        mse_2 = self.mse_loss(pred_2, true_2)
        mse_3 = self.mse_loss(pred_3, true_3)
        base_loss = (mse_1 + mse_2 + mse_3) / 3

        # 2. Create mask based on cycle_indices
        # mask = 1 if cycle_index >= split_threshold, else 0
        if not isinstance(cycle_indices, torch.Tensor):
            cycle_indices = torch.tensor(cycle_indices, device=device)
        elif cycle_indices.device != device:
            cycle_indices = cycle_indices.to(device)

        mask = (cycle_indices >= self.split_threshold).float()  # (batch,)

        # 3. Monotonicity Constraint (1st order, masked)
        # Both transitions should be non-increasing:
        # - pred_2 <= pred_1
        # - pred_3 <= pred_2

        # Squeeze to ensure 1D
        pred_1_flat = pred_1.squeeze() if pred_1.dim() > 1 else pred_1
        pred_2_flat = pred_2.squeeze() if pred_2.dim() > 1 else pred_2
        pred_3_flat = pred_3.squeeze() if pred_3.dim() > 1 else pred_3

        # Calculate differences
        diff_1_2 = pred_2_flat - pred_1_flat  # Should be <= 0
        diff_2_3 = pred_3_flat - pred_2_flat  # Should be <= 0

        # Violations: if diff > tolerance, penalize
        violation_1_2 = torch.relu(diff_1_2 - self.monotonic_tolerance)
        violation_2_3 = torch.relu(diff_2_3 - self.monotonic_tolerance)

        # Apply mask and average
        masked_monotonic_loss = ((violation_1_2 ** 2) + (violation_2_3 ** 2)) * mask
        monotonic_loss = masked_monotonic_loss.mean()

        # 4. Curvature Constraint (2nd order, masked) - KEY INNOVATION!
        # Discrete second derivative (Laplacian):
        # curvature = pred_3 - 2*pred_2 + pred_1
        #
        # Physical interpretation:
        # - If points are on a straight line: curvature = 0
        # - If there's a "bend" (change in direction): curvature != 0
        # - Minimizing curvature² forces smooth, consistent decline

        curvature = pred_3_flat - 2.0 * pred_2_flat + pred_1_flat

        # Penalize curvature (any deviation from straight line)
        curvature_violation = curvature ** 2

        # Apply mask
        masked_curvature_loss = curvature_violation * mask

        # Average over batch
        curvature_loss = masked_curvature_loss.mean()

        # 5. Boundary Constraint (applies to ALL samples)
        # Penalize if SOH goes outside [0, 1]
        boundary_1 = torch.relu(-pred_1) + torch.relu(pred_1 - 1.0)
        boundary_2 = torch.relu(-pred_2) + torch.relu(pred_2 - 1.0)
        boundary_3 = torch.relu(-pred_3) + torch.relu(pred_3 - 1.0)
        boundary_loss = (boundary_1.mean() + boundary_2.mean() + boundary_3.mean()) / 3

        # 6. Total Loss
        total_loss = (
            self.base_loss_weight * base_loss +
            self.monotonic_weight * monotonic_loss +
            self.curvature_weight * curvature_loss +
            self.boundary_weight * boundary_loss
        )

        # Record details
        self.loss_details = {
            'total': total_loss.item(),
            'base': base_loss.item(),
            'mse_1': mse_1.item(),
            'mse_2': mse_2.item(),
            'mse_3': mse_3.item(),
            'monotonic': monotonic_loss.item(),
            'curvature': curvature_loss.item(),
            'boundary': boundary_loss.item(),
            'mask_active_ratio': mask.mean().item()
        }

        if self.verbose:
            print(f"[TripletPhysicsLoss] Total: {total_loss.item():.6f}, "
                  f"Base: {base_loss.item():.6f}, "
                  f"Mono: {monotonic_loss.item():.6f}, "
                  f"Curv: {curvature_loss.item():.6f}, "
                  f"Mask: {mask.mean().item()*100:.1f}%")

        return total_loss

    def get_loss_details(self):
        """Get detailed loss breakdown from last forward pass."""
        return self.loss_details


if __name__ == "__main__":
    """测试物理约束损失函数"""

    print("="*70)
    print("测试物理约束损失函数")
    print("="*70)

    # 创建物理约束损失
    criterion = PhysicsConstrainedLoss(
        base_loss_weight=1.0,
        monotonic_weight=0.1,
        boundary_weight=0.05,
        smoothness_weight=0.0,
        monotonic_tolerance=0.01,
        temporal_decay_enabled=True,
        temporal_max_step=20,
        temporal_decay_type='exp',
        temporal_decay_alpha=0.2,
        verbose=True
    )

    # 模拟一个 batch 的数据（打乱后）
    print("\n[测试1] 合法预测（符合物理规律）")
    print("-"*70)

    predictions = torch.tensor([
        [0.920],  # 电池 1-1, cycle 50
        [0.880],  # 电池 6-2, cycle 40
        [0.915],  # 电池 1-1, cycle 55
        [0.905],  # 电池 1-1, cycle 70
        [0.875],  # 电池 6-2, cycle 45
        [0.950],  # 电池 9-8, cycle 30 (单个样本)
    ])

    targets = predictions.clone()

    battery_ids = ["1-1", "6-2", "1-1", "1-1", "6-2", "9-8"]
    cycle_indices = torch.tensor([50, 40, 55, 70, 45, 30])

    loss = criterion(predictions, targets, battery_ids, cycle_indices)

    print(f"\n总损失: {loss.item():.6f}")
    print(f"详细损失: {criterion.get_loss_details()}")

    # 测试违反物理规律的情况
    print("\n"+"="*70)
    print("[测试2] 违反预测（有上升）")
    print("-"*70)

    predictions_bad = torch.tensor([
        [0.920],  # 电池 1-1, cycle 50
        [0.880],  # 电池 6-2, cycle 40
        [0.935],  # 电池 1-1, cycle 55 - 违反！上升了 0.015
        [0.905],  # 电池 1-1, cycle 70
        [0.875],  # 电池 6-2, cycle 45
        [0.950],  # 电池 9-8, cycle 30
    ])

    loss_bad = criterion(predictions_bad, targets, battery_ids, cycle_indices)

    print(f"\n总损失: {loss_bad.item():.6f}")
    print(f"详细损失: {criterion.get_loss_details()}")

    print("\n"+"="*70)
    print("测试完成！")
    print("="*70)
