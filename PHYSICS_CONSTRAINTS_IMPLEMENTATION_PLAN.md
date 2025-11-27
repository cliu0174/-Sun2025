# 物理约束损失函数实现方案

## 概述

本方案将为 SOH 预测模型添加物理约束损失函数，包括：
1. **软单调性约束**（允许轻微上升）
2. **时间衰减权重**（相邻点权重大，远距离点权重小）
3. **边界约束**（SOH 在 [0, 1] 范围内）
4. **平滑性约束**（可选）

---

## 一、核心改进点

### 1.1 软单调性约束（Soft Monotonic Constraint）

**当前问题**：
- 硬约束（Hard Constraint）：只要 SOH 上升就惩罚
- 过于严格，不允许任何测量误差或小幅波动

**改进方案**：
```python
# 原始硬约束
L_hard = ReLU(SOH[t+1] - SOH[t])

# 软约束（允许 tolerance 范围内的上升）
L_soft = ReLU(SOH[t+1] - SOH[t] - tolerance)
```

**参数说明**：
- `tolerance`（容忍度）：允许的最大上升幅度
  - 例如：tolerance = 0.01 表示允许 SOH 上升不超过 1%
  - 只有超过这个阈值的上升才会被惩罚

**物理意义**：
- 容忍测量噪声和小幅波动
- 仍然惩罚明显违反物理规律的上升趋势
- 更符合实际电池数据的特点

---

### 1.2 时间衰减权重（Temporal Decay Weighting）

**核心思想**：
- 相邻时间步（k=1）：权重最大
- 远距离时间步（k=5）：权重较小
- 强调局部物理规律

**实现方式**：
```python
L_monotonic = Σ w(k) * ReLU(SOH[t+k] - SOH[t] - tolerance)
              k=1 to K

# 权重函数（指数衰减）
w(k) = exp(-alpha * k)
```

**权重示例**（alpha=0.2）：
| 步长 k | 时间间隔 | 权重 w(k) | 物理意义 |
|--------|----------|-----------|----------|
| 1 | 相邻 | 0.82 | 最强约束 |
| 2 | 隔1个 | 0.67 | 较强约束 |
| 3 | 隔2个 | 0.55 | 中等约束 |
| 5 | 隔4个 | 0.37 | 较弱约束 |
| 10 | 隔9个 | 0.14 | 很弱约束 |

---

## 二、文件修改清单

### 2.1 新增文件

#### `models/physics_loss.py`（新建）
物理约束损失函数的实现

**主要类**：
```python
class PhysicsConstrainedLoss(nn.Module):
    """
    物理约束损失函数

    包含：
    1. 软单调性约束 + 时间衰减权重
    2. 边界约束
    3. 平滑性约束
    """
```

---

### 2.2 修改文件

#### `models/__init__.py`
添加 PhysicsConstrainedLoss 的导入

```python
# 修改前
from .model_factory import (
    ConfigLoader,
    ModelFactory,
    UnifiedModelWrapper
)

# 修改后
from .model_factory import (
    ConfigLoader,
    ModelFactory,
    UnifiedModelWrapper
)
from .physics_loss import PhysicsConstrainedLoss

__all__ = [
    # ... 现有内容
    'PhysicsConstrainedLoss',  # 新增
]
```

---

#### `models/model_factory.py`
在 UnifiedModelWrapper 中集成物理损失

**修改点 1**：`__init__` 方法
```python
def __init__(self, model_type, input_size, config=None, device='cuda'):
    # ... 现有初始化代码

    # 创建损失函数（新增物理约束支持）
    self.criterion = self._create_loss_function()

def _create_loss_function(self):
    """根据配置创建损失函数"""
    # 检查是否启用物理约束
    physics_config = self.config.get('physics_constraints', {})

    if physics_config.get('enabled', False):
        # 使用物理约束损失
        return PhysicsConstrainedLoss(
            base_loss_weight=physics_config.get('base_loss_weight', 1.0),
            monotonic_weight=physics_config.get('monotonic_weight', 0.1),
            boundary_weight=physics_config.get('boundary_weight', 0.05),
            smoothness_weight=physics_config.get('smoothness_weight', 0.0),
            # 软约束参数
            monotonic_tolerance=physics_config.get('monotonic_tolerance', 0.01),
            # 时间衰减参数
            temporal_decay_enabled=physics_config.get('temporal_decay', {}).get('enabled', True),
            temporal_max_step=physics_config.get('temporal_decay', {}).get('max_step', 5),
            temporal_decay_type=physics_config.get('temporal_decay', {}).get('decay_type', 'exp'),
            temporal_decay_alpha=physics_config.get('temporal_decay', {}).get('decay_alpha', 0.2)
        ).to(self.device)
    else:
        # 使用标准 MSE 损失
        return nn.MSELoss()
```

**修改点 2**：训练循环中记录物理损失详情（可选）
```python
def train_epoch(self, train_loader):
    # ... 现有代码

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # ... forward pass
        loss = self.criterion(outputs, targets)

        # 如果使用物理约束，记录详细信息
        if isinstance(self.criterion, PhysicsConstrainedLoss):
            loss_dict = self.criterion.get_loss_details()
            # loss_dict 包含：
            # {
            #   'total': total_loss,
            #   'base': base_loss,
            #   'monotonic': monotonic_loss,
            #   'boundary': boundary_loss,
            #   'smoothness': smoothness_loss
            # }
```

---

#### 配置文件（如 `configs/models/lstm_config.json`）
添加物理约束配置项

```json
{
  "model_type": "LSTM",
  "architecture": { ... },
  "training": { ... },
  "data": { ... },

  "physics_constraints": {
    "enabled": false,
    "base_loss_weight": 1.0,
    "monotonic_weight": 0.1,
    "boundary_weight": 0.05,
    "smoothness_weight": 0.0,

    "monotonic_tolerance": 0.01,

    "temporal_decay": {
      "enabled": true,
      "max_step": 5,
      "decay_type": "exp",
      "decay_alpha": 0.2
    },

    "description": "物理约束配置：软单调性 + 时间衰减权重 + 边界约束"
  }
}
```

**配置参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | false | 是否启用物理约束 |
| `base_loss_weight` | float | 1.0 | 基础 MSE 损失权重 |
| `monotonic_weight` | float | 0.1 | 单调性约束权重 |
| `boundary_weight` | float | 0.05 | 边界约束权重 |
| `smoothness_weight` | float | 0.0 | 平滑性约束权重 |
| `monotonic_tolerance` | float | 0.01 | 软约束容忍度（允许的最大上升幅度）|
| `temporal_decay.enabled` | bool | true | 是否启用时间衰减权重 |
| `temporal_decay.max_step` | int | 5 | 最大考虑的时间步长 |
| `temporal_decay.decay_type` | str | "exp" | 衰减类型："exp"、"linear"、"inverse" |
| `temporal_decay.decay_alpha` | float | 0.2 | 衰减系数 |

---

## 三、核心代码实现

### 3.1 `models/physics_loss.py` 完整实现

```python
import torch
import torch.nn as nn


class PhysicsConstrainedLoss(nn.Module):
    """
    物理约束损失函数

    结合：
    1. 基础 MSE 损失（数据拟合）
    2. 软单调性约束 + 时间衰减权重（物理规律）
    3. 边界约束（SOH ∈ [0, 1]）
    4. 平滑性约束（可选）

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
        temporal_max_step=5,
        temporal_decay_type='exp',
        temporal_decay_alpha=0.2
    ):
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

        # 基础损失
        self.mse_loss = nn.MSELoss()

        # 用于记录详细损失（调试用）
        self.loss_details = {}

    def compute_decay_weight(self, k):
        """
        计算步长 k 的时间衰减权重

        Args:
            k: 时间步长（1 表示相邻，2 表示隔一个，...）

        Returns:
            权重值（相邻点权重大，远距离点权重小）
        """
        k = float(k)

        if self.temporal_decay_type == 'exp':
            # 指数衰减：w(k) = exp(-alpha * k)
            return torch.exp(torch.tensor(-self.temporal_decay_alpha * k))

        elif self.temporal_decay_type == 'linear':
            # 线性衰减：w(k) = max(0, 1 - alpha * k)
            return max(0.0, 1.0 - self.temporal_decay_alpha * k)

        elif self.temporal_decay_type == 'inverse':
            # 倒数衰减：w(k) = 1 / k
            return 1.0 / k

        else:
            # 默认：无衰减（所有步长权重相同）
            return 1.0

    def monotonic_loss(self, predictions):
        """
        软单调性约束 + 时间衰减权重

        允许轻微上升（tolerance 范围内），只惩罚明显违反物理规律的上升。
        相邻点约束强，远距离点约束弱。

        Args:
            predictions: 模型预测值 (batch, seq_len, 1) 或 (batch, 1)

        Returns:
            单调性损失
        """
        # 处理 Many-to-One 输出（batch, 1）
        if predictions.dim() == 2 or predictions.shape[1] == 1:
            # 单点输出，无法计算序列单调性
            return torch.tensor(0.0, device=predictions.device)

        # Many-to-Many 输出（batch, seq_len, 1）
        seq_len = predictions.shape[1]

        if not self.temporal_decay_enabled:
            # 不使用时间衰减，简单计算相邻点的软约束
            diff = predictions[:, 1:, :] - predictions[:, :-1, :]
            # 软约束：只惩罚超过 tolerance 的上升
            violation = torch.relu(diff - self.monotonic_tolerance)
            return violation.mean()

        # 使用时间衰减权重
        total_loss = 0.0
        total_weight = 0.0

        # 对不同步长 k 计算约束
        for k in range(1, min(self.temporal_max_step + 1, seq_len)):
            # 计算步长为 k 的差分
            # predictions[:, k:, :] - predictions[:, :-k, :]
            # 例如 k=1: [t=1,2,3,...] - [t=0,1,2,...] = SOH 变化
            diff = predictions[:, k:, :] - predictions[:, :-k, :]

            # 软约束：只惩罚超过 tolerance 的上升
            violation = torch.relu(diff - self.monotonic_tolerance)

            # 计算该步长的衰减权重
            weight = self.compute_decay_weight(k)

            # 加权累加
            total_loss += weight * violation.mean()
            total_weight += weight

        # 归一化（避免权重总和影响损失尺度）
        return total_loss / total_weight if total_weight > 0 else torch.tensor(0.0, device=predictions.device)

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

    def smoothness_loss(self, predictions):
        """
        平滑性约束：SOH 的变化应该平滑（二阶差分小）

        Args:
            predictions: 模型预测值 (batch, seq_len, 1)

        Returns:
            平滑性损失
        """
        # 处理 Many-to-One 输出
        if predictions.dim() == 2 or predictions.shape[1] < 3:
            return torch.tensor(0.0, device=predictions.device)

        # 计算二阶差分
        # first_diff = SOH[t+1] - SOH[t]
        first_diff = predictions[:, 1:, :] - predictions[:, :-1, :]

        # second_diff = first_diff[t+1] - first_diff[t]
        second_diff = first_diff[:, 1:, :] - first_diff[:, :-1, :]

        # 二阶差分的平方（惩罚剧烈变化）
        return (second_diff ** 2).mean()

    def forward(self, predictions, targets):
        """
        计算总损失

        Args:
            predictions: 模型预测值
            targets: 真实目标值

        Returns:
            总损失
        """
        # 1. 基础 MSE 损失（数据拟合）
        base_loss = self.mse_loss(predictions, targets)

        # 2. 软单调性约束 + 时间衰减权重
        mono_loss = self.monotonic_loss(predictions)

        # 3. 边界约束
        bound_loss = self.boundary_loss(predictions)

        # 4. 平滑性约束
        smooth_loss = self.smoothness_loss(predictions)

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
            'base': base_loss.item(),
            'monotonic': mono_loss.item() if isinstance(mono_loss, torch.Tensor) else mono_loss,
            'boundary': bound_loss.item(),
            'smoothness': smooth_loss.item() if isinstance(smooth_loss, torch.Tensor) else smooth_loss
        }

        return total_loss

    def get_loss_details(self):
        """获取最近一次计算的详细损失"""
        return self.loss_details


# 使用示例
if __name__ == "__main__":
    # 创建物理约束损失
    criterion = PhysicsConstrainedLoss(
        base_loss_weight=1.0,
        monotonic_weight=0.1,
        boundary_weight=0.05,
        smoothness_weight=0.0,
        monotonic_tolerance=0.01,  # 允许 1% 的上升
        temporal_decay_enabled=True,
        temporal_max_step=5,
        temporal_decay_type='exp',
        temporal_decay_alpha=0.2
    )

    # 模拟数据
    batch_size = 32
    seq_len = 40

    # 合法预测（单调递减）
    predictions = torch.linspace(1.0, 0.8, seq_len).repeat(batch_size, 1).unsqueeze(-1)
    targets = predictions.clone()

    loss = criterion(predictions, targets)
    print(f"合法预测损失: {loss.item():.6f}")
    print(f"详细损失: {criterion.get_loss_details()}")

    # 违反预测（有上升）
    predictions_bad = predictions.clone()
    predictions_bad[:, 20, 0] = 0.95  # 在第20个位置违反单调性

    loss_bad = criterion(predictions_bad, targets)
    print(f"\n违反预测损失: {loss_bad.item():.6f}")
    print(f"详细损失: {criterion.get_loss_details()}")
```

---

## 四、使用方法

### 4.1 快速启用（推荐）

在配置文件中设置：
```json
{
  "physics_constraints": {
    "enabled": true,
    "monotonic_weight": 0.1,
    "monotonic_tolerance": 0.01
  }
}
```

其他参数使用默认值即可。

---

### 4.2 自定义调整

**调整软约束容忍度**：
```json
"monotonic_tolerance": 0.02  // 允许 2% 的上升（更宽松）
"monotonic_tolerance": 0.005 // 允许 0.5% 的上升（更严格）
```

**调整时间衰减**：
```json
"temporal_decay": {
  "max_step": 10,        // 考虑更远的时间步
  "decay_alpha": 0.3     // 更快的衰减（更强调相邻点）
}
```

**调整权重**：
```json
"base_loss_weight": 1.0,      // 数据拟合
"monotonic_weight": 0.15,     // 单调性约束（增加）
"boundary_weight": 0.05,      // 边界约束
"smoothness_weight": 0.05     // 平滑性约束（启用）
```

---

## 五、预期效果

### 5.1 软约束的优势
- ✅ 允许小幅波动（tolerance 范围内）
- ✅ 只惩罚明显违反的情况
- ✅ 训练更稳定，收敛更快
- ✅ 更符合实际数据特点

### 5.2 时间衰减权重的优势
- ✅ 强调局部物理规律（相邻点）
- ✅ 允许长期小幅波动
- ✅ 避免过度约束
- ✅ 更合理的权重分配

### 5.3 对比实验建议

**实验 1**：无物理约束（baseline）
```json
"physics_constraints": { "enabled": false }
```

**实验 2**：硬约束（原始方法）
```json
"physics_constraints": {
  "enabled": true,
  "monotonic_tolerance": 0.0,
  "temporal_decay": { "enabled": false }
}
```

**实验 3**：软约束（本方案）
```json
"physics_constraints": {
  "enabled": true,
  "monotonic_tolerance": 0.01,
  "temporal_decay": { "enabled": true }
}
```

---

## 六、调试和验证

### 6.1 查看损失详情
```python
if isinstance(criterion, PhysicsConstrainedLoss):
    details = criterion.get_loss_details()
    print(f"总损失: {details['total']:.6f}")
    print(f"  MSE: {details['base']:.6f}")
    print(f"  单调性: {details['monotonic']:.6f}")
    print(f"  边界: {details['boundary']:.6f}")
    print(f"  平滑性: {details['smoothness']:.6f}")
```

### 6.2 可视化损失曲线
建议在训练过程中记录各项损失，训练后绘制：
- 总损失曲线
- MSE 损失曲线
- 单调性损失曲线
- 边界损失曲线

---

## 七、实施步骤

1. ✅ 审阅本方案
2. ⬜ 创建 `models/physics_loss.py`
3. ⬜ 修改 `models/__init__.py`
4. ⬜ 修改 `models/model_factory.py`
5. ⬜ 更新配置文件（如 `lstm_config.json`）
6. ⬜ 测试基本功能
7. ⬜ 训练对比实验
8. ⬜ 调整参数优化

---

## 八、参数推荐值

**保守配置**（轻度约束）：
```json
{
  "enabled": true,
  "base_loss_weight": 1.0,
  "monotonic_weight": 0.05,
  "boundary_weight": 0.02,
  "smoothness_weight": 0.0,
  "monotonic_tolerance": 0.02,
  "temporal_decay": {
    "enabled": true,
    "max_step": 3,
    "decay_alpha": 0.15
  }
}
```

**标准配置**（中等约束）：
```json
{
  "enabled": true,
  "base_loss_weight": 1.0,
  "monotonic_weight": 0.1,
  "boundary_weight": 0.05,
  "smoothness_weight": 0.0,
  "monotonic_tolerance": 0.01,
  "temporal_decay": {
    "enabled": true,
    "max_step": 5,
    "decay_alpha": 0.2
  }
}
```

**激进配置**（强约束）：
```json
{
  "enabled": true,
  "base_loss_weight": 1.0,
  "monotonic_weight": 0.2,
  "boundary_weight": 0.1,
  "smoothness_weight": 0.05,
  "monotonic_tolerance": 0.005,
  "temporal_decay": {
    "enabled": true,
    "max_step": 10,
    "decay_alpha": 0.3
  }
}
```

---

## 九、注意事项

1. **初次使用建议从保守配置开始**，逐步调整
2. **只有 Many-to-Many (Seq2Seq) 模型能充分利用序列约束**
3. **Many-to-One 模型只能使用边界约束**
4. **建议先在小数据集上测试**，确认无误后再大规模训练
5. **如果损失不收敛，降低物理约束权重**

---

**方案准备完毕，请审阅！**
