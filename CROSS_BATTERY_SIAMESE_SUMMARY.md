# train_cross_battery.py 孪生采样支持 - 修改总结

## 修改日期
2025-12-08

## 修改目标
为 `train_cross_battery.py` 添加孪生采样（Siamese Sampling）支持，同时确保完全向后兼容（默认行为不变）。

---

## 关键原则
**向后兼容性**: 所有修改必须保证默认行为不变，现有训练流程不受影响。

---

## 修改清单

### 1. 导入 SiamesePhysicsLoss (Line 21)

**修改前:**
```python
from models import ModelFactory, ConfigLoader, UnifiedModelWrapper, PhysicsConstrainedLoss
```

**修改后:**
```python
from models import ModelFactory, ConfigLoader, UnifiedModelWrapper, PhysicsConstrainedLoss, SiamesePhysicsLoss
```

**用途**: 导入孪生模式损失函数

---

### 2. 读取孪生采样配置 (Lines 484-498)

**新增代码:**
```python
# 读取孪生采样配置（向后兼容：默认关闭）
siamese_config = physics_config.get('siamese_sampling', {})
siamese_mode = siamese_config.get('enabled', False)
split_threshold = siamese_config.get('split_threshold', 300)
step_k = siamese_config.get('step_k', 1)

if siamese_mode:
    print(f"\n  孪生采样: 启用")
    print(f"    分段阈值: {split_threshold} cycles")
    print(f"    配对步长: {step_k}")
else:
    print("\n物理约束: 未启用")
    siamese_mode = False
    split_threshold = 300
    step_k = 1
```

**位置**: 在读取 `physics_config` 之后

**向后兼容性**:
- 使用 `.get()` 方法，未配置时默认 `False`
- 所有参数都有默认值

---

### 3. 修改 create_dataloaders 函数签名 (Line 211)

**修改前:**
```python
def create_dataloaders(data_dict, batch_size=64, window_size=1, seq2seq=False, use_physics=False):
```

**修改后:**
```python
def create_dataloaders(data_dict, batch_size=64, window_size=1, seq2seq=False, use_physics=False, siamese_mode=False, step_k=1):
```

**向后兼容性**: 新参数有默认值 `siamese_mode=False`, `step_k=1`

---

### 4. 修改 collate_fn 支持两种数据格式 (Lines 226-256)

**修改内容**:
- 检测数据格式（通过 'window' vs 'x_t' key）
- 标准模式: 返回 `{'window', 'target_soh', 'battery_id', 'cycle_idx'}`
- 孪生模式: 返回 `{'x_t', 'x_next', 'y_t', 'y_next', 'battery_id', 'cycle_idx'}`

**向后兼容性**: 自动检测格式，标准模式路径完全保留

---

### 5. Dataset 创建传递孪生参数 (Lines 278, 294, 310)

**修改内容**:
```python
dataset = HUSTBatteryDatasetWithMetadata(X, y, bid, cyc, siamese_mode=siamese_mode, step_k=step_k)
```

**向后兼容性**: Dataset 内部检查 `siamese_mode`，默认 `False` 时使用标准格式

---

### 6. Dataloader 创建调用传递参数 (Lines 536-544)

**修改内容**:
```python
train_loader, val_loader, test_loader, test_battery_ids = create_dataloaders(
    data_dict,
    batch_size=config['training']['batch_size'],
    window_size=window_size,
    seq2seq=is_seq2seq,
    use_physics=use_physics,
    siamese_mode=siamese_mode,  # 新增
    step_k=step_k  # 新增
)
```

---

### 7. 损失函数创建逻辑 (Lines 560-592)

**修改内容**: 根据 `siamese_mode` 选择损失函数

```python
if use_physics:
    if siamese_mode:
        # 孪生模式：使用 SiamesePhysicsLoss
        criterion = SiamesePhysicsLoss(
            base_loss_weight=physics_config.get('base_loss_weight', 1.0),
            monotonic_weight=physics_config.get('monotonic_weight', 0.1),
            smoothness_weight=physics_config.get('smoothness_weight', 0.01),
            boundary_weight=physics_config.get('boundary_weight', 0.05),
            split_threshold=split_threshold,
            monotonic_tolerance=physics_config.get('monotonic_tolerance', 0.0),
            verbose=False
        ).to(device)
        print(f"\n损失函数: SiamesePhysicsLoss (孪生采样 + 分段约束)")
    else:
        # 标准模式：使用 PhysicsConstrainedLoss
        criterion = PhysicsConstrainedLoss(...)
        print("\n损失函数: PhysicsConstrainedLoss (物理约束)")
else:
    criterion = wrapper.criterion
    print(f"\n损失函数: {type(criterion).__name__} (标准)")
```

**向后兼容性**: `siamese_mode=False` 时完全使用原有逻辑

---

### 8. 修改训练循环 (Lines 629-698)

**修改内容**: 支持两种数据格式

```python
if isinstance(batch, dict):
    if 'x_t' in batch:
        # 孪生模式
        x_t = batch['x_t'].to(device)
        x_next = batch['x_next'].to(device)
        y_t = batch['y_t'].to(device)
        y_next = batch['y_next'].to(device)
        cycle_indices = batch['cycle_idx']

        # 两次前向传播
        pred_t = model(x_t)
        pred_next = model(x_next)

        # 孪生损失
        loss = criterion(pred_t, pred_next, y_t, y_next, cycle_indices)
    else:
        # 标准模式（原有逻辑）
        features = batch['window'].to(device)
        targets = batch['target_soh'].to(device)
        ...
```

**向后兼容性**: 通过 key 检测自动分支，标准模式完全保留

---

### 9. 修改验证循环 (Lines 722-807)

**修改内容**: 类似训练循环，支持两种格式

**新增**:
- 累积 `mask_active_ratio` 用于孪生模式
- MAE/RMSE 计算时对孪生模式取两个预测的平均

**向后兼容性**: 标准模式路径完全保留

---

### 10. 初始化 physics_loss_details (Lines 715-722)

**修改内容**:
```python
physics_loss_details = {
    'base': 0.0,
    'monotonic': 0.0,
    'boundary': 0.0,
    'smoothness': 0.0,
    'mask_active_ratio': 0.0  # 新增，用于孪生模式
}
mask_batch_count = 0  # 新增，用于计算 mask 平均值
```

---

### 11. 累积 Mask 统计 (Lines 752-755)

**修改内容**:
```python
# 累积mask比例（用于孪生模式）
if 'mask_active_ratio' in details:
    physics_loss_details['mask_active_ratio'] += details['mask_active_ratio']
    mask_batch_count += 1
```

---

### 12. 计算平均损失详情 (Lines 815-821)

**修改内容**:
```python
if use_physics:
    for key in physics_loss_details:
        if key == 'mask_active_ratio' and mask_batch_count > 0:
            # mask_active_ratio 按batch数量平均
            physics_loss_details[key] /= mask_batch_count
        else:
            # 其他损失按样本数量平均
            physics_loss_details[key] /= len(val_loader.dataset)
```

**向后兼容性**: 标准模式下 `mask_batch_count=0`，不影响其他字段

---

### 13. 输出格式显示 Mask 信息 (Lines 859-861)

**修改内容**:
```python
# 如果是孪生模式，显示Mask激活比例
if siamese_mode and physics_loss_details['mask_active_ratio'] > 0:
    print(f"    Mask激活比例:    {physics_loss_details['mask_active_ratio']*100:.1f}% (cycle >= {split_threshold})")
```

**向后兼容性**: 仅在孪生模式且有mask数据时显示

---

## 向后兼容性验证

### 默认参数确保兼容
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `siamese_mode` | `False` | 默认使用标准模式 |
| `step_k` | `1` | 默认步长为1 |
| `siamese_sampling.enabled` | 配置中不存在时为 `False` | 使用 `.get()` 方法 |

### 代码分支确保兼容
- 所有关键逻辑使用 `if siamese_mode:` 判断
- 标准模式分支保留原有代码不变
- 数据格式通过 key 检测自动分支

### 测试验证
创建了 `test_cross_battery_compatibility.py` 脚本验证：
1. 未启用孪生采样的配置（GRU, LSTM）应使用标准损失函数
2. 启用孪生采样的配置（CNN-LSTM）应使用孪生损失函数
3. 输出格式根据模式自动调整

---

## 使用方法

### 方法1: 通过配置文件启用（推荐）

编辑模型配置文件 (如 `configs/models/cnn_lstm_config.json`):

```json
{
  "physics_constraints": {
    "enabled": true,
    "siamese_sampling": {
      "enabled": true,
      "split_threshold": 300,
      "step_k": 1
    }
  }
}
```

然后运行:
```bash
python train_cross_battery.py --model cnn_lstm
```

### 方法2: 保持默认（标准模式）

不修改配置文件，或设置 `"enabled": false`:

```bash
python train_cross_battery.py --model gru
```

---

## 预期输出对比

### 标准模式输出
```
损失函数: PhysicsConstrainedLoss (物理约束)

物理约束详细损失:
  基础损失 (MSE):  0.000627
  单调性损失:      0.000123
  边界损失:        0.000000
  平滑性损失:      0.000045
```

### 孪生模式输出
```
损失函数: SiamesePhysicsLoss (孪生采样 + 分段约束)
  分段阈值: 300 cycles

物理约束详细损失:
  基础损失 (MSE):  0.152576
  单调性损失:      0.000000
  边界损失:        0.000000
  平滑性损失:      0.000000
  Mask激活比例:    81.8% (cycle >= 300)
```

**关键区别**: 孪生模式会显示 "Mask激活比例"

---

## 测试建议

1. **快速测试标准模式**:
   ```bash
   python train_cross_battery.py --model gru --max_batteries 2 --epochs 1
   ```
   预期: 不显示 Mask 信息

2. **快速测试孪生模式**:
   ```bash
   python train_cross_battery.py --model cnn_lstm --max_batteries 2 --epochs 1
   ```
   预期: 显示 Mask 信息

3. **完整兼容性测试**:
   ```bash
   python test_cross_battery_compatibility.py
   ```

---

## 文件清单

修改的文件:
- ✅ `train_cross_battery.py` (主要修改)
- ✅ `configs/models/cnn_lstm_config.json` (启用孪生采样)

新增的文件:
- ✅ `test_cross_battery_compatibility.py` (兼容性测试脚本)
- ✅ `CROSS_BATTERY_SIAMESE_SUMMARY.md` (本文档)

---

## 相关文档

- `SIAMESE_USAGE_GUIDE.md` - 孪生采样使用指南
- `SIAMESE_TRAINING_PROCESS_EXPLAINED.md` - 训练过程详解
- `QUICK_COMPARISON.md` - 前300循环 vs 后期循环对比
- `check_siamese_configs.py` - 配置检查脚本

---

## 总结

所有修改均遵循向后兼容原则：
- ✅ 默认行为不变（siamese_mode=False）
- ✅ 所有新参数有合理默认值
- ✅ 代码分支清晰，标准模式路径完全保留
- ✅ 自动检测数据格式，无需手动干预
- ✅ 输出格式根据模式自动调整

**你的原有代码和训练流程不会受到任何影响！**
