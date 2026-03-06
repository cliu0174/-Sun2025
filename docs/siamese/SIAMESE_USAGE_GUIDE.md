# Siamese Sampling + Split Constraint Usage Guide

## Overview

This implementation adds **Siamese/Pairwise Sampling** with **Split Constraint** to your PINN training, allowing you to handle the early-cycle capacity regeneration phenomenon in battery data.

### Key Features

1. **Dataset returns paired consecutive samples**: `(x_t, x_{t+1}, y_t, y_{t+1})`
2. **Split Constraint**: Physics loss only applies to cycles >= threshold
3. **Backward Compatible**: Standard mode still works with `siamese_mode=False`
4. **Easy Switch**: Toggle between modes with a single flag

---

## Quick Start

### Method 1: Enable via Configuration File (Recommended)

Edit `configs/models/lstm_config.json`, find the `physics_constraints.siamese_sampling` section:

```json
"siamese_sampling": {
  "enabled": true,
  "split_threshold": 300,
  "step_k": 1,
  "description": "孪生采样：前split_threshold循环无约束，后期强制单调"
}
```

Then simply run:

```bash
python train_with_physics.py --model lstm
```

### Method 2: Enable via Command-Line Arguments

```bash
# Enable Siamese mode with split threshold at 300 cycles
python train_with_physics.py --model lstm --siamese --split_threshold 300
```

**Note**: Command-line arguments override configuration file settings.

---

## Configuration Parameters

### Configuration File Location

`configs/models/lstm_config.json` (or other model config files)

### Siamese Sampling Configuration

```json
"physics_constraints": {
  "siamese_sampling": {
    "enabled": false,          // Enable siamese sampling mode
    "split_threshold": 300,    // Cycle threshold for masking physics loss
    "step_k": 1,               // Step size for pairing (1 = consecutive)
    "description": "孪生采样：前split_threshold循环无约束，后期强制单调"
  }
}
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `lstm` | Model type (lstm, gru, bilstm, bigru, etc.) |
| `--device` | str | `cuda` | Device (cuda/cpu) |
| `--max_batteries` | int | `None` | Max number of batteries to load |
| `--no_physics` | flag | False | Disable physics constraints |
| `--cleaning` | flag | False | Enable data cleaning |
| **`--siamese`** | **flag** | **False** | **Enable Siamese sampling mode** |
| **`--split_threshold`** | **int** | **300** | **Cycle threshold for masking physics loss** |
| **`--step_k`** | **int** | **1** | **Step size for pairing (1 = consecutive)** |

---

## Examples

### Example 1: Basic Siamese Mode

```bash
python train_with_physics.py \
    --model lstm \
    --siamese \
    --split_threshold 300
```

**What happens:**
- Dataset returns pairs: `(x_t, x_{t+1})`
- For cycles **< 300**: Only MSE loss applied (allows capacity rise)
- For cycles **≥ 300**: MSE + Physics loss (enforces monotonic decrease)

### Example 2: Different Split Thresholds

Based on our data analysis:

```bash
# Conservative (300 cycles)
python train_with_physics.py --model lstm --siamese --split_threshold 300

# Moderate (500 cycles)
python train_with_physics.py --model lstm --siamese --split_threshold 500

# Aggressive (800 cycles)
python train_with_physics.py --model lstm --siamese --split_threshold 800
```

### Example 3: Quick Test with 3 Batteries

```bash
python train_with_physics.py \
    --model lstm \
    --siamese \
    --split_threshold 300 \
    --max_batteries 3
```

### Example 4: Comparison Experiment

```bash
# Baseline (no physics)
python train_with_physics.py --model lstm --no_physics

# Standard physics (original)
python train_with_physics.py --model lstm

# Siamese physics (new)
python train_with_physics.py --model lstm --siamese --split_threshold 300
```

---

## How It Works

### Data Flow

#### Standard Mode (`siamese_mode=False`)

```
CSV → load_single_hust_battery() → apply_windowing_with_metadata()
  ↓
Dataset returns: {'window': X_t, 'target_soh': Y_t, 'cycle_idx': cycle_t}
  ↓
Training: pred = model(X_t)
         loss = PhysicsConstrainedLoss(pred, Y_t, battery_ids, cycle_indices)
```

#### Siamese Mode (`siamese_mode=True`)

```
CSV → load_single_hust_battery() → apply_windowing_with_metadata()
  ↓
Dataset returns: {
    'x_t': X_t,
    'x_next': X_{t+1},
    'y_t': Y_t,
    'y_next': Y_{t+1},
    'cycle_index': cycle_t
}
  ↓
Training: pred_t = model(X_t)
         pred_next = model(X_{t+1})
         loss = SiamesePhysicsLoss(pred_t, pred_next, Y_t, Y_{t+1}, cycle_t)
           ↓
         if cycle_t < 300:
             loss = MSE(pred_t, Y_t) + MSE(pred_next, Y_{t+1})
         else:
             loss = MSE + Physics_Constraints
```

### Loss Function Components

#### SiamesePhysicsLoss

```python
Total Loss = MSE_component + Physics_component

MSE_component = (MSE(pred_t, y_t) + MSE(pred_next, y_next)) / 2
  # Always applied to ALL samples

Physics_component = mask * (Monotonic_loss + Smoothness_loss)
  # Only applied where mask = 1 (cycle >= threshold)

where:
    mask = 1 if cycle_index >= split_threshold else 0

    Monotonic_loss = ReLU(pred_next - pred_t - tolerance)^2
      # Penalizes capacity increase

    Smoothness_loss = (pred_next - pred_t)^2
      # Penalizes large jumps
```

---

## Configuration

### Model Config (configs/models/lstm_config.json)

For Siamese mode, the physics constraints config is used:

```json
{
  "physics_constraints": {
    "enabled": true,
    "base_loss_weight": 1.0,
    "monotonic_weight": 0.1,        // Weight for monotonicity constraint
    "smoothness_weight": 0.01,      // Weight for smoothness constraint
    "boundary_weight": 0.05,        // Weight for boundary constraint
    "monotonic_tolerance": 0.0      // Tolerance (0 = strict decrease)
  }
}
```

**Note:** In Siamese mode:
- `split_threshold` is set via command-line argument
- `temporal_decay` parameters are **not used** (Siamese uses direct pairing)

---

## Expected Results

### Data Statistics (from analysis)

| Threshold | Early Non-Monotonic | Late Non-Monotonic | Improvement |
|-----------|--------------------|--------------------|-------------|
| 300 cycles | 42.4% | 34.8% | 7.5% |
| 500 cycles | 43.0% | 33.3% | **9.7%** |
| 800 cycles | 42.5% | 30.4% | **12.2%** |

### Training Output

```
加载 77 个电池数据 (孪生采样模式)...
成功加载 77 个电池

数据集大小:
  训练集: 45000 配对
  验证集: 15000 配对
  测试集: 20000 配对

模型: LSTM
  参数数量: 50,000

损失函数: SiamesePhysicsLoss (孪生采样 + 分段约束)
  分段阈值: 300 cycles
  单调性权重: 0.1
  平滑性权重: 0.01
  容忍度: 0.0

Epoch 1/100: Train Loss = 0.003456, Val Loss = 0.002987
  [详细] Base: 0.0023, Mono: 0.0004, Smooth: 0.0001, Bound: 0.0000, Mask: 67.3%
```

**Mask active ratio** = % of samples where cycle >= threshold (physics applies)

---

## Implementation Details

### Modified Files

1. **data_loaders/data_loader_hust.py**
   - `HUSTBatteryDatasetWithMetadata`: Added `siamese_mode` parameter
   - Returns paired samples when `siamese_mode=True`

2. **models/physics_loss.py**
   - Added `SiamesePhysicsLoss` class
   - Implements mask-based split constraint

3. **models/__init__.py**
   - Exported `SiamesePhysicsLoss`

4. **train_with_physics.py**
   - Updated `load_batteries_with_windowing()` to support Siamese mode
   - Updated `custom_collate_fn()` to handle both modes
   - Updated training loop to handle both modes
   - Added command-line arguments: `--siamese`, `--split_threshold`, `--step_k`

### Backward Compatibility

✅ **All existing functionality preserved**:
- Standard mode works exactly as before
- No breaking changes to existing code
- Default parameters maintain original behavior

---

## Troubleshooting

### Issue: "KeyError: 'x_t'"

**Cause:** Dataset is in standard mode but training expects Siamese mode.

**Solution:** Make sure `siamese_mode` is consistently set:

```python
# In load_batteries_with_windowing()
siamese_mode=True

# In train_with_physics_constraints()
siamese_mode=True
```

### Issue: "Mask active ratio = 0%"

**Cause:** `split_threshold` is too high, all samples are masked out.

**Solution:** Lower the threshold:

```bash
python train_with_physics.py --siamese --split_threshold 100
```

### Issue: "Too few samples in dataset"

**Cause:** Siamese mode filters out samples without valid next samples.

**Expected:** Dataset size reduces slightly (~1% for step_k=1)

---

## Performance Comparison

### Recommended Experiments

| Experiment | Command | Expected Outcome |
|------------|---------|------------------|
| Baseline | `--model lstm --no_physics` | Best MSE, no constraints |
| Standard PINN | `--model lstm` | Moderate, penalizes early rise |
| Siamese (300) | `--model lstm --siamese --split_threshold 300` | Should improve over standard |
| Siamese (500) | `--model lstm --siamese --split_threshold 500` | Possibly best balance |
| Siamese (800) | `--model lstm --siamese --split_threshold 800` | Most permissive |

---

## Next Steps

1. **Quick Test**: Run with 3 batteries to verify setup
   ```bash
   python train_with_physics.py --model lstm --siamese --max_batteries 3
   ```

2. **Threshold Tuning**: Try different thresholds
   ```bash
   for threshold in 300 500 800; do
       python train_with_physics.py --model lstm --siamese --split_threshold $threshold
   done
   ```

3. **Full Training**: Train on all 77 batteries
   ```bash
   python train_with_physics.py --model lstm --siamese --split_threshold 500
   ```

4. **Analysis**: Compare results in `results/physics_constraints/`

---

## Technical Notes

### Why Siamese Sampling?

**Problem**: Early cycles show capacity regeneration (35-43% non-monotonic), but standard PINN penalizes all violations.

**Solution**: Siamese sampling with split constraint allows:
- **Early cycles**: Learn freely from data (MSE only)
- **Late cycles**: Enforce physics constraints (MSE + Physics)

### Why Not Just Modify Tolerance?

Increasing `monotonic_tolerance` to 0.05 (5%) helps, but:
- Still penalizes 30%+ of late-cycle data
- Can't differentiate early vs late cycles
- Siamese mode gives surgical control

---

## References

- Data analysis: See section in `PINN_Analysis_Report.md`
- Original PhysicsConstrainedLoss: `models/physics_loss.py:19-380`
- SiamesePhysicsLoss: `models/physics_loss.py:382-528`

---

**Generated**: 2025-12-08
**Author**: Claude Code
**Version**: 1.0
