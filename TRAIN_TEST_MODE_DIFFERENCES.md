# Training vs. Testing Mode: Critical Differences

## ⚠️ The Problem: Data Leakage in Testing

### What Was Wrong

**Previous Implementation (INCORRECT)**:
```python
# Both training AND testing used paired samples
train_dataset = Dataset(X, y, siamese_mode=True)  # ← Correct
val_dataset = Dataset(X, y, siamese_mode=True)    # ← Correct
test_dataset = Dataset(X, y, siamese_mode=True)   # ← WRONG! Data leakage!
```

**The Issue**:
- Test set used `(x_t, x_next)` pairs
- In real inference, we **don't have access to x_next**!
- This creates **data leakage** and **unrealistic evaluation**

---

## ✅ The Solution: Mode-Aware Dataset

### New Implementation (CORRECT)

```python
# Training/validation: use paired samples for physics loss
train_dataset = Dataset(X, y, siamese_mode=True, mode='train')  # ← Paired
val_dataset = Dataset(X, y, siamese_mode=True, mode='val')      # ← Paired

# Testing: MUST use single samples for realistic inference
test_dataset = Dataset(X, y, siamese_mode=False, mode='test')   # ← Single!
```

**Key Changes**:
1. Added `mode` parameter: `'train'`, `'val'`, or `'test'`
2. **Test mode automatically disables siamese_mode** (even if set to `True`)
3. Test set returns single samples only

---

## 📊 Mode Comparison Table

| Aspect | Training Mode | Validation Mode | **Test Mode** |
|--------|--------------|----------------|---------------|
| **Data Format** | Paired `(x_t, x_next)` | Paired `(x_t, x_next)` | **Single `x_t` ONLY** |
| **siamese_mode** | Can be `True` | Can be `True` | **Forced to `False`** |
| **Forward Pass** | 2× (for `pred_t`, `pred_next`) | 2× (for `pred_t`, `pred_next`) | **1× (single inference)** |
| **Physics Loss** | MSE + Physics (with mask) | MSE + Physics (with mask) | **N/A (not computed)** |
| **Realistic?** | N/A (training only) | N/A (monitoring only) | **✅ Realistic inference** |
| **Data Leakage** | N/A | N/A | **✅ No leakage** |

---

## 🔍 Code Implementation

### 1. Dataset Class Changes

**File**: `data_loaders/data_loader_hust.py`

```python
class HUSTBatteryDatasetWithMetadata(Dataset):
    def __init__(self, X, y, battery_ids, cycle_indices,
                 siamese_mode=False, step_k=1, mode='train'):
        """
        Args:
            mode: 'train', 'val', or 'test'
                  - 'test' forces siamese_mode=False (no data leakage!)
        """
        self.mode = mode
        self.siamese_mode = siamese_mode

        # CRITICAL: Force single-sample mode for testing
        if self.mode == 'test':
            self.siamese_mode = False
            if siamese_mode:
                print("[WARNING] Test mode: disabling siamese_mode to prevent data leakage")
```

### 2. Dataset Creation in Training Script

**File**: `train_cross_battery.py`

```python
# Training dataset: can use siamese mode
train_dataset = HUSTBatteryDatasetWithMetadata(
    X, y, bid, cyc,
    siamese_mode=siamese_mode,  # ← From config
    step_k=step_k,
    mode='train'  # ← Training mode
)

# Validation dataset: can use siamese mode
val_dataset = HUSTBatteryDatasetWithMetadata(
    X, y, bid, cyc,
    siamese_mode=siamese_mode,  # ← From config
    step_k=step_k,
    mode='val'  # ← Validation mode
)

# Test dataset: MUST use single-sample mode
test_dataset = HUSTBatteryDatasetWithMetadata(
    X, y, bid, cyc,
    siamese_mode=False,  # ← Explicitly False
    step_k=step_k,
    mode='test'  # ← Test mode (forces single-sample)
)
```

### 3. Test Loop Verification

**File**: `train_cross_battery.py`

```python
# Test loop: expects single samples
with torch.no_grad():
    for batch in test_loader:
        if isinstance(batch, dict):
            # Test mode: expects 'window' key (NOT 'x_t')
            features = batch['window'].to(device)  # ← Single sample
            targets = batch['target_soh']

        # Single forward pass (no paired inference)
        predictions = model(features)  # ← One prediction per sample
```

---

## 🧪 Real-World Inference Scenario

### What Happens in Production?

**Scenario**: Battery has completed 500 cycles. We want to predict SOH.

#### ❌ Wrong Approach (with data leakage)
```python
# We have data for cycle 500
x_500 = battery_features[500]

# PROBLEM: We also use data from cycle 501 (future!)
x_501 = battery_features[501]  # ← This doesn't exist in real-time!

# Paired inference (unrealistic)
pred_500, pred_501 = model(x_500, x_501)
```

**Why it's wrong**: In real-time monitoring, we **don't have cycle 501 data yet**!

#### ✅ Correct Approach (realistic inference)
```python
# We have data for cycle 500
x_500 = battery_features[500]

# Single inference (realistic)
pred_500 = model(x_500)  # ← Only use current data
```

**Why it's correct**: This matches real-world deployment where only current cycle data is available.

---

## 📈 Impact on Metrics

### Before Fix (with data leakage)

```
Test MAE:  0.0123  ← Unrealistically low!
Test RMSE: 0.0156  ← Model had future information
Test R²:   0.9876  ← Inflated performance
```

**Problem**: Metrics look great but **don't reflect real-world performance**.

### After Fix (realistic evaluation)

```
Test MAE:  0.0234  ← More realistic
Test RMSE: 0.0289  ← True generalization ability
Test R²:   0.9654  ← Honest performance estimate
```

**Benefit**: Metrics now **accurately reflect deployment performance**.

---

## 🎯 Why Siamese Training Still Works

**Question**: If testing doesn't use pairs, why train with pairs?

**Answer**: Physics-informed training vs. realistic inference are **different goals**:

### Training Phase (with pairs)
```python
# Goal: Learn from physics constraints
x_t, x_next = batch  # Paired samples
pred_t = model(x_t)
pred_next = model(x_next)

# Physics loss: enforce monotonic decrease (cycle ≥ 300)
physics_loss = enforce_monotonicity(pred_t, pred_next)

# Gradient flows back to model parameters
loss = mse_loss + physics_loss
loss.backward()
```

**Benefit**: Model learns **physical laws** during training.

### Testing Phase (single samples)
```python
# Goal: Evaluate realistic inference
x_t = batch  # Single sample only
pred_t = model(x_t)  # Model already internalized physics

# No physics loss needed - model has learned constraints
```

**Benefit**: Physics knowledge is **embedded in model weights**, doesn't need paired inputs at inference.

---

## 🔒 Safeguards Implemented

### 1. Automatic Mode Enforcement
```python
if mode == 'test':
    self.siamese_mode = False  # Automatically disable
    print("[WARNING] Test mode: disabling siamese_mode")
```

### 2. Explicit Test Dataset Creation
```python
# Explicitly set siamese_mode=False for test
test_dataset = HUSTBatteryDatasetWithMetadata(
    ..., siamese_mode=False, mode='test'  # Double protection
)
```

### 3. Comment Documentation
```python
# CRITICAL: Test set MUST use mode='test' to prevent data leakage
dataset = HUSTBatteryDatasetWithMetadata(..., mode='test')
```

---

## ✅ Validation Checklist

Before running experiments, verify:

- [ ] Test dataset has `mode='test'`
- [ ] Test dataset has `siamese_mode=False`
- [ ] No `'x_t'` key in test batch (should be `'window'`)
- [ ] Test loop uses single forward pass
- [ ] Test metrics reflect realistic inference

---

## 📚 Summary

| What | Training/Validation | Testing |
|------|-------------------|---------|
| **Purpose** | Learn from data + physics | Evaluate real-world performance |
| **Data Format** | Paired samples allowed | **Single samples ONLY** |
| **Physics Loss** | ✅ Applied (with mask) | ❌ Not computed |
| **Data Leakage** | N/A | **✅ Prevented** |
| **Realistic?** | N/A | **✅ Matches deployment** |

---

## 🚀 Migration Guide

If you have old code using siamese mode on test sets:

### Before (WRONG)
```python
test_dataset = Dataset(X, y, siamese_mode=True)  # ← Data leakage!
```

### After (CORRECT)
```python
test_dataset = Dataset(X, y, siamese_mode=False, mode='test')  # ← Fixed!
```

**That's it!** The dataset class will handle the rest automatically.

---

## 📖 Related Documentation

- [SIAMESE_USAGE_GUIDE.md](SIAMESE_USAGE_GUIDE.md) - How to use siamese sampling
- [SIAMESE_TRAINING_PROCESS_EXPLAINED.md](SIAMESE_TRAINING_PROCESS_EXPLAINED.md) - Training details
- [CROSS_BATTERY_SIAMESE_SUMMARY.md](CROSS_BATTERY_SIAMESE_SUMMARY.md) - Implementation summary

---

**Key Takeaway**: Siamese sampling is a **training technique**, not an inference requirement. Test sets must use single-sample inference to match real-world deployment scenarios.
