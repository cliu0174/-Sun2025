"""
Test monotonic constraint with REAL cycle values >= threshold
"""
import torch
import numpy as np
from data_loaders.data_loader_hust import HUSTBatteryDatasetWithMetadata
from torch.utils.data import DataLoader
from models.physics_loss import SiamesePhysicsLoss

print("="*70)
print("Testing Monotonic Constraint with Real Data (cycle >= 300)")
print("="*70)

# Create simulated data with HIGH cycle indices (>= 300)
window_size = 10
feature_dim = 5
num_samples = 100

# IMPORTANT: Cycles 300-399 (all should activate mask!)
X = np.random.randn(num_samples, window_size, feature_dim)
y = np.linspace(0.85, 0.80, num_samples)  # Monotonic decrease
battery_ids = ['battery_1'] * num_samples
cycle_indices = np.arange(300, 300 + num_samples)  # cycle 300-399

print(f"\nSimulated data:")
print(f"  Samples: {num_samples}")
print(f"  SOH range: {y[0]:.3f} to {y[-1]:.3f}")
print(f"  Cycle range: {cycle_indices[0]} to {cycle_indices[-1]}")
print(f"  Split threshold: 300")
print(f"  Expected mask_active_ratio: 100%")

# Create siamese dataset
dataset = HUSTBatteryDatasetWithMetadata(
    X, y, battery_ids, cycle_indices,
    siamese_mode=True, step_k=1, mode='train'
)

print(f"\nDataset created:")
print(f"  Original samples: {num_samples}")
print(f"  Valid pairs: {len(dataset)} (should be {num_samples-1})")

# Create dataloader
loader = DataLoader(dataset, batch_size=8, shuffle=False)
batch = next(iter(loader))

print(f"\nFirst batch:")
print(f"  Batch keys: {list(batch.keys())}")
cycle_key = 'cycle_index' if 'cycle_index' in batch else 'cycle_idx'
print(f"  Batch size: {len(batch[cycle_key])}")
print(f"  Cycle indices: {batch[cycle_key].tolist()}")
print(f"  y_t:   {batch['y_t'].numpy().tolist()}")
print(f"  y_next: {batch['y_next'].numpy().tolist()}")

# Create loss function
criterion = SiamesePhysicsLoss(
    base_loss_weight=1.0,
    monotonic_weight=0.1,
    smoothness_weight=0.05,
    split_threshold=300,  # Should activate for ALL samples
    monotonic_tolerance=0.0,  # Strict
    verbose=False
)

print("\n" + "="*70)
print("Test 1: Perfect Predictions (pred = true)")
print("="*70)
pred_t = batch['y_t']
pred_next = batch['y_next']

loss = criterion(pred_t, pred_next, batch['y_t'], batch['y_next'], batch[cycle_key])
details = criterion.get_loss_details()

print(f"MSE loss: {details['base']:.6f} (should be 0)")
print(f"Monotonic loss: {details['monotonic']:.6f} (should be 0, true values decrease)")
print(f"Smoothness loss: {details['smoothness']:.6f}")
print(f"Mask active ratio: {details['mask_active_ratio']*100:.1f}% (should be 100%)")

print("\n" + "="*70)
print("Test 2: Forced Increase (pred_next > pred_t) - VIOLATION!")
print("="*70)
pred_t = batch['y_t']
pred_next = batch['y_t'] + 0.01  # Force increase!

loss = criterion(pred_t, pred_next, batch['y_t'], batch['y_next'], batch[cycle_key])
details = criterion.get_loss_details()

print(f"MSE loss: {details['base']:.6f}")
print(f"Monotonic loss: {details['monotonic']:.6f} (should be > 0, penalizing increase!)")
print(f"Smoothness loss: {details['smoothness']:.6f}")
print(f"Mask active ratio: {details['mask_active_ratio']*100:.1f}% (should be 100%)")

if details['monotonic'] > 0:
    print("\n[SUCCESS] Monotonic constraint is WORKING! Violations are penalized.")
else:
    print("\n[CRITICAL ERROR] Monotonic loss is STILL 0 even with forced violations!")
    print("This indicates a bug in the loss function!")

print("\n" + "="*70)
print("Test 3: Check Individual Sample Computation")
print("="*70)
# Manually compute what the loss should be
pred_t_val = batch['y_t'][0].item()
pred_next_val = batch['y_t'][0].item() + 0.01
cycle_val = batch[cycle_key][0].item()

diff = pred_next_val - pred_t_val
mask = 1.0 if cycle_val >= 300 else 0.0
violation = max(0, diff - 0.0)  # tolerance = 0.0
expected_mono_loss = (violation ** 2) * mask

print(f"Sample 0:")
print(f"  pred_t = {pred_t_val:.6f}")
print(f"  pred_next = {pred_next_val:.6f}")
print(f"  cycle = {cycle_val}")
print(f"  diff = pred_next - pred_t = {diff:.6f}")
print(f"  mask = {mask} (cycle >= 300)")
print(f"  violation = max(0, diff - 0.0) = {violation:.6f}")
print(f"  Expected monotonic loss = {expected_mono_loss:.6f}")
print(f"  Actual monotonic loss = {details['monotonic']:.6f}")

print("\n" + "="*70)
