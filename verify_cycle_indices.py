"""
Verify what cycle indices actually are in the windowed data
"""
import numpy as np
from data_loaders import load_single_hust_battery
from data_loaders.data_loader_hust import apply_windowing_with_metadata

print("="*70)
print("Verifying Actual Cycle Indices from apply_windowing_with_metadata")
print("="*70)

# Load one battery
file_path = 'data/HUST data/1-1.csv'
data = load_single_hust_battery(file_path, train_ratio=1.0, normalize_target=False)

features = data['train_features']
targets = data['train_capacity']

print(f"\nLoaded battery 1-1:")
print(f"  Total samples: {len(features)}")
print(f"  Feature shape: {features.shape}")
print(f"  Target shape: {targets.shape}")

# Apply windowing
window_size = 10
battery_id = '1-1'

X, y, bid, cyc = apply_windowing_with_metadata(
    features, targets, window_size, battery_id, mode='many_to_one'
)

print(f"\nAfter windowing (window_size={window_size}):")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  battery_ids shape: {bid.shape}")
print(f"  cycle_indices shape: {cyc.shape}")

print(f"\nCycle indices statistics:")
print(f"  Min: {cyc.min()}")
print(f"  Max: {cyc.max()}")
print(f"  Mean: {cyc.mean():.1f}")
print(f"  Median: {np.median(cyc):.1f}")

# Check threshold
threshold = 300
below = (cyc < threshold).sum()
above = (cyc >= threshold).sum()

print(f"\nWith split_threshold={threshold}:")
print(f"  Cycle < {threshold}: {below} ({below/len(cyc)*100:.1f}%)")
print(f"  Cycle >= {threshold}: {above} ({above/len(cyc)*100:.1f}%)")

print(f"\nFirst 10 cycle indices: {cyc[:10]}")
print(f"Last 10 cycle indices: {cyc[-10:]}")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print(f"cycle_idx = sample index within battery (0-indexed)")
print(f"For battery 1-1 with {len(features)} samples:")
print(f"  - Window size = {window_size}")
print(f"  - Valid windows = {len(cyc)}")
print(f"  - cycle_idx range = {cyc.min()} to {cyc.max()}")
print(f"  - {above/len(cyc)*100:.1f}% of samples have cycle_idx >= {threshold}")
print(f"\nThis WILL activate the mask for later samples in each battery!")
print("="*70)
