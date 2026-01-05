"""
测试MIT数据加载器
"""

from data_loaders import load_all_mit_batteries

# 测试加载MIT数据
print("Testing MIT data loader...")
battery_names, all_data = load_all_mit_batteries(
    data_dir='data/MIT data',
    apply_cleaning=False
)

print(f"\nTotal batteries: {len(battery_names)}")
print(f"Battery names (first 10): {battery_names[:10]}")

# 检查第一个电池的数据
first_battery = battery_names[0]
features, soh, capacity = all_data[first_battery]

print(f"\nFirst battery: {first_battery}")
print(f"  Features shape: {features.shape}")
print(f"  SOH shape: {soh.shape}")
print(f"  SOH range: [{soh.min():.4f}, {soh.max():.4f}]")
print(f"  Capacity shape: {capacity.shape}")
print(f"  Capacity range: [{capacity.min():.4f}, {capacity.max():.4f}]")
print(f"  Initial capacity: {capacity[0]:.4f}")

# 统计所有电池的cycle数
cycle_counts = []
for name in battery_names:
    features, soh, capacity = all_data[name]
    cycle_counts.append(len(features))

import numpy as np
print(f"\nCycle statistics:")
print(f"  Min: {min(cycle_counts)}")
print(f"  Max: {max(cycle_counts)}")
print(f"  Mean: {np.mean(cycle_counts):.1f}")
print(f"  Median: {np.median(cycle_counts):.1f}")

print("\n[OK] MIT data loader test PASSED!")
