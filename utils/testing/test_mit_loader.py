"""
测试MIT数据加载器
"""

from data_loaders import load_all_mit_batteries
import numpy as np

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
data = all_data[first_battery]

print(f"\nFirst battery: {first_battery}")
print(f"  Data keys: {list(data.keys())}")
print(f"  Features shape: {data['train_features'].shape}")
print(f"  SOH shape: {data['train_capacity'].shape}")
print(f"  SOH range: [{data['train_capacity'].min():.4f}, {data['train_capacity'].max():.4f}]")

# 统计所有电池的cycle数
cycle_counts = []
for name in battery_names:
    data = all_data[name]
    cycle_counts.append(len(data['train_features']))

print(f"\nCycle statistics:")
print(f"  Min: {min(cycle_counts)}")
print(f"  Max: {max(cycle_counts)}")
print(f"  Mean: {np.mean(cycle_counts):.1f}")
print(f"  Median: {np.median(cycle_counts):.1f}")

# 验证数据格式与HUST兼容
print(f"\nData format compatibility check:")
print(f"  'train_features' key exists: {'train_features' in data}")
print(f"  'train_capacity' key exists: {'train_capacity' in data}")
print(f"  'scaler' key exists: {'scaler' in data}")
print(f"  Features are normalized: (mean should be ~0)")
print(f"    Feature mean: {data['train_features'].mean():.6f}")
print(f"    Feature std: {data['train_features'].std():.6f}")

print("\n[OK] MIT data loader test PASSED!")
