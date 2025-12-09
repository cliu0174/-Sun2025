"""
Verification Script: Ensure Test Mode Uses Single-Sample Inference

This script verifies that:
1. Test dataset forces siamese_mode=False
2. Test batch has 'window' key (not 'x_t')
3. No data leakage in test evaluation
"""

import numpy as np
import torch
from data_loaders.data_loader_hust import HUSTBatteryDatasetWithMetadata

def verify_dataset_mode():
    """Verify Dataset mode enforcement"""
    print("="*70)
    print("Dataset Mode Verification")
    print("="*70)

    # Create dummy data
    N = 100
    window_size = 5
    feature_dim = 16

    X = np.random.randn(N, window_size, feature_dim)
    y = np.random.rand(N)
    battery_ids = ['battery_1'] * N
    cycle_indices = np.arange(N)

    print("\n[Test 1] Training mode with siamese_mode=True")
    print("-"*70)
    train_ds = HUSTBatteryDatasetWithMetadata(
        X, y, battery_ids, cycle_indices,
        siamese_mode=True, step_k=1, mode='train'
    )
    sample_train = train_ds[0]
    print(f"Keys in sample: {list(sample_train.keys())}")
    print(f"Expected: ['x_t', 'x_next', 'y_t', 'y_next', 'battery_id', 'cycle_index']")

    if 'x_t' in sample_train:
        print("[PASS] Training dataset returns paired samples")
    else:
        print("[FAIL] Training dataset should return paired samples!")

    print("\n[Test 2] Validation mode with siamese_mode=True")
    print("-"*70)
    val_ds = HUSTBatteryDatasetWithMetadata(
        X, y, battery_ids, cycle_indices,
        siamese_mode=True, step_k=1, mode='val'
    )
    sample_val = val_ds[0]
    print(f"Keys in sample: {list(sample_val.keys())}")

    if 'x_t' in sample_val:
        print("[OK] PASS: Validation dataset returns paired samples")
    else:
        print("[FAIL] FAIL: Validation dataset should return paired samples!")

    print("\n[Test 3] Test mode with siamese_mode=True (should be overridden!)")
    print("-"*70)
    print("IMPORTANT: Test mode should FORCE siamese_mode=False")

    test_ds = HUSTBatteryDatasetWithMetadata(
        X, y, battery_ids, cycle_indices,
        siamese_mode=True,  # ← Try to enable siamese mode
        step_k=1,
        mode='test'  # ← Test mode should override!
    )

    sample_test = test_ds[0]
    print(f"Keys in sample: {list(sample_test.keys())}")
    print(f"Expected: ['window', 'target_soh', 'battery_id', 'cycle_idx']")

    if 'window' in sample_test and 'x_t' not in sample_test:
        print("[OK] PASS: Test dataset correctly forces single-sample mode!")
        print("[OK] PASS: No data leakage - test mode is safe!")
    else:
        print("[FAIL] FAIL: Test dataset should return single samples only!")
        print("[FAIL] FAIL: Data leakage detected - test mode is unsafe!")

    print("\n[Test 4] Test mode with siamese_mode=False (explicit)")
    print("-"*70)
    test_ds_explicit = HUSTBatteryDatasetWithMetadata(
        X, y, battery_ids, cycle_indices,
        siamese_mode=False,  # ← Explicitly False
        step_k=1,
        mode='test'
    )

    sample_test_explicit = test_ds_explicit[0]
    print(f"Keys in sample: {list(sample_test_explicit.keys())}")

    if 'window' in sample_test_explicit:
        print("[OK] PASS: Test dataset with explicit siamese_mode=False works correctly")
    else:
        print("[FAIL] FAIL: Test dataset configuration error!")

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)

    all_pass = (
        'x_t' in sample_train and
        'x_t' in sample_val and
        'window' in sample_test and
        'x_t' not in sample_test and
        'window' in sample_test_explicit
    )

    if all_pass:
        print("[OK] ALL TESTS PASSED!")
        print("[OK] Test mode correctly prevents data leakage")
        print("[OK] Training/validation modes support siamese sampling")
        print("\nYour implementation is CORRECT and SAFE for deployment!")
    else:
        print("[FAIL] SOME TESTS FAILED!")
        print("[FAIL] Please review the Dataset implementation")
        print("\nDO NOT use this for real experiments until fixed!")

    return all_pass

def verify_batch_format():
    """Verify batch collate function handles test mode correctly"""
    print("\n" + "="*70)
    print("Batch Format Verification")
    print("="*70)

    from torch.utils.data import DataLoader

    # Create dummy data
    N = 20
    window_size = 5
    feature_dim = 16

    X = np.random.randn(N, window_size, feature_dim)
    y = np.random.rand(N)
    battery_ids = ['battery_1'] * N
    cycle_indices = np.arange(N)

    # Create datasets
    train_ds = HUSTBatteryDatasetWithMetadata(
        X, y, battery_ids, cycle_indices,
        siamese_mode=True, mode='train'
    )

    test_ds = HUSTBatteryDatasetWithMetadata(
        X, y, battery_ids, cycle_indices,
        siamese_mode=True,  # ← Should be overridden
        mode='test'
    )

    # Custom collate function (from train_cross_battery.py)
    def custom_collate_fn(batch):
        if 'window' in batch[0]:
            # Single-sample mode
            windows = torch.stack([item['window'] for item in batch])
            targets = torch.cat([item['target_soh'].unsqueeze(0) for item in batch], dim=0)
            battery_ids = [item['battery_id'] for item in batch]
            cycle_indices = torch.stack([item['cycle_idx'] for item in batch])
            return {
                'window': windows,
                'target_soh': targets,
                'battery_id': battery_ids,
                'cycle_idx': cycle_indices
            }
        else:
            # Siamese mode
            x_t = torch.stack([item['x_t'] for item in batch])
            x_next = torch.stack([item['x_next'] for item in batch])
            y_t = torch.cat([item['y_t'].unsqueeze(0) for item in batch], dim=0)
            y_next = torch.cat([item['y_next'].unsqueeze(0) for item in batch], dim=0)
            battery_ids = [item['battery_id'] for item in batch]
            cycle_indices = torch.stack([item['cycle_index'] for item in batch])
            return {
                'x_t': x_t,
                'x_next': x_next,
                'y_t': y_t,
                'y_next': y_next,
                'battery_id': battery_ids,
                'cycle_idx': cycle_indices
            }

    train_loader = DataLoader(train_ds, batch_size=4, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=4, collate_fn=custom_collate_fn)

    print("\n[Test 5] Training batch format")
    print("-"*70)
    train_batch = next(iter(train_loader))
    print(f"Batch keys: {list(train_batch.keys())}")

    if 'x_t' in train_batch:
        print("[OK] PASS: Training batch has paired format")
    else:
        print("[FAIL] FAIL: Training batch should be paired!")

    print("\n[Test 6] Test batch format")
    print("-"*70)
    test_batch = next(iter(test_loader))
    print(f"Batch keys: {list(test_batch.keys())}")

    if 'window' in test_batch and 'x_t' not in test_batch:
        print("[OK] PASS: Test batch has single-sample format")
        print("[OK] PASS: No paired data in test batches!")
    else:
        print("[FAIL] FAIL: Test batch should be single-sample only!")

    return 'window' in test_batch and 'x_t' not in test_batch

if __name__ == '__main__':
    print("\n" + "#"*70)
    print("# Test Mode Verification Script")
    print("# Purpose: Ensure no data leakage in test evaluation")
    print("#"*70)

    test1_pass = verify_dataset_mode()
    test2_pass = verify_batch_format()

    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)

    if test1_pass and test2_pass:
        print("\n*** ALL VERIFICATIONS PASSED! ***")
        print("\n[OK] Dataset mode enforcement: WORKING")
        print("[OK] Batch format handling: CORRECT")
        print("[OK] Data leakage prevention: ACTIVE")
        print("\n*** Your code is SAFE for real experiments! ***")
    else:
        print("\n[WARNING]  VERIFICATION FAILED! [WARNING]")
        print("\n[FAIL] Some tests did not pass")
        print("[FAIL] DO NOT run experiments until issues are fixed")
        print("\n[DOC] Please review: TRAIN_TEST_MODE_DIFFERENCES.md")

    print("\n" + "="*70)
