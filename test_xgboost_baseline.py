"""
Test script for XGBoost baseline models.

Verifies that both XGBoost_Simple and XGBoost_Enhanced can be:
1. Created via model factory
2. Trained on sample data
3. Make predictions
"""

import numpy as np
import torch
from models.model_factory import ModelFactory

def test_xgboost_models():
    """Test both XGBoost baseline models."""

    print("="*70)
    print("Testing XGBoost Baseline Models")
    print("="*70)

    # Create synthetic data
    batch_size = 100
    window_size = 40
    input_size = 16

    # Training data
    X_train = np.random.randn(batch_size, window_size, input_size).astype(np.float32)
    y_train = np.random.rand(batch_size).astype(np.float32)  # SOH in [0, 1]

    # Validation data
    X_val = np.random.randn(20, window_size, input_size).astype(np.float32)
    y_val = np.random.rand(20).astype(np.float32)

    # Test data (as PyTorch tensor to verify compatibility)
    X_test = torch.randn(10, window_size, input_size)

    # ========================================================================
    # Test 1: XGBoost_Simple
    # ========================================================================
    print("\n" + "-"*70)
    print("Test 1: XGBoost_Simple")
    print("-"*70)

    try:
        # Create model via factory
        model_simple = ModelFactory.create_model(
            model_type='xgboost_simple',
            input_size=input_size
        )

        print(f"[OK] Model created: {model_simple.__class__.__name__}")
        print(f"  - Input size: {input_size}")
        print(f"  - Window size: {window_size}")
        print(f"  - Flattened feature dim: {model_simple.feature_dim}")

        # Train model
        print("\nTraining XGBoost_Simple...")
        model_simple.fit(X_train, y_train, X_val, y_val, verbose=False)
        print("[OK] Training completed")

        # Test inference
        print("\nTesting inference...")
        predictions = model_simple(X_test)
        print(f"[OK] Prediction shape: {predictions.shape}")
        print(f"  - Expected: ({len(X_test)}, 1)")
        print(f"  - Sample predictions: {predictions[:3].flatten().tolist()}")

        assert predictions.shape == (len(X_test), 1), "Incorrect output shape"
        print("\n[PASS] XGBoost_Simple test PASSED!")

    except Exception as e:
        print(f"\n[FAIL] XGBoost_Simple test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ========================================================================
    # Test 2: XGBoost_Enhanced
    # ========================================================================
    print("\n" + "-"*70)
    print("Test 2: XGBoost_Enhanced")
    print("-"*70)

    try:
        # Create model via factory
        model_enhanced = ModelFactory.create_model(
            model_type='xgboost_enhanced',
            input_size=input_size
        )

        print(f"[OK] Model created: {model_enhanced.__class__.__name__}")
        print(f"  - Input size: {input_size}")
        print(f"  - Window size: {window_size}")
        print(f"  - N lags: {model_enhanced.n_lags}")
        print(f"  - Rolling windows: {model_enhanced.rolling_windows}")
        print(f"  - Engineered feature dim: {model_enhanced.feature_dim}")

        # Train model
        print("\nTraining XGBoost_Enhanced...")
        model_enhanced.fit(X_train, y_train, X_val, y_val, verbose=False)
        print("[OK] Training completed")

        # Test inference
        print("\nTesting inference...")
        predictions = model_enhanced(X_test)
        print(f"[OK] Prediction shape: {predictions.shape}")
        print(f"  - Expected: ({len(X_test)}, 1)")
        print(f"  - Sample predictions: {predictions[:3].flatten().tolist()}")

        assert predictions.shape == (len(X_test), 1), "Incorrect output shape"
        print("\n[PASS] XGBoost_Enhanced test PASSED!")

    except Exception as e:
        print(f"\n[FAIL] XGBoost_Enhanced test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ========================================================================
    # Test 3: Compare feature dimensions
    # ========================================================================
    print("\n" + "-"*70)
    print("Test 3: Feature Dimension Comparison")
    print("-"*70)

    print(f"\nXGBoost_Simple:")
    print(f"  - Flattened features: {model_simple.feature_dim}")
    print(f"  - Formula: {window_size} x {input_size} = {window_size * input_size}")

    print(f"\nXGBoost_Enhanced:")
    print(f"  - Engineered features: {model_enhanced.feature_dim}")
    print(f"  - Breakdown:")
    print(f"    - Lag features ({model_enhanced.n_lags}): {model_enhanced.n_lags * input_size}")
    print(f"    - Rolling stats ({len(model_enhanced.rolling_windows)} windows x 5 stats): {len(model_enhanced.rolling_windows) * 5 * input_size}")
    print(f"    - Change rates (delta + accel): {2 * input_size}")
    print(f"    - Raw features: {input_size}")

    print(f"\n[OK] Feature engineering adds {model_enhanced.feature_dim - model_simple.feature_dim} more features!")

    print("\n" + "="*70)
    print("All XGBoost Baseline Tests PASSED!")
    print("="*70)

    return True


if __name__ == "__main__":
    success = test_xgboost_models()
    exit(0 if success else 1)
