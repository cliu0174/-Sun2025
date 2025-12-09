# Battery SOH Prediction System - Architecture Visualization Prompt

## System Overview
Please create a detailed architecture diagram for a **Physics-Informed Deep Learning System for Battery State of Health (SOH) Prediction** with the following components:

---

## 1. Data Flow Architecture

### Input Data
- **Source**: HUST Battery Dataset (77 batteries)
- **Raw Features**: 16-dimensional time-series data per charging cycle
  - Voltage, Current, Temperature, Capacity, etc.
- **Windowing**: Sliding window of size 40 time steps
- **Output**: Window shape: (batch_size, 40, 16)

### Data Split
- **Training**: 60% of batteries (46 batteries)
- **Validation**: 20% of batteries (15 batteries)
- **Testing**: 20% of batteries (16 batteries)
- **Cross-Battery Setting**: No battery appears in multiple sets

### Triplet Sampling (Training/Validation only)
- **Purpose**: Enable second-order physics constraints
- **Input**: Windowed sequences from same battery
- **Output**: Three consecutive time points (t, t+k, t+2k) where k=1
- **Format**:
  - x_1 (window at time t)
  - x_2 (window at time t+1)
  - x_3 (window at time t+2)
  - Each: shape (batch_size, 40, 16)
- **Note**: Testing uses single-sample mode (no triplets)

---

## 2. Neural Network Architecture (CNN-LSTM Hybrid)

### Layer-by-Layer Structure

**Input Layer**
- Shape: (batch_size, 40, 16)
- 40 time steps, 16 features per step

**↓ Reshape for CNN**
- Permute to: (batch_size, 16, 40)
- 16 channels (features), 40 sequence length

**CNN Block 1**
- Conv1D: 16 → 256 channels, kernel_size=7, padding=3
- BatchNorm1D(256)
- ReLU activation
- MaxPool1D: pool_size=2, stride=2
- Output: (batch_size, 256, 20)

**CNN Block 2**
- Conv1D: 256 → 128 channels, kernel_size=7, padding=3
- BatchNorm1D(128)
- ReLU activation
- MaxPool1D: pool_size=2, stride=2
- Output: (batch_size, 128, 10)

**↓ Reshape for LSTM**
- Permute to: (batch_size, 10, 128)
- 10 time steps, 128 features per step

**LSTM Block**
- 2-layer LSTM
- Hidden size: 64
- Bidirectional: No
- Dropout: 0.4 (between layers)
- batch_first=True
- Output: (batch_size, 10, 64) → take last time step → (batch_size, 64)

**Fully Connected Layers**
- FC1: Linear(64 → 64)
- ReLU activation
- Dropout(0.4)
- FC2: Linear(64 → 1)
- Sigmoid activation (output in [0,1])

**Output**
- Shape: (batch_size, 1)
- SOH prediction (0-100% normalized to 0-1)

---

## 3. Triplet Physics-Informed Loss Function

### Loss Components (visualize as separate branches that merge)

**Branch 1: Base MSE Loss**
- Compute MSE for all three predictions:
  - MSE(pred_1, true_1)
  - MSE(pred_2, true_2)
  - MSE(pred_3, true_3)
- Average: base_loss = (MSE_1 + MSE_2 + MSE_3) / 3
- Weight: 1.0

**Branch 2: Monotonicity Constraint (1st Order)**
- Check two transitions:
  - diff_1_2 = pred_2 - pred_1 (should be ≤ 0)
  - diff_2_3 = pred_3 - pred_2 (should be ≤ 0)
- Violations:
  - violation_1_2 = ReLU(diff_1_2 - tolerance)
  - violation_2_3 = ReLU(diff_2_3 - tolerance)
- Loss: (violation_1_2² + violation_2_3²) × mask
- Weight: 0.5
- **Physical Meaning**: Enforce SOH decreases monotonically

**Branch 3: Curvature Constraint (2nd Order) - KEY INNOVATION**
- Discrete Laplacian (second derivative):
  - curvature = pred_3 - 2×pred_2 + pred_1
- Loss: curvature² × mask
- Weight: 1000 (adjustable)
- **Physical Meaning**: Eliminate sawtooth/zigzag patterns, enforce smooth decline

**Branch 4: Boundary Constraint**
- Penalize predictions outside [0, 1]:
  - boundary_1 = ReLU(-pred_1) + ReLU(pred_1 - 1)
  - boundary_2 = ReLU(-pred_2) + ReLU(pred_2 - 1)
  - boundary_3 = ReLU(-pred_3) + ReLU(pred_3 - 1)
- Loss: (boundary_1 + boundary_2 + boundary_3) / 3
- Weight: 0.0 (disabled, Sigmoid already bounds output)

**Mask Mechanism (Split Constraint)**
- Early cycles (cycle < 300): mask = 0 (no physics constraints)
  - Reason: Early capacity may increase (formation phase)
- Late cycles (cycle ≥ 300): mask = 1 (full physics constraints)
  - Reason: Clear degradation phase, physics rules apply
- Visualization: Show mask as a switch/gate

**Total Loss**
```
Total = 1.0 × base_loss
      + 0.5 × monotonic_loss
      + 1000 × curvature_loss
      + 0.0 × boundary_loss
```

---

## 4. Training Process Visualization

### Forward Pass (Triplet Mode)
1. Sample triplet (x_1, x_2, x_3) from same battery
2. Three parallel forward passes through CNN-LSTM:
   - pred_1 = Model(x_1)
   - pred_2 = Model(x_2)
   - pred_3 = Model(x_3)
3. Compute TripletPhysicsLoss(pred_1, pred_2, pred_3, true_1, true_2, true_3, cycle_indices)
4. Backward propagation
5. Optimizer step (Adam, lr=0.0005)

### Validation (Triplet Mode)
- Same as training, but no gradient updates
- Used for early stopping and hyperparameter tuning

### Testing (Single-Sample Mode)
- **Different from training!**
- Input: Single window (x)
- Forward: pred = Model(x)
- Loss: Simple MSE(pred, true)
- Metrics: MAE, RMSE, R²

---

## 5. Key Innovation Highlights (emphasize in diagram)

### From Pairwise to Triplet Sampling
**Before (Pairwise)**:
- Samples: (t, t+k) pairs
- Constraint: 1st order (velocity) → smoothness = (pred_next - pred_t)²
- **Problem**: Cannot eliminate zigzag patterns ↓↑↓↑

**After (Triplet)**:
- Samples: (t, t+k, t+2k) triplets
- Constraints:
  - 1st order (velocity/monotonicity)
  - 2nd order (acceleration/curvature)
- **Solution**: Forces smooth monotonic decline ↓↓↓↓

### Physical Interpretation
- **1st Order (Monotonicity)**: "Don't go up" (velocity constraint)
- **2nd Order (Curvature)**: "Don't change direction" (acceleration constraint)
- Together: Guarantee smooth, physically plausible degradation curves

---

## 6. Visualization Style Suggestions

### Diagram Types
1. **Main Architecture Flowchart**: Data → Triplet Sampling → CNN-LSTM → Loss → Backprop
2. **CNN-LSTM Detailed Structure**: Show tensor shapes at each layer
3. **Loss Function Tree**: Four branches (MSE, Mono, Curv, Bound) merging into total loss
4. **Triplet Sampling Illustration**: Visual of (t, t+1, t+2) consecutive windows
5. **Physics Constraint Comparison**: Side-by-side Pairwise vs Triplet

### Color Coding
- **Data Flow**: Blue
- **Model Layers**: Green
- **Loss Components**: Red/Orange
- **Physics Constraints**: Purple
- **Mask Mechanism**: Yellow

### Annotations
- Mark tensor shapes at key points: (batch, seq, features)
- Show key hyperparameters: kernel_size=7, hidden=64, etc.
- Highlight innovation: "Second-Order Curvature Constraint ★"

---

## 7. Technical Specifications Summary

| Component | Value |
|-----------|-------|
| **Model Type** | CNN-LSTM Hybrid |
| **Input Dimension** | (batch, 40, 16) |
| **CNN Layers** | 2 blocks, [256, 128] channels |
| **LSTM Layers** | 2 layers, hidden=64 |
| **Total Parameters** | 346,369 |
| **Batch Size** | 256 |
| **Training Mode** | Triplet sampling |
| **Loss Function** | TripletPhysicsLoss |
| **Optimizer** | Adam (lr=0.0005) |
| **Epochs** | 200 (with early stopping) |
| **Dataset** | HUST 77 batteries |
| **Key Innovation** | 2nd-order curvature constraint |

---

## 8. Expected Output Quality

### What to Emphasize
1. **Three-stage pipeline**: Data → Model → Loss
2. **Triplet architecture**: Show how 3 samples flow through shared CNN-LSTM
3. **Physics-informed loss**: Visualize how 4 loss components combine
4. **Mask mechanism**: Show split constraint (cycle < 300 vs ≥ 300)

### Make it Clear
- This is NOT a standard deep learning model
- This is a **Physics-Informed Neural Network (PINN)**
- Key innovation: Using **second-order smoothness** to eliminate prediction jitter
- Triplet sampling enables constraints that pairwise cannot achieve

---

## Additional Context

**Problem Domain**: Battery degradation follows physical laws (monotonic decrease, smooth curve)

**Challenge**: Standard neural networks can produce physically implausible predictions (zigzag patterns, upward trends)

**Solution**: Inject physics knowledge as soft constraints in the loss function

**Result**: Predictions that are both accurate (low MSE) AND physically plausible (smooth, monotonic)

---

**End of Prompt** - Please create a comprehensive architecture diagram based on this specification.
