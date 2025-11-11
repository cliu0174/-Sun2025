# BPINN for Battery SOH Estimation

This project implements the Battery Physics-Informed Neural Network (BPINN) method for estimating lithium-ion battery State of Health (SOH), based on the paper "A method for estimating lithium-ion battery state of health based on physics-informed machine learning" by Sun et al. (2025).

## Project Structure

```
.
├── data/                  # IC feature data
├── src/                   # Source code
│   ├── data_loader.py     # Data preprocessing
│   ├── model.py           # BPINN model definition
│   ├── train.py           # Training functions
│   └── utils.py           # Utility functions
├── results/               # Training results and figures
├── requirements.txt       # Dependencies
└── main.py                # Main training script
```

## Key Features

- IC curve feature extraction from battery data
- Physics-informed neural network with monotonic constraints
- Secondary optimization during testing phase
- Validation on NASA battery datasets (B05, B06, B07)

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python main.py
```

## Method Overview

The BPINN method incorporates physical constraints derived from the monotonic relationship between Peak IC (P-IC) and SOH into both the training and prediction phases:

1. **Feature Extraction**: Extract 6 key features from IC curves (y_h, V_h, k_l, k_r, t_G, Q_G)
2. **Physical Constraint**: Enforce monotonic relationship ∂SOH/∂P-IC > 0
3. **Model Training**: Train FNN with physically constrained loss function
4. **Secondary Training**: Online optimization during testing to improve accuracy

## References

Sun, G., Liu, Y., & Liu, X. (2025). A method for estimating lithium-ion battery state of health based on physics-informed machine learning. Journal of Power Sources, 627, 235767.
