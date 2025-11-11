"""
Data preprocessing module for BPINN SOH estimation.
Loads and preprocesses IC features from NASA battery datasets.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


class BatteryDataset(Dataset):
    """PyTorch Dataset for battery IC features and SOH."""

    def __init__(self, features, soh, p_ic):
        """
        Args:
            features: numpy array of shape (n_samples, n_features)
            soh: numpy array of shape (n_samples,)
            p_ic: numpy array of shape (n_samples,) - Peak IC values
        """
        self.features = torch.FloatTensor(features)
        self.soh = torch.FloatTensor(soh)
        self.p_ic = torch.FloatTensor(p_ic)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.soh[idx], self.p_ic[idx]


def load_battery_data(file_paths, train_ratio=0.6):
    """
    Load battery data from CSV files.

    IMPORTANT: Following the paper's methodology, each battery's data is split
    separately into train/test sets (60%/40%), then combined. This prevents
    data leakage and maintains independence between batteries.

    Args:
        file_paths: list of paths to CSV files
        train_ratio: ratio of data to use for training (default: 0.6 as per paper)

    Returns:
        Dictionary containing train and test data
    """
    train_data_list = []
    test_data_list = []

    # Split each battery's data separately (as per paper methodology)
    for file_path in file_paths:
        df = pd.read_csv(file_path)

        # Split this battery's data
        n_train = int(len(df) * train_ratio)
        train_df = df.iloc[:n_train]
        test_df = df.iloc[n_train:]

        train_data_list.append(train_df)
        test_data_list.append(test_df)

    # Combine all batteries' train and test data separately
    train_df = pd.concat(train_data_list, ignore_index=True)
    test_df = pd.concat(test_data_list, ignore_index=True)

    # Select the 6 key features from IC curves as mentioned in the paper
    # Based on the paper: y_h, V_h, k_l, k_r, t_G, Q_G
    feature_columns = ['y_h', 'V_h', 'k_l', 'k_r', 't_G', 'Q_G']

    train_features = train_df[feature_columns].values
    train_soh = train_df['SOH'].values
    train_p_ic = train_df['y_h'].values  # Peak IC (P-IC) is y_h

    test_features = test_df[feature_columns].values
    test_soh = test_df['SOH'].values
    test_p_ic = test_df['y_h'].values

    # Normalize features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    return {
        'train_features': train_features,
        'train_soh': train_soh,
        'train_p_ic': train_p_ic,
        'test_features': test_features,
        'test_soh': test_soh,
        'test_p_ic': test_p_ic,
        'scaler': scaler,
        'feature_columns': feature_columns
    }


def create_data_loaders(data_dict, batch_size=32):
    """
    Create PyTorch DataLoaders for training and testing.

    Args:
        data_dict: Dictionary returned by load_battery_data
        batch_size: Batch size for training

    Returns:
        train_loader, test_loader
    """
    train_dataset = BatteryDataset(
        data_dict['train_features'],
        data_dict['train_soh'],
        data_dict['train_p_ic']
    )

    test_dataset = BatteryDataset(
        data_dict['test_features'],
        data_dict['test_soh'],
        data_dict['test_p_ic']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    return train_loader, test_loader


def compute_physical_constraint(model, features, p_ic, epsilon=1e-6):
    """
    Compute the physical constraint: ∂SOH/∂P-IC > 0

    This function computes the gradient of SOH prediction with respect to P-IC (y_h).
    According to the paper, there should be a monotonic positive relationship.

    Args:
        model: BPINN model
        features: Input features tensor (batch_size, n_features)
        p_ic: Peak IC values tensor (batch_size,)
        epsilon: Small value for numerical stability

    Returns:
        Physical constraint loss
    """
    features_with_grad = features.clone().detach().requires_grad_(True)

    # Forward pass
    soh_pred = model(features_with_grad)

    # Compute gradient of SOH with respect to P-IC (first feature is y_h)
    grad_soh_wrt_pic = torch.autograd.grad(
        outputs=soh_pred,
        inputs=features_with_grad,
        grad_outputs=torch.ones_like(soh_pred),
        create_graph=True,
        retain_graph=True
    )[0][:, 0]  # Gradient w.r.t. first feature (y_h which is P-IC)

    # Physical constraint: derivative should be positive
    # Using ReLU-like penalty for negative gradients
    constraint_violation = torch.relu(-grad_soh_wrt_pic)

    return constraint_violation.mean()


if __name__ == "__main__":
    # Test data loading
    file_paths = [
        '../data/B05_IC.csv',
        '../data/B06_IC.csv',
        '../data/B07_IC.csv'
    ]

    data = load_battery_data(file_paths)

    print("Data loaded successfully!")
    print(f"Training samples: {len(data['train_features'])}")
    print(f"Testing samples: {len(data['test_features'])}")
    print(f"Feature shape: {data['train_features'].shape}")
    print(f"Features: {data['feature_columns']}")
