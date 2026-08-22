import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_loaders.data_loader_hust import load_single_hust_battery


FEATURE_COLUMNS = [
    "voltage mean",
    "voltage std",
    "voltage kurtosis",
    "voltage skewness",
    "CC Q",
    "CC charge time",
    "voltage slope",
    "voltage entropy",
    "current mean",
    "current std",
    "current kurtosis",
    "current skewness",
    "CV Q",
    "CV charge time",
    "current slope",
    "current entropy",
]


class FeatureStandardizationScopeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.tmp.name) / "cell.csv"
        rows = 8
        frame = {
            name: np.arange(rows, dtype=float) + 10.0 * j
            for j, name in enumerate(FEATURE_COLUMNS)
        }
        frame["capacity"] = np.linspace(1.10, 0.90, rows)
        pd.DataFrame(frame).to_csv(self.csv_path, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_cross_battery_mode_returns_unscaled_features(self):
        data = load_single_hust_battery(
            self.csv_path,
            train_ratio=1.0,
            standardize_features=False,
        )
        self.assertFalse(data["features_standardized"])
        self.assertIsNone(data["scaler"])
        np.testing.assert_allclose(
            data["train_features"][:, 0], np.arange(8, dtype=float)
        )

    def test_legacy_single_battery_mode_remains_standardized(self):
        data = load_single_hust_battery(
            self.csv_path,
            train_ratio=0.75,
            standardize_features=True,
        )
        self.assertTrue(data["features_standardized"])
        self.assertIsNotNone(data["scaler"])
        np.testing.assert_allclose(
            data["train_features"].mean(axis=0), 0.0, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
