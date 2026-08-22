import unittest

import numpy as np

from train_cross_battery import create_dataloaders


def _synthetic_data():
    # Battery A contains only zeros and battery B only ones.  Any window with
    # both values necessarily crosses a battery boundary.
    features = np.vstack(
        [np.zeros((5, 2), dtype=np.float32), np.ones((5, 2), dtype=np.float32)]
    )
    targets = np.linspace(1.0, 0.9, 10, dtype=np.float32)
    battery_ids = np.asarray(["A"] * 5 + ["B"] * 5, dtype=object)
    mask = np.ones(10, dtype=bool)
    return {
        "train_features": features,
        "train_targets": targets,
        "train_battery_ids": battery_ids,
        "train_supervision_mask": mask,
        "val_features": features,
        "val_targets": targets,
        "val_battery_ids": battery_ids,
        "val_supervision_mask": mask,
        "test_features": features,
        "test_targets": targets,
        "test_battery_ids": battery_ids,
        "test_supervision_mask": mask,
        "train_batteries": ["A", "B"],
        "val_batteries": ["A", "B"],
        "test_batteries": ["A", "B"],
    }


class BatteryBoundaryWindowingTests(unittest.TestCase):
    def test_paper_mode_never_mixes_batteries_in_one_window(self):
        data = _synthetic_data()
        train_loader, _, test_loader, returned_ids = create_dataloaders(
            data,
            batch_size=4,
            window_size=3,
            use_physics=False,
            preserve_battery_boundaries=True,
        )
        self.assertIsNone(returned_ids)
        self.assertEqual(len(train_loader.dataset), 6)
        self.assertEqual(len(test_loader.dataset), 6)
        self.assertEqual(data["windowed_train_samples"], 6)
        self.assertEqual(data["windowed_train_labeled_samples"], 6)
        self.assertEqual(data["windowed_train_label_ratio"], 1.0)
        for item in train_loader.dataset:
            window = item["window"].numpy()
            self.assertEqual(np.unique(window).size, 1)
            expected_value = 0.0 if item["battery_id"] == "A" else 1.0
            self.assertTrue(np.all(window == expected_value))

    def test_legacy_mode_demonstrates_the_old_boundary_problem(self):
        train_loader, _, _, _ = create_dataloaders(
            _synthetic_data(),
            batch_size=4,
            window_size=3,
            use_physics=False,
            preserve_battery_boundaries=False,
        )
        mixed_windows = 0
        for window, _ in train_loader.dataset:
            if np.unique(window.numpy()).size > 1:
                mixed_windows += 1
        self.assertGreater(mixed_windows, 0)


if __name__ == "__main__":
    unittest.main()
