import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import run_paper_main_results as runner


def _fake_training_result(*args, **kwargs):
    predictions = np.asarray([1.0, 0.99, 0.98, 0.97], dtype=float)
    targets = np.asarray([1.0, 0.985, 0.975, 0.965], dtype=float)
    results = {
        "predictions": predictions,
        "targets": targets,
        "battery_ids": np.asarray(["cell-a"] * 4),
        "cycle_indices": np.asarray([300, 301, 302, 303], dtype=np.int32),
        "test_mae": float(np.mean(np.abs(predictions - targets))),
        "test_rmse": float(np.sqrt(np.mean((predictions - targets) ** 2))),
        "test_mape": 0.5,
        "test_r2": 0.9,
        "best_val_mae": 0.01,
        "best_epoch": 1,
        "cycle_level_train_samples": 100,
        "cycle_level_train_labeled_samples": 30,
        "windowed_train_samples": 61,
        "windowed_train_labeled_samples": 18,
        "windowed_train_label_ratio": 18 / 61,
    }
    return object(), results, object()


class PaperRunnerArchivalTests(unittest.TestCase):
    def test_smoke_and_formal_runs_use_separate_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(
                runner, "train_cross_battery_model", side_effect=_fake_training_result
            ) as train:
                runner.run_one(
                    model_id="multi_pi",
                    ratio=0.3,
                    training_seed=7,
                    mask_seed=17,
                    split_seed=42,
                    protocol="fixed_split",
                    output_root=root,
                    device="cpu",
                    smoke=True,
                    force=False,
                )
                runner.run_one(
                    model_id="multi_pi",
                    ratio=0.3,
                    training_seed=7,
                    mask_seed=17,
                    split_seed=42,
                    protocol="fixed_split",
                    output_root=root,
                    device="cpu",
                    smoke=False,
                    force=False,
                )
            self.assertEqual(train.call_count, 2)
            base = root / "fixed_split" / runner.PIPELINE_REVISION / "multi_pi" / "r0p3"
            self.assertTrue((base / "split42_mask17_train7_smoke" / "result.json").exists())
            self.assertTrue((base / "split42_mask17_train7" / "result.json").exists())

    def test_staged_collection_reads_all_completed_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            revision_root = Path(tmp) / runner.PIPELINE_REVISION
            for model_id in ("single_no_pi", "multi_pi"):
                run_dir = revision_root / model_id / "r0p3" / "split42_mask17_train7"
                run_dir.mkdir(parents=True)
                record = {
                    "pipeline_revision": runner.PIPELINE_REVISION,
                    "model_id": model_id,
                    "supervision_ratio": 0.3,
                    "smoke": False,
                }
                (run_dir / "result.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )
                np.savez_compressed(run_dir / "predictions.npz", predictions=[0.9])

            records = runner.collect_completed_records(revision_root)
            self.assertEqual({row["model_id"] for row in records}, {"single_no_pi", "multi_pi"})


if __name__ == "__main__":
    unittest.main()
