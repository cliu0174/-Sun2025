import json
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scripts.paper_plot_style import apply_paper_style
from scripts.plot_paper_trajectories import (
    MODEL_ORDER,
    create_trajectory_figures,
    load_factorial_runs,
    select_cross_model_median_seed_pair,
)


class PaperTrajectoryPlottingTests(unittest.TestCase):
    def _write_run_tree(
        self,
        root: Path,
        *,
        smoke_model: str | None = None,
        mask_seed: int = 17,
        training_seed: int = 7,
        run_mae_base: float = 0.01,
    ) -> None:
        battery_ids = np.asarray(["a"] * 4 + ["b"] * 4 + ["c"] * 4)
        cycles = np.asarray([300, 301, 302, 303] * 3)
        targets = np.asarray(
            [1.00, 0.99, 0.98, 0.97, 1.00, 0.98, 0.96, 0.94, 1.00, 0.97, 0.94, 0.91]
        )
        offsets = {"a": 0.001, "b": 0.010, "c": 0.030}
        for model_index, model_id in enumerate(MODEL_ORDER):
            run_dir = (
                root
                / model_id
                / "r0p3"
                / f"split42_mask{mask_seed}_train{training_seed}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            predictions = targets.copy()
            for battery_id, offset in offsets.items():
                predictions[battery_ids == battery_id] += offset + model_index * 0.001
            np.savez_compressed(
                run_dir / "predictions.npz",
                predictions=predictions,
                targets=targets,
                battery_ids=battery_ids,
                cycle_indices=cycles,
            )
            result = {
                "model_id": model_id,
                "supervision_ratio": 0.3,
                "split_seed": 42,
                "mask_seed": mask_seed,
                "training_seed": training_seed,
                "smoke": model_id == smoke_model,
                "test_mae": run_mae_base + model_index * 0.001,
            }
            (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")

    def test_group_and_child_figures_are_independent_600_dpi_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision_root = root / "revision"
            output_dir = root / "figures"
            self._write_run_tree(revision_root)
            runs = load_factorial_runs(
                revision_root,
                ratio=0.3,
                split_seed=42,
                mask_seed=17,
                training_seed=7,
            )
            apply_paper_style()
            saved, source = create_trajectory_figures(runs, output_dir)
            plt.close("all")
            self.assertEqual(source["selected_battery"], "b")
            self.assertEqual(len(saved), 12)
            for stem in (
                "fig4_trajectory_and_error_group",
                "fig4a_representative_trajectory",
                "fig4b_aligned_prediction_error",
            ):
                self.assertTrue((output_dir / f"{stem}.pdf").exists())
                self.assertTrue((output_dir / f"{stem}.svg").exists())
                self.assertTrue((output_dir / f"{stem}.tiff").exists())
                png = output_dir / f"{stem}.png"
                self.assertTrue(png.exists())
                with Image.open(png) as image:
                    dpi = image.info.get("dpi")
                    self.assertIsNotNone(dpi)
                    self.assertAlmostEqual(dpi[0], 600.0, delta=1.0)
                    self.assertAlmostEqual(dpi[1], 600.0, delta=1.0)
            self.assertTrue((output_dir / "fig4_source.json").exists())

    def test_seed_pair_is_selected_by_cross_model_median_mae(self):
        with tempfile.TemporaryDirectory() as tmp:
            revision_root = Path(tmp) / "revision"
            self._write_run_tree(
                revision_root, mask_seed=17, training_seed=7, run_mae_base=0.03
            )
            self._write_run_tree(
                revision_root, mask_seed=18, training_seed=8, run_mae_base=0.01
            )
            self._write_run_tree(
                revision_root, mask_seed=19, training_seed=9, run_mae_base=0.02
            )
            selected, runs, scores = select_cross_model_median_seed_pair(
                revision_root, ratio=0.3, split_seed=42
            )
            self.assertEqual(selected, (19, 9))
            self.assertEqual(len(scores), 3)
            self.assertEqual(runs["multi_pi"]["result"]["mask_seed"], 19)

    def test_smoke_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            revision_root = Path(tmp) / "revision"
            self._write_run_tree(revision_root, smoke_model="multi_pi")
            with self.assertRaisesRegex(ValueError, "smoke run"):
                load_factorial_runs(
                    revision_root,
                    ratio=0.3,
                    split_seed=42,
                    mask_seed=17,
                    training_seed=7,
                )


if __name__ == "__main__":
    unittest.main()
