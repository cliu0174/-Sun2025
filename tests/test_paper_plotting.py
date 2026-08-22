import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.paper_plot_style import apply_paper_style
from scripts.plot_paper_main_results import (
    MODEL_ORDER,
    PIPELINE_REVISION,
    _effect_series,
    create_ablation_figures,
    create_performance_figures,
    load_summary,
)


def _summary(n=3, smoke=False):
    groups = {}
    runs = []
    for model_index, model_id in enumerate(MODEL_ORDER):
        for ratio in (1.0, 0.7, 0.5, 0.3):
            key = f"{model_id}/r{ratio:.1f}".replace(".", "p")
            groups[key] = {
                "model_id": model_id,
                "model_label": model_id,
                "supervision_ratio": ratio,
                "n": n,
                "test_mae_mean": 0.01 + model_index * 0.0005 + (1 - ratio) * 0.001,
                "test_mae_std": 0.0002,
                "test_rmse_mean": 0.015 + model_index * 0.0006 + (1 - ratio) * 0.001,
                "test_rmse_std": 0.0003,
                "test_r2_mean": 0.92,
                "test_r2_std": 0.01,
                "monotonicity_violation_rate_percent_mean": 1.0,
                "monotonicity_violation_rate_percent_std": 0.1,
                "mean_cumulative_upward_excess_per_battery_mean": 0.01,
                "mean_cumulative_upward_excess_per_battery_std": 0.001,
                "mean_absolute_second_difference_mean": 0.001,
                "mean_absolute_second_difference_std": 0.0001,
            }
            for repeat in range(n):
                mae = 0.01 + model_index * 0.0005 + (1 - ratio) * 0.001
                runs.append(
                    {
                        "run_id": f"{model_id}_{ratio}_{repeat}",
                        "pipeline_revision": PIPELINE_REVISION,
                        "protocol": "fixed_split",
                        "model_id": model_id,
                        "supervision_ratio": ratio,
                        "split_seed": 42,
                        "mask_seed": 100 + repeat,
                        "training_seed": 200 + repeat,
                        "smoke": smoke,
                        "test_mae": mae,
                        "test_rmse": mae * 1.4,
                        "test_r2": 0.95 - mae,
                        "monotonicity_violation_rate_percent": mae * 10,
                        "mean_cumulative_upward_excess_per_battery": mae / 10,
                        "mean_absolute_second_difference": mae / 20,
                    }
                )
    return {
        "smoke": smoke,
        "pipeline_revision": PIPELINE_REVISION,
        "groups": groups,
        "runs": runs,
    }


class PaperPlottingTests(unittest.TestCase):
    def test_ablation_effects_use_paired_run_differences(self):
        summary = _summary()
        means, stds = _effect_series(
            summary, "single_no_pi", "multi_no_pi", (1.0, 0.3)
        )
        self.assertEqual(means.shape, (2,))
        self.assertEqual(stds.shape, (2,))
        self.assertTrue((means < 0).all())

    def test_smoke_and_single_repeat_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps(_summary(smoke=True)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "smoke-test"):
                load_summary(path)
            path.write_text(json.dumps(_summary(n=1)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "complete paired matrix"):
                load_summary(path)

    def test_unpaired_run_level_seeds_are_rejected(self):
        summary = _summary()
        target = next(run for run in summary["runs"] if run["model_id"] == "multi_pi")
        target["mask_seed"] = 9999
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unpaired seeds"):
                load_summary(path)

    def test_group_and_individual_figures_export_at_600_dpi(self):
        apply_paper_style()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            saved = create_performance_figures(_summary(), output)
            saved += create_ablation_figures(_summary(), output)
            self.assertEqual(len(saved), 24)
            pngs = sorted(output.glob("*.png"))
            tiffs = sorted(output.glob("*.tiff"))
            self.assertEqual(len(pngs), 6)
            self.assertEqual(len(tiffs), 6)
            for path in pngs + tiffs:
                with Image.open(path) as image:
                    dpi = image.info.get("dpi", (0, 0))
                    self.assertGreaterEqual(dpi[0], 599.0)
                    self.assertGreaterEqual(dpi[1], 599.0)
                    self.assertGreater(image.width, 2000)
                    self.assertGreater(image.height, 1500)


if __name__ == "__main__":
    unittest.main()
