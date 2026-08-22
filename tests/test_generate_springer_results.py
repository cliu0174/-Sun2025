import json
import tempfile
import unittest
from pathlib import Path

from scripts.generate_springer_results import (
    MODEL_ORDER,
    PIPELINE_REVISION,
    RATIOS,
    generate,
    paired_factorial_effects,
    validate_summary,
)


class SpringerResultGenerationTests(unittest.TestCase):
    @staticmethod
    def _summary(repeats=3):
        runs = []
        model_offsets = {
            "single_no_pi": 0.0040,
            "single_pi": 0.0030,
            "multi_no_pi": 0.0025,
            "multi_pi": 0.0010,
        }
        for model_id in MODEL_ORDER:
            for ratio in RATIOS:
                for repeat in range(repeats):
                    base = 0.010 + (1.0 - ratio) * 0.004 + repeat * 0.0002
                    mae = base + model_offsets[model_id]
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
                            "smoke": False,
                            "test_mae": mae,
                            "test_rmse": mae * 1.4,
                            "test_r2": 0.95 - mae,
                            "monotonicity_violation_rate_percent": mae * 10,
                            "mean_cumulative_upward_excess_per_battery": mae / 10,
                            "mean_absolute_second_difference": mae / 20,
                        }
                    )
        return {
            "pipeline_revision": PIPELINE_REVISION,
            "runs": runs,
            "groups": {"complete": {}},
        }

    def test_complete_paired_matrix_generates_all_assets(self):
        summary = self._summary()
        runs = validate_summary(summary, expected_repeats=3)
        effects = paired_factorial_effects(runs, 0.3)
        self.assertGreater(effects["architecture"][0], 0)
        self.assertGreater(effects["physics"][0], 0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            outputs = generate(summary_path, root / "generated", expected_repeats=3)
            self.assertEqual(len(outputs), 6)
            for path in outputs:
                self.assertTrue(path.exists())
            table = (root / "generated" / "table_label_budget.tex").read_text(
                encoding="utf-8"
            )
            self.assertIn("PI--MSCL", table)
            self.assertIn("r = 0.3", table)
            self.assertIn(r"\textpm{}", table)
            self.assertNotIn("textsuperscript", table)
            self.assertNotIn("±", table)
            self.assertNotIn("Â", table)
            self.assertNotIn(r"\small", table)
            r2_table = (root / "generated" / "table_label_budget_r2.tex").read_text(
                encoding="utf-8"
            )
            self.assertIn("R-squared", r2_table)
            self.assertIn("0.3", r2_table)
            for table_path in (root / "generated").glob("table_*.tex"):
                self.assertNotIn(r"\small", table_path.read_text(encoding="utf-8"))
            manifest = json.loads(
                (root / "generated" / "results_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["formal_run_count"], 48)

    def test_incomplete_matrix_is_rejected(self):
        summary = self._summary()
        summary["runs"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_summary(summary, expected_repeats=3)

    def test_smoke_run_cannot_replace_a_missing_formal_repeat(self):
        summary = self._summary()
        removed = summary["runs"].pop()
        summary["runs"].append(
            {**removed, "run_id": removed["run_id"] + "_smoke", "smoke": True}
        )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_summary(summary, expected_repeats=3)

    def test_unpaired_seed_matrix_is_rejected(self):
        summary = self._summary()
        target = next(
            run
            for run in summary["runs"]
            if run["model_id"] == "multi_pi" and run["supervision_ratio"] == 0.3
        )
        target["mask_seed"] = 9999
        with self.assertRaisesRegex(ValueError, "unpaired seeds"):
            validate_summary(summary, expected_repeats=3)


if __name__ == "__main__":
    unittest.main()
