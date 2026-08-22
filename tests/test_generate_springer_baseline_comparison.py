import json
import tempfile
import unittest
from pathlib import Path

from scripts.generate_springer_baseline_comparison import (
    BASELINE_ORDER,
    BASELINE_REVISION,
    RATIOS,
    generate,
    validate_baseline_summary,
)
from scripts.generate_springer_results import (
    MODEL_ORDER,
    PIPELINE_REVISION,
    RATIOS as PRIMARY_RATIOS,
)


def _run(model_id, ratio, repeat, revision):
    mae = 0.012 + 0.001 * (1.0 - ratio) + 0.0001 * repeat
    return {
        "run_id": f"{model_id}_{ratio}_{repeat}",
        "pipeline_revision": revision,
        "protocol": "fixed_split",
        "model_id": model_id,
        "supervision_ratio": ratio,
        "split_seed": 42,
        "mask_seed": (1929, 3262, 1007)[repeat],
        "training_seed": (929, 2262, 7)[repeat],
        "smoke": False,
        "test_mae": mae,
        "test_rmse": mae * 1.5,
        "test_r2": 0.93 - mae,
        "monotonicity_violation_rate_percent": 1.0,
        "mean_cumulative_upward_excess_per_battery": 0.01,
        "mean_absolute_second_difference": 0.001,
    }


def _primary_summary():
    return {
        "pipeline_revision": PIPELINE_REVISION,
        "runs": [
            _run(model_id, ratio, repeat, PIPELINE_REVISION)
            for model_id in MODEL_ORDER
            for ratio in PRIMARY_RATIOS
            for repeat in range(3)
        ],
    }


def _baseline_summary():
    return {
        "pipeline_revision": BASELINE_REVISION,
        "protocol": "fixed_split",
        "runs": [
            _run(model_id, ratio, repeat, BASELINE_REVISION)
            for model_id in BASELINE_ORDER
            for ratio in RATIOS
            for repeat in range(3)
        ],
    }


class BaselineComparisonGenerationTests(unittest.TestCase):
    def test_complete_archives_generate_three_tables_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = root / "primary.json"
            baseline = root / "baseline.json"
            primary.write_text(json.dumps(_primary_summary()), encoding="utf-8")
            baseline.write_text(json.dumps(_baseline_summary()), encoding="utf-8")
            outputs = generate(primary, baseline, root / "generated")
            self.assertEqual(len(outputs), 4)
            full = (root / "generated" / "table_all_models_full_supervision.tex").read_text(
                encoding="utf-8"
            )
            endpoint = (
                root / "generated" / "table_all_models_endpoint_budget.tex"
            ).read_text(encoding="utf-8")
            trajectory = (
                root / "generated" / "table_all_models_trajectory_r0p3.tex"
            ).read_text(encoding="utf-8")
            self.assertIn("XGBoost", full)
            self.assertIn("PI--MSCL", full)
            self.assertIn("R-squared", full)
            self.assertNotIn("textsuperscript", full)
            self.assertIn(r"\multicolumn{2}{c}{MAE (\%)}", endpoint)
            self.assertIn("Change (pp)", endpoint)
            self.assertIn("Mono. viol.", trajectory)
            self.assertIn("Roughness (pp)", trajectory)
            all_tables = full + endpoint + trajectory
            self.assertIn(r"\textpm{}", all_tables)
            self.assertNotIn("±", all_tables)
            self.assertNotIn("Â", all_tables)
            self.assertNotIn(r"\small", all_tables)

    def test_smoke_run_does_not_replace_missing_formal_run(self):
        summary = _baseline_summary()
        removed = summary["runs"].pop()
        summary["runs"].append({**removed, "run_id": removed["run_id"] + "_smoke", "smoke": True})
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_baseline_summary(summary)

    def test_cross_archive_seed_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary_data = _primary_summary()
            baseline_data = _baseline_summary()
            baseline_data["runs"][0]["mask_seed"] = 9999
            primary = root / "primary.json"
            baseline = root / "baseline.json"
            primary.write_text(json.dumps(primary_data), encoding="utf-8")
            baseline.write_text(json.dumps(baseline_data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not match"):
                generate(primary, baseline, root / "generated")


if __name__ == "__main__":
    unittest.main()
