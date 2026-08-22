import unittest

from evaluation.trajectory_metrics import (
    compute_trajectory_metrics,
    select_median_error_battery,
)


class TrajectoryMetricsTests(unittest.TestCase):
    def test_monotonic_sequence_has_no_violation(self):
        metrics = compute_trajectory_metrics(
            predictions=[1.00, 0.99, 0.98, 0.97],
            targets=[1.00, 0.99, 0.98, 0.97],
            battery_ids=["a"] * 4,
            cycle_indices=[300, 301, 302, 303],
            tolerance=0.005,
            min_cycle=300,
        )
        self.assertEqual(metrics["n_eligible_pairs"], 3)
        self.assertEqual(metrics["n_violations"], 0)
        self.assertEqual(metrics["monotonicity_violation_rate_percent"], 0.0)

    def test_tolerance_and_min_cycle_are_applied(self):
        metrics = compute_trajectory_metrics(
            predictions=[0.90, 0.93, 0.92, 0.94],
            targets=[0.90, 0.91, 0.92, 0.93],
            battery_ids=["a"] * 4,
            cycle_indices=[298, 299, 300, 301],
            tolerance=0.005,
            min_cycle=300,
        )
        self.assertEqual(metrics["n_eligible_pairs"], 1)
        self.assertEqual(metrics["n_violations"], 1)
        self.assertAlmostEqual(metrics["mean_upward_excess"], 0.015)

    def test_input_order_is_sorted_by_cycle_within_battery(self):
        metrics = compute_trajectory_metrics(
            predictions=[0.98, 1.00, 0.97, 0.99],
            targets=[0.98, 1.00, 0.97, 0.99],
            battery_ids=["a", "a", "a", "a"],
            cycle_indices=[302, 300, 303, 301],
            tolerance=0.0,
            min_cycle=300,
        )
        self.assertEqual(metrics["n_violations"], 0)

    def test_median_battery_selection_is_deterministic(self):
        metrics = compute_trajectory_metrics(
            predictions=[1.0, 0.9, 1.0, 0.8, 1.0, 0.7],
            targets=[1.0, 0.9, 1.0, 0.9, 1.0, 0.9],
            battery_ids=["a", "a", "b", "b", "c", "c"],
            cycle_indices=[300, 301, 300, 301, 300, 301],
            tolerance=0.0,
            min_cycle=300,
        )
        self.assertEqual(select_median_error_battery(metrics), "b")


if __name__ == "__main__":
    unittest.main()
