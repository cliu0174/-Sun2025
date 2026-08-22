"""Trajectory-level metrics for battery SOH predictions.

The functions in this module are intentionally independent from training.  They
operate on saved predictions and explicit battery/cycle metadata so that paper
tables can be regenerated without rerunning a model.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np


def _as_1d(values: Iterable[Any], name: str) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def compute_trajectory_metrics(
    predictions: Iterable[float],
    targets: Iterable[float],
    battery_ids: Iterable[Any],
    cycle_indices: Iterable[int],
    *,
    tolerance: float = 0.005,
    min_cycle: int = 300,
) -> Dict[str, Any]:
    """Compute error and physical-consistency metrics by battery trajectory.

    A monotonicity violation is an adjacent prediction increase larger than
    ``tolerance``.  Consistent with the training loss, a pair is eligible only
    when its earlier cycle index is at least ``min_cycle``.

    Returned rates are percentages.  SOH differences and errors remain in the
    same normalized SOH unit as the inputs.
    """

    preds = _as_1d(predictions, "predictions").astype(float)
    tgts = _as_1d(targets, "targets").astype(float)
    bids = _as_1d(battery_ids, "battery_ids")
    cycles = _as_1d(cycle_indices, "cycle_indices").astype(int)

    n = len(preds)
    if not (len(tgts) == len(bids) == len(cycles) == n):
        raise ValueError(
            "predictions, targets, battery_ids, and cycle_indices must have "
            f"the same length; got {n}, {len(tgts)}, {len(bids)}, {len(cycles)}"
        )
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if min_cycle < 0:
        raise ValueError("min_cycle must be non-negative")
    if not np.isfinite(preds).all() or not np.isfinite(tgts).all():
        raise ValueError("predictions and targets must contain only finite values")

    per_battery: Dict[str, Dict[str, Any]] = {}
    all_excesses = []
    all_abs_first_diffs = []
    all_abs_second_diffs = []
    all_abs_rate_variations = []
    total_pairs = 0
    total_violations = 0

    for raw_bid in sorted(np.unique(bids), key=lambda value: str(value)):
        idx = np.where(bids == raw_bid)[0]
        order = np.argsort(cycles[idx], kind="stable")
        idx = idx[order]

        p = preds[idx]
        y = tgts[idx]
        c = cycles[idx]
        pred_diff = np.diff(p)
        eligible = c[:-1] >= min_cycle
        eligible_diff = pred_diff[eligible]
        excess = np.maximum(eligible_diff - tolerance, 0.0)
        n_pairs = int(eligible_diff.size)
        n_viol = int(np.count_nonzero(excess > 0))

        abs_first = np.abs(eligible_diff)
        second_diff = np.diff(p, n=2)
        second_eligible = c[:-2] >= min_cycle
        abs_second = np.abs(second_diff[second_eligible])
        cycle_steps = np.diff(c)
        rates = np.diff(p) / np.maximum(cycle_steps, 1)
        rate_variation = np.abs(np.diff(rates))
        rate_eligible = (c[:-2] >= min_cycle) & (cycle_steps[:-1] > 0) & (cycle_steps[1:] > 0)
        abs_rate_variation = rate_variation[rate_eligible]

        all_excesses.extend(excess.tolist())
        all_abs_first_diffs.extend(abs_first.tolist())
        all_abs_second_diffs.extend(abs_second.tolist())
        all_abs_rate_variations.extend(abs_rate_variation.tolist())
        total_pairs += n_pairs
        total_violations += n_viol

        bid = str(raw_bid)
        per_battery[bid] = {
            "n_samples": int(len(idx)),
            "first_cycle": int(c[0]),
            "last_cycle": int(c[-1]),
            "mae": float(np.mean(np.abs(p - y))),
            "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "n_eligible_pairs": n_pairs,
            "n_violations": n_viol,
            "monotonicity_violation_rate_percent": (
                float(100.0 * n_viol / n_pairs) if n_pairs else 0.0
            ),
            "cumulative_upward_excess": float(np.sum(excess)),
            "mean_absolute_first_difference": (
                float(np.mean(abs_first)) if abs_first.size else 0.0
            ),
            "mean_absolute_second_difference": (
                float(np.mean(abs_second)) if abs_second.size else 0.0
            ),
            "mean_absolute_rate_variation": (
                float(np.mean(abs_rate_variation)) if abs_rate_variation.size else 0.0
            ),
        }

    excess_array = np.asarray(all_excesses, dtype=float)
    first_array = np.asarray(all_abs_first_diffs, dtype=float)
    second_array = np.asarray(all_abs_second_diffs, dtype=float)
    rate_variation_array = np.asarray(all_abs_rate_variations, dtype=float)
    cumulative_values = [
        values["cumulative_upward_excess"] for values in per_battery.values()
    ]

    return {
        "tolerance": float(tolerance),
        "min_cycle": int(min_cycle),
        "n_samples": int(n),
        "n_batteries": int(len(per_battery)),
        "n_eligible_pairs": int(total_pairs),
        "n_violations": int(total_violations),
        "mae": float(np.mean(np.abs(preds - tgts))),
        "rmse": float(np.sqrt(np.mean((preds - tgts) ** 2))),
        "monotonicity_violation_rate_percent": (
            float(100.0 * total_violations / total_pairs) if total_pairs else 0.0
        ),
        "mean_upward_excess": (
            float(np.mean(excess_array[excess_array > 0]))
            if np.any(excess_array > 0)
            else 0.0
        ),
        "mean_cumulative_upward_excess_per_battery": (
            float(np.mean(cumulative_values)) if cumulative_values else 0.0
        ),
        "mean_absolute_first_difference": (
            float(np.mean(first_array)) if first_array.size else 0.0
        ),
        "mean_absolute_second_difference": (
            float(np.mean(second_array)) if second_array.size else 0.0
        ),
        "mean_absolute_rate_variation": (
            float(np.mean(rate_variation_array)) if rate_variation_array.size else 0.0
        ),
        "per_battery": per_battery,
    }


def select_median_error_battery(metrics: Dict[str, Any]) -> str:
    """Select the battery whose MAE is closest to the across-battery median.

    This deterministic rule prevents selecting a visually favourable trajectory
    after inspecting the curves.  Lexicographic battery ID resolves exact ties.
    """

    per_battery = metrics.get("per_battery", {})
    if not per_battery:
        raise ValueError("metrics does not contain per-battery results")
    median = float(np.median([entry["mae"] for entry in per_battery.values()]))
    return min(
        per_battery,
        key=lambda bid: (abs(per_battery[bid]["mae"] - median), str(bid)),
    )
