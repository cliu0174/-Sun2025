"""Queue the remaining Chapter 4 neural experiments with checkpointed outputs.

Runs three independent task families serially on one GPU:
1. PI-MSCL across 100/50/30/20/10% label budgets;
2. the complete 2^3 multi-scale/mono/rate ablation at 10%;
3. the 6x6 lambda_mono/lambda_rate sensitivity scan at 10%.

Every task uses the fixed split and the single screening seed requested by the
author.  Existing completed task directories are verified then skipped, so the
queue can be restarted safely after interruption.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.trajectory_metrics import compute_trajectory_metrics, select_median_error_battery
from train_cross_battery import train_cross_battery_model


PIPELINE_REVISION = "v6_chapter4_remaining_single_seed"
SPLIT_SEED = 42
TRAINING_SEED = 2262
MASK_SEED = 3262
SUPERVISION_MODE = "trajectory_prefix_concentrated"
TOLERANCE = 0.005
MIN_CYCLE = 300
RATE_DEFAULT = 0.10
MONO_DEFAULT = 0.30

COMMON_OVERRIDE: dict[str, Any] = {
    "architecture": {"per_window_norm": False},
    "training": {
        "batch_size": 1024,
        "num_epochs": 200,
        "learning_rate": 0.0005,
        "scheduler": {
            "enabled": True,
            "type": "WarmupCosineDecay",
            "warmup_epochs": 20,
            "warmup_lr": 0.002,
            "base_lr": 0.005,
            "final_lr": 0.0001,
        },
        "early_stopping": {"enabled": True, "patience": 20, "min_delta": 1e-5},
    },
    "data": {"window_size": 40},
}


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def label_tag(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def physics_override(*, multi_scale: bool, lambda_mono: float, lambda_rate: float) -> dict[str, Any]:
    enabled = lambda_mono > 0.0 or lambda_rate > 0.0
    return deep_merge(
        COMMON_OVERRIDE,
        {
            "architecture": {"use_multiscale": multi_scale, "per_window_norm": False},
            "physics_constraints": {
                "enabled": enabled,
                "base_loss_weight": 1.0,
                "monotonic_weight": lambda_mono,
                "boundary_weight": 0.0,
                "smoothness_weight": lambda_rate,
                "monotonic_tolerance": TOLERANCE,
                "min_cycle": MIN_CYCLE,
                "temporal_decay": {"enabled": True, "max_step": 40, "decay_type": "exp", "decay_alpha": 0.2},
            },
        },
    )


def make_tasks() -> Iterator[dict[str, Any]]:
    for ratio in (1.0, 0.5, 0.3, 0.2, 0.1):
        yield {
            "family": "label_budget", "name": f"pi_mscl_r{label_tag(ratio)}", "ratio": ratio,
            "multi_scale": True, "lambda_mono": MONO_DEFAULT, "lambda_rate": 0.0,
        }
    for multi_scale in (False, True):
        for mono in (False, True):
            for rate in (False, True):
                yield {
                    "family": "factorial_ablation",
                    "name": f"ms{int(multi_scale)}_mono{int(mono)}_rate{int(rate)}",
                    "ratio": 0.10, "multi_scale": multi_scale,
                    "lambda_mono": MONO_DEFAULT if mono else 0.0,
                    "lambda_rate": RATE_DEFAULT if rate else 0.0,
                }
    for lambda_mono in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        for lambda_rate in (0.0, 0.01, 0.05, 0.1, 0.2, 0.3):
            yield {
                "family": "sensitivity_2d", "name": f"mono{label_tag(lambda_mono)}_rate{label_tag(lambda_rate)}",
                "ratio": 0.10, "multi_scale": True,
                "lambda_mono": lambda_mono, "lambda_rate": lambda_rate,
            }


def finite(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def run_task(task: dict[str, Any], root: Path, device: str, smoke: bool, force: bool) -> dict[str, Any]:
    override = physics_override(
        multi_scale=task["multi_scale"], lambda_mono=task["lambda_mono"], lambda_rate=task["lambda_rate"]
    )
    if smoke:
        override = deep_merge(override, {"training": {"num_epochs": 1, "scheduler": {"enabled": False}, "early_stopping": {"enabled": False}}})
    run_suffix = "_smoke" if smoke else ""
    # The original 5% factorial/sensitivity runs share task names with the
    # revised 10% protocol. Keep the 10% archives collision-free while still
    # allowing the already-completed label-budget rows to be reused.
    family_dir = task["family"] if task["family"] == "label_budget" else f"{task['family']}_r{label_tag(task['ratio'])}"
    run_dir = root / PIPELINE_REVISION / family_dir / task["name"] / f"split{SPLIT_SEED}_mask{MASK_SEED}_train{TRAINING_SEED}{run_suffix}"
    result_path = run_dir / "result.json"
    prediction_path = run_dir / "predictions.npz"
    manifest = {
        "pipeline_revision": PIPELINE_REVISION, "task": task, "model_id": "multi_pi_rate",
        "model_label": "PI-MSCL variant", "model_type": "ms_cnn_lstm_v2",
        "training_seed": TRAINING_SEED, "mask_seed": MASK_SEED, "split_seed": SPLIT_SEED,
        "supervision_mode": SUPERVISION_MODE, "preserve_battery_boundaries": True,
        "trajectory_tolerance": TOLERANCE, "trajectory_min_cycle": MIN_CYCLE,
        "selection_rule": "validation MAE only", "smoke": smoke, "config_override": override,
    }
    if result_path.exists() and prediction_path.exists() and not force:
        record = json.loads(result_path.read_text(encoding="utf-8"))
        if all(record.get(key) == value for key, value in manifest.items()):
            print(f"[SKIP] {task['family']}/{task['name']}  MAE={record['test_mae'] * 100:.4f}%")
            return record
        raise RuntimeError(f"conflicting archived task: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run_manifest.json", manifest)
    print(f"[RUN] {task['family']}/{task['name']} ratio={task['ratio']} mono={task['lambda_mono']} rate={task['lambda_rate']}")
    started = time.time()
    try:
        _, results, _ = train_cross_battery_model(
            model_type="ms_cnn_lstm_v2", device=device, seed=TRAINING_SEED, split_seed=SPLIT_SEED,
            supervision_ratio=task["ratio"], supervision_seed=MASK_SEED, supervision_mode=SUPERVISION_MODE,
            config_override=override, preserve_battery_boundaries=True, results_dir_override=str(run_dir),
            save_diagnostic_plots=False, color_by_battery=False, highlight_anomalies=False,
        )
        predictions = np.asarray(results["predictions"], dtype=float)
        targets = np.asarray(results["targets"], dtype=float)
        battery_ids = np.asarray(results["battery_ids"], dtype=str)
        cycle_indices = np.asarray(results["cycle_indices"], dtype=np.int32)
        if len({len(predictions), len(targets), len(battery_ids), len(cycle_indices)}) != 1:
            raise RuntimeError("prediction metadata have inconsistent lengths")
        trajectory = compute_trajectory_metrics(predictions, targets, battery_ids, cycle_indices, tolerance=TOLERANCE, min_cycle=MIN_CYCLE)
        np.savez_compressed(prediction_path, predictions=predictions, targets=targets, battery_ids=battery_ids, cycle_indices=cycle_indices)
        write_json(run_dir / "trajectory_metrics.json", trajectory)
        record = {
            **manifest, "elapsed_sec": time.time() - started,
            "test_mae": finite(results["test_mae"]), "test_rmse": finite(results["test_rmse"]),
            "test_mape": finite(results["test_mape"]), "test_r2": finite(results["test_r2"]),
            "best_val_mae": finite(results.get("best_val_mae", np.nan)), "best_epoch": results.get("best_epoch"),
            "cycle_level_train_samples": int(results["cycle_level_train_samples"]),
            "cycle_level_train_labeled_samples": int(results["cycle_level_train_labeled_samples"]),
            "windowed_train_samples": int(results["windowed_train_samples"]),
            "windowed_train_labeled_samples": int(results["windowed_train_labeled_samples"]),
            "windowed_train_label_ratio": finite(results["windowed_train_label_ratio"]),
            "n_test_samples": int(len(predictions)), "representative_battery_by_median_mae": select_median_error_battery(trajectory),
            "monotonicity_violation_rate_percent": trajectory["monotonicity_violation_rate_percent"],
            "mean_cumulative_upward_excess_per_battery": trajectory["mean_cumulative_upward_excess_per_battery"],
            "mean_absolute_rate_variation": trajectory["mean_absolute_rate_variation"],
        }
        write_json(result_path, record)
        print(f"[DONE] {task['family']}/{task['name']}  MAE={record['test_mae'] * 100:.4f}%")
        return record
    except Exception as exc:
        write_json(run_dir / "error.json", {**manifest, "elapsed_sec": time.time() - started, "error": str(exc), "traceback": traceback.format_exc()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", choices=("label_budget", "factorial_ablation", "sensitivity_2d"), default=("label_budget", "factorial_ablation", "sensitivity_2d"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "experiments" / "chapter4_remaining")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N queued tasks (for verification).")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but unavailable")
    root = args.output_root.resolve()
    tasks = [task for task in make_tasks() if task["family"] in args.families]
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        tasks = tasks[:args.limit]
    write_json(root / PIPELINE_REVISION / "execution_plan.json", {"pipeline_revision": PIPELINE_REVISION, "screening_only": True, "device": device, "task_count": len(tasks), "tasks": tasks})
    records = [run_task(task, root, device, args.smoke, args.force) for task in tasks]
    write_json(root / PIPELINE_REVISION / "summary.json", {"pipeline_revision": PIPELINE_REVISION, "screening_only": True, "n_tasks": len(tasks), "completed_records": records})
    print(f"[SUMMARY] completed {len(records)} tasks")


if __name__ == "__main__":
    main()
