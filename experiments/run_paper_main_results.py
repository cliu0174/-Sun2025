"""Reproduce the paper's main 2x2 architecture/physics experiment matrix.

This runner supersedes the old cross-pipeline ablations for manuscript use.  It
enforces battery-boundary-preserving windows for every model, archives complete
predictions and cycle metadata per run, and exposes the split/mask/training seed
policy explicitly on the command line.

Examples
--------
Smoke test one configuration::

    python experiments/run_paper_main_results.py \
        --protocol fixed_split --split-seed 42 --models multi_pi \
        --ratios 0.3 --training-seeds 929 --smoke

Full matrix with repeated independent battery partitions::

    python experiments/run_paper_main_results.py \
        --protocol independent_splits

Full matrix with one fixed battery partition and repeated masks/training::

    python experiments/run_paper_main_results.py \
        --protocol fixed_split --split-seed 42
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
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.trajectory_metrics import (  # noqa: E402
    compute_trajectory_metrics,
    select_median_error_battery,
)
from train_cross_battery import train_cross_battery_model  # noqa: E402


DEFAULT_SEEDS = [929, 2262, 7]
DEFAULT_RATIOS = [1.0, 0.7, 0.5, 0.3]
TOLERANCE = 0.005
MIN_CYCLE = 300
PIPELINE_REVISION = "v4_label_protocols"
SUPERVISION_MODES = ("distributed_random", "trajectory_prefix_concentrated")


COMMON_OVERRIDE: Dict[str, Any] = {
    "architecture": {
        "per_window_norm": False,
    },
    "training": {
        "batch_size": 1024,
        "num_epochs": 200,
        "learning_rate": 0.0005,
        "early_stopping": {
            "enabled": True,
            "patience": 20,
            "min_delta": 1e-5,
        },
    },
}


def _physics_config(enabled: bool, monotonic_weight: float = 0.3) -> Dict[str, Any]:
    return {
        "enabled": enabled,
        "base_loss_weight": 1.0,
        "monotonic_weight": float(monotonic_weight) if enabled else 0.0,
        "boundary_weight": 0.0,
        "smoothness_weight": 0.0,
        "monotonic_tolerance": TOLERANCE,
        "min_cycle": MIN_CYCLE,
        "temporal_decay": {
            "enabled": True,
            "max_step": 40,
            "decay_type": "exp",
            "decay_alpha": 0.2,
        },
    }


MODEL_MATRIX: Dict[str, Dict[str, Any]] = {
    "single_no_pi": {
        "label": "Single-scale CNN-LSTM",
        "use_multiscale": False,
        "use_physics": False,
    },
    "single_pi": {
        "label": "PI-CNN-LSTM",
        "use_multiscale": False,
        "use_physics": True,
    },
    "multi_no_pi": {
        "label": "Multi-scale CNN-LSTM",
        "use_multiscale": True,
        "use_physics": False,
    },
    "multi_pi": {
        "label": "PI-MSCL",
        "use_multiscale": True,
        "use_physics": True,
    },
}


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def model_override(
    model_id: str, smoke: bool = False, monotonic_weight: float = 0.3
) -> Dict[str, Any]:
    spec = MODEL_MATRIX[model_id]
    override = deep_merge(
        COMMON_OVERRIDE,
        {
            "architecture": {
                "use_multiscale": spec["use_multiscale"],
                "per_window_norm": False,
            },
            "physics_constraints": _physics_config(
                spec["use_physics"], monotonic_weight=monotonic_weight
            ),
        },
    )
    if smoke:
        override = deep_merge(
            override,
            {
                "training": {
                    "num_epochs": 1,
                    "scheduler": {"enabled": False},
                    "early_stopping": {"enabled": False},
                }
            },
        )
    return override


def _ratio_tag(ratio: float) -> str:
    """Return a collision-free, human-readable directory label for a ratio.

    One decimal place aliases 0.05 and 0.10 to ``r0p1``.  Results from those
    distinct sparse-supervision conditions must never share an output folder.
    """
    text = f"{float(ratio):.6f}".rstrip("0").rstrip(".")
    return f"r{text}".replace(".", "p")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def _finite_float(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def run_one(
    *,
    model_id: str,
    ratio: float,
    training_seed: int,
    mask_seed: int,
    split_seed: int,
    supervision_mode: str,
    protocol: str,
    output_root: Path,
    device: str,
    smoke: bool,
    force: bool,
    monotonic_weight: float,
    experiment_tag: str | None,
) -> Dict[str, Any]:
    run_suffix = "_smoke" if smoke else ""
    experiment_root = (
        output_root
        / protocol
        / PIPELINE_REVISION
        / supervision_mode
    )
    if experiment_tag:
        experiment_root = experiment_root / experiment_tag
    run_dir = (
        experiment_root
        / model_id
        / _ratio_tag(ratio)
        / f"split{split_seed}_mask{mask_seed}_train{training_seed}{run_suffix}"
    )
    result_path = run_dir / "result.json"
    prediction_path = run_dir / "predictions.npz"
    spec = MODEL_MATRIX[model_id]
    override = model_override(
        model_id, smoke=smoke, monotonic_weight=monotonic_weight
    )
    run_id = (
        f"{model_id}_{_ratio_tag(ratio)}_split{split_seed}_"
        f"mask{mask_seed}_train{training_seed}"
    )
    manifest = {
        "run_id": run_id,
        "pipeline_revision": PIPELINE_REVISION,
        "protocol": protocol,
        "model_id": model_id,
        "model_label": spec["label"],
        "model_type": "ms_cnn_lstm_v2",
        "supervision_ratio": ratio,
        "training_seed": training_seed,
        "mask_seed": mask_seed,
        "split_seed": split_seed,
        "supervision_mode": supervision_mode,
        "experiment_tag": experiment_tag,
        "monotonic_weight": float(monotonic_weight),
        "preserve_battery_boundaries": True,
        "trajectory_tolerance": TOLERANCE,
        "trajectory_min_cycle": MIN_CYCLE,
        "smoke": smoke,
        "device": device,
        "config_override": override,
    }
    if not force and result_path.exists() and prediction_path.exists():
        with result_path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        comparable_keys = (
            "run_id",
            "pipeline_revision",
            "protocol",
            "model_id",
            "supervision_ratio",
            "training_seed",
            "mask_seed",
            "split_seed",
            "supervision_mode",
            "experiment_tag",
            "monotonic_weight",
            "preserve_battery_boundaries",
            "trajectory_tolerance",
            "trajectory_min_cycle",
            "smoke",
            "config_override",
        )
        mismatched = [key for key in comparable_keys if record.get(key) != manifest.get(key)]
        if mismatched:
            raise RuntimeError(
                f"existing run metadata do not match the requested experiment at {run_dir}; "
                f"mismatched fields: {', '.join(mismatched)}. Use --force only after "
                "reviewing the archived run."
            )
        print(f"[SKIP] {record['run_id']}  MAE={record['test_mae'] * 100:.4f}%")
        return record

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "run_manifest.json", manifest)

    print("\n" + "=" * 88)
    print(f"[RUN] {run_id}")
    print(
        f"      model={spec['label']} ratio={ratio} split={split_seed} "
        f"mask={mask_seed} train={training_seed} device={device}"
    )
    print("=" * 88)
    started = time.time()
    try:
        _, results, _ = train_cross_battery_model(
            model_type="ms_cnn_lstm_v2",
            device=device,
            seed=training_seed,
            split_seed=split_seed,
            supervision_ratio=ratio,
            supervision_seed=mask_seed,
            supervision_mode=supervision_mode,
            config_override=override,
            preserve_battery_boundaries=True,
            results_dir_override=str(run_dir),
            save_diagnostic_plots=False,
            color_by_battery=False,
            highlight_anomalies=False,
        )

        predictions = np.asarray(results["predictions"], dtype=float)
        targets = np.asarray(results["targets"], dtype=float)
        battery_ids = np.asarray(results["battery_ids"], dtype=str)
        cycle_indices = np.asarray(results["cycle_indices"], dtype=np.int32)
        if not (
            len(predictions)
            == len(targets)
            == len(battery_ids)
            == len(cycle_indices)
        ):
            raise RuntimeError("saved prediction metadata have inconsistent lengths")

        trajectory = compute_trajectory_metrics(
            predictions,
            targets,
            battery_ids,
            cycle_indices,
            tolerance=TOLERANCE,
            min_cycle=MIN_CYCLE,
        )
        representative_battery = select_median_error_battery(trajectory)

        np.savez_compressed(
            prediction_path,
            predictions=predictions,
            targets=targets,
            battery_ids=battery_ids,
            cycle_indices=cycle_indices,
        )
        _write_json(run_dir / "trajectory_metrics.json", trajectory)

        elapsed = time.time() - started
        record = {
            **manifest,
            "elapsed_sec": elapsed,
            "test_mae": _finite_float(results["test_mae"]),
            "test_rmse": _finite_float(results["test_rmse"]),
            "test_mape": _finite_float(results["test_mape"]),
            "test_r2": _finite_float(results["test_r2"]),
            "best_val_mae": _finite_float(results["best_val_mae"]),
            "best_epoch": int(results["best_epoch"]),
            "cycle_level_train_samples": int(results["cycle_level_train_samples"]),
            "cycle_level_train_labeled_samples": int(
                results["cycle_level_train_labeled_samples"]
            ),
            "windowed_train_samples": int(results["windowed_train_samples"]),
            "windowed_train_labeled_samples": int(
                results["windowed_train_labeled_samples"]
            ),
            "windowed_train_label_ratio": _finite_float(
                results["windowed_train_label_ratio"]
            ),
            "n_test_samples": int(len(predictions)),
            "representative_battery_by_median_mae": representative_battery,
            "monotonicity_violation_rate_percent": trajectory[
                "monotonicity_violation_rate_percent"
            ],
            "mean_cumulative_upward_excess_per_battery": trajectory[
                "mean_cumulative_upward_excess_per_battery"
            ],
            "mean_absolute_second_difference": trajectory[
                "mean_absolute_second_difference"
            ],
        }
        _write_json(result_path, record)
        error_path = run_dir / "error.json"
        if error_path.exists():
            error_path.unlink()
        print(
            f"[DONE] {run_id}  MAE={record['test_mae'] * 100:.4f}%  "
            f"RMSE={record['test_rmse'] * 100:.4f}%  "
            f"mono={record['monotonicity_violation_rate_percent']:.3f}%"
        )
        return record
    except Exception as exc:
        failure = {
            **manifest,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_sec": time.time() - started,
        }
        _write_json(run_dir / "error.json", failure)
        raise


def aggregate(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    records = list(records)
    grouped: Dict[str, Dict[str, Any]] = {}
    for model_id in MODEL_MATRIX:
        ratios = sorted(
            {
                float(row["supervision_ratio"])
                for row in records
                if row["model_id"] == model_id
            },
            reverse=True,
        )
        for ratio in ratios:
            subset = [
                row
                for row in records
                if row["model_id"] == model_id
                and abs(row["supervision_ratio"] - ratio) < 1e-12
                and not row.get("smoke", False)
            ]
            if not subset:
                continue
            key = f"{model_id}/{_ratio_tag(ratio)}"
            grouped[key] = {
                "model_id": model_id,
                "model_label": MODEL_MATRIX[model_id]["label"],
                "supervision_ratio": ratio,
                "n": len(subset),
            }
            for metric in (
                "test_mae",
                "test_rmse",
                "test_r2",
                "monotonicity_violation_rate_percent",
                "mean_cumulative_upward_excess_per_battery",
                "mean_absolute_second_difference",
            ):
                values = np.asarray([row[metric] for row in subset], dtype=float)
                grouped[key][f"{metric}_mean"] = float(np.mean(values))
                grouped[key][f"{metric}_std"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else None
                )
    return {"n_runs": len(records), "groups": grouped, "runs": records}


def collect_completed_records(revision_root: Path) -> List[Dict[str, Any]]:
    """Load every completed run in one pipeline revision for staged execution."""
    records: List[Dict[str, Any]] = []
    for result_path in sorted(revision_root.glob("**/result.json")):
        prediction_path = result_path.with_name("predictions.npz")
        if not prediction_path.exists():
            continue
        with result_path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("pipeline_revision") != PIPELINE_REVISION:
            continue
        records.append(record)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=("independent_splits", "fixed_split"),
        required=True,
        help=(
            "independent_splits: each training seed also defines the battery split; "
            "fixed_split: all repeats share --split-seed"
        ),
    )
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument(
        "--supervision-mode",
        choices=SUPERVISION_MODES,
        default="distributed_random",
        help="Label allocation across the fixed training-cell set.",
    )
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_MATRIX), default=list(MODEL_MATRIX)
    )
    parser.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS)
    parser.add_argument(
        "--training-seeds", nargs="+", type=int, default=DEFAULT_SEEDS
    )
    parser.add_argument(
        "--mask-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional one-to-one mask seeds; defaults to the training seeds",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "paper_main_results",
    )
    parser.add_argument(
        "--monotonic-weight",
        type=float,
        default=0.3,
        help="Weight of the soft monotonicity penalty when physics is enabled.",
    )
    parser.add_argument(
        "--experiment-tag",
        default=None,
        help="Optional separate subdirectory for a sensitivity experiment.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.protocol == "fixed_split" and args.split_seed is None:
        raise SystemExit("--split-seed is required for --protocol fixed_split")
    if args.mask_seeds is not None and len(args.mask_seeds) != len(args.training_seeds):
        raise SystemExit("--mask-seeds must have the same length as --training-seeds")
    for ratio in args.ratios:
        if not 0.0 < ratio <= 1.0:
            raise SystemExit(f"invalid supervision ratio: {ratio}")
    if args.monotonic_weight < 0.0:
        raise SystemExit("--monotonic-weight must be non-negative")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False")

    mask_seeds: List[int] = args.mask_seeds or list(args.training_seeds)
    output_root = args.output_root.resolve()
    revision_root = output_root / args.protocol / PIPELINE_REVISION
    revision_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "pipeline_revision": PIPELINE_REVISION,
        "protocol": args.protocol,
        "models": args.models,
        "ratios": args.ratios,
        "training_seeds": args.training_seeds,
        "mask_seeds": mask_seeds,
        "fixed_split_seed": args.split_seed,
        "supervision_mode": args.supervision_mode,
        "monotonic_weight": args.monotonic_weight,
        "experiment_tag": args.experiment_tag,
        "device": device,
        "smoke": args.smoke,
        "preserve_battery_boundaries": True,
        "trajectory_tolerance": TOLERANCE,
        "trajectory_min_cycle": MIN_CYCLE,
    }
    _write_json(revision_root / "execution_plan.json", plan)

    records = []
    for model_id in args.models:
        for ratio in args.ratios:
            for training_seed, mask_seed in zip(args.training_seeds, mask_seeds):
                split_seed = (
                    training_seed
                    if args.protocol == "independent_splits"
                    else int(args.split_seed)
                )
                record = run_one(
                    model_id=model_id,
                    ratio=ratio,
                    training_seed=training_seed,
                    mask_seed=mask_seed,
                    split_seed=split_seed,
                    supervision_mode=args.supervision_mode,
                    protocol=args.protocol,
                    output_root=output_root,
                    device=device,
                    smoke=args.smoke,
                    force=args.force,
                    monotonic_weight=args.monotonic_weight,
                    experiment_tag=args.experiment_tag,
                )
                records.append(record)

    result_root = revision_root / args.supervision_mode
    if args.experiment_tag:
        result_root = result_root / args.experiment_tag
    completed_records = collect_completed_records(result_root)
    summary = aggregate(completed_records)
    summary["last_invocation"] = plan
    summary["pipeline_revision"] = PIPELINE_REVISION
    summary_path = result_root / "summary.json"
    _write_json(summary_path, summary)
    print(
        f"\nSaved summary: {summary_path} "
        f"({len(completed_records)} completed runs in this revision)"
    )


if __name__ == "__main__":
    main()
