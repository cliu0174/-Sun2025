"""Run same-protocol classical and modern baselines for the manuscript.

The endpoint design complements the full four-ratio 2x2 PI-MSCL matrix. Every
baseline uses the same fixed battery split, 16-feature windows, cycle-level
label masks, fully labelled validation/test cells, and paired seed roles.
Published numbers obtained with different protocols are deliberately excluded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.trajectory_metrics import (  # noqa: E402
    compute_trajectory_metrics,
    select_median_error_battery,
)
from train_cross_battery import train_cross_battery_model  # noqa: E402
from train_xgboost_baseline import train_xgboost_baseline  # noqa: E402


PIPELINE_REVISION = "v2_trajectory_prefix_concentrated"
DEFAULT_SEEDS = [929, 2262, 7]
DEFAULT_MASK_SEEDS = [1929, 3262, 1007]
DEFAULT_RATIOS = [0.1]
SUPERVISION_MODES = ("distributed_random", "trajectory_prefix_concentrated")
TOLERANCE = 0.005
MIN_CYCLE = 300

BASELINES: Dict[str, Dict[str, Any]] = {
    "xgboost": {"label": "XGBoost", "model_type": "xgboost_simple", "kind": "xgb"},
    "lstm": {"label": "LSTM", "model_type": "lstm", "kind": "neural"},
    "gru": {"label": "GRU", "model_type": "gru", "kind": "neural"},
    "attn_cnn_lstm": {
        "label": "Attention CNN-LSTM",
        "model_type": "cnn_lstm_attention",
        "kind": "neural",
    },
}

COMMON_OVERRIDE: Dict[str, Any] = {
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
    "physics_constraints": {
        "enabled": False,
        "base_loss_weight": 1.0,
        "monotonic_weight": 0.0,
        "boundary_weight": 0.0,
        "smoothness_weight": 0.0,
    },
}


def ratio_tag(ratio: float) -> str:
    text = f"{float(ratio):.6f}".rstrip("0").rstrip(".")
    return f"r{text}".replace(".", "p")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def finite_float(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def run_one(
    *,
    model_id: str,
    ratio: float,
    training_seed: int,
    mask_seed: int,
    split_seed: int,
    output_root: Path,
    device: str,
    supervision_mode: str,
    smoke: bool,
    force: bool,
) -> Dict[str, Any]:
    spec = BASELINES[model_id]
    suffix = "_smoke" if smoke else ""
    run_dir = (
        output_root
        / "fixed_split"
        / PIPELINE_REVISION
        / model_id
        / ratio_tag(ratio)
        / f"split{split_seed}_mask{mask_seed}_train{training_seed}{suffix}"
    )
    result_path = run_dir / "result.json"
    prediction_path = run_dir / "predictions.npz"
    override = json.loads(json.dumps(COMMON_OVERRIDE))
    if smoke:
        override["training"]["num_epochs"] = 1
        override["training"]["scheduler"] = {"enabled": False}
        override["training"]["early_stopping"] = {"enabled": False}

    run_id = (
        f"{model_id}_{ratio_tag(ratio)}_split{split_seed}_"
        f"mask{mask_seed}_train{training_seed}"
    )
    manifest = {
        "run_id": run_id,
        "pipeline_revision": PIPELINE_REVISION,
        "protocol": "fixed_split",
        "model_id": model_id,
        "model_label": spec["label"],
        "model_type": spec["model_type"],
        "supervision_ratio": ratio,
        "supervision_mode": supervision_mode,
        "training_seed": training_seed,
        "mask_seed": mask_seed,
        "split_seed": split_seed,
        "preserve_battery_boundaries": True,
        "trajectory_tolerance": TOLERANCE,
        "trajectory_min_cycle": MIN_CYCLE,
        "smoke": smoke,
        "config_override": override if spec["kind"] == "neural" else None,
    }

    if not force and result_path.exists() and prediction_path.exists():
        with result_path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        comparable = (
            "run_id",
            "pipeline_revision",
            "model_id",
            "model_type",
            "supervision_ratio",
            "supervision_mode",
            "training_seed",
            "mask_seed",
            "split_seed",
            "preserve_battery_boundaries",
            "smoke",
            "config_override",
        )
        mismatched = [key for key in comparable if record.get(key) != manifest.get(key)]
        if mismatched:
            raise RuntimeError(
                f"Archived metadata mismatch at {run_dir}: {', '.join(mismatched)}"
            )
        print(f"[SKIP] {run_id} MAE={100 * record['test_mae']:.4f}%")
        return record

    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run_manifest.json", manifest)
    started = time.time()
    print(
        f"[RUN] {run_id} model={spec['label']} ratio={ratio} "
        f"split={split_seed} mask={mask_seed} train={training_seed}"
    )
    try:
        if spec["kind"] == "xgb":
            wrapper, results, _ = train_xgboost_baseline(
                model_type=spec["model_type"],
                seed=training_seed,
                split_seed=split_seed,
                supervision_ratio=ratio,
                supervision_seed=mask_seed,
                supervision_mode=supervision_mode,
                apply_cleaning=False,
                degradation_scenario="none",
            )
            wrapper.model.model.save_model(str(run_dir / "xgboost_model.json"))
        else:
            _, results, _ = train_cross_battery_model(
                model_type=spec["model_type"],
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
        lengths = {len(predictions), len(targets), len(battery_ids), len(cycle_indices)}
        if len(lengths) != 1:
            raise RuntimeError("Prediction metadata have inconsistent lengths")

        trajectory = compute_trajectory_metrics(
            predictions,
            targets,
            battery_ids,
            cycle_indices,
            tolerance=TOLERANCE,
            min_cycle=MIN_CYCLE,
        )
        representative = select_median_error_battery(trajectory)
        np.savez_compressed(
            prediction_path,
            predictions=predictions,
            targets=targets,
            battery_ids=battery_ids,
            cycle_indices=cycle_indices,
        )
        write_json(run_dir / "trajectory_metrics.json", trajectory)

        record = {
            **manifest,
            "elapsed_sec": time.time() - started,
            "test_mae": finite_float(results["test_mae"]),
            "test_rmse": finite_float(results["test_rmse"]),
            "test_mape": finite_float(results["test_mape"]),
            "test_r2": finite_float(results["test_r2"]),
            "best_val_mae": finite_float(results.get("best_val_mae", np.nan)),
            "best_epoch": results.get("best_epoch"),
            "cycle_level_train_samples": int(results["cycle_level_train_samples"]),
            "cycle_level_train_labeled_samples": int(
                results["cycle_level_train_labeled_samples"]
            ),
            "windowed_train_samples": int(results["windowed_train_samples"]),
            "windowed_train_labeled_samples": int(
                results["windowed_train_labeled_samples"]
            ),
            "windowed_train_label_ratio": finite_float(
                results["windowed_train_label_ratio"]
            ),
            "n_test_samples": int(len(predictions)),
            "representative_battery_by_median_mae": representative,
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
        write_json(result_path, record)
        print(
            f"[DONE] {run_id} MAE={100 * record['test_mae']:.4f}% "
            f"RMSE={100 * record['test_rmse']:.4f}%"
        )
        return record
    except Exception as exc:
        write_json(
            run_dir / "error.json",
            {
                **manifest,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_sec": time.time() - started,
            },
        )
        raise


def aggregate(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    records = list(records)
    groups: Dict[str, Dict[str, Any]] = {}
    for model_id, spec in BASELINES.items():
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
                and abs(float(row["supervision_ratio"]) - ratio) < 1e-12
                and not row.get("smoke", False)
            ]
            if not subset:
                continue
            key = f"{model_id}/{ratio_tag(ratio)}"
            groups[key] = {
                "model_id": model_id,
                "model_label": spec["label"],
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
                groups[key][f"{metric}_mean"] = float(values.mean())
                groups[key][f"{metric}_std"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else None
                )
    return {
        "pipeline_revision": PIPELINE_REVISION,
        "protocol": "fixed_split",
        "n_runs": len(records),
        "groups": groups,
        "runs": records,
    }


def collect_records(revision_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(revision_root.glob("*/r*/split*/result.json")):
        if not path.with_name("predictions.npz").exists():
            continue
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("pipeline_revision") == PIPELINE_REVISION:
            records.append(record)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=tuple(BASELINES), default=list(BASELINES))
    parser.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS)
    parser.add_argument(
        "--supervision-mode",
        choices=SUPERVISION_MODES,
        default="trajectory_prefix_concentrated",
    )
    parser.add_argument("--training-seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--mask-seeds", nargs="+", type=int, default=DEFAULT_MASK_SEEDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "paper_baselines",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.training_seeds) != len(args.mask_seeds):
        raise SystemExit("--training-seeds and --mask-seeds must have equal length")
    invalid = [ratio for ratio in args.ratios if not 0 < ratio <= 1]
    if invalid:
        raise SystemExit(f"Ratios must lie in (0, 1]: {invalid}")

    for model_id in args.models:
        for ratio in args.ratios:
            for training_seed, mask_seed in zip(args.training_seeds, args.mask_seeds):
                run_one(
                    model_id=model_id,
                    ratio=ratio,
                    training_seed=training_seed,
                    mask_seed=mask_seed,
                    split_seed=args.split_seed,
                    output_root=args.output_root,
                    device=args.device,
                    supervision_mode=args.supervision_mode,
                    smoke=args.smoke,
                    force=args.force,
                )

    revision_root = args.output_root / "fixed_split" / PIPELINE_REVISION
    records = collect_records(revision_root)
    write_json(revision_root / "summary.json", aggregate(records))
    print(f"[SUMMARY] {len(records)} archived runs -> {revision_root / 'summary.json'}")


if __name__ == "__main__":
    main()
