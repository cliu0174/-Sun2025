"""Run the single-seed, five-model comparison proposed for Chapter 4.3.

All models use the identical 60/20/20 cell split, 40-cycle input windows,
cycle-level trajectory-prefix label mask, validation set, and held-out test
cells.  The script is intentionally a screening run: one existing best PI-MSCL
seed (train=2262, mask=3262) is used for all five models.  It must not be
reported as a multi-seed mean-plus-standard-deviation result.
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
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.trajectory_metrics import compute_trajectory_metrics, select_median_error_battery
from train_cross_battery import train_cross_battery_model
from train_xgboost_baseline import train_xgboost_baseline


PIPELINE_REVISION = "v5_literature_five_model_comparison"
DEFAULT_RATIO = 0.10
DEFAULT_TRAINING_SEED = 2262
DEFAULT_MASK_SEED = 3262
DEFAULT_SPLIT_SEED = 42
TOLERANCE = 0.005
MIN_CYCLE = 300

NEURAL_TRAINING = {
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

PI_MSCL_OVERRIDE = {
    "architecture": {"use_multiscale": True, "per_window_norm": False},
    "physics_constraints": {
        "enabled": True,
        "base_loss_weight": 1.0,
        "monotonic_weight": 0.3,
        "boundary_weight": 0.0,
        "smoothness_weight": 0.0,
        "monotonic_tolerance": TOLERANCE,
        "min_cycle": MIN_CYCLE,
        "temporal_decay": {"enabled": True, "max_step": 40, "decay_type": "exp", "decay_alpha": 0.2},
    },
}

MODELS: dict[str, dict[str, Any]] = {
    "xgboost": {"label": "XGBoost", "kind": "xgb", "model_type": "xgboost_simple", "source": "project baseline"},
    "gru": {"label": "GRU", "kind": "neural", "model_type": "gru", "source": "project baseline"},
    "transformer_soh": {
        "label": "Transformer-SOH",
        "kind": "neural",
        "model_type": "transformer_soh",
        "source": "Shu et al., J. Energy Storage 108 (2025) 115200, DOI: 10.1016/j.est.2024.115200; core-network reimplementation",
    },
    "cnn_bigru_attention": {
        "label": "CNN-BiGRU-Attention",
        "kind": "neural",
        "model_type": "cnn_bigru_attention",
        "source": "Wu et al., World Electric Vehicle Journal 16 (2025) 487, DOI: 10.3390/wevj16090487; core-network reimplementation without HOA",
    },
    "pi_mscl": {"label": "PI-MSCL", "kind": "neural", "model_type": "ms_cnn_lstm_v2", "source": "proposed method", "override": PI_MSCL_OVERRIDE},
}


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def ratio_tag(ratio: float) -> str:
    return f"r{ratio:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def finite(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def run_one(model_id: str, args: argparse.Namespace, device: str) -> dict[str, Any]:
    spec = MODELS[model_id]
    run_id = f"{model_id}_{ratio_tag(args.ratio)}_split{args.split_seed}_mask{args.mask_seed}_train{args.training_seed}"
    run_directory_name = f"{run_id}_smoke" if args.smoke else run_id
    run_dir = args.output_root / PIPELINE_REVISION / model_id / ratio_tag(args.ratio) / run_directory_name
    result_path = run_dir / "result.json"
    prediction_path = run_dir / "predictions.npz"
    override = deep_merge(NEURAL_TRAINING, spec.get("override", {})) if spec["kind"] == "neural" else None
    if args.smoke and override is not None:
        override = deep_merge(override, {"training": {"num_epochs": 1, "scheduler": {"enabled": False}, "early_stopping": {"enabled": False}}})
    manifest = {
        "run_id": run_id,
        "pipeline_revision": PIPELINE_REVISION,
        "model_id": model_id,
        "model_label": spec["label"],
        "model_type": spec["model_type"],
        "model_source": spec["source"],
        "supervision_ratio": args.ratio,
        "supervision_mode": "trajectory_prefix_concentrated",
        "training_seed": args.training_seed,
        "mask_seed": args.mask_seed,
        "split_seed": args.split_seed,
        "window_size": 40,
        "preserve_battery_boundaries": True,
        "selection_rule": "validation MAE only",
        "smoke": args.smoke,
        "config_override": override,
    }
    if result_path.exists() and prediction_path.exists() and not args.force:
        record = json.loads(result_path.read_text(encoding="utf-8"))
        if all(record.get(key) == value for key, value in manifest.items()):
            print(f"[SKIP] {run_id}: MAE={record['test_mae'] * 100:.4f}%")
            return record
        raise RuntimeError(f"existing archived result conflicts with requested manifest: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run_manifest.json", manifest)
    started = time.time()
    print(f"[RUN] {run_id} ({spec['label']})")
    try:
        if spec["kind"] == "xgb":
            wrapper, results, _ = train_xgboost_baseline(
                model_type=spec["model_type"], seed=args.training_seed, split_seed=args.split_seed,
                supervision_ratio=args.ratio, supervision_seed=args.mask_seed,
                supervision_mode="trajectory_prefix_concentrated", apply_cleaning=False,
                degradation_scenario="none",
            )
            wrapper.model.model.save_model(str(run_dir / "xgboost_model.json"))
        else:
            _, results, _ = train_cross_battery_model(
                model_type=spec["model_type"], device=device, seed=args.training_seed,
                split_seed=args.split_seed, supervision_ratio=args.ratio,
                supervision_seed=args.mask_seed, supervision_mode="trajectory_prefix_concentrated",
                config_override=override, preserve_battery_boundaries=True,
                results_dir_override=str(run_dir), save_diagnostic_plots=False,
                color_by_battery=False, highlight_anomalies=False,
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
            **manifest,
            "elapsed_sec": time.time() - started,
            "test_mae": finite(results["test_mae"]), "test_rmse": finite(results["test_rmse"]),
            "test_mape": finite(results["test_mape"]), "test_r2": finite(results["test_r2"]),
            "best_val_mae": finite(results.get("best_val_mae", np.nan)), "best_epoch": results.get("best_epoch"),
            "cycle_level_train_samples": int(results["cycle_level_train_samples"]),
            "cycle_level_train_labeled_samples": int(results["cycle_level_train_labeled_samples"]),
            "windowed_train_samples": int(results["windowed_train_samples"]),
            "windowed_train_labeled_samples": int(results["windowed_train_labeled_samples"]),
            "windowed_train_label_ratio": finite(results["windowed_train_label_ratio"]),
            "n_test_samples": int(len(predictions)),
            "representative_battery_by_median_mae": select_median_error_battery(trajectory),
            "monotonicity_violation_rate_percent": trajectory["monotonicity_violation_rate_percent"],
            "mean_cumulative_upward_excess_per_battery": trajectory["mean_cumulative_upward_excess_per_battery"],
            "mean_absolute_second_difference": trajectory["mean_absolute_second_difference"],
        }
        write_json(result_path, record)
        print(f"[DONE] {run_id}: MAE={record['test_mae'] * 100:.4f}%, RMSE={record['test_rmse'] * 100:.4f}%")
        return record
    except Exception as exc:
        write_json(run_dir / "error.json", {**manifest, "elapsed_sec": time.time() - started, "error": str(exc), "traceback": traceback.format_exc()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--ratio", type=float, default=DEFAULT_RATIO)
    parser.add_argument("--training-seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--mask-seed", type=int, default=DEFAULT_MASK_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "experiments" / "chapter4_five_model_comparison")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.ratio <= 1.0:
        raise SystemExit("--ratio must lie in (0, 1]")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    args.output_root = args.output_root.resolve()
    records = [run_one(model_id, args, device) for model_id in args.models]
    summary = {"pipeline_revision": PIPELINE_REVISION, "screening_only": True, "n_runs": len(records), "protocol": {"ratio": args.ratio, "training_seed": args.training_seed, "mask_seed": args.mask_seed, "split_seed": args.split_seed, "supervision_mode": "trajectory_prefix_concentrated", "window_size": 40, "device": device}, "runs": records}
    write_json(args.output_root / PIPELINE_REVISION / "summary.json", summary)
    print(f"[SUMMARY] {args.output_root / PIPELINE_REVISION / 'summary.json'}")


if __name__ == "__main__":
    main()
