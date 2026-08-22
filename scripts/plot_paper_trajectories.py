"""Generate manuscript trajectory and aligned-error figures from archived runs.

Selection is performed without visual inspection in two stages. First, the
paired mask/training-seed repetition whose four-model mean test MAE is closest
to the median across repetitions is selected. Second, within that repetition,
each test cell's MAE is averaged across the four models and the cell closest to
the across-cell median is selected, with deterministic tie breakers.

The grouped figure and both child figures are rendered independently from the
same archived arrays.  Child panels are never cropped from the group image.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paper_plot_style import (  # noqa: E402
    COLORS,
    LINESTYLES,
    add_panel_label,
    add_panel_title,
    apply_paper_style,
    polish_axes,
    save_figure,
)


MODEL_ORDER = ["single_no_pi", "single_pi", "multi_no_pi", "multi_pi"]
MODEL_LABELS = {
    "single_no_pi": "Single-scale CNN-LSTM",
    "single_pi": "PI-CNN-LSTM",
    "multi_no_pi": "Multi-scale CNN-LSTM",
    "multi_pi": "PI-MSCL",
}


def ratio_tag(ratio: float) -> str:
    return f"r{ratio:.1f}".replace(".", "p")


def load_factorial_run(
    revision_root: Path,
    *,
    model_id: str,
    ratio: float,
    split_seed: int,
    mask_seed: int,
    training_seed: int,
) -> Dict[str, object]:
    run_tag = f"split{split_seed}_mask{mask_seed}_train{training_seed}"
    run_dir = revision_root / model_id / ratio_tag(ratio) / run_tag
    result_path = run_dir / "result.json"
    prediction_path = run_dir / "predictions.npz"
    if not result_path.exists() or not prediction_path.exists():
        raise FileNotFoundError(f"incomplete archived run: {run_dir}")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("smoke"):
        raise ValueError(f"refusing to plot smoke run: {run_dir}")
    expected = {
        "model_id": model_id,
        "supervision_ratio": ratio,
        "split_seed": split_seed,
        "mask_seed": mask_seed,
        "training_seed": training_seed,
    }
    mismatch = [key for key, value in expected.items() if result.get(key) != value]
    if mismatch:
        raise ValueError(f"run metadata mismatch at {run_dir}: {', '.join(mismatch)}")

    with np.load(prediction_path, allow_pickle=False) as data:
        required = {"predictions", "targets", "battery_ids", "cycle_indices"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"missing arrays at {prediction_path}: {sorted(missing)}")
        arrays = {name: np.asarray(data[name]) for name in required}
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"inconsistent prediction-array lengths at {prediction_path}")
    return {"result": result, "arrays": arrays, "run_dir": run_dir}


def load_factorial_runs(
    revision_root: Path,
    *,
    ratio: float,
    split_seed: int,
    mask_seed: int,
    training_seed: int,
) -> Dict[str, Dict[str, object]]:
    runs = {
        model_id: load_factorial_run(
            revision_root,
            model_id=model_id,
            ratio=ratio,
            split_seed=split_seed,
            mask_seed=mask_seed,
            training_seed=training_seed,
        )
        for model_id in MODEL_ORDER
    }
    reference = runs[MODEL_ORDER[0]]["arrays"]
    for model_id in MODEL_ORDER[1:]:
        arrays = runs[model_id]["arrays"]
        for key in ("targets", "battery_ids", "cycle_indices"):
            if not np.array_equal(reference[key], arrays[key]):
                raise ValueError(
                    f"factorial runs do not share identical {key}: {MODEL_ORDER[0]} vs {model_id}"
                )
    return runs


def discover_complete_seed_pairs(
    revision_root: Path, *, ratio: float, split_seed: int
) -> list[Tuple[int, int]]:
    """Discover non-smoke paired repetitions from the first factorial model."""
    model_root = revision_root / MODEL_ORDER[0] / ratio_tag(ratio)
    pairs: list[Tuple[int, int]] = []
    if not model_root.exists():
        return pairs
    for result_path in sorted(model_root.glob(f"split{split_seed}_mask*_train*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("smoke"):
            continue
        if int(result.get("split_seed", -1)) != split_seed:
            continue
        pairs.append((int(result["mask_seed"]), int(result["training_seed"])))
    return sorted(set(pairs))


def select_cross_model_median_seed_pair(
    revision_root: Path, *, ratio: float, split_seed: int
) -> tuple[Tuple[int, int], Dict[str, Dict[str, object]], Dict[str, float]]:
    """Select the paired repetition closest to median four-model test MAE."""
    pairs = discover_complete_seed_pairs(
        revision_root, ratio=ratio, split_seed=split_seed
    )
    if not pairs:
        raise FileNotFoundError(
            f"no completed seed pairs at split={split_seed}, ratio={ratio}"
        )
    loaded: Dict[Tuple[int, int], Dict[str, Dict[str, object]]] = {}
    scores: Dict[Tuple[int, int], float] = {}
    for mask_seed, training_seed in pairs:
        runs = load_factorial_runs(
            revision_root,
            ratio=ratio,
            split_seed=split_seed,
            mask_seed=mask_seed,
            training_seed=training_seed,
        )
        loaded[(mask_seed, training_seed)] = runs
        scores[(mask_seed, training_seed)] = float(
            np.mean(
                [float(runs[model_id]["result"]["test_mae"]) for model_id in MODEL_ORDER]
            )
        )
    median = float(np.median(list(scores.values())))
    selected = min(
        scores,
        key=lambda pair: (abs(scores[pair] - median), pair[0], pair[1]),
    )
    serializable_scores = {
        f"mask{mask_seed}_train{training_seed}": score
        for (mask_seed, training_seed), score in sorted(scores.items())
    }
    return selected, loaded[selected], serializable_scores


def select_cross_model_median_battery(
    runs: Mapping[str, Mapping[str, object]],
) -> tuple[str, Dict[str, Dict[str, float]]]:
    reference = runs[MODEL_ORDER[0]]["arrays"]
    battery_ids = np.asarray(reference["battery_ids"], dtype=str)
    targets = np.asarray(reference["targets"], dtype=float)
    rows: Dict[str, Dict[str, float]] = {}
    for battery_id in sorted(np.unique(battery_ids)):
        mask = battery_ids == battery_id
        model_mae = {}
        for model_id in MODEL_ORDER:
            predictions = np.asarray(runs[model_id]["arrays"]["predictions"], dtype=float)
            model_mae[model_id] = float(np.mean(np.abs(predictions[mask] - targets[mask])))
        rows[battery_id] = {
            **model_mae,
            "cross_model_mean_mae": float(np.mean(list(model_mae.values()))),
        }
    median = float(np.median([row["cross_model_mean_mae"] for row in rows.values()]))
    selected = min(
        rows,
        key=lambda battery_id: (
            abs(rows[battery_id]["cross_model_mean_mae"] - median),
            battery_id,
        ),
    )
    return selected, rows


def extract_selected_trajectory(
    runs: Mapping[str, Mapping[str, object]], battery_id: str
) -> Dict[str, np.ndarray]:
    reference = runs[MODEL_ORDER[0]]["arrays"]
    battery_ids = np.asarray(reference["battery_ids"], dtype=str)
    mask = battery_ids == battery_id
    if not np.any(mask):
        raise ValueError(f"battery {battery_id!r} is absent from the archived predictions")
    cycles = np.asarray(reference["cycle_indices"], dtype=int)[mask]
    order = np.argsort(cycles, kind="stable")
    selected = {
        "cycles": cycles[order],
        "targets": np.asarray(reference["targets"], dtype=float)[mask][order],
    }
    for model_id in MODEL_ORDER:
        selected[model_id] = np.asarray(
            runs[model_id]["arrays"]["predictions"], dtype=float
        )[mask][order]
    return selected


def draw_trajectory_panel(
    ax: plt.Axes,
    trajectory: Mapping[str, np.ndarray],
    *,
    panel_label: str | None = None,
    panel_title: str | None = None,
) -> None:
    cycles = trajectory["cycles"]
    ax.plot(
        cycles,
        trajectory["targets"] * 100.0,
        color=COLORS["truth"],
        linewidth=2.25,
        label="Measured SOH",
        zorder=5,
    )
    for model_id in MODEL_ORDER:
        ax.plot(
            cycles,
            trajectory[model_id] * 100.0,
            color=COLORS[model_id],
            linestyle=LINESTYLES[model_id],
            linewidth=1.45,
            label=MODEL_LABELS[model_id],
        )
    ax.set_xlabel("Cycle index")
    ax.set_ylabel("SOH (%)")
    polish_axes(ax)
    if panel_label:
        add_panel_label(ax, panel_label)
    if panel_title:
        add_panel_title(ax, panel_title)


def draw_error_panel(
    ax: plt.Axes,
    trajectory: Mapping[str, np.ndarray],
    *,
    panel_label: str | None = None,
    panel_title: str | None = None,
) -> None:
    cycles = trajectory["cycles"]
    targets = trajectory["targets"]
    ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--", zorder=1)
    for model_id in MODEL_ORDER:
        error = (trajectory[model_id] - targets) * 100.0
        ax.plot(
            cycles,
            error,
            color=COLORS[model_id],
            linestyle=LINESTYLES[model_id],
            linewidth=1.45,
            label=MODEL_LABELS[model_id],
        )
    ax.set_xlabel("Cycle index")
    ax.set_ylabel("Prediction error (p.p.)")
    polish_axes(ax)
    if panel_label:
        add_panel_label(ax, panel_label)
    if panel_title:
        add_panel_title(ax, panel_title)


def create_trajectory_figures(
    runs: Mapping[str, Mapping[str, object]],
    output_dir: Path,
    *,
    paired_seed_scores: Mapping[str, float] | None = None,
) -> tuple[list[Path], Dict[str, object]]:
    battery_id, battery_metrics = select_cross_model_median_battery(runs)
    trajectory = extract_selected_trajectory(runs, battery_id)
    saved: list[Path] = []

    fig, axes = plt.subplots(2, 1, figsize=(6.15, 6.65), sharex=True)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.09, top=0.84, hspace=0.43)
    draw_trajectory_panel(axes[0], trajectory, panel_label="a", panel_title="Representative unseen-cell trajectory")
    draw_error_panel(axes[1], trajectory, panel_label="b", panel_title="Aligned prediction error")
    axes[0].set_xlabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.985),
        columnspacing=1.25,
        handlelength=2.0,
        fontsize=8.8,
    )
    saved.extend(save_figure(fig, output_dir / "fig4_trajectory_and_error_group"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.35), constrained_layout=True)
    draw_trajectory_panel(ax, trajectory)
    ax.legend(loc="lower left", ncol=1, handlelength=2.5, facecolor="white")
    saved.extend(save_figure(fig, output_dir / "fig4a_representative_trajectory"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.35), constrained_layout=True)
    draw_error_panel(ax, trajectory)
    ax.legend(loc="upper left", ncol=1, handlelength=2.5, facecolor="white")
    saved.extend(save_figure(fig, output_dir / "fig4b_aligned_prediction_error"))
    plt.close(fig)

    first_result = runs[MODEL_ORDER[0]]["result"]
    source = {
        "seed_pair_selection_rule": (
            "four-model mean test MAE closest to the across-pair median"
            if paired_seed_scores is not None
            else "seed pair supplied explicitly"
        ),
        "battery_selection_rule": (
            "cross-model mean test-cell MAE closest to the across-cell median"
        ),
        "selected_battery": battery_id,
        "n_selected_samples": int(len(trajectory["cycles"])),
        "supervision_ratio": first_result["supervision_ratio"],
        "split_seed": first_result["split_seed"],
        "mask_seed": first_result["mask_seed"],
        "training_seed": first_result["training_seed"],
        "paired_seed_cross_model_mean_mae": dict(paired_seed_scores or {}),
        "per_battery_mae": battery_metrics,
        "run_directories": {
            model_id: str(runs[model_id]["run_dir"].resolve()) for model_id in MODEL_ORDER
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fig4_source.json").write_text(
        json.dumps(source, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return saved, source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision_root", type=Path)
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument(
        "--mask-seed",
        type=int,
        default=None,
        help="Optional explicit reproduction override; omit with --training-seed for median-pair selection",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=None,
        help="Optional explicit reproduction override; omit with --mask-seed for median-pair selection",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_paper_style()
    revision_root = args.revision_root.resolve()
    if (args.mask_seed is None) != (args.training_seed is None):
        raise SystemExit("--mask-seed and --training-seed must be supplied together")
    if args.mask_seed is None:
        selected_pair, runs, pair_scores = select_cross_model_median_seed_pair(
            revision_root, ratio=args.ratio, split_seed=args.split_seed
        )
        print(
            f"Selected paired repetition: mask={selected_pair[0]}, "
            f"train={selected_pair[1]}"
        )
    else:
        runs = load_factorial_runs(
            revision_root,
            ratio=args.ratio,
            split_seed=args.split_seed,
            mask_seed=args.mask_seed,
            training_seed=args.training_seed,
        )
        pair_scores = None
    saved, source = create_trajectory_figures(
        runs, args.output_dir.resolve(), paired_seed_scores=pair_scores
    )
    print(f"Selected battery: {source['selected_battery']}")
    print("Generated figures:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
