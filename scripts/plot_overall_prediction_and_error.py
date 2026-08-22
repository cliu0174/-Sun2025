"""Plot full-supervision prediction agreement and per-cell error levels.

The figure is generated entirely from archived, non-smoke predictions.  Panel
(a) uses the median-MAE PI-MSCL repetition under full supervision.  Panel (b)
computes one MAE per unseen test cell and model after averaging the cell-level
MAE across the three paired repetitions.  Group and child panels are rendered
independently; no panel is cropped from the composite figure.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paper_plot_style import (  # noqa: E402
    add_panel_label,
    add_panel_title,
    apply_paper_style,
    polish_axes,
    save_figure,
)


MAIN_ROOT = PROJECT_ROOT / "experiments/paper_main_results/fixed_split/v3_leakage_free"
BASELINE_ROOT = PROJECT_ROOT / "experiments/paper_baselines/fixed_split/v1_leakage_free_endpoints"
OUTPUT_ROOT = PROJECT_ROOT / "output/figures/formal_main_v3"

MODEL_ORDER = [
    "xgboost",
    "lstm",
    "gru",
    "attn_cnn_lstm",
    "single_no_pi",
    "single_pi",
    "multi_no_pi",
    "multi_pi",
]
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "gru": "GRU",
    "attn_cnn_lstm": "Attn. CNN-LSTM",
    "single_no_pi": "CNN-LSTM",
    "single_pi": "PI-CNN-LSTM",
    "multi_no_pi": "MS-CNN-LSTM",
    "multi_pi": "PI-MSCL",
}
MODEL_COLORS = {
    "xgboost": "#8C8C8C",
    "lstm": "#6F7897",
    "gru": "#8795B8",
    "attn_cnn_lstm": "#A8B6D5",
    "single_no_pi": "#4C78A8",
    "single_pi": "#72A0C1",
    "multi_no_pi": "#E08B3E",
    "multi_pi": "#C44E52",
}
SEED_PAIRS = [(1007, 7), (1929, 929), (3262, 2262)]


def run_dir(model_id: str, mask_seed: int, training_seed: int) -> Path:
    tag = f"split42_mask{mask_seed}_train{training_seed}"
    if model_id in {"xgboost", "lstm", "gru", "attn_cnn_lstm"}:
        return BASELINE_ROOT / model_id / "r1p0" / tag
    return MAIN_ROOT / model_id / "r1p0" / tag


def load_run(model_id: str, mask_seed: int, training_seed: int) -> Dict[str, object]:
    directory = run_dir(model_id, mask_seed, training_seed)
    result_path = directory / "result.json"
    prediction_path = directory / "predictions.npz"
    if not result_path.exists() or not prediction_path.exists():
        raise FileNotFoundError(f"incomplete archived run: {directory}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("smoke"):
        raise ValueError(f"refusing to plot smoke run: {directory}")
    expected = {
        "supervision_ratio": 1.0,
        "split_seed": 42,
        "mask_seed": mask_seed,
        "training_seed": training_seed,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"metadata mismatch for {key} at {directory}")
    with np.load(prediction_path, allow_pickle=False) as archive:
        required = {"predictions", "targets", "battery_ids", "cycle_indices"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"missing arrays at {prediction_path}: {sorted(missing)}")
        arrays = {key: np.asarray(archive[key]) for key in required}
    if len({len(value) for value in arrays.values()}) != 1:
        raise ValueError(f"inconsistent array lengths at {prediction_path}")
    return {"directory": directory, "result": result, "arrays": arrays}


def load_all_runs() -> Dict[str, list[Dict[str, object]]]:
    runs = {
        model_id: [load_run(model_id, mask_seed, training_seed) for mask_seed, training_seed in SEED_PAIRS]
        for model_id in MODEL_ORDER
    }
    reference = runs[MODEL_ORDER[0]][0]["arrays"]
    for model_id in MODEL_ORDER:
        for run in runs[model_id]:
            for key in ("targets", "battery_ids", "cycle_indices"):
                if not np.array_equal(reference[key], run["arrays"][key]):
                    raise ValueError(f"unaligned {key} for {model_id}: {run['directory']}")
    return runs


def select_median_pimscl_run(runs: Dict[str, list[Dict[str, object]]]) -> Dict[str, object]:
    ordered = sorted(runs["multi_pi"], key=lambda run: float(run["result"]["test_mae"]))
    return ordered[len(ordered) // 2]


def per_cell_mae(runs: Dict[str, list[Dict[str, object]]]) -> tuple[list[str], Dict[str, np.ndarray]]:
    reference = runs[MODEL_ORDER[0]][0]["arrays"]
    battery_ids = np.asarray(reference["battery_ids"], dtype=str)
    batteries = sorted(np.unique(battery_ids).tolist())
    values: Dict[str, np.ndarray] = {}
    for model_id in MODEL_ORDER:
        model_values = []
        for battery_id in batteries:
            mask = battery_ids == battery_id
            repetition_mae = []
            for run in runs[model_id]:
                predictions = np.asarray(run["arrays"]["predictions"], dtype=float)
                targets = np.asarray(run["arrays"]["targets"], dtype=float)
                repetition_mae.append(float(np.mean(np.abs(predictions[mask] - targets[mask])) * 100.0))
            model_values.append(float(np.mean(repetition_mae)))
        values[model_id] = np.asarray(model_values, dtype=float)
    return batteries, values


def draw_parity(
    ax: plt.Axes,
    run: Dict[str, object],
    panel_label: str | None = None,
    panel_title: str | None = None,
) -> None:
    arrays = run["arrays"]
    targets = np.asarray(arrays["targets"], dtype=float) * 100.0
    predictions = np.asarray(arrays["predictions"], dtype=float) * 100.0
    lo = float(min(targets.min(), predictions.min()))
    hi = float(max(targets.max(), predictions.max()))
    pad = 0.015 * (hi - lo)
    hb = ax.hexbin(
        targets,
        predictions,
        gridsize=52,
        mincnt=1,
        cmap="Blues",
        linewidths=0,
        bins="log",
    )
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#333333", linestyle="--", linewidth=1.4)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Measured SOH (%)")
    ax.set_ylabel("Predicted SOH (%)")
    polish_axes(ax)
    if panel_label:
        add_panel_label(ax, panel_label)
    if panel_title:
        add_panel_title(ax, panel_title)
    return hb


def draw_cell_errors(
    ax: plt.Axes,
    values: Dict[str, np.ndarray],
    panel_label: str | None = None,
    panel_title: str | None = None,
) -> None:
    positions = np.arange(1, len(MODEL_ORDER) + 1)
    series = [values[model_id] for model_id in MODEL_ORDER]
    box = ax.boxplot(
        series,
        vert=False,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.4},
        whiskerprops={"color": "#555555", "linewidth": 1.0},
        capprops={"color": "#555555", "linewidth": 1.0},
    )
    for patch, model_id in zip(box["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model_id])
        patch.set_edgecolor("#333333")
        patch.set_alpha(0.72)
    for index, model_id in enumerate(MODEL_ORDER, start=1):
        y_offsets = np.linspace(-0.16, 0.16, len(values[model_id]))
        ax.scatter(
            values[model_id],
            index + y_offsets,
            s=14,
            color=MODEL_COLORS[model_id],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.82,
            zorder=3,
        )
    ax.set_yticks(positions)
    ax.set_yticklabels([MODEL_LABELS[model_id] for model_id in MODEL_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Per-cell MAE (%)")
    ax.set_ylabel("")
    polish_axes(ax, grid_axis="x")
    if panel_label:
        add_panel_label(ax, panel_label)
    if panel_title:
        add_panel_title(ax, panel_title)


def write_sources(
    selected_run: Dict[str, object],
    batteries: Iterable[str],
    values: Dict[str, np.ndarray],
) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_ROOT / "fig2_per_cell_mae_source.csv"
    batteries = list(batteries)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["battery_id", *MODEL_ORDER])
        for row_index, battery_id in enumerate(batteries):
            writer.writerow([battery_id, *[f"{values[model_id][row_index]:.10f}" for model_id in MODEL_ORDER]])
    result = selected_run["result"]
    manifest = {
        "figure": "overall full-supervision prediction agreement and per-cell MAE",
        "split_seed": 42,
        "supervision_ratio": 1.0,
        "n_test_cells": len(batteries),
        "paired_repetitions": [{"mask_seed": m, "training_seed": t} for m, t in SEED_PAIRS],
        "parity_panel": {
            "model_id": "multi_pi",
            "selection_rule": "median test MAE among the three formal PI-MSCL repetitions",
            "run_directory": str(selected_run["directory"].relative_to(PROJECT_ROOT)),
            "mask_seed": int(result["mask_seed"]),
            "training_seed": int(result["training_seed"]),
            "test_mae": float(result["test_mae"]),
            "test_rmse": float(result["test_rmse"]),
            "test_r2": float(result["test_r2"]),
        },
        "error_panel": {
            "unit": "one point per unseen test cell and model",
            "aggregation": "cell-level MAE averaged across the three paired repetitions",
            "source_csv": csv_path.name,
        },
    }
    (OUTPUT_ROOT / "fig2_source.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    apply_paper_style()
    runs = load_all_runs()
    selected = select_median_pimscl_run(runs)
    batteries, values = per_cell_mae(runs)
    write_sources(selected, batteries, values)

    fig = plt.figure(figsize=(6.15, 7.05))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.0, 1.06), hspace=0.52)
    axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.09, top=0.97)
    draw_parity(axes[0], selected, panel_label="a", panel_title="Prediction agreement, full supervision")
    draw_cell_errors(axes[1], values, panel_label="b", panel_title="Per-cell error distribution")
    save_figure(fig, OUTPUT_ROOT / "fig2_overall_prediction_and_error_group")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.7, 5.2), constrained_layout=True)
    draw_parity(ax, selected)
    save_figure(fig, OUTPUT_ROOT / "fig2a_overall_prediction_agreement")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.9), constrained_layout=True)
    draw_cell_errors(ax, values)
    save_figure(fig, OUTPUT_ROOT / "fig2b_per_cell_mae_distribution")
    plt.close(fig)


if __name__ == "__main__":
    main()
