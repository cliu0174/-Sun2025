"""Create the Chapter 4.3 per-cell error-distribution figure from real 20% runs.

The figure is a single-seed screening visualisation.  Each point is the MAE or
RMSE aggregated over all test windows from one held-out battery; it is never a
window-level pseudo-replicate.  No simulated values are used.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "v5_literature_five_model_comparison"
OUTPUT_DIR = ROOT / "figures"

MODELS = (
    ("xgboost", "XGBoost", "#8B95A1"),
    ("gru", "GRU", "#4C78A8"),
    ("transformer_soh", "Transformer-SOH", "#7A9E57"),
    ("cnn_bigru_attention", "CNN–BiGRU–Attention", "#B279A2"),
    ("pi_mscl", "PI-MSCL", "#E07A5F"),
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8,
            "axes.labelsize": 13,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.85,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def prediction_file(model_id: str) -> Path:
    runs = list((RESULT_ROOT / model_id / "r0p2").glob(f"{model_id}_r0p2_split42_mask3262_train2262/predictions.npz"))
    if len(runs) != 1:
        raise FileNotFoundError(f"Expected exactly one completed 20% prediction file for {model_id}; found {len(runs)}.")
    return runs[0]


def calculate_per_cell_errors(path: Path) -> dict[str, tuple[float, float]]:
    archive = np.load(path)
    prediction = np.asarray(archive["predictions"], dtype=float).reshape(-1)
    target = np.asarray(archive["targets"], dtype=float).reshape(-1)
    battery_ids = np.asarray(archive["battery_ids"], dtype=str).reshape(-1)
    if not (prediction.shape == target.shape == battery_ids.shape):
        raise ValueError(f"Inconsistent prediction archive shapes in {path}.")
    errors: dict[str, tuple[float, float]] = {}
    for battery_id in np.unique(battery_ids):
        residual = prediction[battery_ids == battery_id] - target[battery_ids == battery_id]
        errors[str(battery_id)] = (100 * float(np.mean(np.abs(residual))), 100 * float(np.sqrt(np.mean(residual**2))))
    return errors


def write_source_data(records: dict[str, dict[str, tuple[float, float]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "figure5_chapter4_3_per_cell_error_source_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "battery_id", "mae_percent", "rmse_percent"])
        for _, model_label, _ in MODELS:
            for battery_id, (mae, rmse) in sorted(records[model_label].items()):
                writer.writerow([model_label, battery_id, f"{mae:.8f}", f"{rmse:.8f}"])


def draw_box_panel(
    axis: plt.Axes,
    values: list[np.ndarray],
    metric: str,
    letter: str,
) -> None:
    positions = np.arange(1, len(MODELS) + 1)
    box = axis.boxplot(
        values,
        positions=positions,
        widths=0.62,
        vert=False,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#212529", "linewidth": 1.35},
        whiskerprops={"color": "#495057", "linewidth": 0.9},
        capprops={"color": "#495057", "linewidth": 0.9},
    )
    rng = np.random.default_rng(20260820)
    for index, (patch, (_, _, color)) in enumerate(zip(box["boxes"], MODELS)):
        patch.set_facecolor(color)
        patch.set_alpha(0.62)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.0)
        jitter = rng.uniform(-0.11, 0.11, size=len(values[index]))
        axis.scatter(
            values[index],
            np.full(len(values[index]), positions[index]) + jitter,
            s=16,
            color=color,
            edgecolors="white",
            linewidths=0.35,
            alpha=0.9,
            zorder=3,
        )
    display_labels = ["XGBoost", "GRU", "Transformer", "CNN–BiGRU\nAttention", "PI-MSCL"]
    axis.set_yticks(positions, display_labels)
    axis.invert_yaxis()
    axis.set_xlabel(f"Per-cell {metric} (%)", fontsize=13, fontweight="bold", labelpad=7)
    axis.grid(axis="x", color="#D0D5DB", linewidth=0.65, zorder=0)
    axis.set_axisbelow(True)
    axis.set_xlim(left=0)
    axis.tick_params(axis="both", which="major", width=1.0, length=4.8)
    for tick in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        tick.set_fontweight("bold")
    axis.text(-0.12, 1.025, letter, transform=axis.transAxes, fontsize=11, fontweight="bold")


def save_all(figure: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf"):
        figure.savefig(OUTPUT_DIR / f"{stem}.{suffix}", bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / f"{stem}.tiff", dpi=600, bbox_inches="tight")


def main() -> None:
    configure_style()
    records: dict[str, dict[str, tuple[float, float]]] = {}
    expected_batteries: set[str] | None = None
    for model_id, model_label, _ in MODELS:
        errors = calculate_per_cell_errors(prediction_file(model_id))
        batteries = set(errors)
        if expected_batteries is None:
            expected_batteries = batteries
        elif batteries != expected_batteries:
            raise RuntimeError(f"Held-out battery set differs for {model_label}.")
        records[model_label] = errors
    write_source_data(records)

    mae_values = [np.array([value[0] for value in records[label].values()]) for _, label, _ in MODELS]
    rmse_values = [np.array([value[1] for value in records[label].values()]) for _, label, _ in MODELS]
    figure, axes = plt.subplots(2, 1, figsize=(5.45, 5.6))
    draw_box_panel(axes[0], mae_values, "MAE", "a")
    draw_box_panel(axes[1], rmse_values, "RMSE", "b")
    figure.tight_layout(pad=0.9, h_pad=1.3)
    save_all(figure, "figure5_chapter4_3_per_cell_error_boxplots_20pct")
    plt.close(figure)
    print(f"Wrote figure bundle and source data to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
