"""Render the complete HUST discharge-capacity trajectory overview for the manuscript.

The HUST CSV release used in this study contains one discharge-capacity value per
cycle.  This script therefore visualizes all 77 normalized discharge-capacity
trajectories, rather than implying access to raw voltage-time discharge traces.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.paper_plot_style import apply_paper_style, polish_axes, save_figure


DATA_DIR = ROOT / "data" / "HUST data"
OUTPUT_DIR = ROOT / "output" / "figures" / "formal_main_v3"
OVERLEAF_FIGURE = ROOT / "overleaf_submission_20260813" / "Figure2.png"

# A restrained categorical palette makes overlapping cell trajectories easier to
# trace without implying a ranked or continuous battery attribute.  No legend is
# used because colour has no scientific category in this overview figure.
TRAJECTORY_COLORS = (
    "#3B75AF",  # blue
    "#D4863A",  # orange
    "#4D9A76",  # green
    "#A96578",  # muted red
    "#7B6AAE",  # purple
    "#4E9B9D",  # teal
    "#B07A58",  # brown
    "#6E8EAF",  # slate blue
    "#A88A4E",  # ochre
    "#6E6E6E",  # neutral grey
)


def battery_sort_key(path: Path) -> tuple[int, int]:
    return tuple(int(part) for part in path.stem.split("-"))


def load_curves() -> dict[str, np.ndarray]:
    curves: dict[str, np.ndarray] = {}
    for csv_path in sorted(DATA_DIR.glob("*.csv"), key=battery_sort_key):
        capacity = pd.read_csv(csv_path, usecols=["capacity"])["capacity"].to_numpy(dtype=float)
        capacity = capacity[np.isfinite(capacity)]
        if capacity.size == 0:
            continue
        reference = float(np.mean(capacity[: min(20, capacity.size)]))
        curves[csv_path.stem] = capacity / reference * 100.0
    if len(curves) != 77:
        raise ValueError(f"expected 77 HUST cells, found {len(curves)}")
    return curves


def draw_overview(ax: plt.Axes, curves: dict[str, np.ndarray]) -> None:
    for index, (_, soh) in enumerate(curves.items()):
        cycles = np.arange(soh.size)
        ax.plot(
            cycles,
            soh,
            color=TRAJECTORY_COLORS[index % len(TRAJECTORY_COLORS)],
            linewidth=0.82,
            alpha=0.45,
            zorder=2,
        )
    ax.set_xlabel("Cycle index")
    ax.set_ylabel("Normalized discharge capacity (%)")
    # Retain the full observed lifetime on the horizontal axis, while cropping
    # the low-capacity tail below 75% to focus this overview on the shared
    # degradation region.
    max_cycle = max(soh.size - 1 for soh in curves.values())
    ax.set_xlim(0, max_cycle)
    ax.set_ylim(75, 102)
    ax.set_xticks(np.arange(0, max_cycle + 1, 500))
    ax.set_yticks(np.arange(75, 101, 5))
    polish_axes(ax, grid_axis="both")


def main() -> None:
    apply_paper_style()
    curves = load_curves()
    fig, ax = plt.subplots(figsize=(7.2, 4.35), constrained_layout=True)
    draw_overview(ax, curves)
    save_figure(fig, OUTPUT_DIR / "fig2_hust_all_discharge_capacity")
    fig.savefig(OVERLEAF_FIGURE, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    metadata = {
        "figure": "all HUST normalized discharge-capacity trajectories",
        "n_cells": len(curves),
        "unit": "cycle-level discharge capacity normalized by mean of first 20 available cycles",
        "source": "public HUST per-cycle CSV release",
        "battery_ids": list(curves),
        "palette": list(TRAJECTORY_COLORS),
        "legend": "omitted because colour only distinguishes overlapping trajectories",
    }
    (OUTPUT_DIR / "fig2_hust_all_discharge_capacity_source.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
