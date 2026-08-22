"""Shared plotting style and export helpers for the SOH manuscript."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt


# Restrained, colour-blind-friendly palette used consistently across the paper.
COLORS = {
    "single_no_pi": "#4C78A8",
    "single_pi": "#72A0C1",
    "multi_no_pi": "#E08B3E",
    "multi_pi": "#C44E52",
    "truth": "#222222",
    "unlabeled": "#B8B8B8",
}

MARKERS = {
    "single_no_pi": "o",
    "single_pi": "s",
    "multi_no_pi": "^",
    "multi_pi": "D",
}

LINESTYLES = {
    "single_no_pi": "--",
    "single_pi": "-.",
    "multi_no_pi": ":",
    "multi_pi": "-",
}


def apply_paper_style() -> None:
    """Apply the manuscript-wide Matplotlib style."""

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11.5,
            "axes.labelsize": 13.5,
            "axes.labelweight": "bold",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11,
            "axes.linewidth": 1.15,
            "lines.linewidth": 1.8,
            "lines.markersize": 6.0,
            "grid.alpha": 0.32,
            "grid.linewidth": 0.65,
            "grid.linestyle": "--",
            # Legends must remain lightweight in the manuscript: rely on
            # whitespace rather than a surrounding rectangular frame.
            "legend.frameon": False,
            "legend.fancybox": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.unicode_minus": False,
        }
    )


def polish_axes(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    """Apply consistent spines, ticks, and grid to one axis."""

    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.15)
        spine.set_color("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=1.0, length=4.0, color="#333333")
    for tick_label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick_label.set_fontweight("semibold")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    """Place a small, unboxed panel label just outside an axes' upper-left."""

    ax.text(
        -0.075,
        1.045,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        clip_on=False,
    )


def add_panel_title(ax: plt.Axes, title: str) -> None:
    """Place a short, left-aligned descriptive subtitle above one panel.

    Used alongside ``add_panel_label`` so every multi-panel group figure in
    the manuscript states, in a few words, what each lettered panel shows
    without relying on the reader cross-referencing the caption text.
    """

    ax.set_title(title, loc="left", fontsize=10.5, fontweight="normal", pad=8.0, color="#333333")


def save_figure(
    fig: plt.Figure,
    output_stem: Path | str,
    *,
    formats: Iterable[str] = ("png", "tiff", "pdf", "svg"),
    dpi: int = 600,
) -> list[Path]:
    """Export one figure in independent raster and vector formats."""

    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        suffix = fmt.lower().lstrip(".")
        path = stem.with_suffix(f".{suffix}")
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if suffix in {"png", "jpg", "jpeg", "tif", "tiff"}:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        saved.append(path)
    return saved
