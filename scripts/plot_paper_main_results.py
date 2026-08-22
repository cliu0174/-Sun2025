"""Generate manuscript main-result and 2x2 ablation figures.

Every panel is rendered twice from the same summary data: once as part of the
combined figure and once as an independent figure.  No child panel is cropped
from a group image.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paper_plot_style import (  # noqa: E402
    COLORS,
    LINESTYLES,
    MARKERS,
    add_panel_label,
    add_panel_title,
    apply_paper_style,
    polish_axes,
    save_figure,
)
from scripts.generate_springer_results import validate_summary  # noqa: E402


MODEL_ORDER = ["single_no_pi", "single_pi", "multi_no_pi", "multi_pi"]
PIPELINE_REVISION = "v3_leakage_free"
EXPECTED_RATIOS = (1.0, 0.7, 0.5, 0.3)
EXPECTED_REPEATS = 3
MODEL_LABELS = {
    "single_no_pi": "Single-scale CNN-LSTM",
    "single_pi": "PI-CNN-LSTM",
    "multi_no_pi": "Multi-scale CNN-LSTM",
    "multi_pi": "PI-MSCL",
}


def load_summary(path: Path, *, allow_incomplete: bool = False) -> Dict:
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("smoke") and not allow_incomplete:
        raise ValueError("refusing to plot a smoke-test summary as a paper figure")
    groups = summary.get("groups", {})
    if not groups:
        raise ValueError("summary contains no aggregated non-smoke result groups")
    if not allow_incomplete:
        if summary.get("pipeline_revision") != PIPELINE_REVISION:
            raise ValueError(
                f"expected pipeline_revision={PIPELINE_REVISION!r}, "
                f"got {summary.get('pipeline_revision')!r}"
            )
        expected_keys = {
            f"{model_id}/r{ratio:.1f}".replace(".", "p")
            for model_id in MODEL_ORDER
            for ratio in EXPECTED_RATIOS
        }
        missing = sorted(expected_keys.difference(groups))
        wrong_repeats = sorted(
            key
            for key in expected_keys.intersection(groups)
            if int(groups[key].get("n", 0)) != EXPECTED_REPEATS
        )
        if missing or wrong_repeats:
            details = []
            if missing:
                details.append("missing groups: " + ", ".join(missing))
            if wrong_repeats:
                details.append(
                    f"groups without exactly {EXPECTED_REPEATS} repeats: "
                    + ", ".join(wrong_repeats)
                )
            raise ValueError(
                "formal paper figure requires the complete paired matrix; "
                + "; ".join(details)
            )
        # Group counts alone cannot prove that the four variants share the
        # same split/mask/training seeds. Reuse the table generator's stricter
        # run-level validation before any submission figure is produced.
        validate_summary(summary, expected_repeats=EXPECTED_REPEATS)
    return summary


def available_ratios(summary: Dict) -> list[float]:
    ratios = {
        float(row["supervision_ratio"])
        for row in summary["groups"].values()
    }
    return sorted(ratios)


def metric_series(
    summary: Dict, model_id: str, metric: str, ratios: Iterable[float]
) -> Tuple[np.ndarray, np.ndarray]:
    means = []
    stds = []
    for ratio in ratios:
        key = f"{model_id}/r{ratio:.1f}".replace(".", "p")
        row = summary["groups"].get(key)
        if row is None:
            means.append(np.nan)
            stds.append(np.nan)
            continue
        means.append(float(row[f"{metric}_mean"]))
        std = row.get(f"{metric}_std")
        stds.append(float(std) if std is not None else 0.0)
    return np.asarray(means), np.asarray(stds)


def draw_budget_panel(
    ax: plt.Axes,
    summary: Dict,
    metric: str,
    ylabel: str,
    *,
    panel_label: str | None = None,
    panel_title: str | None = None,
) -> None:
    ratios = available_ratios(summary)
    for model_id in MODEL_ORDER:
        means, stds = metric_series(summary, model_id, metric, ratios)
        scale = 100.0 if metric in {"test_mae", "test_rmse"} else 1.0
        ax.errorbar(
            ratios,
            means * scale,
            yerr=stds * scale,
            color=COLORS[model_id],
            marker=MARKERS[model_id],
            linestyle=LINESTYLES[model_id],
            linewidth=1.8,
            capsize=3.0,
            capthick=1.0,
            label=MODEL_LABELS[model_id],
        )
    ax.set_xlabel("SOH-label retention ratio, $r$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(ratios)
    ax.set_xlim(min(ratios) - 0.035, max(ratios) + 0.035)
    # Full supervision (r=1.0) reads left-to-right into the sparsest budget
    # (r=0.3), matching the "as labels shrink" direction used in the text.
    ax.invert_xaxis()
    if metric == "test_mae":
        # Reserve a data-free lower-right corner for the in-panel shared legend.
        lower, upper = ax.get_ylim()
        ax.set_ylim(min(lower, 1.42), upper)
    polish_axes(ax)
    if panel_label:
        add_panel_label(ax, panel_label)
    if panel_title:
        add_panel_title(ax, panel_title)


def create_performance_figures(summary: Dict, output_dir: Path) -> list[Path]:
    saved = []
    fig, axes = plt.subplots(2, 1, figsize=(6.15, 6.35))
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.09, top=0.84, hspace=0.50)
    draw_budget_panel(axes[0], summary, "test_mae", "MAE (%)", panel_label="a", panel_title="MAE vs. label budget")
    draw_budget_panel(axes[1], summary, "test_rmse", "RMSE (%)", panel_label="b", panel_title="RMSE vs. label budget")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.985),
        columnspacing=1.3,
        handlelength=2.1,
        fontsize=9.3,
    )
    saved.extend(save_figure(fig, output_dir / "fig3_performance_vs_label_budget_group"))
    plt.close(fig)

    for suffix, metric, ylabel in (
        ("fig3a_mae_vs_label_budget", "test_mae", "MAE (%)"),
        ("fig3b_rmse_vs_label_budget", "test_rmse", "RMSE (%)"),
    ):
        fig, ax = plt.subplots(figsize=(6.5, 4.35), constrained_layout=True)
        draw_budget_panel(ax, summary, metric, ylabel)
        ax.legend(loc="lower right", ncol=1, handlelength=2.5, facecolor="white")
        saved.extend(save_figure(fig, output_dir / suffix))
        plt.close(fig)
    return saved


def _effect_series(
    summary: Dict, lhs: str, rhs: str, ratios: Iterable[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return paired MAE effects as mean and sample SD in percentage points."""

    means, stds = [], []
    formal_runs = [row for row in summary.get("runs", []) if not row.get("smoke", False)]
    for ratio in ratios:
        by_model = {}
        for model_id in (lhs, rhs):
            by_model[model_id] = {
                (
                    int(row["split_seed"]),
                    int(row["mask_seed"]),
                    int(row["training_seed"]),
                ): float(row["test_mae"])
                for row in formal_runs
                if row["model_id"] == model_id
                and abs(float(row["supervision_ratio"]) - ratio) < 1e-12
            }
        if by_model[lhs].keys() != by_model[rhs].keys() or not by_model[lhs]:
            raise ValueError(f"unpaired effect runs for {lhs} versus {rhs} at r={ratio:.1f}")
        values = np.asarray(
            [
                (by_model[lhs][key] - by_model[rhs][key]) * 100.0
                for key in sorted(by_model[lhs])
            ],
            dtype=float,
        )
        means.append(float(values.mean()))
        stds.append(float(values.std(ddof=1)) if values.size > 1 else 0.0)
    return np.asarray(means), np.asarray(stds)


def draw_effect_panel(
    ax: plt.Axes,
    summary: Dict,
    comparisons: Iterable[Tuple[str, str, str, str]],
    *,
    panel_label: str | None = None,
    panel_title: str | None = None,
) -> None:
    ratios = available_ratios(summary)
    for lhs, rhs, label, color_key in comparisons:
        effect, effect_sd = _effect_series(summary, lhs, rhs, ratios)
        ax.errorbar(
            ratios,
            effect,
            yerr=effect_sd,
            color=COLORS[color_key],
            marker=MARKERS[color_key],
            linestyle=LINESTYLES[color_key],
            capsize=3.0,
            capthick=1.0,
            label=label,
        )
    ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    ax.set_xlabel("SOH-label retention ratio, $r$")
    ax.set_ylabel("Paired MAE reduction (p.p.)")
    ax.set_xticks(ratios)
    ax.set_xlim(min(ratios) - 0.035, max(ratios) + 0.035)
    # Match the r=1.0-to-r=0.3 reading direction used in the budget panels.
    ax.invert_xaxis()
    # Preserve an empty upper-right corner for the in-panel legend.
    ax.set_ylim(-0.24, 0.46)
    polish_axes(ax)
    if panel_label:
        add_panel_label(ax, panel_label)
    if panel_title:
        add_panel_title(ax, panel_title)


ARCH_COMPARISONS = (
    ("single_no_pi", "multi_no_pi", "No physics constraint", "multi_no_pi"),
    ("single_pi", "multi_pi", "With physics constraint", "multi_pi"),
)
PHYSICS_COMPARISONS = (
    ("single_no_pi", "single_pi", "Single-scale architecture", "single_pi"),
    ("multi_no_pi", "multi_pi", "Multi-scale architecture", "multi_pi"),
)


def create_ablation_figures(summary: Dict, output_dir: Path) -> list[Path]:
    saved = []
    fig, axes = plt.subplots(2, 1, figsize=(6.15, 6.20))
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.09, top=0.96, hspace=0.56)
    draw_effect_panel(axes[0], summary, ARCH_COMPARISONS, panel_label="a", panel_title="Architecture effect, $\\Delta_{\\mathrm{MS}}$")
    draw_effect_panel(axes[1], summary, PHYSICS_COMPARISONS, panel_label="b", panel_title="Constraint effect, $\\Delta_{\\mathrm{PI}}$")
    axes[0].legend(loc="upper left", ncol=2, handlelength=2.1, columnspacing=1.1, fontsize=9.2)
    axes[1].legend(loc="upper left", ncol=2, handlelength=2.1, columnspacing=1.1, fontsize=9.2)
    saved.extend(save_figure(fig, output_dir / "fig5_ablation_effects_group"))
    plt.close(fig)

    for stem, comparisons in (
        (
            "fig5a_multiscale_effect",
            ARCH_COMPARISONS,
        ),
        (
            "fig5b_physics_effect",
            PHYSICS_COMPARISONS,
        ),
    ):
        fig, ax = plt.subplots(figsize=(6.5, 4.35), constrained_layout=True)
        draw_effect_panel(ax, summary, comparisons)
        ax.legend(loc="upper right", ncol=1, facecolor="white")
        saved.extend(save_figure(fig, output_dir / stem))
        plt.close(fig)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Development only: allow fewer than two repeats; never use for submission",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_paper_style()
    summary = load_summary(args.summary, allow_incomplete=args.allow_incomplete)
    saved = []
    saved.extend(create_performance_figures(summary, args.output_dir))
    saved.extend(create_ablation_figures(summary, args.output_dir))
    print("Generated figures:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
