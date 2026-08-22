"""Render traceable candidate figures from the Chinese manuscript's legacy tables.

Figure contract
---------------
Core conclusion:
    The legacy point estimates report a comparatively flat PI-MSCL MAE curve as
    the retained-label ratio decreases, while the legacy 2x2 RMSE matrix reports
    the lowest error for the multi-scale physics-informed configuration.

Evidence chain:
    Panel a uses Table 4 only (four-method MAE trend).
    Panel b uses Table 3 only (architecture x physics RMSE trend).

Review boundary:
    These are legacy point estimates with no recoverable per-seed archive.  The
    script therefore draws no error bars and makes no significance claim.  It
    never combines these values with v3_leakage_free results.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "data" / "paper_results" / "legacy_cn_tables.csv"
SOURCE_META = ROOT / "data" / "paper_results" / "legacy_cn_tables_metadata.json"
OUTPUT_DIR = ROOT / "output" / "figures" / "legacy_cn"

RATIOS = [1.0, 0.7, 0.5, 0.3]
RATIO_LABELS = ["1.0", "0.7", "0.5", "0.3"]

COLORS = {
    "XGBoost": "#8C8C8C",
    "LSTM": "#7A9CC6",
    "CNN-LSTM": "#3D6F9E",
    "PI-MSCL": "#C6534C",
    "Single-scale / no physics": "#5D7896",
    "Single-scale / physics": "#91AEC7",
    "Multi-scale / no physics": "#D59A5A",
    "Multi-scale / physics": "#C6534C",
}

MARKERS = {
    "XGBoost": "o",
    "LSTM": "s",
    "CNN-LSTM": "^",
    "PI-MSCL": "D",
    "Single-scale / no physics": "o",
    "Single-scale / physics": "s",
    "Multi-scale / no physics": "^",
    "Multi-scale / physics": "D",
}

LINESTYLES = {
    "XGBoost": (0, (5, 2)),
    "LSTM": (0, (3, 1, 1, 1)),
    "CNN-LSTM": (0, (2, 1)),
    "PI-MSCL": "-",
    "Single-scale / no physics": (0, (5, 2)),
    "Single-scale / physics": (0, (3, 1, 1, 1)),
    "Multi-scale / no physics": (0, (2, 1)),
    "Multi-scale / physics": "-",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 10.5,
            "axes.labelsize": 13.0,
            "axes.labelweight": "bold",
            "axes.titlesize": 12.0,
            "axes.titleweight": "bold",
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "legend.fontsize": 10.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.45,
            "lines.markersize": 4.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_rows() -> list[dict[str, object]]:
    with SOURCE_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label_ratio"] = float(row["label_ratio"])
        row["value"] = float(row["value"])
    return rows


def select_series(
    rows: list[dict[str, object]], table_id: str, metric: str
) -> dict[str, list[float]]:
    selected: dict[str, dict[float, float]] = defaultdict(dict)
    for row in rows:
        if row["table_id"] == table_id and row["metric"] == metric:
            selected[str(row["method"])][float(row["label_ratio"])] = float(row["value"])
    series = {}
    for method, values in selected.items():
        missing = [ratio for ratio in RATIOS if ratio not in values]
        if missing:
            raise ValueError(f"{table_id}/{metric}/{method} missing ratios: {missing}")
        series[method] = [values[ratio] for ratio in RATIOS]
    return series


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.55, linestyle=(0, (2, 2)))
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", width=0.75, length=3.0, color="#333333")
    for tick_label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick_label.set_fontweight("semibold")
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#333333")
        ax.spines[side].set_linewidth(0.8)


def draw_panel_a(ax: plt.Axes, series: dict[str, list[float]]) -> None:
    order = ["XGBoost", "LSTM", "CNN-LSTM", "PI-MSCL"]
    for method in order:
        ax.plot(
            RATIOS,
            series[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            markerfacecolor="white" if method != "PI-MSCL" else COLORS[method],
            markeredgewidth=0.9,
            label=method,
            zorder=3,
        )
    ax.set_xticks(RATIOS, RATIO_LABELS)
    ax.set_xlim(1.04, 0.26)
    ax.set_ylim(1.02, 1.33)
    ax.set_xlabel("Label-retention ratio, $r$")
    ax.set_ylabel("MAE (%)")
    ax.legend(loc="upper left", ncol=1, handlelength=2.5)
    style_axis(ax)


def draw_panel_b(ax: plt.Axes, series: dict[str, list[float]]) -> None:
    order = [
        "Single-scale / no physics",
        "Single-scale / physics",
        "Multi-scale / no physics",
        "Multi-scale / physics",
    ]
    labels = {
        "Single-scale / no physics": "Single, no PI",
        "Single-scale / physics": "Single, PI",
        "Multi-scale / no physics": "Multi, no PI",
        "Multi-scale / physics": "Multi, PI",
    }
    for method in order:
        ax.plot(
            RATIOS,
            series[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            markerfacecolor="white" if method != "Multi-scale / physics" else COLORS[method],
            markeredgewidth=0.9,
            label=labels[method],
            zorder=3,
        )
    ax.set_xticks(RATIOS, RATIO_LABELS)
    ax.set_xlim(1.04, 0.26)
    ax.set_ylim(1.60, 2.48)
    ax.set_xlabel("Label-retention ratio, $r$")
    ax.set_ylabel("RMSE (%)")
    ax.legend(loc="upper left", ncol=1, handlelength=2.5)
    style_axis(ax)


def export(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "tiff", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        kwargs: dict[str, object] = {"bbox_inches": "tight", "pad_inches": 0.035}
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        outputs.append(path)
    return outputs


def image_record(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        return {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "pixels": [image.width, image.height],
            "dpi": [round(float(x), 2) for x in image.info.get("dpi", (0, 0))],
            "mode": image.mode,
        }


def main() -> None:
    configure_style()
    rows = load_rows()
    mae = select_series(rows, "Table4", "MAE")
    rmse = select_series(rows, "Table3", "RMSE")

    fig, axes = plt.subplots(1, 2, figsize=(8.20, 4.00), constrained_layout=True)
    draw_panel_a(axes[0], mae)
    draw_panel_b(axes[1], rmse)
    axes[0].text(-0.15, 1.045, "a", transform=axes[0].transAxes, fontsize=9, fontweight="bold")
    axes[1].text(-0.15, 1.045, "b", transform=axes[1].transAxes, fontsize=9, fontweight="bold")
    group_outputs = export(fig, OUTPUT_DIR / "legacy_cn_budget_results_group")
    plt.close(fig)

    fig_a, ax_a = plt.subplots(figsize=(6.50, 4.35), constrained_layout=True)
    draw_panel_a(ax_a, mae)
    panel_a_outputs = export(fig_a, OUTPUT_DIR / "legacy_cn_budget_mae_panel_a")
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(6.50, 4.35), constrained_layout=True)
    draw_panel_b(ax_b, rmse)
    panel_b_outputs = export(fig_b, OUTPUT_DIR / "legacy_cn_budget_rmse_panel_b")
    plt.close(fig_b)

    with SOURCE_META.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    manifest = {
        "figure_status": "legacy candidate; not a v3_leakage_free main result",
        "core_conclusion": (
            "Legacy point estimates report a comparatively flat PI-MSCL MAE curve "
            "and the lowest Table-3 RMSE for the multi-scale physics-informed model."
        ),
        "statistical_boundary": "Point estimates only; no error bars or significance inference.",
        "source_csv": str(SOURCE_CSV.relative_to(ROOT)).replace("\\", "/"),
        "source_metadata": str(SOURCE_META.relative_to(ROOT)).replace("\\", "/"),
        "source_role": metadata["role"],
        "panels": {
            "a": "Chinese manuscript Table 4 MAE point estimates.",
            "b": "Chinese manuscript Table 3 RMSE point estimates."
        },
        "exports": [
            str(path.relative_to(ROOT)).replace("\\", "/")
            for path in group_outputs + panel_a_outputs + panel_b_outputs
        ],
        "raster_qa": [
            image_record(path)
            for path in group_outputs + panel_a_outputs + panel_b_outputs
            if path.suffix.lower() in {".png", ".tiff"}
        ],
    }
    manifest_path = OUTPUT_DIR / "legacy_cn_budget_results_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
