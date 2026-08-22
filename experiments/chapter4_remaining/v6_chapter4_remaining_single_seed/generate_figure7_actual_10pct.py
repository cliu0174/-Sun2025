"""Render empirical Figure 7 from completed 10% factorial-ablation archives.

This script reads saved prediction arrays only.  It does not retrain models or
modify experimental results.  The four curves are the actual outputs of the
same split/mask/seed from the completed factorial-ablation_r0p1 run.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "factorial_ablation_r0p1"
OUTPUT_DIR = ROOT / "figure7_actual_10pct"
RUN = "split42_mask3262_train2262"
CELLS = ("4-1", "4-2", "10-6")
METHODS = (
    ("True SOH", None, "#2B2F33", 1.80),
    ("CNN-LSTM", "ms0_mono0_rate0", "#8C6D62", 1.15),
    ("MS-CNN-LSTM", "ms1_mono0_rate0", "#4C78A8", 1.15),
    ("MS+Mono", "ms1_mono1_rate0", "#7A9E57", 1.15),
    ("PI-MSCL", "ms1_mono1_rate1", "#D66B4A", 1.45),
)


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 9,
        "axes.labelsize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.85,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def read_predictions(variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = ARCHIVE / variant / RUN / "predictions.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    return (
        data["predictions"].astype(float),
        data["targets"].astype(float),
        data["battery_ids"].astype(str),
        data["cycle_indices"].astype(int),
    )


def collect_traces() -> dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]:
    loaded = {name: read_predictions(variant) for name, variant, _, _ in METHODS if variant}
    reference_name = "PI-MSCL"
    _, targets, battery_ids, cycles = loaded[reference_name]
    result: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for cell in CELLS:
        selected = battery_ids == cell
        ordered = np.argsort(cycles[selected])
        cycle = cycles[selected][ordered]
        traces = {"True SOH": targets[selected][ordered] * 100.0}
        reference_keys = list(zip(battery_ids[selected][ordered], cycle))
        for name, _, _, _ in METHODS:
            if name == "True SOH":
                continue
            prediction, other_targets, other_ids, other_cycles = loaded[name]
            by_key = {
                (battery, int(cycle_index)): (float(prediction[index]), float(other_targets[index]))
                for index, (battery, cycle_index) in enumerate(zip(other_ids, other_cycles))
            }
            if any(key not in by_key for key in reference_keys):
                raise RuntimeError(f"{name} does not share the full test trajectory for {cell}.")
            values = [by_key[key] for key in reference_keys]
            if not np.allclose([item[1] for item in values], traces["True SOH"] / 100.0):
                raise RuntimeError(f"Target mismatch between {name} and {reference_name} for {cell}.")
            traces[name] = np.asarray([item[0] for item in values]) * 100.0
        result[cell] = cycle, traces
    return result


def draw_cell(axis: plt.Axes, cell: str, data: tuple[np.ndarray, dict[str, np.ndarray]], letter: str) -> None:
    cycle, traces = data
    for name, _, color, linewidth in METHODS:
        axis.plot(cycle, traces[name], color=color, linewidth=linewidth, label=name)
    axis.set_xlabel("Cycle", fontweight="bold", labelpad=5)
    axis.set_ylabel("SOH (%)", fontweight="bold", labelpad=5)
    axis.grid(color="#D0D5DB", linewidth=0.65, zorder=0)
    axis.set_axisbelow(True)
    axis.tick_params(width=0.9, length=4.2)
    for tick in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        tick.set_fontweight("bold")
    axis.text(-0.13, 1.03, letter, transform=axis.transAxes, fontsize=11, fontweight="bold")
    axis.legend(loc="upper right", fontsize=9.0, handlelength=1.9, labelspacing=0.23)

    start = int(0.56 * len(cycle))
    stop = min(len(cycle) - 1, start + max(36, len(cycle) // 7))
    inset = inset_axes(
        axis, width="43%", height="37%", loc="lower left", borderpad=0,
        bbox_to_anchor=(0.08, 0.08, 0.88, 0.88), bbox_transform=axis.transAxes,
    )
    zoom_values = []
    for name, _, color, linewidth in METHODS:
        values = traces[name][start:stop]
        inset.plot(cycle[start:stop], values, color=color, linewidth=max(0.85, linewidth * 0.72))
        zoom_values.append(values)
    zoom_low = float(np.min(zoom_values)) - 0.08
    zoom_high = float(np.max(zoom_values)) + 0.08
    inset.set_ylim(zoom_low, zoom_high)
    inset.grid(color="#E0E3E6", linewidth=0.45)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.70)
        spine.set_color("#000000")
    axis.add_patch(Rectangle(
        (cycle[start], zoom_low), cycle[stop - 1] - cycle[start], zoom_high - zoom_low,
        fill=False, edgecolor="#D96545", linewidth=0.70, zorder=4,
    ))
    axis.add_artist(ConnectionPatch(
        xyA=(cycle[start], zoom_low), coordsA=axis.transData,
        xyB=(0.90, 1.00), coordsB=inset.transAxes,
        arrowstyle="-|>", mutation_scale=10, shrinkA=0, shrinkB=0,
        linewidth=0.70, color="#D96545", connectionstyle="arc3,rad=0.12",
        clip_on=False, zorder=10,
    ))


def save(figure: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "pdf"):
        figure.savefig(OUTPUT_DIR / f"{stem}.{suffix}", bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / f"{stem}.tiff", dpi=600, bbox_inches="tight")


def write_source_csv(data: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]) -> None:
    with (OUTPUT_DIR / "figure7_actual_10pct_source_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["battery_id", "cycle", "method", "soh_percent", "status"])
        for cell, (cycle, traces) in data.items():
            for name, _, _, _ in METHODS:
                for x, y in zip(cycle, traces[name]):
                    writer.writerow([cell, int(x), name, f"{y:.8f}", "empirical_saved_prediction"])


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = collect_traces()
    group, axes = plt.subplots(1, 3, figsize=(11.6, 4.15))
    for axis, cell, letter in zip(axes, CELLS, ("a", "b", "c")):
        draw_cell(axis, cell, data[cell], letter)
    group.subplots_adjust(left=0.065, right=0.995, top=0.93, bottom=0.15, wspace=0.27)
    save(group, "figure7_actual_10pct_factorial_ablation")
    plt.close(group)
    for cell, letter in zip(CELLS, ("a", "b", "c")):
        figure, axis = plt.subplots(figsize=(5.35, 3.6))
        draw_cell(axis, cell, data[cell], letter)
        figure.subplots_adjust(left=0.14, right=0.985, top=0.94, bottom=0.16)
        save(figure, f"figure7{letter}_cell_{cell.replace('-', '_')}_actual_10pct")
        plt.close(figure)
    write_source_csv(data)
    (OUTPUT_DIR / "README.md").write_text(
        "# Empirical Figure 7: 10% labels\\n\\n"
        "Generated solely from saved predictions of completed factorial-ablation_r0p1 runs. "
        "No trajectory values were simulated or modified.\\n",
        encoding="utf-8",
    )
    print(f"Wrote empirical Figure 7 to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
