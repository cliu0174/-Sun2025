from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(r"D:\Projects\1111-soh")
DATA_DIR = ROOT / "data" / "HUST data"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)


def moving_average(values, window=15):
    s = pd.Series(values)
    return s.rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def load_capacity_curves():
    curves = {}
    metrics = []
    for csv_path in sorted(DATA_DIR.glob("*.csv"), key=lambda p: tuple(map(int, p.stem.split("-")))):
        df = pd.read_csv(csv_path)
        cap = df["capacity"].astype(float).to_numpy()
        cap = cap[np.isfinite(cap)]
        if len(cap) == 0:
            continue
        battery_id = csv_path.stem
        initial = float(np.nanmean(cap[: min(20, len(cap))]))
        final = float(np.nanmean(cap[-min(20, len(cap)) :]))
        loss = initial - final
        loss_pct = loss / initial * 100
        rate = loss_pct / len(cap) * 1000
        curves[battery_id] = cap
        metrics.append(
            {
                "battery_id": battery_id,
                "cycles": len(cap),
                "initial": initial,
                "final": final,
                "loss_pct": loss_pct,
                "loss_rate_pct_per_1000_cycles": rate,
            }
        )
    return curves, pd.DataFrame(metrics)


def plot(stretch_normal_x=True):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
        }
    )

    curves, metrics = load_capacity_curves()
    anomaly_n = max(1, int(np.ceil(len(metrics) * 0.15)))
    metrics = metrics.sort_values("loss_rate_pct_per_1000_cycles", ascending=False).reset_index(drop=True)
    anomaly_ids = set(metrics.head(anomaly_n)["battery_id"])
    normal_ids = [bid for bid in curves if bid not in anomaly_ids]

    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)

    normal_palette = plt.cm.tab20(np.linspace(0, 1, 20))
    normal_x_scale = 1.21
    for idx, bid in enumerate(normal_ids):
        y = moving_average(curves[bid], window=15)
        x = np.arange(len(y), dtype=float)
        if stretch_normal_x:
            x = x * normal_x_scale
        ax.plot(
            x,
            y,
            color=normal_palette[idx % len(normal_palette)],
            lw=0.85,
            alpha=0.28,
            zorder=1,
        )

    anomaly_palette = ["#D7191C", "#F04E30", "#B2182B", "#EF8A62"]
    for idx, bid in enumerate(sorted(anomaly_ids, key=lambda x: tuple(map(int, x.split("-"))))):
        y = moving_average(curves[bid], window=15)
        x = np.arange(len(y))
        ax.plot(
            x,
            y,
            color=anomaly_palette[idx % len(anomaly_palette)],
            lw=2.35,
            alpha=0.96,
            zorder=5,
        )
    ax.set_title("")
    ax.set_xlabel("Cycle Number", weight="bold")
    ax.set_ylabel("Capacity (Ah)", weight="bold")
    x_max = max(
        max(len(v) for bid, v in curves.items() if bid in anomaly_ids),
        max(len(v) * normal_x_scale for bid, v in curves.items() if bid not in anomaly_ids),
    )
    ax.set_xlim(-30, x_max + 160)
    ax.set_ylim(0.86, 1.225)
    ax.grid(True, color="#E8E8E8", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)

    avg_loss = metrics["loss_pct"].mean()
    max_cycles = metrics["cycles"].max()
    min_cycles = metrics["cycles"].min()
    threshold = metrics.iloc[anomaly_n - 1]["loss_rate_pct_per_1000_cycles"]
    summary = (
        f"Total Batteries: {len(metrics)}\n"
        f"Normal Batteries: {len(metrics) - anomaly_n}\n"
        f"Fast-Degradation: {anomaly_n}\n"
        f"Avg Capacity Loss: {avg_loss:.2f}%\n"
        f"Criterion: Top 15% loss rate\n"
        f"Rate Threshold: {threshold:.2f}%/k cycles\n"
        f"Max Cycles: {max_cycles}\n"
        f"Min Cycles: {min_cycles}"
    )
    ax.text(
        0.018,
        0.98,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        weight="bold",
        bbox=dict(boxstyle="square,pad=0.35", facecolor="#DDEFF4", edgecolor="#6E8C97", linewidth=0.8, alpha=0.92),
    )

    handles = [
        Line2D([0], [0], color="#7F8C8D", lw=1.2, alpha=0.55, label="Normal batteries"),
        Line2D([0], [0], color="#D7191C", lw=2.8, label="Fast-degradation anomalies"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.92, fontsize=8, edgecolor="#B0B0B0")

    note = "Anomaly definition: batteries with the highest capacity-loss rate per cycle (top 15%)."
    if stretch_normal_x:
        note = f"Display-enhanced: normal trajectories use original capacity values; cycle axis is stretched x{normal_x_scale:.2f}."
    ax.text(0.99, 0.018, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#555555")

    fig.tight_layout()
    suffix = "_normal_x_stretched" if stretch_normal_x else ""
    base = OUT_DIR / f"hust_capacity_degradation_normal_vs_anomaly{suffix}"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    metrics.to_csv(OUT_DIR / "hust_capacity_degradation_anomaly_metrics.csv", index=False, encoding="utf-8-sig")
    print(base.with_suffix(".png"))
    print("Anomaly IDs:", ", ".join(metrics.head(anomaly_n)["battery_id"]))


if __name__ == "__main__":
    plot(stretch_normal_x=True)
