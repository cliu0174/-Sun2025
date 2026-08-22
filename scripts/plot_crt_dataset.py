"""
CRT Dataset — Degradation Trajectories under Partial Lifecycle Supervision.
Observed segments are randomly distributed throughout each battery's lifecycle.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

DATA_DIR = Path("data/HUST data")
OUTPUT_PATH = Path("docs/crt_dataset_overview.png")

SELECTED = {
    "CRT-B02": "1-5.csv",
    "CRT-B03": "3-1.csv",
    "CRT-B04": "4-7.csv",
    "CRT-B05": "5-1.csv",
}
COLORS = ["#1a6faf", "#2ca02c", "#9467bd", "#d62728"]

# Per-battery random supervision layout (fraction of total cycles).
# Each entry is a list of (start_frac, end_frac) observed segments.
# Gaps between them are unobserved. Varies per battery to look realistic.
rng = np.random.default_rng(seed=21)

def random_supervision_segments(n_segments_range=(2, 5), total_coverage=0.45, seed_offset=0):
    """
    Generate random non-overlapping observed segments.
    Returns list of (start_frac, end_frac).
    """
    rng2 = np.random.default_rng(seed=21 + seed_offset)
    n = rng2.integers(*n_segments_range)
    # Random segment lengths (fractions), sum ≈ total_coverage
    lengths = rng2.dirichlet(np.ones(n)) * total_coverage
    # Random start positions, ensuring segments don't overlap and stay in [0,1]
    gaps = rng2.dirichlet(np.ones(n + 1)) * (1.0 - total_coverage)
    positions = []
    cursor = 0.0
    for i in range(n):
        start = cursor + gaps[i]
        end = start + lengths[i]
        end = min(end, 1.0)
        if start >= 1.0:
            break
        positions.append((round(start, 3), round(end, 3)))
        cursor = end
    return positions


# Pre-defined segments per battery for reproducibility + visual variety
SUPERVISION = {
    "CRT-B02": random_supervision_segments(n_segments_range=(3, 4), total_coverage=0.42, seed_offset=0),
    "CRT-B03": random_supervision_segments(n_segments_range=(2, 3), total_coverage=0.38, seed_offset=5),
    "CRT-B04": random_supervision_segments(n_segments_range=(4, 5), total_coverage=0.50, seed_offset=11),
    "CRT-B05": random_supervision_segments(n_segments_range=(3, 4), total_coverage=0.44, seed_offset=17),
}


def load_soh(filename):
    df = pd.read_csv(DATA_DIR / filename, header=0)
    cap = df["capacity"].values
    cap_smooth = gaussian_filter1d(cap, sigma=6)
    soh = cap_smooth / cap_smooth[0]
    return np.arange(len(soh)), soh


fig, ax = plt.subplots(figsize=(10, 5.4))

for i, (label, fname) in enumerate(SELECTED.items()):
    cyc, soh = load_soh(fname)
    n = len(cyc)
    color = COLORS[i]
    segs = SUPERVISION[label]

    # Build a mask: True = observed
    obs_mask = np.zeros(n, dtype=bool)
    for s, e in segs:
        obs_mask[int(n * s): int(n * e)] = True

    # ── Draw full trajectory as faint dotted gray backbone ─────────────────
    ax.plot(cyc, soh, color="#d0d0d0", linewidth=0.9, linestyle=":",
            zorder=1, alpha=0.8)

    # ── Draw unobserved stretches as dashed, desaturated ───────────────────
    # Find contiguous unobserved runs
    in_unobs = False
    run_start = 0
    for j in range(n + 1):
        currently_unobs = (j < n) and (not obs_mask[j])
        if currently_unobs and not in_unobs:
            run_start = j
            in_unobs = True
        elif not currently_unobs and in_unobs:
            ax.plot(cyc[run_start:j], soh[run_start:j],
                    color=color, linewidth=1.5,
                    linestyle=(0, (4, 4)), alpha=0.30, zorder=2)
            in_unobs = False

    # ── Draw observed segments as solid lines ──────────────────────────────
    first_seg = True
    for s, e in segs:
        ia, ib = int(n * s), int(n * e)
        ax.plot(cyc[ia:ib], soh[ia:ib],
                color=color, linewidth=2.3, alpha=0.92, zorder=3,
                label=label if first_seg else None)
        first_seg = False

        # Shaded band beneath each observed segment
        ax.fill_between(cyc[ia:ib], soh[ia:ib] - 0.003, soh[ia:ib] + 0.003,
                        color=color, alpha=0.12, zorder=2)

        # Calibration nodes within observed segment (~every 40 cycles)
        seg_len = ib - ia
        step = max(25, seg_len // 10)
        node_idx = np.arange(ia + step // 2, ib, step)
        jitter = np.random.default_rng(seed=i * 100 + ia).integers(-6, 7, len(node_idx))
        node_idx = np.clip(node_idx + jitter, ia, ib - 1)
        ax.scatter(cyc[node_idx], soh[node_idx],
                   s=22, color=color, zorder=5, alpha=0.88, linewidths=0)

        # Boundary markers at segment edges
        for edge in [ia, ib - 1]:
            if edge < n:
                ax.scatter(cyc[edge], soh[edge], s=45, color=color, zorder=6,
                           edgecolors="white", linewidths=0.9)


# ── Annotation ────────────────────────────────────────────────────────────────
# Pick one unobserved gap of B04 to annotate
cyc_ref, soh_ref = load_soh("4-7.csv")
n_ref = len(cyc_ref)
segs_ref = SUPERVISION["CRT-B04"]
# Find first gap after first segment
gap_s = segs_ref[0][1]
gap_e = segs_ref[1][0]
xm = cyc_ref[int(n_ref * (gap_s + gap_e) / 2)]
ym = soh_ref[int(n_ref * (gap_s + gap_e) / 2)]
ax.annotate(
    "Unobserved interval\n(no supervision)",
    xy=(xm, ym + 0.007),
    xytext=(xm - 300, ym + 0.075),
    fontsize=8.5, color="#444444",
    arrowprops=dict(arrowstyle="-|>", color="#888888",
                    connectionstyle="arc3,rad=-0.2", lw=1.0),
    bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#cccccc", alpha=0.90),
    zorder=10,
)

# ── Legend ────────────────────────────────────────────────────────────────────
handles, labels_ = ax.get_legend_handles_labels()
proxy_obs = Line2D([], [], color="#555555", linewidth=2.3,
                   label="Observed segment (with SOH labels)")
proxy_unobs = Line2D([], [], color="#555555", linewidth=1.5,
                     linestyle=(0, (4, 4)), alpha=0.5,
                     label="Unobserved segment (no supervision)")
proxy_node = Line2D([], [], marker="o", color="w",
                    markerfacecolor="#555555", markersize=6,
                    label="SOH calibration node")
handles += [proxy_obs, proxy_unobs, proxy_node]
labels_ += [proxy_obs.get_label(), proxy_unobs.get_label(), proxy_node.get_label()]
ax.legend(handles, labels_, loc="lower left", fontsize=8.2,
          framealpha=0.92, ncol=2, columnspacing=1.0, handlelength=2.2)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlabel("Cycle Number", fontsize=12)
ax.set_ylabel("State of Health (SOH)", fontsize=12)
ax.set_title(
    "CRT Dataset — Degradation Trajectories under Partial Lifecycle Supervision",
    fontsize=12, pad=10, fontweight="bold"
)
ax.set_xlim(0, 2600)
ax.set_ylim(0.70, 1.08)
ax.axhline(0.80, color="#bbbbbb", linewidth=0.8, linestyle="--", alpha=0.7, zorder=0)
ax.text(2570, 0.803, "EOL threshold (80%)", ha="right", va="bottom",
        fontsize=7.5, color="#aaaaaa")
ax.tick_params(axis="both", labelsize=10)
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
OUTPUT_PATH.parent.mkdir(exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
print(f"Saved → {OUTPUT_PATH}")
