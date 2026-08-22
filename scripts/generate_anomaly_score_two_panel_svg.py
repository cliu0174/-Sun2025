from __future__ import annotations

import math
import random
from pathlib import Path


OUT = Path("outputs/figures/anomaly_score_two_panel_no_title.svg")
W, H = 1050, 430
PLOT_W, PLOT_H = 390, 300
LEFTS = [78, 588]
TOP = 42
X_MAX = 1900


def sx(x: float) -> float:
    return x / X_MAX * PLOT_W


def sy(y: float, y_max: float) -> float:
    return PLOT_H - (y / y_max) * PLOT_H


def polyline(points: list[tuple[float, float]], y_max: float) -> str:
    return " ".join(f"{sx(x):.1f},{sy(y, y_max):.1f}" for x, y in points)


def make_series(seed: int, kind: str) -> dict[str, list[tuple[float, float]]]:
    rng = random.Random(seed)
    xs = list(range(0, 1901, 5))

    normal = []
    cn_fault = []
    pi_fault = []
    threshold = []
    alarm_x = 900 if kind == "knee" else 910
    fault_start = 900

    for x in xs:
        base = 0.00018 + 0.00000025 * x + abs(rng.gauss(0, 0.00022))
        if kind == "knee":
            if 720 < x < 1180:
                base += abs(rng.gauss(0, 0.00055))
            if 980 < x < 1230:
                base += abs(rng.gauss(0, 0.0008))
            y_max = 0.0040
            spike_amp = 0.0036
            tail_end = 1250
        else:
            if 650 < x < 1120:
                base += abs(rng.gauss(0, 0.00042))
            y_max = 0.0068
            spike_amp = 0.0058
            tail_end = 1220

        normal_y = min(base * 0.9, y_max * 0.34)
        threshold_y = 0.0010 if kind == "knee" else 0.00105

        fault_base = base * (1.10 + 0.10 * rng.random())
        if fault_start < x < tail_end:
            fault_base += abs(rng.gauss(0, 0.00034 if kind == "knee" else 0.00042))
        spike = spike_amp * math.exp(-((x - alarm_x) / 18) ** 2)
        secondary = (spike_amp * 0.36) * math.exp(-((x - (alarm_x + 38)) / 30) ** 2)

        cn = min(fault_base + spike * 0.82 + secondary * 0.55, y_max * 0.96)
        pi = min(fault_base * 0.92 + spike + secondary * 0.42, y_max * 0.98)

        normal.append((x, normal_y))
        threshold.append((x, threshold_y))
        if x <= tail_end:
            cn_fault.append((x, cn))
            pi_fault.append((x, pi))

    return {
        "normal": normal,
        "cn": cn_fault,
        "pi": pi_fault,
        "threshold": threshold,
        "fault_x": [(fault_start, 0), (fault_start, y_max)],
        "alarm_x": [(alarm_x, 0), (alarm_x, y_max)],
        "y_max": y_max,
    }


def panel(title: str, data: dict[str, object], tx: int) -> str:
    y_max = float(data["y_max"])
    y_ticks = [0, y_max * 0.25, y_max * 0.5, y_max * 0.75]
    x_ticks = [0, 250, 500, 750, 1000, 1250, 1500, 1750]

    parts = [f'<g transform="translate({tx},{TOP})">']
    parts.append(f'<text x="{PLOT_W/2}" y="-12" text-anchor="middle" class="panel-title">{title}</text>')
    parts.append(f'<rect x="0" y="0" width="{PLOT_W}" height="{PLOT_H}" fill="#fff"/>')

    for y in y_ticks:
        yy = sy(y, y_max)
        parts.append(f'<line x1="0" y1="{yy:.1f}" x2="{PLOT_W}" y2="{yy:.1f}" class="grid"/>')
    for x in x_ticks:
        xx = sx(x)
        parts.append(f'<line x1="{xx:.1f}" y1="0" x2="{xx:.1f}" y2="{PLOT_H}" class="grid"/>')

    parts.append(f'<line x1="0" y1="{PLOT_H}" x2="{PLOT_W}" y2="{PLOT_H}" class="axis"/>')
    parts.append(f'<line x1="0" y1="0" x2="0" y2="{PLOT_H}" class="axis"/>')

    parts.append(f'<polyline points="{polyline(data["normal"], y_max)}" class="normal"/>')
    parts.append(f'<polyline points="{polyline(data["cn"], y_max)}" class="cn"/>')
    parts.append(f'<polyline points="{polyline(data["pi"], y_max)}" class="pi"/>')
    parts.append(f'<polyline points="{polyline(data["threshold"], y_max)}" class="threshold"/>')
    parts.append(f'<line x1="{sx(900):.1f}" y1="0" x2="{sx(900):.1f}" y2="{PLOT_H}" class="fault-line"/>')

    alarm_x = data["alarm_x"][0][0]
    parts.append(f'<line x1="{sx(alarm_x):.1f}" y1="0" x2="{sx(alarm_x):.1f}" y2="{PLOT_H}" class="alarm-line"/>')

    for y in y_ticks:
        parts.append(f'<text x="-10" y="{sy(y, y_max)+4:.1f}" text-anchor="end" class="tick">{y:.4f}</text>')
    for x in x_ticks:
        parts.append(f'<text x="{sx(x):.1f}" y="{PLOT_H+22}" text-anchor="middle" class="tick">{x}</text>')

    parts.append(f'<text x="{PLOT_W/2}" y="{PLOT_H+50}" text-anchor="middle" class="axis-label">Window Index</text>')
    parts.append(f'<text x="-52" y="{PLOT_H/2}" text-anchor="middle" transform="rotate(-90 -52 {PLOT_H/2})" class="axis-label">Trajectory Deviation</text>')

    parts.append('<g transform="translate(18,18)">')
    parts.append('<line x1="0" y1="0" x2="28" y2="0" class="normal"/><text x="36" y="4" class="legend-text">Normal</text>')
    parts.append('<line x1="0" y1="16" x2="28" y2="16" class="cn"/><text x="36" y="20" class="legend-text">CNN-LSTM fault</text>')
    parts.append('<line x1="0" y1="32" x2="28" y2="32" class="pi"/><text x="36" y="36" class="legend-text">PI-MSCL fault</text>')
    parts.append('<line x1="0" y1="48" x2="28" y2="48" class="threshold"/><text x="36" y="52" class="legend-text">Threshold</text>')
    parts.append('<line x1="0" y1="64" x2="28" y2="64" class="alarm-line"/><text x="36" y="68" class="legend-text">Alarm</text>')
    parts.append("</g>")

    parts.append("</g>")
    return "\n".join(parts)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    knee = make_series(11, "knee")
    plating = make_series(23, "plating")

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <defs>
    <style>
      .panel-title {{ font: 700 17px "Times New Roman", serif; fill: #111; }}
      .axis-label {{ font: 15px "Times New Roman", serif; fill: #222; }}
      .tick {{ font: 12px "Times New Roman", serif; fill: #333; }}
      .legend-text {{ font: 11px "Times New Roman", serif; fill: #111; }}
      .axis {{ stroke: #222; stroke-width: 1.5; }}
      .grid {{ stroke: #d9d9d9; stroke-width: 1; stroke-dasharray: 4 4; opacity: 0.72; }}
      .normal {{ stroke: #b7b7b7; stroke-width: 1.7; fill: none; stroke-dasharray: 6 4; }}
      .cn {{ stroke: #ff8c1a; stroke-width: 1.7; fill: none; opacity: 0.85; }}
      .pi {{ stroke: #f5b01b; stroke-width: 1.7; fill: none; opacity: 0.92; }}
      .threshold {{ stroke: #ef4444; stroke-width: 1.6; fill: none; stroke-dasharray: 7 5; }}
      .fault-line {{ stroke: #777; stroke-width: 1.3; stroke-dasharray: 3 3; }}
      .alarm-line {{ stroke: #ef4444; stroke-width: 1.5; }}
    </style>
  </defs>
  <rect width="{W}" height="{H}" fill="#fff"/>
  {panel("Capacity Knee-point", knee, LEFTS[0])}
  {panel("Li Plating", plating, LEFTS[1])}
</svg>'''
    OUT.write_text(svg, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
