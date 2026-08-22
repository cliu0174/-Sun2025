"""Create a top-journal-style, fully editable PI-MSCL architecture figure.

The figure combines three visual panels: controlled sparse supervision,
multi-scale temporal network architecture, and dual-objective learning. Curves,
points, network blocks, arrows, and labels are native Draw.io elements rather
than embedded raster images.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paper_plot_style import apply_paper_style, save_figure  # noqa: E402


W, H = 1800, 1050


@dataclass
class Panel:
    panel_id: str
    x: int
    y: int
    w: int
    h: int
    label: str


@dataclass
class Box:
    box_id: str
    parent: str
    x: int
    y: int
    w: int
    h: int
    label: str = ""
    fill: str = "#FFFFFF"
    stroke: str = "#666666"
    font_size: int = 13
    rounded: bool = True
    dashed: bool = False
    opacity: int = 100


@dataclass
class Dot:
    dot_id: str
    parent: str
    x: int
    y: int
    r: int
    fill: str
    stroke: str
    stroke_width: float = 1.2


@dataclass
class PathItem:
    path_id: str
    parent: str
    points: Sequence[Tuple[int, int]]
    color: str = "#4A4A4A"
    width: float = 1.6
    dashed: bool = False
    arrow: bool = False
    curved: bool = False


@dataclass
class Label:
    label_id: str
    parent: str
    x: int
    y: int
    w: int
    h: int
    text: str
    font_size: int = 13
    color: str = "#222222"
    bold: bool = False
    align: str = "center"


PANELS = {
    "panel_a": Panel("panel_a", 30, 70, 500, 930, "(a) Controlled sparse supervision"),
    "panel_b": Panel("panel_b", 560, 70, 1210, 590, "(b) PI-MSCL architecture"),
    "panel_c": Panel("panel_c", 560, 690, 1210, 310, "(c) Dual-objective learning"),
}

BOXES: List[Box] = []
DOTS: List[Dot] = []
PATHS: List[PathItem] = []
LABELS: List[Label] = []


def add_box(*args, **kwargs):
    BOXES.append(Box(*args, **kwargs))


def add_dot(*args, **kwargs):
    DOTS.append(Dot(*args, **kwargs))


def add_path(*args, **kwargs):
    PATHS.append(PathItem(*args, **kwargs))


def add_label(*args, **kwargs):
    LABELS.append(Label(*args, **kwargs))


def build_spec() -> None:
    # Global legend.
    legend = [
        ("Observed / predicted data", "#DDEED8", "#70A35B"),
        ("Convolutional representation", "#DCEAF7", "#5B87B2"),
        ("Temporal modeling", "#FFF0C2", "#CDA83D"),
        ("Learning objective", "#F8D7D4", "#B85450"),
    ]
    lx = 410
    for i, (text, fill, stroke) in enumerate(legend):
        add_box(f"leg_box_{i}", "1", lx, 22, 28, 16, fill=fill, stroke=stroke, rounded=False)
        add_label(f"leg_text_{i}", "1", lx + 38, 12, 225, 34, text, font_size=12, align="left")
        lx += 315

    # Panel A: sparse targets.
    add_label("a_target_title", "panel_a", 40, 60, 420, 30, "Sparse SOH targets", 14, bold=True)
    add_path("a_y_axis", "panel_a", [(70, 110), (70, 350)], width=1.2)
    add_path("a_x_axis", "panel_a", [(70, 350), (450, 350)], width=1.2)
    target_curve = [(75, 135), (120, 145), (165, 160), (210, 174), (255, 195), (300, 220), (345, 236), (390, 278), (445, 305)]
    add_path("a_true_soh", "panel_a", target_curve, color="#4E7C59", width=2.6, curved=True)
    labeled_indices = {0, 2, 5, 7}
    for i, (x, y) in enumerate(target_curve):
        if i in labeled_indices:
            add_dot(f"a_label_{i}", "panel_a", x, y, 7, "#E58B3A", "#B8641D", 1.4)
        else:
            add_dot(f"a_unlabel_{i}", "panel_a", x, y, 5, "#FFFFFF", "#A0A0A0", 1.1)
    add_label("a_soh_ylabel", "panel_a", 10, 185, 45, 70, "SOH", 12, bold=True)
    add_label("a_cycle_xlabel", "panel_a", 205, 352, 130, 28, "Cycle index", 11)
    add_label("a_ratio", "panel_a", 82, 378, 350, 28, "r = 1.0, 0.7, 0.5, 0.3", 11, color="#555555", align="left")

    # Panel A: complete operating feature trajectories.
    add_label("a_feature_title", "panel_a", 40, 420, 420, 30, "Complete operating-feature trajectories", 14, bold=True)
    add_path("a2_y_axis", "panel_a", [(70, 475), (70, 710)], width=1.2)
    add_path("a2_x_axis", "panel_a", [(70, 710), (450, 710)], width=1.2)
    feature_1 = [(75, 520), (120, 510), (165, 535), (210, 525), (255, 558), (300, 550), (345, 585), (390, 575), (445, 610)]
    feature_2 = [(75, 610), (120, 590), (165, 600), (210, 570), (255, 585), (300, 555), (345, 565), (390, 535), (445, 545)]
    feature_3 = [(75, 665), (120, 655), (165, 635), (210, 645), (255, 620), (300, 630), (345, 600), (390, 610), (445, 580)]
    add_path("a_feature_voltage", "panel_a", feature_1, color="#4C78A8", width=2.0, curved=True)
    add_path("a_feature_current", "panel_a", feature_2, color="#E08B3E", width=2.0, curved=True)
    add_path("a_feature_derived", "panel_a", feature_3, color="#7A9E63", width=2.0, curved=True)
    add_box("a_window", "panel_a", 245, 465, 90, 255, fill="#DCEAF7", stroke="#5B87B2", dashed=True, opacity=22)
    add_label("a_window_label", "panel_a", 240, 720, 100, 28, "window T = 40", 11, color="#4C78A8")
    add_label("a_feature_legend", "panel_a", 85, 735, 350, 35, "voltage     current     derived features", 11)
    add_box("a_takeaway", "panel_a", 65, 800, 380, 80, "Mask labels only; retain features and cycle order", "#F5F5F5", "#777777", 13)

    # Panel B: input tensor stack.
    add_label("b_input_label", "panel_b", 20, 80, 155, 32, "Input window", 13, bold=True)
    add_box("b_tensor_back2", "panel_b", 52, 223, 105, 120, fill="#EEF5EC", stroke="#A6C79A")
    add_box("b_tensor_back1", "panel_b", 42, 213, 105, 120, fill="#E6F1E2", stroke="#8FB982")
    add_box("b_tensor_front", "panel_b", 32, 203, 105, 120, "B × 40 × F", "#DDEED8", "#70A35B", 13)

    # Multi-scale Conv1D branches. Keep each branch visually sparse: the
    # branch card itself represents the repeated convolutional layers.
    branch_specs = [
        ("local", 130, "Local", "k = 3"),
        ("medium", 260, "Medium-range", "k = 7"),
        ("long", 390, "Long-range", "k = 15"),
    ]
    for name, y, title, kernel in branch_specs:
        add_box(
            f"b_{name}_branch",
            "panel_b",
            205,
            y,
            210,
            100,
            f"{title}\nConv1D × 2   {kernel}\n64 → 64 channels",
            "#DCEAF7",
            "#5B87B2",
            12,
        )

    add_box("b_fusion", "panel_b", 465, 245, 130, 120, "Concat\n1 × 1 Conv\n128 ch", "#E7F0F8", "#4C78A8", 13)

    # LSTM network cells.
    add_label("b_lstm_title", "panel_b", 655, 175, 235, 30, "Cross-cycle temporal model", 13, bold=True)
    for i, x in enumerate((665, 735, 805)):
        add_box(f"b_lstm_{i}", "panel_b", x, 250, 58, 100, "LSTM", "#FFF0C2", "#CDA83D", 11)
        if i < 2:
            add_path(f"b_lstm_edge_{i}", "panel_b", [(x + 58, 300), (x + 70, 300)], arrow=True, width=1.2)

    add_box("b_head", "panel_b", 925, 245, 115, 120, "FC 64\nDropout\nSigmoid head", "#FFE6CC", "#D79B00", 12)

    # Predicted SOH curve as the output object.
    add_label("b_output_title", "panel_b", 1055, 125, 140, 35, "Predicted SOH", 12, bold=True)
    add_path("b_out_y", "panel_b", [(1065, 185), (1065, 430)], width=1.0)
    add_path("b_out_x", "panel_b", [(1065, 430), (1190, 430)], width=1.0)
    out_curve = [(1070, 215), (1087, 220), (1104, 235), (1121, 250), (1138, 270), (1155, 320), (1173, 350), (1188, 390)]
    add_path("b_pred_curve", "panel_b", out_curve, color="#4E7C59", width=2.8, curved=True)
    add_label("b_yhat", "panel_b", 1070, 440, 115, 30, "SOH estimate ŷ", 11)

    # Main network arrows.
    for name, y, _, _ in branch_specs:
        add_path(f"b_in_{name}", "panel_b", [(137, 263), (170, 263), (170, y + 50), (205, y + 50)], arrow=True, width=1.4)
        add_path(f"b_{name}_fuse", "panel_b", [(415, y + 50), (440, y + 50), (440, 305), (465, 305)], arrow=True, width=1.4)
    add_path("b_fuse_lstm", "panel_b", [(595, 305), (640, 305), (640, 300), (665, 300)], arrow=True, width=1.5)
    add_path("b_lstm_head", "panel_b", [(863, 300), (900, 300), (900, 305), (925, 305)], arrow=True, width=1.5)
    add_path("b_head_output", "panel_b", [(1040, 305), (1052, 305), (1052, 305), (1065, 305)], arrow=True, width=1.5)

    # Panel C: two evidence paths.
    add_label("c_data_path", "panel_c", 30, 55, 230, 28, "Label-supported path", 13, bold=True)
    add_path("c_data_y", "panel_c", [(55, 105), (55, 235)], width=1.0)
    add_path("c_data_x", "panel_c", [(55, 235), (230, 235)], width=1.0)
    c_curve = [(60, 120), (90, 130), (120, 145), (150, 165), (180, 190), (225, 215)]
    add_path("c_data_curve", "panel_c", c_curve, color="#A0A0A0", width=1.8, curved=True)
    for i in (0, 2, 4, 5):
        x, y = c_curve[i]
        add_dot(f"c_label_{i}", "panel_c", x, y, 6, "#E58B3A", "#B8641D")
    add_box("c_mse", "panel_c", 275, 105, 185, 105, "Masked MSE\n(m = 1 only)", "#F8D7D4", "#B85450", 13)

    add_label("c_phys_path", "panel_c", 500, 55, 255, 28, "Trajectory-structure path", 13, bold=True)
    add_path("c_phys_y", "panel_c", [(520, 105), (520, 235)], width=1.0)
    add_path("c_phys_x", "panel_c", [(520, 235), (740, 235)], width=1.0)
    phys_curve = [(525, 125), (560, 140), (595, 160), (630, 150), (665, 185), (700, 205), (735, 220)]
    add_path("c_phys_curve", "panel_c", phys_curve, color="#4E7C59", width=2.4, curved=True)
    add_path("c_violation", "panel_c", [phys_curve[2], phys_curve[3]], color="#C44E52", width=3.4)
    add_label("c_violation_label", "panel_c", 585, 100, 95, 35, "penalized rise", 10, color="#B85450")
    add_box("c_mono", "panel_c", 775, 105, 195, 105, "Soft monotonicity\nδ = 0.005\ncycle ≥ 300", "#F8D7D4", "#B85450", 12)
    add_box("c_total", "panel_c", 1000, 105, 180, 105, "Total loss\nLdata + λ Lmono", "#F2B9B4", "#9E3F3A", 12)
    add_path("c_data_to_mse", "panel_c", [(230, 170), (255, 170), (255, 158), (275, 158)], arrow=True, width=1.4)
    add_path("c_phys_to_mono", "panel_c", [(740, 170), (755, 170), (755, 158), (775, 158)], arrow=True, width=1.4)
    add_path(
        "c_mse_total",
        "panel_c",
        [(460, 158), (480, 158), (480, 265), (985, 265), (985, 185), (1000, 185)],
        arrow=True,
        width=1.4,
    )
    add_path(
        "c_mono_total",
        "panel_c",
        [(970, 158), (985, 158), (985, 135), (1000, 135)],
        arrow=True,
        width=1.4,
    )

    # Cross-panel flow cues.
    add_path("global_a_b", "1", [(530, 365), (555, 365), (555, 365), (590, 365)], arrow=True, width=1.8)
    add_path("global_b_c", "1", [(1690, 500), (1690, 675), (1200, 675), (1200, 795)], color="#B85450", width=1.5, dashed=True, arrow=True)


def panel_offset(parent: str) -> Tuple[int, int]:
    if parent == "1":
        return 0, 0
    panel = PANELS[parent]
    return panel.x, panel.y


def plain_text(text: str) -> str:
    text = text.replace("<br>", "\n").replace("<br/>", "\n")
    text = text.replace("\\n", "\n")
    text = re.sub(r"<sub>(.*?)</sub>", r"$_{\1}$", text)
    text = re.sub(r"<sup>(.*?)</sup>", r"$^{\1}$", text)
    text = re.sub(r"</?b>", "", text)
    return html.unescape(text)


def render_preview(stem: Path) -> List[Path]:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    for panel in PANELS.values():
        ax.add_patch(
            FancyBboxPatch(
                (panel.x, panel.y), panel.w, panel.h,
                boxstyle="round,pad=0.01,rounding_size=14",
                facecolor="#FCFCFC", edgecolor="#B8B8B8", linewidth=1.3, zorder=0,
            )
        )
        ax.text(panel.x + 18, panel.y + 28, panel.label, ha="left", va="center", fontsize=11.5, fontweight="bold", color="#333333")

    # Light highlight rectangles first.
    for box in BOXES:
        ox, oy = panel_offset(box.parent)
        alpha = box.opacity / 100.0
        patch = FancyBboxPatch(
            (ox + box.x, oy + box.y), box.w, box.h,
            boxstyle="round,pad=0.01,rounding_size=10" if box.rounded else "square,pad=0",
            facecolor=box.fill, edgecolor=box.stroke, linewidth=1.25,
            linestyle="--" if box.dashed else "-", alpha=alpha, zorder=1,
        )
        ax.add_patch(patch)
        if box.label:
            ax.text(ox + box.x + box.w / 2, oy + box.y + box.h / 2, plain_text(box.label), ha="center", va="center", fontsize=8.2 if box.font_size <= 12 else 8.7, linespacing=1.15, zorder=4)

    for path in PATHS:
        ox, oy = panel_offset(path.parent)
        points = [(ox + x, oy + y) for x, y in path.points]
        mpl_path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
        if path.arrow:
            patch = FancyArrowPatch(path=mpl_path, arrowstyle="-|>", mutation_scale=11, color=path.color, linewidth=path.width, linestyle="--" if path.dashed else "-", zorder=3)
            ax.add_patch(patch)
        else:
            xs, ys = zip(*points)
            ax.plot(xs, ys, color=path.color, linewidth=path.width, linestyle="--" if path.dashed else "-", solid_capstyle="round", solid_joinstyle="round", zorder=2)

    for dot in DOTS:
        ox, oy = panel_offset(dot.parent)
        ax.add_patch(Circle((ox + dot.x, oy + dot.y), dot.r, facecolor=dot.fill, edgecolor=dot.stroke, linewidth=dot.stroke_width, zorder=5))

    for label in LABELS:
        ox, oy = panel_offset(label.parent)
        ha = {"left": "left", "right": "right"}.get(label.align, "center")
        x = ox + label.x + (0 if ha == "left" else label.w if ha == "right" else label.w / 2)
        ax.text(x, oy + label.y + label.h / 2, plain_text(label.text), ha=ha, va="center", fontsize=8.0 if label.font_size <= 11 else 8.7, color=label.color, fontweight="bold" if label.bold else "normal", zorder=6)

    return save_figure(fig, stem, dpi=600)


def box_style(box: Box) -> str:
    style = (
        f"rounded={1 if box.rounded else 0};arcSize=10;whiteSpace=wrap;html=1;"
        f"fillColor={box.fill};strokeColor={box.stroke};strokeWidth=1.3;"
        f"fillOpacity={box.opacity};fontFamily=Times New Roman;fontSize={box.font_size};"
        "fontColor=#222222;align=center;verticalAlign=middle;spacing=5;"
    )
    if box.dashed:
        style += "dashed=1;dashPattern=6 4;"
    return style


def add_geometry(cell: ET.Element, x: int, y: int, w: int, h: int) -> None:
    ET.SubElement(cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})


def write_drawio(path: Path) -> None:
    mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "version": "26.0.0"})
    diagram = ET.SubElement(
        mxfile,
        "diagram",
        {"name": path.stem.replace("_", " "), "id": "pi_mscl_architecture"},
    )
    model = ET.SubElement(diagram, "mxGraphModel", {"dx": str(W), "dy": str(H), "grid": "1", "gridSize": "10", "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1", "fold": "1", "page": "0", "math": "1", "shadow": "0"})
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    for panel in PANELS.values():
        cell = ET.SubElement(root, "mxCell", {"id": panel.panel_id, "value": panel.label, "style": "rounded=1;arcSize=10;container=1;pointerEvents=0;whiteSpace=wrap;html=1;fillColor=#FCFCFC;strokeColor=#B8B8B8;strokeWidth=1.3;verticalAlign=top;align=left;spacingTop=10;spacingLeft=14;fontFamily=Times New Roman;fontSize=16;fontStyle=1;fontColor=#333333;", "vertex": "1", "parent": "1"})
        add_geometry(cell, panel.x, panel.y, panel.w, panel.h)

    for box in BOXES:
        cell = ET.SubElement(root, "mxCell", {"id": box.box_id, "value": box.label.replace("\n", "&#xa;"), "style": box_style(box), "vertex": "1", "parent": box.parent})
        add_geometry(cell, box.x, box.y, box.w, box.h)

    for path_item in PATHS:
        style = f"edgeStyle=none;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor={path_item.color};strokeWidth={path_item.width};endArrow={'blockThin' if path_item.arrow else 'none'};endFill=1;"
        if path_item.dashed:
            style += "dashed=1;dashPattern=6 4;"
        if path_item.curved:
            style += "curved=1;"
        cell = ET.SubElement(root, "mxCell", {"id": path_item.path_id, "value": "", "style": style, "edge": "1", "parent": path_item.parent})
        geom = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        ET.SubElement(geom, "mxPoint", {"x": str(path_item.points[0][0]), "y": str(path_item.points[0][1]), "as": "sourcePoint"})
        ET.SubElement(geom, "mxPoint", {"x": str(path_item.points[-1][0]), "y": str(path_item.points[-1][1]), "as": "targetPoint"})
        arr = ET.SubElement(geom, "Array", {"as": "points"})
        for x, y in path_item.points[1:-1]:
            ET.SubElement(arr, "mxPoint", {"x": str(x), "y": str(y)})

    for dot in DOTS:
        cell = ET.SubElement(root, "mxCell", {"id": dot.dot_id, "value": "", "style": f"ellipse;whiteSpace=wrap;html=1;fillColor={dot.fill};strokeColor={dot.stroke};strokeWidth={dot.stroke_width};", "vertex": "1", "parent": dot.parent})
        add_geometry(cell, dot.x - dot.r, dot.y - dot.r, 2 * dot.r, 2 * dot.r)

    for label in LABELS:
        align = label.align if label.align in {"left", "right", "center"} else "center"
        style = f"text;html=1;strokeColor=none;fillColor=none;align={align};verticalAlign=middle;fontFamily=Times New Roman;fontSize={label.font_size};fontColor={label.color};fontStyle={1 if label.bold else 0};whiteSpace=wrap;"
        cell = ET.SubElement(root, "mxCell", {"id": label.label_id, "value": label.text, "style": style, "vertex": "1", "parent": label.parent})
        add_geometry(cell, label.x, label.y, label.w, label.h)

    ET.indent(mxfile, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(mxfile).write(path, encoding="utf-8", xml_declaration=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drawio", type=Path, default=PROJECT_ROOT / "docs" / "PI-MSCL_architecture_v3.drawio")
    parser.add_argument("--preview-stem", type=Path, default=PROJECT_ROOT / "docs" / "PI-MSCL_architecture_v3_preview")
    return parser.parse_args()


def main() -> None:
    build_spec()
    args = parse_args()
    write_drawio(args.drawio)
    saved = render_preview(args.preview_stem)
    print(f"Editable Draw.io: {args.drawio}")
    for item in saved:
        print(f"Preview: {item}")


if __name__ == "__main__":
    main()
