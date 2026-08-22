"""Generate the editable PI-MSCL architecture diagram and a matched preview.

The Draw.io XML and Matplotlib preview share one coordinate specification.  The
preview is used for local visual QA when the Draw.io desktop CLI is unavailable.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paper_plot_style import apply_paper_style, save_figure  # noqa: E402


CANVAS_WIDTH = 1800
CANVAS_HEIGHT = 830


@dataclass(frozen=True)
class Node:
    node_id: str
    label: str
    x: int
    y: int
    width: int
    height: int
    fill: str
    stroke: str
    parent: str = "1"
    font_size: int = 15
    dashed: bool = False
    container: bool = False


@dataclass(frozen=True)
class Edge:
    edge_id: str
    source: str
    target: str
    points: Sequence[Tuple[int, int]]
    color: str = "#444444"
    dashed: bool = False


CONTAINERS: Dict[str, Node] = {
    "ms_box": Node(
        "ms_box",
        "<b>Multi-scale degradation encoder</b>",
        290,
        120,
        650,
        350,
        "#F4F8FD",
        "#6C8EBF",
        font_size=16,
        dashed=True,
        container=True,
    ),
    "objective_box": Node(
        "objective_box",
        "<b>Training objective</b>",
        1000,
        520,
        760,
        270,
        "#FFF9F2",
        "#B85450",
        font_size=16,
        dashed=True,
        container=True,
    ),
}


NODES: List[Node] = [
    Node(
        "input",
        "<b>Cycle window</b><br>X: B × 40 × F<br>all features retained",
        30,
        230,
        220,
        110,
        "#D5E8D4",
        "#82B366",
        font_size=13,
    ),
    Node(
        "branch_local",
        "<b>Local pattern</b><br>Conv1D × 2, k = 3<br>64 → 64 channels",
        40,
        80,
        230,
        80,
        "#DAE8FC",
        "#6C8EBF",
        parent="ms_box",
        font_size=14,
    ),
    Node(
        "branch_mid",
        "<b>Medium-range pattern</b><br>Conv1D × 2, k = 7<br>64 → 64 channels",
        40,
        170,
        230,
        80,
        "#DAE8FC",
        "#6C8EBF",
        parent="ms_box",
        font_size=14,
    ),
    Node(
        "branch_long",
        "<b>Long-range pattern</b><br>Conv1D × 2, k = 15<br>64 → 64 channels",
        40,
        260,
        230,
        80,
        "#DAE8FC",
        "#6C8EBF",
        parent="ms_box",
        font_size=14,
    ),
    Node(
        "fusion",
        "<b>Feature fusion</b><br>Concatenate<br>1 × 1 Conv, BN, ReLU<br>128 channels",
        390,
        120,
        220,
        150,
        "#EAF2FB",
        "#4C78A8",
        parent="ms_box",
        font_size=15,
    ),
    Node(
        "lstm",
        "<b>Stacked LSTM</b><br>2 layers, hidden size 64<br>cross-cycle evolution",
        1000,
        230,
        220,
        110,
        "#FFF2CC",
        "#D6B656",
        font_size=13,
    ),
    Node(
        "head",
        "<b>SOH regression head</b><br>FC64 + ReLU + Dropout(0.4)<br>FC1 + Sigmoid",
        1280,
        220,
        240,
        130,
        "#FFE6CC",
        "#D79B00",
        font_size=12,
    ),
    Node(
        "output",
        "<b>SOH estimate</b><br>ŷ in [0, 1]",
        1580,
        235,
        180,
        100,
        "#D5E8D4",
        "#82B366",
        font_size=13,
    ),
    Node(
        "label_input",
        "<b>Available targets</b><br>SOH y and mask m",
        30,
        70,
        200,
        70,
        "#F5F5F5",
        "#666666",
        parent="objective_box",
        font_size=12,
    ),
    Node(
        "order_input",
        "<b>Trajectory metadata</b><br>battery ID + cycle index",
        30,
        160,
        200,
        70,
        "#F5F5F5",
        "#666666",
        parent="objective_box",
        font_size=12,
    ),
    Node(
        "objective",
        "<b>Masked supervision + structural prior</b><br>ℒ<sub>total</sub> = ℒ<sub>masked-MSE</sub> + λ<sub>mono</sub>ℒ<sub>mono</sub><br>λ<sub>mono</sub> = 0.3, δ = 0.005, cycle ≥ 300",
        300,
        80,
        400,
        130,
        "#F8CECC",
        "#B85450",
        parent="objective_box",
        font_size=13,
    ),
]


EDGES: List[Edge] = [
    Edge("e_input_local", "input", "branch_local", [(250, 285), (270, 285), (270, 240), (330, 240)]),
    Edge("e_input_mid", "input", "branch_mid", [(250, 285), (280, 285), (280, 330), (330, 330)]),
    Edge("e_input_long", "input", "branch_long", [(250, 285), (270, 285), (270, 420), (330, 420)]),
    Edge("e_local_fusion", "branch_local", "fusion", [(560, 240), (620, 240), (620, 275), (680, 275)]),
    Edge("e_mid_fusion", "branch_mid", "fusion", [(560, 330), (620, 330), (620, 315), (680, 315)]),
    Edge("e_long_fusion", "branch_long", "fusion", [(560, 420), (630, 420), (630, 355), (680, 355)]),
    Edge("e_fusion_lstm", "fusion", "lstm", [(900, 315), (960, 315), (960, 285), (1000, 285)]),
    Edge("e_lstm_head", "lstm", "head", [(1220, 285), (1250, 285), (1250, 285), (1280, 285)]),
    Edge("e_head_output", "head", "output", [(1520, 285), (1550, 285), (1550, 285), (1580, 285)]),
    Edge("e_label_objective", "label_input", "objective", [(1230, 625), (1260, 625), (1260, 640), (1300, 640)]),
    Edge("e_order_objective", "order_input", "objective", [(1230, 715), (1260, 715), (1260, 690), (1300, 690)]),
    Edge("e_prediction_objective", "output", "objective", [(1670, 335), (1670, 490), (1500, 490), (1500, 600)], color="#B85450", dashed=True),
]


LEGEND = [
    ("Input / output", "#D5E8D4", "#82B366"),
    ("Convolution / fusion", "#DAE8FC", "#6C8EBF"),
    ("Temporal model", "#FFF2CC", "#D6B656"),
    ("Regression", "#FFE6CC", "#D79B00"),
    ("Training objective", "#F8CECC", "#B85450"),
]


def absolute_node(node: Node) -> Node:
    if node.parent == "1":
        return node
    parent = CONTAINERS[node.parent]
    return Node(
        node.node_id,
        node.label,
        parent.x + node.x,
        parent.y + node.y,
        node.width,
        node.height,
        node.fill,
        node.stroke,
        parent=node.parent,
        font_size=node.font_size,
        dashed=node.dashed,
        container=node.container,
    )


def plain_label(label: str) -> str:
    text = label.replace("<br>", "\n").replace("<br/>", "\n")
    text = re.sub(r"<sub>(.*?)</sub>", r"$_{\1}$", text)
    text = re.sub(r"<sup>(.*?)</sup>", r"$^{\1}$", text)
    text = re.sub(r"</?b>", "", text)
    return html.unescape(text)


def render_preview(output_stem: Path) -> List[Path]:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(12.0, 6.25))
    ax.set_xlim(0, CANVAS_WIDTH)
    ax.set_ylim(CANVAS_HEIGHT, 0)
    ax.axis("off")

    for container in CONTAINERS.values():
        patch = FancyBboxPatch(
            (container.x, container.y),
            container.width,
            container.height,
            boxstyle="round,pad=0.015,rounding_size=14",
            facecolor=container.fill,
            edgecolor=container.stroke,
            linewidth=1.6,
            linestyle="--" if container.dashed else "-",
            zorder=0,
        )
        ax.add_patch(patch)
        ax.text(
            container.x + 16,
            container.y + 25,
            plain_label(container.label),
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=container.stroke,
        )

    absolute_nodes = {node.node_id: absolute_node(node) for node in NODES}
    for node in absolute_nodes.values():
        preview_font_size = 8.2 if node.font_size <= 12 else (8.8 if node.font_size <= 14 else 9.4)
        patch = FancyBboxPatch(
            (node.x, node.y),
            node.width,
            node.height,
            boxstyle="round,pad=0.015,rounding_size=12",
            facecolor=node.fill,
            edgecolor=node.stroke,
            linewidth=1.45,
            zorder=2,
        )
        ax.add_patch(patch)
        ax.text(
            node.x + node.width / 2,
            node.y + node.height / 2,
            plain_label(node.label),
            ha="center",
            va="center",
            fontsize=preview_font_size,
            color="#222222",
            linespacing=1.18,
            zorder=3,
        )

    for edge in EDGES:
        path = MplPath(
            edge.points,
            [MplPath.MOVETO] + [MplPath.LINETO] * (len(edge.points) - 1),
        )
        arrow = FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.35,
            linestyle="--" if edge.dashed else "-",
            color=edge.color,
            zorder=1,
        )
        ax.add_patch(arrow)

    legend_x = 260
    for label, fill, stroke in LEGEND:
        ax.add_patch(Rectangle((legend_x, 36), 32, 18, facecolor=fill, edgecolor=stroke, linewidth=1.1))
        ax.text(legend_x + 42, 45, label, ha="left", va="center", fontsize=9.4)
        legend_x += 270

    return save_figure(fig, output_stem, dpi=600)


def _node_style(node: Node) -> str:
    style = (
        "rounded=1;arcSize=12;whiteSpace=wrap;html=1;"
        f"fillColor={node.fill};strokeColor={node.stroke};strokeWidth=1.5;"
        f"fontFamily=Times New Roman;fontSize={node.font_size};fontColor=#222222;"
        "align=center;verticalAlign=middle;spacing=6;"
    )
    if node.container:
        style += "container=1;pointerEvents=0;verticalAlign=top;align=left;spacingTop=10;spacingLeft=12;fontStyle=1;"
    if node.dashed:
        style += "dashed=1;dashPattern=6 4;"
    return style


def _add_geometry(parent: ET.Element, node: Node) -> None:
    ET.SubElement(
        parent,
        "mxGeometry",
        {
            "x": str(node.x),
            "y": str(node.y),
            "width": str(node.width),
            "height": str(node.height),
            "as": "geometry",
        },
    )


def write_drawio(path: Path) -> None:
    mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "version": "26.0.0"})
    diagram = ET.SubElement(mxfile, "diagram", {"name": "PI-MSCL architecture", "id": "pi_mscl_v2"})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": "1800",
            "dy": "830",
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "0",
            "math": "1",
            "shadow": "0",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    for container in CONTAINERS.values():
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": container.node_id,
                "value": container.label,
                "style": _node_style(container),
                "vertex": "1",
                "parent": "1",
            },
        )
        _add_geometry(cell, container)

    for node in NODES:
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": node.node_id,
                "value": node.label,
                "style": _node_style(node),
                "vertex": "1",
                "parent": node.parent,
            },
        )
        _add_geometry(cell, node)

    for index, (label, fill, stroke) in enumerate(LEGEND, start=1):
        x = 260 + (index - 1) * 270
        swatch = Node(f"legend_swatch_{index}", "", x, 36, 32, 18, fill, stroke, font_size=12)
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": swatch.node_id,
                "value": "",
                "style": _node_style(swatch),
                "vertex": "1",
                "parent": "1",
            },
        )
        _add_geometry(cell, swatch)
        text_node = Node(f"legend_text_{index}", label, x + 42, 28, 190, 34, "none", "none", font_size=13)
        text_style = (
            "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;"
            "fontFamily=Times New Roman;fontSize=13;fontColor=#222222;"
        )
        text_cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": text_node.node_id,
                "value": label,
                "style": text_style,
                "vertex": "1",
                "parent": "1",
            },
        )
        _add_geometry(text_cell, text_node)

    for edge in EDGES:
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
            f"html=1;strokeColor={edge.color};strokeWidth=1.5;endArrow=blockThin;endFill=1;"
        )
        if edge.dashed:
            style += "dashed=1;dashPattern=6 4;"
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": edge.edge_id,
                "value": "",
                "style": style,
                "edge": "1",
                "parent": "1",
                "source": edge.source,
                "target": edge.target,
            },
        )
        geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        points = ET.SubElement(geometry, "Array", {"as": "points"})
        for x, y in edge.points[1:-1]:
            ET.SubElement(points, "mxPoint", {"x": str(x), "y": str(y)})

    ET.indent(mxfile, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(mxfile)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drawio",
        type=Path,
        default=PROJECT_ROOT / "docs" / "PI-MSCL_architecture_v2.drawio",
    )
    parser.add_argument(
        "--preview-stem",
        type=Path,
        default=PROJECT_ROOT / "docs" / "PI-MSCL_architecture_v2_preview",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_drawio(args.drawio)
    saved = render_preview(args.preview_stem)
    print(f"Editable Draw.io: {args.drawio}")
    for output in saved:
        print(f"Preview: {output}")


if __name__ == "__main__":
    main()
