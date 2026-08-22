"""Render a single-column-readable, editable PI-MSCL architecture figure.

The diagram deliberately favors a small number of large, connected visual
elements over an exhaustive depiction of every intermediate tensor. Detailed
dimensions and hyperparameters remain in the manuscript tables.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib as mpl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_BASE_PATH = Path(__file__).with_name("generate_architecture_drawio_v3.py")
_SPEC = importlib.util.spec_from_file_location("pi_mscl_drawio_base_v6", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load architecture drawing utilities from {_BASE_PATH}")
base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = base
_SPEC.loader.exec_module(base)


def reset_canvas() -> None:
    base.W, base.H = 1400, 1090
    base.PANELS = {
        "panel_a": base.Panel("panel_a", 30, 35, 1340, 600, "a  PI-MSCL architecture"),
        "panel_b": base.Panel("panel_b", 30, 670, 1340, 380, "b  Sparse-label learning objective"),
    }
    base.BOXES.clear()
    base.DOTS.clear()
    base.PATHS.clear()
    base.LABELS.clear()


def add_feature_maps(prefix: str, y: int, fill: str, stroke: str, kernel: str) -> None:
    """One large branch label and three layered slabs, readable at column width."""
    box, label = base.add_box, base.add_label
    label(f"{prefix}_label", "panel_a", 346, y + 14, 72, 28, kernel, 11, bold=True, color=stroke, align="left")
    for index in range(3, -1, -1):
        box(
            f"{prefix}_map_{index}", "panel_a", 446 + index * 10, y + 8 - index * 7,
            72, 74, fill=fill, stroke=stroke, rounded=False, opacity=65 + 8 * index,
        )


def build_spec() -> None:
    reset_canvas()
    box, dot, path, label = base.add_box, base.add_dot, base.add_path, base.add_label

    # (a) Large left-to-right explanatory chain.
    box("input", "panel_a", 55, 100, 205, 330, fill="#F7FAFC", stroke="#AAB7C4")
    label("input_title", "panel_a", 76, 118, 165, 30, "Complete operating\nfeatures", 12, bold=True)
    path("voltage", "panel_a", [(84, 190), (112, 160), (150, 154), (184, 173), (222, 228)], color="#3182BD", width=2.7, curved=True)
    path("current", "panel_a", [(84, 264), (122, 254), (160, 260), (194, 288), (222, 326)], color="#35A98B", width=2.7, curved=True)
    label("input_note", "panel_a", 72, 344, 175, 28, "40 cycles × 16 features", 9, bold=True)
    # Sparse labels use open versus filled points rather than a long explanation.
    soh = [(84, 384), (112, 388), (140, 397), (168, 407), (196, 421), (224, 438)]
    path("sparse_soh", "panel_a", soh, color="#65758A", width=1.8, curved=True)
    for index, (x, y) in enumerate(soh):
        dot(f"target_{index}", "panel_a", x, y, 6 if index in {0, 2, 5} else 4,
            "#F28E2B" if index in {0, 2, 5} else "#FFFFFF", "#B96313" if index in {0, 2, 5} else "#9AA4AF", 1.0)
    label("mask_note", "panel_a", 72, 453, 178, 24, "SOH targets masked only", 9, color="#B96313")

    label("cnn_title", "panel_a", 348, 88, 220, 28, "Multi-scale Conv1D", 13, bold=True, align="left")
    label("cnn_note", "panel_a", 348, 460, 245, 24, "Three receptive fields: 3 / 7 / 15", 9, color="#5F6B75", align="left")
    add_feature_maps("short", 145, "#CFE8F3", "#3F88A8", "k = 3")
    add_feature_maps("medium", 245, "#D8EFE6", "#3C9D79", "k = 7")
    add_feature_maps("long", 345, "#E7DCF2", "#8B6BAE", "k = 15")
    path("input_to_cnn", "panel_a", [(260, 263), (318, 263), (318, 185), (410, 185)], color="#60717F", width=1.5, arrow=True)
    path("input_to_cnn_2", "panel_a", [(318, 263), (318, 285), (410, 285)], color="#60717F", width=1.5, arrow=True)
    path("input_to_cnn_3", "panel_a", [(318, 263), (318, 385), (410, 385)], color="#60717F", width=1.5, arrow=True)

    box("fusion", "panel_a", 620, 220, 135, 130, "Fuse\n1 × 1 Conv\n128 channels", "#DDE8F6", "#4C78A8", 11)
    for y, stroke in ((185, "#3F88A8"), (285, "#3C9D79"), (385, "#8B6BAE")):
        path(f"branch_to_fusion_{y}", "panel_a", [(548, y), (585, y), (585, 285), (620, 285)], color=stroke, width=1.35, arrow=True)

    box("lstm", "panel_a", 820, 180, 205, 210, "Two-layer LSTM\nCross-cycle state modeling\n(hidden size 64)", "#FFF1C9", "#D39B25", 12)
    # Minimal recurrent-cell signature, placed below the label text so the
    # dots never sit under the box caption regardless of font metrics.
    for x in (858, 895, 932, 969):
        dot(f"lstm_gate_{x}", "panel_a", x, 358, 7, "#80B1D3", "#FFFFFF", 0.8)
    path("fusion_to_lstm", "panel_a", [(755, 285), (820, 285)], color="#4C78A8", width=1.6, arrow=True)

    box("head", "panel_a", 1090, 220, 120, 130, "FC 64\nReLU\nDropout\nSigmoid", "#FDE3C8", "#E18E3B", 11)
    path("lstm_to_head", "panel_a", [(1025, 285), (1090, 285)], color="#9A7A30", width=1.5, arrow=True)

    label("output_title", "panel_a", 1240, 110, 100, 26, "Predicted SOH", 11, bold=True)
    path("output_y", "panel_a", [(1248, 160), (1248, 420)], width=1.0)
    path("output_x", "panel_a", [(1248, 420), (1340, 420)], width=1.0)
    path("output_curve", "panel_a", [(1256, 182), (1272, 193), (1288, 214), (1304, 255), (1322, 320), (1334, 390)], color="#2F7D5A", width=3.0, curved=True)
    label("output_note", "panel_a", 1248, 434, 95, 22, "0 ≤ SOH ≤ 1", 9, color="#2F7D5A")
    path("head_to_output", "panel_a", [(1210, 285), (1248, 285)], color="#E18E3B", width=1.5, arrow=True)

    # (b) Two explicitly separated paths feed the same objective.
    label("b_left", "panel_b", 76, 82, 250, 25, "Retained SOH targets", 11, bold=True, color="#B96313", align="left")
    box("masked_mse", "panel_b", 100, 140, 255, 105, "Masked data loss\nonly retained targets", "#FFF1E6", "#E18E3B", 12)
    label("b_mid", "panel_b", 445, 82, 290, 25, "Complete unlabeled trajectory", 11, bold=True, color="#A64545", align="left")
    path("b_traj", "panel_b", [(450, 185), (490, 170), (530, 186), (570, 214), (610, 238), (650, 228), (690, 250)], color="#2F7D5A", width=3.0, curved=True)
    path("b_violation", "panel_b", [(610, 238), (650, 228)], color="#C44E52", width=4.0)
    box("mono", "panel_b", 750, 140, 255, 105, "Soft monotonicity\nstructural loss", "#FBE2E0", "#C44E52", 12)
    path("traj_to_mono", "panel_b", [(690, 215), (750, 215)], color="#C44E52", width=1.5, arrow=True)
    box("objective", "panel_b", 1070, 120, 230, 145, "Training objective\n\nL = Ldata + λLmono", "#F3C6C2", "#A64545", 13)
    # Route the numerical-loss path through otherwise unused lower whitespace.
    # Parameters belong in Table 2, not in this overview figure.
    path("mse_to_obj", "panel_b", [(355, 192), (410, 192), (410, 280), (1035, 280), (1035, 225), (1070, 225)], color="#E18E3B", width=1.4, arrow=True)
    path("mono_to_obj", "panel_b", [(1005, 192), (1070, 192)], color="#C44E52", width=1.4, arrow=True)


def use_sans_serif(path: Path) -> None:
    tree = ET.parse(path)
    for cell in tree.getroot().iter("mxCell"):
        style = cell.attrib.get("style", "")
        if "fontFamily=Times New Roman" in style:
            cell.set("style", style.replace("fontFamily=Times New Roman", "fontFamily=Arial"))
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drawio", type=Path, default=PROJECT_ROOT / "docs" / "PI-MSCL_architecture_single_column.drawio")
    parser.add_argument("--preview-stem", type=Path, default=PROJECT_ROOT / "docs" / "PI-MSCL_architecture_single_column_preview")
    args = parser.parse_args()
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "svg.fonttype": "none", "pdf.fonttype": 42})
    build_spec()
    base.write_drawio(args.drawio)
    use_sans_serif(args.drawio)
    paths = base.render_preview(args.preview_stem)
    print(f"Editable Draw.io: {args.drawio}")
    for path in paths:
        print(f"Preview: {path}")


if __name__ == "__main__":
    main()
