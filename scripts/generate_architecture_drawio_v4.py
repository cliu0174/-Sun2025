"""Generate the editable v4 PI-MSCL architecture figure.

Visual reference: Xu et al., Energy 276 (2023) 127585.  The borrowed visual
grammar is limited to tensor slices, parallel feature-map stacks, and an
explicitly unrolled recurrent block.  The scientific content is the project's
own three-scale Conv1D, two-layer LSTM, sparse-label protocol, and physics loss.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_BASE_PATH = Path(__file__).with_name("generate_architecture_drawio_v3.py")
_BASE_SPEC = importlib.util.spec_from_file_location("pi_mscl_drawio_base", _BASE_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise ImportError(f"Cannot load architecture base module from {_BASE_PATH}")
base = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_SPEC.name] = base
_BASE_SPEC.loader.exec_module(base)


def reset_canvas() -> None:
    base.PANELS = {
        "panel_a": base.Panel("panel_a", 30, 55, 1740, 660, "(a) PI-MSCL network architecture"),
        "panel_b": base.Panel("panel_b", 30, 745, 650, 270, "(b) Controlled sparse supervision"),
        "panel_c": base.Panel("panel_c", 710, 745, 1060, 270, "(c) Physics-informed learning"),
    }
    base.BOXES.clear()
    base.DOTS.clear()
    base.PATHS.clear()
    base.LABELS.clear()


def build_spec() -> None:
    reset_canvas()
    box, dot, path, label = base.add_box, base.add_dot, base.add_path, base.add_label

    # Input tensor.
    label("a_input_title", "panel_a", 24, 78, 150, 28, "Input sequence", 13, bold=True)
    box("a_tensor_back2", "panel_a", 55, 232, 105, 130, fill="#EEF5EC", stroke="#A6C79A")
    box("a_tensor_back1", "panel_a", 45, 222, 105, 130, fill="#E6F1E2", stroke="#8FB982")
    box("a_tensor_front", "panel_a", 35, 212, 105, 130, "B × 40 × 16", "#DDEED8", "#70A35B", 13)
    label("a_input_note", "panel_a", 22, 370, 158, 58, "40-cycle window\n16 charging features", 11, color="#4E6B45")

    # Three-scale CNN: each lane contains a convolution kernel and stacked maps.
    branches = [
        ("short", 95, "Short-range branch", "k = 3", "#D7E8F6", "#5B87B2"),
        ("medium", 265, "Medium-range branch", "k = 7", "#DDEAF5", "#668EAF"),
        ("long", 435, "Long-range branch", "k = 15", "#E5EEF5", "#7895AB"),
    ]
    for name, y, title, kernel, fill, stroke in branches:
        box(f"a_{name}_lane", "panel_a", 210, y, 390, 125, fill="#FAFCFE", stroke=stroke)
        label(f"a_{name}_title", "panel_a", 225, y + 8, 170, 26, title, 12, bold=True, align="left")
        box(f"a_{name}_kernel", "panel_a", 230, y + 48, 78, 54, f"Conv1D\n{kernel}", fill, stroke, 11)
        for j in range(5):
            box(
                f"a_{name}_map_{j}", "panel_a", 355 + 18 * j, y + 42 - 3 * j,
                60, 64, fill=fill, stroke=stroke, rounded=False, opacity=92,
            )
        label(f"a_{name}_channels", "panel_a", 470, y + 48, 112, 50, "Conv1D × 2\n64 channels", 11)
        path(f"a_{name}_inner", "panel_a", [(308, y + 75), (350, y + 75)], color=stroke, width=1.4, arrow=True)
        path(f"a_input_{name}", "panel_a", [(140, 277), (180, 277), (180, y + 63), (210, y + 63)], width=1.4, arrow=True)

    # Concatenation and 1x1 fusion are represented as an upright tensor stack.
    label("a_fusion_title", "panel_a", 625, 112, 165, 28, "Feature fusion", 13, bold=True)
    for j in range(7):
        box(
            f"a_fusion_slice_{j}", "panel_a", 655 + 9 * j, 225 - 5 * j,
            72, 190, fill="#DCEAF7", stroke="#4C78A8", rounded=False, opacity=90,
        )
    label("a_fusion_note", "panel_a", 620, 438, 190, 55, "Concatenate + 1 × 1 Conv\n128 channels", 11)
    for name, y, _, _, _, stroke in branches:
        path(f"a_{name}_fusion", "panel_a", [(600, y + 63), (630, y + 63), (630, 315), (655, 315)], color=stroke, width=1.4, arrow=True)

    # Explicit 2-layer LSTM avoids the common ambiguity between cells and layers.
    label("a_lstm_title", "panel_a", 810, 80, 450, 32, "Cross-cycle temporal model: two-layer LSTM (hidden size 64)", 13, bold=True)
    time_labels = ["t − 1", "t", "…", "t + 1"]
    cell_xs = [865, 965, 1065, 1165]
    for layer_no, y in ((1, 205), (2, 375)):
        label(f"a_layer_{layer_no}", "panel_a", 790, y + 22, 65, 28, f"Layer {layer_no}", 11, bold=True, align="right")
        for i, (x, time_text) in enumerate(zip(cell_xs, time_labels)):
            if time_text == "…":
                label(f"a_l{layer_no}_ellipsis", "panel_a", x, y + 18, 70, 40, "•••", 14, color="#777777")
                continue
            box(f"a_l{layer_no}_cell_{i}", "panel_a", x, y, 70, 78, "LSTM", "#FFF0C2", "#CDA83D", 11)
            if layer_no == 1:
                label(f"a_l{layer_no}_time_{i}", "panel_a", x, y - 30, 70, 24, time_text, 10, color="#666666")
        path(f"a_l{layer_no}_h1", "panel_a", [(935, y + 39), (965, y + 39)], width=1.25, arrow=True)
        path(f"a_l{layer_no}_h2", "panel_a", [(1035, y + 39), (1065, y + 39)], width=1.25, dashed=True)
        path(f"a_l{layer_no}_h3", "panel_a", [(1135, y + 39), (1165, y + 39)], width=1.25, dashed=True)
    for i, x in enumerate((900, 1000, 1200)):
        path(f"a_lstm_vertical_{i}", "panel_a", [(x, 283), (x, 375)], color="#B08B2F", width=1.2, arrow=True)
    path("a_fusion_lstm", "panel_a", [(727, 315), (810, 315), (810, 244), (865, 244)], width=1.5, arrow=True)

    # Regression head and trajectory output.
    box("a_head", "panel_a", 1285, 275, 125, 150, "FC 64\nDropout\nSigmoid", "#FFE6CC", "#D79B00", 12)
    path("a_lstm_head", "panel_a", [(1235, 414), (1260, 414), (1260, 350), (1285, 350)], width=1.5, arrow=True)
    label("a_output_title", "panel_a", 1450, 112, 230, 28, "Cycle-wise SOH estimate", 13, bold=True)
    path("a_out_y", "panel_a", [(1470, 175), (1470, 470)], width=1.1)
    path("a_out_x", "panel_a", [(1470, 470), (1690, 470)], width=1.1)
    out_curve = [(1480, 190), (1515, 205), (1550, 225), (1585, 250), (1620, 292), (1655, 360), (1680, 445)]
    path("a_output_curve", "panel_a", out_curve, color="#4E7C59", width=3.0, curved=True)
    path("a_head_output", "panel_a", [(1410, 350), (1440, 350), (1440, 325), (1470, 325)], width=1.5, arrow=True)
    label("a_output_note", "panel_a", 1490, 486, 180, 28, "ŷ₁, …, ŷₜ", 11, color="#4E6B45")

    # Controlled within-cell label retention.
    path("b_y", "panel_b", [(55, 70), (55, 215)], width=1.1)
    path("b_x", "panel_b", [(55, 215), (390, 215)], width=1.1)
    sparse_curve = [(65, 82), (105, 90), (145, 101), (185, 116), (225, 132), (265, 153), (305, 178), (345, 198), (382, 210)]
    path("b_soh_curve", "panel_b", sparse_curve, color="#4E7C59", width=2.6, curved=True)
    retained = {0, 2, 5, 7}
    for i, (x, y) in enumerate(sparse_curve):
        if i in retained:
            dot(f"b_label_{i}", "panel_b", x, y, 7, "#E58B3A", "#B8641D", 1.4)
        else:
            dot(f"b_unlabeled_{i}", "panel_b", x, y, 5, "#FFFFFF", "#A0A0A0", 1.1)
    label("b_y_label", "panel_b", 5, 110, 42, 55, "SOH", 11, bold=True)
    label("b_x_label", "panel_b", 170, 220, 120, 25, "Cycle index", 11, bold=True)
    box("b_protocol", "panel_b", 425, 62, 195, 150, "Mask labels only\nKeep features + order\n\nr = 1.0, 0.7, 0.5, 0.3", "#F5F7F8", "#777777", 12)

    # Complementary data and physics objectives.
    label("c_prediction_label", "panel_c", 25, 62, 190, 26, "Predicted trajectory", 12, bold=True)
    path("c_y", "panel_c", [(45, 105), (45, 215)], width=1.0)
    path("c_x", "panel_c", [(45, 215), (245, 215)], width=1.0)
    phys_curve = [(52, 115), (82, 124), (112, 140), (142, 132), (172, 165), (205, 188), (235, 210)]
    path("c_curve", "panel_c", phys_curve, color="#4E7C59", width=2.5, curved=True)
    path("c_violation", "panel_c", [phys_curve[2], phys_curve[3]], color="#C44E52", width=3.4)
    label("c_violation_note", "panel_c", 102, 91, 95, 26, "penalized rise", 10, color="#B85450")
    box("c_data_loss", "panel_c", 295, 55, 200, 78, "Masked data loss\nLdata (mᵢ = 1)", "#F8D7D4", "#B85450", 12)
    box("c_physics_loss", "panel_c", 295, 158, 200, 78, "Soft monotonicity\nLmono (cycle ≥ 300)", "#F8D7D4", "#B85450", 12)
    box("c_total_loss", "panel_c", 585, 92, 220, 112, "Total objective\nL = Ldata + λLmono", "#F2B9B4", "#9E3F3A", 13)
    path("c_curve_data", "panel_c", [(245, 150), (270, 150), (270, 94), (295, 94)], width=1.3, arrow=True)
    path("c_curve_phys", "panel_c", [(245, 175), (270, 175), (270, 197), (295, 197)], width=1.3, arrow=True)
    path("c_data_total", "panel_c", [(495, 94), (540, 94), (540, 135), (585, 135)], width=1.3, arrow=True)
    path("c_phys_total", "panel_c", [(495, 197), (540, 197), (540, 162), (585, 162)], width=1.3, arrow=True)
    box("c_takeaway", "panel_c", 850, 82, 180, 135, "Unlabeled cycles\nstill constrain\ntrajectory structure", "#E8F2E5", "#70A35B", 12)
    path("c_total_takeaway", "panel_c", [(805, 148), (850, 148)], color="#70A35B", width=1.4, arrow=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drawio", type=Path, default=PROJECT_ROOT / "docs" / "PI-MSCL_architecture_v4.drawio")
    parser.add_argument("--preview-stem", type=Path, default=PROJECT_ROOT / "docs" / "PI-MSCL_architecture_v4_preview")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_spec()
    base.write_drawio(args.drawio)
    saved = base.render_preview(args.preview_stem)
    print(f"Editable Draw.io: {args.drawio}")
    for item in saved:
        print(f"Preview: {item}")


if __name__ == "__main__":
    main()
