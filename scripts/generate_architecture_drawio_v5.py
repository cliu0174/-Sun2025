"""Generate the editable v5 PI-MSCL architecture figure.

The visual grammar follows high-impact battery/sequence-model papers: compact
signal glyphs, colour-coded feature-map slabs, and an explicitly unrolled
stacked recurrent block.  The scientific structure is project-specific and
matches the manuscript implementation.
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
_BASE_SPEC = importlib.util.spec_from_file_location("pi_mscl_drawio_base_v5", _BASE_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise ImportError(f"Cannot load architecture base module from {_BASE_PATH}")
base = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_SPEC.name] = base
_BASE_SPEC.loader.exec_module(base)


def reset_canvas() -> None:
    base.W, base.H = 1800, 860
    base.PANELS = {
        "panel_a": base.Panel("panel_a", 30, 45, 1740, 535, "(a) PI-MSCL under sparse lifecycle supervision"),
        "panel_b": base.Panel("panel_b", 30, 610, 1740, 210, "(b) Learning with a fixed cycle-level label budget"),
    }
    base.BOXES.clear()
    base.DOTS.clear()
    base.PATHS.clear()
    base.LABELS.clear()


def add_slab_stack(prefix: str, parent: str, x: int, y: int, fill: str, stroke: str) -> None:
    """Add a compact pseudo-3D stack of feature maps."""
    for j in range(5, -1, -1):
        base.add_box(
            f"{prefix}_slab_{j}", parent, x + 8 * j, y - 5 * j, 72, 70,
            fill=fill, stroke=stroke, rounded=False, opacity=58 + 7 * j,
        )


def add_lstm_cell(prefix: str, parent: str, x: int, y: int, fill: str, stroke: str) -> None:
    """Draw a recurrent cell with a minimal internal gate signature."""
    base.add_box(prefix, parent, x, y, 78, 72, fill=fill, stroke=stroke)
    gate_colors = ("#80B1D3", "#B3DE69", "#FDB462", "#BC80BD")
    for j, color in enumerate(gate_colors):
        base.add_dot(f"{prefix}_gate_{j}", parent, x + 15 + 16 * j, y + 26, 5, color, "#FFFFFF", 0.7)
    base.add_label(f"{prefix}_label", parent, x + 12, y + 43, 54, 22, "LSTM", 10, bold=True)


def build_spec() -> None:
    reset_canvas()
    box, dot, path, label = base.add_box, base.add_dot, base.add_path, base.add_label

    # Input card: complete features and a sparse SOH target live on one cycle axis.
    label("input_heading", "panel_a", 35, 52, 205, 26, "Operating trajectory", 13, bold=True, align="left")
    box("input_card", "panel_a", 35, 88, 210, 360, fill="#F7FAFC", stroke="#AAB7C4")
    label("input_voltage", "panel_a", 52, 103, 68, 20, "Voltage", 9, bold=True, align="left")
    path("voltage_curve", "panel_a", [(54, 164), (78, 142), (104, 133), (132, 130), (162, 132), (193, 146), (224, 182)], color="#3182BD", width=2.2, curved=True)
    label("input_current", "panel_a", 52, 190, 68, 20, "Current", 9, bold=True, align="left")
    path("current_curve", "panel_a", [(54, 253), (84, 248), (115, 245), (145, 247), (172, 258), (198, 281), (224, 303)], color="#35A98B", width=2.2, curved=True)
    label("input_soh", "panel_a", 52, 307, 80, 20, "SOH labels", 9, bold=True, align="left")
    soh_points = [(55, 350), (79, 354), (103, 360), (127, 368), (151, 379), (175, 392), (199, 408), (224, 426)]
    path("input_soh_curve", "panel_a", soh_points, color="#64748B", width=1.8, curved=True)
    for i, (x, y) in enumerate(soh_points):
        if i in {0, 2, 5, 7}:
            dot(f"input_label_{i}", "panel_a", x, y, 6, "#F28E2B", "#B96313", 1.1)
        else:
            dot(f"input_unlabel_{i}", "panel_a", x, y, 4, "#FFFFFF", "#A0A0A0", 1.0)
    # Keep this note visually separate from the tensor-shape annotation at the
    # lower edge of the panel; the extra whitespace matters at single-column size.
    label("input_policy", "panel_a", 50, 445, 160, 32, "mask targets only", 9, color="#B96313")

    # Tensor glyph between the measured trajectories and convolutional branches.
    for j in range(4, -1, -1):
        box(f"tensor_{j}", "panel_a", 275 + 8 * j, 214 - 7 * j, 82, 112, fill="#E6F4EA", stroke="#62A66A", rounded=False, opacity=62 + 7 * j)
    label("tensor_shape", "panel_a", 260, 482, 135, 35, "40 cycles × 16 features", 10, bold=True)
    path("input_to_tensor", "panel_a", [(245, 250), (275, 250)], color="#60717F", width=1.5, arrow=True)

    # Three receptive-field branches. Repeated layers are encoded graphically,
    # while the single shared annotation avoids text-heavy branch cards.
    branch_specs = [
        ("short", 105, "k = 3", "short", "#CFE8F3", "#3F88A8"),
        ("medium", 235, "k = 7", "medium", "#D8EFE6", "#3C9D79"),
        ("long", 365, "k = 15", "long", "#E7DCF2", "#8B6BAE"),
    ]
    label("cnn_heading", "panel_a", 420, 52, 300, 26, "Multi-scale Conv1D", 13, bold=True, align="left")
    label("cnn_shared", "panel_a", 570, 80, 180, 24, "2 layers · 64 channels", 9, color="#5F6B75")
    for name, y, kernel, scale, fill, stroke in branch_specs:
        label(f"{name}_scale", "panel_a", 408, y + 24, 68, 24, scale, 9, bold=True, color=stroke, align="left")
        path(f"{name}_signal", "panel_a", [(485, y + 54), (503, y + 39), (520, y + 61), (538, y + 30), (557, y + 49)], color=stroke, width=2.0)
        box(f"{name}_kernel", "panel_a", 510, y + 22, 30, 54, kernel, "#FFFFFF", stroke, 9)
        path(f"{name}_conv_arrow", "panel_a", [(563, y + 50), (595, y + 50)], color=stroke, width=1.4, arrow=True)
        add_slab_stack(name, "panel_a", 595, y + 18, fill, stroke)
        tensor_exit_y = {"short": 235, "medium": 270, "long": 305}[name]
        path(f"tensor_to_{name}", "panel_a", [(357, tensor_exit_y), (390, tensor_exit_y), (390, y + 50), (408, y + 50)], color=stroke, width=1.3, arrow=True)

    # Concatenation and 1x1 fusion: three coloured streams converge on one slab.
    label("fusion_heading", "panel_a", 770, 52, 135, 26, "Feature fusion", 12, bold=True)
    for j in range(6, -1, -1):
        box(f"fusion_slab_{j}", "panel_a", 775 + 7 * j, 218 - 5 * j, 82, 116, fill="#DDE8F6", stroke="#4C78A8", rounded=False, opacity=58 + 6 * j)
    label("fusion_note", "panel_a", 742, 360, 160, 48, "concat + 1 × 1 Conv\n128 channels", 10, bold=True)
    for name, y, _, _, _, stroke in branch_specs:
        path(f"{name}_to_fusion", "panel_a", [(707, y + 52), (742, y + 52), (742, 276), (775, 276)], color=stroke, width=1.3, arrow=True)

    # Two LSTM layers are shown as two rows; three glyphs denote time steps.
    label("lstm_heading", "panel_a", 930, 52, 370, 26, "Cross-cycle dynamics", 13, bold=True, align="left")
    label("time_axis", "panel_a", 990, 91, 292, 22, "t − 1                 t                 t + 1", 9, color="#64748B")
    xs = (955, 1065, 1175)
    ys = (140, 280)
    for layer_no, y in enumerate(ys, start=1):
        label(f"layer_{layer_no}_label", "panel_a", 902, y + 21, 44, 28, f"L{layer_no}", 10, bold=True, color="#9A6A16", align="right")
        for time_no, x in enumerate(xs):
            add_lstm_cell(f"lstm_{layer_no}_{time_no}", "panel_a", x, y, "#FFF1C9", "#D39B25")
            if time_no < 2:
                path(f"lstm_h_{layer_no}_{time_no}", "panel_a", [(x + 78, y + 36), (x + 110, y + 36)], color="#9A7A30", width=1.3, arrow=True)
    for time_no, x in enumerate(xs):
        path(f"lstm_vertical_{time_no}", "panel_a", [(x + 39, 212), (x + 39, 280)], color="#D39B25", width=1.2, arrow=True)
    path("fusion_to_lstm", "panel_a", [(857, 276), (914, 276), (914, 176), (955, 176)], color="#4C78A8", width=1.5, arrow=True)
    label("lstm_note", "panel_a", 968, 383, 270, 25, "2 layers · hidden size 64", 9, color="#7A5E1E")

    # Compact head and an actual curve make the regression endpoint visual.
    label("head_heading", "panel_a", 1307, 52, 135, 26, "Regression head", 13, bold=True)
    box("head_fc", "panel_a", 1320, 140, 92, 54, "FC 64", "#FDE3C8", "#E18E3B", 10)
    box("head_reg", "panel_a", 1320, 216, 92, 86, "ReLU\nDropout 0.4\nSigmoid", "#FAD8B4", "#D97822", 9)
    path("lstm_to_head", "panel_a", [(1253, 316), (1285, 316), (1285, 167), (1320, 167)], color="#9A7A30", width=1.5, arrow=True)
    path("head_fc_reg", "panel_a", [(1366, 194), (1366, 216)], color="#D97822", width=1.3, arrow=True)

    label("output_heading", "panel_a", 1470, 52, 210, 26, "Cycle-wise SOH", 13, bold=True)
    path("out_y", "panel_a", [(1490, 136), (1490, 395)], width=1.0)
    path("out_x", "panel_a", [(1490, 395), (1690, 395)], width=1.0)
    out_curve = [(1498, 155), (1530, 165), (1560, 180), (1590, 205), (1620, 240), (1650, 296), (1680, 370)]
    path("output_curve", "panel_a", out_curve, color="#2F7D5A", width=3.0, curved=True)
    path("head_to_output", "panel_a", [(1412, 259), (1450, 259), (1450, 265), (1490, 265)], color="#D97822", width=1.5, arrow=True)
    label("output_note", "panel_a", 1520, 407, 150, 25, "0 ≤ SOH ≤ 1", 9, color="#2F7D5A")

    # Bottom ribbon: two distinct evidence paths and one concise objective.
    label("mask_title", "panel_b", 45, 50, 215, 24, "Sparse-label data path", 11, bold=True, color="#B96313", align="left")
    path("mask_line", "panel_b", [(55, 126), (90, 119), (125, 113), (160, 105), (195, 95), (230, 84)], color="#64748B", width=1.6, curved=True)
    for i, (x, y) in enumerate(((55, 126), (90, 119), (125, 113), (160, 105), (195, 95), (230, 84))):
        if i in {0, 2, 5}:
            dot(f"mask_dot_{i}", "panel_b", x, y, 6, "#F28E2B", "#B96313", 1.0)
        else:
            dot(f"mask_open_{i}", "panel_b", x, y, 4, "#FFFFFF", "#A0A0A0", 1.0)
    box("masked_mse", "panel_b", 290, 74, 190, 82, "Masked MSE\nretained targets only", "#FFF1E6", "#E18E3B", 10)
    path("mask_to_mse", "panel_b", [(230, 112), (290, 112)], color="#E18E3B", width=1.4, arrow=True)

    label("physics_title", "panel_b", 570, 50, 230, 24, "Unlabeled trajectory path", 11, bold=True, color="#A64545", align="left")
    phys_curve = [(580, 90), (620, 97), (660, 109), (700, 102), (740, 128), (780, 142)]
    path("physics_curve", "panel_b", phys_curve, color="#2F7D5A", width=2.2, curved=True)
    path("physics_violation", "panel_b", [phys_curve[2], phys_curve[3]], color="#C44E52", width=3.3)
    box("mono_loss", "panel_b", 835, 74, 225, 82, "Soft monotonicity\neligible predicted pairs", "#FBE2E0", "#C44E52", 10)
    path("physics_to_mono", "panel_b", [(780, 120), (835, 120)], color="#C44E52", width=1.4, arrow=True)

    box("total_loss", "panel_b", 1190, 62, 255, 104, "L = Ldata + 0.3 Lmono", "#F3C6C2", "#A64545", 12)
    path("mse_to_total", "panel_b", [(480, 115), (520, 115), (520, 177), (1145, 177), (1145, 93), (1190, 93)], color="#E18E3B", width=1.3, arrow=True)
    path("mono_to_total", "panel_b", [(1060, 115), (1128, 115), (1128, 137), (1190, 137)], color="#C44E52", width=1.3, arrow=True)
    label("physics_params", "panel_b", 1110, 178, 420, 22, "ε = .005 · cmin = 300 · K = 40 · α = .2", 8, color="#755252")

    box("takeaway", "panel_b", 1510, 62, 205, 104, "Complete features\n+ sparse targets\n+ structural prior", "#E5F3EA", "#5A9B70", 10)
    path("total_to_takeaway", "panel_b", [(1445, 114), (1510, 114)], color="#5A9B70", width=1.5, arrow=True)


def use_sans_serif_in_drawio(path: Path) -> None:
    """Use one clean sans-serif family while preserving native editable text."""
    tree = ET.parse(path)
    root = tree.getroot()
    for cell in root.iter("mxCell"):
        style = cell.attrib.get("style", "")
        if "fontFamily=Times New Roman" in style:
            cell.set("style", style.replace("fontFamily=Times New Roman", "fontFamily=Arial"))
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def apply_v5_style() -> None:
    """Clean sans-serif typography for the publication preview/export."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drawio", type=Path, default=PROJECT_ROOT / "docs" / "PI-MSCL_architecture_v5.drawio")
    parser.add_argument("--preview-stem", type=Path, default=PROJECT_ROOT / "docs" / "PI-MSCL_architecture_v5_preview")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_spec()
    base.write_drawio(args.drawio)
    use_sans_serif_in_drawio(args.drawio)
    base.apply_paper_style = apply_v5_style
    saved = base.render_preview(args.preview_stem)
    print(f"Editable Draw.io: {args.drawio}")
    for item in saved:
        print(f"Preview: {item}")


if __name__ == "__main__":
    main()
