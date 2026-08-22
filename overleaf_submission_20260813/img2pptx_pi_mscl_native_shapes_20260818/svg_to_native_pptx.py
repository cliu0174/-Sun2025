from __future__ import annotations

import json
import math
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

from PIL import ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from lxml import etree
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT.parent / "img2pptx_pi_mscl_proportional_cells_20260818"
SOURCE_SVG = SOURCE_DIR / "full.svg"
OUTPUT_PPTX = ROOT / "PI-MSCL_architecture_native_shapes_20260818.pptx"
OUTPUT_PPTX_UNGROUPED = ROOT / "PI-MSCL_architecture_native_shapes_fully_ungrouped_20260818.pptx"
COPIED_SVG = ROOT / "full.svg"
QA_DIR = ROOT / "qa"
AUDIT_JSON = QA_DIR / "native_shape_audit.json"

CANVAS_W = 1672.0
CANVAS_H = 941.0
SLIDE_W_IN = 13.333333
SLIDE_H_IN = SLIDE_W_IN * CANVAS_H / CANVAS_W
PX_TO_IN = SLIDE_W_IN / CANVAS_W
PX_TO_PT = PX_TO_IN * 72.0

SVG_NS = "http://www.w3.org/2000/svg"
NS = {"svg": SVG_NS}

# Avoid a single full-slide group. These semantic groups remain editable as modules;
# their children remain independently editable after entering or ungrouping the module.
GROUP_IDS = {
    "model_connectors",
    "panel_a_header",
    "input_sequence",
    "cnn_k3",
    "cnn_k3_40",
    "cnn_k3_20",
    "cnn_k3_10",
    "cnn_k7",
    "cnn_k7_40",
    "cnn_k7_20",
    "cnn_k7_10",
    "cnn_k15",
    "cnn_k15_40",
    "cnn_k15_20",
    "cnn_k15_10",
    "fusion",
    "lstm",
    "regression",
    "output_curve",
    "training_connectors",
    "panel_b_header",
    "supervised_path",
    "monotonic_path",
    "rate_path",
    "objective",
}

FONT_FILES = {
    "Arial": Path(r"C:\Windows\Fonts\arial.ttf"),
    "Times New Roman": Path(r"C:\Windows\Fonts\times.ttf"),
    "Cambria Math": Path(r"C:\Windows\Fonts\cambria.ttc"),
}


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def rgb(value: str | None, fallback: str = "000000") -> RGBColor:
    if not value or value == "none":
        value = f"#{fallback}"
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    return RGBColor.from_string(value.upper())


def opacity_of(element: ET.Element) -> float:
    try:
        return max(0.0, min(1.0, float(element.attrib.get("opacity", "1"))))
    except ValueError:
        return 1.0


def set_alpha_on_solid_fill(fill_element, opacity: float) -> None:
    if opacity >= 0.999:
        return
    srgb = fill_element.find(qn("a:srgbClr"))
    if srgb is None:
        return
    for child in list(srgb):
        if child.tag == qn("a:alpha"):
            srgb.remove(child)
    alpha = etree.Element(qn("a:alpha"))
    alpha.set("val", str(int(round(opacity * 100000))))
    srgb.append(alpha)


def apply_fill(shape, element: ET.Element) -> None:
    fill = element.attrib.get("fill", "none")
    if fill == "none":
        shape.fill.background()
        return
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    set_alpha_on_solid_fill(shape._element.spPr.solidFill, opacity_of(element))


def dash_style(dasharray: str | None):
    if not dasharray:
        return MSO_LINE_DASH_STYLE.SOLID
    parts = [float(x) for x in re.findall(r"[0-9.]+", dasharray)]
    if not parts:
        return MSO_LINE_DASH_STYLE.DASH
    if len(parts) == 2 and parts[0] <= 2.0:
        return MSO_LINE_DASH_STYLE.ROUND_DOT
    return MSO_LINE_DASH_STYLE.DASH


def apply_line(shape, element: ET.Element) -> None:
    stroke = element.attrib.get("stroke", "none")
    if stroke == "none":
        shape.line.fill.background()
        return
    shape.line.color.rgb = rgb(stroke)
    width_px = float(element.attrib.get("stroke-width", "1"))
    shape.line.width = Pt(max(0.25, width_px * PX_TO_PT))
    shape.line.dash_style = dash_style(element.attrib.get("stroke-dasharray"))


def apply_name(shape, name: str) -> None:
    shape.name = name[:250]
    # python-pptx emits a theme style reference for newly added shapes. In the
    # project's PowerPoint theme that reference adds an unintended drop shadow.
    # The SVG has no shadow, so remove the inherited style and rely entirely on
    # the explicit fill/line/text properties above.
    style = shape._element.find(qn("p:style"))
    if style is not None:
        shape._element.remove(style)


def add_rect(tree, element: ET.Element, name: str):
    x = float(element.attrib["x"])
    y = float(element.attrib["y"])
    w = float(element.attrib["width"])
    h = float(element.attrib["height"])
    rx = float(element.attrib.get("rx", "0"))
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rx > 0 else MSO_SHAPE.RECTANGLE
    shape = tree.add_shape(shape_type, Inches(x * PX_TO_IN), Inches(y * PX_TO_IN), Inches(w * PX_TO_IN), Inches(h * PX_TO_IN))
    if rx > 0 and getattr(shape, "adjustments", None):
        try:
            shape.adjustments[0] = min(0.5, rx / max(1.0, min(w, h)))
        except (IndexError, ValueError):
            pass
    apply_fill(shape, element)
    apply_line(shape, element)
    apply_name(shape, name)
    return shape


def add_ellipse(tree, element: ET.Element, name: str):
    if local_name(element.tag) == "circle":
        cx = float(element.attrib["cx"])
        cy = float(element.attrib["cy"])
        rx = ry = float(element.attrib["r"])
    else:
        cx = float(element.attrib["cx"])
        cy = float(element.attrib["cy"])
        rx = float(element.attrib["rx"])
        ry = float(element.attrib["ry"])
    shape = tree.add_shape(
        MSO_SHAPE.OVAL,
        Inches((cx - rx) * PX_TO_IN),
        Inches((cy - ry) * PX_TO_IN),
        Inches(2 * rx * PX_TO_IN),
        Inches(2 * ry * PX_TO_IN),
    )
    apply_fill(shape, element)
    apply_line(shape, element)
    apply_name(shape, name)
    return shape


def parse_points(raw: str) -> list[tuple[float, float]]:
    values = [float(v) for v in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", raw)]
    return list(zip(values[0::2], values[1::2]))


def add_freeform(tree, element: ET.Element, name: str, closed: bool):
    points = parse_points(element.attrib["points"])
    if len(points) < 2:
        return None
    builder = tree.build_freeform(points[0][0], points[0][1], scale=Inches(PX_TO_IN))
    builder.add_line_segments(points[1:], close=closed)
    shape = builder.convert_to_shape()
    apply_fill(shape, element)
    apply_line(shape, element)
    apply_name(shape, name)
    return shape


def add_line(tree, element: ET.Element, name: str):
    x1 = float(element.attrib["x1"])
    y1 = float(element.attrib["y1"])
    x2 = float(element.attrib["x2"])
    y2 = float(element.attrib["y2"])
    shape = tree.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(x1 * PX_TO_IN),
        Inches(y1 * PX_TO_IN),
        Inches(x2 * PX_TO_IN),
        Inches(y2 * PX_TO_IN),
    )
    apply_line(shape, element)
    apply_name(shape, name)
    return shape


def font_file(font_family: str, bold: bool, italic: bool) -> Path:
    if font_family == "Arial":
        suffix = "bi" if bold and italic else "bd" if bold else "i" if italic else ""
        candidate = Path(rf"C:\Windows\Fonts\arial{suffix}.ttf")
        return candidate if candidate.exists() else FONT_FILES["Arial"]
    if font_family == "Times New Roman":
        suffix = "bi" if bold and italic else "bd" if bold else "i" if italic else ""
        candidate = Path(rf"C:\Windows\Fonts\times{suffix}.ttf")
        return candidate if candidate.exists() else FONT_FILES["Times New Roman"]
    return FONT_FILES.get(font_family, FONT_FILES["Arial"])


def text_metrics(text: str, font_family: str, size_px: float, bold: bool, italic: bool) -> tuple[float, float, float]:
    path = font_file(font_family, bold, italic)
    try:
        font = ImageFont.truetype(str(path), max(1, int(round(size_px))))
        bbox = font.getbbox(text or " ")
        ascent, descent = font.getmetrics()
        return max(2.0, float(bbox[2] - bbox[0])), float(ascent), float(descent)
    except OSError:
        return max(2.0, len(text) * size_px * 0.56), size_px * 0.82, size_px * 0.22


def add_text(tree, element: ET.Element, name: str):
    text = "".join(element.itertext())
    x = float(element.attrib["x"])
    baseline_y = float(element.attrib["y"])
    size_px = float(element.attrib.get("font-size", "12"))
    font_family = element.attrib.get("font-family", "Arial")
    bold = element.attrib.get("font-weight", "normal") == "bold"
    italic = element.attrib.get("font-style", "normal") == "italic"
    anchor = element.attrib.get("text-anchor", "start")
    width_px, ascent_px, descent_px = text_metrics(text, font_family, size_px, bold, italic)
    pad_px = max(2.0, size_px * 0.12)
    box_w = width_px + 2 * pad_px
    box_h = max(size_px * 1.28, ascent_px + descent_px + 2.0)

    transform = element.attrib.get("transform", "")
    rotate_match = re.search(r"rotate\(([-0-9.]+)", transform)
    rotation = float(rotate_match.group(1)) % 360 if rotate_match else 0.0

    if rotation:
        # Rotated SVG text is centered on its anchor; use a centered box so the
        # PowerPoint rotation pivot matches the SVG rotation pivot closely.
        left = x - box_w / 2
        top = baseline_y - box_h / 2
    else:
        left = x - (box_w / 2 if anchor == "middle" else box_w if anchor == "end" else 0)
        top = baseline_y - ascent_px - 1.0

    shape = tree.add_textbox(
        Inches(left * PX_TO_IN),
        Inches(top * PX_TO_IN),
        Inches(box_w * PX_TO_IN),
        Inches(box_h * PX_TO_IN),
    )
    shape.rotation = rotation
    shape.text_frame.clear()
    shape.text_frame.margin_left = 0
    shape.text_frame.margin_right = 0
    shape.text_frame.margin_top = 0
    shape.text_frame.margin_bottom = 0
    shape.text_frame.word_wrap = False
    shape.text_frame.auto_size = MSO_AUTO_SIZE.NONE
    shape.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = shape.text_frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.CENTER
    run = paragraph.add_run()
    run.text = text
    run.font.name = font_family
    run.font.size = Pt(size_px * PX_TO_PT)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = rgb(element.attrib.get("fill", "#111111"))
    apply_name(shape, name)
    return shape


def add_element(tree, element: ET.Element, path: list[str], counters: Counter, group_ids: set[str]):
    tag = local_name(element.tag)
    if tag == "g":
        group_id = element.attrib.get("id", "group")
        child_path = path + [group_id]
        if group_id in group_ids:
            group = tree.add_group_shape()
            apply_name(group, "module__" + "__".join(child_path))
            for child in list(element):
                add_element(group.shapes, child, child_path, counters, group_ids)
            counters["group"] += 1
            return group
        for child in list(element):
            add_element(tree, child, child_path, counters, group_ids)
        return None

    counters[tag] += 1
    semantic = "__".join(path[-3:]) if path else "slide"
    name = f"{semantic}__{tag}_{counters[tag]:03d}"
    if tag == "rect":
        return add_rect(tree, element, name)
    if tag in {"circle", "ellipse"}:
        return add_ellipse(tree, element, name)
    if tag == "polygon":
        return add_freeform(tree, element, name, closed=True)
    if tag == "polyline":
        return add_freeform(tree, element, name, closed=False)
    if tag == "line":
        return add_line(tree, element, name)
    if tag == "text":
        return add_text(tree, element, name)
    raise ValueError(f"Unsupported SVG element: {tag}")


def pptx_xml_counts(pptx_path: Path) -> dict:
    with zipfile.ZipFile(pptx_path) as zf:
        slide_xml = zf.read("ppt/slides/slide1.xml").decode("utf-8")
        media = [name for name in zf.namelist() if name.startswith("ppt/media/")]
    return {
        "native_shapes": slide_xml.count("<p:sp>"),
        "connectors": slide_xml.count("<p:cxnSp>"),
        "groups": slide_xml.count("<p:grpSp>"),
        "pictures": slide_xml.count("<p:pic>"),
        "editable_text_runs": slide_xml.count("<a:t>"),
        "custom_geometries": slide_xml.count("<a:custGeom>"),
        "media_files": media,
    }


def build_presentation(output_path: Path, group_ids: set[str]):
    svg_root = ET.parse(SOURCE_SVG).getroot()
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W_IN)
    prs.slide_height = Inches(SLIDE_H_IN)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor(255, 255, 255)

    counters: Counter = Counter()
    for child in list(svg_root):
        add_element(slide.shapes, child, [], counters, group_ids)

    # Remove default empty slide that can appear in some templates.
    if len(prs.slides) > 1:
        first_slide_id = prs.slides._sldIdLst[0]
        prs.part.drop_rel(first_slide_id.rId)
        del prs.slides._sldIdLst[0]

    suffix = "grouped modules" if group_ids else "fully ungrouped"
    prs.core_properties.title = f"PI-MSCL architecture — native editable shapes ({suffix})"
    prs.core_properties.subject = "Native PowerPoint/DrawingML reconstruction from img2pptx SVG"
    prs.core_properties.author = "Codex / img2pptx native-shape workflow"
    prs.core_properties.keywords = "PI-MSCL, SOH, CNN-LSTM, editable PowerPoint, DrawingML"
    prs.save(output_path)
    return prs, counters


def main() -> None:
    QA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_SVG, COPIED_SVG)
    for filename in ("component_manifest.json", "original.png"):
        source = SOURCE_DIR / filename
        if source.exists():
            shutil.copy2(source, ROOT / filename)

    prs, counters = build_presentation(OUTPUT_PPTX, GROUP_IDS)
    ungrouped_prs, ungrouped_counters = build_presentation(OUTPUT_PPTX_UNGROUPED, set())

    package_counts = pptx_xml_counts(OUTPUT_PPTX)
    ungrouped_package_counts = pptx_xml_counts(OUTPUT_PPTX_UNGROUPED)
    audit = {
        "source_svg": str(SOURCE_SVG),
        "output_pptx": str(OUTPUT_PPTX),
        "output_pptx_fully_ungrouped": str(OUTPUT_PPTX_UNGROUPED),
        "canvas_px": [CANVAS_W, CANVAS_H],
        "slide_inches": [SLIDE_W_IN, SLIDE_H_IN],
        "svg_element_counts": dict(counters),
        "pptx_package_counts": package_counts,
        "fully_ungrouped_package_counts": ungrouped_package_counts,
        "hard_checks": {
            "single_slide": len(prs.slides) == 1,
            "no_picture_objects": package_counts["pictures"] == 0,
            "no_embedded_media": len(package_counts["media_files"]) == 0,
            "native_shape_count_at_least_svg_graphics": package_counts["native_shapes"] + package_counts["connectors"] >= 500,
            "editable_text_runs_match_svg": package_counts["editable_text_runs"] >= counters["text"],
            "custom_geometry_present": package_counts["custom_geometries"] >= counters["polygon"] + counters["polyline"],
            "semantic_groups_present": package_counts["groups"] >= 20,
            "fully_ungrouped_has_no_groups": ungrouped_package_counts["groups"] == 0,
            "fully_ungrouped_has_no_pictures": ungrouped_package_counts["pictures"] == 0,
            "fully_ungrouped_preserves_all_text": ungrouped_package_counts["editable_text_runs"] >= ungrouped_counters["text"],
        },
    }
    audit["all_hard_checks_pass"] = all(audit["hard_checks"].values())
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if not audit["all_hard_checks_pass"]:
        raise SystemExit(json.dumps(audit["hard_checks"], indent=2))
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
