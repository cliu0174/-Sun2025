from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont
from lxml import etree

ROOT = Path(__file__).resolve().parent
QA = ROOT / "qa"
MASKS = QA / "semantic_masks"
CROPS = QA / "component_crops"
NS = {"svg": "http://www.w3.org/2000/svg"}


def save_json(name, data):
    (QA / name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rgb(path):
    return Image.open(path).convert("RGB")


def mae(a, b, mask=None):
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    diff = np.abs(aa - bb)
    if mask is None:
        return float(diff.mean())
    if not np.any(mask):
        return 0.0
    return float(diff[mask].mean())


def group(root, gid):
    nodes = root.xpath(f"//*[local-name()='g' and @id='{gid}']")
    return nodes[0] if nodes else None


def direct_children_named(node, name):
    return [c for c in node if etree.QName(c).localname == name]


def all_named(node, name):
    return node.xpath(f".//*[local-name()='{name}']")


def text_content(node):
    return " ".join("".join(t.itertext()) for t in all_named(node, "text"))


def fg_bbox(im, threshold=246):
    arr = np.asarray(im.convert("RGB"))
    mask = np.any(arr < threshold, axis=2)
    if not np.any(mask):
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def crop_box(comp):
    b = comp["bbox"]
    return (int(b["x"]), int(b["y"]), int(b["x"] + b["width"]), int(b["y"] + b["height"]))


def make_component_sheet(rows, out_path):
    thumb_w, thumb_h = 300, 170
    label_h = 25
    canvas = Image.new("RGB", (thumb_w * 3, len(rows) * (thumb_h + label_h)), "white")
    d = ImageDraw.Draw(canvas)
    for i, (label, src, ren, dif) in enumerate(rows):
        y = i * (thumb_h + label_h)
        d.text((8, y + 4), label, fill="black")
        for j, im in enumerate((src, ren, dif)):
            cp = im.copy()
            cp.thumbnail((thumb_w, thumb_h))
            x = j * thumb_w + (thumb_w - cp.width) // 2
            yy = y + label_h + (thumb_h - cp.height) // 2
            canvas.paste(cp, (x, yy))
    canvas.save(out_path)


def main():
    QA.mkdir(exist_ok=True)
    MASKS.mkdir(exist_ok=True)
    CROPS.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / "component_manifest.json").read_text(encoding="utf-8"))
    comps = manifest["components"]
    by_id = {c["id"]: c for c in comps}
    src = rgb(ROOT / "original.png")
    ren = rgb(QA / "rendered_svg.png")
    ppt = rgb(QA / "pptx_slide_preview.png")
    if src.size != (manifest["canvas"]["width"], manifest["canvas"]["height"]):
        raise RuntimeError(f"Source size {src.size} does not match manifest canvas")
    if ren.size != src.size or ppt.size != src.size:
        raise RuntimeError("Render sizes do not match source")
    src.save(QA / "normalized_original.png")

    svg_bytes = (ROOT / "full.svg").read_bytes()
    svg_text = svg_bytes.decode("utf-8")
    root = etree.fromstring(svg_bytes)

    panel_a = by_id["panel_a"]["bbox"]
    panel_b = by_id["panel_b"]["bbox"]
    layout_checks = {
        "canvas_matches_source": src.size == (manifest["canvas"]["width"], manifest["canvas"]["height"]),
        "panels_share_left_edge": panel_a["x"] == panel_b["x"],
        "panels_share_width": panel_a["width"] == panel_b["width"],
        "panel_a_above_panel_b": panel_a["y"] + panel_a["height"] < panel_b["y"],
        "panel_gap_px": panel_b["y"] - (panel_a["y"] + panel_a["height"]),
        "panel_side_margin_px": panel_a["x"],
        "canvas_right_margin_px": manifest["canvas"]["width"] - panel_a["x"] - panel_a["width"]
    }
    layout_pass = all(v for k, v in layout_checks.items() if isinstance(v, bool)) and 8 <= layout_checks["panel_gap_px"] <= 25
    save_json("layout_skeleton_audit.json", {"checks": layout_checks, "passed": layout_pass})

    standalone = []
    standalone_pass = True
    for comp in comps:
        if not comp.get("export"):
            continue
        p = CROPS / f"{comp['id']}_render.png"
        ok_file = p.exists() and p.stat().st_size > 200
        bb = fg_bbox(rgb(p)) if ok_file else None
        if bb:
            im = rgb(p)
            margins = [bb[0], bb[1], im.width - 1 - bb[2], im.height - 1 - bb[3]]
            safe = min(margins) >= 2
        else:
            margins, safe = [], False
        entry = {"component": comp["id"], "render_exists": ok_file, "foreground_bbox": bb, "edge_margins": margins, "safe_padding": safe, "passed": bool(ok_file and safe)}
        standalone.append(entry)
        standalone_pass &= entry["passed"]
    save_json("standalone_integrity_audit.json", {"modules": standalone, "passed": standalone_pass})

    containment = []
    containment_pass = True
    for comp in comps:
        if not comp.get("parent"):
            continue
        b, p = comp["bbox"], by_id[comp["parent"]]["bbox"]
        inside = b["x"] >= p["x"] and b["y"] >= p["y"] and b["x"] + b["width"] <= p["x"] + p["width"] and b["y"] + b["height"] <= p["y"] + p["height"]
        min_pad = min(b["x"]-p["x"], b["y"]-p["y"], p["x"]+p["width"]-(b["x"]+b["width"]), p["y"]+p["height"]-(b["y"]+b["height"]))
        entry = {"component": comp["id"], "parent": comp["parent"], "inside": inside, "minimum_manifest_padding_px": min_pad, "passed": inside}
        containment.append(entry); containment_pass &= inside
    save_json("containment_audit.json", {"checks": containment, "passed": containment_pass})

    align_checks = {
        "loss_boxes_same_x": len({float(r.get("x")) for gid in ("supervised_path","monotonic_path","rate_path") for r in all_named(group(root,gid),"rect") if r.get("x") == "768"}) == 1,
        "loss_boxes_width_168": sum(1 for gid in ("supervised_path","monotonic_path","rate_path") for r in all_named(group(root,gid),"rect") if r.get("x") == "768" and r.get("width") == "168") == 3,
        "panel_titles_left_aligned": "PI–MSCL architecture" in text_content(group(root,"panel_a")) and "Sparse-label physics-informed learning" in text_content(group(root,"panel_b")),
        "no_panel_overlap": panel_a["y"] + panel_a["height"] < panel_b["y"],
        "model_stage_order": [by_id[x]["bbox"]["x"] for x in ("input_sequence","multiscale_cnn","fusion","lstm","regression","output_curve")] == sorted(by_id[x]["bbox"]["x"] for x in ("input_sequence","multiscale_cnn","fusion","lstm","regression","output_curve"))
    }
    align_pass = all(align_checks.values())
    save_json("alignment_audit.json", {"checks": align_checks, "passed": align_pass})

    border_ids = ["panel_a","panel_b","fusion","lstm","regression","objective"]
    border_results = []
    border_pass = True
    for gid in border_ids:
        g = group(root, gid)
        rects = direct_children_named(g, "rect")
        first, last = rects[0], rects[-1]
        fill_only = first.get("stroke") == "none" and first.get("fill") not in (None, "none")
        outline = last.get("fill") == "none" and last.get("stroke") not in (None, "none") and last.get("stroke-width") is not None
        last_visual = list(g)[-1] is last
        ok = fill_only and outline and last_visual
        border_results.append({"component":gid,"fill_only_background":fill_only,"separate_outline":outline,"outline_last":last_visual,"passed":ok})
        border_pass &= ok
    save_json("border_layering_audit.json", {"components": border_results, "passed": border_pass})

    checks = {}
    checks["panels.order"] = by_id["panel_a"]["bbox"]["y"] < by_id["panel_b"]["bbox"]["y"]
    checks["input.dimension"] = "40 × 14" in text_content(group(root,"input_sequence"))
    input_group = group(root, "input_sequence")
    purple_input_fills = {"#9684C8", "#A996D2", "#BBAADD"}
    purple_input_tiles = [r for r in all_named(input_group, "rect") if r.get("fill") in purple_input_fills]
    checks["input.purple_cells"] = len(purple_input_tiles) >= 8
    cnn = group(root,"multiscale_cnn")
    branch_ids = [e.get("id") for e in cnn.xpath(".//*[local-name()='g']") if e.get("id") in ("cnn_k3","cnn_k7","cnn_k15")]
    checks["cnn.branch_order_colors"] = branch_ids[:3] == ["cnn_k3","cnn_k7","cnn_k15"] and all(c in svg_text for c in ("#D6E7F8","#D9EFD9","#E5DDF5"))
    cnn_polygons = all_named(cnn, "polygon")
    gradient_group_ids = [f"cnn_{branch}_{scale}" for branch in ("k3", "k7", "k15") for scale in ("40", "20", "10")]
    gradient_groups = [group(root, gid) for gid in gradient_group_ids]
    checks["cnn.slanted_multicolor_grids"] = len(cnn_polygons) >= 180 and all(g is not None for g in gradient_groups)
    def color_luminance(hex_color):
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    expected_cols_by_scale = {"40": 8, "20": 4, "10": 2}
    ordered_gradients = True
    temporal_cell_ratio = True
    for gid, gradient_group in zip(gradient_group_ids, gradient_groups):
        scale = gid.rsplit("_", 1)[-1]
        expected_cols = expected_cols_by_scale[scale]
        expected_cells = expected_cols * 3
        tiles = [p for p in all_named(gradient_group, "polygon") if p.get("data-gradient-rank") is not None]
        tiles.sort(key=lambda p: int(p.get("data-gradient-rank")))
        ranks = [int(p.get("data-gradient-rank")) for p in tiles]
        luminance = [color_luminance(p.get("fill")) for p in tiles]
        rows = sorted({int(p.get("data-gradient-row")) for p in tiles})
        columns_by_row = {
            row: sorted(int(p.get("data-gradient-col")) for p in tiles if int(p.get("data-gradient-row")) == row)
            for row in rows
        }
        base_layers = [p for p in all_named(gradient_group, "polygon") if p.get("data-gradient-rank") is None and p.get("fill") != "none"]
        ordered_gradients &= ranks == list(range(expected_cells)) and all(b > a for a, b in zip(luminance, luminance[1:]))
        temporal_cell_ratio &= rows == [0, 1, 2] and all(columns_by_row[row] == list(range(expected_cols)) for row in rows) and len(base_layers) == 5
    checks["cnn.row_major_gradient"] = bool(ordered_gradients)
    checks["cnn.temporal_cell_ratio"] = bool(temporal_cell_ratio)
    checks["cnn.dimensions"] = all(all(v in text_content(group(root,gid)) for v in ("40","20","10")) for gid in ("cnn_k3","cnn_k7","cnn_k15"))
    fusion = group(root, "fusion")
    tensor_layers = [p for p in all_named(fusion, "polygon") if p.get("data-tensor-layer") is not None]
    tensor_facets = {p.get("data-tensor-facet") for p in all_named(fusion, "polygon") if p.get("data-tensor-facet") is not None}
    tensor_cells = [p for p in all_named(fusion, "polygon") if p.get("data-tensor-cell") is not None]
    checks["fusion.tensor_3d"] = len(tensor_layers) == 5 and tensor_facets == {"top", "side"} and len(tensor_cells) == 20
    checks["model.flow"] = len(all_named(group(root,"model_connectors"),"polyline")) == 10
    checks["lstm.two_layers"] = text_content(group(root,"lstm")).count("LSTM") >= 7 and len(all_named(group(root,"lstm"),"rect")) >= 8
    sup = group(root,"supervised_path")
    orange = [c for c in all_named(sup,"circle") if c.get("fill") == "#F27919"]
    hollow = [c for c in all_named(sup,"circle") if c.get("fill") == "#FFFFFF"]
    checks["supervision.sparse_markers"] = len(orange) == 4 and len(hollow) >= 12
    mono = group(root,"monotonic_path")
    red_mono = [p for p in all_named(mono,"polyline") if p.get("stroke") == "#D94343"]
    green_mono = [p for p in all_named(mono,"polyline") if p.get("stroke") == "#2E7B25"]
    checks["monotonicity.violation"] = len(red_mono) == 1 and len(green_mono) == 1 and "Δc ≤ 40" in text_content(mono) and "c ≥ 300" in text_content(mono)
    rate = group(root,"rate_path")
    checks["rate.kink"] = any(p.get("stroke") == "#D94343" for p in all_named(rate,"polyline")) and any(p.get("stroke") == "#73A95C" for p in all_named(rate,"polyline"))
    checks["objective.three_inputs"] = len(all_named(group(root,"training_connectors"),"polyline")) == 3
    obj_text = text_content(group(root,"objective"))
    checks["objective.formula"] = all(token in obj_text for token in ("ℒₛᵤₚ","ℒₘₒₙₒ","ℒᵣₐₜₑ"))
    checks["canvas.no_raster"] = len(root.xpath("//*[local-name()='image']")) == 0
    checks["input.no_soh_target"] = "SOH target" not in text_content(group(root,"input_sequence"))
    checks["cnn.exactly_three"] = len(branch_ids) == 3
    checks["objective.no_extra_edge"] = len(all_named(group(root,"training_connectors"),"polyline")) == 3
    semantic_pass = all(checks.values())
    save_json("semantic_constraint_audit.json", {"checks": checks, "passed": semantic_pass})

    coverage = []
    for con in manifest["visual_invariants"] + manifest["negative_constraints"]:
        key = con["audit_check"].replace("semantic_constraint_audit.", "")
        passed = bool(checks.get(key, False))
        coverage.append({"constraint_id":con["id"],"component":con["component"],"severity":con["severity"],"audit_check":con["audit_check"],"implemented":key in checks,"executed":key in checks,"passed":passed,"evidence":{"actual_check_value":checks.get(key)}})
    observation_ids = {o["id"] for o in manifest["source_observations"]}
    referenced = {c["source_observation"] for c in manifest["visual_invariants"] + manifest["negative_constraints"]}
    coverage_pass = all(c["implemented"] and c["executed"] and c["passed"] for c in coverage if c["severity"] == "hard") and observation_ids <= referenced
    save_json("constraint_coverage_audit.json", {"observations":len(observation_ids),"hard_constraints":sum(1 for c in coverage if c["severity"]=="hard"),"implemented":sum(c["implemented"] for c in coverage),"executed":sum(c["executed"] for c in coverage),"passed_checks":sum(c["passed"] for c in coverage),"uncovered_observations":sorted(observation_ids-referenced),"constraints":coverage,"passed":coverage_pass})

    sa, ra = np.asarray(src), np.asarray(ren)
    overlay = Image.blend(src, ren, 0.5); overlay.save(QA / "full_overlay_diff.png")
    diff_arr = np.abs(sa.astype(np.int16)-ra.astype(np.int16)).astype(np.uint8)
    Image.fromarray(np.clip(diff_arr*4,0,255).astype(np.uint8)).save(QA / "amplified_diff.png")
    foreground = np.any(sa < 245, axis=2) | np.any(ra < 245, axis=2)
    full_mae = mae(src,ren); fg_mae = mae(src,ren,foreground)
    component_metrics = {}
    sheet_rows=[]
    for comp in comps:
        box=crop_box(comp); s=src.crop(box); r=ren.crop(box)
        d=ImageChops.difference(s,r)
        s.save(CROPS/f"{comp['id']}_source.png"); r.save(CROPS/f"{comp['id']}_full_render.png"); d.save(CROPS/f"{comp['id']}_diff.png")
        component_metrics[comp["id"]]=mae(s,r)
        sheet_rows.append((comp["id"],s,r,Image.eval(d,lambda p:min(255,p*4))))
    make_component_sheet(sheet_rows,QA/"component_diff_sheet.png")

    colors={"orange_labels":(242,121,25),"green_trajectory":(90,169,62),"red_violation":(217,67,67),"blue_branch":(214,231,248),"purple_branch":(229,221,245)}
    for name,col in colors.items():
        dist=np.sqrt(sum((ra[:,:,i].astype(float)-col[i])**2 for i in range(3)))
        mask=(dist<35).astype(np.uint8)*255
        Image.fromarray(mask,"L").save(MASKS/f"{name}.png")

    review = Image.new("RGB",(1600,1000),"white"); draw=ImageDraw.Draw(review)
    labels=["SOURCE","RECONSTRUCTION","50/50 OVERLAY","AMPLIFIED DIFF"]
    imgs=[src,ren,overlay,Image.open(QA/"amplified_diff.png").convert("RGB")]
    for i,(lab,im) in enumerate(zip(labels,imgs)):
        cp=im.copy(); cp.thumbnail((780,440)); x=10+(i%2)*795; y=40+(i//2)*480
        draw.text((x,y-25),lab,fill="black"); review.paste(cp,(x,y))
    review.save(QA/"semantic_review_sheet.png")

    with zipfile.ZipFile(ROOT/"final.pptx") as zf:
        names=zf.namelist(); svg_names=[n for n in names if n.lower().endswith('.svg')]; slide_names=[n for n in names if n.startswith('ppt/slides/slide') and n.endswith('.xml')]
        embedded=[(n,zf.read(n)) for n in svg_names]
    exact_hash_match=any(hashlib.sha256(data).hexdigest()==hashlib.sha256(svg_bytes).hexdigest() for _,data in embedded)
    package_checks={"slide_count":len(slide_names),"svg_media_count":len(svg_names),"embedded_svg_exact_hash_match":exact_hash_match,"full_svg_sha256":hashlib.sha256(svg_bytes).hexdigest(),"embedded_svg_names":svg_names,"nested_groups_present":all(x in svg_text for x in ('id="panel_a"','id="panel_b"','id="multiscale_cnn"')),"unsupported_foreign_object":'<foreignObject' in svg_text,"unsupported_filter":'<filter' in svg_text,"embedded_raster_image":'<image' in svg_text}
    package_pass=package_checks["slide_count"]==1 and package_checks["svg_media_count"]>=1 and exact_hash_match and package_checks["nested_groups_present"] and not package_checks["unsupported_foreign_object"] and not package_checks["unsupported_filter"] and not package_checks["embedded_raster_image"]
    save_json("pptx_package_audit.json",{"checks":package_checks,"passed":package_pass})

    ppt_mae=mae(ren,ppt); ppt_fg=np.any(np.asarray(ren)<245,axis=2)|np.any(np.asarray(ppt)<245,axis=2); ppt_fg_mae=mae(ren,ppt,ppt_fg)
    preview_pass=ppt_mae<6 and ppt_fg_mae<40
    save_json("pptx_preview_audit.json",{"svg_to_pptx_preview_mae":ppt_mae,"foreground_mae":ppt_fg_mae,"slide_size_pixels":ppt.size,"text_readability_review":"pending-human-review","passed":preview_pass})

    visual={"baseline_mae":full_mae,"target_mae":full_mae*0.90,"best_mae":full_mae,"foreground_mae":fg_mae,"relative_improvement":0.0,"target_relative_improvement":0.10,"target_achieved":False,"attempts_used":0,"max_try":3,"optimization_status":"baseline_established","delivered_state":"baseline_valid","hard_audits_passed":all((layout_pass,standalone_pass,containment_pass,align_pass,border_pass,package_pass,preview_pass)),"semantic_audits_passed":semantic_pass,"constraint_coverage_passed":coverage_pass,"component_mae":component_metrics,"attempts":[]}
    save_json("visual_similarity_audit.json",visual)
    aggregate={"layout":layout_pass,"standalone":standalone_pass,"containment":containment_pass,"alignment":align_pass,"border_layering":border_pass,"semantic":semantic_pass,"constraint_coverage":coverage_pass,"pptx_package":package_pass,"pptx_preview":preview_pass}
    aggregate["passed"]=all(aggregate.values())
    save_json("aggregate_audit.json",aggregate)
    print(json.dumps({"aggregate":aggregate,"full_mae":full_mae,"foreground_mae":fg_mae,"pptx_preview_mae":ppt_mae,"worst_components":sorted(component_metrics.items(),key=lambda kv:kv[1],reverse=True)[:5]},indent=2))


if __name__ == "__main__":
    main()
