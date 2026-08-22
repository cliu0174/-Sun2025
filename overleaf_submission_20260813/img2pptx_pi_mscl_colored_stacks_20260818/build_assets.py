from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent
W, H = 1672, 941


def attrs(**kwargs):
    out = []
    for key, value in kwargs.items():
        if value is None:
            continue
        key = key.replace("_", "-")
        out.append(f'{key}="{escape(str(value), {chr(34): "&quot;"})}"')
    return " ".join(out)


def rect(x, y, w, h, fill="none", stroke="none", sw=0, rx=0, extra=""):
    return f'<rect {attrs(x=x,y=y,width=w,height=h,fill=fill,stroke=stroke,stroke_width=sw,rx=rx)} {extra}/>'


def line(x1, y1, x2, y2, stroke="#333333", sw=1.4, dash=None):
    return f'<line {attrs(x1=x1,y1=y1,x2=x2,y2=y2,stroke=stroke,stroke_width=sw,stroke_dasharray=dash,stroke_linecap="round")}/>'


def text(x, y, s, size=14, anchor="start", weight="normal", family="Arial", fill="#111111", italic=False, rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    style = "italic" if italic else "normal"
    return f'<text {attrs(x=x,y=y,font_family=family,font_size=size,font_weight=weight,font_style=style,text_anchor=anchor,fill=fill)}{transform}>{escape(s)}</text>'


def circle(cx, cy, r, fill="#FFFFFF", stroke="#333333", sw=1):
    return f'<circle {attrs(cx=cx,cy=cy,r=r,fill=fill,stroke=stroke,stroke_width=sw)}/>'


def polygon(points, fill="none", stroke="none", sw=0, extra=""):
    pts = " ".join(f"{x},{y}" for x, y in points)
    return f'<polygon {attrs(points=pts,fill=fill,stroke=stroke,stroke_width=sw,stroke_linejoin="round")} {extra}/>'


def polyline(points, stroke="#333333", sw=1.5, fill="none", dash=None):
    pts = " ".join(f"{x},{y}" for x, y in points)
    return f'<polyline {attrs(points=pts,fill=fill,stroke=stroke,stroke_width=sw,stroke_dasharray=dash,stroke_linecap="round",stroke_linejoin="round")}/>'


def arrow(points, color="#333333", sw=1.6, dash=None, head=8):
    body = polyline(points, stroke=color, sw=sw, dash=dash)
    (x1, y1), (x2, y2) = points[-2], points[-1]
    if abs(x2 - x1) >= abs(y2 - y1):
        if x2 >= x1:
            tri = [(x2, y2), (x2-head, y2-head/2), (x2-head, y2+head/2)]
        else:
            tri = [(x2, y2), (x2+head, y2-head/2), (x2+head, y2+head/2)]
    else:
        if y2 >= y1:
            tri = [(x2, y2), (x2-head/2, y2-head), (x2+head/2, y2-head)]
        else:
            tri = [(x2, y2), (x2-head/2, y2+head), (x2+head/2, y2+head)]
    pts = " ".join(f"{x},{y}" for x, y in tri)
    return body + f'<polygon points="{pts}" fill="{color}" stroke="none"/>'


def group(gid, children):
    return f'<g id="{gid}">' + "".join(children) + "</g>"


def panel_header(letter, title_s, y):
    return group(f"panel_{letter}_header", [
        rect(23, y+14, 30, 30, fill="#111111", rx=1),
        text(38, y+37, letter, 18, "middle", "bold", fill="#FFFFFF"),
        text(70, y+39, title_s, 22, weight="bold")
    ])


def matrix_module():
    c = [text(133, 102, "Input", 15, "middle", "bold"), text(133, 130, "40 × 14", 15, "middle", family="Times New Roman")]
    c.append(rect(70, 150, 127, 222, fill="#FFFFFF", stroke="#20262A", sw=1.6))
    blue = ["#82ACD8", "#91B8DD", "#A5C7E4"]
    green = ["#8FC3AF", "#A4CEBC", "#B8D9C8"]
    purple = ["#9684C8", "#A996D2", "#BBAADD"]
    input_colors = [
        [blue[0], blue[1], blue[2], green[0], green[1], green[2]],
        [blue[1], blue[0], blue[1], green[1], green[0], green[2]],
        [blue[0], blue[1], blue[2], green[0], green[1], green[2]],
        [blue[1], blue[0], blue[1], green[1], green[2], green[0]],
        [purple[0], blue[1], blue[2], green[0], green[1], green[2]],
        [purple[1], purple[0], blue[1], blue[2], green[0], green[1]],
        [purple[2], purple[1], purple[0], blue[1], green[1], green[2]],
        [purple[0], purple[2], purple[1], blue[2], green[0], green[1]],
    ]
    for r in range(8):
        for col in range(6):
            color = input_colors[r][col]
            c.append(rect(78 + 19*col, 155 + 19*r, 17, 17, fill=color, stroke="#FFFFFF", sw=0.8))
    c.extend([text(88, 324, "⋮", 24, "middle"), text(126, 324, "⋮", 24, "middle"), text(164, 324, "⋮", 24, "middle")])
    tail_colors = ["#D8DC9C", "#DDE1A3", "#E4E8B0", "#C4DCCB", "#CEE4D4", "#D8EADD"]
    for r in range(2):
        for col, color in enumerate(tail_colors):
            c.append(rect(78 + 19*col, 334 + 19*r, 17, 17, fill=color, stroke="#FFFFFF", sw=0.8))
    c.append(line(53, 150, 53, 368, sw=1.2)); c.append(arrow([(53,150),(53,368)], sw=1.2, head=7))
    c.append(text(39, 261, "features (14)", 11, "middle", rotate=-90))
    c.append(arrow([(70,387),(198,387)], sw=1.2, head=7)); c.append(text(134, 410, "time (40 cycles)", 11, "middle", "bold"))
    return group("input_sequence", c)


def feature_stack(prefix, x, y, w, h, fill, stroke):
    c = []
    shade_map = {
        "#D6E7F8": ["#89B4DE", "#A5C8E8", "#C1DBF1", "#DCECF9"],
        "#D9EFD9": ["#86C18B", "#A4D1A7", "#BFE0C1", "#DCEEDC"],
        "#E5DDF5": ["#9A82C8", "#B19BD6", "#C9B9E4", "#E2D9F2"],
    }
    shades = shade_map[fill]
    slant = min(7, max(4, h * 0.10))
    # Five editable feature maps, diagonally offset to reproduce the layered perspective.
    for layer in range(5):
        depth = 4 - layer
        lx, ly = x - depth * 5, y - depth * 3
        pts = [(lx, ly + slant), (lx + w, ly), (lx + w, ly + h - slant), (lx, ly + h)]
        c.append(polygon(pts, fill=fill, stroke=stroke, sw=0.8, extra=f'opacity="{0.30 + 0.13*layer:.2f}"'))

    # The front map is a slanted 6 × 3 grid with varied shades inside the branch color family.
    cols, rows = 6, 3
    for row in range(rows):
        for col in range(cols):
            u0, u1 = col / cols, (col + 1) / cols
            x0, x1 = x + w * u0, x + w * u1
            top0, top1 = y + slant * (1 - u0), y + slant * (1 - u1)
            y00, y01 = top0 + h * row / rows, top1 + h * row / rows
            y10, y11 = top0 + h * (row + 1) / rows, top1 + h * (row + 1) / rows
            tile_fill = shades[(row * 2 + col) % len(shades)]
            c.append(polygon([(x0, y00), (x1, y01), (x1, y11), (x0, y10)], fill=tile_fill, stroke="#FFFFFF", sw=0.65, extra='opacity="0.90"'))
    c.append(polygon([(x, y + slant), (x + w, y), (x + w, y + h - slant), (x, y + h)], fill="none", stroke=stroke, sw=1.0))
    return group(prefix, c)


def cnn_module():
    c = [text(490, 70, "Conv1D blocks", 16, "middle", "bold", fill="#0756A0"), text(490, 94, "(64 ch.)", 14, "middle", "bold", fill="#0756A0")]
    branch_specs = [
        ("cnn_k3", 142, "k = 3", "#D6E7F8", "#2E6DA4", "#0A5AA6"),
        ("cnn_k7", 257, "k = 7", "#D9EFD9", "#398C43", "#148129"),
        ("cnn_k15", 372, "k = 15", "#E5DDF5", "#7251A7", "#6842A2")
    ]
    for gid, cy, label, fill, stroke, color in branch_specs:
        parts = [text(304, cy-12, label, 15, "end", family="Times New Roman", fill=color, italic=True)]
        parts.append(feature_stack(gid+"_40", 345, cy-46, 100, 58, fill, stroke))
        parts.append(arrow([(445,cy-17),(476,cy-17)], color="#111111", sw=1.3, dash="4 4", head=7))
        parts.append(feature_stack(gid+"_20", 495, cy-44, 92, 56, fill, stroke))
        parts.append(arrow([(587,cy-16),(616,cy-16)], color="#111111", sw=1.3, dash="4 4", head=7))
        parts.append(feature_stack(gid+"_10", 635, cy-40, 63, 53, fill, stroke))
        parts.extend([text(390, cy+38, "40", 11, "middle"), text(540, cy+38, "20", 11, "middle"), text(665, cy+38, "10", 11, "middle")])
        c.append(group(gid, parts))
    c.extend([
        rect(104, 449, 17, 17, fill="#D6E7F8", stroke="#2E6DA4", sw=0.8), text(129, 463, "k = 3 (local)", 12, family="Times New Roman"),
        rect(250, 449, 17, 17, fill="#D9EFD9", stroke="#398C43", sw=0.8), text(275, 463, "k = 7 (intermediate)", 12, family="Times New Roman"),
        rect(464, 449, 17, 17, fill="#E5DDF5", stroke="#7251A7", sw=0.8), text(489, 463, "k = 15 (longer-range)", 12, family="Times New Roman")
    ])
    return group("multiscale_cnn", c)


def fusion_module():
    c = [
        rect(782, 197, 100, 137, fill="#EDF4FA", stroke="none", rx=10),
        text(832, 245, "Fuse", 15, "middle", "bold"), text(832, 272, "1 × 1 Conv", 14, "middle"), text(832, 299, "(128 ch.)", 14, "middle", "bold", fill="#0756A0"),
    ]
    for i in reversed(range(4)):
        c.append(rect(936-5*i, 186+5*i, 39, 151, fill="#C9DDF0", stroke="#5D7D98", sw=0.8, extra=f'opacity="{0.42+0.13*i:.2f}"'))
    for yy in [201,216,231,246,261,276,291,306,321]: c.append(line(936, yy, 975, yy, stroke="#5D7D98", sw=0.45))
    c.extend([text(955, 310, "⋮", 18, "middle"), text(955, 358, "10", 11, "middle"), rect(782.75,197.75,98.5,135.5,fill="none",stroke="#2B69A1",sw=1.5,rx=9)])
    return group("fusion", c)


def lstm_module():
    c = [rect(1028, 165, 290, 219, fill="#FFF9E9", stroke="none", rx=12), text(1173, 148, "LSTM (2 layers)", 15, "middle", "bold")]
    xs = [1048, 1131, 1254]
    for row_y in [202, 309]:
        for x in xs:
            c.append(rect(x, row_y, 51, 44, fill="#FFF1BE", stroke="#5C532A", sw=1.2, rx=8))
            c.append(text(x+25.5, row_y+27, "LSTM", 11, "middle", "bold"))
    c.extend([text(1210, 232, "•••", 16, "middle"), text(1210, 339, "•••", 16, "middle")])
    c.extend([arrow([(1099,224),(1129,224)], sw=1.2, head=7), arrow([(1182,224),(1249,224)], sw=1.2, head=7), arrow([(1099,331),(1129,331)], sw=1.2, head=7), arrow([(1182,331),(1249,331)], sw=1.2, head=7)])
    for x in [1074,1157,1280]: c.append(arrow([(x,309),(x,250)], sw=1.1, head=7))
    c.extend([text(1045, 412, "t = 1", 13, family="Times New Roman", italic=True), text(1252, 412, "t = 10", 13, family="Times New Roman", italic=True)])
    c.append(rect(1028.75,165.75,288.5,217.5,fill="none",stroke="#EEA51C",sw=1.5,rx=11))
    return group("lstm", c)


def regression_module():
    c = [rect(1340, 136, 127, 248, fill="#FFF2E9", stroke="none", rx=10), text(1403.5, 165, "Regression head", 14, "middle", "bold"), text(1403.5, 186, "(FC + Dropout)", 13, "middle", "bold"), rect(1381, 200, 47, 169, fill="#FFFFFF", stroke="#222222", sw=1.1, rx=8)]
    for cy in [222, 254, 309, 346]: c.append(circle(1405, cy, 12, fill="#FFD8BF", stroke="#8A4B20", sw=1.0))
    c.append(text(1405, 286, "⋮", 18, "middle")); c.append(rect(1340.75,136.75,125.5,246.5,fill="none",stroke="#E86C1C",sw=1.5,rx=9))
    return group("regression", c)


def output_module():
    c = [text(1571, 142, "Predicted SOH", 15, "middle", "bold"), text(1571, 174, "ŷᵢ,ₜ", 20, "middle", family="Times New Roman", italic=True)]
    c.extend([line(1517,350,1640,350,stroke="#333333",sw=1.1), line(1517,199,1517,350,stroke="#333333",sw=1.1)])
    for yy, val in [(204,"1.0"),(233,"0.8"),(263,"0.6"),(292,"0.4"),(321,"0.2"),(350,"0.0")]:
        c.append(line(1517,yy,1640,yy,stroke="#C8C8C8",sw=0.8,dash="4 4")); c.append(text(1509,yy+4,val,10,"end"))
    pts=[(1528,209),(1542,219),(1558,232),(1570,250),(1580,274),(1593,298),(1607,313),(1620,323),(1634,340)]
    c.append(polyline(pts,stroke="#2F7D22",sw=2.0))
    for x,y in pts: c.append(circle(x,y,4,fill="#5CAB3E",stroke="#1E5F1A",sw=0.9))
    c.extend([text(1528,368,"1",10,"middle"),text(1634,368,"10",10,"middle"),text(1582,392,"cycle index",11,"middle")])
    return group("output_curve", c)


def model_connectors():
    return group("model_connectors", [
        arrow([(246,258),(246,142),(322,142)], color="#0A5AA6", sw=1.8, head=8),
        arrow([(207,258),(322,258)], color="#148129", sw=1.8, head=8),
        arrow([(246,258),(246,373),(322,373)], color="#6842A2", sw=1.8, head=8),
        arrow([(698,142),(740,142),(740,246),(760,258),(780,258)], color="#0A5AA6", sw=1.8, head=8),
        arrow([(698,258),(780,258)], color="#148129", sw=1.8, head=8),
        arrow([(698,373),(740,373),(740,270),(760,258),(780,258)], color="#6842A2", sw=1.8, head=8),
        arrow([(882,263),(922,263)], sw=1.6, head=8), arrow([(976,263),(1016,263)], sw=1.6, head=8), arrow([(1318,260),(1337,260)], sw=1.6, head=8), arrow([(1467,260),(1488,260)], sw=1.6, head=8)
    ])


def panel_a_group():
    children = [rect(10,10,1652,475,fill="#FFFFFF",stroke="none",rx=13), model_connectors(), panel_header("a","PI–MSCL architecture",10), matrix_module(), cnn_module(), fusion_module(), lstm_module(), regression_module(), output_module(), rect(10.75,10.75,1650.5,473.5,fill="none",stroke="#AEB4B8",sw=1.5,rx=12)]
    return group("panel_a", children)


def sparse_path():
    c=[text(112,580,"Sparse labels",14,"middle","bold"),text(112,603,"(one cell)",12,"middle")]
    orange={1,9,17,22}
    xs=[202+18*i for i in range(27)]
    for i,x in enumerate(xs): c.append(circle(x,579,7,fill="#F27919" if i in orange else "#FFFFFF",stroke="#A95A14" if i in orange else "#B5B5B5",sw=1))
    c.extend([line(195,606,690,606,sw=1.0),line(219,602,219,611,sw=1.0),line(340,602,340,611,sw=1.0),line(460,602,460,611,sw=1.0),line(580,602,580,611,sw=1.0),line(682,602,682,611,sw=1.0)])
    c.extend([text(219,627,"1",10,"middle"),text(340,627,"10",10,"middle"),text(460,627,"20",10,"middle"),text(580,627,"30",10,"middle"),text(682,627,"40",10,"middle"),text(445,649,"cycle index",11,"middle")])
    c.extend([rect(768,541,168,90,fill="#FFF3EC",stroke="none",rx=10),text(852,568,"Masked",14,"middle","bold"),text(852,588,"supervision",14,"middle","bold"),text(852,616,"ℒₛᵤₚ",20,"middle",family="Times New Roman",italic=True),rect(768.75,541.75,166.5,88.5,fill="none",stroke="#E45F13",sw=1.5,rx=9)])
    c.append(arrow([(706,579),(756,579)],color="#E45F13",sw=1.6,head=8))
    return group("supervised_path",c)


def monotonic_path():
    c=[text(104,690,"Predicted SOH",14,"middle","bold"),text(104,713,"(ordered cell)",12,"middle")]
    pts=[(199,666),(227,672),(256,682),(284,691),(312,700),(340,706),(368,712),(396,718),(424,699),(452,691),(480,699),(510,706),(539,713),(568,720),(597,726),(626,732),(654,737),(683,742)]
    c.append(polyline(pts,stroke="#2E7B25",sw=2.0))
    for i,(x,y) in enumerate(pts): c.append(circle(x,y,4.5,fill="#D94343" if i in (7,8,9) else "#5AA93E",stroke="#8E2222" if i in (7,8,9) else "#1E5D19",sw=0.9))
    c.append(polyline(pts[7:10],stroke="#D94343",sw=2.5)); c.append(f'<ellipse cx="438" cy="704" rx="38" ry="32" fill="none" stroke="#E33A3A" stroke-width="1.3" stroke-dasharray="5 4"/>')
    c.extend([line(195,742,690,742,sw=1.0),text(199,765,"1",10,"middle"),text(683,765,"40",10,"middle")])
    c.extend([rect(768,660,168,84,fill="#FDEDEE",stroke="none",rx=10),text(852,690,"Soft monotonicity",14,"middle","bold"),text(852,722,"ℒₘₒₙₒ",20,"middle",family="Times New Roman",italic=True),rect(768.75,660.75,166.5,82.5,fill="none",stroke="#DC2832",sw=1.5,rx=9)])
    c.append(arrow([(706,700),(756,700)],color="#DC2832",sw=1.6,head=8))
    c.append(arrow([(936,701),(981,701)],color="#DC2832",sw=1.4,head=7))
    c.extend([rect(983,665,90,36,fill="#FFFFFF",stroke="#DC2832",sw=1.2,rx=4,extra='stroke-dasharray="4 3"'),text(1028,688,"Δc ≤ 40",14,"middle",family="Times New Roman",italic=True),rect(983,707,90,36,fill="#FFFFFF",stroke="#DC2832",sw=1.2,rx=4,extra='stroke-dasharray="4 3"'),text(1028,731,"c ≥ 300",14,"middle",family="Times New Roman",italic=True)])
    return group("monotonic_path",c)


def rate_path():
    c=[text(104,816,"Predicted SOH",14,"middle","bold"),text(104,839,"(local segment)",12,"middle")]
    pts1=[(204,808),(228,803),(252,797),(277,791),(302,786),(326,780),(350,773),(366,778)]
    pts2=[(366,778),(380,804),(395,814),(411,822)]
    pts3=[(411,822),(438,820),(468,823),(500,827),(534,831),(570,835),(606,839),(644,842),(684,846)]
    c.extend([polyline(pts1,stroke="#2E7B25",sw=2.0),polyline(pts2,stroke="#D94343",sw=2.5),polyline(pts3,stroke="#73A95C",sw=2.0)])
    for x,y in pts1[:-1]: c.append(circle(x,y,4.2,fill="#5AA93E",stroke="#1E5D19",sw=0.8))
    for x,y in pts2: c.append(circle(x,y,4.2,fill="#D94343",stroke="#8E2222",sw=0.8))
    for x,y in pts3[1:]: c.append(circle(x,y,4.2,fill="#8BC66D",stroke="#4A842F",sw=0.8))
    c.append(f'<ellipse cx="384" cy="803" rx="39" ry="47" fill="none" stroke="#E33A3A" stroke-width="1.3" stroke-dasharray="5 4"/>')
    c.extend([line(195,861,690,861,sw=1.0),text(199,885,"t − w",11,"middle",family="Times New Roman",italic=True),text(368,885,"t",11,"middle",family="Times New Roman",italic=True),text(682,885,"t + w",11,"middle",family="Times New Roman",italic=True),text(444,907,"cycle index",11,"middle")])
    c.extend([rect(768,789,168,84,fill="#F3EFFA",stroke="none",rx=10),text(852,819,"Rate continuity",14,"middle","bold"),text(852,851,"ℒᵣₐₜₑ",20,"middle",family="Times New Roman",italic=True),rect(768.75,789.75,166.5,82.5,fill="none",stroke="#6642A3",sw=1.5,rx=9)])
    c.append(arrow([(706,831),(756,831)],color="#6642A3",sw=1.6,head=8))
    return group("rate_path",c)


def objective_group():
    c=[rect(1255,663,360,103,fill="#EEF5FB",stroke="none",rx=10),text(1435,703,"Training objective",15,"middle","bold"),text(1435,739,"ℒ = ℒₛᵤₚ + λₘₒₙₒℒₘₒₙₒ + λᵣₐₜₑℒᵣₐₜₑ",18,"middle",family="Times New Roman",italic=True),rect(1255.75,663.75,358.5,101.5,fill="none",stroke="#1F5E9E",sw=1.5,rx=9)]
    return group("objective",c)


def training_connectors():
    return group("training_connectors",[
        arrow([(936,579),(1198,579),(1198,699),(1253,699)],color="#E45F13",sw=1.7,head=8),
        arrow([(1073,701),(1198,701),(1198,715),(1253,715)],color="#DC2832",sw=1.7,head=8),
        arrow([(936,831),(1198,831),(1198,731),(1253,731)],color="#6642A3",sw=1.7,head=8)
    ])


def panel_b_group():
    children=[rect(10,499,1652,428,fill="#FFFFFF",stroke="none",rx=13),training_connectors(),panel_header("b","Sparse-label physics-informed learning",499),sparse_path(),monotonic_path(),rate_path(),objective_group(),rect(10.75,499.75,1650.5,426.5,fill="none",stroke="#AEB4B8",sw=1.5,rx=12)]
    return group("panel_b",children)


def svg_doc(contents, viewbox=(0,0,W,H), width=None, height=None):
    x,y,w,h=viewbox
    width = width or w
    height = height or h
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="{x} {y} {w} {h}">
<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#FFFFFF"/>
{contents}
</svg>'''


def main():
    manifest=json.loads((ROOT/'component_manifest.json').read_text(encoding='utf-8'))
    pa=panel_a_group(); pb=panel_b_group()
    full=svg_doc(group('pi_mscl_figure',[pa,pb]))
    (ROOT/'full.svg').write_text(full,encoding='utf-8')

    module_map={
        'panel_a':pa,'input_sequence':matrix_module(),'multiscale_cnn':cnn_module(),'fusion':fusion_module(),'lstm':lstm_module(),'regression':regression_module(),'output_curve':output_module(),
        'panel_b':pb,'supervised_path':sparse_path(),'monotonic_path':monotonic_path(),'rate_path':rate_path(),'objective':objective_group()
    }
    modules=ROOT/'modules'; modules.mkdir(exist_ok=True)
    by_id={c['id']:c for c in manifest['components']}
    for cid, content in module_map.items():
        b=by_id[cid]['bbox']; pad=8
        vb=(b['x']-pad,b['y']-pad,b['width']+2*pad,b['height']+2*pad)
        (modules/f'{cid}.svg').write_text(svg_doc(content,viewbox=vb),encoding='utf-8')
    print(f'Wrote full.svg and {len(module_map)} modules')


if __name__=='__main__':
    main()
