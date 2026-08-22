#!/usr/bin/env python3
"""Convert docx to LaTeX (XeLaTeX/ctex, Chinese thesis)."""

import re
import sys
from docx import Document
from docx.oxml.ns import qn

DOCX_PATH = r"D:\Projects\1111-soh\第四章_面向部分生命周期监督的多尺度物理一致性SOH估计方法_全文扩写参考文献增强稿.docx"
TEX_PATH  = r"D:\Projects\1111-soh\第四章_面向部分生命周期监督的多尺度物理一致性SOH估计方法_全文扩写参考文献增强稿.tex"

# ---------------------------------------------------------------------------
# LaTeX preamble
# ---------------------------------------------------------------------------
PREAMBLE = r"""\documentclass[12pt, a4paper]{ctexart}

% ---- 基础宏包 ----
\usepackage{amsmath, amssymb, amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{multirow}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{geometry}
\usepackage{setspace}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{enumitem}
\usepackage{float}

% ---- 页面设置 ----
\geometry{
  a4paper,
  top=2.54cm, bottom=2.54cm,
  left=3.17cm, right=3.17cm
}

\onehalfspacing

% ---- 字体 ----
\setCJKmainfont{SimSun}        % 宋体（或改为 Source Han Serif CN 等）
\setCJKsansfont{SimHei}        % 黑体
\setmainfont{Times New Roman}

% ---- 章节编号深度 ----
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}

\hypersetup{
  colorlinks=true,
  linkcolor=black,
  citecolor=blue,
  urlcolor=blue
}

\begin{document}

"""

POSTAMBLE = r"""
\end{document}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPECIAL = {
    '&': r'\&',
    '%': r'\%',
    '$': r'\$',
    '#': r'\#',
    '_': r'\_',
    '{': r'\{',
    '}': r'\}',
    '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}',
    '\\': r'\textbackslash{}',
}
_SPECIAL_RE = re.compile('|'.join(re.escape(k) for k in _SPECIAL))


def escape(text: str) -> str:
    """Escape LaTeX special characters (skip math mode markers)."""
    return _SPECIAL_RE.sub(lambda m: _SPECIAL[m.group()], text)


def run_text(run) -> str:
    """Convert a single Run to LaTeX, handling bold/italic."""
    text = run.text
    if not text:
        return ''
    result = escape(text)
    if run.bold:
        result = r'\textbf{' + result + '}'
    if run.italic:
        result = r'\textit{' + result + '}'
    return result


def para_text(para) -> str:
    """Concatenate all runs of a paragraph."""
    return ''.join(run_text(r) for r in para.runs)


def is_formula_line(text: str) -> bool:
    """Heuristic: line looks like an inline formula label."""
    stripped = text.strip()
    return bool(re.match(r'^(公式|式)\s*[\d\-\–]+', stripped) or
                re.match(r'^[A-Za-z0-9_\^{}\s=\+\-\*\/\(\)\[\],.]+$', stripped) and len(stripped) < 120)


def is_figure_placeholder(text: str) -> bool:
    stripped = text.strip()
    return ('图片占位' in stripped or '图片留白' in stripped or
            stripped.startswith('图') and '占位' in stripped)


def convert_formula_text(text: str) -> str:
    """Wrap obvious formula lines in equation environment (best-effort)."""
    # Remove "公式X-X" prefix/suffix labels
    cleaned = re.sub(r'(公式|式)\s*[\d\-\–]+\s*', '', text).strip()
    if cleaned:
        return '\n\\begin{equation}\n' + cleaned + '\n\\end{equation}\n'
    return ''


def table_to_tex(table) -> str:
    """Convert a docx Table to a LaTeX table."""
    n_cols = max(len(row.cells) for row in table.rows)
    col_spec = 'l' * n_cols

    lines = [
        '',
        '\\begin{table}[H]',
        '  \\centering',
        '  \\begin{tabular}{' + col_spec + '}',
        '  \\toprule',
    ]

    for i, row in enumerate(table.rows):
        cells = []
        for cell in row.cells:
            cell_text = escape(cell.text.strip().replace('\n', ' '))
            cells.append(cell_text)
        # Pad if merged cells reduce count
        while len(cells) < n_cols:
            cells.append('')
        row_str = ' & '.join(cells) + r' \\'
        lines.append('  ' + row_str)
        if i == 0:
            lines.append('  \\midrule')

    lines += [
        '  \\bottomrule',
        '  \\end{tabular}',
        '  \\caption{}',
        '  \\label{tab:}',
        '\\end{table}',
        '',
    ]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def convert(docx_path: str, tex_path: str) -> None:
    doc = Document(docx_path)

    # Collect tables indexed by their XML element id so we can detect inline
    table_elements = {id(t._tbl): t for t in doc.tables}
    table_done = set()

    out = [PREAMBLE]

    # Walk all block-level elements (paragraphs + tables) in order
    body = doc.element.body
    for child in body:
        tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag

        # ---------- TABLE ----------
        if tag == 'tbl':
            tbl_id = id(child)
            if tbl_id not in table_done:
                table_done.add(tbl_id)
                table = table_elements.get(tbl_id)
                if table:
                    out.append(table_to_tex(table))
            continue

        # ---------- PARAGRAPH ----------
        if tag != 'p':
            continue

        # Find matching paragraph object
        para = None
        for p in doc.paragraphs:
            if p._p is child:
                para = p
                break
        if para is None:
            continue

        text = para.text
        stripped = text.strip()
        style = para.style.name

        if not stripped:
            out.append('')
            continue

        # --- Headings ---
        if style == 'Heading 1':
            # Remove leading chapter number if present (e.g. "第四章 ...")
            title = escape(stripped)
            out.append(f'\n\\section{{{title}}}\n')
            continue

        if style == 'Heading 2':
            title = escape(stripped)
            out.append(f'\n\\subsection{{{title}}}\n')
            continue

        if style == 'Heading 3':
            title = escape(stripped)
            out.append(f'\n\\subsubsection{{{title}}}\n')
            continue

        # --- List ---
        if style == 'List Paragraph':
            content = para_text(para)
            out.append(f'\\begin{{itemize}}\n  \\item {content}\n\\end{{itemize}}')
            continue

        # --- Figure placeholder ---
        if is_figure_placeholder(stripped):
            out.append(
                '\n\\begin{figure}[H]\n'
                '  \\centering\n'
                '  % \\includegraphics[width=0.8\\textwidth]{figures/placeholder}\n'
                f'  \\caption{{{escape(stripped)}}}\n'
                '  \\label{fig:}\n'
                '\\end{figure}\n'
            )
            continue

        # --- Caption lines (图X-X / 表X-X) ---
        if re.match(r'^[图表]\s*\d', stripped):
            out.append(f'% {escape(stripped)}\n')
            continue

        # --- Formula label lines ---
        if re.match(r'^[（(]公式|^公式\s*\(', stripped):
            out.append(f'% formula label: {escape(stripped)}\n')
            continue

        # --- Normal paragraph ---
        content = para_text(para)
        if content.strip():
            out.append(escape(content) + '\n')
            out.append('')

    out.append(POSTAMBLE)

    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))

    print(f'Written: {tex_path}')


if __name__ == '__main__':
    convert(DOCX_PATH, TEX_PATH)
