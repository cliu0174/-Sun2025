# Overleaf compilation package

This folder is a self-contained Springer manuscript project. Upload the **contents of this folder** to one Overleaf project and set `main.tex` as the main document.

## Compiler

- Compiler: `pdfLaTeX`
- Bibliography tool: `BibTeX`
- Recommended build sequence: `pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX`

## Folder layout

- `main.tex`: manuscript source
- `references.bib`: bibliography database
- `sn-jnl.cls`, `sn-mathphys-num.bst`: Springer template files
- `Figure2.png`: the HUST trajectory figure cited by the current manuscript
- `Figure1.png`, `Figure3.png`--`Figure6.png`: legacy, unreferenced figure files retained only for local history; they are not part of the current manuscript argument and may be omitted from an Overleaf upload

All formal results are embedded directly in `main.tex`; no red provisional values or placeholder result figures remain. All tables are embedded directly in `main.tex`; no auxiliary `.tex` files or external relative paths are required. The generated PDF after compilation is `main.pdf`.
