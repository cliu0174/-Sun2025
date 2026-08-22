# PI-MSCL English Manuscript Log

> Working directory: `latex/els-cas-templates/`  
> Main TeX file: `cas-dc-template.tex`  
> Bibliography file: `references.bib`  
> Template: Elsevier CAS double-column (`cas-dc.cls`)  
> Target: SCI pre-submission draft, journal not yet specified  
> Method name: PI-MSCL (Physics-Informed Multi-Scale CNN-LSTM)  
> Scope decision: conservative version based on the current Word manuscript only; do not include Stage 1-5 full-stack modules. The title now emphasizes robust SOH estimation rather than making partial lifecycle supervision the sole framing.

## Global Decisions

- Use double-column Elsevier CAS template.
- Remove standalone `Problem Formulation` section; merge it into `Methodology` as `Partial lifecycle supervision setting`.
- Keep `Trajectory-Based Abnormal Degradation Risk Indication` as a full section, but position it as an extension/application rather than the main contribution.
- Leave in-house dataset metadata as placeholders until the user provides details.
- Use Overleaf for compilation because local LaTeX tools (`pdflatex`, `latexmk`, `bibtex`, etc.) are not available in the current environment.
- Current `.bib` is a manually organized draft based mainly on the Chinese Word manuscript references plus light web/checking support; it is not yet an official publisher/Crossref-exported BibTeX library.

## Directory State

### Kept Files

- `cas-dc-template.tex` - main manuscript file.
- `references.bib` - current bibliography draft.
- `cas-dc.cls` - Elsevier double-column class file.
- `cas-common.sty` - Elsevier CAS common style macros.
- `cas-model2-names.bst` - BibTeX style.
- `soh_cn_version.docx` - source Chinese manuscript.
- `figures/` - figure folder, currently empty.

### Removed Template Files

- Single-column class/template/sample files.
- Elsevier sample PDFs and sample `.tex` files.
- `README`, `manifest.txt`, `doc/`, `thumbnails/`, and sample `figs/`.

## Section Status

| Section | Status | Notes | Next Work |
|---|---|---|---|
| Front matter | Drafted | Title updated to `Physics-Informed Multi-Scale CNN-LSTM for Robust Lithium-Ion Battery State-of-Health Estimation`. Abstract is intentionally a working draft. Highlights, keywords, and author placeholder written. | Fill affiliation; expand abstract after Methodology, Results, and risk-indication sections are complete. |
| 1. Introduction | Drafted | Full English draft written with single-key citations rather than stacked multi-citation groups. Structure: SOH importance -> existing methods -> incomplete supervision/robustness gap -> physics-informed motivation -> PI-MSCL contributions. Contribution list uses bullet items, not numbered items. | Later polish after Methodology/Results are finalized; verify all citations from official sources. |
| 2. Methodology | Drafted | Full English method section written with equations for input sequence, masked supervised loss, multi-scale branches, feature fusion, LSTM mapping, soft monotonic loss, temporal decay, and total objective. | Later polish after Overleaf compilation; replace framework placeholder with final figure. |
| 2.1 Partial lifecycle supervision setting | Drafted | Defines input sequence, SOH label, supervision mask, and masked MSE. Uses T=40 and F=14. | Check notation against Experimental Setup after dataset section is written. |
| 2.2 Overview of PI-MSCL | Drafted | Describes full PI-MSCL pipeline and includes a compilable placeholder figure. | Replace placeholder with final architecture figure in `figures/`. |
| 2.3 Multi-scale convolutional feature extraction | Drafted | Describes kernels 3/7/15, two 64-channel layers per branch, concatenation, and 1x1 fusion to 128 channels. | Verify exact wording after architecture figure is drawn. |
| 2.4 LSTM-based temporal degradation modeling | Drafted | Describes two-layer LSTM with hidden dimension 64 and sigmoid-bounded SOH regression. | Add dropout value only if final training config confirms it should be reported here. |
| 2.5 Physics-informed training objective | Drafted | Defines pair set, soft monotonic loss, exponential temporal decay, and total loss. Final PI-MSCL settings: lambda_mono=0.3, epsilon=0.005, c_min=300, K=20, alpha=0.2, boundary/smoothness weights 0. | Check Overleaf line breaks for equations in double-column layout. |
| 3. Experimental Setup | Drafted | Written with HUST dataset description, SOH definition, 14-feature table, preprocessing, 60/20/20 battery-level split, supervision ratios, baseline/ablation matrix, training configuration, and evaluation metrics. In-house dataset metadata remains explicitly pending. | Later verify table width on Overleaf; fill in-house metadata when available. |
| 4. Results and Discussion | Drafted | Full-supervision benchmark, architecture/physics ablation, partial-supervision robustness, and unseen-cell trajectory case study have been written. Tables migrated: full-supervision metrics, 2x2 ablation matrix, ablation across supervision ratios, MAE across ratios, and partial-supervision MAE/RMSE/R2 comparison. | Replace trajectory placeholders with final figures; verify table width on Overleaf. |
| 5. Trajectory-Based Abnormal Degradation Risk Indication | Drafted | Written as a bounded engineering extension, not a standalone fault-diagnosis system. Includes trajectory-deviation score, tau=mu+2sigma threshold, N=20 persistence rule, AUC/Det@FPR5%/Delay metrics, and abnormal-scenario performance table. | Replace risk-trajectory and risk-score placeholders with final figures; confirm in-house dataset metadata and abnormal-scenario description before submission. |
| 6. Conclusion | Drafted | Written with contribution summary, full/partial supervision evidence, abnormal-risk extension, and explicit limitations. | Revisit after final figures, in-house metadata, and abstract are finalized. |
| Data Availability | Placeholder | HUST public; in-house details pending. | Add exact HUST dataset link/reference and in-house availability statement. |
| Declaration / Acknowledgements | Placeholder | Generic declaration and empty acknowledgements. | Fill before submission. |

## Bibliography Status

### Current State

- `references.bib` now contains 38 entries after the user-supplied BibTeX update: 37 article entries plus the HUST dataset entry `Yuan2022HUSTDataset`. All 38 entries are currently cited in the manuscript.
- Citation key consistency checked after the BibTeX expansion: all `\cite{}` keys in `cas-dc-template.tex` exist in `references.bib`; no current `\cite{a,b,c}` stacked multi-citation groups remain; duplicate key `ma2022real` was removed from the supplied BibTeX.
- Unescaped `&` in journal names has been fixed as `\&`.

### Important Caveat

The current BibTeX file is still not a final official-export bibliography. It now includes the user-supplied Google Scholar-style entries plus light cleanup, so author lists, DOI fields, title capitalization, and publisher metadata should still be verified against DOI/Crossref/publisher pages before submission.

### High-Priority Entries to Verify Later

- `Lu2023DeepLearning`
- `Ma2022PersonalizedTransfer`
- `Lin2023UnlabeledChargingData`
- `Wang2024PINNBattery`
- `Ye2024PINNSOH`
- `Lin2025PhysicsInformedSOH`
- `Zhu2022AttentionCNNBiLSTM`

### Bibliography QA Checklist

- Verify each entry against DOI/Crossref/publisher page.
- Add missing DOI values where available.
- Replace incomplete author lists or `others` fields with correct BibTeX syntax.
- Protect terms such as `{SOH}`, `{BMS}`, `{CNN-LSTM}`, `{PI-MSCL}`, `{Li-ion}` if needed.
- Remove entries not used in the final manuscript.

## Figure and Table Status

| Item | Status | Notes |
|---|---|---|
| `figures/` folder | Created | Empty for now. |
| Architecture figure | Missing | Need to export/redraw PI-MSCL framework figure. |
| HUST degradation trajectories | Missing | Need figure from source data or existing Word media. |
| Feature correlation heatmap | Missing | Need figure from source data or existing Word media. |
| Feature table | Drafted in TeX | 14 charging-cycle features migrated from Word Table 4-1. |
| Full-supervision benchmark table | Drafted in TeX | Migrated from Chinese Word manuscript Table 4-2. |
| Ablation tables | Drafted in TeX | Migrated 2x2 architecture/physics ablation and supervision-ratio ablation tables. |
| Partial-supervision MAE/RMSE/R2 tables | Drafted in TeX | Migrated MAE ratio table and CNN-LSTM vs PI-MSCL multi-metric table. |
| Abnormal risk indication figures/tables | Table drafted; figures pending | Risk detection table migrated from Chinese Word manuscript. Two figure placeholders added for abnormal trajectories and trajectory-deviation scores. In-house metadata still pending. |

## Compilation Notes

- Local environment cannot compile TeX: `latexmk`, `pdflatex`, `xelatex`, `lualatex`, `bibtex`, `miktex`, `tlmgr`, and `tectonic` were not found in PATH.
- Use Overleaf for now.
- Recommended Overleaf compiler: `pdfLaTeX`.
- Main file: `cas-dc-template.tex`.
- Upload the whole `latex/els-cas-templates/` folder to preserve `cas-dc.cls`, `cas-common.sty`, and `cas-model2-names.bst`.

## Next Recommended Step

Expand the abstract so it reflects the completed Methodology, Results, and abnormal-risk indication sections. Then update Data Availability and verify Overleaf layout for all double-column tables and placeholders.







