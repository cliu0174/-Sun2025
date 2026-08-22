# Springer manuscript goal-completion audit

> Last evidence check: 2026-08-08  
> Rule: `verified` requires an inspectable artifact or command result; work that is merely planned remains `in progress`.

| Requirement | Status | Authoritative evidence | Remaining gate |
|---|---|---|---|
| Use the Chinese document as the factual/narrative basis and avoid contradictions | In progress | Final cross-language audit completed against all 201 Chinese paragraphs and 7 tables; exact conflicts and replacement logic are recorded in `docs/CHINESE_SYNC_PROPOSAL_20260808.md`. The active English source contains no active self-built-data, 14-feature, or legacy-result wording. | Author approval is required before copying and rewriting the Chinese DOCX to the later-approved HUST masking, 16-feature, and formal multi-seed protocol. |
| Back up the pre-restructure manuscript and supporting notes | Verified | `backups/manuscript_pre_restructure_20260806/` contains the pre-restructure English TeX, Chinese DOCX, bibliography, improvement plan, narrative memo, and original template files. | None. |
| Use the leakage-free fixed split and paired multi-seed main matrix | Verified | `experiments/paper_main_results/fixed_split/v3_leakage_free`: 48/48 non-smoke runs, 16/16 cells x 3 repeats, 0 missing predictions, 0 errors; strict `summary.json` generated. | None. |
| Add fair same-protocol classical and advanced baselines | Verified | `experiments/paper_baselines/fixed_split/v1_leakage_free_endpoints`: 24/24 formal runs, 0 errors, 0 missing predictions; XGBoost/LSTM/GRU/attention CNN--LSTM share the fixed split and paired seeds; smoke excluded. | None. |
| Reuse existing results where protocol-compatible and retain unfavorable results | Verified | Legacy results were inventoried and isolated where protocols differed; N-001--N-011 preserve mixed/unfavorable outcomes. The active manuscript reports GRU/LSTM/attention advantages alongside PI-MSCL results. | Author decision remains only for whether the isolated legacy anomaly exploration receives a brief Discussion mention. |
| Generate 600 dpi group figures and independently rendered child panels | Verified | `output/figures/formal_main_v3`: every group and child panel has independently generated PNG/TIFF/PDF/SVG; PNG metadata 599.9988 dpi; compiled-page visual QA passed. | None. |
| Maintain an editable, high-impact-journal-style architecture diagram | Verified | `docs/PI-MSCL_architecture_v5.drawio` plus PDF/SVG/600 dpi PNG/TIFF exports; configuration audit; validator: 0 errors and 0 edge crossings; Springer page-6 visual check passed. | Optional final open in Draw.io desktop to check local font fallback; no structural work remains. |
| Use at least 40 real, searchable references in the Introduction | Verified | `latex/springer-sn-2024/references.bib`: 43 DOI-bearing entries; Introduction: 40 unique citation keys; compiled bibliography: 42 items; Crossref verified 42 journal articles and DataCite verified the HUST dataset. See `docs/REFERENCE_AUDIT_20260807.md`. | Claim-level citation placement remains part of the final prose review, but the numerical and identity requirements are proven. |
| Follow SCI academic writing conventions and avoid thin subsections | Verified | Active Results and Analysis integrates aggregate, per-cell, budget, baseline, trajectory, and factorial evidence; the standalone Discussion was removed without losing its bounded interpretation; reviewer Rounds 14--17 completed. | None. |
| Keep captions concise and explain figures/tables in the prose | Verified | Final figure/table captions identify content briefly; numerical meaning and limitations are developed in prose. Final pages 6 and 13--20 were visually inspected. | None. |
| Keep the Springer single-column manuscript above 20 pages without padding | Verified | `latex/springer-sn-2024/manuscript.pdf` has 23 pages after removing Discussion and integrating its analysis into Section 4; no overfull/undefined/float-too-large errors; all 23 pages were re-rendered and visually inspected. | None. |
| Record uncertainties, implementation deviations, and author decisions | Verified | `docs/NEGATIVE_RESULTS_AND_DECISIONS.md` records N-001--N-012 and D-001--D-013; `docs/EXECUTION_UPDATE_20260807.md`, `docs/CURRENT_COMPLETION_AND_FIGURE_PREVIEW.md`, `docs/CHINESE_SYNC_PROPOSAL_20260808.md`, and this audit preserve the remaining author choices. | Decisions remain open, but their existence, evidence, and permitted actions are fully recorded. |

## Explicit unresolved author decisions

- **Chinese synchronization:** the original DOCX still contains self-built-data, 14-feature, legacy numerical, and anomaly-risk claims contradicted by the later-approved formal protocol. `docs/CHINESE_SYNC_PROPOSAL_20260808.md` identifies exact paragraphs and tables. Approval is required before creating a dated synchronized copy; the original will not be overwritten.
- **Legacy anomaly-risk extension:** the source retains an entire anomaly section under `\iffalse`, so its two placeholders and old numerical table are not active in the compiled manuscript. The Exp-12 summary uses legacy 14-feature/checkpoint naming and lacks local per-run raw archives. Because the standalone Discussion has been removed, decide only whether a bounded sentence belongs in Section 4.5 or whether the extension remains fully excluded; do not restore a standalone section without traceable evidence. See D-008 in `docs/NEGATIVE_RESULTS_AND_DECISIONS.md`.
- **Code availability:** repository identifier and release status remain unconfirmed.

## Current gating path

1. Obtain the author's decision on synchronizing the Chinese DOCX's early self-built-data/14-feature/legacy-result wording to the later-approved HUST masking and 16-feature protocol.
2. Obtain the author's decision on whether the legacy anomaly-risk exploration remains excluded or receives one bounded sentence in Section 4.5.
3. Fill the code-availability repository identifier when the author provides it.
4. Optionally open the editable `.drawio` in the author's native Draw.io installation to verify local font fallback; structural and rendered QA are already complete.
# 2026-08-07 最终主矩阵状态更新

- 正式固定划分主矩阵：**48/48 完成**。
- 严格配对完整性：16/16 单元均为 3 次预注册配对重复；0 缺失预测；0 错误归档。
- 正式主结果表：已生成。
- Fig. 3--5：组图与独立子图均已生成 PNG/TIFF/PDF/SVG；PNG 元数据为 599.9988 dpi（即 600 dpi 导出）。
- 当前剩余关键路径：同协议端点基线、正文 Results/Discussion/Abstract/Conclusion 重写、最终编译与审稿人式自检。
