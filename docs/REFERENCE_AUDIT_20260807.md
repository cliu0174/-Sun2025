# Reference audit for the Springer manuscript (2026-08-07)

## Scope

- Active manuscript: `latex/springer-sn-2024/manuscript.tex`
- Bibliography: `latex/springer-sn-2024/references.bib`
- Compiled citation evidence: `latex/springer-sn-2024/manuscript.aux` and `manuscript.bbl`
- Requirement under audit: at least 40 real, searchable references used by the manuscript.

## Local consistency audit

- BibTeX entries: **43**.
- Entries with a DOI: **43/43**.
- References present in the compiled bibliography: **42**.
- Unique citation keys used by the active compiled manuscript: **42**.
- Unique citation keys used specifically in the Introduction before `\section{Methodology}`: **40**.
- Citation keys missing from the BibTeX database: **0**.
- BibTeX entries not cited by the active manuscript: **1** (`zhang2023deep`).

The active manuscript therefore satisfies both interpretations of the numerical requirement: 42 references are compiled in the manuscript as a whole, and 40 unique sources are cited within the Introduction itself. The uncited entry must not be inserted merely to increase the count; it should be cited only if a revised sentence genuinely needs it.

## External metadata verification

The DOI list was checked against authoritative registration metadata on 2026-08-07.

- **42/42 journal-article DOIs** resolved through the Crossref REST API. Crossref returned a `journal-article` record, publisher, and title for every item. The returned titles matched the corresponding local BibTeX titles after ignoring capitalization, punctuation, and typographic dash differences.
- The HUST dataset DOI, **10.17632/nsc7hnsg4s.2**, is not registered in Crossref, as expected for this repository record. It resolved through the DataCite REST API as:
  - title: *The Dataset for: Real-time personalized health status prediction of lithium-ion batteries using deep transfer learning*;
  - publisher/repository: Mendeley;
  - publication year: 2022;
  - resource type: Dataset;
  - landing page: `https://data.mendeley.com/datasets/nsc7hnsg4s/2`.

## Decision

The requirement of at least 40 real and searchable references is **verified for the current compiled manuscript**. No additional reference should be added solely for numerical padding. Future edits must preserve at least 40 compiled bibliography items and must continue to keep claim-to-citation placement defensible.

## Remaining citation-quality check

DOI validity and title identity do not by themselves prove that every citation supports the exact sentence in which it appears. The final manuscript review must therefore retain a separate claim-level check, especially for statements about sparse supervision, weak labels, physics-informed learning, and the interpretation of HUST label masking.
