# DELTA NeurIPS 2026 Submission

## Setup

1. Download the NeurIPS 2026 style file (`neurips_2024.sty` or the latest available) from:
   https://neurips.cc/Conferences/2026/CallForPapers

2. Place `neurips_2024.sty` in this directory.

3. Compile:
   ```bash
   pdflatex delta_neurips2026.tex
   pdflatex delta_neurips2026.tex  # run twice for references
   ```

## Structure

- `delta_neurips2026.tex` — Main paper (9 pages + references + appendix)
- References are inline via `thebibliography` (no separate .bib needed)

## Key Tables

| Table | Content |
|-------|---------|
| 1 | Multi-hop MRR by depth (headline result) |
| 2 | Full 7-model multi-hop comparison |
| 3 | Multi-seed validation |
| 4 | Standard link prediction |
| 5 | Inference timing |
| 6 | Scaling analysis |

## Appendix Tables

| Table | Content |
|-------|---------|
| A.1 | Hits@10 by depth |
| A.2 | DropEdge robustness |
| A.3 | Brain architecture results |
| A.4 | Domain transfer (preliminary) |
