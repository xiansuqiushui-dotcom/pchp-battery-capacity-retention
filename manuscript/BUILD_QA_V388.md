# Build and visual QA — V388

Run date: 2026-08-03

## Build commands

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error main_en.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement_en.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error supplement_zh.tex
```

## Final artifacts

| File | Pages | SHA-256 |
|---|---:|---|
| `main_en.pdf` | 44 | `1b17597fb6968febdaaf162216c1bce6460aea59b9b8a141765441c585482fbe` |
| `main_zh.pdf` | 28 | `8f79a93e994a84e7ebda52d83ab3eb6505707bd24a787e60c73b288482c08ea3` |
| `supplement_en.pdf` | 20 | `e6edc590f39285a0f857dd89ad04884a048eb1c2a48a19f38c9b58bd1a406863` |
| `supplement_zh.pdf` | 18 | `46afeec65da46126dc512b4a40d4055a6dba9f94d001a33c919ae55785519c5e` |

## Automated log scan

The final four logs contain zero matches for:

- overfull boxes;
- undefined citations or references;
- LaTeX errors;
- fatal errors;
- emergency stops;
- missing referenced files;
- objects reported as too wide.

BibTeX reports three non-blocking metadata warnings for entries whose sources do not provide conventional page ranges. These warnings do not create undefined references or broken links.

## Visual QA

Rendered pages and contact sheets are stored in `qa_render_v388/`.

- English main: title/abstract, method workflow, closest-prior table, core theorem, external confirmation, BaSyTec, limitations, declarations.
- Chinese main: title/abstract, method workflow, closest-prior table, core theorem, external confirmation, BaSyTec, limitations/conclusion, declarations.
- English and Chinese supplements: title pages, monotonicity/output-bound audits, NASA target-compatibility figure, complete theory appendix, and theory verification records.
- Graphical abstract: verified at (2340\times900) pixels with no clipping, overlap, or unreadable labels at the inspected resolution.

No overlap, clipping, displaced caption, missing glyph, or broken figure was found in the inspected pages.
