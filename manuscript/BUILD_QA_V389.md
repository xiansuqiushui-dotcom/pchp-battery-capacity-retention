# Build, render, and clean-replay QA — V389

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
| `main_en.pdf` | 44 | `8bc5af58f5eb55702f48d7bcd854e29b3e5b80f8d53213e4675a3b86199b9bb4` |
| `main_zh.pdf` | 28 | `55f2436e4a74e44a75cf9e4e065a2fa36a68bce7a372f9c3ba671fb9e2ec60e1` |
| `supplement_en.pdf` | 20 | `e6edc590f39285a0f857dd89ad04884a048eb1c2a48a19f38c9b58bd1a406863` |
| `supplement_zh.pdf` | 18 | `46afeec65da46126dc512b4a40d4055a6dba9f94d001a33c919ae55785519c5e` |

## Automated log scan

All four logs contain zero matches for overfull boxes, undefined citations,
undefined references, LaTeX errors, fatal errors, emergency stops, missing
referenced files, and objects reported as too wide. Three BibTeX warnings concern
empty conventional page ranges for sources that publish article identifiers or
proceedings metadata; no citation is unresolved.

## Visual inspection

Latest rendered pages are stored under `tmp/pdfs/v389/`.

- English main: pages 1, 3, 8, 22, 26, 28, 33, and 44.
- Chinese main: pages 1, 3, 5, 13, 16, 20, 21, and 28.
- Figure 3 was regenerated after shortening the bilingual horizontal-axis label;
  both PDF and PNG variants were inspected directly.
- Prior V388 supplement renders remain valid because neither supplement source
  changed in V389.
- After public-repository URL integration, English page 35 and Chinese page 21
  were re-rendered and inspected; the named GitHub links are legible and remain
  within the text block.

No overlap, clipping, missing glyph, displaced caption, unreadable coordinate,
or broken reference was found.

## Clean reproducibility replay

- Package: `paper_q1/pchp_battery_capacity_retention_release_v389`.
- Tracked release files: 418; the current manifest SHA-256 is recorded in
  `verification_receipt_v389.json` to avoid a circular self-reference.
- `.gitattributes` disables end-of-line conversion so mixed LF/CRLF frozen
  artifacts retain the exact bytes recorded by the manifest on every platform.
- Author-created code license: MIT.
- Forbidden third-party raw/archive/model suffixes: none.
- Files at or above the GitHub 100 MB hard limit: none.
- Public repository:
  `https://github.com/xiansuqiushui-dotcom/pchp-battery-capacity-retention`.
- Isolated environment: Python 3.10.11 with dependencies reinstalled from
  `requirements.txt`.
- Command: `python verify_release_v389.py --write-receipt`.
- Outcome: 29/29 top-level release checks passed, including the full V368 core,
  exact final external estimand and two-stage bootstrap replay, citation-key
  resolution, and V384--V387 tests.
