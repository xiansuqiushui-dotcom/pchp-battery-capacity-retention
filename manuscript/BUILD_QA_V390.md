# Build, render, and clean-replay QA — V390

Run date: 2026-08-03

## Build commands

```text
latexmk -g -pdf -interaction=nonstopmode -halt-on-error main_en.tex
latexmk -g -xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
latexmk -g -pdf -interaction=nonstopmode -halt-on-error supplement_en.tex
latexmk -g -xelatex -interaction=nonstopmode -halt-on-error supplement_zh.tex
```

## Final artifacts

| File | Pages | SHA-256 |
|---|---:|---|
| `main_en.pdf` | 44 | `883a12b73f71940aecdc1b13c48dd8e6ca0a469bc4fefafec2203a5ee4788f23` |
| `main_zh.pdf` | 28 | `233a26f2a75fb86295ad2d2c20e629cc969daeeca43c21f75b046ea24fa9a7fa` |
| `supplement_en.pdf` | 20 | `e8ec99a02924d32b6a2689a8267f4b93e346be36166b95aeeba3c773b9db45f9` |
| `supplement_zh.pdf` | 18 | `104ec4f216f038ee47904e2815f8e6a462d8b16ead38db1a712e26aa3693308c` |

## Automated log scan

The final four logs contain zero overfull boxes, undefined citations,
undefined references, LaTeX errors, fatal errors, emergency stops, missing
glyphs, missing referenced files, or objects reported as too wide. The two
remaining BibTeX notices concern `zhang2024batteryml` and
`deb2025conservative`, which are ICLR proceedings records without conventional
page ranges. The verified AAAI SAFER entry now includes pages 2217--2223.

## Visual inspection

All 110 pages were rendered at 90 dpi and inspected in contact sheets. The
English and Chinese title/abstract pages and representative dense reference
pages were also inspected at full rendered resolution. No overlap, clipping,
unexpected blank page, broken table, displaced caption, unreadable axis,
missing Chinese glyph, or malformed hyperlink was found. The revised bounded
novelty sentence fits both abstract layouts without changing the page counts.

## Clean reproducibility replay

- Package: `paper_q1/pchp_battery_capacity_retention_release_v390`.
- Release payload files: 425; all are SHA-256 pinned except the manifest and
  generated receipt, which are excluded to prevent circular self-reference.
- Strict JSON: all 74 payload JSON files parse with non-standard constants
  rejected.
- Standard discovery: 40 core tests and 15 update tests pass; the validated
  V390 public API adds 2 passing tests.
- Portable verifier: `python verify_release_v390.py --write-receipt` passes
  44 named top-level checks, including the immutable V368 verifier, exact
  external estimand and hierarchical bootstrap replay, V384--V387 result tests,
  record identity, manuscript metadata, citation keys, and all four PDFs.
- The external prediction ledger still contains one legitimate tied nominal
  cycle key; `external_record_identity_v390.csv` assigns unique stable record
  identifiers and within-key ordinals to all 9,712 records.
- No third-party raw archive, fitted model bundle, secret, or file at or above
  the GitHub 100 MB hard limit is present.

The manifest pins the distributed PDF bytes. Rebuilding can alter only PDF
creation/modification metadata while preserving text, geometry and content
streams; such a rebuild must not be mistaken for scientific drift.
