# Build and submission QA — V391

Date: 2026-08-03 (Asia/Shanghai)

## Scope

V391 revises the title, abstract, terminology, applied framing, and submission presentation. The frozen datasets, models, predictions, estimands, figures, and V390 scientific-code release are unchanged.

## Manuscript builds

| Artifact | Pages | Bytes | SHA-256 |
|---|---:|---:|---|
| `main_en.pdf` | 44 | 770865 | `57862d3ae37d81fff9893c9ff4b2763fc878a934b8a3290f59cf5b7e5533fa6e` |
| `supplement_en.pdf` | 20 | 442515 | `dfb664876f41d23297c34ae2b428016525dc9819f7509f108da6f4d73df1a218` |
| `main_zh.pdf` | 28 | 743592 | `a977816ed273e0516793343dfd9b35196b8cbaf6ab1659211c35b6dae3106067` |
| `supplement_zh.pdf` | 18 | 320580 | `cddb4b6b923adff7cb6652e4d1cb59985f45e8b4ff978bdaa5e80942a37c5dad` |

- All four final logs were scanned for undefined citations/references, overfull boxes, missing characters, and fatal LaTeX errors; none were found.
- The English abstract contains exactly 250 words under the package verifier's tokenizer; the five highlights contain 73, 78, 75, 82, and 77 characters, respectively.
- All 110 pages were rasterized and visually inspected. Modified title, abstract, attribution, external-confirmation, and Transformer-portability pages received full-resolution review.
- No overlap, clipping, broken table, formula truncation, missing glyph, or abnormal blank-page defect was observed.

## Source and package reproducibility

- `09_LaTeX_Source.zip` contains exactly 12 required source files and no LaTeX build auxiliaries.
- Clean extraction and compilation produced 44-page and 20-page English PDFs whose rendered pages are pixel-identical to the upload references.
- The final package verifier passed 68/68 checks.
- The outer submission ZIP and nested source ZIP both passed CRC checks.
- The 19 manifest-controlled package entries passed byte-size and SHA-256 verification.

Final submission ZIP:

- File: `PCHP_Applied_Energy_Submission_v391.zip`
- Bytes: 1781965
- SHA-256: `fd5ec9f6ed45c7737c10c99ffe2f6bd16d4230c5e5b35946c74e2f32d5dbba03`

## DOCX rendering

- The local Documents Skill now prefers `soffice.com` on Windows and creates a fresh per-run LibreOffice `UserInstallation` under the render output directory.
- Windows `HOME` and `USERPROFILE` are preserved to avoid LibreOffice bootstrap failures.
- The cover letter and official Declaration of Interests were each rendered to PDF and PNG through independent profiles and visually inspected.
- The authoritative `cover_letter_en_v391.txt` source is paragraph-for-paragraph identical to the uploaded cover-letter DOCX.
- The modified Documents Skill passed Python syntax validation and `quick_validate.py`.
