# Prefix-Causal Harm-Budget Projection for Battery Capacity-Retention Estimation

This is the frozen reproducibility release accompanying the manuscript
*Prefix-Causal Harm-Budget Projection for Cross-Domain Lithium-Ion Battery
Capacity-Retention Estimation*.

Repository: https://github.com/xiansuqiushui-dotcom/pchp-battery-capacity-retention

PCHP is an output-side deployment layer. It preserves every emitted prefix,
enforces a recursively feasible non-increasing trajectory, and bounds the
per-record increase in absolute loss relative to a protected causal state
before the capacity outcome is observed.

## Release contents

- `manuscript/`: final bilingual manuscript and supplementary sources, compiled
  PDFs, figures, figure builders, submission assets, and audit records.
- `core_v368/`: the hash-pinned reproducibility core audited through V378. It
  retains the causal method, development evaluation, BaSyTec validation,
  theoretical falsification, backbone portability, and finite-sample audits.
  Historical submission-planning files inside this exact legacy snapshot are
  preserved solely to satisfy its published hashes and are not current
  submission documents.
- `updates_v389/`: the final external mechanism confirmation, decision-cost
  analysis, external robustness, threshold-event audit, Transformer stress
  test, monotonicity ablation, output-bound sensitivity, and sample-flow audit.
- `verify_release_v389.py`: portable release verifier.
- `manifest_v389.json`: SHA-256 manifest for every tracked release file.
- `verification_receipt_v389.json`: clean-environment verification receipt for
  the published snapshot.

No third-party raw archive or fitted model bundle is redistributed. The saved
tables are derived research artifacts required to verify reported estimands and
deterministic certificates. See `NOTICE.md` and `core_v368/DATA_SOURCES.md`.

## Quick verification

Python 3.10 is the frozen reference environment.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python verify_release_v389.py --write-receipt
```

The verifier checks the complete manifest, replays the legacy core verifier,
recomputes the final six-dataset mechanism estimand and its two-stage bootstrap,
runs the V384--V387 tests, validates the manuscript's citation keys and central
numbers, and rejects forbidden raw-data or model-bundle file types.

The manifest excludes itself and the generated verification receipt to avoid
self-referential hashes; all payload files remain hash-pinned.

To rebuild the four PDFs, run from `manuscript/`:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main_en.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement_en.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error supplement_zh.tex
```

## Evidence boundary

The deterministic PCHP certificate is pointwise and does not require a sampling
assumption. Empirical utility is evaluated at the declared hierarchical units:
complete development domains, external datasets with nested physical cells, or
physical cells within BaSyTec. The six-dataset analysis is protocol-frozen
external mechanism confirmation because those outcomes did not influence the
tested method, comparator, estimand, cost protocol, or decision criteria. It is
not newly collected laboratory evidence. BaSyTec is outcome-blind,
protocol-locked validation on a held-out public laboratory program.

## License and citation

Author-created software and documentation are released under the MIT License.
Third-party data and derived artifacts remain subject to their source terms; see
`NOTICE.md`. Citation metadata are provided in `CITATION.cff`.
