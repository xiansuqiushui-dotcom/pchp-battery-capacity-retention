# Prefix-Causal Harm-Budget Projection for Battery Capacity-Retention Estimation

This is the V390 frozen reproducibility release accompanying the manuscript
*Prefix-Causal Harm-Budget Projection for Cross-Domain Lithium-Ion Battery
Capacity-Retention Estimation*.

Repository: https://github.com/xiansuqiushui-dotcom/pchp-battery-capacity-retention

PCHP is an output-side deployment layer. It preserves every emitted prefix,
enforces a recursively feasible non-increasing trajectory, and bounds the
per-record increase in absolute loss relative to a protected causal state
before the capacity outcome is observed.

## Release contents

- `manuscript/`: final bilingual manuscript and supplementary sources, compiled
  PDFs, figures, submission assets, and reviewer-risk records.
- `core_v368/`: the immutable, hash-pinned reproducibility core audited through
  V378. Historical planning files inside this snapshot are provenance records,
  not current submission documents.
- `updates_v389/`: the frozen V380--V387 scientific update snapshot, including
  final prediction ledgers, result tables, protocols, tests, and historical
  full-recomputation scripts.
- `pchp_release_api_v390.py`: validated public wrapper around the unchanged
  hash-pinned PCHP operator. It rejects missing or empty cell identifiers and
  produces exactly the frozen output for valid inputs.
- `replay_v390.py` and `REPLAY_READINESS_V390.md`: executable preflight and an
  explicit classification of direct replay versus provider-dependent refitting.
- `verify_release_v390.py`: authoritative portable release verifier.
- `manifest_v390.json`: SHA-256 manifest for every release payload file.
- `verification_receipt_v390.json`: clean-environment verification receipt.

No third-party raw archive or fitted model bundle is redistributed. Saved CSV
and Parquet tables are derived research artifacts required to verify the
reported estimands and deterministic certificates. See `NOTICE.md`,
`core_v368/DATA_SOURCES.md`, and `core_v368/FULL_REPRODUCTION.md`.

## Quick verification

Python 3.10 is the frozen reference environment.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python verify_release_v390.py --write-receipt
```

The verifier checks every payload hash, strict JSON syntax, the release
boundary, the unique V390 external record ledger, the final six-dataset
estimand and two-stage bootstrap, the validated public API, standard unit-test
discovery, the immutable V368 verifier, V384--V387 result tests, bilingual
manuscript numbers and citation keys, and the four compiled PDFs.

The manifest excludes itself and the generated verification receipt to avoid
self-referential hashes. All other non-ignored payload files are hash-pinned.

## Replay boundary

Run the following command before invoking an individual historical audit:

```powershell
python replay_v390.py
```

V384 external robustness and V383 threshold-decision audits replay directly
from the included V380 prediction ledger. Several refitting and record-level
historical audits require provider-authorized raw or derived inputs that are
intentionally absent. `replay_v390.py --run <script>` fails before execution
with an explicit missing-input report; it never downloads or substitutes data.
The included result-ledger tests remain the authoritative review-lite checks for
those workflows. See `REPLAY_READINESS_V390.md` for the complete matrix.

## PDF rebuild

From `manuscript/`, run:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main_en.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement_en.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error supplement_zh.tex
```

The manifest pins the distributed PDF bytes. A local LaTeX rebuild can produce
identical text, geometry, figures, and content streams while changing only PDF
creation/modification timestamps; rerun `make_manifest_v390.py` only when
preparing a new versioned release, never merely to conceal an unexplained
scientific difference.

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
Third-party data and derived artifacts remain subject to their source terms;
see `NOTICE.md`. Citation metadata are provided in `CITATION.cff`.
