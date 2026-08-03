# PCHP reproducibility package (V368 review-lite release, audited through V378)

This package supports the manuscript *Risk-Controlled Cross-Domain Lithium-Ion Battery State-of-Health Estimation via Prefix-Causal Harm-Budget Projection*.

## What can be verified without third-party raw data

The package retains the complete causal-method, comparator, NASA, BaSyTec, budget-path, loss-geometry, exact-viability, rejected-recovery, and untuned-backbone evidence layers. It can independently:

- verify prefix causality, monotonicity, physical range, exact harm-tube membership, asymmetric costs, time-varying recursive feasibility, and the exact online viability kernel;
- recompute the development-domain, candidate-control, source-group, NASA, and BaSyTec aggregate statistics at their declared independent-unit levels;
- verify that the source-tuned unprotected comparator is more accurate but violates the declared displacement budget in all $12$ development domains;
- authenticate the BaSyTec schema-failure lineage, header-only audit, parser-development exclusion, final freeze, outcome-blind predictions, one-shot scoring, and independent statistical audit;
- enforce the conclusion that BaSyTec externally confirms the declared harm-control and relative-utility contract but does not identify adaptive superiority over a fixed positive shift in that dataset;
- verify the V361 budget path and the separate V362 precision-anchor audit while retaining the original V361 `NARROW` record;
- replay all $27$ V366 loss-geometry aggregate checks and enforce the bounded-outcome and full-real-line claim boundaries;
- verify the V373 exact-viability and closest-neighbor closure and its future-continuation and maximal-kernel property tests;
- retain V374 as a rejected bounded-recovery extension because its frozen $95\%$ interval crossed zero and only $7/12$ domains improved, despite all deterministic certificates passing;
- independently recompute the V375 Ridge, HistGradientBoosting, and LightGBM domain effects and deterministic certificates from the included record-level derived predictions, passing $196$ named checks;
- verify the V376 closest-neighbor positioning against conservative OCO, stage-wise conservative bandits, and online learning with unknown constraints, while enforcing design independence rather than absolute dataset-lineage isolation as the external-confirmation standard;
- replay the V377 label-free operator-stability validator, including the exact local proximal form, exhaustive local nonexpansiveness checks, $30{,}000$ random trajectory pairs, and a tight unit-Lipschitz witness;
- verify that the V377 theorem is positioned against moving-horizon monotonic-trend and monotonic-walk filters as a scoped prefix-level stability result, not as the first stable monotone filter;
- replay the V378 post-hoc finite-sample audit over all $2^{12}=4{,}096$ sign assignments for three domain-level mean effects and enforce the distinction between a domain-equal mean gain and universal or majority-probability dominance;
- verify the bilingual manuscript, abstracts, highlights, AI-use declaration, reviewer-risk audits, graphical abstract, and submission assets;
- regenerate every bilingual manuscript figure and the graphical abstract and verify deterministic PNG hashes;
- reject missing, modified, untracked, or forbidden third-party raw/archive/model files.

Run:

```powershell
python make_manifest_v368.py
python verify_reproducibility_v368.py --regenerate-figures --write-receipt
```

The current verifier also calls `verify_pchp_backbone_portability_v375.py`, which reconstructs the V375 statistics and certificates directly from the saved prediction table. PDF byte hashes are recorded by the package manifest, while regenerated scientific figures use deterministic PNG hashes because equivalent PDF builds may differ in metadata or font-object ordering.

## Evidence labels and boundaries

BaSyTec is classified as an **outcome-blind, prospectively locked external validation on held-out public data**. Its confirmatory status follows from design independence: held-out capacity outcomes did not affect the method, fitted predictors, hyperparameters, estimand, exclusion logic, or decision gates, and predictions were frozen before capacity access. This does not require newly collected data or an impossible claim that no person had ever encountered the dataset. It is not a prospectively collected laboratory study or an independent-laboratory replication.

The budget path, V374, and V375 are **protocol-locked retrospective development analyses** because all $12$ development-domain outcomes had already been opened. V374 must remain rejected. V375 excludes an ExtraTrees-specific explanation within one linear and two boosting families under zero additional backbone tuning; it does not establish external confirmation, universal model independence, deep-model portability, or best-backbone superiority.

The V366 loss geometry is a **pre-frozen theoretical and implementation falsification layer**. The metric identity follows from the triangle inequality and is not claimed as a new primitive. The bounded squared-loss result requires a truthful outcome interval, and the impossibility proof covers outcomes spanning $\mathbb{R}$ rather than every unbounded subset. No squared-loss battery-utility result is claimed.

The V377 prefix-nonexpansiveness result is also a **pre-frozen theoretical and implementation falsification layer**. It proves that the complete causal two-predictor cascade is prefix-wise $\ell_\infty$-nonexpansive under a fixed output contract, with tight constant $1$. This is incremental algorithmic stability only; it does not bound baseline error, statistical uncertainty, calibration, distribution shift, or electrochemical safety.

The V378 finite-sample audit is a **post-hoc sensitivity on the $12$ opened development domains**, not a new confirmation. Exact sign-flip, exact Wilcoxon, and Student-$t$ calculations retain the negative domain-equal mean directions. The candidate-free comparison wins in $9/12$ domains but its direction-only sign-test value is $0.145996$; the manuscript therefore claims a heterogeneous domain-equal mean utility contribution, not universal dominance or a probability-of-improvement statement for a new domain.

No third-party raw archive, MAT file, BaSyTec record-level table, development source table, or fitted model bundle is included. The package contains the historically permitted derived NASA Parquet table and the V375 derived prediction Parquet required for independent portability verification. Full retraining still requires provider-authorized inputs.

## Full recomputation and release status

`FULL_REPRODUCTION.md` lists provider records, required local inputs, and the historical execution order. Numerical replay after outcome access can reproduce results but cannot recreate the original information chronology; the archived freeze and failure receipts remain the chronology evidence.

Public release still requires the author and institution to choose a licence for author-created code. Until then, `LICENSE_PENDING.md` is authoritative.
