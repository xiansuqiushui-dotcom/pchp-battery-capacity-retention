# Local full-recomputation protocol

The review-lite package omits third-party raw data and fitted source-model bundles. Full local recomputation therefore requires provider-authorized data acquisition. Replaying an already observed dataset can reproduce numerical results but cannot recreate the historical outcome-isolation claim.

## Method-level checks

```powershell
python -m unittest -v test_prefix_causal_harm_projection_v321.py
python validate_asymmetric_harm_extension_v340.py
python validate_time_varying_viability_v341.py
```

## Development inputs

| Required input | Frozen identity source |
|---|---|
| `batterylife_early_charge_soh_v109.parquet` | `rccp_source_model_freeze_v316/rccp_source_models_v316_metadata.json` |
| `batterylife_external_hust_rwth_v124.parquet` | same metadata record |
| `batterylife_sdu_early_charge_soh_v145.parquet` | same metadata record |
| `batterylife_matr_early_charge_soh_v151.parquet` | same metadata record |

Do not substitute same-named files with different hashes. Acquire BatteryLife V11 from Zenodo record `10.5281/zenodo.19688272`, construct the frozen domain tables with the included builders, and verify every feature-table identity before fitting.

The historical development/NASA execution order was:

```powershell
python freeze_rccp_source_models_v316.py
python build_nested_source_only_alpha_selections_v326.py
python score_nested_prefix_causal_outer_v327.py
python freeze_external_causal_source_spec_v328.py
python build_nasa_label_blind_predictions_v330.py
python score_nasa_frozen_confirmation_v331.py
python evaluate_candidate_information_control_v333.py
python analyze_source_group_sensitivity_v339.py
python analyze_source_tuned_causal_candidate_v342.py
python audit_source_tuned_causal_candidate_statistics_v342.py
python analyze_budget_path_candidate_control_v361.py
python audit_budget_path_anchor_v362.py
python evaluate_bounded_recovery_pchp_v374.py
python audit_pchp_backbone_portability_v375.py
python verify_pchp_backbone_portability_v375.py
```

The V328 source-only freeze preceded NASA archive access, and V330 label-blind predictions preceded V331 scoring. The official NASA archive must match SHA-256 `82302a7db4fc1b34e0b6676326610438d43b816bdf11a69d1d012a464ef2f92e`.

The V361 budget path was locked before its own computation but after development outcomes had been opened. Its original `NARROW` report must remain in the lineage. V362 is an audit of the V361 historical-anchor implementation against already existing high-precision V326/V327 CSV files; it must not overwrite V361 or be presented as a rerun that restores prospective status.

V374 and V375 are also protocol-locked retrospective development audits because all twelve development-domain outcomes had already been opened. V374 tested a bounded positive-recovery extension and failed its frozen interval and domain-win gates; the failed decision must not be rescued by widening the recovery grid or changing the estimand. V375 replaced both regressors with Ridge, HistGradientBoosting, or LightGBM without secondary-backbone tuning and reused the exact V326 outer-domain $\alpha$ schedule. Its retained result excludes an ExtraTrees-specific explanation within those three audited families only; it is not external confirmation, universal model independence, deep-model portability, or best-backbone superiority.

The review-lite package includes the V375 record-level derived predictions so that `verify_pchp_backbone_portability_v375.py` can independently reconstruct all domain effects and deterministic certificates. It omits the third-party raw development tables and the V374 rejected outer-prediction table; full retraining still requires the provider-authorized inputs above.

## BaSyTec numerical replay

Acquire Zenodo record `10.5281/zenodo.15755725` and verify all $48$ ZIP files against `external_basytec_v343/download_receipt_v343.json`. Keep F0001 only as the declared parser-development cell; it is excluded from the $47$-cell confirmation roster.

The archived chronology is V343 schema failure, V347 pilot declaration, V348 heterogeneous-header failure, V351 header-only inventory, V352 final freeze, V353 label-blind prediction, and V354 one-shot scoring. The numerical replay order after acquiring the provider files is:

```powershell
python prepare_basytec_confirmation_roster_v347.py
python inventory_basytec_headers_v351.py
python test_basytec_header_family_v355.py
python build_basytec_label_blind_predictions_v353.py
python score_basytec_frozen_confirmation_v354.py
python audit_basytec_external_statistics_v357.py
python analyze_basytec_excluded_condition_v358.py
```

The statistical audit requires the V354 scored-record table, and the full-roster sensitivity requires the original provider ZIP files. Neither is redistributed in review-lite. The condition map derives from the provider's official degradation-metrics workbook and is included only as a compact derived JSON record.

## Figure replay

```powershell
python paper_q1\rccp_causal_manuscript_v368\make_figures_v335.py
python paper_q1\rccp_causal_manuscript_v368\make_basytec_figure_v359.py
python paper_q1\rccp_causal_manuscript_v368\make_budget_path_figure_v363.py
```

## V366 loss-geometry replay

The V366 validator requires SymPy in addition to the inherited numerical stack. It reads no battery outcomes and deterministically reconstructs the metric identity checks, the bounded squared-loss supremum and feasible interval, active-boundary tightness, exterior-point violations, recursive feasibility, and the two full-real-line divergence directions:

```powershell
python validate_loss_geometry_v366.py
```

The expected result is status `RETAIN_FOR_V366`, seed $20{,}260{,}802$, tolerance $10^{-12}$, and all $27$ named checks passing. This replay validates formulas and implementation only. It does not establish literature priority, empirical squared-loss utility, a prospective laboratory result, deployment cost calibration, or electrochemical safety.

## V368 Applied Energy submission-asset replay

The V368 manuscript preserves the frozen primary scientific output and adds the reviewer-risk closures documented through V376. V376 changes only closest-neighbor positioning and the explicit design-independence evidence boundary; it adds no dataset, method component, tuning, or empirical result. Rebuild the graphical abstract with:

```powershell
python paper_q1\rccp_causal_manuscript_v368\make_graphical_abstract_v368.py
```

The expected PNG is $2340\times900$ pixels. The V368 verifier regenerates it in a temporary directory and checks its deterministic PNG hash, while also checking the abstract word limit, highlight character limits, current concurrent-work identities, AI-use declaration, V374 negative-result preservation, V375 claim boundary, and V376 direct-neighbor citations. These checks establish submission-artifact integrity, not journal acceptance or institutional Q1/Top status.

The full local rebuild remains subject to every upstream licence and to the eventual licence selected for author-created code.
