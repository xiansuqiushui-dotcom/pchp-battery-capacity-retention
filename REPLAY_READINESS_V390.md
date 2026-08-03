# V390 replay-readiness matrix

This release distinguishes evidence verification from full model refitting. It
does not redistribute third-party raw archives, fitted model bundles, or large
rejected intermediate ledgers when their source terms or review-lite scope do
not permit or require redistribution. A script's presence records provenance;
it does not by itself mean that every upstream input is included.

Run `python replay_v390.py` to list machine-readable readiness, or
`python replay_v390.py --check <script>` for one entry point. `--run` refuses to
start when a required input or optional dependency is absent; it never downloads
or silently substitutes data.

| Entry point | Release status | What can be verified here |
|---|---|---|
| `audit_external_robustness_v384.py` | Direct replay | Recomputes aggregation, leave-one-dataset-out and exact small-sample sensitivity from the frozen V380 predictions. |
| `audit_external_threshold_decisions_v383.py` | Direct replay | Recomputes threshold-event and decision summaries from the frozen V380 predictions. |
| `audit_monotonicity_ablation_v385.py` | Provider-dependent full recomputation | The generated tables and tests are included; the rejected V374 record ledger is intentionally absent. |
| `audit_output_range_sensitivity_v386.py` | Provider-dependent full recomputation | Generated cap summaries and tests are included; V374, BaSyTec scored-record and NASA record ledgers are not redistributed. |
| `audit_data_flow_v387.py` | Provider-dependent full recomputation | The final surface-flow ledger and tests are included; upstream provider-derived ingestion tables are omitted. |
| `audit_feature_weighting_pchp_factorial_v383.py` | Provider-dependent refit | Frozen result tables are included; raw development tables and several pre-projection ledgers are omitted. |
| `audit_pchp_sequence_transformer_v383.py` | Historical archival refit | Frozen result tables document the audit; the historical sequence engines, raw curve tables and optional PyTorch environment are not part of the review-lite release. |
| `run_pchp_external_mechanism_decision_v380.py` | Provider-dependent refit | The final record ledger, estimand replay and deterministic certificates are included; full refitting requires all provider-authorized source tables. |

The portable `verify_release_v390.py` is the authoritative direct verification
entry point. It checks every payload hash, strictly parses every JSON file,
recomputes the six-dataset estimand and hierarchical bootstrap, verifies record
identities, runs both unit-test suites, replays the immutable V368 verifier and
runs the V384--V387 result tests. Full refitting remains governed by
`core_v368/FULL_REPRODUCTION.md` and the source terms in `NOTICE.md`.

