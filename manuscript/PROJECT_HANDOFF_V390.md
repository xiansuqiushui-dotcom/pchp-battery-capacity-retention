# Project handoff — V390

## Target and paper memory point

- Target: Applied Energy, research article.
- Memory point: in cross-domain battery capacity-retention updates issued before reference outcomes arrive, PCHP couples a protected prefix-causal state with a per-update absolute-loss harm budget and prefix immutability; its value is supported by nested development evidence, protocol-frozen six-dataset confirmation, decision-cost analysis, and outcome-blind BaSyTec boundary validation.

## Canonical terminology

- Method: prefix-causal harm-budget projection (PCHP); 中文：前缀因果损害预算投影。
- Target: capacity retention; 中文：容量保持率。
- Protected object: protected state; 中文：受保护状态。
- Main comparator: development-selected constant budget-feasible offset; 中文：开发数据选出的常数预算可行偏移。
- Development estimand: domain-equal cell-macro MAE.
- External estimand: surface-equal mean PCHP-minus-fixed-shift cell-macro MAE.
- Evidence labels: nested development evidence; protocol-frozen external mechanism confirmation; outcome-blind BaSyTec validation; NASA target-incompatibility stress test.

## Supported claims and evidence

1. Deterministic contract: `core_v368/prefix_causal_harm_projection_v321.py`, theory and falsification records in `core_v368/paper_q1/`.
2. Development evidence: 12 domains, 586 cells, 601,932 records; main Tables 2--3 and Figures 3--5.
3. External mechanism confirmation: 6 datasets, 659 cells, 9,712 records; `updates_v389/external_mechanism_decision_v380/` and main Figure 6.
4. Decision value: prespecified threshold/cost grid in V380 and main Figure 7; the primary continuous cost comparison uses a 5:1 missed-degradation-to-unnecessary-review ratio.
5. Boundary evidence: 45-cell BaSyTec evaluation in main Figure 8; complete conditions in the supplement.
6. Reviewer-risk closures: V384 robustness, V385 monotonicity ablation, V386 output-range sensitivity and V387 sample flow.

## Authoritative files

- English manuscript: `main_en.tex` and `main_en.pdf`.
- Chinese manuscript: `main_zh.tex` and `main_zh.pdf`.
- Supplements: `supplement_en.tex/.pdf` and `supplement_zh.tex/.pdf`.
- Bibliography: `references.bib`.
- Submission metadata: `submission_metadata_en_v390.md`.
- Replay boundary: `../REPLAY_READINESS_V390.md` and `../replay_v390.py`.
- Release verifier: `../verify_release_v390.py`.
- Manifest and receipt: `../manifest_v390.json`, `../verification_receipt_v390.json`.

## Frozen identifiers and QA

- Scientific parent: Git commit `157f411f38f42a195786ce14f7ab9b6afdf241f0`, tag `v389`.
- V390 changes submission wording and reproducibility engineering only; frozen predictions and headline estimands are unchanged.
- Release payload: 425 files; portable verifier: 44/44 checks passed.
- PDF hashes and complete render QA are recorded in `BUILD_QA_V390.md`.
- Public repository: https://github.com/xiansuqiushui-dotcom/pchp-battery-capacity-retention.

## Remaining author-controlled actions

1. Confirm the corresponding-author phone in the submission portal.
2. Generate the journal-required Declaration of Interests file.
3. Confirm both authors approved the submission and it is not under review elsewhere.
4. Confirm that the AI-use statement's manual source, derivation, analysis, code-output, figure and text review is factually accurate.
5. Optionally mint and insert a Zenodo DOI after the exact V390 release is public.

No additional dataset, model refit, experiment, or manuscript claim expansion is required before submission.
