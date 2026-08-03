# Project handoff — Applied-energy journal version V383

## Authoritative version

`rccp_causal_manuscript_v383_applied_energy` is the current authoritative manuscript directory. V382 and earlier directories are archival baselines and must not be edited in place.

## Current completion state

- The six mandatory revision items are closed and documented in `MANDATORY_REVISION_CHECKLIST_V383.md`.
- R1 external aggregation and omission robustness is closed. The unchanged V380 predictions were audited in V384 across dataset-, cell-, record-equal and median estimands, all six leave-one-dataset-out estimates, largest-dataset removal, and all \(2^6=64\) exact sign assignments. All seven verification tests passed.
- R2 monotonicity-contract ablation is closed. V385 compared the strict trajectory contract with a clean pointwise-tube arm that removes only the previous-output cap and with the pre-existing bounded-recovery alternative contract. All eight verification tests passed. Strict PCHP removed \(38{,}580\) upward revisions across \(483/586\) cells, while its domain-equal MAE difference from the pointwise arm was \(+0.000260\) with interval \([-0.000538,0.001095]\).
- R3 numerical-output-bound sensitivity is closed. V386 replayed the unchanged operator at upper bounds \(1.1,1.2,1.3,1.5\). Development, the six external datasets, and BaSyTec had zero changed outputs, invariant MAE, and complete deterministic-certificate passage. The audit classifies \(1.3\) as a nonbinding pre-existing computational bound, not an accuracy-tuned constant. NASA was replayed on 34 cells and 2,598 label-blind records before scoring 33 cells and 2,556 matched labels; its incompatible target scale cannot select the bound.
- R4 evidence and sample flow is closed. V387 reconstructs 20 dataset surfaces from frozen ingestion reports and final ledgers, with eight verification tests passed. The development identity is \(605{,}005-143-5\times586=601{,}932\) scored records; the six-external identity is \(945-286=659\) scored cells and 9,712 records. Missing upstream raw-record totals are explicitly NR rather than inferred. Aggregate flow is in both main texts, and the complete surface ledger and exclusion reasons are in both supplements.
- R5--R10 are closed: the main texts are compressed, the visible theory is focused, NASA is supplementary only, BaSyTec has one consistent interpretation, and the final bilingual title, abstract, introduction and contribution framing are stable.
- R11 scientific and file preparation is complete. The submission package now includes separate Highlights, a programmatically generated graphical abstract, submission metadata, a cover letter, a declaration-of-interests consistency text, and an Applied Energy checklist. Only author-entered portal fields remain.
- English and Chinese main texts and supplements compile successfully and have been visually checked after the V389 completion audit.
- Current PDFs: English main text 44 pages; Chinese main text 28 pages; English supplement 20 pages; Chinese supplement 18 pages.
- Current SHA-256 values: `main_en.pdf` `8bc5af58f5eb55702f48d7bcd854e29b3e5b80f8d53213e4675a3b86199b9bb4`; `main_zh.pdf` `55f2436e4a74e44a75cf9e4e065a2fa36a68bce7a372f9c3ba671fb9e2ec60e1`; `supplement_en.pdf` `e6edc590f39285a0f857dd89ad04884a048eb1c2a48a19f38c9b58bd1a406863`; `supplement_zh.pdf` `46afeec65da46126dc512b4a40d4055a6dba9f94d001a33c919ae55785519c5e`.
- Final local release package: `paper_q1/pchp_battery_capacity_retention_release_v389`, with 418 manifest-tracked files, byte-preserving Git attributes, MIT licensing for author-created code, no forbidden third-party raw archive/model bundle, and 29/29 top-level checks passed in a clean Python 3.10.11 environment.
- Public repository: <https://github.com/xiansuqiushui-dotcom/pchp-battery-capacity-retention>. The bilingual Data Availability statements, README, NOTICE and `CITATION.cff` contain the real URL; no third-party raw archive was added during publication.
- All manuscript and repository tasks in `REMAINING_REVISION_TRACKER_V383.md` are closed. Remaining actions are author-controlled submission-portal fields and an optional Zenodo DOI.

## Evidence hierarchy

1. Deterministic guarantee: each admissible update remains within the declared absolute-loss harm budget relative to the protected state.
2. Nested development evidence: 12 complete domains, 586 cells and 601,932 scored records.
3. Mechanism control: comparison with the development-selected constant budget-feasible offset.
4. Frozen external mechanism confirmation: 6 independent battery data surfaces, 659 cells and 9,712 post-reference records.
5. External robustness: the PCHP-minus-constant-offset direction remains negative under all declared weighting, median, leave-one-dataset-out, and largest-dataset-removal analyses; the two-sided exact sign-flip value is \(0.0625\) for six top-level units and is reported as a small-sample sensitivity rather than a second confirmation claim.
6. Operational value: threshold-based timing metrics and decision-linked asymmetric cost.
7. Boundary evidence: BaSyTec for constraint transfer and accuracy trade-off; NASA is retained only in the supplement as severe target-incompatibility stress testing.

## Data-use boundary

Local presence authorizes computation within this project, not automatic scientific independence or redistribution. Before a data asset enters the manuscript evidence chain, verify its source identity, physical-cell identity, target definition, information chronology and independent analysis unit. Before public release, verify the upstream license separately and omit third-party raw archives unless redistribution is unambiguously allowed and intentionally chosen.

## Fixed claim boundaries

- Do not count processed copies or experimental branches as new datasets.
- Do not claim universal domain-wise dominance.
- Do not turn an interval for a domain-equal mean into a probability statement about a new domain.
- Do not describe retrospective development evidence as external confirmation.
- Do not use NASA to claim high-accuracy absolute capacity-retention estimation.
- Do not claim fewer binary review errors where the supported result is a reduction in continuous asymmetric cost with unchanged binary decisions.
- State supported contributions directly and confidently; do not dilute them with repetitive defensive qualifications.

## Remaining execution order

1. The authors enter the corresponding-author phone number and generate the Elsevier declaration-of-interests file in the submission system.
2. The GitHub repository is public and the real URL has been integrated into both manuscripts and release metadata; the linked pages have been recompiled and visually checked.
3. A Zenodo DOI may be added later if a versioned persistent archive is desired. Independently, inspect the PDF generated by the Elsevier portal before submission.

## V389 completion and submission artifacts

- `TERMINOLOGY_CLAIM_AND_REVIEWER_AUDIT_V388.md`
- `FINAL_SUPER_SKILL_COMPLETION_AUDIT_V389.md`
- `BUILD_QA_V389.md`
- `APPLIED_ENERGY_SUBMISSION_CHECKLIST_V389.md`
- `highlights.txt`
- `graphical_abstract_applied_energy_v388.pdf`
- `graphical_abstract_applied_energy_v388.png`
- `submission_metadata_en_v388.md`
- `cover_letter_en_v388.txt`
- `competing_interest_declaration_text_v388.txt`
- `qa_render_v388/`
