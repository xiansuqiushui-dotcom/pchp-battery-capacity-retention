# Applied Energy submission components (V368 draft)

Status: synchronized with the V368 submission working manuscript; author and compliance fields remain open.

## Proposed title

Risk-Controlled Cross-Domain Lithium-Ion Battery State-of-Health Estimation via Prefix-Causal Harm-Budget Projection

## Abstract

Battery-management systems in a new fleet can receive partial charging records before reference capacity labels become available. Cross-domain learned updates can then cause negative transfer, while retrospective monotone smoothing can revise already-issued SOH records. We propose prefix-causal harm-budget projection (PCHP), an output operator separating a protected source predictor from a learned candidate. PCHP constructs a one-sided causal baseline and projects each candidate into the intersection of a declared harm tube, the physical range, and recursively feasible non-increasing outputs. For every possible outcome, the absolute-loss increase relative to the protected state is at most the budget without distributional assumptions; this intersection is the exact online viability kernel: every admitted output remains extendable, and the projection is the unique closest-candidate solution. Loss-geometry analysis derives exact metric and bounded squared-loss regions and proves that no nontrivial finite-budget squared-loss update exists over the real line. Nested leave-one-domain-out evaluation across 12 domains, 586 cells, and 601,932 records reduced domain-equal cell-macro mean absolute error from 0.07524 to 0.07038. The paired change was -0.00486 with a domain-cluster 95% interval of [-0.00671, -0.00301], and 11 of 12 domains improved. A stronger unprotected causal comparator reached 0.05470 but violated the harm tube in every domain, quantifying the harm-control--accuracy opportunity cost. In outcome-blind, protocol-locked external confirmation, PCHP reduced cell-equal error from 0.17496 to 0.16510 across 45 cells, with all cells improving. A NASA stress test retained the deterministic certificate but exposed target incompatibility. PCHP therefore enables auditable, risk-bounded online battery-health adaptation before reference capacity becomes available.

## Highlights

- PCHP admits cross-domain SOH updates before capacity labels are available.
- Exact projection bounds per-record harm and never rewrites issued SOH.
- Exact viability kernel joins harm budgets with irreversible SOH trajectories.
- Nested evaluation improves 11 of 12 domains under the harm budget.
- Outcome-blind external confirmation improves all 45 eligible cells.

## Recommended generative-AI declaration for author verification

During the preparation of this work, the author used OpenAI Codex to assist with research-code development and verification, literature-search organization, manuscript structuring, translation, and language editing. The author reviewed and independently verified all source claims, mathematical derivations, analyses, code outputs, figures, and text and takes full responsibility for the content of the work. All scientific figures were generated programmatically from the reported data and scripts; no generative image model was used to create or alter manuscript figures.

This statement must be checked and approved by every listed author before submission. It is not authorized for insertion until the final authorship record is known.
