# Applied Energy cover letter — V368 draft

> Submission placeholders must be completed and approved by all authors. Text in square brackets is not submission-ready.

[Date]

Editor-in-Chief  
Applied Energy

Dear Editor,

Please consider our Full Length Research Article, “Risk-Controlled Cross-Domain Lithium-Ion Battery State-of-Health Estimation via Prefix-Causal Harm-Budget Projection,” for publication in *Applied Energy*.

In a new battery fleet, partial charging records can arrive during ordinary operation before an accurate reference-capacity label is available. A cross-domain learned update may then improve average accuracy yet harm an unknown domain, while retrospective monotone smoothing may revise SOH records that have already been issued. The manuscript addresses this operational gap with prefix-causal harm-budget projection (PCHP), an online output-space operator that separates a protected source predictor from a learned candidate. For every hidden outcome, PCHP bounds the per-record increase in absolute loss relative to the protected causal state, enforces a declared non-increasing trend without accessing future records, and returns the unique feasible output closest to the current learned proposal.

The central innovation is a recursively feasible prediction contract that unifies prefix-only execution, a per-record harm budget, and an irreversible non-increasing output trajectory. The theory proves that the intersection of the exact harm region, the physical range, and the issued-record constraint is the exact online viability kernel: every admitted output remains extendable under every future non-increasing protected-state path, whereas every excluded output already violates the contract. It also proves unconditional zero-harm impossibility and identifies PCHP as the unique closest-candidate solution in that kernel. The evidence chain combines strict nested leave-one-domain-out evaluation across 12 public battery domains, 586 cells, and 601,932 records; a matched candidate-free control; a stronger unprotected comparator that quantifies the harm-control-accuracy opportunity cost; outcome-blind, protocol-locked external confirmation across 45 eligible cells held out from development; and a separate stress test that retains the deterministic certificate while exposing target incompatibility. The complete review-lite package reproduces the reported evidence, statistical replay checks, theory-implementation checks, and all figures without redistributing third-party raw archives.

We believe the manuscript fits *Applied Energy* because it contributes an artificial-intelligence and decision-control method for battery energy storage, connects partial-charge sensing and delayed reference-capacity outcomes to an operational maintenance constraint, and evaluates the resulting contract across heterogeneous battery systems. The result is a bounded, inspectable rule for deciding when a learned SOH update may be issued before target outcomes are known; its guarantee is relative to the protected predictor and is not presented as electrochemical or universal battery safety.

[AUTHOR CONFIRMATION REQUIRED: This manuscript is original, is not under consideration elsewhere, and has been approved by all authors.]  
[AUTHOR CONFIRMATION REQUIRED: All authors agree with the authorship order, competing-interest declaration, funding statement, data/code statement, and generative-AI disclosure.]  
[OPTIONAL: Suggested handling editor and reviewer information, only after conflict checks.]

Thank you for your consideration.

Sincerely,

Yuyang Wu  
[Affiliation]  
[Postal address]  
[Email]  
[ORCID]
