# Project handoff — Applied Energy V382

## Authoritative version

`rccp_causal_manuscript_v382_applied_energy` is the authoritative journal-adapted version. The preceding V381 directory remains unchanged as the archival baseline.

## Manuscript identity

- English title: *Risk-Controlled Cross-Domain Lithium-Ion Battery Capacity-Retention Estimation via Prefix-Causal Harm-Budget Projection*.
- Chinese title: *基于前缀因果损害预算投影的跨域锂离子电池容量保持率风险可控估计*.
- Method: prefix-causal harm-budget projection (PCHP; 前缀因果损害预算投影).
- Primary application: updating deployed capacity-retention records from partial charge traces before reference-capacity labels become available.

## Controlled terminology

| English | Chinese | Usage note |
|---|---|---|
| capacity retention | 容量保持率 | Primary prediction target; do not replace with remaining capacity. |
| state of health (SOH) | 健康状态（SOH） | Use only where the broader battery-health concept or cited literature requires it. |
| domain adaptation | 域适配 | Preferred over literal variants such as domain fitting. |
| protected state | 受保护状态 | Causal reference prediction used in the harm guarantee. |
| learned candidate | 学习候选值 | Model-proposed adaptive update before projection. |
| harm budget | 损害预算 | Maximum permitted increase in absolute loss relative to the protected state. |
| prefix causality | 前缀因果性 | Previously issued outputs cannot depend on future records. |
| matched constant-offset control | 匹配常数偏移对照 | Replaces the opaque phrase “candidate-free control / 候选无关控制”. |
| domain-equal MAE | 域等权平均绝对误差 | Each complete domain contributes equally. |
| dataset-equal MAE | 数据集等权平均绝对误差 | Each external dataset contributes equally. |
| continuous asymmetric cost | 连续非对称代价 | Decision-linked loss; distinct from binary review-decision error. |

## Evidence hierarchy

1. Deterministic guarantee: for every admissible hidden outcome, each issued update remains within the declared absolute-loss harm budget relative to the protected state.
2. Nested development evaluation: 12 complete domains, 586 cells, and 601,932 records; domain-equal MAE improves by 0.00486 and 11 of 12 domains improve.
3. Matched constant-offset control: PCHP improves domain-equal MAE by 0.00477, ruling out a purely uniform offset explanation on average while retaining the stated heterogeneity caveat.
4. Frozen six-dataset external mechanism confirmation: 659 cells and 9,712 records; dataset-equal MAE improves by 0.00690 and 5 of 6 datasets improve.
5. Operational connection: at a 5:1 missed-degradation-to-unnecessary-review cost ratio, continuous asymmetric cost decreases by 7.54%; the binary review decision at 80% retention remains unchanged.
6. Boundary evidence: BaSyTec quantifies constraint transfer and the accuracy trade-off; NASA supports relative harm control under severe target shift but not a high-accuracy absolute-SOH claim.

## Claim boundaries that must remain

- Do not claim universal domain-wise dominance: PCHP improves 9 of 12 domains against the matched constant-offset control.
- Do not convert a confidence interval on the domain-equal mean into a claim that a randomly drawn new domain is more likely than not to improve.
- Do not call the retrospective budget-path control independent confirmation.
- Do not use the NASA stress test to claim high-accuracy absolute SOH estimation.
- Do not claim fewer binary review errors in the reported decision experiment; the supported benefit is lower continuous asymmetric cost with an unchanged binary decision.

## Reproducibility and release state

- Main and supplemental sources compile successfully.
- Scientific figures are generated programmatically; no generative image model was used for manuscript figures.
- The public repository is intentionally pending. No GitHub upload was performed in V382.
- Before submission, either publish the frozen author-created package and insert its persistent DOI/URL or retain the current on-request availability statement if the journal permits it.

## Last validation

- English manuscript: 52 pages.
- Chinese manuscript: 33 pages.
- English supplement: 5 pages.
- Chinese supplement: 5 pages.
- No undefined references, overfull boxes, or float-placement warnings in the final logs.
- Only benign underfull line warnings remain: one English body line and three long-URL reference lines in the Chinese manuscript.

