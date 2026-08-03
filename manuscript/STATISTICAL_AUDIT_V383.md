# V383 statistical audit

## Unit-of-analysis rules

- Development effects and the $2\times2$ attribution audit use the complete held-out dataset domain as the independent unit ($12$ paired domains).
- External mechanism confirmation uses the dataset as the top-level unit and physical cells as nested units ($6$ datasets, $659$ cells).
- Transformer results use the physical cell as the resampling unit within each evaluated domain (MATR and SDU).
- Threshold-event metrics are calculated within physical cells and then aggregated equally by dataset. Repeated cycle records are never treated as independent replications.

## Exact $2\times2$ attribution

The audit varied only the feature representation (raw versus commissioning-reference change) and source weighting (pooled versus domain-equal). All folds, regressors, row caps, imputers, random seed, protected predictor, harm budget, and outer-domain $\alpha$ schedule were held fixed. The selected arm reproduced the authoritative V327 ledger with maximum absolute error $0$.

Effects use $100{,}000$ complete-domain bootstrap resamples with seed $20{,}260{,}805$. Two-sided exact sign-flip values enumerate all $2^{12}=4{,}096$ assignments. After PCHP, change features improved MAE under pooled weighting by $-0.00566$ with $95\%$ interval $[-0.00941,-0.00235]$ and under domain-equal weighting by $-0.00550$ with interval $[-0.00970,-0.00184]$. Weighting effects and the interaction had intervals spanning zero.

## Transformer stress test

The battery-sequence Transformer was evaluated on all available MATR and SDU trajectories. It did not outperform the protected state; this negative result is retained. PCHP maintained the declared $0.01$ maximum displacement and realized-harm bounds with zero range or monotonicity violations. The experiment therefore supports architecture compatibility and harm containment, not Transformer utility superiority.

## Threshold-event estimands

Late warning means a predicted first threshold crossing after the observed first crossing or no predicted crossing during follow-up. Premature review means a predicted crossing before the observed first crossing; any predicted crossing is premature for a cell that does not cross during follow-up. Absent predicted crossings are censored to one record beyond the last observation only for restricted-delay calculation. Results describe the observed horizon and do not extrapolate unobserved end-of-life time.

At the primary $80\%$ threshold, protected state, constant offset, and PCHP issued identical first-crossing actions. Therefore the manuscript makes no binary timing-benefit claim; it retains the supported continuous asymmetric-cost result.
