"""Exact current-pipeline 2x2 feature-by-weighting attribution for PCHP V383.

The audit varies only (i) raw versus raw-plus-commissioning-change features
and (ii) pooled versus domain-equal source weights.  All four raw candidate
streams use the same strict LODO folds, ExtraTrees specification, source-row
cap, imputation, and seed.  The frozen V326 outer-domain alpha schedule and
the same raw/pooled protected predictor are then used to wrap every candidate
with the current prefix-causal PCHP operator.

Two raw arms are reused from frozen ledgers and two missing arms are fitted.
Complete held-out dataset domains are the independent statistical units.
"""

from __future__ import annotations

import hashlib
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_method_ablation_v315 import RAW_ABSOLUTE_CHANGE_FEATURES
from develop_anchor_invariant_soh_v306 import (
    RAW_FEATURES,
    TARGET,
    balanced_source_rows,
    cell_metrics,
    domain_summary,
    fit_predict,
    load_data,
)
from develop_context_change_soh_v310 import add_context_change
from prefix_causal_harm_projection_v321 import prefix_causal_cellwise_projection


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "feature_weighting_pchp_factorial_v383"
V315 = ROOT / "method_ablation_v315" / "ablation_predictions_v315.parquet"
V322 = ROOT / "prefix_causal_rccp_v322" / "prefix_causal_predictions_v322.parquet"
V327 = ROOT / "nested_prefix_causal_outer_v327" / "nested_outer_predictions_v327.parquet"
ALPHAS = ROOT / "nested_prefix_causal_selection_v326" / "nested_alpha_selections_v326.csv"
DELTA = 0.01
BOOTSTRAP_REPETITIONS = 100_000
BOOTSTRAP_SEED = 20_260_805
TOLERANCE = 1e-12
KEYS = ["domain", "cell_id", "target_cycle_number", "truth"]

ARMS = (
    "raw_pooled",
    "raw_domain_equal",
    "change_pooled",
    "change_domain_equal",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_raw_arms() -> pd.DataFrame:
    previous = pd.read_parquet(V315)
    raw_pooled = previous.loc[
        previous["method"] == "raw_early_charge", KEYS + ["prediction"]
    ].copy()
    raw_pooled["method"] = "raw_pooled"
    current = pd.read_parquet(V322)
    change_equal = current[KEYS + ["raw_candidate"]].rename(
        columns={"raw_candidate": "prediction"}
    )
    change_equal["method"] = "change_domain_equal"
    return pd.concat([raw_pooled, change_equal], ignore_index=True)


def fit_missing_raw_arms() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    enriched = add_context_change(load_data())
    evaluation = enriched.loc[enriched["after_initial5_reference_window"]].copy()
    domains = sorted(evaluation["domain"].astype(str).unique())
    blocks: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    specifications = {
        "raw_domain_equal": (RAW_FEATURES, True),
        "change_pooled": (RAW_ABSOLUTE_CHANGE_FEATURES, False),
    }
    for fold_index, target_domain in enumerate(domains, start=1):
        source = evaluation.loc[evaluation["domain"] != target_domain].copy()
        target = evaluation.loc[evaluation["domain"] == target_domain].copy()
        source_fit = balanced_source_rows(source)
        for method, (features, domain_equal) in specifications.items():
            prediction = fit_predict(
                source_fit,
                target,
                features,
                domain_equal=domain_equal,
            )
            block = target[["domain", "cell_id", "target_cycle_number", TARGET]].copy()
            block = block.rename(columns={TARGET: "truth"})
            block["method"] = method
            block["prediction"] = prediction
            blocks.append(block)
        folds.append(
            {
                "target_domain": target_domain,
                "source_fit_rows": int(len(source_fit)),
                "target_rows": int(len(target)),
                "target_cells": int(target["cell_id"].nunique()),
            }
        )
        print(f"completed exact factorial fold {fold_index}/{len(domains)}: {target_domain}", flush=True)
    return pd.concat(blocks, ignore_index=True), folds


def validate_rosters(predictions: pd.DataFrame) -> None:
    reference: pd.DataFrame | None = None
    for method, rows in predictions.groupby("method", sort=True):
        roster = rows[KEYS].sort_values(KEYS).reset_index(drop=True)
        if roster.duplicated(KEYS[:-1]).any():
            raise RuntimeError(f"duplicate scoring keys in {method}")
        if reference is None:
            reference = roster
        elif not roster.equals(reference):
            raise RuntimeError(f"scoring roster mismatch in {method}")


def score(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blocks: list[pd.DataFrame] = []
    for method, rows in predictions.groupby("method", sort=False):
        frame = rows[KEYS].rename(columns={"truth": TARGET})
        blocks.append(cell_metrics(frame, rows["prediction"].to_numpy(float), str(method)))
    cells = pd.concat(blocks, ignore_index=True)
    domains = domain_summary(cells)
    summary = (
        domains.groupby("method", as_index=False)
        .agg(
            domain_equal_cell_macro_mae=("cell_macro_mae", "mean"),
            worst_domain_cell_macro_mae=("cell_macro_mae", "max"),
        )
        .sort_values("domain_equal_cell_macro_mae")
    )
    return cells, domains, summary


def paired_effect(domains: pd.DataFrame, method: str, reference: str) -> dict[str, object]:
    wide = domains.pivot(index="domain", columns="method", values="cell_macro_mae")
    difference = (wide[method] - wide[reference]).sort_index()
    values = difference.to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPETITIONS, len(values)))
    bootstrap = values[draws].mean(axis=1)
    signs = np.asarray(list(product((-1.0, 1.0), repeat=len(values))), dtype=float)
    sign_flip = (signs * values[None, :]).mean(axis=1)
    observed = float(values.mean())
    return {
        "method": method,
        "reference": reference,
        "difference_direction": "negative favors method",
        "mean_domain_difference": observed,
        "median_domain_difference": float(np.median(values)),
        "ci95_domain_bootstrap": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "exact_two_sided_sign_flip_p": float(np.mean(np.abs(sign_flip) >= abs(observed) - 1e-15)),
        "domain_wins_ties_losses": [
            int(np.sum(values < -TOLERANCE)),
            int(np.sum(np.abs(values) <= TOLERANCE)),
            int(np.sum(values > TOLERANCE)),
        ],
        "per_domain_difference": difference.to_dict(),
    }


def project_all_arms(candidates: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    wide = candidates.pivot(index=KEYS, columns="method", values="prediction").reset_index()
    alpha_rows = pd.read_csv(ALPHAS)
    alpha_map = dict(
        zip(
            alpha_rows["outer_target_domain"].astype(str),
            alpha_rows["selected_alpha"].astype(float),
        )
    )
    blocks: list[pd.DataFrame] = []
    for domain, rows in wide.groupby("domain", sort=True):
        alpha = alpha_map[str(domain)]
        ordered = rows.sort_values(["cell_id", "target_cycle_number"]).copy()
        protected_state: np.ndarray | None = None
        for arm in ARMS:
            baseline, method = prefix_causal_cellwise_projection(
                ordered["cell_id"].astype(str).to_numpy(),
                ordered["target_cycle_number"].to_numpy(float),
                ordered["raw_pooled"].to_numpy(float),
                ordered[arm].to_numpy(float),
                DELTA,
                assimilation=float(alpha),
            )
            if protected_state is None:
                protected_state = baseline
            elif not np.allclose(protected_state, baseline, atol=0.0, rtol=0.0):
                raise RuntimeError("protected state changed across candidate arms")
            block = ordered[KEYS].copy()
            block["method"] = f"pchp_{arm}"
            block["prediction"] = method
            blocks.append(block)
    projected = pd.concat(blocks, ignore_index=True)
    reference = pd.read_parquet(V327)[KEYS + ["selected_causal_method"]]
    selected = projected.loc[projected["method"] == "pchp_change_domain_equal"].merge(
        reference,
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    reproduction_error = float(
        np.max(np.abs(selected["prediction"] - selected["selected_causal_method"]))
    )
    return projected, reproduction_error


def factorial_effects(domains: pd.DataFrame, prefix: str) -> dict[str, object]:
    names = {arm: f"{prefix}{arm}" for arm in ARMS}
    effects = {
        "feature_effect_under_pooled_weighting": paired_effect(
            domains, names["change_pooled"], names["raw_pooled"]
        ),
        "feature_effect_under_domain_equal_weighting": paired_effect(
            domains, names["change_domain_equal"], names["raw_domain_equal"]
        ),
        "weighting_effect_with_raw_features": paired_effect(
            domains, names["raw_domain_equal"], names["raw_pooled"]
        ),
        "weighting_effect_with_change_features": paired_effect(
            domains, names["change_domain_equal"], names["change_pooled"]
        ),
    }
    wide = domains.pivot(index="domain", columns="method", values="cell_macro_mae")
    interaction = (
        wide[names["change_domain_equal"]]
        - wide[names["change_pooled"]]
        - wide[names["raw_domain_equal"]]
        + wide[names["raw_pooled"]]
    )
    values = interaction.to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPETITIONS, len(values)))
    effects["feature_by_weighting_interaction"] = {
        "mean_domain_difference": float(values.mean()),
        "ci95_domain_bootstrap": np.quantile(values[draws].mean(axis=1), [0.025, 0.975]).tolist(),
        "per_domain_difference": interaction.to_dict(),
    }
    return effects


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidate_predictions": OUT / "candidate_predictions_v383.parquet",
        "projected_predictions": OUT / "projected_predictions_v383.parquet",
        "candidate_cell_metrics": OUT / "candidate_cell_metrics_v383.csv",
        "candidate_domain_metrics": OUT / "candidate_domain_metrics_v383.csv",
        "candidate_summary": OUT / "candidate_summary_v383.csv",
        "projected_cell_metrics": OUT / "projected_cell_metrics_v383.csv",
        "projected_domain_metrics": OUT / "projected_domain_metrics_v383.csv",
        "projected_summary": OUT / "projected_summary_v383.csv",
        "report": OUT / "feature_weighting_pchp_factorial_v383_report.json",
    }
    if paths["candidate_predictions"].exists():
        candidates = pd.read_parquet(paths["candidate_predictions"])
        folds: list[dict[str, object]] = []
        print("reused cached exact four-arm candidate predictions", flush=True)
    else:
        missing, folds = fit_missing_raw_arms()
        candidates = pd.concat([frozen_raw_arms(), missing], ignore_index=True)
    validate_rosters(candidates)
    candidate_cells, candidate_domains, candidate_summary = score(candidates)
    projected, reproduction_error = project_all_arms(candidates)
    validate_rosters(projected)
    projected_cells, projected_domains, projected_summary = score(projected)
    if reproduction_error > TOLERANCE:
        raise RuntimeError(f"current PCHP reproduction error: {reproduction_error}")

    candidates.to_parquet(paths["candidate_predictions"], index=False)
    projected.to_parquet(paths["projected_predictions"], index=False)
    candidate_cells.to_csv(paths["candidate_cell_metrics"], index=False)
    candidate_domains.to_csv(paths["candidate_domain_metrics"], index=False)
    candidate_summary.to_csv(paths["candidate_summary"], index=False)
    projected_cells.to_csv(paths["projected_cell_metrics"], index=False)
    projected_domains.to_csv(paths["projected_domain_metrics"], index=False)
    projected_summary.to_csv(paths["projected_summary"], index=False)

    report = {
        "status": "EXACT_CURRENT_PCHP_FEATURE_WEIGHTING_FACTORIAL_COMPLETED",
        "scope": "post-freeze development-only strict LODO attribution audit",
        "independent_statistical_unit": "complete held-out dataset domain",
        "number_of_domains": int(candidate_domains["domain"].nunique()),
        "new_raw_arms_fitted": ["raw_domain_equal", "change_pooled"],
        "frozen_raw_arms_reused": ["raw_pooled", "change_domain_equal"],
        "pchp_budget_soh_units": DELTA,
        "v327_selected_method_maximum_reproduction_error": reproduction_error,
        "candidate_summary": json.loads(candidate_summary.to_json(orient="records")),
        "projected_summary": json.loads(projected_summary.to_json(orient="records")),
        "candidate_factorial_effects": factorial_effects(candidate_domains, prefix=""),
        "projected_factorial_effects": factorial_effects(projected_domains, prefix="pchp_"),
        "folds": folds,
        "interpretation_contract": [
            "This is a post-freeze development-domain attribution audit, not external confirmation.",
            "Feature and weighting effects are paired over identical complete held-out domains.",
            "Every PCHP arm uses the raw/pooled model as the protected predictor and the frozen V326 alpha for its outer domain.",
            "The selected change/domain-equal PCHP arm must reproduce the authoritative V327 prediction ledger exactly within 1e-12.",
        ],
        "runtime_seconds": float(time.perf_counter() - started),
        "files": {},
    }
    for name, path in paths.items():
        if name != "report":
            report["files"][name] = {"path": str(path), "sha256": sha256_file(path)}
    paths["report"].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
