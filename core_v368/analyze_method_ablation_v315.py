"""Component ablations for monotone regret-capped context--change SOH.

All variants use the same strict leave-one-dataset-domain-out splits.  The
ablation distinguishes absolute change, relative change, source-domain
weighting, monotone baseline processing, and the final regret cap.  These are
development analyses on already opened domains.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from develop_anchor_invariant_soh_v306 import (
    RAW_FEATURES,
    TARGET,
    balanced_source_rows,
    cell_metrics,
    domain_summary,
    fit_predict,
    load_data,
    monotone_project,
    paired_domain_comparison,
)
from develop_context_change_soh_v310 import (
    RAW_CHANGE_FEATURES,
    add_context_change,
)
from evaluate_monotone_regret_capped_soh_v313 import prepare_components
from pilot_batterylife_early_charge_lodo_v110 import EARLY_CHARGE_FEATURES
from regret_capped_projection_v312 import regret_capped_projection


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "method_ablation_v315"
BUDGET = 0.01
RAW_ABSOLUTE_CHANGE_FEATURES = ["target_cycle_number", *EARLY_CHARGE_FEATURES] + [
    f"change__{feature}" for feature in EARLY_CHARGE_FEATURES
]
RAW_RELATIVE_CHANGE_FEATURES = ["target_cycle_number", *EARLY_CHARGE_FEATURES] + [
    f"relative_change__{feature}" for feature in EARLY_CHARGE_FEATURES
]
TRAINED_VARIANTS = {
    "raw_absolute_change_domain_equal_monotone": (
        RAW_ABSOLUTE_CHANGE_FEATURES,
        True,
    ),
    "raw_relative_change_domain_equal_monotone": (
        RAW_RELATIVE_CHANGE_FEATURES,
        True,
    ),
    "raw_change_unweighted_monotone": (RAW_CHANGE_FEATURES, False),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def primary_predictions() -> pd.DataFrame:
    components = prepare_components()
    protected = components["protected_baseline_prediction"].to_numpy(float)
    candidate = components["candidate_prediction"].to_numpy(float)
    capped = regret_capped_projection(protected, candidate, BUDGET)
    keys = components[["domain", "cell_id", "target_cycle_number", "truth"]].copy()
    blocks = []
    for method, prediction in (
        ("raw_early_charge", components["baseline_prediction"].to_numpy(float)),
        ("raw_early_charge_isotonic", protected),
        ("raw_change_domain_equal_monotone", candidate),
        ("monotone_regret_capped_context_change", capped),
    ):
        block = keys.copy()
        block["method"] = method
        block["prediction"] = prediction
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def train_ablation_variants() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    enriched = add_context_change(load_data())
    evaluation = enriched.loc[enriched["after_initial5_reference_window"]].copy()
    domains = sorted(evaluation["domain"].astype(str).unique())
    blocks: list[pd.DataFrame] = []
    fold_metadata: list[dict[str, object]] = []
    for fold_index, target_domain in enumerate(domains, start=1):
        source = evaluation.loc[evaluation["domain"] != target_domain].copy()
        target = evaluation.loc[evaluation["domain"] == target_domain].copy()
        source_fit = balanced_source_rows(source)
        for method, (features, domain_equal) in TRAINED_VARIANTS.items():
            raw_prediction = fit_predict(
                source_fit,
                target,
                features,
                domain_equal=domain_equal,
            )
            prediction = monotone_project(target, raw_prediction)
            blocks.append(
                pd.DataFrame(
                    {
                        "domain": target["domain"].astype(str).to_numpy(),
                        "cell_id": target["cell_id"].astype(str).to_numpy(),
                        "target_cycle_number": target["target_cycle_number"].to_numpy(float),
                        "truth": target[TARGET].to_numpy(float),
                        "method": method,
                        "prediction": prediction,
                    }
                )
            )
        fold_metadata.append(
            {
                "target_domain": target_domain,
                "source_fit_rows": int(len(source_fit)),
                "target_rows": int(len(target)),
                "target_cells": int(target["cell_id"].nunique()),
            }
        )
        print(f"completed ablation fold {fold_index}/{len(domains)}: {target_domain}", flush=True)
    return pd.concat(blocks, ignore_index=True), fold_metadata


def score(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks: list[pd.DataFrame] = []
    for method, rows in predictions.groupby("method", sort=False):
        frame = rows[["domain", "cell_id", "target_cycle_number", "truth"]].rename(
            columns={"truth": TARGET}
        )
        blocks.append(
            cell_metrics(frame, rows["prediction"].to_numpy(float), str(method))
        )
    cells = pd.concat(blocks, ignore_index=True)
    return cells, domain_summary(cells)


def comparison_row(
    domains: pd.DataFrame,
    method: str,
    baseline: str,
) -> dict[str, object]:
    item = paired_domain_comparison(domains, method, baseline)
    return {
        "method": method,
        "baseline": baseline,
        "domain_equal_mean_difference": item["domain_equal_mean_difference"],
        "ci95_lower": item["ci95_domain_cluster_percentile"][0],
        "ci95_upper": item["ci95_domain_cluster_percentile"][1],
        "domain_wins": item["domain_wins_ties_losses"][0],
        "domain_ties": item["domain_wins_ties_losses"][1],
        "domain_losses": item["domain_wins_ties_losses"][2],
        "maximum_domain_harm": item["maximum_domain_harm"],
        "per_domain_difference": item["per_domain_difference"],
    }


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    primary = primary_predictions()
    trained, folds = train_ablation_variants()
    predictions = pd.concat([primary, trained], ignore_index=True)
    cells, domains = score(predictions)
    summary = (
        domains.groupby("method", as_index=False)
        .agg(
            domain_equal_cell_macro_mae=("cell_macro_mae", "mean"),
            worst_domain_cell_macro_mae=("cell_macro_mae", "max"),
            domain_equal_trajectory_increase_fraction=(
                "mean_trajectory_increase_fraction", "mean"
            ),
        )
        .sort_values("domain_equal_cell_macro_mae")
    )

    protected = "raw_early_charge_isotonic"
    versus_protected = [
        comparison_row(domains, method, protected)
        for method in summary["method"]
        if method != protected
    ]
    attribution = [
        comparison_row(
            domains,
            "raw_change_domain_equal_monotone",
            "raw_absolute_change_domain_equal_monotone",
        ),
        comparison_row(
            domains,
            "raw_change_domain_equal_monotone",
            "raw_relative_change_domain_equal_monotone",
        ),
        comparison_row(
            domains,
            "raw_change_domain_equal_monotone",
            "raw_change_unweighted_monotone",
        ),
        comparison_row(
            domains,
            "monotone_regret_capped_context_change",
            "raw_change_domain_equal_monotone",
        ),
        comparison_row(domains, protected, "raw_early_charge"),
    ]

    paths = {
        "summary": OUT / "ablation_summary_v315.csv",
        "domain_metrics": OUT / "domain_metrics_v315.csv",
        "cell_metrics": OUT / "cell_metrics_v315.csv",
        "predictions": OUT / "ablation_predictions_v315.parquet",
        "report": OUT / "method_ablation_v315_report.json",
    }
    summary.to_csv(paths["summary"], index=False)
    domains.to_csv(paths["domain_metrics"], index=False)
    cells.to_csv(paths["cell_metrics"], index=False)
    predictions.to_parquet(paths["predictions"], index=False)
    report = {
        "status": "METHOD_COMPONENT_ABLATION_COMPLETED",
        "scope": "development-only strict leave-one-dataset-domain-out ablation",
        "target_outcome_access_during_training_or_projection": False,
        "budget_soh_units": BUDGET,
        "summary": json.loads(summary.to_json(orient="records")),
        "comparisons_versus_protected_baseline": versus_protected,
        "component_attribution_comparisons": attribution,
        "folds": folds,
        "interpretation_contract": [
            "A lower mean error for the uncapped candidate does not establish deployment safety if its maximum domain harm is large.",
            "The capped method is expected to sacrifice some candidate utility in exchange for the deterministic harm budget.",
            "Ablations on already opened domains are mechanistic development evidence, not independent confirmation.",
        ],
        "runtime_seconds": float(time.perf_counter() - started),
        "files": {},
    }
    for name, path in paths.items():
        if name != "report":
            report["files"][name] = {"path": str(path), "sha256": sha256_file(path)}
    paths["report"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
