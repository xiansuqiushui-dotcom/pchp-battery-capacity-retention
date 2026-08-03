"""Commissioning-context and within-cell change decomposition for SOH.

Each early-charge channel is decomposed into a first-five-cycle commissioning
median and a subsequent change.  The model sees the protocol/context anchor,
the dimensional change, and a clipped dimensionless relative change.  Unlike
the V306 IQR normalization, this representation does not divide by tiny
commissioning variability and therefore does not amplify stable-channel noise.

The audit leaves one complete battery dataset domain out.  All twelve domains
were opened previously, so results remain development-only.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from develop_anchor_invariant_soh_v306 import (
    ANCHOR_CYCLES,
    OUT as V306_OUT,
    TARGET,
    balanced_source_rows,
    cell_metrics,
    domain_summary,
    fit_predict,
    load_data,
    monotone_project,
    paired_domain_comparison,
)
from pilot_batterylife_early_charge_lodo_v110 import EARLY_CHARGE_FEATURES


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "context_change_soh_v310"
RELATIVE_DENOMINATOR_FLOOR = 1e-6
RELATIVE_CHANGE_CLIP = 5.0
RAW_FEATURES = ["target_cycle_number", *EARLY_CHARGE_FEATURES]
CHANGE_FEATURES = ["target_cycle_number"] + [
    item
    for feature in EARLY_CHARGE_FEATURES
    for item in (f"change__{feature}", f"relative_change__{feature}")
]
CONTEXT_CHANGE_FEATURES = ["target_cycle_number"] + [
    item
    for feature in EARLY_CHARGE_FEATURES
    for item in (
        f"commissioning_context__{feature}",
        f"change__{feature}",
        f"relative_change__{feature}",
    )
]
RAW_CHANGE_FEATURES = ["target_cycle_number"] + [
    item
    for feature in EARLY_CHARGE_FEATURES
    for item in (feature, f"change__{feature}", f"relative_change__{feature}")
]
METHODS = [
    "raw_early_charge",
    "change_only",
    "context_change",
    "context_change_domain_equal",
    "context_change_domain_equal_monotone",
    "raw_change_domain_equal_monotone",
]
PRIMARY_METHOD = "context_change_domain_equal_monotone"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_context_change(frame: pd.DataFrame) -> pd.DataFrame:
    reference = frame.loc[frame["aligned_cycle_rank"] <= ANCHOR_CYCLES]
    context = (
        reference.groupby("cell_id", sort=False)[list(EARLY_CHARGE_FEATURES)]
        .median()
        .add_prefix("commissioning_context__")
    )
    output = frame.merge(context, on="cell_id", how="left", validate="many_to_one")
    for feature in EARLY_CHARGE_FEATURES:
        anchor = output[f"commissioning_context__{feature}"].to_numpy(float)
        observed = output[feature].to_numpy(float)
        change = observed - anchor
        denominator = np.maximum(np.abs(anchor), RELATIVE_DENOMINATOR_FLOOR)
        relative = change / denominator
        output[f"change__{feature}"] = change
        output[f"relative_change__{feature}"] = np.clip(
            relative, -RELATIVE_CHANGE_CLIP, RELATIVE_CHANGE_CLIP
        )
    return output


def raw_outer_predictions() -> pd.DataFrame:
    predictions = pd.read_parquet(V306_OUT / "outer_lodo_predictions_v306.parquet")
    return predictions.loc[
        predictions["method"] == "raw_early_charge",
        ["domain", "cell_id", "target_cycle_number", "truth", "prediction"],
    ].copy()


def align_raw(target: pd.DataFrame, raw_predictions: pd.DataFrame) -> np.ndarray:
    keys = target[["domain", "cell_id", "target_cycle_number", TARGET]].copy()
    aligned = keys.merge(
        raw_predictions.rename(columns={"truth": TARGET}),
        on=["domain", "cell_id", "target_cycle_number", TARGET],
        how="left",
        validate="one_to_one",
    )
    if aligned["prediction"].isna().any():
        raise RuntimeError("raw prediction alignment failed")
    return aligned["prediction"].to_numpy(float)


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_data()
    enriched = add_context_change(raw)
    evaluation = enriched.loc[enriched["after_initial5_reference_window"]].copy()
    domains = sorted(evaluation["domain"].astype(str).unique())
    raw_predictions = raw_outer_predictions()

    cell_blocks: list[pd.DataFrame] = []
    prediction_blocks: list[pd.DataFrame] = []
    fold_metadata: list[dict[str, object]] = []
    for fold_index, target_domain in enumerate(domains, start=1):
        source = evaluation.loc[evaluation["domain"] != target_domain].copy()
        target = evaluation.loc[evaluation["domain"] == target_domain].copy()
        source_fit = balanced_source_rows(source)
        predictions: dict[str, np.ndarray] = {
            "raw_early_charge": align_raw(target, raw_predictions)
        }
        predictions["change_only"] = fit_predict(
            source_fit, target, CHANGE_FEATURES, domain_equal=False
        )
        predictions["context_change"] = fit_predict(
            source_fit, target, CONTEXT_CHANGE_FEATURES, domain_equal=False
        )
        predictions["context_change_domain_equal"] = fit_predict(
            source_fit, target, CONTEXT_CHANGE_FEATURES, domain_equal=True
        )
        predictions[PRIMARY_METHOD] = monotone_project(
            target, predictions["context_change_domain_equal"]
        )
        raw_change = fit_predict(
            source_fit, target, RAW_CHANGE_FEATURES, domain_equal=True
        )
        predictions["raw_change_domain_equal_monotone"] = monotone_project(
            target, raw_change
        )

        for method in METHODS:
            cell_blocks.append(cell_metrics(target, predictions[method], method))
            prediction_blocks.append(
                pd.DataFrame(
                    {
                        "domain": target["domain"].astype(str).to_numpy(),
                        "cell_id": target["cell_id"].astype(str).to_numpy(),
                        "target_cycle_number": target["target_cycle_number"].to_numpy(float),
                        "truth": target[TARGET].to_numpy(float),
                        "method": method,
                        "prediction": predictions[method],
                    }
                )
            )
        fold_metadata.append(
            {
                "target_domain": target_domain,
                "source_domains": sorted(source["domain"].unique()),
                "source_cells": int(source["cell_id"].nunique()),
                "source_fit_rows": int(len(source_fit)),
                "target_cells": int(target["cell_id"].nunique()),
                "target_rows": int(len(target)),
            }
        )
        print(f"completed outer domain {fold_index}/{len(domains)}: {target_domain}", flush=True)

    cells = pd.concat(cell_blocks, ignore_index=True)
    predictions_frame = pd.concat(prediction_blocks, ignore_index=True)
    domains_frame = domain_summary(cells)
    overall = (
        domains_frame.groupby("method", as_index=False)
        .agg(
            domain_equal_cell_macro_mae=("cell_macro_mae", "mean"),
            domain_equal_cell_macro_rmse=("cell_macro_rmse", "mean"),
            worst_domain_cell_macro_mae=("cell_macro_mae", "max"),
            mean_domain_worst_cell_mae=("worst_cell_mae", "mean"),
            domain_equal_trajectory_increase_fraction=(
                "mean_trajectory_increase_fraction", "mean"
            ),
        )
        .sort_values("domain_equal_cell_macro_mae")
    )
    comparisons = {
        method: paired_domain_comparison(domains_frame, method, "raw_early_charge")
        for method in METHODS
        if method != "raw_early_charge"
    }

    paths = {
        "cell_metrics": OUT / "cell_level_metrics_v310.csv",
        "domain_metrics": OUT / "domain_level_metrics_v310.csv",
        "overall": OUT / "domain_equal_summary_v310.csv",
        "predictions": OUT / "outer_lodo_predictions_v310.parquet",
        "report": OUT / "context_change_soh_v310_report.json",
    }
    cells.to_csv(paths["cell_metrics"], index=False)
    domains_frame.to_csv(paths["domain_metrics"], index=False)
    overall.to_csv(paths["overall"], index=False)
    predictions_frame.to_parquet(paths["predictions"], index=False)

    primary = comparisons[PRIMARY_METHOD]
    status = (
        "CONTEXT_CHANGE_DEVELOPMENT_GATE_PASSED"
        if (
            primary["domain_equal_mean_difference"] < 0.0
            and primary["ci95_domain_cluster_percentile"][1] < 0.0
            and primary["maximum_domain_harm"] <= 0.01
        )
        else "CONTEXT_CHANGE_DEVELOPMENT_GATE_NOT_PASSED"
    )
    report = {
        "status": status,
        "scope": (
            "retrospective development-only leave-one-complete-battery-domain-out "
            "audit on twelve previously opened public domains"
        ),
        "independent_unit_for_cross_domain_claim": "battery dataset domain",
        "nested_units": "physical cells within domain; cycles within physical cell",
        "target_outcome_access_during_training_or_feature_construction": False,
        "decomposition": {
            "commissioning_context": "within-cell median of each channel over the first five cycles",
            "change": "current channel value minus commissioning context",
            "relative_change": (
                "change divided by the absolute commissioning context with a "
                f"fixed {RELATIVE_DENOMINATOR_FLOOR:g} denominator floor and "
                f"clipped to plus/minus {RELATIVE_CHANGE_CLIP:g}"
            ),
        },
        "folds": fold_metadata,
        "overall": overall.where(pd.notna(overall), None).to_dict(orient="records"),
        "comparisons_against_raw_early_charge": comparisons,
        "primary_development_gate": {
            "method": PRIMARY_METHOD,
            "requirements": [
                "negative domain-equal MAE difference",
                "upper endpoint of 95% domain-cluster bootstrap interval below zero",
                "maximum domain harm no greater than 0.01 MAE",
            ],
        },
        "limitations": [
            "All twelve domains were opened before this method was proposed.",
            "Relative changes are scale-normalized but not invariant to arbitrary sensor offsets.",
            "Commissioning context may encode both chemistry/protocol and nuisance instrumentation effects.",
        ],
        "runtime_seconds": float(time.perf_counter() - started),
        "files": {},
    }
    for name, path in paths.items():
        if name == "report":
            continue
        report["files"][name] = {"path": str(path), "sha256": sha256_file(path)}
    paths["report"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
