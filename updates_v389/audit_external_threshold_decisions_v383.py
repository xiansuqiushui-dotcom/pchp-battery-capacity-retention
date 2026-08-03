"""Observed-horizon threshold-decision audit for the six V380 datasets.

The audit complements continuous asymmetric cost with cell-level operational
quantities: first-crossing timing, late warning, premature review, and on-time
review.  It uses the already frozen V380 predictions and retains all tested
thresholds.  Complete datasets are the top-level transport units; cells are
nested replicates.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "external_mechanism_decision_v380" / "external_mechanism_predictions_v380.parquet"
OUT = ROOT / "external_threshold_decisions_v383"
THRESHOLDS = (0.70, 0.80, 0.90)
METHODS = ("protected_state", "fixed_shift", "pchp_method")
TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_true(array: np.ndarray) -> int | None:
    positions = np.flatnonzero(array)
    return int(positions[0]) if len(positions) else None


def cell_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (domain, cell_id), group in frame.groupby(["domain", "cell_id"], sort=True):
        ordered = group.sort_values("target_cycle_number").reset_index(drop=True)
        truth = ordered["truth"].to_numpy(float)
        cycles = ordered["target_cycle_number"].to_numpy(float)
        for threshold in THRESHOLDS:
            truth_degraded = truth <= threshold
            truth_cross = first_true(truth_degraded)
            for method in METHODS:
                prediction = ordered[method].to_numpy(float)
                predicted_review = prediction <= threshold
                prediction_cross = first_true(predicted_review)
                eligible = truth_cross is not None
                restricted_prediction_cross = (
                    prediction_cross if prediction_cross is not None else len(ordered)
                )
                if eligible:
                    signed_delay = int(restricted_prediction_cross - truth_cross)
                    absolute_delay = abs(signed_delay)
                    late = int(signed_delay > 0)
                    premature = int(signed_delay < 0)
                    on_time = int(signed_delay == 0)
                    both_observed = prediction_cross is not None
                    cycle_delay = (
                        float(cycles[prediction_cross] - cycles[truth_cross])
                        if both_observed
                        else np.nan
                    )
                else:
                    signed_delay = np.nan
                    absolute_delay = np.nan
                    late = np.nan
                    premature = int(prediction_cross is not None)
                    on_time = np.nan
                    both_observed = False
                    cycle_delay = np.nan

                late_record_count = int(np.sum(truth_degraded & ~predicted_review))
                degraded_record_count = int(np.sum(truth_degraded))
                premature_record_count = int(np.sum(~truth_degraded & predicted_review))
                healthy_record_count = int(np.sum(~truth_degraded))
                rows.append(
                    {
                        "domain": domain,
                        "cell_id": cell_id,
                        "threshold": threshold,
                        "method": method,
                        "records": int(len(ordered)),
                        "truth_crossing_observed": bool(eligible),
                        "prediction_crossing_observed": bool(prediction_cross is not None),
                        "both_crossings_observed": bool(eligible and both_observed),
                        "truth_first_crossing_record": (
                            int(truth_cross + 1) if truth_cross is not None else np.nan
                        ),
                        "prediction_first_crossing_record": (
                            int(prediction_cross + 1) if prediction_cross is not None else np.nan
                        ),
                        "restricted_signed_delay_records": signed_delay,
                        "restricted_absolute_delay_records": absolute_delay,
                        "observed_cycle_delay": cycle_delay,
                        "late_warning": late,
                        "premature_review": premature,
                        "on_time_review": on_time,
                        "degraded_records": degraded_record_count,
                        "late_warning_records": late_record_count,
                        "cell_late_warning_record_rate": (
                            late_record_count / degraded_record_count
                            if degraded_record_count
                            else np.nan
                        ),
                        "healthy_records": healthy_record_count,
                        "premature_review_records": premature_record_count,
                        "cell_premature_review_record_rate": (
                            premature_record_count / healthy_record_count
                            if healthy_record_count
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (domain, threshold, method), group in cells.groupby(
        ["domain", "threshold", "method"], sort=True
    ):
        crossing = group.loc[group["truth_crossing_observed"]].copy()
        both = group.loc[group["both_crossings_observed"]].copy()
        late_record_rates = group["cell_late_warning_record_rate"].dropna()
        premature_record_rates = group["cell_premature_review_record_rate"].dropna()
        rows.append(
            {
                "domain": domain,
                "threshold": threshold,
                "method": method,
                "cells": int(group["cell_id"].nunique()),
                "truth_crossing_cells": int(len(crossing)),
                "prediction_crossing_cells": int(group["prediction_crossing_observed"].sum()),
                "mean_restricted_signed_delay_records": float(crossing["restricted_signed_delay_records"].mean()) if len(crossing) else np.nan,
                "mean_restricted_absolute_delay_records": float(crossing["restricted_absolute_delay_records"].mean()) if len(crossing) else np.nan,
                "p95_restricted_late_delay_records": float(np.quantile(np.maximum(crossing["restricted_signed_delay_records"].to_numpy(float), 0.0), 0.95)) if len(crossing) else np.nan,
                "mean_observed_absolute_cycle_delay": float(np.abs(both["observed_cycle_delay"]).mean()) if len(both) else np.nan,
                "late_warning_cell_rate": float(crossing["late_warning"].mean()) if len(crossing) else np.nan,
                "premature_review_cell_rate_among_crossers": float(crossing["premature_review"].mean()) if len(crossing) else np.nan,
                "on_time_review_cell_rate": float(crossing["on_time_review"].mean()) if len(crossing) else np.nan,
                "premature_review_cell_rate_without_observed_truth_crossing": float(group.loc[~group["truth_crossing_observed"], "premature_review"].mean()) if (~group["truth_crossing_observed"]).any() else np.nan,
                "cell_macro_late_warning_record_rate": float(late_record_rates.mean()) if len(late_record_rates) else np.nan,
                "cell_macro_premature_review_record_rate": float(premature_record_rates.mean()) if len(premature_record_rates) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def dataset_equal(summary: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in summary.columns
        if column
        not in {"domain", "threshold", "method", "cells", "truth_crossing_cells", "prediction_crossing_cells"}
    ]
    return (
        summary.groupby(["threshold", "method"], as_index=False)[metric_columns]
        .mean(numeric_only=True)
    )


def comparator_differences(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_restricted_absolute_delay_records",
        "p95_restricted_late_delay_records",
        "late_warning_cell_rate",
        "premature_review_cell_rate_among_crossers",
        "cell_macro_late_warning_record_rate",
        "cell_macro_premature_review_record_rate",
    ]
    rows: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        subset = summary.loc[summary["threshold"] == threshold]
        for reference in ("protected_state", "fixed_shift"):
            for metric in metrics:
                wide = subset.pivot(index="domain", columns="method", values=metric)
                difference = (wide["pchp_method"] - wide[reference]).dropna()
                rows.append(
                    {
                        "threshold": threshold,
                        "reference": reference,
                        "metric": metric,
                        "datasets_contributing": int(len(difference)),
                        "dataset_equal_mean_difference": float(difference.mean()) if len(difference) else np.nan,
                        "median_dataset_difference": float(difference.median()) if len(difference) else np.nan,
                        "dataset_wins": int(np.sum(difference < -TOLERANCE)),
                        "dataset_ties": int(np.sum(np.abs(difference) <= TOLERANCE)),
                        "dataset_losses": int(np.sum(difference > TOLERANCE)),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_parquet(INPUT)
    cells = cell_rows(predictions)
    summary = summarize(cells)
    overall = dataset_equal(summary)
    differences = comparator_differences(summary)
    paths = {
        "cell_metrics": OUT / "external_threshold_cell_metrics_v383.csv",
        "dataset_summary": OUT / "external_threshold_dataset_summary_v383.csv",
        "dataset_equal_summary": OUT / "external_threshold_dataset_equal_summary_v383.csv",
        "comparisons": OUT / "external_threshold_comparisons_v383.csv",
        "report": OUT / "external_threshold_decisions_v383_report.json",
    }
    cells.to_csv(paths["cell_metrics"], index=False)
    summary.to_csv(paths["dataset_summary"], index=False)
    overall.to_csv(paths["dataset_equal_summary"], index=False)
    differences.to_csv(paths["comparisons"], index=False)
    primary = overall.loc[np.isclose(overall["threshold"], 0.80)].copy()
    report = {
        "status": "EXTERNAL_THRESHOLD_DECISION_AUDIT_COMPLETED",
        "scope": "six-dataset frozen-prediction observed-horizon decision replay",
        "thresholds": list(THRESHOLDS),
        "methods": list(METHODS),
        "definitions": {
            "late_warning": "predicted first crossing occurs after the observed true first crossing or is absent within follow-up",
            "premature_review": "predicted first crossing occurs before the observed true first crossing; in non-crossing cells, any predicted crossing is premature within follow-up",
            "restricted_delay": "record-index delay with an absent predicted crossing censored to one record beyond the final observation",
            "record_rates": "cell-level false-negative rate among degraded records and false-positive rate among non-degraded records, then averaged by cell and dataset",
        },
        "independent_top_level_unit": "dataset",
        "nested_unit": "physical cell",
        "primary_threshold_0p80": json.loads(primary.to_json(orient="records")),
        "interpretation_contract": [
            "First-crossing metrics describe the observed follow-up window; they do not extrapolate unobserved end-of-life time.",
            "At a threshold where two methods issue identical binary actions, their threshold-event metrics must also be identical even when continuous costs differ.",
            "All thresholds and adverse directions are retained.",
        ],
        "files": {},
    }
    for name, path in paths.items():
        if name != "report":
            report["files"][name] = {"path": str(path), "sha256": sha256_file(path)}
    paths["report"].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
