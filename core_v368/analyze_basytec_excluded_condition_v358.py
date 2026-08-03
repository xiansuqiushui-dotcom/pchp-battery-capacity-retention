"""Post-hoc sensitivity that includes the frozen low-reference exclusions."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from basytec_external_v352_common import capacity_map


ROOT = Path(__file__).resolve().parent
EXTERNAL = ROOT / "external_basytec_v352"
PREDICTION_REPORT = (
    EXTERNAL / "label_blind_v353" / "basytec_label_blind_prediction_report_v353.json"
)
FROZEN_SCORED = EXTERNAL / "scored_v354" / "basytec_scored_records_v354.parquet"
CONDITIONS = EXTERNAL / "condition_map_v356.json"
DOWNLOADS = ROOT / "external_basytec_v343" / "downloads"
OUT = EXTERNAL / "scored_v354" / "basytec_excluded_condition_sensitivity_v358.json"

EXCLUDED = ["F0009", "F0048"]
ANCHOR = 5
REPLICATES = 100_000
SEED = 20260810
TOLERANCE = 1e-12
BUDGET = 0.01


def bootstrap(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(values), size=(REPLICATES, len(values)))
    return [
        float(value)
        for value in np.percentile(values[indices].mean(axis=1), [2.5, 97.5])
    ]


def sign_p(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    tail = min(wins, losses)
    return float(
        min(1.0, 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n))
    )


def comparison(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    wins = int((values < -TOLERANCE).sum())
    ties = int((np.abs(values) <= TOLERANCE).sum())
    losses = int((values > TOLERANCE).sum())
    return {
        "mean": float(values.mean()),
        "bootstrap_ci95": bootstrap(values),
        "wins_ties_losses": [wins, ties, losses],
        "exact_two_sided_sign_p": sign_p(wins, losses),
    }


def score_excluded(features: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for cell_id in EXCLUDED:
        mapping, _ = capacity_map(DOWNLOADS / f"{cell_id}.zip")
        cell_features = features.loc[features["cell_id"] == cell_id].copy()
        cell_features["discharge_capacity_ah"] = cell_features["cycle_key"].map(mapping)
        cell_features = cell_features.sort_values("aligned_cycle_rank", kind="mergesort")
        anchor = cell_features.loc[cell_features["aligned_cycle_rank"] <= ANCHOR]
        reference = float(anchor["discharge_capacity_ah"].median())
        labels = cell_features.loc[
            (cell_features["aligned_cycle_rank"] > ANCHOR)
            & cell_features["discharge_capacity_ah"].notna(),
            ["cell_id", "cycle_key", "discharge_capacity_ah"],
        ].copy()
        labels["truth"] = labels["discharge_capacity_ah"] / reference
        scored = predictions.loc[predictions["cell_id"] == cell_id].merge(
            labels,
            on=["cell_id", "cycle_key"],
            how="inner",
            validate="one_to_one",
        )
        truth = scored["truth"].to_numpy(float)
        baseline = scored["causal_baseline_prediction"].to_numpy(float)
        pchp = scored["prefix_causal_harm_capped_prediction"].to_numpy(float)
        comparator = scored["source_tuned_causal_candidate_prediction"].to_numpy(float)
        scored["baseline_abs_error"] = np.abs(baseline - truth)
        scored["pchp_abs_error"] = np.abs(pchp - truth)
        scored["source_tuned_candidate_abs_error"] = np.abs(comparator - truth)
        scored["pchp_loss_regret"] = (
            scored["pchp_abs_error"] - scored["baseline_abs_error"]
        )
        scored["source_tuned_candidate_loss_regret"] = (
            scored["source_tuned_candidate_abs_error"]
            - scored["baseline_abs_error"]
        )
        scored["pchp_displacement"] = np.abs(pchp - baseline)
        scored["source_tuned_candidate_displacement"] = np.abs(
            comparator - baseline
        )
        scored["reference_capacity_ah"] = reference
        outputs.append(scored)
    return pd.concat(outputs, ignore_index=True)


def main() -> int:
    report = json.loads(PREDICTION_REPORT.read_text(encoding="utf-8"))
    features = pd.read_parquet(report["frozen_outputs"]["features"]["path"])
    predictions = pd.read_parquet(report["frozen_outputs"]["predictions"]["path"])
    frozen = pd.read_parquet(FROZEN_SCORED)
    excluded = score_excluded(features, predictions)
    combined = pd.concat([frozen, excluded[frozen.columns]], ignore_index=True)

    cell = (
        combined.groupby("cell_id", sort=True)
        .agg(
            records=("truth", "size"),
            baseline_mae=("baseline_abs_error", "mean"),
            pchp_mae=("pchp_abs_error", "mean"),
            comparator_mae=("source_tuned_candidate_abs_error", "mean"),
            maximum_pchp_loss_regret=("pchp_loss_regret", "max"),
            maximum_pchp_displacement=("pchp_displacement", "max"),
        )
        .reset_index()
    )
    cell["pchp_minus_baseline"] = cell["pchp_mae"] - cell["baseline_mae"]
    cell["comparator_minus_baseline"] = cell["comparator_mae"] - cell["baseline_mae"]
    cell["pchp_minus_comparator"] = cell["pchp_mae"] - cell["comparator_mae"]

    metadata = pd.DataFrame(
        json.loads(CONDITIONS.read_text(encoding="utf-8"))["rows"]
    ).rename(columns={"cell_name": "cell_id"})
    cell = cell.merge(
        metadata[["cell_id", "temperature_c", "c_rate"]],
        on="cell_id",
        validate="one_to_one",
    )
    cell["condition"] = (
        cell["temperature_c"].astype(str)
        + "C_"
        + cell["c_rate"].astype(str)
        + "Crate"
    )
    condition = (
        cell.groupby("condition", sort=True)
        .agg(
            cells=("cell_id", "size"),
            pchp_minus_baseline=("pchp_minus_baseline", "mean"),
            comparator_minus_baseline=("comparator_minus_baseline", "mean"),
            pchp_minus_comparator=("pchp_minus_comparator", "mean"),
        )
        .reset_index()
    )

    excluded_cell = cell.loc[cell["cell_id"].isin(EXCLUDED)].copy()
    payload = {
        "status": "POSTHOC_FULL_47_CELL_SENSITIVITY_COMPLETE",
        "warning": "Outcome-informed sensitivity; does not replace the frozen 45-cell confirmatory analysis.",
        "included_previously_excluded_condition": "0 C, 1.5C",
        "excluded_cell_details": [
            {
                "cell_id": row.cell_id,
                "reference_capacity_ah": float(
                    excluded.loc[excluded["cell_id"] == row.cell_id, "reference_capacity_ah"].iloc[0]
                ),
                "records": int(row.records),
                "pchp_minus_baseline": float(row.pchp_minus_baseline),
                "comparator_minus_baseline": float(row.comparator_minus_baseline),
            }
            for row in excluded_cell.itertuples()
        ],
        "full_47_cell_analysis": {
            "physical_cells": int(len(cell)),
            "conditions": int(len(condition)),
            "scored_records": int(len(combined)),
            "pchp_minus_baseline": comparison(cell["pchp_minus_baseline"].to_numpy(float)),
            "comparator_minus_baseline": comparison(
                cell["comparator_minus_baseline"].to_numpy(float)
            ),
            "pchp_minus_comparator": comparison(
                cell["pchp_minus_comparator"].to_numpy(float)
            ),
            "maximum_pchp_displacement": float(
                combined["pchp_displacement"].max()
            ),
            "maximum_observed_pchp_loss_regret": float(
                combined["pchp_loss_regret"].max()
            ),
            "maximum_cell_macro_pchp_harm": float(
                cell["pchp_minus_baseline"].max()
            ),
        },
        "posthoc_24_condition_equal_analysis": {
            "conditions": int(len(condition)),
            "cells_per_condition": {
                str(int(key)): int(value)
                for key, value in condition["cells"].value_counts().sort_index().items()
            },
            "pchp_minus_baseline": comparison(
                condition["pchp_minus_baseline"].to_numpy(float)
            ),
            "comparator_minus_baseline": comparison(
                condition["comparator_minus_baseline"].to_numpy(float)
            ),
            "pchp_minus_comparator": comparison(
                condition["pchp_minus_comparator"].to_numpy(float)
            ),
        },
    }
    OUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
