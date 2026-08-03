"""Independent statistical and reviewer-risk audit of the V354 confirmation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SCORED = (
    ROOT
    / "external_basytec_v352"
    / "scored_v354"
    / "basytec_scored_records_v354.parquet"
)
CELL = (
    ROOT
    / "external_basytec_v352"
    / "scored_v354"
    / "basytec_cell_metrics_v354.csv"
)
REPORT = (
    ROOT
    / "external_basytec_v352"
    / "scored_v354"
    / "basytec_frozen_confirmation_v354_report.json"
)
CONDITIONS = ROOT / "external_basytec_v352" / "condition_map_v356.json"
ROSTER = ROOT / "external_basytec_v347" / "confirmation_roster_v347.json"
OUT = (
    ROOT
    / "external_basytec_v352"
    / "scored_v354"
    / "basytec_external_statistics_audit_v357.json"
)

TOLERANCE = 1e-12
BUDGET = 0.01
REPLICATES = 100_000
CELL_SEED = 20260808
CONDITION_SEED = 20260809


def percentile_bootstrap(values: np.ndarray, seed: int) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(REPLICATES, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.percentile(means, [2.5, 97.5])]


def exact_two_sided_sign_p(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    tail = min(wins, losses)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return float(min(1.0, 2.0 * probability))


def comparison(values: np.ndarray, seed: int) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    wins = int((values < -TOLERANCE).sum())
    ties = int((np.abs(values) <= TOLERANCE).sum())
    losses = int((values > TOLERANCE).sum())
    return {
        "mean": float(values.mean()),
        "bootstrap_ci95": percentile_bootstrap(values, seed),
        "wins_ties_losses": [wins, ties, losses],
        "exact_two_sided_sign_p": exact_two_sided_sign_p(wins, losses),
        "leave_one_unit_out_mean_range": [
            float(((values.sum() - values) / (len(values) - 1)).min()),
            float(((values.sum() - values) / (len(values) - 1)).max()),
        ],
    }


def main() -> int:
    scored = pd.read_parquet(SCORED)
    cell = pd.read_csv(CELL)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    condition_rows = json.loads(CONDITIONS.read_text(encoding="utf-8"))["rows"]
    conditions = pd.DataFrame(condition_rows).rename(columns={"cell_name": "cell_id"})

    reconstructed = (
        scored.groupby("cell_id", sort=True)
        .agg(
            records=("truth", "size"),
            baseline_mae=("baseline_abs_error", "mean"),
            pchp_mae=("pchp_abs_error", "mean"),
            source_tuned_causal_candidate_mae=(
                "source_tuned_candidate_abs_error",
                "mean",
            ),
            maximum_pchp_loss_regret=("pchp_loss_regret", "max"),
            maximum_pchp_displacement=("pchp_displacement", "max"),
            maximum_source_tuned_candidate_displacement=(
                "source_tuned_candidate_displacement",
                "max",
            ),
        )
        .reset_index()
    )
    check_columns = [
        "records",
        "baseline_mae",
        "pchp_mae",
        "source_tuned_causal_candidate_mae",
        "maximum_pchp_loss_regret",
        "maximum_pchp_displacement",
        "maximum_source_tuned_candidate_displacement",
    ]
    merged = cell.merge(
        reconstructed,
        on="cell_id",
        suffixes=("_reported", "_reconstructed"),
        validate="one_to_one",
    )
    reconstruction_errors = {}
    for column in check_columns:
        left = merged[f"{column}_reported"].to_numpy(float)
        right = merged[f"{column}_reconstructed"].to_numpy(float)
        reconstruction_errors[column] = float(np.max(np.abs(left - right)))

    cell_diff = cell["pchp_minus_baseline"].to_numpy(float)
    cell_primary = comparison(cell_diff, CELL_SEED)
    frozen_primary = report["accuracy"]["comparisons"]["pchp_minus_baseline"]

    cell_condition = cell.merge(
        conditions[["cell_id", "temperature_c", "c_rate"]],
        on="cell_id",
        how="left",
        validate="one_to_one",
    )
    if cell_condition[["temperature_c", "c_rate"]].isna().any().any():
        raise RuntimeError("condition metadata missing for an eligible cell")
    cell_condition["condition"] = (
        cell_condition["temperature_c"].astype(str)
        + "C_"
        + cell_condition["c_rate"].astype(str)
        + "Crate"
    )
    condition = (
        cell_condition.groupby("condition", sort=True)
        .agg(
            cells=("cell_id", "size"),
            pchp_minus_baseline=("pchp_minus_baseline", "mean"),
            source_tuned_candidate_minus_baseline=(
                "source_tuned_candidate_minus_baseline",
                "mean",
            ),
            pchp_minus_source_tuned_candidate=(
                "pchp_minus_source_tuned_candidate",
                "mean",
            ),
        )
        .reset_index()
    )
    condition_sensitivity = {
        name: comparison(condition[name].to_numpy(float), CONDITION_SEED)
        for name in (
            "pchp_minus_baseline",
            "source_tuned_candidate_minus_baseline",
            "pchp_minus_source_tuned_candidate",
        )
    }

    baseline = scored["causal_baseline_prediction"].to_numpy(float)
    pchp = scored["prefix_causal_harm_capped_prediction"].to_numpy(float)
    comparator = scored["source_tuned_causal_candidate_prediction"].to_numpy(float)
    truth = scored["truth"].to_numpy(float)
    safe_up = np.clip(baseline + BUDGET, 0.0, 1.3)
    scored_shift = scored[["cell_id"]].copy()
    scored_shift["safe_up_abs_error"] = np.abs(safe_up - truth)
    shift_cell = (
        scored_shift.groupby("cell_id", sort=True)["safe_up_abs_error"].mean()
    )
    baseline_cell = cell.set_index("cell_id")["baseline_mae"].sort_index()
    pchp_cell = cell.set_index("cell_id")["pchp_mae"].sort_index()
    shift_cell = shift_cell.sort_index()

    roster = json.loads(ROSTER.read_text(encoding="utf-8"))["downloaded_cell_zips"]
    roster_cells = {Path(item["key"]).stem for item in roster}
    eligible_cells = set(cell["cell_id"].astype(str))
    excluded = sorted(roster_cells.difference(eligible_cells))

    payload = {
        "status": "BASYTEC_V354_INDEPENDENT_STATISTICAL_AUDIT_COMPLETE",
        "frozen_decision": report["decision"],
        "reconstruction": {
            "scored_records": int(len(scored)),
            "eligible_cells": int(len(cell)),
            "maximum_absolute_errors": reconstruction_errors,
            "all_reconstructed_within_1e-12": bool(
                max(reconstruction_errors.values()) <= TOLERANCE
            ),
        },
        "exclusions": {
            "confirmation_roster_cells": len(roster_cells),
            "eligible_cells": len(eligible_cells),
            "excluded_cells": excluded,
            "frozen_reason": "reference capacity outside 0.08-0.16 Ah",
        },
        "confirmatory_cell_level": {
            "independent_unit": "physical cell",
            "comparison": cell_primary,
            "reported_mean_difference": frozen_primary["cell_equal_mean"],
            "reported_bootstrap_ci95": frozen_primary["cell_bootstrap_ci95"],
            "mean_reproduction_error": float(
                abs(cell_primary["mean"] - frozen_primary["cell_equal_mean"])
            ),
            "ci_reproduction_max_error": float(
                np.max(
                    np.abs(
                        np.asarray(cell_primary["bootstrap_ci95"])
                        - np.asarray(frozen_primary["cell_bootstrap_ci95"])
                    )
                )
            ),
        },
        "posthoc_condition_cluster_sensitivity": {
            "status": "POSTHOC_NOT_A_REPLACEMENT_FOR_THE_FROZEN_PRIMARY_ANALYSIS",
            "conditions": int(len(condition)),
            "condition_cell_counts": {
                str(int(key)): int(value)
                for key, value in condition["cells"].value_counts().sort_index().items()
            },
            "bootstrap_replicates": REPLICATES,
            "bootstrap_seed": CONDITION_SEED,
            "comparisons": condition_sensitivity,
        },
        "reviewer_risk_diagnostics": {
            "fraction_baseline_predictions_below_truth": float((baseline < truth).mean()),
            "fraction_pchp_moves_upward": float((pchp > baseline + TOLERANCE).mean()),
            "fraction_pchp_at_full_positive_budget": float(
                np.isclose(pchp, baseline + BUDGET, rtol=0.0, atol=TOLERANCE).mean()
            ),
            "max_abs_pchp_minus_constant_safe_up": float(
                np.max(np.abs(pchp - safe_up))
            ),
            "cell_equal_constant_safe_up_mae": float(shift_cell.mean()),
            "cell_equal_constant_safe_up_gain_over_baseline": float(
                (baseline_cell - shift_cell).mean()
            ),
            "cell_equal_pchp_minus_constant_safe_up": float(
                (pchp_cell - shift_cell).mean()
            ),
            "source_tuned_candidate_fraction_below_truth": float(
                (comparator < truth).mean()
            ),
            "interpretation": (
                "The external domain tests the frozen safety-and-utility contract, "
                "but it may not distinguish adaptive PCHP behavior from a constant "
                "upward budget shift when baseline underprediction is nearly universal."
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
