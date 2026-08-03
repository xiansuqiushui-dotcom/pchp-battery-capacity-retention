"""Formal development audit of regret-capped context--change updates.

The baseline and candidate are frozen strict-LODO predictions from V310.  This
script applies no fitted calibration or target-outcome-dependent threshold.  A
primary absolute SOH update budget of 0.01 is declared before the audit.  The
full budget grid is reported only as a sensitivity frontier.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from develop_anchor_invariant_soh_v306 import (
    TARGET,
    cell_metrics,
    domain_summary,
    paired_domain_comparison,
)
from develop_context_change_soh_v310 import OUT as V310_OUT
from regret_capped_projection_v312 import (
    absolute_loss_regret,
    regret_capped_projection,
    verify_absolute_loss_budget,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "regret_capped_context_change_v312"
PRIMARY_BUDGET = 0.01
BUDGET_GRID = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03)
BASELINE_METHOD = "raw_early_charge"
CANDIDATE_METHOD = "raw_change_domain_equal_monotone"
PRIMARY_METHOD = "regret_capped_context_change_delta_0p01"
TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_components() -> pd.DataFrame:
    predictions = pd.read_parquet(V310_OUT / "outer_lodo_predictions_v310.parquet")
    keys = ["domain", "cell_id", "target_cycle_number", "truth"]
    baseline = predictions.loc[
        predictions["method"] == BASELINE_METHOD, keys + ["prediction"]
    ].rename(columns={"prediction": "baseline_prediction"})
    candidate = predictions.loc[
        predictions["method"] == CANDIDATE_METHOD, keys + ["prediction"]
    ].rename(columns={"prediction": "candidate_prediction"})
    combined = baseline.merge(candidate, on=keys, how="inner", validate="one_to_one")
    if len(combined) != len(baseline) or len(combined) != len(candidate):
        raise RuntimeError("baseline and candidate predictions do not align")
    return combined


def scoring_frame(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[["domain", "cell_id", "target_cycle_number", "truth"]].rename(
        columns={"truth": TARGET}
    )


def audit_budget(
    components: pd.DataFrame,
    budget: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], np.ndarray, dict[str, object]]:
    truth = components["truth"].to_numpy(float)
    baseline = components["baseline_prediction"].to_numpy(float)
    candidate = components["candidate_prediction"].to_numpy(float)
    projected = regret_capped_projection(baseline, candidate, budget)
    method = f"regret_capped_context_change_delta_{budget:g}"
    cells = cell_metrics(scoring_frame(components), projected, method)
    domains = domain_summary(cells)

    baseline_cells = cell_metrics(
        scoring_frame(components), baseline, BASELINE_METHOD
    )
    baseline_domains = domain_summary(baseline_cells)
    comparison_input = pd.concat([baseline_domains, domains], ignore_index=True)
    comparison = paired_domain_comparison(comparison_input, method, BASELINE_METHOD)

    row_regret = absolute_loss_regret(truth, baseline, projected)
    update = np.abs(projected - baseline)
    paired_cells = cells[["domain", "cell_id", "mae"]].merge(
        baseline_cells[["domain", "cell_id", "mae"]],
        on=["domain", "cell_id"],
        suffixes=("_method", "_baseline"),
        validate="one_to_one",
    )
    cell_regret = paired_cells["mae_method"] - paired_cells["mae_baseline"]
    domain_method = domains.set_index("domain")["cell_macro_mae"]
    domain_baseline = baseline_domains.set_index("domain")["cell_macro_mae"]
    domain_regret = domain_method - domain_baseline
    certificate = {
        "budget": budget,
        "maximum_absolute_prediction_update": float(update.max()),
        "maximum_row_absolute_loss_regret": float(row_regret.max()),
        "maximum_cell_macro_mae_regret": float(cell_regret.max()),
        "maximum_domain_cell_macro_mae_regret": float(domain_regret.max()),
        "casewise_budget_verified": verify_absolute_loss_budget(
            truth, baseline, candidate, budget
        ),
        "row_bound_verified": bool(row_regret.max() <= budget + TOLERANCE),
        "cell_bound_verified": bool(cell_regret.max() <= budget + TOLERANCE),
        "domain_bound_verified": bool(domain_regret.max() <= budget + TOLERANCE),
    }
    return cells, domains, comparison, projected, certificate


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    components = load_components()
    cell_blocks: list[pd.DataFrame] = []
    domain_blocks: list[pd.DataFrame] = []
    frontier_rows: list[dict[str, object]] = []
    comparisons: dict[str, dict[str, object]] = {}
    certificates: dict[str, dict[str, object]] = {}
    primary_prediction: np.ndarray | None = None

    for budget in BUDGET_GRID:
        cells, domains, comparison, prediction, certificate = audit_budget(
            components, budget
        )
        method = str(comparison["method"])
        cell_blocks.append(cells.assign(budget=budget))
        domain_blocks.append(domains.assign(budget=budget))
        comparisons[f"budget_{budget:g}"] = comparison
        certificates[f"budget_{budget:g}"] = certificate
        frontier_rows.append(
            {
                "budget": budget,
                "domain_equal_cell_macro_mae": float(domains["cell_macro_mae"].mean()),
                "worst_domain_cell_macro_mae": float(domains["cell_macro_mae"].max()),
                "domain_equal_difference_from_raw": comparison[
                    "domain_equal_mean_difference"
                ],
                "ci95_lower": comparison["ci95_domain_cluster_percentile"][0],
                "ci95_upper": comparison["ci95_domain_cluster_percentile"][1],
                "domain_wins": comparison["domain_wins_ties_losses"][0],
                "domain_ties": comparison["domain_wins_ties_losses"][1],
                "domain_losses": comparison["domain_wins_ties_losses"][2],
                "observed_maximum_domain_harm": comparison["maximum_domain_harm"],
                "deterministic_domain_harm_bound": budget,
                "all_certificate_checks_passed": bool(
                    certificate["casewise_budget_verified"]
                    and certificate["row_bound_verified"]
                    and certificate["cell_bound_verified"]
                    and certificate["domain_bound_verified"]
                ),
            }
        )
        if np.isclose(budget, PRIMARY_BUDGET):
            primary_prediction = prediction

    if primary_prediction is None:
        raise RuntimeError("primary budget is absent from the sensitivity grid")
    frontier = pd.DataFrame(frontier_rows)
    primary_comparison = comparisons[f"budget_{PRIMARY_BUDGET:g}"]
    primary_certificate = certificates[f"budget_{PRIMARY_BUDGET:g}"]
    baseline_domain_mae = float(
        frontier.loc[np.isclose(frontier["budget"], 0.0), "domain_equal_cell_macro_mae"].iloc[0]
    )
    primary_domain_mae = float(
        frontier.loc[
            np.isclose(frontier["budget"], PRIMARY_BUDGET),
            "domain_equal_cell_macro_mae",
        ].iloc[0]
    )

    primary_predictions = components[
        ["domain", "cell_id", "target_cycle_number", "truth", "baseline_prediction", "candidate_prediction"]
    ].copy()
    primary_predictions["projected_prediction"] = primary_prediction
    primary_predictions["absolute_update"] = np.abs(
        primary_predictions["projected_prediction"]
        - primary_predictions["baseline_prediction"]
    )
    primary_predictions["absolute_loss_regret"] = absolute_loss_regret(
        primary_predictions["truth"].to_numpy(float),
        primary_predictions["baseline_prediction"].to_numpy(float),
        primary_predictions["projected_prediction"].to_numpy(float),
    )

    paths = {
        "frontier": OUT / "risk_utility_frontier_v312.csv",
        "cell_metrics": OUT / "cell_level_metrics_all_budgets_v312.csv",
        "domain_metrics": OUT / "domain_level_metrics_all_budgets_v312.csv",
        "primary_predictions": OUT / "primary_delta_0p01_predictions_v312.parquet",
        "report": OUT / "regret_capped_context_change_v312_report.json",
    }
    frontier.to_csv(paths["frontier"], index=False)
    pd.concat(cell_blocks, ignore_index=True).to_csv(paths["cell_metrics"], index=False)
    pd.concat(domain_blocks, ignore_index=True).to_csv(paths["domain_metrics"], index=False)
    primary_predictions.to_parquet(paths["primary_predictions"], index=False)

    certificate_passed = all(
        bool(primary_certificate[key])
        for key in (
            "casewise_budget_verified",
            "row_bound_verified",
            "cell_bound_verified",
            "domain_bound_verified",
        )
    )
    status = (
        "REGRET_CAPPED_CONTEXT_CHANGE_DEVELOPMENT_GATE_PASSED"
        if (
            primary_comparison["domain_equal_mean_difference"] < 0.0
            and primary_comparison["ci95_domain_cluster_percentile"][1] < 0.0
            and primary_comparison["maximum_domain_harm"] <= PRIMARY_BUDGET
            and certificate_passed
        )
        else "REGRET_CAPPED_CONTEXT_CHANGE_DEVELOPMENT_GATE_NOT_PASSED"
    )
    report = {
        "status": status,
        "scope": (
            "retrospective development-only audit of frozen strict-LODO predictions "
            "from twelve previously opened public battery domains"
        ),
        "independent_unit_for_cross_domain_effect": "battery dataset domain",
        "nested_units": "physical cells within domain; cycles within physical cell",
        "target_outcome_access_during_projection": False,
        "baseline": BASELINE_METHOD,
        "candidate": CANDIDATE_METHOD,
        "primary_method": PRIMARY_METHOD,
        "primary_budget_soh_units": PRIMARY_BUDGET,
        "primary_result": {
            "baseline_domain_equal_cell_macro_mae": baseline_domain_mae,
            "method_domain_equal_cell_macro_mae": primary_domain_mae,
            "relative_improvement_percent": float(
                100.0 * (baseline_domain_mae - primary_domain_mae) / baseline_domain_mae
            ),
            **primary_comparison,
        },
        "deterministic_certificate": {
            "statement": (
                "for every outcome value, casewise absolute-loss regret relative "
                "to the baseline is at most the declared update budget; every "
                "non-negative weighted MAE aggregation inherits that bound"
            ),
            **primary_certificate,
        },
        "sensitivity_frontier": frontier.to_dict(orient="records"),
        "limitations": [
            "All twelve domains were opened before the method was proposed.",
            "The deterministic certificate bounds harm relative to the baseline; it does not guarantee improvement.",
            "The 0.01 SOH-unit budget is a declared practical tolerance and still requires independent confirmation for the empirical utility claim.",
            "The candidate context-change model and baseline share public development data and model family.",
        ],
        "files": {},
    }
    for name, path in paths.items():
        if name == "report":
            continue
        report["files"][name] = {"path": str(path), "sha256": sha256_file(path)}
    paths["report"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
