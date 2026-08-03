"""Development audit of monotone regret-capped context--change SOH prediction.

The procedure protects a label-free isotonic version of the frozen raw-feature
baseline.  A frozen commissioning context--change candidate is projected into
a fixed-width tube around that protected baseline.  Both input trajectories
are non-increasing, so a constant-width pointwise projection preserves the
trajectory constraint while retaining the exact absolute-loss regret cap.

All twelve battery domains were opened before this refinement.  The output is
therefore development evidence and must not be described as external
confirmation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from develop_anchor_invariant_soh_v306 import (
    TARGET,
    cell_metrics,
    domain_summary,
    monotone_project,
    paired_domain_comparison,
)
from evaluate_regret_capped_context_change_v312 import load_components
from regret_capped_projection_v312 import (
    absolute_loss_regret,
    regret_capped_projection,
    verify_absolute_loss_budget,
    worst_case_absolute_loss_regret,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "monotone_regret_capped_soh_v313"
PRIMARY_BUDGET = 0.01
BUDGET_GRID = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03)
BASELINE_METHOD = "raw_early_charge_isotonic"
CANDIDATE_METHOD = "raw_change_domain_equal_monotone"
PRIMARY_METHOD = "monotone_regret_capped_context_change_delta_0p01"
TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scoring_frame(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[["domain", "cell_id", "target_cycle_number", "truth"]].rename(
        columns={"truth": TARGET}
    )


def all_cell_trajectories_nonincreasing(
    rows: pd.DataFrame,
    prediction: np.ndarray,
    *,
    tolerance: float = TOLERANCE,
) -> tuple[bool, float, int]:
    working = rows[["domain", "cell_id", "target_cycle_number"]].copy()
    working["prediction"] = np.asarray(prediction, dtype=float)
    maximum_increase = 0.0
    violating_cells = 0
    for _, cell in working.groupby(["domain", "cell_id"], sort=False):
        ordered = cell.sort_values("target_cycle_number", kind="mergesort")
        differences = np.diff(ordered["prediction"].to_numpy(float))
        if len(differences):
            maximum_increase = max(maximum_increase, float(differences.max()))
            violating_cells += int(bool(np.any(differences > tolerance)))
    return violating_cells == 0, maximum_increase, violating_cells


def two_sided_exact_sign_pvalue(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    smaller = min(wins, losses)
    probability = sum(math.comb(n, k) for k in range(smaller + 1)) / (2**n)
    return min(1.0, 2.0 * probability)


def prepare_components() -> pd.DataFrame:
    components = load_components()
    frame = scoring_frame(components)
    protected_baseline = monotone_project(
        frame, components["baseline_prediction"].to_numpy(float)
    )
    output = components.copy()
    output["protected_baseline_prediction"] = protected_baseline
    baseline_check = all_cell_trajectories_nonincreasing(output, protected_baseline)
    candidate_check = all_cell_trajectories_nonincreasing(
        output, output["candidate_prediction"].to_numpy(float)
    )
    if not baseline_check[0] or not candidate_check[0]:
        raise RuntimeError("protected baseline and candidate must be non-increasing")
    return output


def audit_budget(
    components: pd.DataFrame,
    budget: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], np.ndarray, dict[str, object]]:
    truth = components["truth"].to_numpy(float)
    baseline = components["protected_baseline_prediction"].to_numpy(float)
    candidate = components["candidate_prediction"].to_numpy(float)
    projected = regret_capped_projection(baseline, candidate, budget)
    method = f"monotone_regret_capped_context_change_delta_{budget:g}"
    frame = scoring_frame(components)

    baseline_cells = cell_metrics(frame, baseline, BASELINE_METHOD)
    baseline_domains = domain_summary(baseline_cells)
    method_cells = cell_metrics(frame, projected, method)
    method_domains = domain_summary(method_cells)
    comparison = paired_domain_comparison(
        pd.concat([baseline_domains, method_domains], ignore_index=True),
        method,
        BASELINE_METHOD,
    )

    row_regret = absolute_loss_regret(truth, baseline, projected)
    paired_cells = method_cells[["domain", "cell_id", "mae"]].merge(
        baseline_cells[["domain", "cell_id", "mae"]],
        on=["domain", "cell_id"],
        suffixes=("_method", "_baseline"),
        validate="one_to_one",
    )
    cell_regret = paired_cells["mae_method"] - paired_cells["mae_baseline"]
    domain_regret = (
        method_domains.set_index("domain")["cell_macro_mae"]
        - baseline_domains.set_index("domain")["cell_macro_mae"]
    )
    monotone, maximum_increase, violating_cells = all_cell_trajectories_nonincreasing(
        components, projected
    )
    worst_case_regret = worst_case_absolute_loss_regret(baseline, projected)
    certificate = {
        "budget": budget,
        "maximum_absolute_prediction_update": float(np.abs(projected - baseline).max()),
        "maximum_exact_worst_case_absolute_loss_regret": float(worst_case_regret.max()),
        "maximum_observed_row_absolute_loss_regret": float(row_regret.max()),
        "maximum_cell_macro_mae_regret": float(cell_regret.max()),
        "maximum_domain_cell_macro_mae_regret": float(domain_regret.max()),
        "casewise_budget_verified_for_observed_truths": verify_absolute_loss_budget(
            truth, baseline, candidate, budget
        ),
        "worst_case_identity_budget_verified": bool(
            worst_case_regret.max() <= budget + TOLERANCE
        ),
        "row_bound_verified": bool(row_regret.max() <= budget + TOLERANCE),
        "cell_bound_verified": bool(cell_regret.max() <= budget + TOLERANCE),
        "domain_bound_verified": bool(domain_regret.max() <= budget + TOLERANCE),
        "all_projected_cell_trajectories_nonincreasing": monotone,
        "maximum_projected_trajectory_increase": maximum_increase,
        "violating_projected_cells": violating_cells,
    }
    wins, _, losses = comparison["domain_wins_ties_losses"]
    comparison["two_sided_exact_sign_pvalue_descriptive"] = two_sided_exact_sign_pvalue(
        int(wins), int(losses)
    )
    return method_cells, method_domains, comparison, projected, certificate


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    components = prepare_components()
    frame = scoring_frame(components)
    protected_baseline = components["protected_baseline_prediction"].to_numpy(float)
    baseline_cells = cell_metrics(frame, protected_baseline, BASELINE_METHOD)
    baseline_domains = domain_summary(baseline_cells)
    baseline_domain_equal_mae = float(baseline_domains["cell_macro_mae"].mean())

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
        cell_blocks.append(cells.assign(budget=budget))
        domain_blocks.append(domains.assign(budget=budget))
        comparisons[f"budget_{budget:g}"] = comparison
        certificates[f"budget_{budget:g}"] = certificate
        frontier_rows.append(
            {
                "budget": budget,
                "domain_equal_cell_macro_mae": float(domains["cell_macro_mae"].mean()),
                "domain_equal_difference_from_protected_baseline": comparison[
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
                    certificate["worst_case_identity_budget_verified"]
                    and certificate["row_bound_verified"]
                    and certificate["cell_bound_verified"]
                    and certificate["domain_bound_verified"]
                    and certificate["all_projected_cell_trajectories_nonincreasing"]
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
    primary_domain_mae = float(
        frontier.loc[
            np.isclose(frontier["budget"], PRIMARY_BUDGET),
            "domain_equal_cell_macro_mae",
        ].iloc[0]
    )

    primary_predictions = components[
        [
            "domain",
            "cell_id",
            "target_cycle_number",
            "truth",
            "baseline_prediction",
            "protected_baseline_prediction",
            "candidate_prediction",
        ]
    ].copy()
    primary_predictions["projected_prediction"] = primary_prediction
    primary_predictions["absolute_update"] = np.abs(
        primary_prediction - protected_baseline
    )
    primary_predictions["observed_absolute_loss_regret"] = absolute_loss_regret(
        primary_predictions["truth"].to_numpy(float),
        protected_baseline,
        primary_prediction,
    )

    paths = {
        "frontier": OUT / "risk_utility_frontier_v313.csv",
        "cell_metrics": OUT / "cell_level_metrics_all_budgets_v313.csv",
        "domain_metrics": OUT / "domain_level_metrics_all_budgets_v313.csv",
        "primary_predictions": OUT / "primary_delta_0p01_predictions_v313.parquet",
        "report": OUT / "monotone_regret_capped_soh_v313_report.json",
    }
    frontier.to_csv(paths["frontier"], index=False)
    pd.concat(cell_blocks, ignore_index=True).to_csv(paths["cell_metrics"], index=False)
    pd.concat(domain_blocks, ignore_index=True).to_csv(paths["domain_metrics"], index=False)
    primary_predictions.to_parquet(paths["primary_predictions"], index=False)

    certificate_passed = all(
        bool(primary_certificate[key])
        for key in (
            "worst_case_identity_budget_verified",
            "row_bound_verified",
            "cell_bound_verified",
            "domain_bound_verified",
            "all_projected_cell_trajectories_nonincreasing",
        )
    )
    status = (
        "MONOTONE_REGRET_CAPPED_CONTEXT_CHANGE_DEVELOPMENT_GATE_PASSED"
        if (
            primary_comparison["domain_equal_mean_difference"] < 0.0
            and primary_comparison["ci95_domain_cluster_percentile"][1] < 0.0
            and primary_comparison["maximum_domain_harm"] <= PRIMARY_BUDGET
            and certificate_passed
        )
        else "MONOTONE_REGRET_CAPPED_CONTEXT_CHANGE_DEVELOPMENT_GATE_NOT_PASSED"
    )
    report = {
        "status": status,
        "scope": (
            "retrospective development-only strict leave-one-battery-dataset-domain-out "
            "audit on twelve previously opened public domains"
        ),
        "independent_unit_for_cross_domain_effect": "battery dataset domain",
        "nested_units": "physical cells within domain; cycles within physical cell",
        "domains": int(components["domain"].nunique()),
        "physical_cells": int(components["cell_id"].nunique()),
        "post_reference_rows": int(len(components)),
        "target_outcome_access_during_training_projection_or_monotone_processing": False,
        "protected_baseline": BASELINE_METHOD,
        "candidate": CANDIDATE_METHOD,
        "primary_method": PRIMARY_METHOD,
        "primary_budget_soh_units": PRIMARY_BUDGET,
        "primary_result": {
            "baseline_domain_equal_cell_macro_mae": baseline_domain_equal_mae,
            "method_domain_equal_cell_macro_mae": primary_domain_mae,
            "relative_improvement_percent": float(
                100.0
                * (baseline_domain_equal_mae - primary_domain_mae)
                / baseline_domain_equal_mae
            ),
            **primary_comparison,
        },
        "deterministic_certificate": {
            "statement": (
                "for every real outcome, casewise absolute-loss regret relative to "
                "the protected baseline equals at most the declared update budget; "
                "all non-negative weighted MAE aggregations inherit the bound"
            ),
            "trajectory_statement": (
                "with a constant budget, coordinatewise projection of two "
                "non-increasing input trajectories remains non-increasing"
            ),
            **primary_certificate,
        },
        "sensitivity_frontier": frontier.to_dict(orient="records"),
        "limitations": [
            "All twelve domains were opened before this method refinement.",
            "The deterministic certificate limits harm relative to the protected baseline; it does not guarantee improvement.",
            "The primary 0.01 SOH-unit budget is a declared engineering tolerance and requires independently frozen confirmation.",
            "The exact sign-test value is descriptive because method development used these domains.",
            "The monotonicity constraint suppresses upward recovery and short-term capacity noise even when such fluctuations are physically observed.",
        ],
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
