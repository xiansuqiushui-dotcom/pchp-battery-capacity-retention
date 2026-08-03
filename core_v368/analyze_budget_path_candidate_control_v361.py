"""Protocol-locked retrospective budget-path candidate-information control.

The analysis asks whether record-specific candidate information remains useful
across a frozen harm-budget path after replacing the earlier discrete safe-shift
control with the exact source-optimal constant shift for each causal baseline.
Development outcomes were already open before this protocol; the result is a
mechanism sensitivity, not a new confirmatory experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit


ROOT = Path(__file__).resolve().parent
PREFREEZE = ROOT / "paper_q1" / "BUDGET_PATH_CANDIDATE_CONTROL_PREFREEZE_V361.json"
V322 = ROOT / "prefix_causal_rccp_v322" / "prefix_causal_predictions_v322.parquet"
V326 = (
    ROOT
    / "nested_prefix_causal_selection_v326"
    / "nested_source_only_alpha_selection_v326_report.json"
)
V327 = (
    ROOT
    / "nested_prefix_causal_outer_v327"
    / "nested_prefix_causal_outer_v327_report.json"
)
V333 = (
    ROOT
    / "candidate_information_control_v333"
    / "candidate_information_control_v333_report.json"
)
PROJECTION = ROOT / "prefix_causal_harm_projection_v321.py"
CACHE = ROOT / "nested_prefix_causal_selection_v326" / "double_holdout_prediction_cache"
OUT = ROOT / "budget_path_candidate_control_v361"

BUDGETS = np.asarray([0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03], dtype=float)
ALPHAS = np.asarray([1.0, 0.5, 0.2, 0.1, 0.05, 0.03, 0.02, 0.015, 0.01, 0.005], dtype=float)
TOL = 1e-12
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 20260811
KEYS = ["domain", "cell_id", "target_cycle_number"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def cache_path(first: str, second: str) -> Path:
    a, b = sorted((first, second))
    return CACHE / f"exclude__{a}__{b}.parquet"


@njit(cache=True)
def causal_budget_grid(
    cell_codes: np.ndarray,
    raw_baseline: np.ndarray,
    raw_candidate: np.ndarray,
    alpha: float,
    budgets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one causal state and all frozen budgets in sorted cell order."""

    n = len(raw_baseline)
    k = len(budgets)
    baseline = np.empty(n, dtype=np.float64)
    projected = np.empty((k, n), dtype=np.float64)
    previous_projected = np.empty(k, dtype=np.float64)
    previous_cell = -1
    state = 0.0
    for index in range(n):
        cell = int(cell_codes[index])
        raw = raw_baseline[index]
        if index == 0 or cell != previous_cell:
            state = min(1.3, max(0.0, raw))
            previous_projected[:] = 1.3
            previous_cell = cell
        else:
            innovation = min(raw - state, 0.0)
            state = min(1.3, max(0.0, state + alpha * innovation))
        baseline[index] = state
        candidate = raw_candidate[index]
        for budget_index in range(k):
            budget = budgets[budget_index]
            lower = max(0.0, state - budget)
            upper = min(1.3, state + budget, previous_projected[budget_index])
            value = min(upper, max(lower, candidate))
            projected[budget_index, index] = value
            previous_projected[budget_index] = value
    return baseline, projected


def sorted_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    ordered = frame.sort_values(["cell_id", "target_cycle_number"], kind="mergesort").reset_index(drop=True)
    cell_codes, unique_cells = pd.factorize(ordered["cell_id"].astype(str), sort=False)
    counts = np.bincount(cell_codes, minlength=len(unique_cells)).astype(float)
    weights = 1.0 / (len(unique_cells) * counts[cell_codes])
    return {
        "cell_codes": cell_codes.astype(np.int64),
        "raw_baseline": ordered["raw_baseline"].to_numpy(float),
        "raw_candidate": ordered["raw_candidate"].to_numpy(float),
        "truth": ordered["truth"].to_numpy(float),
        "weights": weights,
        "cycles": ordered["target_cycle_number"].to_numpy(float),
        "cells": ordered["cell_id"].astype(str).to_numpy(),
    }


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    threshold = 0.5 * float(sorted_weights.sum())
    index = int(np.searchsorted(np.cumsum(sorted_weights), threshold, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def trajectory_nonincreasing(cells: np.ndarray, values: np.ndarray) -> bool:
    for cell in pd.unique(cells):
        block = values[cells == cell]
        if bool((np.diff(block) > TOL).any()):
            return False
    return True


def bootstrap_interval(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for start in range(0, BOOTSTRAP_REPLICATES, 10_000):
        stop = min(start + 10_000, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return [float(value) for value in np.percentile(means, [2.5, 97.5])]


def exact_sign_p(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    tail = min(wins, losses)
    return float(min(1.0, 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)))


def verify_freeze() -> tuple[dict[str, object], dict[str, object]]:
    frozen = json.loads(PREFREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN_PROTOCOL_LOCKED_RETROSPECTIVE_BUDGET_PATH_BEFORE_EXECUTION":
        raise RuntimeError("budget-path prefreeze status mismatch")
    paths = {
        "script": Path(__file__).resolve(),
        "v322_predictions": V322,
        "v326_selection_report": V326,
        "v327_outer_report": V327,
        "v333_control_report": V333,
        "projection_implementation": PROJECTION,
    }
    for name, path in paths.items():
        observed = sha256_file(path)
        expected = str(frozen["frozen_artifacts"][name]["sha256"]).upper()
        if observed != expected:
            raise RuntimeError(f"frozen hash mismatch for {name}: {observed}")
    if [float(value) for value in frozen["budget_grid"]] != BUDGETS.tolist():
        raise RuntimeError("budget grid mismatch")
    if [float(value) for value in frozen["alpha_grid"]] != ALPHAS.tolist():
        raise RuntimeError("alpha grid mismatch")

    selection = json.loads(V326.read_text(encoding="utf-8"))
    if selection.get("status") != "SOURCE_ONLY_NESTED_ALPHA_SELECTIONS_FROZEN_BEFORE_OUTER_SCORING":
        raise RuntimeError("V326 selection status mismatch")
    for item in selection["double_holdout_caches"]:
        path = cache_path(*item["excluded_domains"])
        if sha256_file(path) != str(item["sha256"]).upper():
            raise RuntimeError(f"double-holdout cache mismatch: {path.name}")
    return frozen, selection


def build_inner_selections(
    outer_domain: str,
    domains: list[str],
    truth_table: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    inner_domains = [domain for domain in domains if domain != outer_domain]
    pchp_scores = np.zeros((len(ALPHAS), len(BUDGETS)), dtype=float)
    residuals: list[list[np.ndarray]] = [[] for _ in ALPHAS]
    residual_weights: list[list[np.ndarray]] = [[] for _ in ALPHAS]
    baseline_margins = np.full((len(ALPHAS), 2), [np.inf, np.inf], dtype=float)

    for inner_domain in inner_domains:
        cached = pd.read_parquet(cache_path(outer_domain, inner_domain))
        cached = cached.loc[cached["domain"].astype(str) == inner_domain].copy()
        truth = truth_table.loc[truth_table["domain"].astype(str) == inner_domain]
        validation = truth.merge(cached, on=KEYS, how="inner", validate="one_to_one")
        if len(validation) != len(cached):
            raise RuntimeError(f"inner alignment failed: outer={outer_domain}, inner={inner_domain}")
        arrays = sorted_arrays(validation)
        for alpha_index, alpha in enumerate(ALPHAS):
            baseline, methods = causal_budget_grid(
                arrays["cell_codes"],
                arrays["raw_baseline"],
                arrays["raw_candidate"],
                float(alpha),
                BUDGETS,
            )
            pchp_scores[alpha_index] += np.sum(
                np.abs(methods - arrays["truth"][None, :]) * arrays["weights"][None, :],
                axis=1,
            )
            residuals[alpha_index].append(arrays["truth"] - baseline)
            residual_weights[alpha_index].append(arrays["weights"])
            baseline_margins[alpha_index, 0] = min(
                baseline_margins[alpha_index, 0], float(baseline.min())
            )
            baseline_margins[alpha_index, 1] = min(
                baseline_margins[alpha_index, 1], float(1.3 - baseline.max())
            )

    pchp_scores /= len(inner_domains)
    control_scores = np.empty((len(ALPHAS), len(BUDGETS)), dtype=float)
    control_offsets = np.empty_like(control_scores)
    for alpha_index in range(len(ALPHAS)):
        residual = np.concatenate(residuals[alpha_index])
        weights = np.concatenate(residual_weights[alpha_index]) / len(inner_domains)
        unconstrained = weighted_median(residual, weights)
        for budget_index, budget in enumerate(BUDGETS):
            offset = float(np.clip(unconstrained, -budget, budget))
            control_offsets[alpha_index, budget_index] = offset
            control_scores[alpha_index, budget_index] = float(
                np.sum(weights * np.abs(residual - offset))
            )

    pchp_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    for budget_index, budget in enumerate(BUDGETS):
        pchp_minimum = float(pchp_scores[:, budget_index].min())
        pchp_candidates = np.flatnonzero(
            pchp_scores[:, budget_index] <= pchp_minimum + TOL
        )
        pchp_index = int(pchp_candidates[0])  # ALPHAS is descending.
        pchp_rows.append(
            {
                "outer_target_domain": outer_domain,
                "budget": float(budget),
                "selected_pchp_alpha": float(ALPHAS[pchp_index]),
                "selected_inner_domain_equal_pchp_mae": float(
                    pchp_scores[pchp_index, budget_index]
                ),
                "inner_domains": len(inner_domains),
                "outer_target_labels_used_for_selection": False,
            }
        )

        candidates = []
        for alpha_index, alpha in enumerate(ALPHAS):
            candidates.append(
                (
                    float(control_scores[alpha_index, budget_index]),
                    abs(float(control_offsets[alpha_index, budget_index])),
                    -float(alpha),
                    -float(control_offsets[alpha_index, budget_index]),
                    alpha_index,
                )
            )
        control_index = int(min(candidates)[-1])
        control_rows.append(
            {
                "outer_target_domain": outer_domain,
                "budget": float(budget),
                "selected_control_alpha": float(ALPHAS[control_index]),
                "selected_exact_constant_offset": float(
                    control_offsets[control_index, budget_index]
                ),
                "selected_inner_domain_equal_control_mae": float(
                    control_scores[control_index, budget_index]
                ),
                "minimum_baseline_distance_to_range_boundary": float(
                    baseline_margins[control_index].min()
                ),
                "inner_domains": len(inner_domains),
                "outer_target_labels_used_for_selection": False,
            }
        )
    return pchp_rows, control_rows


def score_outer_domain(
    domain: str,
    outer: pd.DataFrame,
    pchp_selections: list[dict[str, object]],
    control_selections: list[dict[str, object]],
) -> list[dict[str, object]]:
    arrays = sorted_arrays(outer)
    rows: list[dict[str, object]] = []
    for pchp_selection, control_selection in zip(pchp_selections, control_selections):
        budget = float(pchp_selection["budget"])
        if not np.isclose(budget, float(control_selection["budget"]), rtol=0.0, atol=TOL):
            raise RuntimeError("selection budget alignment failed")
        pchp_baseline, pchp_grid = causal_budget_grid(
            arrays["cell_codes"],
            arrays["raw_baseline"],
            arrays["raw_candidate"],
            float(pchp_selection["selected_pchp_alpha"]),
            np.asarray([budget], dtype=float),
        )
        pchp = pchp_grid[0]
        control_baseline, _ = causal_budget_grid(
            arrays["cell_codes"],
            arrays["raw_baseline"],
            arrays["raw_candidate"],
            float(control_selection["selected_control_alpha"]),
            np.asarray([0.0], dtype=float),
        )
        offset = float(control_selection["selected_exact_constant_offset"])
        control = np.clip(control_baseline + offset, 0.0, 1.3)
        truth = arrays["truth"]
        weights = arrays["weights"]
        baseline_mae = float(np.sum(weights * np.abs(pchp_baseline - truth)))
        pchp_mae = float(np.sum(weights * np.abs(pchp - truth)))
        control_mae = float(np.sum(weights * np.abs(control - truth)))
        maximum_displacement = float(np.max(np.abs(pchp - pchp_baseline)))
        maximum_observed_regret = float(
            np.max(np.abs(pchp - truth) - np.abs(pchp_baseline - truth))
        )
        cell_frame = pd.DataFrame(
            {
                "cell": arrays["cells"],
                "pchp_error": np.abs(pchp - truth),
                "baseline_error": np.abs(pchp_baseline - truth),
                "control_error": np.abs(control - truth),
            }
        )
        cell = cell_frame.groupby("cell", sort=False).mean()
        rows.append(
            {
                "domain": domain,
                "budget": budget,
                "physical_cells": int(cell.shape[0]),
                "prediction_rows": int(len(truth)),
                "selected_pchp_alpha": float(pchp_selection["selected_pchp_alpha"]),
                "selected_control_alpha": float(control_selection["selected_control_alpha"]),
                "selected_control_offset": offset,
                "baseline_cell_macro_mae": baseline_mae,
                "pchp_cell_macro_mae": pchp_mae,
                "control_cell_macro_mae": control_mae,
                "pchp_minus_baseline": pchp_mae - baseline_mae,
                "pchp_minus_control": pchp_mae - control_mae,
                "maximum_pchp_displacement": maximum_displacement,
                "maximum_observed_pchp_loss_regret": maximum_observed_regret,
                "maximum_cell_macro_pchp_harm": float(
                    (cell["pchp_error"] - cell["baseline_error"]).max()
                ),
                "pchp_nonincreasing": trajectory_nonincreasing(arrays["cells"], pchp),
                "baseline_nonincreasing": trajectory_nonincreasing(
                    arrays["cells"], pchp_baseline
                ),
                "control_nonincreasing": trajectory_nonincreasing(
                    arrays["cells"], control
                ),
            }
        )
    return rows


def summarize(domain_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    budget_rows: list[dict[str, object]] = []
    for budget, group in domain_metrics.groupby("budget", sort=True):
        differences = group["pchp_minus_control"].to_numpy(float)
        baseline_differences = group["pchp_minus_baseline"].to_numpy(float)
        wins = int((differences < -TOL).sum())
        ties = int((np.abs(differences) <= TOL).sum())
        losses = int((differences > TOL).sum())
        budget_rows.append(
            {
                "budget": float(budget),
                "domains": int(len(group)),
                "domain_equal_baseline_mae": float(group["baseline_cell_macro_mae"].mean()),
                "domain_equal_pchp_mae": float(group["pchp_cell_macro_mae"].mean()),
                "domain_equal_exact_shift_control_mae": float(group["control_cell_macro_mae"].mean()),
                "domain_equal_pchp_minus_baseline": float(baseline_differences.mean()),
                "domain_equal_pchp_minus_control": float(differences.mean()),
                "descriptive_domain_bootstrap_ci95_pchp_minus_control": bootstrap_interval(differences),
                "wins_ties_losses_pchp_vs_control": [wins, ties, losses],
                "maximum_pchp_displacement": float(group["maximum_pchp_displacement"].max()),
                "maximum_observed_pchp_loss_regret": float(
                    group["maximum_observed_pchp_loss_regret"].max()
                ),
                "maximum_domain_pchp_harm": float(
                    group["pchp_minus_baseline"].max()
                ),
            }
        )
    budget_summary = pd.DataFrame(budget_rows).sort_values("budget").reset_index(drop=True)

    auc_rows: list[dict[str, object]] = []
    for domain, group in domain_metrics.groupby("domain", sort=True):
        ordered = group.sort_values("budget")
        auc = float(
            np.trapz(
                ordered["pchp_minus_control"].to_numpy(float),
                ordered["budget"].to_numpy(float),
            )
            / float(BUDGETS.max())
        )
        utility_auc = float(
            np.trapz(
                ordered["pchp_minus_baseline"].to_numpy(float),
                ordered["budget"].to_numpy(float),
            )
            / float(BUDGETS.max())
        )
        auc_rows.append(
            {
                "domain": domain,
                "budget_normalized_auc_pchp_minus_control": auc,
                "budget_normalized_auc_pchp_minus_baseline": utility_auc,
            }
        )
    auc_table = pd.DataFrame(auc_rows)
    values = auc_table["budget_normalized_auc_pchp_minus_control"].to_numpy(float)
    wins = int((values < -TOL).sum())
    ties = int((np.abs(values) <= TOL).sum())
    losses = int((values > TOL).sum())
    primary = {
        "estimand": (
            "domain-equal mean of the budget-normalized trapezoidal integral "
            "of PCHP-minus-exact-safe-shift cell-macro MAE over delta in [0, 0.03]"
        ),
        "independent_unit": "complete battery dataset domain",
        "domains": int(len(values)),
        "domain_equal_mean": float(values.mean()),
        "domain_bootstrap_ci95": bootstrap_interval(values),
        "wins_ties_losses": [wins, ties, losses],
        "exact_two_sided_sign_p_descriptive": exact_sign_p(wins, losses),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    return budget_summary, auc_table, primary


def main() -> int:
    started = time.perf_counter()
    frozen, selection_report = verify_freeze()
    v322 = pd.read_parquet(V322)[
        KEYS + ["truth", "raw_baseline", "raw_candidate"]
    ].copy()
    if v322.duplicated(KEYS).any():
        raise RuntimeError("V322 key duplication")
    domains = sorted(v322["domain"].astype(str).unique())
    if len(domains) != 12:
        raise RuntimeError(f"unexpected domain count: {len(domains)}")
    truth_table = v322[KEYS + ["truth"]].copy()

    pchp_selection_rows: list[dict[str, object]] = []
    control_selection_rows: list[dict[str, object]] = []
    domain_rows: list[dict[str, object]] = []
    for index, domain in enumerate(domains, start=1):
        pchp_rows, control_rows = build_inner_selections(
            domain, domains, truth_table
        )
        pchp_selection_rows.extend(pchp_rows)
        control_selection_rows.extend(control_rows)
        outer = v322.loc[v322["domain"].astype(str) == domain].copy()
        domain_rows.extend(
            score_outer_domain(domain, outer, pchp_rows, control_rows)
        )
        print(f"budget-path outer domain {index}/{len(domains)} complete: {domain}", flush=True)

    pchp_selections = pd.DataFrame(pchp_selection_rows)
    control_selections = pd.DataFrame(control_selection_rows)
    domain_metrics = pd.DataFrame(domain_rows)
    budget_summary, auc_table, primary = summarize(domain_metrics)

    original = {
        str(item["outer_target_domain"]): float(item["selected_alpha"])
        for item in selection_report["selections"]
    }
    delta_001 = pchp_selections.loc[
        np.isclose(pchp_selections["budget"], 0.01, rtol=0.0, atol=TOL)
    ]
    reproduced_alpha = {
        str(row.outer_target_domain): float(row.selected_pchp_alpha)
        for row in delta_001.itertuples(index=False)
    }
    original_v327 = json.loads(V327.read_text(encoding="utf-8"))
    anchor = budget_summary.loc[
        np.isclose(budget_summary["budget"], 0.01, rtol=0.0, atol=TOL)
    ].iloc[0]
    anchor_checks = {
        "selected_alpha_roster_matches_v326": reproduced_alpha == original,
        "pchp_mae_matches_v327": bool(
            np.isclose(
                float(anchor["domain_equal_pchp_mae"]),
                next(
                    item["domain_equal_cell_macro_mae"]
                    for item in original_v327["summary"]
                    if item["method"] == "selected_causal_method"
                ),
                rtol=0.0,
                atol=TOL,
            )
        ),
        "pchp_minus_baseline_matches_v327": bool(
            np.isclose(
                float(anchor["domain_equal_pchp_minus_baseline"]),
                original_v327["primary_comparison"]["domain_equal_mean_difference"],
                rtol=0.0,
                atol=TOL,
            )
        ),
    }
    positive = budget_summary.loc[budget_summary["budget"] > 0.0]
    structural = {
        "zero_budget_identity": bool(
            np.allclose(
                domain_metrics.loc[domain_metrics["budget"] == 0.0, "maximum_pchp_displacement"],
                0.0,
                rtol=0.0,
                atol=TOL,
            )
        ),
        "all_displacements_within_budget": bool(
            (
                domain_metrics["maximum_pchp_displacement"]
                <= domain_metrics["budget"] + TOL
            ).all()
        ),
        "all_observed_regrets_within_budget": bool(
            (
                domain_metrics["maximum_observed_pchp_loss_regret"]
                <= domain_metrics["budget"] + TOL
            ).all()
        ),
        "all_trajectories_nonincreasing": bool(
            domain_metrics[
                ["pchp_nonincreasing", "baseline_nonincreasing", "control_nonincreasing"]
            ].to_numpy(bool).all()
        ),
        "exact_shift_has_no_range_clipping_on_inner_selection": bool(
            (
                control_selections["minimum_baseline_distance_to_range_boundary"]
                >= control_selections["budget"] - TOL
            ).all()
        ),
    }
    gates = {
        "all_anchor_checks": all(anchor_checks.values()),
        "all_structural_checks": all(structural.values()),
        "primary_auc_mean_below_zero": primary["domain_equal_mean"] < 0.0,
        "primary_auc_bootstrap_upper_below_zero": primary["domain_bootstrap_ci95"][1] < 0.0,
        "majority_positive_budgets_favor_pchp": int(
            (positive["domain_equal_pchp_minus_control"] < 0.0).sum()
        )
        >= 4,
    }
    decision = "RETAIN" if all(gates.values()) else "NARROW"
    status = (
        "PROTOCOL_LOCKED_RETROSPECTIVE_BUDGET_PATH_GATE_PASSED"
        if decision == "RETAIN"
        else "PROTOCOL_LOCKED_RETROSPECTIVE_BUDGET_PATH_GATE_NOT_PASSED"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "pchp_selections": OUT / "budget_path_pchp_selections_v361.csv",
        "control_selections": OUT / "budget_path_exact_shift_selections_v361.csv",
        "domain_metrics": OUT / "budget_path_domain_metrics_v361.csv",
        "budget_summary": OUT / "budget_path_summary_v361.csv",
        "auc_table": OUT / "budget_path_domain_auc_v361.csv",
        "report": OUT / "budget_path_candidate_control_v361_report.json",
    }
    pchp_selections.to_csv(paths["pchp_selections"], index=False)
    control_selections.to_csv(paths["control_selections"], index=False)
    domain_metrics.to_csv(paths["domain_metrics"], index=False)
    budget_summary.to_csv(paths["budget_summary"], index=False)
    auc_table.to_csv(paths["auc_table"], index=False)
    report = {
        "status": status,
        "decision": decision,
        "generated_at_local": datetime.now().astimezone().isoformat(),
        "evidence_role": "protocol_locked_retrospective_mechanism_sensitivity",
        "warning": (
            "Development outcomes were open before V361. The analysis may support a "
            "bounded mechanism sensitivity but is not a new confirmatory experiment."
        ),
        "question": (
            "Does record-specific candidate information improve over the exact "
            "source-selected constant safe shift across a frozen harm-budget path?"
        ),
        "design": {
            "independent_unit": "complete battery dataset domain",
            "nested_units": "records within physical cells within domains",
            "domains": len(domains),
            "physical_cells": int(v322["cell_id"].nunique()),
            "records": int(len(v322)),
            "budget_grid": BUDGETS.tolist(),
            "alpha_grid": ALPHAS.tolist(),
            "pchp_selection": (
                "for each outer domain and budget, minimize inner-domain-equal "
                "cell-macro MAE over alpha using only the other eleven domains"
            ),
            "control_selection": (
                "for each alpha, compute the exact weighted-median constant shift "
                "under inner-domain-equal cell-macro MAE, clip it to [-delta, delta], "
                "then select alpha using only the other eleven domains"
            ),
            "outer_target_labels_used_for_selection": False,
        },
        "primary": primary,
        "budget_summary": json.loads(budget_summary.to_json(orient="records")),
        "anchor_checks": anchor_checks,
        "structural_checks": structural,
        "gates": gates,
        "selection_counts": {
            "pchp_alpha": {
                str(key): int(value)
                for key, value in sorted(
                    Counter(pchp_selections["selected_pchp_alpha"]).items()
                )
            },
            "control_alpha": {
                str(key): int(value)
                for key, value in sorted(
                    Counter(control_selections["selected_control_alpha"]).items()
                )
            },
        },
        "interpretation_if_not_passed": frozen["interpretation_if_not_passed"],
        "frozen_protocol": {
            "path": str(PREFREEZE),
            "sha256": sha256_file(PREFREEZE),
        },
        "outputs": {},
        "runtime_seconds": float(time.perf_counter() - started),
    }
    for name, path in paths.items():
        if name != "report":
            report["outputs"][name] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
    paths["report"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(status, flush=True)
    print(json.dumps({"primary": primary, "gates": gates}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
