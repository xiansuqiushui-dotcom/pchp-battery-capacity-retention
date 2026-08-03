"""Protocol-locked retrospective evaluation of bounded-recovery PCHP V374."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from bounded_recovery_pchp_v374 import (
    prefix_causal_bounded_recovery_cellwise_projection,
    prefix_causal_bounded_recovery_projection,
)
from prefix_causal_harm_projection_v321 import (
    prefix_causal_cellwise_projection,
)


ROOT = Path(__file__).resolve().parent
PREFREEZE = ROOT / "BOUNDED_RECOVERY_PCHP_PREFREEZE_V374.json"
SELECTION_REPORT = (
    ROOT
    / "nested_prefix_causal_selection_v326"
    / "nested_source_only_alpha_selection_v326_report.json"
)
CACHE = (
    ROOT
    / "nested_prefix_causal_selection_v326"
    / "double_holdout_prediction_cache"
)
OUTER_PREDICTIONS = (
    ROOT / "prefix_causal_rccp_v322" / "prefix_causal_predictions_v322.parquet"
)
OUT = ROOT / "bounded_recovery_pchp_v374"
TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_path(first: str, second: str) -> Path:
    a, b = sorted((first, second))
    return CACHE / f"exclude__{a}__{b}.parquet"


def verify_prefreeze() -> dict[str, object]:
    prefreeze = json.loads(PREFREEZE.read_text(encoding="utf-8"))
    if prefreeze["status"] != "PREFROZEN_BEFORE_V374_EXECUTION":
        raise RuntimeError("V374 prefreeze status is not executable")
    identities = {
        SELECTION_REPORT: prefreeze["inputs"]["nested_selection_report"]["sha256"],
        OUTER_PREDICTIONS: prefreeze["inputs"]["outer_predictions"]["sha256"],
        ROOT / prefreeze["inputs"]["authoritative_operator"]["path"]: (
            prefreeze["inputs"]["authoritative_operator"]["sha256"]
        ),
    }
    for path, expected in identities.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"V374 frozen input identity mismatch: {path}")
    return prefreeze


@njit(cache=True)
def macro_mae_for_configurations(
    raw: np.ndarray,
    candidate: np.ndarray,
    truth: np.ndarray,
    new_cell: np.ndarray,
    alphas: np.ndarray,
    recoveries: np.ndarray,
    budget: float,
) -> np.ndarray:
    """Compute cell-macro MAE for every configuration in one ordered domain."""

    n_configs = alphas.size
    state = np.zeros(n_configs, dtype=np.float64)
    previous = np.zeros(n_configs, dtype=np.float64)
    cell_error = np.zeros(n_configs, dtype=np.float64)
    macro_sum = np.zeros(n_configs, dtype=np.float64)
    cell_rows = 0
    cells = 0

    for row in range(raw.size):
        if new_cell[row]:
            if row > 0:
                macro_sum += cell_error / cell_rows
            cells += 1
            cell_rows = 0
            cell_error[:] = 0.0
            first_state = min(1.3, max(0.0, raw[row]))
            for config in range(n_configs):
                state[config] = first_state
                lower = max(0.0, first_state - budget)
                upper = min(1.3, first_state + budget)
                value = min(upper, max(lower, candidate[row]))
                previous[config] = value
                cell_error[config] += abs(value - truth[row])
        else:
            for config in range(n_configs):
                innovation = raw[row] - state[config]
                if innovation <= 0.0:
                    state[config] += alphas[config] * innovation
                else:
                    state[config] += min(innovation, recoveries[config])
                state[config] = min(1.3, max(0.0, state[config]))
                lower = max(0.0, state[config] - budget)
                upper = min(
                    1.3,
                    state[config] + budget,
                    previous[config] + recoveries[config],
                )
                if lower > upper:
                    upper = lower
                value = min(upper, max(lower, candidate[row]))
                previous[config] = value
                cell_error[config] += abs(value - truth[row])
        cell_rows += 1

    macro_sum += cell_error / cell_rows
    return macro_sum / cells


def ordered_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
    ordered = frame.sort_values(
        ["cell_id", "target_cycle_number"], kind="mergesort"
    ).reset_index(drop=True)
    identifiers = ordered["cell_id"].astype(str).to_numpy()
    new_cell = np.empty(len(ordered), dtype=np.bool_)
    new_cell[0] = True
    new_cell[1:] = identifiers[1:] != identifiers[:-1]
    return (
        ordered["raw_baseline"].to_numpy(float),
        ordered["raw_candidate"].to_numpy(float),
        ordered["truth"].to_numpy(float),
        new_cell,
    )


def configuration_table(prefreeze: dict[str, object]) -> pd.DataFrame:
    rows = [
        {"alpha": float(alpha), "recovery": float(recovery)}
        for alpha in prefreeze["alpha_grid"]
        for recovery in prefreeze["recovery_grid_soh_per_record"]
    ]
    return pd.DataFrame(rows).sort_values(
        ["recovery", "alpha"], ascending=[True, False]
    ).reset_index(drop=True)


def select_configuration(
    aggregate: pd.DataFrame,
    *,
    strict: bool,
) -> pd.Series:
    candidates = aggregate.loc[aggregate["recovery"] == 0.0] if strict else aggregate
    minimum = float(candidates["inner_domain_equal_method_mae"].min())
    tied = candidates.loc[
        candidates["inner_domain_equal_method_mae"] <= minimum + TOLERANCE
    ]
    return tied.sort_values(
        ["recovery", "alpha"], ascending=[True, False]
    ).iloc[0]


def nested_source_only_selection(
    outer: pd.DataFrame,
    configurations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth_lookup = outer[
        ["domain", "cell_id", "target_cycle_number", "truth"]
    ].copy()
    domains = sorted(outer["domain"].astype(str).unique())
    alpha_array = configurations["alpha"].to_numpy(float)
    recovery_array = configurations["recovery"].to_numpy(float)
    inner_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    keys = ["domain", "cell_id", "target_cycle_number"]

    for outer_index, outer_domain in enumerate(domains, start=1):
        source_domains = [domain for domain in domains if domain != outer_domain]
        outer_records: list[dict[str, object]] = []
        for inner_domain in source_domains:
            cached = pd.read_parquet(cache_path(outer_domain, inner_domain))
            cached = cached.loc[
                cached["domain"].astype(str) == inner_domain
            ].copy()
            validation = cached.merge(
                truth_lookup.loc[
                    truth_lookup["domain"].astype(str) == inner_domain
                ],
                on=keys,
                how="inner",
                validate="one_to_one",
            )
            if len(validation) != len(cached):
                raise RuntimeError(
                    f"V374 inner alignment failed: outer={outer_domain}, "
                    f"inner={inner_domain}"
                )
            raw, candidate, truth, new_cell = ordered_arrays(validation)
            maes = macro_mae_for_configurations(
                raw,
                candidate,
                truth,
                new_cell,
                alpha_array,
                recovery_array,
                0.01,
            )
            for config, mae in zip(configurations.itertuples(index=False), maes):
                record = {
                    "outer_target_domain": outer_domain,
                    "inner_validation_domain": inner_domain,
                    "alpha": float(config.alpha),
                    "recovery": float(config.recovery),
                    "physical_cells": int(validation["cell_id"].nunique()),
                    "validation_rows": int(len(validation)),
                    "method_cell_macro_mae": float(mae),
                }
                inner_rows.append(record)
                outer_records.append(record)

        aggregate = (
            pd.DataFrame(outer_records)
            .groupby(["alpha", "recovery"], as_index=False)
            .agg(
                inner_domain_equal_method_mae=("method_cell_macro_mae", "mean"),
                inner_domains=("inner_validation_domain", "nunique"),
            )
        )
        selected = select_configuration(aggregate, strict=False)
        strict = select_configuration(aggregate, strict=True)
        selection_rows.append(
            {
                "outer_target_domain": outer_domain,
                "selected_alpha": float(selected["alpha"]),
                "selected_recovery": float(selected["recovery"]),
                "selected_inner_domain_equal_method_mae": float(
                    selected["inner_domain_equal_method_mae"]
                ),
                "strict_selected_alpha": float(strict["alpha"]),
                "strict_selected_inner_domain_equal_method_mae": float(
                    strict["inner_domain_equal_method_mae"]
                ),
                "inner_domains": int(selected["inner_domains"]),
                "outer_target_labels_used_for_selection": False,
            }
        )
        print(
            f"V374 nested selection {outer_index}/{len(domains)}: "
            f"{outer_domain} -> alpha={float(selected['alpha'])}, "
            f"rho={float(selected['recovery'])}",
            flush=True,
        )
    return pd.DataFrame(inner_rows), pd.DataFrame(selection_rows)


def prefix_replay_max_difference(
    rows: pd.DataFrame,
    *,
    alpha: float,
    recovery: float,
    full_state: np.ndarray,
    full_output: np.ndarray,
) -> float:
    maximum = 0.0
    identifiers = rows["cell_id"].astype(str).to_numpy()
    cycles = rows["target_cycle_number"].to_numpy(float)
    raw = rows["raw_baseline"].to_numpy(float)
    candidate = rows["raw_candidate"].to_numpy(float)
    for identifier in np.unique(identifiers):
        positions = np.flatnonzero(identifiers == identifier)
        order = positions[np.argsort(cycles[positions], kind="mergesort")]
        length = max(1, len(order) // 2)
        prefix = order[:length]
        state, output = prefix_causal_bounded_recovery_projection(
            raw[prefix],
            candidate[prefix],
            0.01,
            assimilation=alpha,
            recovery_allowance=recovery,
        )
        maximum = max(
            maximum,
            float(np.max(np.abs(state - full_state[prefix]))),
            float(np.max(np.abs(output - full_output[prefix]))),
        )
    return maximum


def outer_evaluation(
    outer: pd.DataFrame,
    selections: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    selection_map = selections.set_index("outer_target_domain").to_dict("index")
    blocks: list[pd.DataFrame] = []
    certificate = {
        "maximum_absolute_displacement": 0.0,
        "maximum_observed_absolute_loss_regret": -np.inf,
        "maximum_state_recovery_excess": -np.inf,
        "maximum_output_recovery_excess": -np.inf,
        "maximum_prefix_replay_difference": 0.0,
        "maximum_zero_recovery_difference_from_v321": 0.0,
        "minimum_prediction": np.inf,
        "maximum_prediction": -np.inf,
    }

    for domain, source_rows in outer.groupby("domain", sort=True):
        domain_name = str(domain)
        setting = selection_map[domain_name]
        rows = source_rows.copy()
        identifiers = rows["cell_id"].astype(str).to_numpy()
        cycles = rows["target_cycle_number"].to_numpy(float)
        raw = rows["raw_baseline"].to_numpy(float)
        candidate = rows["raw_candidate"].to_numpy(float)
        truth = rows["truth"].to_numpy(float)
        alpha = float(setting["selected_alpha"])
        recovery = float(setting["selected_recovery"])
        strict_alpha = float(setting["strict_selected_alpha"])

        state, method = prefix_causal_bounded_recovery_cellwise_projection(
            identifiers,
            cycles,
            raw,
            candidate,
            0.01,
            assimilation=alpha,
            recovery_allowance=recovery,
        )
        strict_state, strict_method = (
            prefix_causal_bounded_recovery_cellwise_projection(
                identifiers,
                cycles,
                raw,
                candidate,
                0.01,
                assimilation=strict_alpha,
                recovery_allowance=0.0,
            )
        )
        v321_state, v321_method = prefix_causal_cellwise_projection(
            identifiers,
            cycles,
            raw,
            candidate,
            0.01,
            assimilation=strict_alpha,
        )
        certificate["maximum_zero_recovery_difference_from_v321"] = max(
            certificate["maximum_zero_recovery_difference_from_v321"],
            float(np.max(np.abs(strict_state - v321_state))),
            float(np.max(np.abs(strict_method - v321_method))),
        )
        certificate["maximum_absolute_displacement"] = max(
            certificate["maximum_absolute_displacement"],
            float(np.max(np.abs(method - state))),
        )
        certificate["maximum_observed_absolute_loss_regret"] = max(
            certificate["maximum_observed_absolute_loss_regret"],
            float(np.max(np.abs(method - truth) - np.abs(state - truth))),
        )
        certificate["minimum_prediction"] = min(
            certificate["minimum_prediction"], float(np.min(method))
        )
        certificate["maximum_prediction"] = max(
            certificate["maximum_prediction"], float(np.max(method))
        )

        for identifier in np.unique(identifiers):
            positions = np.flatnonzero(identifiers == identifier)
            order = positions[np.argsort(cycles[positions], kind="mergesort")]
            if len(order) > 1:
                certificate["maximum_state_recovery_excess"] = max(
                    certificate["maximum_state_recovery_excess"],
                    float(np.max(np.diff(state[order]) - recovery)),
                )
                certificate["maximum_output_recovery_excess"] = max(
                    certificate["maximum_output_recovery_excess"],
                    float(np.max(np.diff(method[order]) - recovery)),
                )
        certificate["maximum_prefix_replay_difference"] = max(
            certificate["maximum_prefix_replay_difference"],
            prefix_replay_max_difference(
                rows,
                alpha=alpha,
                recovery=recovery,
                full_state=state,
                full_output=method,
            ),
        )

        block = rows[
            [
                "domain",
                "cell_id",
                "target_cycle_number",
                "truth",
                "raw_baseline",
                "raw_candidate",
            ]
        ].copy()
        block["selected_alpha"] = alpha
        block["selected_recovery"] = recovery
        block["bounded_recovery_state"] = state
        block["bounded_recovery_method"] = method
        block["strict_selected_alpha"] = strict_alpha
        block["strict_state"] = strict_state
        block["strict_method"] = strict_method
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True), certificate


def domain_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    roles = (
        "bounded_recovery_state",
        "bounded_recovery_method",
        "strict_state",
        "strict_method",
    )
    blocks: list[pd.DataFrame] = []
    for role in roles:
        working = predictions[["domain", "cell_id", "truth"]].copy()
        working["absolute_error"] = np.abs(
            predictions[role].to_numpy(float) - working["truth"].to_numpy(float)
        )
        cells = (
            working.groupby(["domain", "cell_id"], as_index=False)
            .agg(cell_mae=("absolute_error", "mean"))
        )
        domain = (
            cells.groupby("domain", as_index=False)
            .agg(cell_macro_mae=("cell_mae", "mean"), physical_cells=("cell_id", "nunique"))
        )
        domain["method"] = role
        blocks.append(domain)
    return pd.concat(blocks, ignore_index=True)


def primary_comparison(
    domains: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, object]:
    recovery = domains.loc[
        domains["method"] == "bounded_recovery_method",
        ["domain", "cell_macro_mae"],
    ]
    strict = domains.loc[
        domains["method"] == "strict_method",
        ["domain", "cell_macro_mae"],
    ]
    paired = recovery.merge(
        strict,
        on="domain",
        suffixes=("_recovery", "_strict"),
        validate="one_to_one",
    )
    paired["difference"] = (
        paired["cell_macro_mae_recovery"] - paired["cell_macro_mae_strict"]
    )
    differences = paired["difference"].to_numpy(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(repetitions, len(differences)))
    replicates = differences[indices].mean(axis=1)
    wins = int((differences < -TOLERANCE).sum())
    ties = int((np.abs(differences) <= TOLERANCE).sum())
    losses = int((differences > TOLERANCE).sum())
    return {
        "method": "bounded_recovery_method",
        "baseline": "strict_method",
        "difference_direction": "negative favors bounded recovery",
        "domain_equal_mean_difference": float(differences.mean()),
        "ci95_domain_cluster_percentile": [
            float(np.quantile(replicates, 0.025)),
            float(np.quantile(replicates, 0.975)),
        ],
        "domain_wins_ties_losses": [wins, ties, losses],
        "per_domain": json.loads(paired.to_json(orient="records")),
    }


def run_unit_tests() -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "test_bounded_recovery_pchp_v374.py", "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    match = re.search(r"Ran (\d+) tests", output)
    return {
        "return_code": int(completed.returncode),
        "tests": int(match.group(1)) if match else None,
        "passed": completed.returncode == 0 and match is not None,
    }


def main() -> None:
    started = time.perf_counter()
    prefreeze = verify_prefreeze()
    OUT.mkdir(parents=True, exist_ok=True)
    outer = pd.read_parquet(OUTER_PREDICTIONS)[
        [
            "domain",
            "cell_id",
            "target_cycle_number",
            "truth",
            "raw_baseline",
            "raw_candidate",
        ]
    ].copy()
    if (
        outer["domain"].nunique() != prefreeze["information_boundary"]["outer_domains"]
        or outer["cell_id"].nunique()
        != prefreeze["information_boundary"]["physical_cells"]
        or len(outer) != prefreeze["information_boundary"]["post_reference_rows"]
    ):
        raise RuntimeError("V374 outer roster does not match the prefreeze")

    configurations = configuration_table(prefreeze)
    inner, selections = nested_source_only_selection(outer, configurations)
    if len(selections) != 12 or selections["inner_domains"].min() != 11:
        raise RuntimeError("V374 nested selections are incomplete")
    if selections["outer_target_labels_used_for_selection"].any():
        raise RuntimeError("V374 outer-label isolation failed")

    predictions, certificate = outer_evaluation(outer, selections)
    domains = domain_metrics(predictions)
    comparison = primary_comparison(
        domains,
        repetitions=int(prefreeze["bootstrap"]["repetitions"]),
        seed=int(prefreeze["bootstrap"]["seed"]),
    )
    unit_tests = run_unit_tests()
    positive_recovery_domains = int((selections["selected_recovery"] > 0.0).sum())

    gates = {
        "mean_difference_below_zero": (
            comparison["domain_equal_mean_difference"] < 0.0
        ),
        "ci95_upper_below_zero": (
            comparison["ci95_domain_cluster_percentile"][1] < 0.0
        ),
        "minimum_improved_outer_domains": (
            comparison["domain_wins_ties_losses"][0]
            >= prefreeze["retain_gates"]["minimum_improved_outer_domains"]
        ),
        "minimum_positive_recovery_domains": (
            positive_recovery_domains
            >= prefreeze["retain_gates"][
                "minimum_outer_domains_selecting_positive_recovery"
            ]
        ),
        "absolute_displacement_budget": (
            certificate["maximum_absolute_displacement"] <= 0.01 + TOLERANCE
        ),
        "observed_absolute_loss_regret": (
            certificate["maximum_observed_absolute_loss_regret"]
            <= 0.01 + TOLERANCE
        ),
        "state_recovery_envelope": (
            certificate["maximum_state_recovery_excess"] <= TOLERANCE
        ),
        "output_recovery_envelope": (
            certificate["maximum_output_recovery_excess"] <= TOLERANCE
        ),
        "physical_range": (
            certificate["minimum_prediction"] >= -TOLERANCE
            and certificate["maximum_prediction"] <= 1.3 + TOLERANCE
        ),
        "prefix_replay_exact": (
            certificate["maximum_prefix_replay_difference"] == 0.0
        ),
        "zero_recovery_exactly_recovers_v321": (
            certificate["maximum_zero_recovery_difference_from_v321"] == 0.0
        ),
        "unit_tests": unit_tests["passed"],
    }
    retained = all(gates.values())

    summary = (
        domains.groupby("method", as_index=False)
        .agg(
            domain_equal_cell_macro_mae=("cell_macro_mae", "mean"),
            worst_domain_cell_macro_mae=("cell_macro_mae", "max"),
        )
        .sort_values("domain_equal_cell_macro_mae")
    )
    paths = {
        "inner_metrics": OUT / "bounded_recovery_inner_metrics_v374.csv",
        "selections": OUT / "bounded_recovery_selections_v374.csv",
        "summary": OUT / "bounded_recovery_summary_v374.csv",
        "domain_metrics": OUT / "bounded_recovery_domain_metrics_v374.csv",
        "outer_predictions": OUT / "bounded_recovery_outer_predictions_v374.parquet",
        "report": OUT / "bounded_recovery_pchp_v374_report.json",
    }
    inner.to_csv(paths["inner_metrics"], index=False)
    selections.to_csv(paths["selections"], index=False)
    summary.to_csv(paths["summary"], index=False)
    domains.to_csv(paths["domain_metrics"], index=False)
    predictions.to_parquet(paths["outer_predictions"], index=False)

    report = {
        "version": "v374",
        "status": (
            "BOUNDED_RECOVERY_PCHP_RETAINED"
            if retained
            else "BOUNDED_RECOVERY_PCHP_REJECTED"
        ),
        "decision": "RETAIN" if retained else "REJECT",
        "scope": prefreeze["scope"],
        "prefreeze": {"path": str(PREFREEZE), "sha256": sha256_file(PREFREEZE)},
        "information_boundary": prefreeze["information_boundary"],
        "budget_soh_units": prefreeze["budget_soh_units"],
        "alpha_grid": prefreeze["alpha_grid"],
        "recovery_grid_soh_per_record": prefreeze[
            "recovery_grid_soh_per_record"
        ],
        "selection_rule": {
            "metric": prefreeze["selection_metric"],
            "tie_breaker": prefreeze["tie_breaker"],
            "strict_comparator_selected_independently": True,
            "outer_target_labels_used_for_selection": False,
        },
        "selected_parameters": json.loads(selections.to_json(orient="records")),
        "positive_recovery_outer_domains": positive_recovery_domains,
        "summary": json.loads(summary.to_json(orient="records")),
        "primary_comparison": comparison,
        "deterministic_certificate": certificate,
        "unit_tests": unit_tests,
        "gates": gates,
        "limitations": [
            "All twelve domains were historically opened; this is protocol-locked retrospective development evidence.",
            "The recovery allowance is per emitted record and is not an electrochemical recovery-rate calibration.",
            "No external dataset was rerun and no V368 confirmatory claim is changed by this decision.",
        ],
        "runtime_seconds": float(time.perf_counter() - started),
        "files": {},
    }
    for name, path in paths.items():
        if name != "report":
            report["files"][name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    paths["report"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
