"""Three-arm monotonicity-contract ablation on frozen development predictions.

The pointwise-tube arm removes only the previous-output cap from strict PCHP,
holding the strict protected state, raw candidate, range and harm budget fixed.
The bounded-recovery arm is the already frozen V374 alternative contract and
is reported separately because it also permits bounded state recovery.
"""

from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "bounded_recovery_pchp_v374" / "bounded_recovery_outer_predictions_v374.parquet"
V327 = ROOT / "nested_prefix_causal_outer_v327" / "nested_outer_predictions_v327.parquet"
PROTOCOL = ROOT / "MONOTONICITY_ABLATION_PROTOCOL_V385.json"
OUT = ROOT / "monotonicity_ablation_v385"
EXPECTED_INPUT_SHA256 = "c7c0045abfa8fdaa862604a273fe1da8f4409e0741148a74c6220f2212c11bdb"
EXPECTED_COUNTS = {"domains": 12, "cells": 586, "records": 601932}
METHODS = ("pointwise_tube", "strict_monotone", "bounded_recovery")
DELTA = 0.01
LOWER = 0.0
UPPER = 1.3
TOLERANCE = 1e-12
BOOTSTRAP_REPETITIONS = 100_000
BOOTSTRAP_SEED = 20_260_806


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate() -> pd.DataFrame:
    if sha256_file(INPUT) != EXPECTED_INPUT_SHA256:
        raise RuntimeError("V374 prediction hash mismatch")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["input"]["sha256"] != EXPECTED_INPUT_SHA256:
        raise RuntimeError("protocol input hash mismatch")
    frame = pd.read_parquet(INPUT)
    required = {
        "domain",
        "cell_id",
        "target_cycle_number",
        "truth",
        "raw_candidate",
        "strict_state",
        "strict_method",
        "bounded_recovery_state",
        "bounded_recovery_method",
        "selected_recovery",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"missing V374 columns: {sorted(missing)}")
    counts = {
        "domains": int(frame["domain"].nunique()),
        "cells": int(frame[["domain", "cell_id"]].drop_duplicates().shape[0]),
        "records": int(len(frame)),
    }
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"unexpected development roster: {counts}")
    numeric = list(required.difference({"domain", "cell_id"}))
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise RuntimeError("non-finite V374 inputs")
    return frame


def add_pointwise_arm(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    lower = np.maximum(LOWER, result["strict_state"].to_numpy(float) - DELTA)
    upper = np.minimum(UPPER, result["strict_state"].to_numpy(float) + DELTA)
    result["pointwise_tube"] = np.clip(
        result["raw_candidate"].to_numpy(float), lower, upper
    )
    result["strict_monotone"] = result["strict_method"].to_numpy(float)
    result["bounded_recovery"] = result["bounded_recovery_method"].to_numpy(float)
    return result


def score(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cell_blocks: list[pd.DataFrame] = []
    for method in METHODS:
        work = frame[["domain", "cell_id", "truth"]].copy()
        work["absolute_error"] = np.abs(
            frame[method].to_numpy(float) - frame["truth"].to_numpy(float)
        )
        cells = (
            work.groupby(["domain", "cell_id"], as_index=False)
            .agg(cell_mae=("absolute_error", "mean"))
        )
        cells["method"] = method
        cell_blocks.append(cells)
    cell_metrics = pd.concat(cell_blocks, ignore_index=True)
    domain_metrics = (
        cell_metrics.groupby(["domain", "method"], as_index=False)
        .agg(
            cell_macro_mae=("cell_mae", "mean"),
            physical_cells=("cell_id", "nunique"),
        )
    )
    summary = (
        domain_metrics.groupby("method", as_index=False)
        .agg(
            domain_equal_cell_macro_mae=("cell_macro_mae", "mean"),
            worst_domain_cell_macro_mae=("cell_macro_mae", "max"),
        )
        .sort_values("domain_equal_cell_macro_mae")
        .reset_index(drop=True)
    )
    return cell_metrics, domain_metrics, summary


def pairwise_effect(
    domain_metrics: pd.DataFrame,
    method: str,
    reference: str,
) -> dict[str, object]:
    wide = domain_metrics.pivot(index="domain", columns="method", values="cell_macro_mae")
    difference = (wide[method] - wide[reference]).sort_index()
    values = difference.to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPETITIONS, len(values)))
    bootstrap = values[draws].mean(axis=1)
    signs = np.asarray(list(product((-1.0, 1.0), repeat=len(values))), dtype=float)
    permutation = (signs * values[None, :]).mean(axis=1)
    observed = float(values.mean())
    return {
        "method": method,
        "reference": reference,
        "difference_direction": f"MAE({method}) - MAE({reference})",
        "domain_equal_mean_difference": observed,
        "median_domain_difference": float(np.median(values)),
        "domain_bootstrap_95_percent_interval": np.quantile(
            bootstrap, [0.025, 0.975]
        ).tolist(),
        "exact_two_sided_sign_flip_p": float(
            np.mean(np.abs(permutation) >= abs(observed) - 1e-15)
        ),
        "exact_assignments": int(len(permutation)),
        "domain_wins_ties_losses_for_method": [
            int(np.sum(values < -TOLERANCE)),
            int(np.sum(np.abs(values) <= TOLERANCE)),
            int(np.sum(values > TOLERANCE)),
        ],
        "per_domain_difference": difference.to_dict(),
    }


def transition_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in METHODS:
        positive_transitions = 0
        total_transitions = 0
        affected_cells = 0
        maximum_increase = 0.0
        recovery_excess = -np.inf
        for (_, _), cell in frame.groupby(["domain", "cell_id"], sort=False):
            ordered = cell.sort_values("target_cycle_number", kind="mergesort")
            increments = np.diff(ordered[method].to_numpy(float))
            if increments.size == 0:
                continue
            positive = increments > TOLERANCE
            positive_transitions += int(positive.sum())
            total_transitions += int(len(increments))
            affected_cells += int(positive.any())
            maximum_increase = max(maximum_increase, float(np.max(increments)))
            if method == "bounded_recovery":
                allowance = ordered["selected_recovery"].to_numpy(float)[1:]
                recovery_excess = max(
                    recovery_excess, float(np.max(increments - allowance))
                )
        rows.append(
            {
                "method": method,
                "positive_output_transitions": positive_transitions,
                "total_output_transitions": total_transitions,
                "fraction_positive_transitions": positive_transitions / total_transitions,
                "affected_cells": affected_cells,
                "fraction_affected_cells": affected_cells / EXPECTED_COUNTS["cells"],
                "maximum_output_increase": maximum_increase,
                "maximum_recovery_envelope_excess": (
                    recovery_excess if method == "bounded_recovery" else None
                ),
            }
        )
    return pd.DataFrame(rows)


def deterministic_certificates(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    references = {
        "pointwise_tube": "strict_state",
        "strict_monotone": "strict_state",
        "bounded_recovery": "bounded_recovery_state",
    }
    reports: dict[str, dict[str, object]] = {}
    truth = frame["truth"].to_numpy(float)
    for method, reference in references.items():
        prediction = frame[method].to_numpy(float)
        state = frame[reference].to_numpy(float)
        displacement = np.abs(prediction - state)
        regret = np.abs(prediction - truth) - np.abs(state - truth)
        reports[method] = {
            "protected_reference": reference,
            "maximum_absolute_displacement": float(np.max(displacement)),
            "maximum_observed_absolute_loss_increase": float(np.max(regret)),
            "minimum_prediction": float(np.min(prediction)),
            "maximum_prediction": float(np.max(prediction)),
            "harm_budget_passed": bool(
                np.max(displacement) <= DELTA + TOLERANCE
                and np.max(regret) <= DELTA + TOLERANCE
            ),
            "range_passed": bool(
                np.min(prediction) >= LOWER - TOLERANCE
                and np.max(prediction) <= UPPER + TOLERANCE
            ),
        }
    return reports


def strict_reproduction_error(frame: pd.DataFrame) -> float:
    keys = ["domain", "cell_id", "target_cycle_number", "truth"]
    reference = pd.read_parquet(V327)[keys + ["selected_causal_method"]]
    current = frame[keys + ["strict_monotone"]]
    merged = current.merge(reference, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(frame):
        raise RuntimeError("V327 strict roster mismatch")
    return float(
        np.max(np.abs(merged["strict_monotone"] - merged["selected_causal_method"]))
    )


def main() -> None:
    frame = add_pointwise_arm(load_and_validate())
    cell_metrics, domain_metrics, summary = score(frame)
    transitions = transition_metrics(frame)
    comparisons = {
        "strict_minus_pointwise": pairwise_effect(
            domain_metrics, "strict_monotone", "pointwise_tube"
        ),
        "bounded_minus_strict": pairwise_effect(
            domain_metrics, "bounded_recovery", "strict_monotone"
        ),
        "bounded_minus_pointwise": pairwise_effect(
            domain_metrics, "bounded_recovery", "pointwise_tube"
        ),
    }
    certificates = deterministic_certificates(frame)
    reproduction_error = strict_reproduction_error(frame)

    OUT.mkdir(parents=True, exist_ok=True)
    prediction_path = OUT / "monotonicity_ablation_predictions_v385.parquet"
    cell_path = OUT / "monotonicity_ablation_cell_metrics_v385.csv"
    domain_path = OUT / "monotonicity_ablation_domain_metrics_v385.csv"
    summary_path = OUT / "monotonicity_ablation_summary_v385.csv"
    transition_path = OUT / "monotonicity_ablation_transitions_v385.csv"
    report_path = OUT / "monotonicity_ablation_v385_report.json"

    frame[
        [
            "domain",
            "cell_id",
            "target_cycle_number",
            "truth",
            "strict_state",
            "bounded_recovery_state",
            "selected_recovery",
            *METHODS,
        ]
    ].to_parquet(prediction_path, index=False)
    cell_metrics.to_csv(cell_path, index=False)
    domain_metrics.to_csv(domain_path, index=False)
    summary.to_csv(summary_path, index=False)
    transitions.to_csv(transition_path, index=False)

    report = {
        "version": "v385",
        "status": "MONOTONICITY_ABLATION_COMPLETE",
        "classification": json.loads(PROTOCOL.read_text(encoding="utf-8"))[
            "classification"
        ],
        "information_boundary": {
            **EXPECTED_COUNTS,
            "independent_unit": "complete held-out development dataset domain",
            "nested_units": ["physical cell", "record"],
            "refitting": False,
            "all_development_outcomes_historically_opened": True,
        },
        "summary": summary.to_dict("records"),
        "comparisons": comparisons,
        "transition_metrics": transitions.to_dict("records"),
        "deterministic_certificates": certificates,
        "strict_v327_maximum_reproduction_error": reproduction_error,
        "interpretation_contract": (
            "The strict-minus-pointwise comparison isolates the output-order constraint. "
            "The bounded-recovery comparison is an alternative state-and-output contract, "
            "not a single-component deletion."
        ),
        "artifacts": {},
    }
    artifacts = {
        "protocol": PROTOCOL,
        "implementation": Path(__file__).resolve(),
        "input": INPUT,
        "predictions": prediction_path,
        "cell_metrics": cell_path,
        "domain_metrics": domain_path,
        "summary": summary_path,
        "transitions": transition_path,
    }
    report["artifacts"] = {
        name: {"path": str(path), "sha256": sha256_file(path), "bytes": int(path.stat().st_size)}
        for name, path in artifacts.items()
    }
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
