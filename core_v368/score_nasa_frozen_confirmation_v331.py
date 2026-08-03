"""Release NASA discharge capacities once and score frozen causal predictions.

The independent unit is the physical battery.  Cycle-level errors are first
averaged within battery and only then aggregated equally across batteries.
All model predictions and structural eligibility decisions must already exist
in the label-blind V330 artifacts before this script can access outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
EXTERNAL_ROOT = ROOT / "external_nasa_v329"
EXTRACTED = EXTERNAL_ROOT / "extracted"
LABEL_BLIND_ROOT = EXTERNAL_ROOT / "label_blind_v330"
FEATURES = LABEL_BLIND_ROOT / "nasa_label_blind_features_v330.parquet"
PREDICTIONS = LABEL_BLIND_ROOT / "nasa_frozen_predictions_v330.parquet"
PREDICTION_REPORT = LABEL_BLIND_ROOT / "nasa_label_blind_prediction_report_v330.json"
PREFREEZE = ROOT / "paper_q1" / "rccp_nasa_prefreeze_v329.json"
PREDICTION_SCRIPT = ROOT / "build_nasa_label_blind_predictions_v330.py"
OUT = EXTERNAL_ROOT / "scored_v331"

EXPECTED_PREDICTION_SCRIPT_SHA256 = (
    "305D623F711758AD1B4BE0B518F302D468B6D4BAE17F5642EA09F4CE00D11B5C"
)
ANCHOR_RECORDS = 5
MINIMUM_SCORABLE_POST_ANCHOR_RECORDS = 10
MINIMUM_CONFIRMATORY_CELLS = 4
BUDGET = 0.01
TOLERANCE = 1e-12
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 20260801


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def require_hash(path: Path, expected: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"frozen artifact hash mismatch for {path}: {observed} != {expected}"
        )


def as_cycles(container: Any) -> list[Any]:
    cycles = getattr(container, "cycle", None)
    if cycles is None:
        return []
    return np.atleast_1d(cycles).ravel().tolist()


def normalized_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    array = np.asarray(value)
    if array.size == 1:
        return str(array.item()).strip().lower()
    return str(value).strip().lower()


def read_capacity_label(
    row: pd.Series, cache: dict[str, dict[str, Any]]
) -> tuple[float, str | None]:
    source_member = str(row["discharge_source_member"])
    variable_name = str(row["discharge_variable_name"])
    cycle_index = int(row["discharge_cycle_index_zero_based"])
    if source_member not in cache:
        path = EXTRACTED / Path(source_member)
        if not path.exists():
            return math.nan, "discharge_source_member_missing"
        cache[source_member] = loadmat(
            path, squeeze_me=True, struct_as_record=False
        )
    payload = cache[source_member]
    if variable_name not in payload:
        return math.nan, "discharge_variable_missing"
    cycles = as_cycles(payload[variable_name])
    if cycle_index < 0 or cycle_index >= len(cycles):
        return math.nan, "discharge_cycle_index_out_of_range"
    cycle = cycles[cycle_index]
    if normalized_text(getattr(cycle, "type", "")) != "discharge":
        return math.nan, "paired_locator_is_not_discharge"
    data = getattr(cycle, "data", None)
    if data is None or not hasattr(data, "Capacity"):
        return math.nan, "discharge_capacity_field_missing"
    try:
        values = np.asarray(getattr(data, "Capacity"), dtype=float).ravel()
    except (TypeError, ValueError):
        return math.nan, "discharge_capacity_not_numeric"
    finite = values[np.isfinite(values)]
    if finite.size != 1 or finite[0] <= 0.0:
        return math.nan, "discharge_capacity_not_one_positive_scalar"
    return float(finite[0]), None


def release_labels(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {
        "cell_id",
        "aligned_cycle_rank",
        "target_cycle_number",
        "discharge_source_member",
        "discharge_variable_name",
        "discharge_cycle_index_zero_based",
    }
    missing = sorted(required.difference(features.columns))
    if missing:
        raise RuntimeError(f"label-blind features missing locator columns: {missing}")
    cache: dict[str, dict[str, Any]] = {}
    capacities: list[float] = []
    reasons: list[str | None] = []
    rejection_counts: Counter[str] = Counter()
    for _, row in features.iterrows():
        capacity, reason = read_capacity_label(row, cache)
        capacities.append(capacity)
        reasons.append(reason)
        if reason is not None:
            rejection_counts[reason] += 1
    released = features[
        [
            "domain",
            "cell_id",
            "aligned_cycle_rank",
            "target_cycle_number",
            "discharge_source_member",
            "discharge_variable_name",
            "discharge_cycle_index_zero_based",
        ]
    ].copy()
    released["capacity_ah"] = capacities
    released["label_release_reason"] = reasons

    reference_rows: list[dict[str, object]] = []
    valid_reference_cells: set[str] = set()
    for cell_id, cell in released.groupby("cell_id", sort=False):
        anchor = cell.loc[cell["aligned_cycle_rank"] <= ANCHOR_RECORDS]
        valid = bool(
            len(anchor) == ANCHOR_RECORDS
            and anchor["capacity_ah"].notna().all()
            and np.isfinite(anchor["capacity_ah"].to_numpy(float)).all()
            and (anchor["capacity_ah"] > 0.0).all()
        )
        reference = (
            float(anchor["capacity_ah"].median()) if valid else math.nan
        )
        if valid:
            valid_reference_cells.add(str(cell_id))
        reference_rows.append(
            {
                "cell_id": str(cell_id),
                "initial5_reference_capacity_ah": reference,
                "initial5_reference_valid": valid,
            }
        )
    references = pd.DataFrame(reference_rows)
    released = released.merge(references, on="cell_id", validate="many_to_one")
    released["soh_initial5"] = (
        released["capacity_ah"] / released["initial5_reference_capacity_ah"]
    )
    audit = {
        "capacity_rejection_counts": dict(sorted(rejection_counts.items())),
        "cells_with_valid_initial5_reference": len(valid_reference_cells),
        "mat_files_loaded_for_label_release": len(cache),
    }
    return released, audit


def nonincreasing_by_cell(frame: pd.DataFrame, column: str) -> bool:
    for _, cell in frame.groupby("cell_id", sort=False):
        ordered = cell.sort_values("target_cycle_number", kind="mergesort")
        if (np.diff(ordered[column].to_numpy(float)) > TOLERANCE).any():
            return False
    return True


def exact_two_sided_sign_pvalue(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(trials, k) for k in range(tail + 1)) / (2**trials)
    return min(1.0, 2.0 * probability)


def bootstrap_mean_ci(differences: np.ndarray) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    batch = 10_000
    for start in range(0, BOOTSTRAP_REPLICATES, batch):
        stop = min(start + batch, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def main() -> None:
    started = time.perf_counter()
    for path in (FEATURES, PREDICTIONS, PREDICTION_REPORT, PREFREEZE, PREDICTION_SCRIPT):
        if not path.exists():
            raise FileNotFoundError(path)
    require_hash(PREDICTION_SCRIPT, EXPECTED_PREDICTION_SCRIPT_SHA256)
    prefreeze = json.loads(PREFREEZE.read_text(encoding="utf-8"))
    if prefreeze.get("status") != "FROZEN_BEFORE_NASA_ARCHIVE_DOWNLOAD":
        raise RuntimeError("NASA prefreeze status mismatch")
    prediction_report = json.loads(PREDICTION_REPORT.read_text(encoding="utf-8"))
    if prediction_report.get("status") != (
        "NASA_LABEL_BLIND_PREDICTIONS_FROZEN_BEFORE_OUTCOME_ACCESS"
    ):
        raise RuntimeError("label-blind prediction status mismatch")
    expected_features_hash = prediction_report["frozen_outputs"]["features"]["sha256"]
    expected_predictions_hash = prediction_report["frozen_outputs"]["predictions"]["sha256"]
    require_hash(FEATURES, expected_features_hash)
    require_hash(PREDICTIONS, expected_predictions_hash)

    features = pd.read_parquet(FEATURES)
    predictions = pd.read_parquet(PREDICTIONS)
    released, label_audit = release_labels(features)
    post = released.loc[released["aligned_cycle_rank"] > ANCHOR_RECORDS].copy()
    keys = ["domain", "cell_id", "aligned_cycle_rank", "target_cycle_number"]
    scored = predictions.merge(
        post[keys + ["capacity_ah", "initial5_reference_capacity_ah", "soh_initial5"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    valid_label = (
        np.isfinite(scored["soh_initial5"].to_numpy(float))
        & (scored["soh_initial5"].to_numpy(float) > 0.0)
    )
    scored = scored.loc[valid_label].copy()

    counts = scored.groupby("cell_id").size()
    eligible_cells = set(
        counts.loc[counts >= MINIMUM_SCORABLE_POST_ANCHOR_RECORDS].index.astype(str)
    )
    scored = scored.loc[scored["cell_id"].astype(str).isin(eligible_cells)].copy()
    if scored.empty:
        cell_metrics = pd.DataFrame(
            columns=[
                "cell_id",
                "prediction_rows",
                "baseline_cell_mae",
                "method_cell_mae",
                "method_minus_baseline",
            ]
        )
    else:
        truth = scored["soh_initial5"].to_numpy(float)
        scored["baseline_absolute_error"] = np.abs(
            scored["causal_baseline_prediction"].to_numpy(float) - truth
        )
        scored["method_absolute_error"] = np.abs(
            scored["prefix_causal_harm_capped_prediction"].to_numpy(float) - truth
        )
        scored["raw_baseline_absolute_error"] = np.abs(
            scored["raw_baseline_prediction"].to_numpy(float) - truth
        )
        scored["raw_candidate_absolute_error"] = np.abs(
            scored["raw_candidate_prediction"].to_numpy(float) - truth
        )
        cell_metrics = (
            scored.groupby("cell_id", as_index=False)
            .agg(
                prediction_rows=("soh_initial5", "size"),
                baseline_cell_mae=("baseline_absolute_error", "mean"),
                method_cell_mae=("method_absolute_error", "mean"),
                raw_baseline_cell_mae=("raw_baseline_absolute_error", "mean"),
                raw_candidate_cell_mae=("raw_candidate_absolute_error", "mean"),
            )
        )
        cell_metrics["method_minus_baseline"] = (
            cell_metrics["method_cell_mae"] - cell_metrics["baseline_cell_mae"]
        )

    independent_cells = int(len(cell_metrics))
    execution_gate = independent_cells >= MINIMUM_CONFIRMATORY_CELLS
    if independent_cells:
        differences = cell_metrics["method_minus_baseline"].to_numpy(float)
        wins = int((differences < -TOLERANCE).sum())
        ties = int((np.abs(differences) <= TOLERANCE).sum())
        losses = int((differences > TOLERANCE).sum())
        mean_difference = float(differences.mean())
        maximum_cell_harm = float(differences.max())
        improvement_fraction = float(wins / independent_cells)
        ci_lower, ci_upper = bootstrap_mean_ci(differences)
        sign_p = exact_two_sided_sign_pvalue(wins, losses)
        displacement = np.abs(
            scored["prefix_causal_harm_capped_prediction"].to_numpy(float)
            - scored["causal_baseline_prediction"].to_numpy(float)
        )
        observed_regret = (
            scored["method_absolute_error"].to_numpy(float)
            - scored["baseline_absolute_error"].to_numpy(float)
        )
        certificate = {
            "maximum_absolute_displacement": float(displacement.max()),
            "maximum_observed_absolute_loss_regret": float(observed_regret.max()),
            "baseline_trajectories_nonincreasing": nonincreasing_by_cell(
                scored, "causal_baseline_prediction"
            ),
            "method_trajectories_nonincreasing": nonincreasing_by_cell(
                scored, "prefix_causal_harm_capped_prediction"
            ),
        }
        certificate["deterministic_certificate_numerically_verified"] = bool(
            certificate["maximum_absolute_displacement"] <= BUDGET + TOLERANCE
            and certificate["maximum_observed_absolute_loss_regret"]
            <= BUDGET + TOLERANCE
            and certificate["baseline_trajectories_nonincreasing"]
            and certificate["method_trajectories_nonincreasing"]
        )
    else:
        wins = ties = losses = 0
        mean_difference = maximum_cell_harm = improvement_fraction = math.nan
        ci_lower = ci_upper = sign_p = math.nan
        certificate = {
            "deterministic_certificate_numerically_verified": False,
            "reason": "no confirmatory cells",
        }

    primary_gates = {
        "minimum_four_independent_cells": execution_gate,
        "cell_equal_mean_method_minus_baseline_below_zero": bool(
            execution_gate and mean_difference < 0.0
        ),
        "at_least_75_percent_cells_improve": bool(
            execution_gate and improvement_fraction >= 0.75
        ),
        "maximum_cell_macro_harm_at_most_budget": bool(
            execution_gate and maximum_cell_harm <= BUDGET + TOLERANCE
        ),
        "deterministic_certificate_verified": bool(
            certificate["deterministic_certificate_numerically_verified"]
        ),
    }
    if not execution_gate:
        status = "NASA_ONE_SHOT_EXTERNAL_GATE_INCONCLUSIVE"
        decision = "INCONCLUSIVE"
    elif all(primary_gates.values()):
        status = "NASA_ONE_SHOT_EXTERNAL_GATE_PASSED"
        decision = "RETAIN"
    else:
        status = "NASA_ONE_SHOT_EXTERNAL_GATE_NOT_PASSED"
        decision = "REJECT"

    OUT.mkdir(parents=True, exist_ok=True)
    released_path = OUT / "nasa_released_labels_v331.parquet"
    scored_path = OUT / "nasa_scored_predictions_v331.parquet"
    cells_path = OUT / "nasa_cell_metrics_v331.csv"
    report_path = OUT / "nasa_frozen_confirmation_v331_report.json"
    released.to_parquet(released_path, index=False)
    scored.to_parquet(scored_path, index=False)
    cell_metrics.to_csv(cells_path, index=False)
    report = {
        "status": status,
        "decision": decision,
        "generated_at_local": datetime.now().astimezone().isoformat(),
        "external_outcomes_accessed_in_this_stage": True,
        "independent_unit": "physical battery",
        "nested_measurements": "post-anchor charge-discharge records within battery",
        "primary_estimand": (
            "cell-equal mean of paired within-cell MAE difference: prefix-causal "
            "harm-capped method minus its causal protected baseline"
        ),
        "independent_cells": independent_cells,
        "scored_prediction_rows": int(len(scored)),
        "minimum_scorable_post_anchor_records_per_cell": (
            MINIMUM_SCORABLE_POST_ANCHOR_RECORDS
        ),
        "cell_equal_baseline_mae": (
            float(cell_metrics["baseline_cell_mae"].mean())
            if independent_cells
            else None
        ),
        "cell_equal_method_mae": (
            float(cell_metrics["method_cell_mae"].mean())
            if independent_cells
            else None
        ),
        "cell_equal_method_minus_baseline": (
            mean_difference if independent_cells else None
        ),
        "cell_wins_ties_losses": [wins, ties, losses],
        "cell_improvement_fraction": (
            improvement_fraction if independent_cells else None
        ),
        "maximum_cell_macro_harm": (
            maximum_cell_harm if independent_cells else None
        ),
        "cell_cluster_percentile_bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "ci95": (
                [ci_lower, ci_upper] if independent_cells else [None, None]
            ),
            "confirmatory_gate": False,
        },
        "exact_two_sided_sign_test": {
            "p_value": sign_p if independent_cells else None,
            "ties_excluded": ties,
            "confirmatory_gate": False,
        },
        "primary_gates": primary_gates,
        "certificate": certificate,
        "label_audit": label_audit,
        "no_post_outcome_rescue_permitted": True,
        "frozen_inputs": {
            "prefreeze": {"path": str(PREFREEZE), "sha256": sha256_file(PREFREEZE)},
            "prediction_report": {
                "path": str(PREDICTION_REPORT),
                "sha256": sha256_file(PREDICTION_REPORT),
            },
            "features": {"path": str(FEATURES), "sha256": sha256_file(FEATURES)},
            "predictions": {
                "path": str(PREDICTIONS),
                "sha256": sha256_file(PREDICTIONS),
            },
            "scoring_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "outputs": {
            "released_labels": {
                "path": str(released_path),
                "sha256": sha256_file(released_path),
            },
            "scored_predictions": {
                "path": str(scored_path),
                "sha256": sha256_file(scored_path),
            },
            "cell_metrics": {
                "path": str(cells_path),
                "sha256": sha256_file(cells_path),
            },
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{status}: {decision}", flush=True)
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
