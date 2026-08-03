"""Unseal capacities and score the frozen V344 BaSyTec predictions once."""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from basytec_external_v343_common import (
    ANCHOR_RECORDS,
    CURRENT_ACTIVITY_THRESHOLD_A,
    MINIMUM_POST_ANCHOR_RECORDS,
    detect_schema,
    exact_two_sided_sign_p,
    load_aging_bytes,
    numeric_series,
    read_scoring_fields,
    sha256_file,
    stable_cycle_key,
)


ROOT = Path(__file__).resolve().parent
EXTERNAL = ROOT / "external_basytec_v343"
DOWNLOADS = EXTERNAL / "downloads"
PREDICTION_DIR = EXTERNAL / "label_blind_v344"
PREDICTION_REPORT = PREDICTION_DIR / "basytec_label_blind_prediction_report_v344.json"
OUT = EXTERNAL / "scored_v345"
RECORD_PATH = OUT / "basytec_scored_records_v345.parquet"
CELL_PATH = OUT / "basytec_cell_metrics_v345.csv"
REPORT_PATH = OUT / "basytec_frozen_confirmation_v345_report.json"

BUDGET = 0.01
TOLERANCE = 1e-12
MINIMUM_CONFIRMATORY_CELLS = 36
MINIMUM_WIN_FRACTION = 2.0 / 3.0
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 20260808
REFERENCE_CAPACITY_MIN_AH = 0.08
REFERENCE_CAPACITY_MAX_AH = 0.16


def capacity_map(zip_path: Path) -> tuple[dict[str, float], dict[str, object]]:
    raw, member = load_aging_bytes(zip_path)
    schema = detect_schema(raw)
    frame = read_scoring_fields(raw, schema)
    keys = frame["cycle_raw"].map(stable_cycle_key)
    current = numeric_series(frame["current_raw"])
    capacity = numeric_series(frame["discharge_capacity_raw"])
    mapping: dict[str, float] = {}
    missing = 0
    for key in dict.fromkeys(keys.tolist()):
        positions = np.flatnonzero(keys.to_numpy(str) == key)
        discharge = positions[current[positions] < -CURRENT_ACTIVITY_THRESHOLD_A]
        values = np.abs(capacity[discharge])
        values = values[np.isfinite(values) & (values > 0.0)]
        if len(values):
            mapping[key] = float(values.max())
        else:
            missing += 1
    return mapping, {
        "cell_id": zip_path.stem,
        "zip_name": zip_path.name,
        "aging_member": member,
        "cycles_with_capacity": len(mapping),
        "cycles_without_valid_capacity": missing,
        "capacity_field": schema.discharge_capacity,
    }


def percentile_bootstrap(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.percentile(means, [2.5, 97.5])]


def comparison(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    wins = int((values < -TOLERANCE).sum())
    ties = int((np.abs(values) <= TOLERANCE).sum())
    losses = int((values > TOLERANCE).sum())
    return {
        "cell_equal_mean": float(values.mean()),
        "cell_bootstrap_ci95": percentile_bootstrap(values),
        "wins_ties_losses": [wins, ties, losses],
        "win_fraction_excluding_ties": (
            float(wins / (wins + losses)) if wins + losses else None
        ),
        "exact_two_sided_sign_p_descriptive": exact_two_sided_sign_p(wins, losses),
    }


def main() -> int:
    started = time.perf_counter()
    if not PREDICTION_REPORT.is_file():
        raise FileNotFoundError(PREDICTION_REPORT)
    prediction_report = json.loads(PREDICTION_REPORT.read_text(encoding="utf-8"))
    if prediction_report.get("status") != (
        "BASYTEC_LABEL_BLIND_PREDICTIONS_FROZEN_BEFORE_CAPACITY_ACCESS"
    ):
        raise RuntimeError("prediction freeze report status mismatch")
    if prediction_report.get("capacity_column_values_accessed") is not False:
        raise RuntimeError("prediction stage did not preserve the outcome boundary")

    feature_meta = prediction_report["frozen_outputs"]["features"]
    prediction_meta = prediction_report["frozen_outputs"]["predictions"]
    feature_path = Path(feature_meta["path"])
    prediction_path = Path(prediction_meta["path"])
    if sha256_file(feature_path) != feature_meta["sha256"]:
        raise RuntimeError("frozen feature-table hash mismatch")
    if sha256_file(prediction_path) != prediction_meta["sha256"]:
        raise RuntimeError("frozen prediction-table hash mismatch")
    features = pd.read_parquet(feature_path)
    predictions = pd.read_parquet(prediction_path)

    capacity_rows = []
    capacity_audit = []
    for index, zip_name in enumerate(sorted(features["zip_name"].unique()), start=1):
        mapping, audit = capacity_map(DOWNLOADS / zip_name)
        capacity_audit.append(audit)
        for cycle_key, capacity in mapping.items():
            capacity_rows.append(
                {
                    "cell_id": Path(zip_name).stem,
                    "cycle_key": cycle_key,
                    "discharge_capacity_ah": capacity,
                }
            )
        print(
            f"unsealed capacity-only fields {index}/{features['zip_name'].nunique()}: "
            f"{zip_name}",
            flush=True,
        )
    capacities = pd.DataFrame(capacity_rows)
    feature_labels = features.merge(
        capacities,
        on=["cell_id", "cycle_key"],
        how="left",
        validate="one_to_one",
    )

    eligible_frames = []
    exclusions: Counter[str] = Counter()
    cell_contract = []
    for cell_id, cell in feature_labels.groupby("cell_id", sort=True):
        ordered = cell.sort_values("aligned_cycle_rank", kind="mergesort")
        anchor = ordered.loc[ordered["aligned_cycle_rank"] <= ANCHOR_RECORDS]
        if len(anchor) != ANCHOR_RECORDS or anchor["discharge_capacity_ah"].isna().any():
            exclusions["missing_initial_five_capacity_contract"] += 1
            continue
        reference = float(anchor["discharge_capacity_ah"].median())
        if not (
            math.isfinite(reference)
            and REFERENCE_CAPACITY_MIN_AH <= reference <= REFERENCE_CAPACITY_MAX_AH
        ):
            exclusions["reference_capacity_outside_frozen_physical_range"] += 1
            continue
        post = ordered.loc[ordered["aligned_cycle_rank"] > ANCHOR_RECORDS].copy()
        post = post.loc[post["discharge_capacity_ah"].notna()].copy()
        if len(post) < MINIMUM_POST_ANCHOR_RECORDS:
            exclusions["fewer_than_ten_scorable_post_anchor_cycles"] += 1
            continue
        post["truth"] = post["discharge_capacity_ah"] / reference
        post = post.loc[np.isfinite(post["truth"]) & (post["truth"] > 0.0)].copy()
        if len(post) < MINIMUM_POST_ANCHOR_RECORDS:
            exclusions["fewer_than_ten_positive_finite_soh_targets"] += 1
            continue
        cell_contract.append(
            {
                "cell_id": cell_id,
                "reference_capacity_ah": reference,
                "scorable_post_anchor_cycles": len(post),
            }
        )
        eligible_frames.append(post[["cell_id", "cycle_key", "truth", "discharge_capacity_ah"]])
    labels = (
        pd.concat(eligible_frames, ignore_index=True)
        if eligible_frames
        else pd.DataFrame(columns=["cell_id", "cycle_key", "truth", "discharge_capacity_ah"])
    )
    scored = predictions.merge(
        labels,
        on=["cell_id", "cycle_key"],
        how="inner",
        validate="one_to_one",
    )
    if scored.empty:
        raise RuntimeError("no cells satisfy the frozen label contract")

    truth = scored["truth"].to_numpy(float)
    baseline = scored["causal_baseline_prediction"].to_numpy(float)
    pchp = scored["prefix_causal_harm_capped_prediction"].to_numpy(float)
    comparator = scored["source_tuned_causal_candidate_prediction"].to_numpy(float)
    scored["baseline_abs_error"] = np.abs(baseline - truth)
    scored["pchp_abs_error"] = np.abs(pchp - truth)
    scored["source_tuned_candidate_abs_error"] = np.abs(comparator - truth)
    scored["pchp_loss_regret"] = scored["pchp_abs_error"] - scored["baseline_abs_error"]
    scored["source_tuned_candidate_loss_regret"] = (
        scored["source_tuned_candidate_abs_error"] - scored["baseline_abs_error"]
    )
    scored["pchp_displacement"] = np.abs(pchp - baseline)
    scored["source_tuned_candidate_displacement"] = np.abs(comparator - baseline)

    cell = (
        scored.groupby("cell_id", sort=True)
        .agg(
            records=("truth", "size"),
            reference_capacity_ah=("discharge_capacity_ah", lambda _: np.nan),
            baseline_mae=("baseline_abs_error", "mean"),
            pchp_mae=("pchp_abs_error", "mean"),
            source_tuned_causal_candidate_mae=("source_tuned_candidate_abs_error", "mean"),
            maximum_pchp_loss_regret=("pchp_loss_regret", "max"),
            maximum_pchp_displacement=("pchp_displacement", "max"),
            maximum_source_tuned_candidate_displacement=(
                "source_tuned_candidate_displacement",
                "max",
            ),
        )
        .reset_index()
    )
    reference_map = {item["cell_id"]: item["reference_capacity_ah"] for item in cell_contract}
    cell["reference_capacity_ah"] = cell["cell_id"].map(reference_map)
    cell["pchp_minus_baseline"] = cell["pchp_mae"] - cell["baseline_mae"]
    cell["source_tuned_candidate_minus_baseline"] = (
        cell["source_tuned_causal_candidate_mae"] - cell["baseline_mae"]
    )
    cell["pchp_minus_source_tuned_candidate"] = (
        cell["pchp_mae"] - cell["source_tuned_causal_candidate_mae"]
    )

    comparisons = {
        name: comparison(cell[name].to_numpy(float))
        for name in (
            "pchp_minus_baseline",
            "source_tuned_candidate_minus_baseline",
            "pchp_minus_source_tuned_candidate",
        )
    }
    primary = comparisons["pchp_minus_baseline"]
    execution = len(cell) >= MINIMUM_CONFIRMATORY_CELLS
    effect = float(primary["cell_equal_mean"]) < 0.0
    win_fraction = (
        primary["win_fraction_excluding_ties"] is not None
        and float(primary["win_fraction_excluding_ties"]) >= MINIMUM_WIN_FRACTION
    )
    interval = float(primary["cell_bootstrap_ci95"][1]) < 0.0
    pchp_contract = bool(
        float(scored["pchp_displacement"].max()) <= BUDGET + TOLERANCE
        and float(scored["pchp_loss_regret"].max()) <= BUDGET + TOLERANCE
        and float(cell["pchp_minus_baseline"].max()) <= BUDGET + TOLERANCE
    )
    gates = {
        "execution_at_least_36_cells": execution,
        "cell_equal_mean_pchp_minus_baseline_below_zero": effect,
        "at_least_two_thirds_cells_improve": bool(win_fraction),
        "cell_bootstrap_interval_upper_below_zero": interval,
        "all_harm_contract_checks": pchp_contract,
    }
    if not execution:
        decision = "INCONCLUSIVE"
    elif all(gates.values()):
        decision = "RETAIN"
    else:
        decision = "REJECT"

    comparator_gain = -float(comparisons["source_tuned_candidate_minus_baseline"]["cell_equal_mean"])
    pchp_gain = -float(primary["cell_equal_mean"])
    utility_retention = (
        pchp_gain / comparator_gain if comparator_gain > TOLERANCE else None
    )
    OUT.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(RECORD_PATH, index=False)
    cell.to_csv(CELL_PATH, index=False, float_format="%.17g", lineterminator="\n")
    report = {
        "status": "BASYTEC_FROZEN_EXTERNAL_CONFIRMATION_COMPLETED",
        "decision": decision,
        "generated_at_local": datetime.now().astimezone().isoformat(),
        "chronology": (
            "Predictions were written and SHA-256 frozen before this scorer accessed "
            "any discharge-capacity value. Team-level prior-exposure attestation remains required."
        ),
        "design": {
            "official_cells": int(features["cell_id"].nunique()),
            "eligible_physical_cells": int(len(cell)),
            "scored_cycle_records": int(len(scored)),
            "independent_unit": "physical_cell",
            "nested_unit": "aging_cycle_within_physical_cell",
            "estimand": "cell-equal mean of within-cell MAE differences",
            "target": "cycle discharge capacity divided by the median of the first five structurally eligible cycle capacities",
            "exclusions": dict(sorted(exclusions.items())),
        },
        "accuracy": {
            "cell_equal_baseline_mae": float(cell["baseline_mae"].mean()),
            "cell_equal_pchp_mae": float(cell["pchp_mae"].mean()),
            "cell_equal_source_tuned_candidate_mae": float(
                cell["source_tuned_causal_candidate_mae"].mean()
            ),
            "pchp_gain_over_baseline": pchp_gain,
            "source_tuned_candidate_gain_over_baseline": comparator_gain,
            "pchp_utility_retention_fraction": utility_retention,
            "comparisons": comparisons,
        },
        "contract": {
            "budget_soh_units": BUDGET,
            "maximum_pchp_displacement": float(scored["pchp_displacement"].max()),
            "maximum_observed_pchp_loss_regret": float(scored["pchp_loss_regret"].max()),
            "maximum_cell_macro_pchp_harm": float(cell["pchp_minus_baseline"].max()),
            "maximum_source_tuned_candidate_displacement": float(
                scored["source_tuned_candidate_displacement"].max()
            ),
            "source_tuned_candidate_violating_records": int(
                (scored["source_tuned_candidate_displacement"] > BUDGET + TOLERANCE).sum()
            ),
            "source_tuned_candidate_violating_cells": int(
                cell.groupby("cell_id")["maximum_source_tuned_candidate_displacement"]
                .max()
                .gt(BUDGET + TOLERANCE)
                .sum()
            ),
        },
        "gates": gates,
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "unit": "physical_cell",
            "interval": "95% percentile",
        },
        "capacity_unseal_audit": capacity_audit,
        "artifacts": [
            {"path": str(RECORD_PATH), "sha256": sha256_file(RECORD_PATH)},
            {"path": str(CELL_PATH), "sha256": sha256_file(CELL_PATH)},
            {"path": str(PREDICTION_REPORT), "sha256": sha256_file(PREDICTION_REPORT)},
            {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        ],
        "runtime_seconds": time.perf_counter() - started,
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
