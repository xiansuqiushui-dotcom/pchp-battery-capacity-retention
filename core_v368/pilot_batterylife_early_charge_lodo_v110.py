"""Leave-one-domain-out viability pilot for the future-safe SOH task."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "batterylife_early_charge_soh_v109.parquet"
DEFAULT_JSON = ROOT / "batterylife_early_charge_lodo_v110_results.json"
DEFAULT_METRICS = ROOT / "batterylife_early_charge_lodo_v110_metrics.csv"
SEED = 20_260_730
TARGET = "soh_initial5"

FORBIDDEN_MODEL_COLUMNS = {
    "capacity_ah",
    "soh_nominal",
    "soh_initial5",
    "initial5_reference_capacity_ah",
    "member_sha256",
    "source_member",
    "cell_id",
    "domain",
    "feature_cycle_number",
    "early_charge_points",
}
STATIC_NUMERIC_FEATURES = (
    "target_cycle_number",
    "charge_rate_c_metadata",
    "discharge_rate_c_metadata",
    "nominal_capacity_ah",
    "max_voltage_limit_v",
    "min_voltage_limit_v",
    "cycling_soc_lower",
    "cycling_soc_upper",
    "full_soc_window",
)
EARLY_CHARGE_FEATURES = (
    "early_charge_delta_q_per_nominal",
    "early_charge_voltage_mean",
    "early_charge_voltage_std",
    "early_charge_voltage_slope_per_hour",
    "early_charge_current_c_mean",
    "early_charge_current_c_std",
    "early_charge_current_c_abs_mean",
    "temperature_available",
    "early_charge_temperature_mean",
    "early_charge_temperature_std",
    "voltage_at_0s",
    "voltage_at_60s",
    "voltage_at_120s",
    "voltage_at_180s",
    "voltage_at_240s",
    "voltage_at_300s",
    "voltage_at_360s",
    "voltage_at_420s",
    "voltage_at_480s",
    "voltage_at_540s",
    "voltage_at_600s",
    "current_c_at_0s",
    "current_c_at_300s",
    "current_c_at_600s",
)
FULL_FEATURES = STATIC_NUMERIC_FEATURES + EARLY_CHARGE_FEATURES


def validate_feature_contract(columns: Sequence[str]) -> None:
    selected = tuple(map(str, columns))
    overlap = sorted(set(selected) & FORBIDDEN_MODEL_COLUMNS)
    if overlap:
        raise ValueError(f"model feature list contains forbidden columns: {overlap}")
    if len(set(selected)) != len(selected):
        raise ValueError("model feature list contains duplicates")


def balanced_source_rows(
    frame: pd.DataFrame,
    *,
    maximum_rows_per_cell: int,
) -> pd.DataFrame:
    limit = int(maximum_rows_per_cell)
    if limit <= 0:
        raise ValueError("maximum_rows_per_cell must be positive")
    pieces: list[pd.DataFrame] = []
    for _, cell in frame.groupby("cell_id", sort=True):
        ordered = cell.sort_values("target_cycle_number")
        if len(ordered) <= limit:
            pieces.append(ordered)
            continue
        positions = np.linspace(0, len(ordered) - 1, limit)
        indices = np.unique(np.rint(positions).astype(int))
        pieces.append(ordered.iloc[indices])
    sampled = pd.concat(pieces, ignore_index=True)
    if sampled.groupby("cell_id").size().max() > limit:
        raise RuntimeError("cell-balanced sampling exceeded its cap")
    if sampled["cell_id"].nunique() != frame["cell_id"].nunique():
        raise RuntimeError("cell-balanced sampling dropped a physical cell")
    return sampled


def feature_matrix(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> np.ndarray:
    validate_feature_contract(columns)
    matrix = frame.loc[:, list(columns)].astype(float).to_numpy()
    cycle_index = list(columns).index("target_cycle_number")
    if np.any(matrix[:, cycle_index] < 0.0):
        raise ValueError("negative target cycle number")
    matrix[:, cycle_index] = np.log1p(matrix[:, cycle_index])
    return matrix


def metrics_by_physical_cell(
    frame: pd.DataFrame,
    prediction: np.ndarray,
) -> dict[str, float | int]:
    truth = frame[TARGET].to_numpy(float)
    predicted = np.asarray(prediction, dtype=float)
    if predicted.shape != truth.shape or np.any(~np.isfinite(predicted)):
        raise ValueError("predictions are nonfinite or shape-incompatible")
    error = predicted - truth
    working = frame[["cell_id"]].copy()
    working["absolute_error"] = np.abs(error)
    working["squared_error"] = error**2
    cells = working.groupby("cell_id", sort=False).agg(
        mae=("absolute_error", "mean"),
        mse=("squared_error", "mean"),
    )
    operational = (truth >= 0.7) & (truth <= 1.1)
    return {
        "cycle_weighted_mae": float(np.mean(np.abs(error))),
        "cycle_weighted_rmse": float(np.sqrt(np.mean(error**2))),
        "cell_macro_mae": float(cells["mae"].mean()),
        "cell_macro_rmse": float(np.sqrt(cells["mse"]).mean()),
        "worst_cell_mae": float(cells["mae"].max()),
        "operational_0p7_to_1p1_mae": (
            float(np.mean(np.abs(error[operational])))
            if operational.any()
            else math.nan
        ),
        "test_rows": int(len(frame)),
        "test_cells": int(frame["cell_id"].nunique()),
    }


def run_pilot(
    data_path: Path,
    *,
    maximum_rows_per_source_cell: int = 200,
    trees: int = 160,
    seed: int = SEED,
) -> tuple[dict[str, object], pd.DataFrame]:
    started = time.perf_counter()
    frame = pd.read_parquet(data_path)
    frame = frame.loc[frame["after_initial5_reference_window"]].copy()
    required = set(FULL_FEATURES) | {
        "domain",
        "cell_id",
        TARGET,
        "after_initial5_reference_window",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"SOH table lacks required columns: {missing}")
    if frame.empty or frame["domain"].nunique() < 3:
        raise ValueError("SOH table does not contain enough target domains")
    validate_feature_contract(FULL_FEATURES)

    model_specs = {
        "cycle_only": ("target_cycle_number",),
        "early_charge_without_static": (
            "target_cycle_number",
            *EARLY_CHARGE_FEATURES,
        ),
        "early_charge_with_static": FULL_FEATURES,
    }
    metric_rows: list[dict[str, object]] = []
    fold_metadata: list[dict[str, object]] = []
    for target_domain in sorted(frame["domain"].unique()):
        source = frame.loc[frame["domain"] != target_domain].copy()
        target = frame.loc[frame["domain"] == target_domain].copy()
        source_fit = balanced_source_rows(
            source,
            maximum_rows_per_cell=maximum_rows_per_source_cell,
        )
        fold_metadata.append(
            {
                "target_domain": str(target_domain),
                "source_domains": sorted(source["domain"].unique()),
                "source_cells": int(source["cell_id"].nunique()),
                "source_rows": int(len(source)),
                "source_fit_rows": int(len(source_fit)),
                "target_cells": int(target["cell_id"].nunique()),
                "target_rows": int(len(target)),
            }
        )
        for model_name, columns in model_specs.items():
            source_matrix = feature_matrix(source_fit, columns)
            target_matrix = feature_matrix(target, columns)
            imputer = SimpleImputer(strategy="median", add_indicator=True)
            source_imputed = imputer.fit_transform(source_matrix)
            target_imputed = imputer.transform(target_matrix)
            scaler = StandardScaler()
            source_scaled = scaler.fit_transform(source_imputed)
            target_scaled = scaler.transform(target_imputed)
            model = ExtraTreesRegressor(
                n_estimators=int(trees),
                min_samples_leaf=20,
                max_features=0.8,
                random_state=int(seed),
                n_jobs=1,
            )
            model.fit(source_scaled, source_fit[TARGET].to_numpy(float))
            prediction = model.predict(target_scaled)
            metric_rows.append(
                {
                    "target_domain": str(target_domain),
                    "model": model_name,
                    **metrics_by_physical_cell(target, prediction),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    comparison: list[dict[str, object]] = []
    for target_domain, group in metrics.groupby("target_domain", sort=True):
        baseline = float(
            group.loc[
                group["model"] == "cycle_only",
                "cell_macro_mae",
            ].iloc[0]
        )
        proposed = float(
            group.loc[
                group["model"] == "early_charge_with_static",
                "cell_macro_mae",
            ].iloc[0]
        )
        comparison.append(
            {
                "target_domain": str(target_domain),
                "cycle_only_cell_macro_mae": baseline,
                "early_charge_cell_macro_mae": proposed,
                "relative_improvement_percent": (
                    100.0 * (baseline - proposed) / baseline
                ),
            }
        )
    results: dict[str, object] = {
        "status": "EIGHT_DOMAIN_EARLY_CHARGE_TASK_VIABILITY_PILOT",
        "claim_boundary": {
            "allowed": (
                "The first 600 seconds of charge contain transferable signal "
                "for retrospective leave-one-domain-out relative-capacity "
                "estimation under a commissioning-reference assumption."
            ),
            "forbidden": [
                "submission readiness",
                "online deployment",
                "causal effect of capacity checks",
                "independence of cycle rows",
                "method superiority before target-label acquisition is tested",
            ],
        },
        "protocol": {
            "outer_evaluation": "leave one complete BatteryLife domain out",
            "target": TARGET,
            "commissioning_reference": (
                "median of the first five aligned discharge capacities per "
                "physical cell; never supplied as a model feature"
            ),
            "evaluation_rows": "aligned cycles after the first five",
            "source_balance": (
                "at most the frozen number of evenly spaced cycle rows per "
                "physical source cell"
            ),
            "maximum_rows_per_source_cell": int(maximum_rows_per_source_cell),
            "trees": int(trees),
            "seed": int(seed),
            "features": {
                key: list(value) for key, value in model_specs.items()
            },
            "forbidden_model_columns": sorted(FORBIDDEN_MODEL_COLUMNS),
        },
        "data": {
            "path": str(data_path.resolve()),
            "domains": sorted(frame["domain"].unique()),
            "domain_count": int(frame["domain"].nunique()),
            "physical_cells": int(frame["cell_id"].nunique()),
            "post_reference_cycle_rows": int(len(frame)),
        },
        "folds": fold_metadata,
        "comparisons": comparison,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return results, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--maximum-rows-per-source-cell", type=int, default=200)
    parser.add_argument("--trees", type=int, default=160)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results, metrics = run_pilot(
        args.data,
        maximum_rows_per_source_cell=args.maximum_rows_per_source_cell,
        trees=args.trees,
        seed=args.seed,
    )
    args.output_json.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics.to_csv(args.output_metrics, index=False, float_format="%.15g")
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
