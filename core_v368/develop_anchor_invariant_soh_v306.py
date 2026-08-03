"""Twelve-domain development audit for anchor-invariant SOH estimation.

The method uses only the first five commissioning-cycle charge signals to form
per-cell robust location and scale anchors.  Later early-charge signals are
expressed in reference coordinates.  When the reference interquartile range is
positive, these coordinates are exactly invariant to a positive affine sensor
transformation applied consistently within a cell.  Invalid anchor channels
are failed closed to zero and accompanied by an availability indicator.

Every outer fold leaves one complete battery dataset domain out.  Target labels
are used only for final scoring.  The twelve domains have all been opened in
earlier work, so this is development evidence rather than external
confirmation.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression

from pilot_batterylife_early_charge_lodo_v110 import EARLY_CHARGE_FEATURES


ROOT = Path(__file__).resolve().parent
DATA_PATHS = [
    ROOT / "batterylife_early_charge_soh_v109.parquet",
    ROOT / "batterylife_external_hust_rwth_v124.parquet",
    ROOT / "batterylife_sdu_early_charge_soh_v145.parquet",
    ROOT / "batterylife_matr_early_charge_soh_v151.parquet",
]
OUT = ROOT / "anchor_invariant_soh_v306"
TARGET = "soh_initial5"
SEED = 20_260_801
MAXIMUM_SOURCE_ROWS_PER_CELL = 80
TREES = 80
MINIMUM_SAMPLES_LEAF = 20
ANCHOR_CYCLES = 5
ANCHOR_SCALE_TOLERANCE = 1e-12
NORMALIZED_CHANGE_CLIP = 20.0
BOOTSTRAP_REPETITIONS = 100_000
BOOTSTRAP_SEED = 20_260_802

RAW_FEATURES = ["target_cycle_number", *EARLY_CHARGE_FEATURES]
ANCHOR_FEATURES = ["target_cycle_number"] + [
    item
    for feature in EARLY_CHARGE_FEATURES
    for item in (f"anchor_z__{feature}", f"anchor_valid__{feature}")
]
MODEL_ORDER = [
    "cycle_only",
    "raw_early_charge",
    "anchor_invariant",
    "anchor_invariant_domain_equal",
    "anchor_invariant_domain_equal_monotone",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data() -> pd.DataFrame:
    required = {
        "domain",
        "cell_id",
        "aligned_cycle_rank",
        "target_cycle_number",
        "after_initial5_reference_window",
        TARGET,
        *EARLY_CHARGE_FEATURES,
    }
    blocks: list[pd.DataFrame] = []
    seen_domains: set[str] = set()
    for path in DATA_PATHS:
        frame = pd.read_parquet(path)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise RuntimeError(f"{path.name} lacks required columns: {missing}")
        domains = set(frame["domain"].astype(str).unique())
        overlap = seen_domains & domains
        if overlap:
            raise RuntimeError(f"domain identities overlap across input files: {overlap}")
        seen_domains |= domains
        blocks.append(frame[list(required)].copy())
    combined = pd.concat(blocks, ignore_index=True)
    if combined["domain"].nunique() != 12:
        raise RuntimeError("the frozen development roster must contain 12 domains")
    if combined[["domain", "cell_id"]].drop_duplicates()["cell_id"].duplicated().any():
        raise RuntimeError("cell identifiers are not globally unique across domains")
    return combined


def add_anchor_coordinates(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = frame.loc[frame["aligned_cycle_rank"] <= ANCHOR_CYCLES].copy()
    reference_counts = reference.groupby("cell_id")["aligned_cycle_rank"].nunique()
    invalid_cells = reference_counts.loc[reference_counts < ANCHOR_CYCLES]
    if len(invalid_cells):
        raise RuntimeError(
            f"{len(invalid_cells)} cells lack the required commissioning anchor cycles"
        )

    grouped = reference.groupby("cell_id", sort=False)[list(EARLY_CHARGE_FEATURES)]
    median = grouped.median().add_prefix("anchor_median__")
    q25 = grouped.quantile(0.25)
    q75 = grouped.quantile(0.75)
    scale = (q75 - q25).abs().add_prefix("anchor_scale__")
    anchor = median.join(scale, how="inner")

    output = frame.merge(anchor, on="cell_id", how="left", validate="many_to_one")
    coverage_rows: list[dict[str, object]] = []
    for feature in EARLY_CHARGE_FEATURES:
        location = output[f"anchor_median__{feature}"].to_numpy(float)
        spread = output[f"anchor_scale__{feature}"].to_numpy(float)
        observed = output[feature].to_numpy(float)
        valid = (
            np.isfinite(location)
            & np.isfinite(spread)
            & (spread > ANCHOR_SCALE_TOLERANCE)
        )
        normalized = np.zeros(len(output), dtype=float)
        normalized[valid] = (observed[valid] - location[valid]) / spread[valid]
        normalized[~np.isfinite(normalized)] = 0.0
        output[f"anchor_z__{feature}"] = np.clip(
            normalized, -NORMALIZED_CHANGE_CLIP, NORMALIZED_CHANGE_CLIP
        )
        output[f"anchor_valid__{feature}"] = valid.astype(float)
        cell_validity = (
            output[["domain", "cell_id", f"anchor_valid__{feature}"]]
            .drop_duplicates(["domain", "cell_id"])
        )
        coverage_rows.append(
            {
                "feature": feature,
                "cells": int(len(cell_validity)),
                "valid_anchor_cells": int(cell_validity[f"anchor_valid__{feature}"].sum()),
                "valid_anchor_fraction": float(
                    cell_validity[f"anchor_valid__{feature}"].mean()
                ),
            }
        )
    drop_columns = [
        column
        for column in output.columns
        if column.startswith("anchor_median__") or column.startswith("anchor_scale__")
    ]
    output = output.drop(columns=drop_columns)
    return output, pd.DataFrame(coverage_rows)


def balanced_source_rows(frame: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, cell in frame.groupby("cell_id", sort=True):
        ordered = cell.sort_values("target_cycle_number", kind="mergesort")
        if len(ordered) <= MAXIMUM_SOURCE_ROWS_PER_CELL:
            pieces.append(ordered)
        else:
            positions = np.unique(
                np.rint(
                    np.linspace(0, len(ordered) - 1, MAXIMUM_SOURCE_ROWS_PER_CELL)
                ).astype(int)
            )
            pieces.append(ordered.iloc[positions])
    return pd.concat(pieces, ignore_index=True)


def domain_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    domain_cell_counts = frame.groupby("domain")["cell_id"].nunique().to_dict()
    cell_row_counts = frame.groupby("cell_id").size().to_dict()
    number_of_domains = frame["domain"].nunique()
    weights = np.asarray(
        [
            1.0
            / (
                number_of_domains
                * domain_cell_counts[row.domain]
                * cell_row_counts[row.cell_id]
            )
            for row in frame[["domain", "cell_id"]].itertuples(index=False)
        ],
        dtype=float,
    )
    return weights / weights.mean()


def raw_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    matrix = frame[columns].astype(float).to_numpy()
    cycle_index = columns.index("target_cycle_number")
    if (matrix[:, cycle_index] < 0.0).any():
        raise ValueError("target cycle number must be non-negative")
    matrix[:, cycle_index] = np.log1p(matrix[:, cycle_index])
    return matrix


def fit_predict(
    source: pd.DataFrame,
    target: pd.DataFrame,
    columns: list[str],
    *,
    domain_equal: bool,
) -> np.ndarray:
    source_raw = raw_matrix(source, columns)
    target_raw = raw_matrix(target, columns)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    source_matrix = imputer.fit_transform(source_raw)
    target_matrix = imputer.transform(target_raw)
    model = ExtraTreesRegressor(
        n_estimators=TREES,
        min_samples_leaf=MINIMUM_SAMPLES_LEAF,
        max_features=0.8,
        random_state=SEED,
        n_jobs=-1,
    )
    weights = domain_equal_weights(source) if domain_equal else None
    model.fit(source_matrix, source[TARGET].to_numpy(float), sample_weight=weights)
    return np.clip(model.predict(target_matrix), 0.0, 1.3)


def monotone_project(frame: pd.DataFrame, prediction: np.ndarray) -> np.ndarray:
    output = np.empty(len(frame), dtype=float)
    working = frame[["cell_id", "target_cycle_number"]].copy()
    working["position"] = np.arange(len(frame))
    working["prediction"] = np.asarray(prediction, dtype=float)
    for _, cell in working.groupby("cell_id", sort=False):
        ordered = cell.sort_values("target_cycle_number", kind="mergesort")
        projected = IsotonicRegression(
            increasing=False,
            out_of_bounds="clip",
            y_min=0.0,
            y_max=1.3,
        ).fit_transform(
            ordered["target_cycle_number"].to_numpy(float),
            ordered["prediction"].to_numpy(float),
        )
        output[ordered["position"].to_numpy(int)] = projected
    return output


def cell_metrics(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    method: str,
) -> pd.DataFrame:
    working = frame[["domain", "cell_id", "target_cycle_number", TARGET]].copy()
    working["prediction"] = np.asarray(prediction, dtype=float)
    working["absolute_error"] = np.abs(working["prediction"] - working[TARGET])
    working["squared_error"] = np.square(working["prediction"] - working[TARGET])
    records: list[dict[str, object]] = []
    for (domain, cell_id), cell in working.groupby(["domain", "cell_id"], sort=False):
        ordered = cell.sort_values("target_cycle_number", kind="mergesort")
        differences = np.diff(ordered["prediction"].to_numpy(float))
        records.append(
            {
                "domain": domain,
                "cell_id": cell_id,
                "method": method,
                "rows": len(cell),
                "mae": float(cell["absolute_error"].mean()),
                "rmse": float(np.sqrt(cell["squared_error"].mean())),
                "trajectory_increase_fraction": (
                    float((differences > TOLERANCE).mean()) if len(differences) else 0.0
                ),
                "maximum_predicted_increase": (
                    float(np.maximum(differences, 0.0).max())
                    if len(differences)
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(records)


TOLERANCE = 1e-12


def domain_summary(cells: pd.DataFrame) -> pd.DataFrame:
    return (
        cells.groupby(["domain", "method"], as_index=False)
        .agg(
            physical_cells=("cell_id", "nunique"),
            cell_macro_mae=("mae", "mean"),
            cell_macro_rmse=("rmse", "mean"),
            worst_cell_mae=("mae", "max"),
            mean_trajectory_increase_fraction=("trajectory_increase_fraction", "mean"),
            maximum_predicted_increase=("maximum_predicted_increase", "max"),
        )
    )


def paired_domain_comparison(
    domains: pd.DataFrame,
    method: str,
    baseline: str,
) -> dict[str, object]:
    method_rows = domains.loc[domains["method"] == method, ["domain", "cell_macro_mae"]]
    baseline_rows = domains.loc[
        domains["method"] == baseline, ["domain", "cell_macro_mae"]
    ]
    paired = method_rows.merge(
        baseline_rows, on="domain", suffixes=("_method", "_baseline"), validate="one_to_one"
    )
    paired["difference"] = (
        paired["cell_macro_mae_method"] - paired["cell_macro_mae_baseline"]
    )
    differences = paired["difference"].to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, len(differences), size=(BOOTSTRAP_REPETITIONS, len(differences))
    )
    replicates = differences[indices].mean(axis=1)
    return {
        "method": method,
        "baseline": baseline,
        "difference_direction": "negative favors method",
        "domain_equal_mean_difference": float(differences.mean()),
        "ci95_domain_cluster_percentile": np.quantile(replicates, [0.025, 0.975]).tolist(),
        "domain_wins_ties_losses": [
            int((differences < -TOLERANCE).sum()),
            int((np.abs(differences) <= TOLERANCE).sum()),
            int((differences > TOLERANCE).sum()),
        ],
        "maximum_domain_harm": float(differences.max()),
        "maximum_domain_improvement": float(differences.min()),
        "per_domain_difference": dict(zip(paired["domain"], differences)),
    }


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_data()
    enriched, anchor_coverage = add_anchor_coordinates(raw)
    evaluation = enriched.loc[enriched["after_initial5_reference_window"]].copy()
    domains = sorted(evaluation["domain"].unique())
    cell_blocks: list[pd.DataFrame] = []
    prediction_blocks: list[pd.DataFrame] = []
    fold_metadata: list[dict[str, object]] = []

    for fold_index, target_domain in enumerate(domains, start=1):
        source = evaluation.loc[evaluation["domain"] != target_domain].copy()
        target = evaluation.loc[evaluation["domain"] == target_domain].copy()
        source_fit = balanced_source_rows(source)
        predictions: dict[str, np.ndarray] = {}
        predictions["cycle_only"] = fit_predict(
            source_fit, target, ["target_cycle_number"], domain_equal=False
        )
        predictions["raw_early_charge"] = fit_predict(
            source_fit, target, RAW_FEATURES, domain_equal=False
        )
        predictions["anchor_invariant"] = fit_predict(
            source_fit, target, ANCHOR_FEATURES, domain_equal=False
        )
        predictions["anchor_invariant_domain_equal"] = fit_predict(
            source_fit, target, ANCHOR_FEATURES, domain_equal=True
        )
        predictions["anchor_invariant_domain_equal_monotone"] = monotone_project(
            target, predictions["anchor_invariant_domain_equal"]
        )

        for method in MODEL_ORDER:
            cell_blocks.append(cell_metrics(target, predictions[method], method))
            prediction_blocks.append(
                pd.DataFrame(
                    {
                        "domain": target["domain"].astype(str).to_numpy(),
                        "cell_id": target["cell_id"].astype(str).to_numpy(),
                        "target_cycle_number": target["target_cycle_number"].to_numpy(float),
                        "truth": target[TARGET].to_numpy(float),
                        "method": method,
                        "prediction": predictions[method],
                    }
                )
            )
        fold_metadata.append(
            {
                "target_domain": target_domain,
                "source_domains": sorted(source["domain"].unique()),
                "source_cells": int(source["cell_id"].nunique()),
                "source_rows": int(len(source)),
                "source_fit_rows": int(len(source_fit)),
                "target_cells": int(target["cell_id"].nunique()),
                "target_rows": int(len(target)),
            }
        )
        print(f"completed outer domain {fold_index}/{len(domains)}: {target_domain}", flush=True)

    cells = pd.concat(cell_blocks, ignore_index=True)
    predictions = pd.concat(prediction_blocks, ignore_index=True)
    domains_frame = domain_summary(cells)
    overall = (
        domains_frame.groupby("method", as_index=False)
        .agg(
            domain_equal_cell_macro_mae=("cell_macro_mae", "mean"),
            domain_equal_cell_macro_rmse=("cell_macro_rmse", "mean"),
            worst_domain_cell_macro_mae=("cell_macro_mae", "max"),
            mean_domain_worst_cell_mae=("worst_cell_mae", "mean"),
            domain_equal_trajectory_increase_fraction=(
                "mean_trajectory_increase_fraction",
                "mean",
            ),
        )
        .sort_values("domain_equal_cell_macro_mae")
    )
    comparisons = {
        method: paired_domain_comparison(domains_frame, method, "raw_early_charge")
        for method in MODEL_ORDER
        if method != "raw_early_charge"
    }

    paths = {
        "anchor_coverage": OUT / "anchor_channel_coverage_v306.csv",
        "cell_metrics": OUT / "cell_level_metrics_v306.csv",
        "domain_metrics": OUT / "domain_level_metrics_v306.csv",
        "overall": OUT / "domain_equal_summary_v306.csv",
        "predictions": OUT / "outer_lodo_predictions_v306.parquet",
        "report": OUT / "anchor_invariant_soh_v306_report.json",
    }
    anchor_coverage.to_csv(paths["anchor_coverage"], index=False)
    cells.to_csv(paths["cell_metrics"], index=False)
    domains_frame.to_csv(paths["domain_metrics"], index=False)
    overall.to_csv(paths["overall"], index=False)
    predictions.to_parquet(paths["predictions"], index=False)

    proposed = "anchor_invariant_domain_equal_monotone"
    proposed_comparison = comparisons[proposed]
    status = (
        "ANCHOR_INVARIANT_DEVELOPMENT_GATE_PASSED"
        if (
            proposed_comparison["domain_equal_mean_difference"] < 0.0
            and proposed_comparison["ci95_domain_cluster_percentile"][1] < 0.0
            and proposed_comparison["maximum_domain_harm"] <= 0.01
        )
        else "ANCHOR_INVARIANT_DEVELOPMENT_GATE_NOT_PASSED"
    )
    report = {
        "status": status,
        "scope": (
            "retrospective development-only leave-one-complete-battery-domain-out "
            "audit on twelve previously opened public domains"
        ),
        "independent_unit_for_cross_domain_claim": "battery dataset domain",
        "nested_units": "physical cells within domain; cycles within physical cell",
        "target_outcome_access_during_training_or_preprocessing": False,
        "data": {
            "input_paths": [str(path) for path in DATA_PATHS],
            "domains": domains,
            "domain_count": len(domains),
            "physical_cells": int(evaluation["cell_id"].nunique()),
            "post_reference_cycle_rows": int(len(evaluation)),
        },
        "method_contract": {
            "commissioning_anchor_cycles": ANCHOR_CYCLES,
            "anchor_location": "within-cell median of the first five early-charge signals",
            "anchor_scale": "within-cell interquartile range of the first five early-charge signals",
            "invalid_channel_policy": "set normalized change to zero and expose a label-free validity indicator",
            "normalized_change_clip": NORMALIZED_CHANGE_CLIP,
            "source_balance": (
                "at most 80 evenly spaced rows per physical cell; optional weights "
                "give equal total mass to domains and then cells"
            ),
            "monotone_projection": (
                "target-label-free non-increasing isotonic projection within each physical cell"
            ),
        },
        "model": {
            "family": "ExtraTreesRegressor",
            "trees": TREES,
            "minimum_samples_leaf": MINIMUM_SAMPLES_LEAF,
            "seed": SEED,
        },
        "folds": fold_metadata,
        "overall": overall.where(pd.notna(overall), None).to_dict(orient="records"),
        "comparisons_against_raw_early_charge": comparisons,
        "primary_development_gate": {
            "method": proposed,
            "baseline": "raw_early_charge",
            "requirements": [
                "negative domain-equal MAE difference",
                "upper endpoint of 95% domain-cluster bootstrap interval below zero",
                "maximum domain harm no greater than 0.01 MAE",
            ],
        },
        "limitations": [
            "All twelve domains were opened before this method was proposed.",
            "The first-five-cycle capacity reference is required to define the target but is not a model feature.",
            "Observed SOH contains measurement noise and is not assumed to be exactly monotone.",
            "Passing this gate would justify originality and mechanism audits, not an external-confirmation claim.",
        ],
        "runtime_seconds": float(time.perf_counter() - started),
        "files": {},
    }
    for name, path in paths.items():
        if name == "report":
            continue
        report["files"][name] = {"path": str(path), "sha256": sha256_file(path)}
    paths["report"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
