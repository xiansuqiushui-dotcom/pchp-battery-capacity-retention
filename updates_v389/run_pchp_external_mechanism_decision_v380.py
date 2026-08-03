"""Execute the protocol-frozen V380 PCHP external mechanism and decision audit."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_method_ablation_v315 import RAW_ABSOLUTE_CHANGE_FEATURES
from develop_anchor_invariant_soh_v306 import (
    RAW_FEATURES,
    TARGET,
    balanced_source_rows,
    fit_predict,
    load_data,
)
from develop_context_change_soh_v310 import add_context_change
from prefix_causal_harm_projection_v321 import prefix_causal_cellwise_projection


ROOT = Path(__file__).resolve().parent
PREFREEZE = ROOT / "PCHP_EXTERNAL_MECHANISM_DECISION_PREFREEZE_V380.json"
EXPECTED_PREFREEZE_SHA256 = (
    "128fa29a2b21ad8022adf3afecc6351e8b480e770c4c8833a032c7097575b5cc"
)
OUT = ROOT / "external_mechanism_decision_v380"
V322 = ROOT / "prefix_causal_rccp_v322" / "prefix_causal_predictions_v322.parquet"
TOLERANCE = 1e-12

EXTERNAL_PATHS = {
    "AMPERE_A123_LFP": ROOT
    / "external_confirmation_v237"
    / "ampere_a123_lfp_early_charge_soh_v243.parquet",
    "IMPERIAL_M50T": ROOT
    / "external_confirmation_v237"
    / "imperial_m50t_early_charge_soh_v249.parquet",
    "LUH_NMC_SIO": ROOT
    / "external_luh_v255"
    / "luh_nmc_sio_early_charge_soh_v256.parquet",
    "ISU_ILCC_NMC": ROOT
    / "external_isu_ilcc_v270"
    / "isu_ilcc_early_charge_soh_v271.parquet",
    "STANFORD_CALENDAR": ROOT
    / "external_stanford_calendar_v280"
    / "stanford_calendar_early_charge_soh_v281.parquet",
    "MULTISTAGE_50E": ROOT
    / "external_multistage_50e_v290"
    / "multistage_50e_early_charge_soh_v291.parquet",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_prefreeze() -> dict[str, object]:
    if sha256_file(PREFREEZE) != EXPECTED_PREFREEZE_SHA256:
        raise RuntimeError("V380 prefreeze identity changed after protocol lock")
    frozen = json.loads(PREFREEZE.read_text(encoding="utf-8"))
    if frozen["status"] != "PREFROZEN_BEFORE_V380_SCORING":
        raise RuntimeError("V380 prefreeze status mismatch")
    for relative, expected in frozen["inputs"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"V380 frozen input mismatch: {relative}")
    return frozen


def load_external(frozen: dict[str, object]) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    observed_roster: list[dict[str, object]] = []
    required = {
        "domain",
        "cell_id",
        "aligned_cycle_rank",
        "after_initial5_reference_window",
        "target_cycle_number",
        TARGET,
        *RAW_FEATURES,
    }
    for surface, path in EXTERNAL_PATHS.items():
        frame = pd.read_parquet(path)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise RuntimeError(f"{surface} lacks required columns: {missing}")
        if set(frame["domain"].astype(str).unique()) != {surface}:
            raise RuntimeError(f"external identity mismatch: {surface}")
        enriched = add_context_change(frame)
        post = enriched.loc[
            enriched["after_initial5_reference_window"].astype(bool)
        ].copy()
        observed_roster.append(
            {
                "surface": surface,
                "cells": int(post["cell_id"].nunique()),
                "post_reference_records": int(len(post)),
            }
        )
        blocks.append(post)
    if observed_roster != frozen["external_surface_roster"]:
        raise RuntimeError("V380 external roster differs from the prefreeze")
    combined = pd.concat(blocks, ignore_index=True)
    boundary = frozen["information_boundary"]
    if (
        combined["domain"].nunique() != boundary["external_surfaces"]
        or combined["cell_id"].nunique() != boundary["external_cells"]
        or len(combined) != boundary["external_post_reference_records"]
    ):
        raise RuntimeError("V380 external boundary mismatch")
    if combined[["domain", "cell_id"]].drop_duplicates()["cell_id"].duplicated().any():
        raise RuntimeError("V380 external cell identifiers are not globally unique")
    return combined


def domain_cell_row_weights(frame: pd.DataFrame) -> np.ndarray:
    domains = frame["domain"].nunique()
    cells_per_domain = frame.groupby("domain")["cell_id"].nunique().to_dict()
    rows_per_cell = frame.groupby("cell_id").size().to_dict()
    return np.asarray(
        [
            1.0
            / (
                domains
                * cells_per_domain[row.domain]
                * rows_per_cell[row.cell_id]
            )
            for row in frame[["domain", "cell_id"]].itertuples(index=False)
        ],
        dtype=float,
    )


def first_weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    normalized = weights[order] / weights.sum()
    index = int(np.searchsorted(np.cumsum(normalized), 0.5, side="left"))
    return float(sorted_values[index])


def select_development_shift(alpha: float, delta: float) -> tuple[float, dict[str, object]]:
    frame = pd.read_parquet(V322)[
        ["domain", "cell_id", "target_cycle_number", "truth", "raw_baseline"]
    ].copy()
    protected, _ = prefix_causal_cellwise_projection(
        frame["cell_id"].astype(str).to_numpy(),
        frame["target_cycle_number"].to_numpy(float),
        frame["raw_baseline"].to_numpy(float),
        frame["raw_baseline"].to_numpy(float),
        delta,
        assimilation=alpha,
    )
    residual = frame["truth"].to_numpy(float) - protected
    unconstrained = first_weighted_median(
        residual, domain_cell_row_weights(frame)
    )
    selected = float(np.clip(unconstrained, -delta, delta))
    return selected, {
        "unconstrained_weighted_median_residual": unconstrained,
        "selected_budget_feasible_shift": selected,
        "selection_domains": int(frame["domain"].nunique()),
        "selection_cells": int(frame["cell_id"].nunique()),
        "selection_records": int(len(frame)),
    }


def maximum_trajectory_increase(
    frame: pd.DataFrame, values: np.ndarray
) -> float:
    maximum = 0.0
    identifiers = frame["cell_id"].astype(str).to_numpy()
    cycles = frame["target_cycle_number"].to_numpy(float)
    for identifier in np.unique(identifiers):
        positions = np.flatnonzero(identifiers == identifier)
        order = positions[np.argsort(cycles[positions], kind="mergesort")]
        if len(order) > 1:
            maximum = max(maximum, float(np.max(np.diff(values[order]))))
    return maximum


def maximum_prefix_replay_difference(
    frame: pd.DataFrame,
    state: np.ndarray,
    method: np.ndarray,
    *,
    alpha: float,
    delta: float,
) -> float:
    identifiers = frame["cell_id"].astype(str).to_numpy()
    cycles = frame["target_cycle_number"].to_numpy(float)
    raw = frame["raw_baseline"].to_numpy(float)
    candidate = frame["raw_candidate"].to_numpy(float)
    maximum = 0.0
    for identifier in np.unique(identifiers):
        positions = np.flatnonzero(identifiers == identifier)
        order = positions[np.argsort(cycles[positions], kind="mergesort")]
        prefix = order[: max(1, len(order) // 2)]
        replay_state, replay_method = prefix_causal_cellwise_projection(
            np.repeat(identifier, len(prefix)),
            cycles[prefix],
            raw[prefix],
            candidate[prefix],
            delta,
            assimilation=alpha,
        )
        maximum = max(
            maximum,
            float(np.max(np.abs(replay_state - state[prefix]))),
            float(np.max(np.abs(replay_method - method[prefix]))),
        )
    return maximum


def build_cell_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    methods = ("protected_state", "fixed_shift", "pchp_method", "raw_candidate")
    blocks: list[pd.DataFrame] = []
    truth = predictions["truth"].to_numpy(float)
    for method in methods:
        working = predictions[["domain", "cell_id"]].copy()
        working["method"] = method
        working["absolute_error"] = np.abs(
            predictions[method].to_numpy(float) - truth
        )
        blocks.append(
            working.groupby(["domain", "cell_id", "method"], as_index=False)
            .agg(mae=("absolute_error", "mean"), records=("absolute_error", "size"))
        )
    return pd.concat(blocks, ignore_index=True)


def surface_metrics(cell_metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        cell_metrics.groupby(["domain", "method"], as_index=False)
        .agg(cell_macro_mae=("mae", "mean"), cells=("cell_id", "nunique"))
    )
    wide = summary.pivot(index="domain", columns="method", values="cell_macro_mae")
    wide["pchp_minus_fixed_shift"] = wide["pchp_method"] - wide["fixed_shift"]
    wide["pchp_minus_protected"] = wide["pchp_method"] - wide["protected_state"]
    return wide.reset_index()


def cell_difference_map(
    long_metrics: pd.DataFrame,
    left: str,
    right: str,
    value: str,
) -> dict[str, np.ndarray]:
    subset = long_metrics.loc[
        long_metrics["method"].isin([left, right]),
        ["domain", "cell_id", "method", value],
    ]
    wide = subset.pivot(
        index=["domain", "cell_id"], columns="method", values=value
    ).reset_index()
    wide["difference"] = wide[left] - wide[right]
    return {
        domain: rows["difference"].to_numpy(float)
        for domain, rows in wide.groupby("domain", sort=True)
    }


def two_stage_bootstrap(
    values: dict[str, np.ndarray], repetitions: int, seed: int
) -> tuple[float, list[float]]:
    names = sorted(values)
    point = float(np.mean([values[name].mean() for name in names]))
    rng = np.random.default_rng(seed)
    replicates = np.empty(repetitions, dtype=float)
    chunk = 2500
    for start in range(0, repetitions, chunk):
        stop = min(start + chunk, repetitions)
        count = stop - start
        sampled_surfaces = rng.integers(0, len(names), size=(count, len(names)))
        totals = np.zeros(count, dtype=float)
        for position in range(len(names)):
            chosen = sampled_surfaces[:, position]
            contribution = np.zeros(count, dtype=float)
            for surface_index, name in enumerate(names):
                rows = np.flatnonzero(chosen == surface_index)
                if not len(rows):
                    continue
                vector = values[name]
                indices = rng.integers(0, len(vector), size=(len(rows), len(vector)))
                contribution[rows] = vector[indices].mean(axis=1)
            totals += contribution
        replicates[start:stop] = totals / len(names)
    return point, [
        float(np.quantile(replicates, 0.025)),
        float(np.quantile(replicates, 0.975)),
    ]


def build_decision_metrics(
    predictions: pd.DataFrame,
    thresholds: list[float],
    ratios: list[float],
) -> pd.DataFrame:
    methods = ("protected_state", "fixed_shift", "pchp_method")
    truth = predictions["truth"].to_numpy(float)
    blocks: list[pd.DataFrame] = []
    for threshold in thresholds:
        for ratio in ratios:
            for method in methods:
                prediction = predictions[method].to_numpy(float)
                premature = (truth > threshold) & (prediction <= threshold)
                missed = (truth <= threshold) & (prediction > threshold)
                binary = premature.astype(float) + ratio * missed.astype(float)
                continuous = np.maximum(truth - prediction, 0.0) + ratio * np.maximum(
                    prediction - truth, 0.0
                )
                working = predictions[["domain", "cell_id"]].copy()
                working["threshold"] = threshold
                working["cost_ratio"] = ratio
                working["method"] = method
                working["binary_cost"] = binary
                working["continuous_cost"] = continuous
                working["premature_review"] = premature.astype(float)
                working["missed_degraded_state"] = missed.astype(float)
                blocks.append(
                    working.groupby(
                        ["domain", "cell_id", "threshold", "cost_ratio", "method"],
                        as_index=False,
                    ).agg(
                        binary_cost=("binary_cost", "mean"),
                        continuous_cost=("continuous_cost", "mean"),
                        premature_review_rate=("premature_review", "mean"),
                        missed_degraded_state_rate=("missed_degraded_state", "mean"),
                    )
                )
    return pd.concat(blocks, ignore_index=True)


def summarize_decisions(decision_cells: pd.DataFrame) -> pd.DataFrame:
    surfaces = (
        decision_cells.groupby(
            ["domain", "threshold", "cost_ratio", "method"], as_index=False
        )
        .agg(
            binary_cost=("binary_cost", "mean"),
            continuous_cost=("continuous_cost", "mean"),
            premature_review_rate=("premature_review_rate", "mean"),
            missed_degraded_state_rate=("missed_degraded_state_rate", "mean"),
            cells=("cell_id", "nunique"),
        )
    )
    return (
        surfaces.groupby(["threshold", "cost_ratio", "method"], as_index=False)
        .agg(
            surface_equal_binary_cost=("binary_cost", "mean"),
            surface_equal_continuous_cost=("continuous_cost", "mean"),
            surface_equal_premature_review_rate=("premature_review_rate", "mean"),
            surface_equal_missed_degraded_state_rate=(
                "missed_degraded_state_rate",
                "mean",
            ),
            external_surfaces=("domain", "nunique"),
        )
    )


def main() -> None:
    started = time.perf_counter()
    frozen = verify_prefreeze()
    OUT.mkdir(parents=True, exist_ok=True)
    alpha = float(frozen["pchp_contract"]["assimilation_alpha"])
    delta = float(frozen["pchp_contract"]["absolute_harm_budget_delta"])

    development = add_context_change(load_data())
    development = development.loc[
        development["after_initial5_reference_window"].astype(bool)
    ].copy()
    source_fit = balanced_source_rows(development)
    external = load_external(frozen)

    raw_baseline = fit_predict(
        source_fit, external, RAW_FEATURES, domain_equal=False
    )
    raw_candidate = fit_predict(
        source_fit,
        external,
        RAW_ABSOLUTE_CHANGE_FEATURES,
        domain_equal=True,
    )
    predictions = external[
        ["domain", "cell_id", "target_cycle_number", TARGET]
    ].rename(columns={TARGET: "truth"})
    predictions["raw_baseline"] = raw_baseline
    predictions["raw_candidate"] = raw_candidate
    protected, pchp = prefix_causal_cellwise_projection(
        predictions["cell_id"].astype(str).to_numpy(),
        predictions["target_cycle_number"].to_numpy(float),
        raw_baseline,
        raw_candidate,
        delta,
        assimilation=alpha,
    )
    shift, shift_metadata = select_development_shift(alpha, delta)
    fixed = np.clip(protected + shift, 0.0, 1.3)
    predictions["protected_state"] = protected
    predictions["fixed_shift"] = fixed
    predictions["pchp_method"] = pchp

    cells = build_cell_metrics(predictions)
    surfaces = surface_metrics(cells)
    mechanism_map = cell_difference_map(
        cells, "pchp_method", "fixed_shift", "mae"
    )
    primary_point, primary_interval = two_stage_bootstrap(
        mechanism_map,
        int(frozen["primary_estimand"]["bootstrap_repetitions"]),
        int(frozen["primary_estimand"]["bootstrap_seed"]),
    )

    thresholds = [
        float(frozen["decision_cost_audit"]["primary_capacity_retention_threshold"]),
        *map(float, frozen["decision_cost_audit"]["sensitivity_thresholds"]),
    ]
    thresholds = sorted(set(thresholds))
    ratios = list(map(float, frozen["decision_cost_audit"]["miss_to_early_cost_ratios"]))
    decision_cells = build_decision_metrics(predictions, thresholds, ratios)
    decision_summary = summarize_decisions(decision_cells)

    primary_threshold = float(
        frozen["decision_cost_audit"]["primary_capacity_retention_threshold"]
    )
    primary_ratio = float(frozen["decision_cost_audit"]["primary_cost_ratio"])
    primary_decision_cells = decision_cells.loc[
        (decision_cells["threshold"] == primary_threshold)
        & (decision_cells["cost_ratio"] == primary_ratio)
    ]
    decision_fixed_map = cell_difference_map(
        primary_decision_cells,
        "pchp_method",
        "fixed_shift",
        "binary_cost",
    )
    decision_protected_map = cell_difference_map(
        primary_decision_cells,
        "pchp_method",
        "protected_state",
        "binary_cost",
    )
    decision_fixed_point, decision_fixed_interval = two_stage_bootstrap(
        decision_fixed_map,
        int(frozen["primary_estimand"]["bootstrap_repetitions"]),
        int(frozen["primary_estimand"]["bootstrap_seed"]),
    )
    decision_protected_point, decision_protected_interval = two_stage_bootstrap(
        decision_protected_map,
        int(frozen["primary_estimand"]["bootstrap_repetitions"]),
        int(frozen["primary_estimand"]["bootstrap_seed"]),
    )

    truth = predictions["truth"].to_numpy(float)
    certificate = {
        "maximum_absolute_displacement": float(np.max(np.abs(pchp - protected))),
        "maximum_observed_absolute_loss_regret": float(
            np.max(np.abs(pchp - truth) - np.abs(protected - truth))
        ),
        "maximum_protected_state_increase": maximum_trajectory_increase(
            predictions, protected
        ),
        "maximum_pchp_output_increase": maximum_trajectory_increase(
            predictions, pchp
        ),
        "minimum_pchp_output": float(pchp.min()),
        "maximum_pchp_output": float(pchp.max()),
        "maximum_prefix_replay_difference": maximum_prefix_replay_difference(
            predictions,
            protected,
            pchp,
            alpha=alpha,
            delta=delta,
        ),
    }
    surface_wins = int((surfaces["pchp_minus_fixed_shift"] < -TOLERANCE).sum())
    gates = {
        "mean_difference_below_zero": primary_point < 0.0,
        "bootstrap_interval_upper_below_zero": primary_interval[1] < 0.0,
        "minimum_improved_external_surfaces": surface_wins
        >= int(frozen["primary_retain_gates"]["minimum_improved_external_surfaces"]),
        "maximum_absolute_displacement": certificate["maximum_absolute_displacement"]
        <= delta + TOLERANCE,
        "maximum_observed_absolute_loss_regret": certificate[
            "maximum_observed_absolute_loss_regret"
        ]
        <= delta + TOLERANCE,
        "protected_state_nonincreasing": certificate[
            "maximum_protected_state_increase"
        ]
        <= TOLERANCE,
        "pchp_output_nonincreasing": certificate["maximum_pchp_output_increase"]
        <= TOLERANCE,
        "physical_range": certificate["minimum_pchp_output"] >= -TOLERANCE
        and certificate["maximum_pchp_output"] <= 1.3 + TOLERANCE,
        "prefix_replay_exact": certificate["maximum_prefix_replay_difference"]
        == 0.0,
    }
    deterministic_names = {
        "maximum_absolute_displacement",
        "maximum_observed_absolute_loss_regret",
        "protected_state_nonincreasing",
        "pchp_output_nonincreasing",
        "physical_range",
        "prefix_replay_exact",
    }
    deterministic_passed = all(gates[name] for name in deterministic_names)
    mechanism_retained = all(gates.values())
    decision_gate = {
        "pchp_cost_below_fixed_shift": decision_fixed_point < 0.0,
        "pchp_cost_below_protected_state": decision_protected_point < 0.0,
        "fixed_shift_interval_upper_below_zero": decision_fixed_interval[1] < 0.0,
    }
    decision_retained = all(decision_gate.values())
    if not deterministic_passed:
        decision = "REJECT"
        status = "V380_DETERMINISTIC_CERTIFICATE_FAILED"
    elif mechanism_retained:
        decision = "RETAIN"
        status = "V380_EXTERNAL_MECHANISM_RETAINED"
    else:
        decision = "NARROW"
        status = "V380_EXTERNAL_MECHANISM_NOT_CONFIRMED"

    paths = {
        "predictions": OUT / "external_mechanism_predictions_v380.parquet",
        "cell_metrics": OUT / "external_mechanism_cell_metrics_v380.csv",
        "surface_metrics": OUT / "external_mechanism_surface_metrics_v380.csv",
        "decision_cell_metrics": OUT / "external_decision_cell_metrics_v380.csv",
        "decision_summary": OUT / "external_decision_summary_v380.csv",
        "report": OUT / "external_mechanism_decision_v380_report.json",
    }
    predictions.to_parquet(paths["predictions"], index=False)
    cells.to_csv(paths["cell_metrics"], index=False)
    surfaces.to_csv(paths["surface_metrics"], index=False)
    decision_cells.to_csv(paths["decision_cell_metrics"], index=False)
    decision_summary.to_csv(paths["decision_summary"], index=False)

    method_means = (
        surfaces[["protected_state", "fixed_shift", "pchp_method", "raw_candidate"]]
        .mean()
        .to_dict()
    )
    report = {
        "version": "v380",
        "status": status,
        "decision": decision,
        "prefreeze": {
            "path": str(PREFREEZE),
            "sha256": sha256_file(PREFREEZE),
        },
        "classification": frozen["chronology"]["classification"],
        "information_boundary": frozen["information_boundary"],
        "model_contract": frozen["model_contract"],
        "pchp_contract": frozen["pchp_contract"],
        "development_shift_control": shift_metadata,
        "surface_equal_cell_macro_mae": method_means,
        "primary_mechanism_estimand": {
            "pchp_minus_fixed_shift": primary_point,
            "two_stage_95_percent_interval": primary_interval,
            "surface_wins_ties_losses": [
                surface_wins,
                int((np.abs(surfaces["pchp_minus_fixed_shift"]) <= TOLERANCE).sum()),
                int((surfaces["pchp_minus_fixed_shift"] > TOLERANCE).sum()),
            ],
        },
        "primary_mechanism_gates": gates,
        "mechanism_conjunction_passed": mechanism_retained,
        "deterministic_certificate": certificate,
        "primary_decision_cost": {
            "threshold": primary_threshold,
            "miss_to_early_cost_ratio": primary_ratio,
            "pchp_minus_fixed_shift": decision_fixed_point,
            "pchp_minus_fixed_shift_two_stage_95_percent_interval": decision_fixed_interval,
            "pchp_minus_protected_state": decision_protected_point,
            "pchp_minus_protected_two_stage_95_percent_interval": decision_protected_interval,
            "gates": decision_gate,
            "decision_claim_retained": decision_retained,
        },
        "surface_roster": frozen["external_surface_roster"],
        "runtime_seconds": float(time.perf_counter() - started),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
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
