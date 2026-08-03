"""Run the pre-frozen source-tuned causal-candidate control for PCHP."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from develop_anchor_invariant_soh_v306 import DATA_PATHS, TARGET, load_data
from prefix_causal_harm_projection_v321 import causal_nonincreasing_state


ROOT = Path(__file__).resolve().parent
PREFREEZE_PATH = ROOT / "paper_q1" / "SOURCE_TUNED_CAUSAL_CANDIDATE_PREFREEZE_V342.json"
CACHE_REPORT_PATH = ROOT / "nested_prefix_causal_selection_v326" / "nested_source_only_alpha_selection_v326_report.json"
CACHE_DIR = ROOT / "nested_prefix_causal_selection_v326" / "double_holdout_prediction_cache"
OUTER_CANDIDATE_PATH = ROOT / "prefix_causal_rccp_v322" / "prefix_causal_predictions_v322.parquet"
OUTER_SELECTED_PATH = ROOT / "nested_prefix_causal_outer_v327" / "nested_outer_predictions_v327.parquet"
OUTPUT_DIR = ROOT / "source_tuned_causal_candidate_v342"
INNER_PATH = OUTPUT_DIR / "source_tuned_causal_candidate_inner_metrics_v342.csv"
SELECTION_PATH = OUTPUT_DIR / "source_tuned_causal_candidate_selections_v342.csv"
CELL_PATH = OUTPUT_DIR / "source_tuned_causal_candidate_cell_metrics_v342.csv"
DOMAIN_PATH = OUTPUT_DIR / "source_tuned_causal_candidate_domain_metrics_v342.csv"
REPORT_PATH = OUTPUT_DIR / "source_tuned_causal_candidate_v342_report.json"

ALPHA_GRID = (1.0, 0.5, 0.2, 0.1, 0.05, 0.03, 0.02, 0.015, 0.01, 0.005)
BUDGET = 0.01
TOL = 1e-12
Y_MIN = 0.0
Y_MAX = 1.3
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 20260806
KEYS = ["domain", "cell_id", "target_cycle_number"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def cache_path(first: str, second: str) -> Path:
    a, b = sorted((first, second))
    return CACHE_DIR / f"exclude__{a}__{b}.parquet"


def causal_candidate_state(frame: pd.DataFrame, alpha: float) -> np.ndarray:
    output = np.empty(len(frame), dtype=float)
    groups = frame.groupby(["domain", "cell_id"], sort=False).indices
    for indices in groups.values():
        index = np.asarray(indices, dtype=int)
        output[index] = causal_nonincreasing_state(
            frame.loc[index, "raw_candidate"].to_numpy(float),
            assimilation=alpha,
            y_min=Y_MIN,
            y_max=Y_MAX,
        )
    return output


def causal_candidate_state_grid(frame: pd.DataFrame) -> np.ndarray:
    """Compute every frozen-alpha trajectory in one cellwise pass."""

    alphas = np.asarray(ALPHA_GRID, dtype=float)
    output = np.empty((len(frame), len(alphas)), dtype=float)
    groups = frame.groupby(["domain", "cell_id"], sort=False).indices
    for indices in groups.values():
        index = np.asarray(indices, dtype=int)
        values = frame.loc[index, "raw_candidate"].to_numpy(float)
        output[index[0], :] = np.clip(values[0], Y_MIN, Y_MAX)
        for local_index in range(1, len(index)):
            previous = output[index[local_index - 1], :]
            innovation = np.minimum(values[local_index] - previous, 0.0)
            output[index[local_index], :] = np.clip(
                previous + alphas * innovation,
                Y_MIN,
                Y_MAX,
            )
    return output


def causal_candidate_replay(values: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.empty_like(values)
    if values.size == 0:
        return output
    output[0] = np.clip(values[0], Y_MIN, Y_MAX)
    for index in range(1, values.size):
        innovation = min(values[index] - output[index - 1], 0.0)
        output[index] = np.clip(
            output[index - 1] + alpha * innovation,
            Y_MIN,
            Y_MAX,
        )
    return output


def cell_macro_mae(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    working = frame[["domain", "cell_id", TARGET]].copy()
    working["absolute_error"] = np.abs(
        np.asarray(prediction, dtype=float) - working[TARGET].to_numpy(float)
    )
    return float(
        working.groupby(["domain", "cell_id"], sort=False)["absolute_error"]
        .mean()
        .mean()
    )


def percentile_bootstrap(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.percentile(means, [2.5, 97.5])]


def verify_frozen_inputs(prefreeze: dict) -> tuple[dict[str, str], list[dict]]:
    expected = {item["path"]: item["sha256"].upper() for item in prefreeze["inputs"]}
    paths = {
        relative(CACHE_REPORT_PATH): CACHE_REPORT_PATH,
        relative(OUTER_CANDIDATE_PATH): OUTER_CANDIDATE_PATH,
        relative(OUTER_SELECTED_PATH): OUTER_SELECTED_PATH,
    }
    paths.update({relative(path): path for path in DATA_PATHS})
    observed = {name: sha256_file(path) for name, path in paths.items()}
    if observed != expected:
        raise RuntimeError(
            f"frozen input identity failure: expected={expected}, observed={observed}"
        )

    cache_report = json.loads(CACHE_REPORT_PATH.read_text(encoding="utf-8"))
    cache_metadata = cache_report["double_holdout_caches"]
    if len(cache_metadata) != 66:
        raise RuntimeError("expected 66 double-holdout caches")
    for item in cache_metadata:
        path = CACHE_DIR / Path(item["path"]).name
        if not path.is_file() or sha256_file(path) != item["sha256"].upper():
            raise RuntimeError(f"double-holdout cache identity failure: {path.name}")
    return observed, cache_metadata


def source_only_select_alphas() -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    evaluation = load_data()
    evaluation = (
        evaluation.loc[evaluation["after_initial5_reference_window"]]
        .copy()
        .sort_values(KEYS, kind="mergesort")
        .reset_index(drop=True)
    )
    domains = sorted(evaluation["domain"].astype(str).unique())
    inner_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    all_align = True
    for outer_domain in domains:
        source_pool = evaluation.loc[
            evaluation["domain"].astype(str) != outer_domain
        ].copy()
        inner_domains = sorted(source_pool["domain"].astype(str).unique())
        current_rows: list[dict[str, object]] = []
        for inner_domain in inner_domains:
            predictions = pd.read_parquet(cache_path(outer_domain, inner_domain))
            predictions = predictions.loc[
                predictions["domain"].astype(str) == inner_domain,
                KEYS + ["raw_candidate"],
            ].copy()
            validation = source_pool.loc[
                source_pool["domain"].astype(str) == inner_domain
            ].merge(predictions, on=KEYS, how="inner", validate="one_to_one")
            validation = validation.sort_values(KEYS, kind="mergesort").reset_index(
                drop=True
            )
            if len(validation) != len(predictions):
                all_align = False
                raise RuntimeError(
                    f"inner alignment failed for outer={outer_domain}, inner={inner_domain}"
                )
            q_grid = causal_candidate_state_grid(validation)
            error_frame = pd.DataFrame(
                np.abs(q_grid - validation[TARGET].to_numpy(float)[:, None]),
                columns=[f"alpha_{index}" for index in range(len(ALPHA_GRID))],
            )
            error_frame["domain"] = validation["domain"].to_numpy()
            error_frame["cell_id"] = validation["cell_id"].to_numpy()
            macro_by_alpha = (
                error_frame.groupby(["domain", "cell_id"], sort=False)
                .mean()
                .mean(axis=0)
            )
            for alpha_index, alpha in enumerate(ALPHA_GRID):
                row = {
                    "outer_target_domain": outer_domain,
                    "inner_validation_domain": inner_domain,
                    "alpha": alpha,
                    "physical_cells": int(validation["cell_id"].nunique()),
                    "validation_rows": int(len(validation)),
                    "causal_candidate_cell_macro_mae": float(
                        macro_by_alpha.iloc[alpha_index]
                    ),
                }
                inner_rows.append(row)
                current_rows.append(row)
        aggregate = (
            pd.DataFrame(current_rows)
            .groupby("alpha", as_index=False)
            .agg(
                inner_domain_equal_causal_candidate_mae=(
                    "causal_candidate_cell_macro_mae",
                    "mean",
                ),
                inner_domains=("inner_validation_domain", "nunique"),
            )
            .sort_values(
                ["inner_domain_equal_causal_candidate_mae", "alpha"],
                ascending=[True, False],
            )
        )
        minimum = float(
            aggregate["inner_domain_equal_causal_candidate_mae"].min()
        )
        selected = (
            aggregate.loc[
                aggregate["inner_domain_equal_causal_candidate_mae"]
                <= minimum + TOL
            ]
            .sort_values("alpha", ascending=False)
            .iloc[0]
        )
        selection_rows.append(
            {
                "outer_target_domain": outer_domain,
                "selected_alpha": float(selected["alpha"]),
                "selected_inner_domain_equal_causal_candidate_mae": float(
                    selected["inner_domain_equal_causal_candidate_mae"]
                ),
                "inner_domains": int(selected["inner_domains"]),
                "outer_target_labels_used_for_selection": False,
            }
        )
    return pd.DataFrame(inner_rows), pd.DataFrame(selection_rows), all_align


def evaluate_outer(selections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    candidate = pd.read_parquet(
        OUTER_CANDIDATE_PATH,
        columns=KEYS + ["truth", "raw_candidate"],
    ).sort_values(KEYS, kind="mergesort").reset_index(drop=True)
    selected = pd.read_parquet(
        OUTER_SELECTED_PATH,
        columns=KEYS
        + [
            "truth",
            "selected_causal_baseline",
            "selected_causal_method",
        ],
    ).sort_values(KEYS, kind="mergesort").reset_index(drop=True)
    row_keys_equal = candidate[KEYS].equals(selected[KEYS])
    truths_equal = np.array_equal(
        candidate["truth"].to_numpy(float), selected["truth"].to_numpy(float)
    )
    if not row_keys_equal or not truths_equal:
        raise RuntimeError("outer prediction rows do not align")
    selected["raw_candidate"] = candidate["raw_candidate"].to_numpy(float)
    alpha_map = selections.set_index("outer_target_domain")["selected_alpha"].to_dict()
    selected["comparator_alpha"] = selected["domain"].map(alpha_map)
    if selected["comparator_alpha"].isna().any():
        raise RuntimeError("missing comparator alpha")

    q = np.empty(len(selected), dtype=float)
    q_replay = np.empty(len(selected), dtype=float)
    groups = selected.groupby(["domain", "cell_id"], sort=False).indices
    for indices in groups.values():
        index = np.asarray(indices, dtype=int)
        alpha_values = selected.loc[index, "comparator_alpha"].to_numpy(float)
        if not np.all(alpha_values == alpha_values[0]):
            raise RuntimeError("alpha must be constant within a cell")
        alpha = float(alpha_values[0])
        values = selected.loc[index, "raw_candidate"].to_numpy(float)
        q[index] = causal_nonincreasing_state(
            values, assimilation=alpha, y_min=Y_MIN, y_max=Y_MAX
        )
        q_replay[index] = causal_candidate_replay(values, alpha)
    replay_exact = np.array_equal(q, q_replay)
    if not replay_exact:
        raise RuntimeError("independent source-tuned comparator replay mismatch")

    truth = selected["truth"].to_numpy(float)
    baseline = selected["selected_causal_baseline"].to_numpy(float)
    pchp = selected["selected_causal_method"].to_numpy(float)
    selected["source_tuned_causal_candidate"] = q
    selected["baseline_abs_error"] = np.abs(baseline - truth)
    selected["pchp_abs_error"] = np.abs(pchp - truth)
    selected["comparator_abs_error"] = np.abs(q - truth)
    selected["pchp_displacement"] = np.abs(pchp - baseline)
    selected["comparator_displacement"] = np.abs(q - baseline)
    selected["pchp_violation"] = selected["pchp_displacement"] > BUDGET + TOL
    selected["comparator_violation"] = (
        selected["comparator_displacement"] > BUDGET + TOL
    )

    cell = (
        selected.groupby(["domain", "cell_id"], sort=True)
        .agg(
            records=("truth", "size"),
            comparator_alpha=("comparator_alpha", "first"),
            baseline_mae=("baseline_abs_error", "mean"),
            pchp_mae=("pchp_abs_error", "mean"),
            source_tuned_causal_candidate_mae=("comparator_abs_error", "mean"),
            pchp_max_displacement=("pchp_displacement", "max"),
            source_tuned_candidate_max_displacement=(
                "comparator_displacement",
                "max",
            ),
            pchp_violation_records=("pchp_violation", "sum"),
            source_tuned_candidate_violation_records=(
                "comparator_violation",
                "sum",
            ),
        )
        .reset_index()
    )
    cell["pchp_minus_baseline"] = cell["pchp_mae"] - cell["baseline_mae"]
    cell["source_tuned_candidate_minus_baseline"] = (
        cell["source_tuned_causal_candidate_mae"] - cell["baseline_mae"]
    )
    cell["pchp_minus_source_tuned_candidate"] = (
        cell["pchp_mae"] - cell["source_tuned_causal_candidate_mae"]
    )
    cell["pchp_violates"] = cell["pchp_violation_records"] > 0
    cell["source_tuned_candidate_violates"] = (
        cell["source_tuned_candidate_violation_records"] > 0
    )

    domain = (
        cell.groupby("domain", sort=True)
        .agg(
            cells=("cell_id", "size"),
            records=("records", "sum"),
            comparator_alpha=("comparator_alpha", "first"),
            baseline_mae=("baseline_mae", "mean"),
            pchp_mae=("pchp_mae", "mean"),
            source_tuned_causal_candidate_mae=(
                "source_tuned_causal_candidate_mae",
                "mean",
            ),
            pchp_violating_cells=("pchp_violates", "sum"),
            source_tuned_candidate_violating_cells=(
                "source_tuned_candidate_violates",
                "sum",
            ),
            pchp_max_displacement=("pchp_max_displacement", "max"),
            source_tuned_candidate_max_displacement=(
                "source_tuned_candidate_max_displacement",
                "max",
            ),
        )
        .reset_index()
    )
    domain["pchp_minus_baseline"] = domain["pchp_mae"] - domain["baseline_mae"]
    domain["source_tuned_candidate_minus_baseline"] = (
        domain["source_tuned_causal_candidate_mae"] - domain["baseline_mae"]
    )
    domain["pchp_minus_source_tuned_candidate"] = (
        domain["pchp_mae"] - domain["source_tuned_causal_candidate_mae"]
    )
    domain["source_tuned_candidate_violating_cell_fraction"] = (
        domain["source_tuned_candidate_violating_cells"] / domain["cells"]
    )
    domain["pchp_violating_cell_fraction"] = (
        domain["pchp_violating_cells"] / domain["cells"]
    )

    nonincreasing = all(
        bool((np.diff(q[np.asarray(indices, dtype=int)]) <= TOL).all())
        for indices in groups.values()
    )
    checks = {
        "row_keys_equal": row_keys_equal,
        "truths_equal": truths_equal,
        "independent_replay_exact": replay_exact,
        "range_valid": bool(((q >= Y_MIN - TOL) & (q <= Y_MAX + TOL)).all()),
        "nonincreasing_within_every_cell": nonincreasing,
        "outer_outcome_used_in_prediction_construction": False,
    }
    return cell, domain, checks


def main() -> int:
    prefreeze = json.loads(PREFREEZE_PATH.read_text(encoding="utf-8"))
    observed_hashes, cache_metadata = verify_frozen_inputs(prefreeze)
    inner, selections, inner_alignment = source_only_select_alphas()
    cell, domain, implementation_checks = evaluate_outer(selections)

    selection_integrity = bool(
        inner_alignment
        and len(selections) == 12
        and selections["inner_domains"].eq(11).all()
        and not selections["outer_target_labels_used_for_selection"].any()
        and len(cache_metadata) == 66
    )
    roster_and_statistics = bool(
        len(domain) == 12
        and len(cell) == 586
        and int(cell["records"].sum()) == 601_932
    )
    violating_domains = int(
        (domain["source_tuned_candidate_violating_cells"] > 0).sum()
    )
    violating_cells = int(cell["source_tuned_candidate_violates"].sum())
    violating_cell_fraction = violating_cells / len(cell)
    pchp_violating_records = int(cell["pchp_violation_records"].sum())
    comparator_violating_records = int(
        cell["source_tuned_candidate_violation_records"].sum()
    )
    contract_activity = bool(
        violating_domains >= 9
        and violating_cell_fraction >= 0.25
        and pchp_violating_records == 0
    )

    baseline_mae = float(domain["baseline_mae"].mean())
    pchp_mae = float(domain["pchp_mae"].mean())
    comparator_mae = float(domain["source_tuned_causal_candidate_mae"].mean())
    pchp_gain = baseline_mae - pchp_mae
    comparator_gain = baseline_mae - comparator_mae
    utility_retention = None
    if comparator_gain > TOL:
        utility_retention = pchp_gain / comparator_gain
    utility = bool(
        (comparator_gain <= TOL and pchp_gain > TOL)
        or (
            comparator_gain > TOL
            and utility_retention is not None
            and utility_retention >= 0.50 - TOL
        )
    )
    online_properties = bool(
        implementation_checks["row_keys_equal"]
        and implementation_checks["truths_equal"]
        and implementation_checks["independent_replay_exact"]
        and implementation_checks["range_valid"]
        and implementation_checks["nonincreasing_within_every_cell"]
        and not implementation_checks[
            "outer_outcome_used_in_prediction_construction"
        ]
    )
    gates = {
        "selection_integrity": selection_integrity,
        "contract_activity": contract_activity,
        "utility": utility,
        "online_properties": online_properties,
        "roster_and_statistics": roster_and_statistics,
    }
    science_failures = int(not contract_activity) + int(not utility)
    if (
        not selection_integrity
        or not online_properties
        or not roster_and_statistics
        or science_failures == 2
    ):
        decision = "REJECT"
    elif science_failures == 1:
        decision = "NARROW"
    else:
        decision = "RETAIN"

    comparisons = {}
    for name in (
        "pchp_minus_baseline",
        "source_tuned_candidate_minus_baseline",
        "pchp_minus_source_tuned_candidate",
    ):
        values = domain[name].to_numpy(float)
        comparisons[name] = {
            "domain_equal_mean": float(values.mean()),
            "domain_bootstrap_ci95": percentile_bootstrap(values),
            "wins_ties_losses": [
                int((values < -TOL).sum()),
                int((np.abs(values) <= TOL).sum()),
                int((values > TOL).sum()),
            ],
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inner.to_csv(INNER_PATH, index=False, float_format="%.17g", lineterminator="\n")
    selections.to_csv(
        SELECTION_PATH, index=False, float_format="%.17g", lineterminator="\n"
    )
    cell.to_csv(CELL_PATH, index=False, float_format="%.17g", lineterminator="\n")
    domain.to_csv(DOMAIN_PATH, index=False, float_format="%.17g", lineterminator="\n")

    report = {
        "version": "v342",
        "report_date": prefreeze["date"],
        "status": "SOURCE_TUNED_CAUSAL_CANDIDATE_CONTROL_COMPLETED",
        "decision": decision,
        "evidence_role": prefreeze["role"],
        "budget": BUDGET,
        "tolerance": TOL,
        "alpha_grid": list(ALPHA_GRID),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "unit": "complete_dataset_domain",
        },
        "inputs": [
            {"path": path, "sha256": digest}
            for path, digest in observed_hashes.items()
        ],
        "cache_identity": {
            "double_holdout_caches": len(cache_metadata),
            "all_hashes_matched": True,
        },
        "selection": {
            "outer_domains": int(len(selections)),
            "inner_domains_per_outer": int(selections["inner_domains"].min()),
            "outer_target_labels_used_for_selection": bool(
                selections["outer_target_labels_used_for_selection"].any()
            ),
            "selected_alphas": json.loads(
                selections[["outer_target_domain", "selected_alpha"]].to_json(
                    orient="records"
                )
            ),
        },
        "roster": {
            "domains": int(len(domain)),
            "physical_cells": int(len(cell)),
            "records": int(cell["records"].sum()),
        },
        "accuracy": {
            "domain_equal_baseline_mae": baseline_mae,
            "domain_equal_pchp_mae": pchp_mae,
            "domain_equal_source_tuned_candidate_mae": comparator_mae,
            "pchp_gain_over_baseline": pchp_gain,
            "source_tuned_candidate_gain_over_baseline": comparator_gain,
            "pchp_utility_retention_fraction": utility_retention,
            "comparisons": comparisons,
        },
        "contract_activity": {
            "source_tuned_candidate_violating_domains": violating_domains,
            "source_tuned_candidate_violating_cells": violating_cells,
            "source_tuned_candidate_violating_cell_fraction": violating_cell_fraction,
            "source_tuned_candidate_violating_records": comparator_violating_records,
            "source_tuned_candidate_max_displacement": float(
                cell["source_tuned_candidate_max_displacement"].max()
            ),
            "pchp_violating_records": pchp_violating_records,
            "pchp_max_displacement": float(cell["pchp_max_displacement"].max()),
        },
        "implementation_checks": implementation_checks,
        "gates": gates,
        "artifacts": [],
        "boundary": prefreeze["chronology_boundary"],
    }
    for path in (
        PREFREEZE_PATH,
        Path(__file__).resolve(),
        INNER_PATH,
        SELECTION_PATH,
        CELL_PATH,
        DOMAIN_PATH,
    ):
        report["artifacts"].append(
            {"path": relative(path), "sha256": sha256_file(path)}
        )
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
