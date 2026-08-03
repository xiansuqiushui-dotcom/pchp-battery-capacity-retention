"""Test whether the learned candidate adds value beyond a constant safe shift.

For every outer development domain, a candidate-free comparator jointly
selects the causal assimilation value and a constant offset inside the same
harm tube using only the other eleven source domains.  It is then scored once
on the outer domain and compared with the nested source-selected PCHP method.
No NASA artifact is read by this script.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from build_nested_source_only_alpha_selections_v326 import cache_path
from develop_anchor_invariant_soh_v306 import TARGET, load_data
from develop_context_change_soh_v310 import add_context_change


ROOT = Path(__file__).resolve().parent
PREFREEZE = (
    ROOT / "paper_q1" / "rccp_candidate_information_control_prefreeze_v333.json"
)
OUT = ROOT / "candidate_information_control_v333"
V322_PREDICTIONS = (
    ROOT / "prefix_causal_rccp_v322" / "prefix_causal_predictions_v322.parquet"
)
V327_PREDICTIONS = (
    ROOT / "nested_prefix_causal_outer_v327" / "nested_outer_predictions_v327.parquet"
)
ALPHA_GRID = (1.0, 0.5, 0.2, 0.1, 0.05, 0.03, 0.02, 0.015, 0.01, 0.005)
OFFSET_GRID = tuple(float(value) for value in np.linspace(-0.01, 0.01, 21))
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


@njit(cache=True)
def _ordered_causal_baseline(
    ordered_cell_codes: np.ndarray,
    ordered_raw: np.ndarray,
    alpha: float,
) -> np.ndarray:
    output = np.empty_like(ordered_raw)
    previous_cell = -1
    state = 0.0
    for index in range(len(ordered_raw)):
        cell = int(ordered_cell_codes[index])
        raw = ordered_raw[index]
        if index == 0 or cell != previous_cell:
            state = min(1.3, max(0.0, raw))
            previous_cell = cell
        else:
            innovation = min(raw - state, 0.0)
            state = min(1.3, max(0.0, state + alpha * innovation))
        output[index] = state
    return output


def causal_baseline(frame: pd.DataFrame, alpha: float) -> np.ndarray:
    raw = frame["raw_baseline"].to_numpy(float)
    cycles = frame["target_cycle_number"].to_numpy(float)
    cell_codes, _ = pd.factorize(frame["cell_id"].astype(str), sort=False)
    order = np.lexsort((cycles, cell_codes))
    ordered = _ordered_causal_baseline(cell_codes[order], raw[order], alpha)
    baseline = np.empty_like(raw)
    baseline[order] = ordered
    return baseline


def cell_macro_mae(
    frame: pd.DataFrame, prediction: np.ndarray, truth_column: str = TARGET
) -> float:
    working = frame[["cell_id", truth_column]].copy()
    working["absolute_error"] = np.abs(
        np.asarray(prediction, dtype=float)
        - working[truth_column].to_numpy(float)
    )
    return float(working.groupby("cell_id")["absolute_error"].mean().mean())


def cell_macro_mae_grid(
    truth: np.ndarray,
    baseline: np.ndarray,
    cell_codes: np.ndarray,
    cell_counts: np.ndarray,
) -> np.ndarray:
    """Vectorized cell-macro MAE for every frozen constant offset."""

    values = np.empty(len(OFFSET_GRID), dtype=float)
    for index, offset in enumerate(OFFSET_GRID):
        prediction = np.clip(baseline + offset, 0.0, 1.3)
        errors = np.abs(prediction - truth)
        per_cell = np.bincount(
            cell_codes, weights=errors, minlength=len(cell_counts)
        ) / cell_counts
        values[index] = float(per_cell.mean())
    return values


def build_source_only_selections(
    evaluation: pd.DataFrame, domains: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["domain", "cell_id", "target_cycle_number"]
    inner_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for outer_index, outer_domain in enumerate(domains, start=1):
        inner_domains = [domain for domain in domains if domain != outer_domain]
        for inner_domain in inner_domains:
            cached = pd.read_parquet(cache_path(outer_domain, inner_domain))
            cached = cached.loc[cached["domain"].astype(str) == inner_domain].copy()
            truth = evaluation.loc[
                evaluation["domain"].astype(str) == inner_domain,
                keys + [TARGET],
            ]
            validation = truth.merge(cached, on=keys, validate="one_to_one")
            if len(validation) != len(cached):
                raise RuntimeError(
                    f"inner alignment failed: outer={outer_domain}, inner={inner_domain}"
                )
            cell_codes, unique_cells = pd.factorize(
                validation["cell_id"].astype(str), sort=False
            )
            cell_counts = np.bincount(
                cell_codes, minlength=len(unique_cells)
            ).astype(float)
            truth_values = validation[TARGET].to_numpy(float)
            for alpha in ALPHA_GRID:
                baseline = causal_baseline(validation, alpha)
                mae_grid = cell_macro_mae_grid(
                    truth_values, baseline, cell_codes, cell_counts
                )
                for offset, metric in zip(OFFSET_GRID, mae_grid):
                    inner_rows.append(
                        {
                            "outer_target_domain": outer_domain,
                            "inner_validation_domain": inner_domain,
                            "alpha": alpha,
                            "offset": offset,
                            "cell_macro_mae": float(metric),
                            "physical_cells": int(validation["cell_id"].nunique()),
                            "prediction_rows": int(len(validation)),
                        }
                    )
        outer_inner = pd.DataFrame(
            [row for row in inner_rows if row["outer_target_domain"] == outer_domain]
        )
        aggregate = (
            outer_inner.groupby(["alpha", "offset"], as_index=False)
            .agg(
                inner_domain_equal_cell_macro_mae=("cell_macro_mae", "mean"),
                inner_domains=("inner_validation_domain", "nunique"),
            )
        )
        minimum = float(aggregate["inner_domain_equal_cell_macro_mae"].min())
        tied = aggregate.loc[
            aggregate["inner_domain_equal_cell_macro_mae"] <= minimum + TOLERANCE
        ].copy()
        tied["absolute_offset"] = tied["offset"].abs()
        selected = tied.sort_values(
            ["absolute_offset", "alpha", "offset"],
            ascending=[True, False, False],
        ).iloc[0]
        selection_rows.append(
            {
                "outer_target_domain": outer_domain,
                "selected_control_alpha": float(selected["alpha"]),
                "selected_control_offset": float(selected["offset"]),
                "selected_inner_domain_equal_cell_macro_mae": float(
                    selected["inner_domain_equal_cell_macro_mae"]
                ),
                "inner_domains": int(selected["inner_domains"]),
                "outer_target_labels_used_for_selection": False,
            }
        )
        print(
            f"control selection {outer_index}/{len(domains)}: {outer_domain} -> "
            f"alpha={float(selected['alpha']):g}, "
            f"offset={float(selected['offset']):+.3f}",
            flush=True,
        )
    return pd.DataFrame(inner_rows), pd.DataFrame(selection_rows)


def build_outer_predictions(
    selections: pd.DataFrame, domains: list[str]
) -> pd.DataFrame:
    v322 = pd.read_parquet(V322_PREDICTIONS)[
        ["domain", "cell_id", "target_cycle_number", "truth", "raw_baseline"]
    ]
    v327 = pd.read_parquet(V327_PREDICTIONS)[
        [
            "domain",
            "cell_id",
            "target_cycle_number",
            "selected_alpha",
            "selected_causal_baseline",
            "selected_causal_method",
        ]
    ]
    keys = ["domain", "cell_id", "target_cycle_number"]
    blocks: list[pd.DataFrame] = []
    for domain in domains:
        selection = selections.loc[
            selections["outer_target_domain"] == domain
        ].iloc[0]
        raw = v322.loc[v322["domain"].astype(str) == domain].copy()
        pchp = v327.loc[v327["domain"].astype(str) == domain].copy()
        outer = raw.merge(pchp, on=keys, validate="one_to_one")
        control_baseline = causal_baseline(
            outer.rename(columns={"truth": TARGET}),
            float(selection["selected_control_alpha"]),
        )
        offset = float(selection["selected_control_offset"])
        outer["control_alpha"] = float(selection["selected_control_alpha"])
        outer["control_offset"] = offset
        outer["candidate_free_control"] = np.clip(
            control_baseline + offset, 0.0, 1.3
        )
        outer["control_causal_baseline"] = control_baseline
        blocks.append(outer)
    return pd.concat(blocks, ignore_index=True)


def score_outer(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rows = predictions[
        ["domain", "cell_id", "target_cycle_number", "truth"]
    ].copy()
    truth = rows["truth"].to_numpy(float)
    rows["pchp_absolute_error"] = np.abs(
        predictions["selected_causal_method"].to_numpy(float) - truth
    )
    rows["control_absolute_error"] = np.abs(
        predictions["candidate_free_control"].to_numpy(float) - truth
    )
    cells = (
        rows.groupby(["domain", "cell_id"], as_index=False)
        .agg(
            pchp_cell_mae=("pchp_absolute_error", "mean"),
            control_cell_mae=("control_absolute_error", "mean"),
            prediction_rows=("truth", "size"),
        )
    )
    cells["pchp_minus_control"] = (
        cells["pchp_cell_mae"] - cells["control_cell_mae"]
    )
    domains = (
        cells.groupby("domain", as_index=False)
        .agg(
            pchp_cell_macro_mae=("pchp_cell_mae", "mean"),
            control_cell_macro_mae=("control_cell_mae", "mean"),
            physical_cells=("cell_id", "nunique"),
            prediction_rows=("prediction_rows", "sum"),
        )
    )
    domains["pchp_minus_control"] = (
        domains["pchp_cell_macro_mae"] - domains["control_cell_macro_mae"]
    )
    differences = domains["pchp_minus_control"].to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        len(differences),
        size=(BOOTSTRAP_REPLICATES, len(differences)),
    )
    bootstrap = differences[indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap, [2.5, 97.5])
    comparison = {
        "domain_equal_pchp_cell_macro_mae": float(
            domains["pchp_cell_macro_mae"].mean()
        ),
        "domain_equal_candidate_free_control_cell_macro_mae": float(
            domains["control_cell_macro_mae"].mean()
        ),
        "domain_equal_pchp_minus_control": float(differences.mean()),
        "domain_cluster_percentile_bootstrap_ci95": [
            float(lower),
            float(upper),
        ],
        "domain_wins": int((differences < -TOLERANCE).sum()),
        "domain_ties": int((np.abs(differences) <= TOLERANCE).sum()),
        "domain_losses": int((differences > TOLERANCE).sum()),
        "maximum_domain_harm_vs_control": float(differences.max()),
    }
    comparison["candidate_information_gate_passed"] = bool(
        comparison["domain_equal_pchp_minus_control"] < 0.0
        and comparison["domain_cluster_percentile_bootstrap_ci95"][1] < 0.0
    )
    return cells, domains, comparison


def main() -> None:
    started = time.perf_counter()
    if not PREFREEZE.exists():
        raise FileNotFoundError(PREFREEZE)
    for path in (V322_PREDICTIONS, V327_PREDICTIONS):
        if not path.exists():
            raise FileNotFoundError(path)
    frozen = json.loads(PREFREEZE.read_text(encoding="utf-8"))
    if frozen.get("status") != (
        "FROZEN_BEFORE_CANDIDATE_INFORMATION_CONTROL_EXECUTION"
    ):
        raise RuntimeError("candidate-information prefreeze status mismatch")
    expected = frozen["frozen_artifacts"]
    bound_paths = {
        "script": Path(__file__).resolve(),
        "v322_predictions": V322_PREDICTIONS,
        "v327_predictions": V327_PREDICTIONS,
    }
    for name, path in bound_paths.items():
        observed = sha256_file(path)
        if observed != expected[name]["sha256"]:
            raise RuntimeError(
                f"frozen artifact hash mismatch for {name}: {observed}"
            )
    evaluation = add_context_change(load_data())
    evaluation = evaluation.loc[evaluation["after_initial5_reference_window"]].copy()
    domains = sorted(evaluation["domain"].astype(str).unique())
    inner, selections = build_source_only_selections(evaluation, domains)
    if (
        len(selections) != len(domains)
        or selections["inner_domains"].min() != len(domains) - 1
        or selections["outer_target_labels_used_for_selection"].any()
    ):
        raise RuntimeError("candidate-free control selection isolation failed")
    predictions = build_outer_predictions(selections, domains)
    cells, domain_metrics, comparison = score_outer(predictions)
    status = (
        "CANDIDATE_INFORMATION_CONTROL_GATE_PASSED"
        if comparison["candidate_information_gate_passed"]
        else "CANDIDATE_INFORMATION_CONTROL_GATE_NOT_PASSED"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "inner": OUT / "candidate_free_inner_metrics_v333.csv",
        "selections": OUT / "candidate_free_selections_v333.csv",
        "predictions": OUT / "candidate_information_outer_predictions_v333.parquet",
        "cells": OUT / "candidate_information_cell_metrics_v333.csv",
        "domains": OUT / "candidate_information_domain_metrics_v333.csv",
        "report": OUT / "candidate_information_control_v333_report.json",
    }
    inner.to_csv(paths["inner"], index=False)
    selections.to_csv(paths["selections"], index=False)
    predictions.to_parquet(paths["predictions"], index=False)
    cells.to_csv(paths["cells"], index=False)
    domain_metrics.to_csv(paths["domains"], index=False)
    report = {
        "status": status,
        "generated_at_local": datetime.now().astimezone().isoformat(),
        "question": (
            "Does the learned candidate add domain-level accuracy beyond a "
            "candidate-free source-selected constant offset inside the same harm tube?"
        ),
        "control": {
            "alpha_grid": list(ALPHA_GRID),
            "offset_grid": list(OFFSET_GRID),
            "selection": (
                "jointly minimize inner-domain-equal cell-macro MAE using only "
                "the eleven source domains for each outer target"
            ),
            "outer_target_labels_used_for_selection": False,
        },
        "comparison": comparison,
        "interpretation_if_failed": (
            "Do not claim that candidate-specific information is necessary; "
            "retain only the bounded causal projection contribution and redesign "
            "the candidate mechanism on a new validation surface."
        ),
        "nasa_artifacts_accessed": False,
        "frozen_inputs": {
            "prefreeze": {"path": str(PREFREEZE), "sha256": sha256_file(PREFREEZE)},
            "v322_predictions": {
                "path": str(V322_PREDICTIONS),
                "sha256": sha256_file(V322_PREDICTIONS),
            },
            "v327_predictions": {
                "path": str(V327_PREDICTIONS),
                "sha256": sha256_file(V327_PREDICTIONS),
            },
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "runtime_seconds": time.perf_counter() - started,
        "outputs": {},
    }
    for name, path in paths.items():
        if name == "report":
            continue
        report["outputs"][name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    paths["report"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{status}", flush=True)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
