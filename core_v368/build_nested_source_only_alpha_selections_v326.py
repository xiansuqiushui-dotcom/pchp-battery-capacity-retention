"""Build source-only nested alpha selections for prefix-causal RCCP.

For each outer target domain, alpha is selected by leave-one-domain-out
validation over the remaining source domains.  Pair caches contain predictions
and identifiers only; labels of the two excluded domains are not serialized.
The final outer-domain outcomes are not scored in this script.
"""

from __future__ import annotations

import hashlib
import itertools
import json
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
from prefix_causal_harm_projection_v321 import (
    prefix_causal_cellwise_projection,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "nested_prefix_causal_selection_v326"
CACHE = OUT / "double_holdout_prediction_cache"
ALPHA_GRID = (1.0, 0.5, 0.2, 0.1, 0.05, 0.03, 0.02, 0.015, 0.01, 0.005)
BUDGET = 0.01
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


def validate_cache(path: Path, expected_domains: tuple[str, str]) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return False
    required = {
        "domain",
        "cell_id",
        "target_cycle_number",
        "raw_baseline",
        "raw_candidate",
    }
    return bool(
        set(frame.columns) == required
        and set(frame["domain"].astype(str).unique()) == set(expected_domains)
        and not frame.isna().any().any()
    )


def build_double_holdout_caches(
    evaluation: pd.DataFrame, domains: list[str]
) -> list[dict[str, object]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, object]] = []
    pairs = list(itertools.combinations(domains, 2))
    for pair_index, (first, second) in enumerate(pairs, start=1):
        path = cache_path(first, second)
        if not validate_cache(path, (first, second)):
            source = evaluation.loc[
                ~evaluation["domain"].astype(str).isin((first, second))
            ].copy()
            target = evaluation.loc[
                evaluation["domain"].astype(str).isin((first, second))
            ].copy()
            source_fit = balanced_source_rows(source)
            target_features = target.drop(columns=[TARGET])
            raw_baseline = fit_predict(
                source_fit,
                target_features,
                RAW_FEATURES,
                domain_equal=False,
            )
            raw_candidate = fit_predict(
                source_fit,
                target_features,
                RAW_ABSOLUTE_CHANGE_FEATURES,
                domain_equal=True,
            )
            prediction_only = target[
                ["domain", "cell_id", "target_cycle_number"]
            ].copy()
            prediction_only["raw_baseline"] = raw_baseline
            prediction_only["raw_candidate"] = raw_candidate
            prediction_only.to_parquet(path, index=False)
        frame = pd.read_parquet(path)
        metadata.append(
            {
                "excluded_domains": [first, second],
                "prediction_rows": int(len(frame)),
                "physical_cells": int(frame["cell_id"].nunique()),
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
        print(
            f"double-holdout cache {pair_index}/{len(pairs)} ready: {first}, {second}",
            flush=True,
        )
    return metadata


def cell_macro_mae(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    working = frame[["cell_id", TARGET]].copy()
    working["absolute_error"] = np.abs(
        np.asarray(prediction, dtype=float) - working[TARGET].to_numpy(float)
    )
    return float(working.groupby("cell_id")["absolute_error"].mean().mean())


def build_selections(
    evaluation: pd.DataFrame, domains: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inner_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    keys = ["domain", "cell_id", "target_cycle_number"]
    for outer_index, outer_domain in enumerate(domains, start=1):
        source_pool = evaluation.loc[
            evaluation["domain"].astype(str) != outer_domain
        ].copy()
        source_domains = sorted(source_pool["domain"].astype(str).unique())
        for inner_domain in source_domains:
            predictions = pd.read_parquet(cache_path(outer_domain, inner_domain))
            predictions = predictions.loc[
                predictions["domain"].astype(str) == inner_domain
            ].copy()
            validation = source_pool.loc[
                source_pool["domain"].astype(str) == inner_domain
            ].merge(predictions, on=keys, how="inner", validate="one_to_one")
            if len(validation) != len(predictions):
                raise RuntimeError(
                    f"inner alignment failed for outer={outer_domain}, inner={inner_domain}"
                )
            for alpha in ALPHA_GRID:
                causal_baseline, causal_method = prefix_causal_cellwise_projection(
                    validation["cell_id"].astype(str).to_numpy(),
                    validation["target_cycle_number"].to_numpy(float),
                    validation["raw_baseline"].to_numpy(float),
                    validation["raw_candidate"].to_numpy(float),
                    BUDGET,
                    assimilation=alpha,
                )
                inner_rows.append(
                    {
                        "outer_target_domain": outer_domain,
                        "inner_validation_domain": inner_domain,
                        "alpha": alpha,
                        "physical_cells": int(validation["cell_id"].nunique()),
                        "validation_rows": int(len(validation)),
                        "baseline_cell_macro_mae": cell_macro_mae(
                            validation, causal_baseline
                        ),
                        "method_cell_macro_mae": cell_macro_mae(
                            validation, causal_method
                        ),
                    }
                )
        outer_inner = pd.DataFrame(
            [row for row in inner_rows if row["outer_target_domain"] == outer_domain]
        )
        aggregate = (
            outer_inner.groupby("alpha", as_index=False)
            .agg(
                inner_domain_equal_baseline_mae=(
                    "baseline_cell_macro_mae",
                    "mean",
                ),
                inner_domain_equal_method_mae=("method_cell_macro_mae", "mean"),
                inner_domains=("inner_validation_domain", "nunique"),
            )
            .sort_values(
                ["inner_domain_equal_method_mae", "alpha"],
                ascending=[True, False],
            )
        )
        minimum = float(aggregate["inner_domain_equal_method_mae"].min())
        tied = aggregate.loc[
            aggregate["inner_domain_equal_method_mae"] <= minimum + TOLERANCE
        ]
        selected = tied.sort_values("alpha", ascending=False).iloc[0]
        selection_rows.append(
            {
                "outer_target_domain": outer_domain,
                "selected_alpha": float(selected["alpha"]),
                "selected_inner_domain_equal_method_mae": float(
                    selected["inner_domain_equal_method_mae"]
                ),
                "selected_inner_domain_equal_baseline_mae": float(
                    selected["inner_domain_equal_baseline_mae"]
                ),
                "inner_domains": int(selected["inner_domains"]),
                "outer_target_labels_used_for_selection": False,
            }
        )
        print(
            f"source-only selection {outer_index}/{len(domains)}: "
            f"{outer_domain} -> alpha={float(selected['alpha'])}",
            flush=True,
        )
    return pd.DataFrame(inner_rows), pd.DataFrame(selection_rows)


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    enriched = add_context_change(load_data())
    evaluation = enriched.loc[enriched["after_initial5_reference_window"]].copy()
    domains = sorted(evaluation["domain"].astype(str).unique())
    cache_metadata = build_double_holdout_caches(evaluation, domains)
    inner_metrics, selections = build_selections(evaluation, domains)
    if len(selections) != len(domains) or selections["inner_domains"].min() != 11:
        raise RuntimeError("nested source-only selection roster is incomplete")
    if selections["outer_target_labels_used_for_selection"].any():
        raise RuntimeError("outer target label isolation failed")

    paths = {
        "inner_metrics": OUT / "nested_alpha_inner_metrics_v326.csv",
        "selections": OUT / "nested_alpha_selections_v326.csv",
        "report": OUT / "nested_source_only_alpha_selection_v326_report.json",
    }
    inner_metrics.to_csv(paths["inner_metrics"], index=False)
    selections.to_csv(paths["selections"], index=False)
    report = {
        "status": "SOURCE_ONLY_NESTED_ALPHA_SELECTIONS_FROZEN_BEFORE_OUTER_SCORING",
        "outer_domains": len(domains),
        "inner_domains_per_outer": 11,
        "training_domains_per_inner_model": 10,
        "alpha_grid": list(ALPHA_GRID),
        "selection_metric": "inner-domain-equal cell-macro MAE of the final causal method",
        "tie_breaker": "largest alpha only within absolute tolerance 1e-12",
        "outer_target_labels_used_for_selection": False,
        "oxford_data_accessed": False,
        "selections": json.loads(selections.to_json(orient="records")),
        "double_holdout_caches": cache_metadata,
        "runtime_seconds": float(time.perf_counter() - started),
        "files": {},
    }
    for name, path in paths.items():
        if name != "report":
            report["files"][name] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
    paths["report"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
