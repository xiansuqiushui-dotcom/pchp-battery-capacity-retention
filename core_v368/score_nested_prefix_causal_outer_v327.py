"""One-shot outer scoring of frozen source-only PCHP alpha selections."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from develop_anchor_invariant_soh_v306 import (
    TARGET,
    cell_metrics,
    domain_summary,
    paired_domain_comparison,
)
from prefix_causal_harm_projection_v321 import (
    prefix_causal_cellwise_projection,
)


ROOT = Path(__file__).resolve().parent
SELECTION_REPORT = (
    ROOT
    / "nested_prefix_causal_selection_v326"
    / "nested_source_only_alpha_selection_v326_report.json"
)
OUTER_PREDICTIONS = (
    ROOT / "prefix_causal_rccp_v322" / "prefix_causal_predictions_v322.parquet"
)
OUT = ROOT / "nested_prefix_causal_outer_v327"
BUDGET = 0.01
TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def all_trajectories_nonincreasing(
    frame: pd.DataFrame, prediction: np.ndarray
) -> bool:
    working = frame[["cell_id", "target_cycle_number"]].copy()
    working["prediction"] = np.asarray(prediction, dtype=float)
    for _, cell in working.groupby("cell_id", sort=False):
        ordered = cell.sort_values("target_cycle_number", kind="mergesort")
        if (np.diff(ordered["prediction"].to_numpy(float)) > TOLERANCE).any():
            return False
    return True


def build_outer_predictions(
    source: pd.DataFrame, selections: dict[str, float]
) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for domain, rows in source.groupby("domain", sort=True):
        domain_name = str(domain)
        if domain_name not in selections:
            raise RuntimeError(f"missing frozen alpha for {domain_name}")
        alpha = selections[domain_name]
        baseline, method = prefix_causal_cellwise_projection(
            rows["cell_id"].astype(str).to_numpy(),
            rows["target_cycle_number"].to_numpy(float),
            rows["raw_baseline"].to_numpy(float),
            rows["raw_candidate"].to_numpy(float),
            BUDGET,
            assimilation=alpha,
        )
        block = rows[
            ["domain", "cell_id", "target_cycle_number", "truth"]
        ].copy()
        block["selected_alpha"] = alpha
        block["selected_causal_baseline"] = baseline
        block["selected_causal_method"] = method
        block["offline_baseline"] = rows["offline_baseline"].to_numpy(float)
        block["offline_harm_capped"] = rows["offline_harm_capped"].to_numpy(float)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def score(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = predictions[
        ["domain", "cell_id", "target_cycle_number", "truth"]
    ].rename(columns={"truth": TARGET})
    methods = (
        "selected_causal_baseline",
        "selected_causal_method",
        "offline_baseline",
        "offline_harm_capped",
    )
    cells = pd.concat(
        [
            cell_metrics(frame, predictions[method].to_numpy(float), method)
            for method in methods
        ],
        ignore_index=True,
    )
    domains = domain_summary(cells)
    summary = (
        domains.groupby("method", as_index=False)
        .agg(
            domain_equal_cell_macro_mae=("cell_macro_mae", "mean"),
            worst_domain_cell_macro_mae=("cell_macro_mae", "max"),
            domain_equal_trajectory_increase_fraction=(
                "mean_trajectory_increase_fraction",
                "mean",
            ),
        )
        .sort_values("domain_equal_cell_macro_mae")
    )
    return cells, domains, summary


def comparison(
    domains: pd.DataFrame, method: str, baseline: str
) -> dict[str, object]:
    item = paired_domain_comparison(domains, method, baseline)
    return {
        "method": method,
        "baseline": baseline,
        "domain_equal_mean_difference": item["domain_equal_mean_difference"],
        "ci95_lower": item["ci95_domain_cluster_percentile"][0],
        "ci95_upper": item["ci95_domain_cluster_percentile"][1],
        "domain_wins": item["domain_wins_ties_losses"][0],
        "domain_ties": item["domain_wins_ties_losses"][1],
        "domain_losses": item["domain_wins_ties_losses"][2],
        "maximum_domain_harm": item["maximum_domain_harm"],
        "per_domain_difference": item["per_domain_difference"],
    }


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    selection_report = json.loads(SELECTION_REPORT.read_text(encoding="utf-8"))
    if (
        selection_report["status"]
        != "SOURCE_ONLY_NESTED_ALPHA_SELECTIONS_FROZEN_BEFORE_OUTER_SCORING"
    ):
        raise RuntimeError("source-only selections are not frozen")
    if selection_report["outer_target_labels_used_for_selection"]:
        raise RuntimeError("outer target label isolation failed")
    selections = {
        str(item["outer_target_domain"]): float(item["selected_alpha"])
        for item in selection_report["selections"]
    }
    source = pd.read_parquet(OUTER_PREDICTIONS)
    predictions = build_outer_predictions(source, selections)
    cells, domains, summary = score(predictions)
    primary = comparison(
        domains, "selected_causal_method", "selected_causal_baseline"
    )
    versus_offline = comparison(
        domains, "selected_causal_method", "offline_harm_capped"
    )
    baseline_versus_offline = comparison(
        domains, "selected_causal_baseline", "offline_baseline"
    )

    baseline = predictions["selected_causal_baseline"].to_numpy(float)
    method = predictions["selected_causal_method"].to_numpy(float)
    truth = predictions["truth"].to_numpy(float)
    maximum_displacement = float(np.max(np.abs(method - baseline)))
    maximum_observed_regret = float(
        np.max(np.abs(method - truth) - np.abs(baseline - truth))
    )
    frame = predictions[["cell_id", "target_cycle_number"]]
    baseline_monotone = all_trajectories_nonincreasing(frame, baseline)
    method_monotone = all_trajectories_nonincreasing(frame, method)
    certificate_passed = bool(
        maximum_displacement <= BUDGET + TOLERANCE
        and maximum_observed_regret <= BUDGET + TOLERANCE
        and baseline_monotone
        and method_monotone
    )
    mae_lookup = dict(
        zip(summary["method"], summary["domain_equal_cell_macro_mae"])
    )
    causal_not_worse_than_offline = bool(
        mae_lookup["selected_causal_method"]
        <= mae_lookup["offline_harm_capped"] + TOLERANCE
    )
    gate_passed = bool(
        primary["domain_equal_mean_difference"] < 0.0
        and primary["ci95_upper"] < 0.0
        and primary["maximum_domain_harm"] <= BUDGET + TOLERANCE
        and certificate_passed
        and causal_not_worse_than_offline
    )

    paths = {
        "summary": OUT / "nested_outer_summary_v327.csv",
        "domain_metrics": OUT / "nested_outer_domain_metrics_v327.csv",
        "cell_metrics": OUT / "nested_outer_cell_metrics_v327.csv",
        "predictions": OUT / "nested_outer_predictions_v327.parquet",
        "report": OUT / "nested_prefix_causal_outer_v327_report.json",
    }
    summary.to_csv(paths["summary"], index=False)
    domains.to_csv(paths["domain_metrics"], index=False)
    cells.to_csv(paths["cell_metrics"], index=False)
    predictions.to_parquet(paths["predictions"], index=False)
    report = {
        "status": (
            "NESTED_PREFIX_CAUSAL_OUTER_GATE_PASSED"
            if gate_passed
            else "NESTED_PREFIX_CAUSAL_OUTER_GATE_NOT_PASSED"
        ),
        "selection_report": {
            "path": str(SELECTION_REPORT),
            "sha256": sha256_file(SELECTION_REPORT),
        },
        "outer_predictions_input": {
            "path": str(OUTER_PREDICTIONS),
            "sha256": sha256_file(OUTER_PREDICTIONS),
        },
        "independent_unit": "complete battery dataset domain",
        "domains": int(predictions["domain"].nunique()),
        "physical_cells": int(predictions["cell_id"].nunique()),
        "post_reference_rows": int(len(predictions)),
        "budget_soh_units": BUDGET,
        "outer_target_labels_used_for_alpha_selection": False,
        "selected_alpha_by_domain": selections,
        "selected_alpha_counts": {
            str(key): value for key, value in sorted(Counter(selections.values()).items())
        },
        "summary": json.loads(summary.to_json(orient="records")),
        "primary_comparison": primary,
        "comparison_with_retrospective_method": versus_offline,
        "causal_baseline_comparison_with_retrospective_baseline": (
            baseline_versus_offline
        ),
        "deterministic_certificate": {
            "maximum_absolute_displacement": maximum_displacement,
            "maximum_observed_absolute_loss_regret": maximum_observed_regret,
            "baseline_trajectories_nonincreasing": baseline_monotone,
            "method_trajectories_nonincreasing": method_monotone,
            "passed": certificate_passed,
        },
        "causal_method_not_worse_than_retrospective_method": (
            causal_not_worse_than_offline
        ),
        "decision": "RETAIN" if gate_passed else "REJECT",
        "limitations": [
            "All twelve outer domains were historically opened; the nested design prevents current target-label tuning but is development evidence.",
            "The selected causal method requires a new frozen external domain before replacing the V318-confirmed retrospective method.",
            "Strict non-increase remains inappropriate when local capacity regeneration is a target phenomenon rather than measurement noise.",
        ],
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
