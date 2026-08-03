"""Robustness audit for the six frozen V380 external battery datasets.

The complete external battery dataset is the top-level independent unit.
Physical cells and post-reference records are nested observations.  The
primary estimand is the dataset-equal mean of within-dataset cell-macro MAE
differences between PCHP and the matched constant-offset comparator.  All
predictions are reused unchanged; this script performs no fitting or tuning.
"""

from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "external_mechanism_decision_v380" / "external_mechanism_predictions_v380.parquet"
PROTOCOL = ROOT / "EXTERNAL_ROBUSTNESS_PROTOCOL_V384.json"
OUT = ROOT / "external_robustness_v384"
METHOD = "pchp_method"
COMPARATORS = ("fixed_shift", "protected_state")
EXPECTED_INPUT_SHA256 = "474d01b7473ed30aaa79f117ab1a9237e5028201ecab74dbce37932d3ac9f5e3"
EXPECTED_COUNTS = {"datasets": 6, "cells": 659, "records": 9712}
TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_input(frame: pd.DataFrame) -> dict[str, int]:
    required = {
        "domain",
        "cell_id",
        "target_cycle_number",
        "truth",
        METHOD,
        *COMPARATORS,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"missing required columns: {sorted(missing)}")
    # MULTISTAGE_50E legitimately contains two distinct diagnostic records at
    # the same nominal target-cycle number.  The frozen row is therefore the
    # record unit; target_cycle_number is an ordering variable, not a unique
    # key.  Exact duplicate rows would still indicate accidental replication.
    if frame.duplicated().any():
        raise RuntimeError("exact duplicate external prediction rows")
    if frame[list(required.difference({"domain", "cell_id"}))].isna().any().any():
        raise RuntimeError("non-finite or missing numeric input")
    counts = {
        "datasets": int(frame["domain"].nunique()),
        "cells": int(frame[["domain", "cell_id"]].drop_duplicates().shape[0]),
        "records": int(len(frame)),
    }
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"unexpected external roster: {counts}")
    return counts


def build_error_ledgers(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = frame[["domain", "cell_id", "target_cycle_number", "truth"]].copy()
    for method in (METHOD, *COMPARATORS):
        records[f"absolute_error_{method}"] = np.abs(
            frame[method].to_numpy(float) - frame["truth"].to_numpy(float)
        )
    cell_columns = [f"absolute_error_{method}" for method in (METHOD, *COMPARATORS)]
    cells = (
        records.groupby(["domain", "cell_id"], as_index=False)
        .agg(
            records=("target_cycle_number", "size"),
            **{f"mae_{method}": (f"absolute_error_{method}", "mean") for method in (METHOD, *COMPARATORS)},
        )
        .sort_values(["domain", "cell_id"])
        .reset_index(drop=True)
    )
    if records[cell_columns].isna().any().any() or cells.isna().any().any():
        raise RuntimeError("missing values in error ledgers")
    return records, cells


def exact_sign_flip(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    signs = np.asarray(list(product((-1.0, 1.0), repeat=len(values))), dtype=float)
    distribution = (signs * values[None, :]).mean(axis=1)
    observed = float(values.mean())
    p_value = float(np.mean(np.abs(distribution) >= abs(observed) - 1e-15))
    return {
        "assignments": int(len(distribution)),
        "observed_mean": observed,
        "two_sided_p": p_value,
        "minimum_attainable_two_sided_p": float(2.0 / (2 ** len(values))),
    }


def comparison_results(
    records: pd.DataFrame,
    cells: pd.DataFrame,
    comparator: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    cell = cells.copy()
    cell["difference"] = cell[f"mae_{METHOD}"] - cell[f"mae_{comparator}"]
    domain = (
        cell.groupby("domain", as_index=False)
        .agg(
            cells=("cell_id", "size"),
            records=("records", "sum"),
            pchp_cell_macro_mae=(f"mae_{METHOD}", "mean"),
            comparator_cell_macro_mae=(f"mae_{comparator}", "mean"),
            difference=("difference", "mean"),
        )
        .sort_values("domain")
        .reset_index(drop=True)
    )
    record_difference = (
        records[f"absolute_error_{METHOD}"] - records[f"absolute_error_{comparator}"]
    )
    values = domain["difference"].to_numpy(float)
    leave_one_out_rows: list[dict[str, object]] = []
    for omitted in domain["domain"]:
        retained = domain.loc[domain["domain"] != omitted, "difference"].to_numpy(float)
        leave_one_out_rows.append(
            {
                "comparator": comparator,
                "omitted_domain": omitted,
                "retained_domains": int(len(retained)),
                "dataset_equal_mean_difference": float(retained.mean()),
            }
        )
    leave_one_out = pd.DataFrame(leave_one_out_rows)
    largest_by_cells = str(domain.sort_values(["cells", "records", "domain"]).iloc[-1]["domain"])
    largest_by_records = str(domain.sort_values(["records", "cells", "domain"]).iloc[-1]["domain"])
    without_largest_cells = float(
        domain.loc[domain["domain"] != largest_by_cells, "difference"].mean()
    )
    without_largest_records = float(
        domain.loc[domain["domain"] != largest_by_records, "difference"].mean()
    )
    wins = int(np.sum(values < -TOLERANCE))
    ties = int(np.sum(np.abs(values) <= TOLERANCE))
    losses = int(np.sum(values > TOLERANCE))
    result = {
        "method": METHOD,
        "comparator": comparator,
        "difference_direction": "negative favors PCHP",
        "estimands": {
            "dataset_equal_cell_macro_mean_difference": float(values.mean()),
            "cell_equal_mean_difference": float(cell["difference"].mean()),
            "record_equal_mean_difference": float(record_difference.mean()),
            "median_dataset_difference": float(np.median(values)),
            "median_cell_difference": float(np.median(cell["difference"].to_numpy(float))),
        },
        "exact_dataset_sign_flip": exact_sign_flip(values),
        "dataset_wins_ties_losses": [wins, ties, losses],
        "leave_one_dataset_out": {
            "minimum_mean_difference": float(leave_one_out["dataset_equal_mean_difference"].min()),
            "maximum_mean_difference": float(leave_one_out["dataset_equal_mean_difference"].max()),
            "all_negative": bool((leave_one_out["dataset_equal_mean_difference"] < 0.0).all()),
        },
        "largest_dataset_sensitivity": {
            "largest_by_cells": largest_by_cells,
            "difference_without_largest_by_cells": without_largest_cells,
            "largest_by_records": largest_by_records,
            "difference_without_largest_by_records": without_largest_records,
        },
    }
    domain.insert(0, "comparator", comparator)
    return result, domain, leave_one_out


def classify_primary(result: dict[str, object]) -> str:
    estimands = result["estimands"]
    exact = result["exact_dataset_sign_flip"]
    leave_one_out = result["leave_one_dataset_out"]
    largest = result["largest_dataset_sensitivity"]
    mean_negative = estimands["dataset_equal_cell_macro_mean_difference"] < 0.0
    if not mean_negative:
        return "CONTRADICTED"
    robust = (
        exact["two_sided_p"] <= 0.05
        and leave_one_out["all_negative"]
        and largest["difference_without_largest_by_cells"] < 0.0
        and largest["difference_without_largest_by_records"] < 0.0
    )
    return "ROBUST" if robust else "QUALIFIED"


def main() -> None:
    if sha256_file(INPUT) != EXPECTED_INPUT_SHA256:
        raise RuntimeError("frozen V380 prediction hash mismatch")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["input"]["sha256"] != EXPECTED_INPUT_SHA256:
        raise RuntimeError("protocol and implementation disagree on input hash")
    frame = pd.read_parquet(INPUT)
    counts = validate_input(frame)
    records, cells = build_error_ledgers(frame)

    results: dict[str, dict[str, object]] = {}
    domain_blocks: list[pd.DataFrame] = []
    leave_one_out_blocks: list[pd.DataFrame] = []
    for comparator in COMPARATORS:
        result, domain, leave_one_out = comparison_results(records, cells, comparator)
        results[comparator] = result
        domain_blocks.append(domain)
        leave_one_out_blocks.append(leave_one_out)

    primary_decision = classify_primary(results["fixed_shift"])
    OUT.mkdir(parents=True, exist_ok=True)
    domain_path = OUT / "external_robustness_domain_metrics_v384.csv"
    cell_path = OUT / "external_robustness_cell_metrics_v384.csv"
    leave_one_out_path = OUT / "external_robustness_leave_one_out_v384.csv"
    report_path = OUT / "external_robustness_v384_report.json"
    pd.concat(domain_blocks, ignore_index=True).to_csv(domain_path, index=False)
    cells.to_csv(cell_path, index=False)
    pd.concat(leave_one_out_blocks, ignore_index=True).to_csv(leave_one_out_path, index=False)

    report = {
        "version": "v384",
        "status": f"EXTERNAL_ROBUSTNESS_{primary_decision}",
        "classification": protocol["classification"],
        "information_boundary": {
            "top_level_independent_unit": "complete external battery dataset surface",
            "nested_units": ["physical cell", "post-reference record"],
            **counts,
            "model_refitting": False,
            "external_target_aware_selection": False,
        },
        "multiplicity": {
            "confirmatory_test_family": "one primary exact test: PCHP versus fixed shift",
            "protected_state_test_role": "secondary descriptive sensitivity",
            "adjustment": "not applicable to the single primary test; no confirmatory interpretation assigned to secondary p-value",
        },
        "primary_interpretation": primary_decision,
        "comparisons": results,
        "artifacts": {},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    artifacts = {
        "protocol": PROTOCOL,
        "implementation": Path(__file__).resolve(),
        "input": INPUT,
        "domain_metrics": domain_path,
        "cell_metrics": cell_path,
        "leave_one_out": leave_one_out_path,
    }
    report["artifacts"] = {
        name: {"path": str(path), "sha256": sha256_file(path), "bytes": int(path.stat().st_size)}
        for name, path in artifacts.items()
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
