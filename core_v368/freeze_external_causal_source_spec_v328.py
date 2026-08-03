"""Freeze one source-only assimilation value for untouched external deployment.

This script is deliberately unable to read any external battery data.  It uses
only the twelve already-opened development domains and their leave-one-domain-
out raw predictions.  The selected value is subsequently fixed for every
untouched external cell; external outcomes cannot affect this selection.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from prefix_causal_harm_projection_v321 import prefix_causal_cellwise_projection


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "prefix_causal_rccp_v322" / "prefix_causal_predictions_v322.parquet"
MODEL_BUNDLE = (
    ROOT / "rccp_source_model_freeze_v316" / "rccp_source_models_v316.joblib"
)
PROJECTION_CODE = ROOT / "prefix_causal_harm_projection_v321.py"
PREFREEZE = ROOT / "paper_q1" / "rccp_external_source_alpha_prefreeze_v328.json"
OUT = ROOT / "external_causal_source_freeze_v328"

ALPHA_GRID = (1.0, 0.5, 0.2, 0.1, 0.05, 0.03, 0.02, 0.015, 0.01, 0.005)
BUDGET = 0.01
Y_MIN = 0.0
Y_MAX = 1.3
TOLERANCE = 1e-12
EXPECTED_DOMAINS = (
    "CALB",
    "CALCE",
    "HNEI",
    "HUST",
    "MATR",
    "MICH",
    "MICH_EXP",
    "RWTH",
    "SDU",
    "SNL",
    "UL_PUR",
    "XJTU",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def nonincreasing_by_cell(
    frame: pd.DataFrame, prediction: np.ndarray, tolerance: float = TOLERANCE
) -> bool:
    check = frame[["domain", "cell_id", "target_cycle_number"]].copy()
    check["prediction"] = np.asarray(prediction, dtype=float)
    for _, cell in check.groupby(["domain", "cell_id"], sort=False):
        ordered = cell.sort_values("target_cycle_number", kind="mergesort")
        if (np.diff(ordered["prediction"].to_numpy(float)) > tolerance).any():
            return False
    return True


def validate_input(frame: pd.DataFrame) -> None:
    required = {
        "domain",
        "cell_id",
        "target_cycle_number",
        "truth",
        "raw_baseline",
        "raw_candidate",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"missing required development columns: {missing}")
    domains = tuple(sorted(frame["domain"].astype(str).unique()))
    if domains != EXPECTED_DOMAINS:
        raise RuntimeError(f"unexpected development domains: {domains}")
    keys = ["domain", "cell_id", "target_cycle_number"]
    if frame.duplicated(keys).any():
        raise RuntimeError("duplicate development prediction keys")
    numeric = frame[["target_cycle_number", "truth", "raw_baseline", "raw_candidate"]]
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise RuntimeError("non-finite development values")


def score_alpha(
    frame: pd.DataFrame, alpha: float
) -> tuple[dict[str, object], pd.DataFrame]:
    unit_id = (
        frame["domain"].astype(str) + "::" + frame["cell_id"].astype(str)
    ).to_numpy()
    baseline, method = prefix_causal_cellwise_projection(
        unit_id,
        frame["target_cycle_number"].to_numpy(float),
        frame["raw_baseline"].to_numpy(float),
        frame["raw_candidate"].to_numpy(float),
        BUDGET,
        assimilation=alpha,
        y_min=Y_MIN,
        y_max=Y_MAX,
    )
    truth = frame["truth"].to_numpy(float)
    rows = frame[["domain", "cell_id"]].copy()
    rows["baseline_absolute_error"] = np.abs(baseline - truth)
    rows["method_absolute_error"] = np.abs(method - truth)
    cell = (
        rows.groupby(["domain", "cell_id"], as_index=False)
        .agg(
            baseline_cell_mae=("baseline_absolute_error", "mean"),
            method_cell_mae=("method_absolute_error", "mean"),
            prediction_rows=("method_absolute_error", "size"),
        )
    )
    domain = (
        cell.groupby("domain", as_index=False)
        .agg(
            baseline_cell_macro_mae=("baseline_cell_mae", "mean"),
            method_cell_macro_mae=("method_cell_mae", "mean"),
            physical_cells=("cell_id", "nunique"),
            prediction_rows=("prediction_rows", "sum"),
        )
    )
    domain["method_minus_baseline"] = (
        domain["method_cell_macro_mae"] - domain["baseline_cell_macro_mae"]
    )
    domain.insert(0, "alpha", alpha)

    displacement = np.abs(method - baseline)
    regret = np.abs(method - truth) - np.abs(baseline - truth)
    summary = {
        "alpha": alpha,
        "domain_equal_method_cell_macro_mae": float(
            domain["method_cell_macro_mae"].mean()
        ),
        "domain_equal_baseline_cell_macro_mae": float(
            domain["baseline_cell_macro_mae"].mean()
        ),
        "domain_equal_method_minus_baseline": float(
            domain["method_minus_baseline"].mean()
        ),
        "domain_wins": int((domain["method_minus_baseline"] < -TOLERANCE).sum()),
        "domain_ties": int(
            (domain["method_minus_baseline"].abs() <= TOLERANCE).sum()
        ),
        "domain_losses": int((domain["method_minus_baseline"] > TOLERANCE).sum()),
        "maximum_domain_harm": float(domain["method_minus_baseline"].max()),
        "maximum_absolute_displacement": float(displacement.max()),
        "maximum_observed_absolute_loss_regret": float(regret.max()),
        "baseline_trajectories_nonincreasing": nonincreasing_by_cell(
            frame, baseline
        ),
        "method_trajectories_nonincreasing": nonincreasing_by_cell(frame, method),
    }
    summary["deterministic_certificate_numerically_verified"] = bool(
        summary["maximum_absolute_displacement"] <= BUDGET + TOLERANCE
        and summary["maximum_observed_absolute_loss_regret"]
        <= BUDGET + TOLERANCE
        and summary["baseline_trajectories_nonincreasing"]
        and summary["method_trajectories_nonincreasing"]
    )
    return summary, domain


def select_alpha(summary: pd.DataFrame) -> float:
    best = float(summary["domain_equal_method_cell_macro_mae"].min())
    tied = summary.loc[
        np.abs(summary["domain_equal_method_cell_macro_mae"] - best) <= TOLERANCE
    ]
    return float(tied["alpha"].max())


def main() -> None:
    started = time.perf_counter()
    if not PREFREEZE.exists():
        raise RuntimeError("source-only alpha prefreeze is missing")
    for path in (INPUT, MODEL_BUNDLE, PROJECTION_CODE):
        if not path.exists():
            raise FileNotFoundError(path)

    frame = pd.read_parquet(INPUT)
    validate_input(frame)
    summaries: list[dict[str, object]] = []
    domain_blocks: list[pd.DataFrame] = []
    for alpha in ALPHA_GRID:
        item, domain = score_alpha(frame, alpha)
        summaries.append(item)
        domain_blocks.append(domain)
        print(
            f"alpha={alpha:g}: domain-equal final MAE="
            f"{item['domain_equal_method_cell_macro_mae']:.9f}",
            flush=True,
        )

    summary = pd.DataFrame(summaries).sort_values("alpha", ascending=False)
    domains = pd.concat(domain_blocks, ignore_index=True)
    selected = select_alpha(summary)
    selected_row = summary.loc[summary["alpha"] == selected].iloc[0]
    all_certificates = bool(
        summary["deterministic_certificate_numerically_verified"].all()
    )
    if not all_certificates:
        raise RuntimeError("a structural certificate failed")

    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "external_source_alpha_grid_v328.csv"
    domain_path = OUT / "external_source_alpha_domain_metrics_v328.csv"
    report_path = OUT / "external_source_alpha_freeze_v328_report.json"
    summary.to_csv(summary_path, index=False)
    domains.to_csv(domain_path, index=False)

    report = {
        "status": "SOURCE_ONLY_EXTERNAL_DEPLOYMENT_ALPHA_FROZEN",
        "generated_at_local": datetime.now().astimezone().isoformat(),
        "external_data_accessed": False,
        "selection_population": "12 already-opened development domains",
        "selection_predictions": "leave-one-dataset-domain-out raw predictions",
        "independent_selection_unit": "dataset domain",
        "aggregation": "cell-macro MAE within domain, then equal mean across domains",
        "selection_rule": (
            "minimum domain-equal final-method cell-macro MAE; largest alpha only "
            "for numerical ties within 1e-12"
        ),
        "alpha_grid": list(ALPHA_GRID),
        "selected_alpha": selected,
        "selected_metrics": {
            key: (
                bool(value)
                if isinstance(value, (bool, np.bool_))
                else int(value)
                if isinstance(value, (int, np.integer))
                else float(value)
                if isinstance(value, (float, np.floating))
                else value
            )
            for key, value in selected_row.to_dict().items()
        },
        "budget_soh_units": BUDGET,
        "prediction_bounds": [Y_MIN, Y_MAX],
        "development_domains": list(EXPECTED_DOMAINS),
        "development_physical_cells": int(
            frame[["domain", "cell_id"]].drop_duplicates().shape[0]
        ),
        "development_prediction_rows": int(len(frame)),
        "all_grid_certificates_verified": all_certificates,
        "frozen_inputs": {
            "prefreeze": {
                "path": str(PREFREEZE),
                "sha256": sha256_file(PREFREEZE),
            },
            "development_lodo_predictions": {
                "path": str(INPUT),
                "sha256": sha256_file(INPUT),
            },
            "all_source_model_bundle": {
                "path": str(MODEL_BUNDLE),
                "sha256": sha256_file(MODEL_BUNDLE),
            },
            "projection_implementation": {
                "path": str(PROJECTION_CODE),
                "sha256": sha256_file(PROJECTION_CODE),
            },
            "selection_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"selected external deployment alpha={selected:g}", flush=True)
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
