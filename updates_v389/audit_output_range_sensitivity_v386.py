"""Frozen upper-output-guard sensitivity audit for PCHP.

The audit replays the same projection operator on saved pre-projection
predictions at upper guards 1.1, 1.2, 1.3 and 1.5. Development and the six
external surfaces form the compatible evidence chain; BaSyTec and NASA are
reported as boundary surfaces and cannot select the guard.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prefix_causal_harm_projection_v321 import prefix_causal_cellwise_projection


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "OUTPUT_RANGE_SENSITIVITY_PROTOCOL_V386.json"
OUT = ROOT / "output_range_sensitivity_v386"
CAPS = (1.1, 1.2, 1.3, 1.5)
LOWER = 0.0
DELTA = 0.01
TOL = 1e-12


SURFACES = {
    "development": {
        "path": ROOT / "bounded_recovery_pchp_v374" / "bounded_recovery_outer_predictions_v374.parquet",
        "columns": {
            "truth": "truth", "raw_baseline": "raw_baseline", "raw_candidate": "raw_candidate",
            "saved_method": "strict_method", "saved_state": "strict_state",
        },
        "alpha_column": "strict_selected_alpha",
        "expected": (12, 586, 601_932),
        "compatible": True,
    },
    "external": {
        "path": ROOT / "external_mechanism_decision_v380" / "external_mechanism_predictions_v380.parquet",
        "columns": {
            "truth": "truth", "raw_baseline": "raw_baseline", "raw_candidate": "raw_candidate",
            "saved_method": "pchp_method", "saved_state": "protected_state",
        },
        "alpha": 0.01,
        "expected": (6, 659, 9_712),
        "compatible": True,
    },
    "basytec": {
        "path": ROOT / "external_basytec_v352" / "scored_v354" / "basytec_scored_records_v354.parquet",
        "columns": {
            "truth": "truth", "raw_baseline": "raw_baseline_prediction", "raw_candidate": "raw_candidate_prediction",
            "saved_method": "prefix_causal_harm_capped_prediction", "saved_state": "causal_baseline_prediction",
        },
        "alpha": 0.01,
        "expected": (1, 45, 2_969),
        "compatible": True,
    },
    "nasa": {
        "path": ROOT / "external_nasa_v329" / "label_blind_v330" / "nasa_frozen_predictions_v330.parquet",
        "score_path": ROOT / "external_nasa_v329" / "scored_v331" / "nasa_scored_predictions_v331.parquet",
        "score_truth_column": "soh_initial5",
        "columns": {
            "raw_baseline": "raw_baseline_prediction", "raw_candidate": "raw_candidate_prediction",
            "saved_method": "prefix_causal_harm_capped_prediction", "saved_state": "causal_baseline_prediction",
        },
        "alpha": 0.01,
        "expected": (1, 34, 2_598),
        "expected_scored": 2_556,
        "compatible": False,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_surface(name: str, spec: dict[str, object], protocol: dict[str, object]) -> pd.DataFrame:
    expected_hash = protocol["inputs"][name]["sha256"]
    if sha256_file(spec["path"]) != expected_hash:
        raise RuntimeError(f"{name} input hash mismatch")
    frame = pd.read_parquet(spec["path"]).copy()
    if "domain" not in frame:
        frame["domain"] = name.upper()
    rename = {source: target for target, source in spec["columns"].items()}
    frame = frame.rename(columns=rename)
    if "score_path" in spec:
        if sha256_file(spec["score_path"]) != protocol["inputs"][name]["score_sha256"]:
            raise RuntimeError(f"{name} score input hash mismatch")
        keys = ["domain", "cell_id", "target_cycle_number"]
        scores = pd.read_parquet(spec["score_path"])[keys + [spec["score_truth_column"]]].rename(columns={spec["score_truth_column"]: "truth"})
        if scores.duplicated(keys).any():
            raise RuntimeError(f"{name} score keys are not unique")
        frame = frame.merge(scores, on=keys, how="left", validate="one_to_one")
    frame["score_mask"] = frame["truth"].notna()
    needed = {"domain", "cell_id", "target_cycle_number", *spec["columns"].keys()}
    if "alpha_column" in spec:
        needed.add(spec["alpha_column"])
    missing = needed.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{name} missing columns: {sorted(missing)}")
    counts = (frame["domain"].nunique(), frame[["domain", "cell_id"]].drop_duplicates().shape[0], len(frame))
    if counts != spec["expected"]:
        raise RuntimeError(f"{name} roster mismatch: {counts}")
    numeric = ["target_cycle_number", "truth", "raw_baseline", "raw_candidate", "saved_method", "saved_state"]
    if not np.isfinite(frame.loc[frame["score_mask"], numeric].to_numpy(float)).all():
        raise RuntimeError(f"{name} has non-finite audit values")
    if int(frame["score_mask"].sum()) != int(spec.get("expected_scored", len(frame))):
        raise RuntimeError(f"{name} scored roster mismatch")
    return frame


def replay(frame: pd.DataFrame, spec: dict[str, object], cap: float) -> tuple[np.ndarray, np.ndarray]:
    state = np.empty(len(frame), dtype=float)
    method = np.empty(len(frame), dtype=float)
    for domain, rows in frame.groupby("domain", sort=True):
        idx = rows.index.to_numpy()
        if "alpha_column" in spec:
            values = rows[spec["alpha_column"]].drop_duplicates().to_numpy(float)
            if len(values) != 1:
                raise RuntimeError(f"non-unique development alpha in {domain}")
            alpha = float(values[0])
        else:
            alpha = float(spec["alpha"])
        identifiers = (rows["domain"].astype(str) + "::" + rows["cell_id"].astype(str)).to_numpy()
        b, p = prefix_causal_cellwise_projection(
            identifiers,
            rows["target_cycle_number"].to_numpy(float),
            rows["raw_baseline"].to_numpy(float),
            rows["raw_candidate"].to_numpy(float),
            DELTA,
            assimilation=alpha,
            y_min=LOWER,
            y_max=cap,
        )
        state[idx] = b
        method[idx] = p
    return state, method


def nonincreasing(frame: pd.DataFrame, values: np.ndarray) -> bool:
    work = frame[["domain", "cell_id", "target_cycle_number"]].copy()
    work["value"] = values
    for _, rows in work.groupby(["domain", "cell_id"], sort=False):
        ordered = rows.sort_values("target_cycle_number", kind="mergesort")
        if (np.diff(ordered["value"].to_numpy(float)) > TOL).any():
            return False
    return True


def score(frame: pd.DataFrame, values: np.ndarray) -> tuple[float, float, float, pd.DataFrame]:
    mask = frame["score_mask"].to_numpy(bool)
    work = frame.loc[mask, ["domain", "cell_id", "truth"]].copy()
    work["absolute_error"] = np.abs(values[mask] - work["truth"].to_numpy(float))
    cells = work.groupby(["domain", "cell_id"], as_index=False).agg(cell_mae=("absolute_error", "mean"))
    domains = cells.groupby("domain", as_index=False).agg(cell_macro_mae=("cell_mae", "mean"), physical_cells=("cell_id", "nunique"))
    return float(domains["cell_macro_mae"].mean()), float(domains["cell_macro_mae"].max()), float(work["absolute_error"].mean()), domains


def distribution_row(name: str, frame: pd.DataFrame, variable: str) -> dict[str, object]:
    values = frame.loc[frame[variable].notna(), variable].to_numpy(float)
    return {
        "surface": name, "variable": variable, "records": int(len(values)),
        "minimum": float(np.min(values)), "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)), "q99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
        **{f"fraction_above_{cap}": float(np.mean(values > cap)) for cap in CAPS},
    }


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if tuple(protocol["upper_bounds"]) != CAPS:
        raise RuntimeError("cap grid mismatch")
    operator_hash = sha256_file(ROOT / protocol["inputs"]["operator"]["path"])
    if operator_hash != protocol["inputs"]["operator"]["sha256"]:
        raise RuntimeError("projection operator hash mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    cap_rows: list[dict[str, object]] = []
    domain_rows: list[pd.DataFrame] = []
    distribution_rows: list[dict[str, object]] = []
    surface_reports: dict[str, object] = {}

    for name, spec in SURFACES.items():
        frame = load_surface(name, spec, protocol).reset_index(drop=True)
        for variable in ("truth", "raw_baseline", "raw_candidate", "saved_state", "saved_method"):
            distribution_rows.append(distribution_row(name, frame, variable))
        replays: dict[float, tuple[np.ndarray, np.ndarray]] = {cap: replay(frame, spec, cap) for cap in CAPS}
        reference_state, reference_method = replays[1.3]
        reproduction = {
            "state": float(np.max(np.abs(reference_state - frame["saved_state"].to_numpy(float)))),
            "method": float(np.max(np.abs(reference_method - frame["saved_method"].to_numpy(float)))),
        }
        for cap, (state, method) in replays.items():
            domain_mae, worst_mae, record_mae, domains = score(frame, method)
            displacement = np.abs(method - state)
            scored = frame["score_mask"].to_numpy(bool)
            truth = frame.loc[scored, "truth"].to_numpy(float)
            regret = np.abs(method[scored] - truth) - np.abs(state[scored] - truth)
            row = {
                "surface": name, "compatible_target": bool(spec["compatible"]), "upper_bound": cap,
                "domain_equal_cell_macro_mae": domain_mae, "worst_domain_cell_macro_mae": worst_mae,
                "record_equal_mae": record_mae, "maximum_state": float(np.max(state)),
                "maximum_output": float(np.max(method)),
                "raw_baseline_clipped_records": int(np.sum(frame["raw_baseline"].to_numpy(float) > cap)),
                "raw_candidate_above_cap_records": int(np.sum(frame["raw_candidate"].to_numpy(float) > cap)),
                "truth_above_cap_records": int(np.sum(truth > cap)),
                "changed_outputs_vs_1.3": int(np.sum(np.abs(method - reference_method) > TOL)),
                "maximum_output_change_vs_1.3": float(np.max(np.abs(method - reference_method))),
                "maximum_absolute_displacement": float(np.max(displacement)),
                "maximum_observed_absolute_loss_increase": float(np.max(regret)),
                "state_nonincreasing": nonincreasing(frame, state), "output_nonincreasing": nonincreasing(frame, method),
                "range_passed": bool(np.min(method) >= LOWER - TOL and np.max(method) <= cap + TOL),
                "harm_certificate_passed": bool(np.max(displacement) <= DELTA + TOL and np.max(regret) <= DELTA + TOL),
            }
            cap_rows.append(row)
            domains = domains.assign(surface=name, upper_bound=cap)
            domain_rows.append(domains)
        surface_reports[name] = {
            "counts": {"domains": int(frame["domain"].nunique()), "cells": int(frame[["domain", "cell_id"]].drop_duplicates().shape[0]), "projection_records": int(len(frame)), "scored_records": int(frame["score_mask"].sum())},
            "cap_1.3_reproduction_maximum_error": reproduction,
            "compatible_target": bool(spec["compatible"]),
        }

    cap_table = pd.DataFrame(cap_rows)
    distributions = pd.DataFrame(distribution_rows)
    domain_table = pd.concat(domain_rows, ignore_index=True)
    compatible = cap_table[cap_table["compatible_target"]]
    invariant = bool(
        (compatible["changed_outputs_vs_1.3"] == 0).all()
        and (compatible["harm_certificate_passed"]).all()
        and (compatible["range_passed"]).all()
        and (compatible["state_nonincreasing"]).all()
        and (compatible["output_nonincreasing"]).all()
    )
    classification = "NONBINDING" if invariant else "SENSITIVE"
    report = {
        "protocol_version": "V386", "classification": classification,
        "interpretation": (
            "The upper guard is nonbinding on every compatible evidence surface over 1.1--1.5; 1.3 is retained as the pre-existing conservative engineering guard, not as an accuracy-tuned constant."
            if invariant else
            "At least one compatible result changes over the guard grid; manuscript claims require revision."
        ),
        "surface_reports": surface_reports,
        "nasa_boundary_note": "NASA normalized targets above the grid demonstrate target-scale incompatibility and do not select the deployment guard.",
        "artifacts": {
            "protocol_sha256": sha256_file(PROTOCOL),
            "operator_sha256": operator_hash,
        },
    }
    cap_table.to_csv(OUT / "cap_summary_v386.csv", index=False)
    distributions.to_csv(OUT / "saved_value_distributions_v386.csv", index=False)
    domain_table.to_csv(OUT / "domain_metrics_v386.csv", index=False)
    (OUT / "output_range_sensitivity_v386_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
