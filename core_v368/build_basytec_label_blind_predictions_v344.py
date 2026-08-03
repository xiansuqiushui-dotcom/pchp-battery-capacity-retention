"""Freeze BaSyTec external predictions before discharge-capacity access."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from basytec_external_v343_common import (
    ANCHOR_RECORDS,
    CURRENT_ACTIVITY_THRESHOLD_A,
    EXPECTED_ZIP_COUNT,
    NOMINAL_CAPACITY_AH,
    WINDOW_SECONDS,
    contiguous_blocks,
    detect_schema,
    load_aging_bytes,
    numeric_series,
    ordered_cycle_keys,
    read_prediction_fields,
    sha256_file,
    stable_cycle_key,
    time_seconds,
)
from build_batterylife_early_charge_soh_v109 import early_charge_features
from develop_context_change_soh_v310 import add_context_change
from prefix_causal_harm_projection_v321 import (
    causal_nonincreasing_state,
    prefix_causal_cellwise_projection,
)


ROOT = Path(__file__).resolve().parent
EXTERNAL = ROOT / "external_basytec_v343"
DOWNLOADS = EXTERNAL / "downloads"
DOWNLOAD_RECEIPT = EXTERNAL / "download_receipt_v343.json"
OUT = EXTERNAL / "label_blind_v344"
MODEL_BUNDLE = ROOT / "rccp_source_model_freeze_v316" / "rccp_source_models_v316.joblib"
ALPHA_REPORT = ROOT / "external_causal_source_freeze_v328" / "external_source_alpha_freeze_v328_report.json"
COMPARATOR_REPORT = (
    ROOT
    / "source_tuned_causal_candidate_v342"
    / "source_tuned_causal_candidate_v342_report.json"
)
PROJECTION_CODE = ROOT / "prefix_causal_harm_projection_v321.py"

EXPECTED_MODEL_SHA256 = "F492F4B87C1CE7F49C3718F5C2A7D7DADE79450A9E548B33540452B4DC4817FD"
EXPECTED_ALPHA_REPORT_SHA256 = "CBB9EBC235878B9C1E914F5AE136BDCA38064040D9957D0410DF82A1AE299AE9"
EXPECTED_COMPARATOR_REPORT_SHA256 = "ED11518EA3E1EAA42148BF4421F0570D775074AE697260A8EDF9FBAC0798C252"
EXPECTED_PROJECTION_SHA256 = "CE7288A129C17114E1CA57432C6417BEBA7938D58DB2B1FD0A87171C479EB54C"
ASSIMILATION = 0.01
COMPARATOR_ASSIMILATION = 0.05
BUDGET = 0.01
Y_MIN = 0.0
Y_MAX = 1.3
TOLERANCE = 1e-12


def cumulative_charge_ah(elapsed: np.ndarray, current: np.ndarray) -> np.ndarray:
    time_values = np.asarray(elapsed, dtype=float)
    current_values = np.asarray(current, dtype=float)
    if len(time_values) == 0:
        return np.asarray([], dtype=float)
    increments = np.diff(time_values)
    increments = np.where(np.isfinite(increments), increments, 0.0)
    increments = np.clip(increments, 0.0, 3600.0)
    trapezoids = (
        0.5 * (current_values[:-1] + current_values[1:]) * increments / 3600.0
    )
    return np.concatenate([[0.0], np.cumsum(trapezoids)])


def feature_from_block(
    time_values: np.ndarray,
    current: np.ndarray,
    voltage: np.ndarray,
    temperature: np.ndarray,
) -> tuple[dict[str, float], str | None]:
    finite = np.isfinite(time_values) & np.isfinite(current) & np.isfinite(voltage)
    if finite.sum() < 3:
        return {}, "insufficient_finite_charge_points"
    time_values = time_values[finite]
    current = current[finite]
    voltage = voltage[finite]
    temperature = temperature[finite]
    order = np.argsort(time_values, kind="mergesort")
    time_values = time_values[order]
    current = current[order]
    voltage = voltage[order]
    temperature = temperature[order]
    elapsed = time_values - time_values[0]
    if not np.isfinite(elapsed[-1]) or elapsed[-1] < WINDOW_SECONDS:
        return {}, "charge_window_not_covered"
    charge = cumulative_charge_ah(elapsed, current)
    return early_charge_features(
        {
            "current_in_A": current,
            "voltage_in_V": voltage,
            "charge_capacity_in_Ah": charge,
            "time_in_s": elapsed,
            "temperature_in_C": temperature,
        },
        nominal_capacity_ah=NOMINAL_CAPACITY_AH,
        window_seconds=WINDOW_SECONDS,
    )


def features_for_zip(zip_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    raw, member = load_aging_bytes(zip_path)
    schema = detect_schema(raw)
    frame = read_prediction_fields(raw, schema)
    cycle_keys = frame["cycle_raw"].map(stable_cycle_key)
    current = numeric_series(frame["current_raw"])
    voltage = numeric_series(frame["voltage_raw"])
    times = time_seconds(frame["time_raw"])
    if "temperature_raw" in frame:
        temperature = numeric_series(frame["temperature_raw"])
    else:
        temperature = np.full(len(frame), np.nan, dtype=float)

    rows: list[dict[str, object]] = []
    rejections: Counter[str] = Counter()
    cycle_order = ordered_cycle_keys(frame["cycle_raw"])
    for raw_rank, key in enumerate(cycle_order, start=1):
        group_positions = np.flatnonzero(cycle_keys.to_numpy(str) == key)
        if len(group_positions) == 0:
            continue
        positive_local = np.flatnonzero(
            current[group_positions] > CURRENT_ACTIVITY_THRESHOLD_A
        )
        selected: dict[str, float] | None = None
        last_reason = "no_positive_charge_segment"
        for local_block in contiguous_blocks(positive_local):
            positions = group_positions[local_block]
            feature, reason = feature_from_block(
                times[positions],
                current[positions],
                voltage[positions],
                temperature[positions],
            )
            if reason is None:
                selected = feature
                break
            last_reason = reason
        if selected is None:
            rejections[last_reason] += 1
            continue
        rank = len(rows) + 1
        row: dict[str, object] = {
            "domain": "BAYREUTH_PATH_INDICATORS_2025",
            "cell_id": zip_path.stem,
            "zip_name": zip_path.name,
            "aging_member": member,
            "cycle_key": key,
            "source_cycle_order": raw_rank,
            "aligned_cycle_rank": rank,
            "target_cycle_number": float(rank),
            "after_initial5_reference_window": rank > ANCHOR_RECORDS,
            "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
        }
        row.update(selected)
        rows.append(row)
    output = pd.DataFrame(rows)
    audit = {
        "cell_id": zip_path.stem,
        "zip_name": zip_path.name,
        "zip_sha256": sha256_file(zip_path),
        "aging_member": member,
        "schema": {
            "encoding": schema.encoding,
            "delimiter": repr(schema.delimiter),
            "header_row_zero_based": schema.header_row_zero_based,
            "cycle": schema.cycle,
            "time": schema.time,
            "current": schema.current,
            "voltage": schema.voltage,
            "temperature": schema.temperature,
            "discharge_capacity_name_only": schema.discharge_capacity,
        },
        "source_cycles": len(cycle_order),
        "valid_fixed_window_charge_cycles": len(output),
        "post_anchor_prediction_cycles": max(len(output) - ANCHOR_RECORDS, 0),
        "rejections": dict(sorted(rejections.items())),
    }
    return output, audit


def predict_block(frame: pd.DataFrame, block: dict[str, object]) -> np.ndarray:
    features = list(block["features"])
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise RuntimeError(f"missing frozen model features: {missing}")
    transformed = block["imputer"].transform(frame[features])
    prediction = np.asarray(block["model"].predict(transformed), dtype=float)
    lower, upper = block["prediction_clip"]
    return np.clip(prediction, float(lower), float(upper))


def source_tuned_candidate(
    cell_ids: np.ndarray, cycles: np.ndarray, candidate: np.ndarray
) -> np.ndarray:
    working = pd.DataFrame(
        {
            "cell_id": cell_ids.astype(str),
            "cycle": cycles.astype(float),
            "candidate": candidate.astype(float),
            "position": np.arange(len(candidate)),
        }
    )
    output = np.empty(len(working), dtype=float)
    for _, cell in working.groupby("cell_id", sort=False):
        ordered = cell.sort_values("cycle", kind="mergesort")
        values = causal_nonincreasing_state(
            ordered["candidate"].to_numpy(float),
            assimilation=COMPARATOR_ASSIMILATION,
            y_min=Y_MIN,
            y_max=Y_MAX,
        )
        output[ordered["position"].to_numpy(int)] = values
    return output


def nonincreasing(frame: pd.DataFrame, values: np.ndarray) -> bool:
    working = frame[["cell_id", "target_cycle_number"]].copy()
    working["value"] = np.asarray(values, dtype=float)
    return all(
        bool(
            (
                np.diff(
                    cell.sort_values("target_cycle_number", kind="mergesort")[
                        "value"
                    ].to_numpy(float)
                )
                <= TOLERANCE
            ).all()
        )
        for _, cell in working.groupby("cell_id", sort=False)
    )


def main() -> int:
    started = time.perf_counter()
    required = [
        DOWNLOAD_RECEIPT,
        MODEL_BUNDLE,
        ALPHA_REPORT,
        COMPARATOR_REPORT,
        PROJECTION_CODE,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_hashes = {
        MODEL_BUNDLE: EXPECTED_MODEL_SHA256,
        ALPHA_REPORT: EXPECTED_ALPHA_REPORT_SHA256,
        COMPARATOR_REPORT: EXPECTED_COMPARATOR_REPORT_SHA256,
        PROJECTION_CODE: EXPECTED_PROJECTION_SHA256,
    }
    for path, expected in expected_hashes.items():
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"frozen artifact hash mismatch: {path}: {observed}")
    alpha_report = json.loads(ALPHA_REPORT.read_text(encoding="utf-8"))
    if not math.isclose(
        float(alpha_report["selected_alpha"]),
        ASSIMILATION,
        rel_tol=0.0,
        abs_tol=TOLERANCE,
    ):
        raise RuntimeError("source-only PCHP assimilation mismatch")
    comparator_report = json.loads(COMPARATOR_REPORT.read_text(encoding="utf-8"))
    selected = {
        float(item["selected_alpha"])
        for item in comparator_report["selection"]["selected_alphas"]
    }
    if selected != {COMPARATOR_ASSIMILATION}:
        raise RuntimeError("source-only comparator assimilation is not unanimous")

    receipt = json.loads(DOWNLOAD_RECEIPT.read_text(encoding="utf-8"))
    roster = receipt["downloaded_cell_zips"]
    if len(roster) != EXPECTED_ZIP_COUNT:
        raise RuntimeError("download receipt does not contain 48 cell ZIPs")
    tables = []
    audits = []
    for index, item in enumerate(sorted(roster, key=lambda value: value["key"]), start=1):
        path = DOWNLOADS / item["key"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"download receipt hash mismatch: {path.name}")
        table, audit = features_for_zip(path)
        tables.append(table)
        audits.append(audit)
        print(
            f"parsed charge-only fields {index}/{len(roster)}: {path.name}, "
            f"valid cycles={len(table)}",
            flush=True,
        )
    features = pd.concat(tables, ignore_index=True)
    enriched = add_context_change(features)
    predictions = enriched.loc[enriched["after_initial5_reference_window"]].copy()
    if predictions.empty:
        raise RuntimeError("no post-anchor external predictions")

    bundle = joblib.load(MODEL_BUNDLE)
    raw_baseline = predict_block(predictions, bundle["protected_baseline"])
    raw_candidate = predict_block(predictions, bundle["absolute_change_candidate"])
    causal_baseline, pchp = prefix_causal_cellwise_projection(
        predictions["cell_id"].astype(str).to_numpy(),
        predictions["target_cycle_number"].to_numpy(float),
        raw_baseline,
        raw_candidate,
        BUDGET,
        assimilation=ASSIMILATION,
        y_min=Y_MIN,
        y_max=Y_MAX,
    )
    comparator = source_tuned_candidate(
        predictions["cell_id"].astype(str).to_numpy(),
        predictions["target_cycle_number"].to_numpy(float),
        raw_candidate,
    )
    displacement = np.abs(pchp - causal_baseline)
    certificate = {
        "maximum_pchp_displacement": float(displacement.max()),
        "baseline_nonincreasing": nonincreasing(predictions, causal_baseline),
        "pchp_nonincreasing": nonincreasing(predictions, pchp),
        "source_tuned_candidate_nonincreasing": nonincreasing(predictions, comparator),
        "prediction_range_valid": bool(
            (
                (causal_baseline >= Y_MIN - TOLERANCE)
                & (causal_baseline <= Y_MAX + TOLERANCE)
                & (pchp >= Y_MIN - TOLERANCE)
                & (pchp <= Y_MAX + TOLERANCE)
                & (comparator >= Y_MIN - TOLERANCE)
                & (comparator <= Y_MAX + TOLERANCE)
            ).all()
        ),
    }
    certificate["label_blind_structural_certificate_passed"] = bool(
        certificate["maximum_pchp_displacement"] <= BUDGET + TOLERANCE
        and certificate["baseline_nonincreasing"]
        and certificate["pchp_nonincreasing"]
        and certificate["source_tuned_candidate_nonincreasing"]
        and certificate["prediction_range_valid"]
    )
    if not certificate["label_blind_structural_certificate_passed"]:
        raise RuntimeError("label-blind structural certificate failed")

    OUT.mkdir(parents=True, exist_ok=True)
    feature_path = OUT / "basytec_label_blind_features_v344.parquet"
    prediction_path = OUT / "basytec_frozen_predictions_v344.parquet"
    report_path = OUT / "basytec_label_blind_prediction_report_v344.json"
    enriched.to_parquet(feature_path, index=False)
    frozen = predictions[
        [
            "domain",
            "cell_id",
            "zip_name",
            "aging_member",
            "cycle_key",
            "source_cycle_order",
            "aligned_cycle_rank",
            "target_cycle_number",
        ]
    ].copy()
    frozen["raw_baseline_prediction"] = raw_baseline
    frozen["raw_candidate_prediction"] = raw_candidate
    frozen["causal_baseline_prediction"] = causal_baseline
    frozen["prefix_causal_harm_capped_prediction"] = pchp
    frozen["source_tuned_causal_candidate_prediction"] = comparator
    frozen.to_parquet(prediction_path, index=False)
    report = {
        "status": "BASYTEC_LABEL_BLIND_PREDICTIONS_FROZEN_BEFORE_CAPACITY_ACCESS",
        "generated_at_local": datetime.now().astimezone().isoformat(),
        "external_outcome_values_accessed": [],
        "capacity_column_values_accessed": False,
        "capacity_header_name_seen_for_schema_only": True,
        "permitted_prediction_fields": [
            "cycle identity",
            "charge-side time",
            "charge-side current",
            "charge-side voltage",
            "charge-side temperature when present",
        ],
        "cells_in_official_roster": len(roster),
        "cells_with_structural_features": int(enriched["cell_id"].nunique()),
        "feature_cycles": int(len(enriched)),
        "prediction_cells": int(frozen["cell_id"].nunique()),
        "prediction_cycles": int(len(frozen)),
        "fixed_nominal_capacity_ah": NOMINAL_CAPACITY_AH,
        "fixed_charge_window_seconds": WINDOW_SECONDS,
        "anchor_cycles": ANCHOR_RECORDS,
        "pchp_assimilation": ASSIMILATION,
        "source_tuned_candidate_assimilation": COMPARATOR_ASSIMILATION,
        "harm_budget_soh_units": BUDGET,
        "certificate": certificate,
        "cell_schema_and_structure_audit": audits,
        "frozen_outputs": {
            "features": {"path": str(feature_path), "sha256": sha256_file(feature_path)},
            "predictions": {"path": str(prediction_path), "sha256": sha256_file(prediction_path)},
        },
        "frozen_inputs": {
            "download_receipt_sha256": sha256_file(DOWNLOAD_RECEIPT),
            "model_bundle_sha256": sha256_file(MODEL_BUNDLE),
            "alpha_report_sha256": sha256_file(ALPHA_REPORT),
            "source_tuned_comparator_report_sha256": sha256_file(COMPARATOR_REPORT),
            "projection_code_sha256": sha256_file(PROJECTION_CODE),
            "prediction_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"froze {len(frozen)} label-blind predictions across "
        f"{frozen['cell_id'].nunique()} cells",
        flush=True,
    )
    print(f"wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
