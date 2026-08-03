"""Build frozen NASA PCoE predictions without reading discharge outcomes.

Only cycle types and charge-record measurements are accessed.  Discharge
records are represented by immutable locators for the later, separate scoring
stage.  The all-source models, assimilation value, feature contract, and harm
budget are already frozen before the official archive is downloaded.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat

from build_batterylife_early_charge_soh_v109 import early_charge_features
from develop_context_change_soh_v310 import add_context_change
from prefix_causal_harm_projection_v321 import prefix_causal_cellwise_projection


ROOT = Path(__file__).resolve().parent
EXTERNAL_ROOT = ROOT / "external_nasa_v329"
ARCHIVE = EXTERNAL_ROOT / "5_Battery_Data_Set.zip"
EXTRACTED = EXTERNAL_ROOT / "extracted"
OUT = EXTERNAL_ROOT / "label_blind_v330"
MODEL_BUNDLE = (
    ROOT / "rccp_source_model_freeze_v316" / "rccp_source_models_v316.joblib"
)
ALPHA_FREEZE_REPORT = (
    ROOT
    / "external_causal_source_freeze_v328"
    / "external_source_alpha_freeze_v328_report.json"
)
PROJECTION_CODE = ROOT / "prefix_causal_harm_projection_v321.py"

EXPECTED_MODEL_SHA256 = (
    "F492F4B87C1CE7F49C3718F5C2A7D7DADE79450A9E548B33540452B4DC4817FD"
)
EXPECTED_ALPHA_REPORT_SHA256 = (
    "CBB9EBC235878B9C1E914F5AE136BDCA38064040D9957D0410DF82A1AE299AE9"
)
EXPECTED_PROJECTION_SHA256 = (
    "9E2903909E9D6A8C80C29BBA0404C94C11C1A3452721BD105F9D4991069CCB23"
)
NOMINAL_CAPACITY_AH = 2.0
WINDOW_SECONDS = 600.0
ANCHOR_RECORDS = 5
ASSIMILATION = 0.01
BUDGET = 0.01
Y_MIN = 0.0
Y_MAX = 1.3
TOLERANCE = 1e-12


@dataclass(frozen=True)
class CycleRef:
    cell_id: str
    source_member: str
    variable_name: str
    cycle_index: int
    cycle_type: str
    sort_key: tuple[Any, ...]
    cycle_object: Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def require_hash(path: Path, expected: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"frozen artifact hash mismatch for {path}: {observed} != {expected}"
        )


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as stream:
        for member in stream.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.filename}")
        stream.extractall(destination)


def ensure_extracted(archive: Path, destination: Path) -> list[Path]:
    marker = destination / ".source_archive_sha256"
    archive_hash = sha256_file(archive)
    if destination.exists():
        if not marker.exists() or marker.read_text(encoding="utf-8").strip() != archive_hash:
            raise RuntimeError("existing extraction is not bound to the frozen archive")
    else:
        safe_extract_zip(archive, destination)
        marker.write_text(archive_hash + "\n", encoding="utf-8")

    expanded: set[Path] = set()
    for _ in range(3):
        nested = [
            path
            for path in destination.rglob("*.zip")
            if path.resolve() not in expanded
        ]
        if not nested:
            break
        for nested_archive in nested:
            nested_destination = nested_archive.with_suffix("")
            nested_marker = nested_destination / ".source_archive_sha256"
            nested_hash = sha256_file(nested_archive)
            if nested_destination.exists():
                if (
                    not nested_marker.exists()
                    or nested_marker.read_text(encoding="utf-8").strip()
                    != nested_hash
                ):
                    raise RuntimeError(
                        f"existing nested extraction mismatch: {nested_archive}"
                    )
            else:
                safe_extract_zip(nested_archive, nested_destination)
                nested_marker.write_text(nested_hash + "\n", encoding="utf-8")
            expanded.add(nested_archive.resolve())
    return sorted(destination.rglob("*.mat"))


def one_dimensional(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.ravel(array)


def normalized_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    array = np.asarray(value)
    if array.size == 1:
        return str(array.item()).strip().lower()
    return str(value).strip().lower()


def cycle_list(container: Any) -> list[Any]:
    cycles = getattr(container, "cycle", None)
    if cycles is None:
        return []
    return np.atleast_1d(cycles).ravel().tolist()


def temporal_sort_key(cycle: Any, source_member: str, index: int) -> tuple[Any, ...]:
    raw = getattr(cycle, "time", None)
    try:
        values = one_dimensional(raw)
    except (TypeError, ValueError):
        values = np.asarray([], dtype=float)
    if values.size >= 6 and np.isfinite(values[:6]).all():
        six = tuple(float(value) for value in values[:6])
        return (0, *six, source_member, index)
    return (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, source_member, index)


def physical_cell_id(variable_name: str, source_member: str) -> str:
    if re.fullmatch(r"[Bb]\d+", variable_name):
        return variable_name.upper()
    return f"{source_member}::{variable_name}"


def discover_cycles(
    mat_files: list[Path], extracted_root: Path
) -> tuple[dict[str, list[CycleRef]], list[dict[str, object]]]:
    by_cell: dict[str, list[CycleRef]] = defaultdict(list)
    inventory: list[dict[str, object]] = []
    for mat_path in mat_files:
        source_member = mat_path.relative_to(extracted_root).as_posix()
        payload = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        variables = 0
        cycles_found = 0
        for variable_name, container in payload.items():
            if variable_name.startswith("__") or not hasattr(container, "cycle"):
                continue
            cycles = cycle_list(container)
            if not cycles:
                continue
            variables += 1
            cell_id = physical_cell_id(variable_name, source_member)
            for index, cycle in enumerate(cycles):
                cycle_type = normalized_text(getattr(cycle, "type", ""))
                by_cell[cell_id].append(
                    CycleRef(
                        cell_id=cell_id,
                        source_member=source_member,
                        variable_name=variable_name,
                        cycle_index=index,
                        cycle_type=cycle_type,
                        sort_key=temporal_sort_key(cycle, source_member, index),
                        cycle_object=cycle,
                    )
                )
                cycles_found += 1
        inventory.append(
            {
                "source_member": source_member,
                "sha256": sha256_file(mat_path),
                "top_level_cycle_variables": variables,
                "cycles_found": cycles_found,
            }
        )
    return dict(by_cell), inventory


def pair_charge_with_following_discharge(
    cycles: list[CycleRef],
) -> list[tuple[CycleRef, CycleRef]]:
    ordered = sorted(cycles, key=lambda item: item.sort_key)
    pairs: list[tuple[CycleRef, CycleRef]] = []
    for position, charge in enumerate(ordered):
        if charge.cycle_type != "charge":
            continue
        for later in ordered[position + 1 :]:
            if later.cycle_type == "charge":
                break
            if later.cycle_type == "discharge":
                pairs.append((charge, later))
                break
    return pairs


def cumulative_charge_ah(time_seconds: np.ndarray, current_amp: np.ndarray) -> np.ndarray:
    elapsed = np.asarray(time_seconds, dtype=float)
    current = np.abs(np.asarray(current_amp, dtype=float))
    if elapsed.ndim != 1 or current.ndim != 1 or len(elapsed) != len(current):
        raise ValueError("time and current must be aligned vectors")
    if len(elapsed) == 0:
        return np.asarray([], dtype=float)
    increments = np.diff(elapsed)
    increments = np.where(np.isfinite(increments), increments, 0.0)
    increments = np.clip(increments, 0.0, 3600.0)
    trapezoids = 0.5 * (current[:-1] + current[1:]) * increments / 3600.0
    return np.concatenate([[0.0], np.cumsum(trapezoids)])


def extract_charge_feature_row(charge: CycleRef) -> tuple[dict[str, float], str | None]:
    data = getattr(charge.cycle_object, "data", None)
    if data is None:
        return {}, "charge_data_missing"
    try:
        time_seconds = one_dimensional(getattr(data, "Time"))
        current_amp = np.abs(one_dimensional(getattr(data, "Current_measured")))
        voltage = one_dimensional(getattr(data, "Voltage_measured"))
    except (AttributeError, TypeError, ValueError):
        return {}, "required_charge_field_missing_or_invalid"
    if not (len(time_seconds) == len(current_amp) == len(voltage)):
        return {}, "misaligned_required_charge_fields"
    try:
        temperature = one_dimensional(getattr(data, "Temperature_measured"))
    except (AttributeError, TypeError, ValueError):
        temperature = np.full(len(time_seconds), np.nan, dtype=float)
    if len(temperature) != len(time_seconds):
        temperature = np.full(len(time_seconds), np.nan, dtype=float)

    charge_ah = cumulative_charge_ah(time_seconds, current_amp)
    normalized = {
        "current_in_A": current_amp,
        "voltage_in_V": voltage,
        "charge_capacity_in_Ah": charge_ah,
        "time_in_s": time_seconds,
        "temperature_in_C": temperature,
    }
    return early_charge_features(
        normalized,
        nominal_capacity_ah=NOMINAL_CAPACITY_AH,
        window_seconds=WINDOW_SECONDS,
    )


def build_label_blind_feature_table(
    cycles_by_cell: dict[str, list[CycleRef]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    rejections: Counter[str] = Counter()
    cell_audit: list[dict[str, object]] = []
    for cell_id in sorted(cycles_by_cell):
        pairs = pair_charge_with_following_discharge(cycles_by_cell[cell_id])
        valid_rank = 0
        valid_for_cell = 0
        for charge, discharge in pairs:
            features, reason = extract_charge_feature_row(charge)
            if reason is not None:
                rejections[reason] += 1
                continue
            valid_rank += 1
            valid_for_cell += 1
            row: dict[str, object] = {
                "domain": "NASA_PCOE_BATTERY_5",
                "cell_id": cell_id,
                "aligned_cycle_rank": valid_rank,
                "target_cycle_number": float(valid_rank),
                "after_initial5_reference_window": valid_rank > ANCHOR_RECORDS,
                "charge_source_member": charge.source_member,
                "charge_variable_name": charge.variable_name,
                "charge_cycle_index_zero_based": charge.cycle_index,
                "discharge_source_member": discharge.source_member,
                "discharge_variable_name": discharge.variable_name,
                "discharge_cycle_index_zero_based": discharge.cycle_index,
                "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
            }
            row.update(features)
            rows.append(row)
        cell_audit.append(
            {
                "cell_id": cell_id,
                "cycle_records": len(cycles_by_cell[cell_id]),
                "structural_charge_discharge_pairs": len(pairs),
                "valid_fixed_window_charge_records": valid_for_cell,
                "post_anchor_prediction_records": max(
                    valid_for_cell - ANCHOR_RECORDS, 0
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no structurally valid NASA charge records")
    enriched = add_context_change(frame)
    audit = {
        "rejection_counts": dict(sorted(rejections.items())),
        "cell_structural_audit": cell_audit,
    }
    return enriched, audit


def predict_block(frame: pd.DataFrame, block: dict[str, object]) -> np.ndarray:
    features = list(block["features"])
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise RuntimeError(f"missing frozen model features: {missing}")
    transformed = block["imputer"].transform(frame[features])
    prediction = np.asarray(block["model"].predict(transformed), dtype=float)
    lower, upper = block["prediction_clip"]
    return np.clip(prediction, float(lower), float(upper))


def trajectories_nonincreasing(
    frame: pd.DataFrame, values: np.ndarray
) -> bool:
    working = frame[["cell_id", "target_cycle_number"]].copy()
    working["prediction"] = np.asarray(values, dtype=float)
    for _, cell in working.groupby("cell_id", sort=False):
        ordered = cell.sort_values("target_cycle_number", kind="mergesort")
        if (np.diff(ordered["prediction"].to_numpy(float)) > TOLERANCE).any():
            return False
    return True


def main() -> None:
    started = time.perf_counter()
    for path in (ARCHIVE, MODEL_BUNDLE, ALPHA_FREEZE_REPORT, PROJECTION_CODE):
        if not path.exists():
            raise FileNotFoundError(path)
    require_hash(MODEL_BUNDLE, EXPECTED_MODEL_SHA256)
    require_hash(ALPHA_FREEZE_REPORT, EXPECTED_ALPHA_REPORT_SHA256)
    require_hash(PROJECTION_CODE, EXPECTED_PROJECTION_SHA256)
    alpha_report = json.loads(ALPHA_FREEZE_REPORT.read_text(encoding="utf-8"))
    if (
        alpha_report.get("status") != "SOURCE_ONLY_EXTERNAL_DEPLOYMENT_ALPHA_FROZEN"
        or not math.isclose(
            float(alpha_report.get("selected_alpha")),
            ASSIMILATION,
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        )
    ):
        raise RuntimeError("frozen external assimilation contract mismatch")

    mat_files = ensure_extracted(ARCHIVE, EXTRACTED)
    if not mat_files:
        raise RuntimeError("official archive contains no MATLAB files")
    cycles_by_cell, inventory = discover_cycles(mat_files, EXTRACTED)
    features, audit = build_label_blind_feature_table(cycles_by_cell)
    predictions = features.loc[features["after_initial5_reference_window"]].copy()
    if predictions.empty:
        raise RuntimeError("no post-anchor NASA prediction records")

    bundle = joblib.load(MODEL_BUNDLE)
    raw_baseline = predict_block(predictions, bundle["protected_baseline"])
    raw_candidate = predict_block(predictions, bundle["absolute_change_candidate"])
    causal_baseline, causal_method = prefix_causal_cellwise_projection(
        predictions["cell_id"].astype(str).to_numpy(),
        predictions["target_cycle_number"].to_numpy(float),
        raw_baseline,
        raw_candidate,
        BUDGET,
        assimilation=ASSIMILATION,
        y_min=Y_MIN,
        y_max=Y_MAX,
    )
    displacement = np.abs(causal_method - causal_baseline)
    certificate = {
        "maximum_absolute_displacement": float(displacement.max()),
        "baseline_trajectories_nonincreasing": trajectories_nonincreasing(
            predictions, causal_baseline
        ),
        "method_trajectories_nonincreasing": trajectories_nonincreasing(
            predictions, causal_method
        ),
    }
    certificate["label_free_structural_certificate_passed"] = bool(
        certificate["maximum_absolute_displacement"] <= BUDGET + TOLERANCE
        and certificate["baseline_trajectories_nonincreasing"]
        and certificate["method_trajectories_nonincreasing"]
    )
    if not certificate["label_free_structural_certificate_passed"]:
        raise RuntimeError("label-free structural certificate failed")

    prediction_columns = [
        "domain",
        "cell_id",
        "aligned_cycle_rank",
        "target_cycle_number",
        "charge_source_member",
        "charge_variable_name",
        "charge_cycle_index_zero_based",
        "discharge_source_member",
        "discharge_variable_name",
        "discharge_cycle_index_zero_based",
    ]
    frozen = predictions[prediction_columns].copy()
    frozen["raw_baseline_prediction"] = raw_baseline
    frozen["raw_candidate_prediction"] = raw_candidate
    frozen["causal_baseline_prediction"] = causal_baseline
    frozen["prefix_causal_harm_capped_prediction"] = causal_method

    OUT.mkdir(parents=True, exist_ok=True)
    feature_path = OUT / "nasa_label_blind_features_v330.parquet"
    prediction_path = OUT / "nasa_frozen_predictions_v330.parquet"
    report_path = OUT / "nasa_label_blind_prediction_report_v330.json"
    features.to_parquet(feature_path, index=False)
    frozen.to_parquet(prediction_path, index=False)
    report = {
        "status": "NASA_LABEL_BLIND_PREDICTIONS_FROZEN_BEFORE_OUTCOME_ACCESS",
        "generated_at_local": datetime.now().astimezone().isoformat(),
        "external_outcome_fields_accessed": [],
        "permitted_cycle_fields_accessed": {
            "all_cycle_records": ["type", "time"],
            "charge_records_only": [
                "data.Time",
                "data.Current_measured",
                "data.Voltage_measured",
                "data.Temperature_measured",
            ],
        },
        "archive": {
            "path": str(ARCHIVE),
            "sha256": sha256_file(ARCHIVE),
        },
        "mat_files": len(mat_files),
        "discovered_physical_cells": len(cycles_by_cell),
        "structurally_valid_feature_cells": int(features["cell_id"].nunique()),
        "structurally_valid_feature_rows": int(len(features)),
        "prediction_cells": int(frozen["cell_id"].nunique()),
        "prediction_rows": int(len(frozen)),
        "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
        "fixed_charge_window_seconds": WINDOW_SECONDS,
        "initial_charge_feature_anchor_records": ANCHOR_RECORDS,
        "current_convention": "absolute measured charge-current magnitude",
        "charge_throughput_convention": (
            "trapezoidal integral of absolute measured charge current"
        ),
        "selected_source_only_assimilation": ASSIMILATION,
        "harm_budget_soh_units": BUDGET,
        "certificate": certificate,
        "inventory": inventory,
        "structural_audit": audit,
        "frozen_outputs": {
            "features": {
                "path": str(feature_path),
                "sha256": sha256_file(feature_path),
            },
            "predictions": {
                "path": str(prediction_path),
                "sha256": sha256_file(prediction_path),
            },
        },
        "frozen_inputs": {
            "model_bundle_sha256": sha256_file(MODEL_BUNDLE),
            "alpha_freeze_report_sha256": sha256_file(ALPHA_FREEZE_REPORT),
            "projection_code_sha256": sha256_file(PROJECTION_CODE),
            "prediction_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"froze {len(frozen)} label-blind predictions across "
        f"{frozen['cell_id'].nunique()} cells",
        flush=True,
    )
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
