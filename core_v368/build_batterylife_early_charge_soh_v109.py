"""Build a future-safe early-charge SOH table from BatteryLife v11 archives.

Only the first fixed-duration charge segment is used as an input.  Full charge
capacity, full-trajectory extrema, terminal cycle count, and discharge traces
are never features.  Discharge capacity is read only as the hidden SOH label.

For charge-first protocols, the early charge segment and discharge-capacity
label come from the same recorded cycle.  For discharge-first protocols, the
charge segment from cycle t is aligned to the discharge-capacity label from
the next recorded cycle, so every input precedes its target measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
DEFAULT_ARCHIVE_DIR = ROOT / "public_data_batterylife_v109" / "archives"
DEFAULT_PARQUET = ROOT / "batterylife_early_charge_soh_v109.parquet"
DEFAULT_CELLS = ROOT / "batterylife_early_charge_soh_v109_cells.csv"
DEFAULT_JSON = ROOT / "batterylife_early_charge_soh_v109_results.json"
DEFAULT_DOMAINS = (
    "CALB",
    "CALCE",
    "HNEI",
    "MICH",
    "MICH_EXP",
    "SNL",
    "UL_PUR",
    "XJTU",
)
DISCHARGE_FIRST_DOMAINS = {"CALB"}
EXPECTED_OBJECT_KEYS = {
    "cell_id",
    "cycle_data",
    "form_factor",
    "anode_material",
    "cathode_material",
    "electrolyte_material",
    "nominal_capacity_in_Ah",
    "depth_of_charge",
    "depth_of_discharge",
    "already_spent_cycles",
    "max_voltage_limit_in_V",
    "min_voltage_limit_in_V",
    "max_current_limit_in_A",
    "min_current_limit_in_A",
    "reference",
    "description",
    "charge_protocol",
    "discharge_protocol",
    "SOC_interval",
}
EXPECTED_CYCLE_KEYS = {
    "cycle_number",
    "current_in_A",
    "voltage_in_V",
    "charge_capacity_in_Ah",
    "discharge_capacity_in_Ah",
    "time_in_s",
    "temperature_in_C",
    "internal_resistance_in_ohm",
}
GRID_SECONDS = np.arange(0.0, 601.0, 60.0)
SOC_WINDOW_PATTERN = re.compile(r"_(\d+)-(\d+)_")


def numeric_array(value: object) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=float)
    array = np.asarray(value, dtype=float).reshape(-1)
    return array


def discharge_capacity_label(cycle: dict[str, object]) -> float | None:
    values = numeric_array(cycle.get("discharge_capacity_in_Ah"))
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None
    capacity = float(np.max(finite))
    if not math.isfinite(capacity) or capacity <= 0.0:
        return None
    return capacity


def reconstructed_elapsed_seconds(time_values: np.ndarray) -> np.ndarray:
    values = np.asarray(time_values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("time_values must be a nonempty vector")
    differences = np.diff(values)
    fallback = np.maximum(values[1:], 0.0)
    increments = np.where(differences >= 0.0, differences, fallback)
    increments = np.where(np.isfinite(increments), increments, 0.0)
    increments = np.clip(increments, 0.0, 3600.0)
    return np.concatenate([[0.0], np.cumsum(increments)])


def _interpolate(
    elapsed: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(elapsed) & np.isfinite(values)
    x = elapsed[finite]
    y = values[finite]
    if len(x) < 2:
        raise ValueError("fewer than two finite interpolation points")
    unique, indices = np.unique(x, return_index=True)
    y = y[indices]
    if len(unique) < 2 or unique[-1] < grid[-1]:
        raise ValueError("charge segment does not cover the fixed time grid")
    return np.interp(grid, unique, y)


def early_charge_features(
    cycle: dict[str, object],
    *,
    nominal_capacity_ah: float,
    window_seconds: float = 600.0,
) -> tuple[dict[str, float], str | None]:
    """Extract features without consulting the target discharge trace."""

    current = numeric_array(cycle.get("current_in_A"))
    voltage = numeric_array(cycle.get("voltage_in_V"))
    charge = numeric_array(cycle.get("charge_capacity_in_Ah"))
    time_values = numeric_array(cycle.get("time_in_s"))
    temperature = numeric_array(cycle.get("temperature_in_C"))
    lengths = {len(current), len(voltage), len(charge), len(time_values)}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 3:
        return {}, "misaligned_or_short_charge_arrays"
    if not math.isfinite(nominal_capacity_ah) or nominal_capacity_ah <= 0.0:
        return {}, "invalid_nominal_capacity"

    differences = np.diff(charge)
    threshold = max(1e-10, 1e-8 * nominal_capacity_ah)
    starts = np.flatnonzero(np.isfinite(differences) & (differences > threshold))
    if len(starts) == 0:
        return {}, "no_positive_charge_increment"
    start = int(starts[0])
    current = current[start:]
    voltage = voltage[start:]
    charge = charge[start:]
    time_values = time_values[start:]
    if len(temperature) == start + len(current):
        temperature = temperature[start:]
    elif len(temperature) != len(current):
        temperature = np.full(len(current), np.nan, dtype=float)

    finite_core = (
        np.isfinite(current)
        & np.isfinite(voltage)
        & np.isfinite(charge)
        & np.isfinite(time_values)
    )
    if finite_core.sum() < 3:
        return {}, "insufficient_finite_charge_points"
    current = current[finite_core]
    voltage = voltage[finite_core]
    charge = charge[finite_core]
    time_values = time_values[finite_core]
    temperature = temperature[finite_core]
    elapsed = reconstructed_elapsed_seconds(time_values)

    window = float(window_seconds)
    if elapsed[-1] < window:
        return {}, "charge_window_not_covered"
    grid = GRID_SECONDS * (window / 600.0)
    try:
        voltage_grid = _interpolate(elapsed, voltage, grid)
        current_grid = _interpolate(elapsed, current, grid)
        charge_grid = _interpolate(elapsed, charge, grid)
    except ValueError:
        return {}, "charge_window_interpolation_failed"

    within = elapsed <= window
    if within.sum() < 3:
        return {}, "insufficient_points_in_charge_window"
    current_window = current[within] / nominal_capacity_ah
    voltage_window = voltage[within]
    charge_delta = float(charge_grid[-1] - charge_grid[0])
    if not math.isfinite(charge_delta) or charge_delta <= 0.0:
        return {}, "nonpositive_early_charge_throughput"

    features: dict[str, float] = {
        "early_charge_delta_q_per_nominal": (
            charge_delta / nominal_capacity_ah
        ),
        "early_charge_voltage_mean": float(np.mean(voltage_window)),
        "early_charge_voltage_std": float(np.std(voltage_window)),
        "early_charge_voltage_slope_per_hour": float(
            (voltage_grid[-1] - voltage_grid[0]) / (window / 3600.0)
        ),
        "early_charge_current_c_mean": float(np.mean(current_window)),
        "early_charge_current_c_std": float(np.std(current_window)),
        "early_charge_current_c_abs_mean": float(
            np.mean(np.abs(current_window))
        ),
        "early_charge_points": float(within.sum()),
        "temperature_available": float(np.isfinite(temperature[within]).any()),
    }
    finite_temperature = temperature[within][
        np.isfinite(temperature[within])
    ]
    features["early_charge_temperature_mean"] = (
        float(np.mean(finite_temperature))
        if len(finite_temperature)
        else math.nan
    )
    features["early_charge_temperature_std"] = (
        float(np.std(finite_temperature))
        if len(finite_temperature)
        else math.nan
    )
    for seconds, value in zip(grid.astype(int), voltage_grid):
        features[f"voltage_at_{seconds}s"] = float(value)
    for seconds, value in zip(
        (0, int(window / 2), int(window)),
        (current_grid[0], current_grid[len(current_grid) // 2], current_grid[-1]),
    ):
        features[f"current_c_at_{seconds}s"] = float(
            value / nominal_capacity_ah
        )
    return features, None


def protocol_rate(protocol: object) -> float:
    if not isinstance(protocol, list):
        return math.nan
    rates = []
    for step in protocol:
        if isinstance(step, dict):
            value = step.get("rate_in_C")
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                rates.append(numeric)
    return float(rates[0]) if rates else math.nan


def infer_cycling_soc_window(
    battery: dict[str, object],
) -> tuple[float, float, str]:
    """Prefer the cell identifier because some v11 metadata are inaccurate."""

    cell_id = str(battery.get("cell_id", ""))
    match = SOC_WINDOW_PATTERN.search(cell_id)
    if match:
        lower = float(match.group(1)) / 100.0
        upper = float(match.group(2)) / 100.0
        if 0.0 <= lower < upper <= 1.0:
            return lower, upper, "cell_id"
    interval = battery.get("SOC_interval")
    if isinstance(interval, list) and len(interval) == 2:
        try:
            lower = float(interval[0])
            upper = float(interval[1])
        except (TypeError, ValueError):
            pass
        else:
            if 0.0 <= lower < upper <= 1.0:
                return lower, upper, "metadata"
    return math.nan, math.nan, "unresolved"


def aligned_rows_for_cell(
    battery: dict[str, object],
    *,
    domain: str,
    source_member: str,
    member_sha256: str,
    window_seconds: float,
) -> tuple[list[dict[str, object]], Counter[str]]:
    cycles = battery["cycle_data"]
    if not isinstance(cycles, list):
        raise ValueError("cycle_data is not a list")
    nominal = float(battery["nominal_capacity_in_Ah"])
    features: list[dict[str, float] | None] = []
    failures: list[str | None] = []
    labels: list[float | None] = []
    for cycle in cycles:
        if not isinstance(cycle, dict) or set(cycle) != EXPECTED_CYCLE_KEYS:
            features.append(None)
            failures.append("cycle_schema_mismatch")
            labels.append(None)
            continue
        feature, failure = early_charge_features(
            cycle,
            nominal_capacity_ah=nominal,
            window_seconds=window_seconds,
        )
        features.append(feature if failure is None else None)
        failures.append(failure)
        labels.append(discharge_capacity_label(cycle))

    discharge_first = domain in DISCHARGE_FIRST_DOMAINS
    charge_rate = protocol_rate(battery.get("charge_protocol"))
    discharge_rate = protocol_rate(battery.get("discharge_protocol"))
    soc_lower, soc_upper, soc_window_source = infer_cycling_soc_window(battery)
    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    last_feature_index = len(cycles) - 1 if discharge_first else len(cycles)
    for feature_index in range(last_feature_index):
        target_index = feature_index + 1 if discharge_first else feature_index
        feature = features[feature_index]
        if feature is None:
            counts[str(failures[feature_index])] += 1
            continue
        capacity = labels[target_index]
        if capacity is None:
            counts["missing_or_invalid_discharge_capacity"] += 1
            continue
        soh = float(capacity / nominal)
        if not math.isfinite(soh) or soh <= 0.0:
            counts["invalid_soh"] += 1
            continue
        feature_cycle = cycles[feature_index]
        target_cycle = cycles[target_index]
        row: dict[str, object] = {
            "domain": domain,
            "cell_id": f"{domain}/{battery['cell_id']}",
            "source_member": source_member,
            "member_sha256": member_sha256,
            "feature_cycle_number": int(feature_cycle["cycle_number"]),
            "target_cycle_number": int(target_cycle["cycle_number"]),
            "feature_precedes_target": bool(
                discharge_first or feature_index == target_index
            ),
            "protocol_order": (
                "discharge_then_charge" if discharge_first else "charge_then_discharge"
            ),
            "charge_rate_c_metadata": charge_rate,
            "discharge_rate_c_metadata": discharge_rate,
            "nominal_capacity_ah": nominal,
            "max_voltage_limit_v": float(battery["max_voltage_limit_in_V"]),
            "min_voltage_limit_v": float(battery["min_voltage_limit_in_V"]),
            "depth_of_charge": float(battery["depth_of_charge"]),
            "depth_of_discharge": float(battery["depth_of_discharge"]),
            "cycling_soc_lower": soc_lower,
            "cycling_soc_upper": soc_upper,
            "cycling_soc_window_source": soc_window_source,
            "full_soc_window": bool(
                math.isfinite(soc_lower)
                and math.isfinite(soc_upper)
                and soc_lower <= 0.01
                and soc_upper >= 0.99
            ),
            "form_factor": str(battery["form_factor"]),
            "anode_material": str(battery["anode_material"]),
            "cathode_material": str(battery["cathode_material"]),
            "capacity_ah": capacity,
            "soh_nominal": soh,
            **feature,
        }
        rows.append(row)
        counts["valid"] += 1
    if discharge_first:
        counts["unpaired_terminal_charge_cycle"] += 1
    if rows:
        ordered_capacities = [
            float(row["capacity_ah"])
            for row in sorted(
                rows,
                key=lambda row: int(row["target_cycle_number"]),
            )[:5]
        ]
        initial_reference = float(np.median(ordered_capacities))
        if not math.isfinite(initial_reference) or initial_reference <= 0.0:
            raise RuntimeError("invalid first-five capacity reference")
        for row in rows:
            row["initial5_reference_capacity_ah"] = initial_reference
            row["soh_initial5"] = float(
                float(row["capacity_ah"]) / initial_reference
            )
        for rank, row in enumerate(
            sorted(rows, key=lambda row: int(row["target_cycle_number"])),
            start=1,
        ):
            row["aligned_cycle_rank"] = int(rank)
            row["after_initial5_reference_window"] = bool(rank > 5)
    return rows, counts


def run_build(
    archive_dir: Path,
    output_parquet: Path,
    *,
    domains: Sequence[str],
    window_seconds: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    started = time.perf_counter()
    temporary = output_parquet.with_suffix(output_parquet.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    writer: pq.ParquetWriter | None = None
    cell_rows: list[dict[str, object]] = []
    member_hashes: dict[str, list[str]] = {}
    all_cell_ids: set[str] = set()
    total_rows = 0
    try:
        for domain in domains:
            archive = archive_dir / f"{domain}.zip"
            if not archive.is_file():
                raise FileNotFoundError(archive)
            with zipfile.ZipFile(archive) as handle:
                members = sorted(
                    (
                        info
                        for info in handle.infolist()
                        if not info.is_dir()
                        and info.filename.lower().endswith(".pkl")
                    ),
                    key=lambda info: info.filename,
                )
                if not members:
                    raise RuntimeError(f"{archive}: no pickle members")
                for member in members:
                    raw = handle.read(member)
                    member_hash = hashlib.sha256(raw).hexdigest()
                    battery = pickle.loads(raw)
                    if not isinstance(battery, dict):
                        raise ValueError(f"{member.filename}: not a dictionary")
                    if set(battery) != EXPECTED_OBJECT_KEYS:
                        raise ValueError(
                            f"{member.filename}: battery object schema mismatch"
                        )
                    cell_id = f"{domain}/{battery['cell_id']}"
                    if cell_id in all_cell_ids:
                        raise RuntimeError(f"duplicate physical cell id: {cell_id}")
                    all_cell_ids.add(cell_id)
                    member_hashes.setdefault(member_hash, []).append(cell_id)
                    rows, counts = aligned_rows_for_cell(
                        battery,
                        domain=domain,
                        source_member=member.filename,
                        member_sha256=member_hash,
                        window_seconds=window_seconds,
                    )
                    if not rows:
                        raise RuntimeError(
                            f"{member.filename}: no valid early-charge SOH rows"
                        )
                    frame = pd.DataFrame(rows)
                    table = pa.Table.from_pandas(
                        frame,
                        preserve_index=False,
                    )
                    if writer is None:
                        writer = pq.ParquetWriter(
                            temporary,
                            table.schema,
                            compression="zstd",
                        )
                    else:
                        if table.schema != writer.schema:
                            table = table.cast(writer.schema)
                    writer.write_table(table)
                    total_rows += len(frame)
                    cell_rows.append(
                        {
                            "domain": domain,
                            "cell_id": cell_id,
                            "source_member": member.filename,
                            "member_sha256": member_hash,
                            "nominal_capacity_ah": float(
                                battery["nominal_capacity_in_Ah"]
                            ),
                            "protocol_order": (
                                "discharge_then_charge"
                                if domain in DISCHARGE_FIRST_DOMAINS
                                else "charge_then_discharge"
                            ),
                            "recorded_cycles": int(len(battery["cycle_data"])),
                            "cycling_soc_lower": float(
                                frame["cycling_soc_lower"].iloc[0]
                            ),
                            "cycling_soc_upper": float(
                                frame["cycling_soc_upper"].iloc[0]
                            ),
                            "cycling_soc_window_source": str(
                                frame["cycling_soc_window_source"].iloc[0]
                            ),
                            "full_soc_window": bool(
                                frame["full_soc_window"].iloc[0]
                            ),
                            "valid_rows": int(counts["valid"]),
                            "invalid_rows": int(
                                sum(
                                    value
                                    for key, value in counts.items()
                                    if key
                                    not in {
                                        "valid",
                                        "unpaired_terminal_charge_cycle",
                                    }
                                )
                            ),
                            "failure_counts_json": json.dumps(
                                dict(sorted(counts.items())),
                                sort_keys=True,
                            ),
                            "first_soh_nominal": float(
                                frame["soh_nominal"].iloc[0]
                            ),
                            "last_soh_nominal": float(
                                frame["soh_nominal"].iloc[-1]
                            ),
                            "minimum_soh_nominal": float(
                                frame["soh_nominal"].min()
                            ),
                            "maximum_soh_nominal": float(
                                frame["soh_nominal"].max()
                            ),
                            "first_soh_initial5": float(
                                frame["soh_initial5"].iloc[0]
                            ),
                            "last_soh_initial5": float(
                                frame["soh_initial5"].iloc[-1]
                            ),
                            "minimum_soh_initial5": float(
                                frame["soh_initial5"].min()
                            ),
                            "maximum_soh_initial5": float(
                                frame["soh_initial5"].max()
                            ),
                        }
                    )
                    print(
                        f"processed {cell_id}: {len(frame)} valid rows",
                        flush=True,
                    )
        if writer is None:
            raise RuntimeError("no Parquet rows were written")
    finally:
        if writer is not None:
            writer.close()

    duplicate_hash_groups = {
        digest: cells
        for digest, cells in member_hashes.items()
        if len(cells) > 1
    }
    if duplicate_hash_groups:
        raise RuntimeError(
            f"duplicate physical cell payloads found: {duplicate_hash_groups}"
        )
    os.replace(temporary, output_parquet)
    cells = pd.DataFrame(cell_rows).sort_values(["domain", "cell_id"])
    table = pq.read_table(output_parquet)
    if table.num_rows != total_rows:
        raise RuntimeError("Parquet row count does not match build ledger")
    results: dict[str, object] = {
        "status": "FUTURE_SAFE_EARLY_CHARGE_SOH_TABLE_BUILT",
        "archive_dir": str(archive_dir.resolve()),
        "output_parquet": str(output_parquet.resolve()),
        "output_parquet_sha256": hashlib.sha256(
            output_parquet.read_bytes()
        ).hexdigest(),
        "contract": {
            "input_window_seconds": float(window_seconds),
            "feature_source": (
                "only the first fixed-duration charge segment and static "
                "battery metadata"
            ),
            "hidden_label": "maximum recorded discharge capacity in Ah",
            "soh_definitions": {
                "soh_nominal": (
                    "discharge capacity divided by nominal capacity; primary "
                    "only for cells with a verified full 0-100% cycling window"
                ),
                "soh_initial5": (
                    "discharge capacity divided by the median of the first "
                    "five aligned discharge capacities; supports partial-window "
                    "relative-retention analysis under an explicit commissioning "
                    "reference assumption"
                ),
            },
            "charge_first_alignment": (
                "charge segment and later discharge label from the same cycle"
            ),
            "discharge_first_alignment": (
                "charge segment from the preceding recorded cycle and "
                "discharge label from the next recorded cycle"
            ),
            "forbidden_features": [
                "full charge capacity",
                "full discharge trace",
                "future cycle rows",
                "terminal trajectory length",
                "target-aware row deletion",
                "capacity_ah",
                "soh_nominal",
                "soh_initial5",
                "initial5_reference_capacity_ah",
            ],
        },
        "domains": sorted(cells["domain"].unique()),
        "domain_count": int(cells["domain"].nunique()),
        "physical_cells": int(len(cells)),
        "full_soc_window_cells": int(cells["full_soc_window"].sum()),
        "partial_soc_window_cells": int((~cells["full_soc_window"]).sum()),
        "cycle_rows": int(total_rows),
        "duplicate_member_hash_groups": duplicate_hash_groups,
        "per_domain": [
            {
                "domain": str(domain),
                "cells": int(len(group)),
                "valid_rows": int(group["valid_rows"].sum()),
                "invalid_rows": int(group["invalid_rows"].sum()),
                "minimum_cell_rows": int(group["valid_rows"].min()),
                "median_cell_rows": float(group["valid_rows"].median()),
                "maximum_cell_rows": int(group["valid_rows"].max()),
            }
            for domain, group in cells.groupby("domain", sort=True)
        ],
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return results, cells


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--output-cells", type=Path, default=DEFAULT_CELLS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(DEFAULT_DOMAINS),
    )
    parser.add_argument("--window-seconds", type=float, default=600.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results, cells = run_build(
        args.archive_dir,
        args.output_parquet,
        domains=args.domains,
        window_seconds=args.window_seconds,
    )
    cells.to_csv(args.output_cells, index=False, float_format="%.15g")
    args.output_json.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
