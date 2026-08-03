"""Header-family-complete BaSyTec utilities after a 47-cell header-only audit."""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

import basytec_external_v347_common as v347


ANCHOR_RECORDS = v347.ANCHOR_RECORDS
CURRENT_ACTIVITY_THRESHOLD_A = v347.CURRENT_ACTIVITY_THRESHOLD_A
EXPECTED_ZIP_COUNT = 47
MINIMUM_POST_ANCHOR_RECORDS = v347.MINIMUM_POST_ANCHOR_RECORDS
NOMINAL_CAPACITY_AH = v347.NOMINAL_CAPACITY_AH
WINDOW_SECONDS = v347.WINDOW_SECONDS
Schema = v347.Schema
contiguous_blocks = v347.contiguous_blocks
exact_two_sided_sign_p = v347.exact_two_sided_sign_p
load_aging_bytes = v347.load_aging_bytes
numeric_series = v347.numeric_series
ordered_cycle_keys = v347.ordered_cycle_keys
recover_cycle_discharge_capacity = v347.recover_cycle_discharge_capacity
sha256_file = v347.sha256_file
stable_cycle_key = v347.stable_cycle_key
time_seconds = v347.time_seconds


def _split_header(line: str) -> tuple[str, list[str]]:
    if "," in line:
        return ",", [item.strip() for item in line.split(",")]
    return r"\s+", re.split(r"\s+", line.strip())


def _unique_field(fields: list[str], normalized_name: str) -> str:
    matches = [
        field
        for field in fields
        if v347.v343.normalized_header(field) == normalized_name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"V352 requires one {normalized_name} field: {fields}")
    return matches[0]


def detect_schema(raw: bytes) -> Schema:
    text, encoding = v347.v343._decode_prefix(raw[:1_000_000])
    lines = text.splitlines()
    candidates: list[tuple[int, str, list[str]]] = []
    for index, line in enumerate(lines[:512]):
        if not line.startswith("~Time") or "Cyc-Count" not in line:
            continue
        delimiter, fields = _split_header(line)
        normalized = {v347.v343.normalized_header(field) for field in fields}
        if {"timeh", "uv", "ia", "ahah", "cyccount"}.issubset(normalized):
            candidates.append((index, delimiter, fields))
    if len(candidates) != 1:
        raise RuntimeError(f"V352 expected one frozen header, found {len(candidates)}")
    header_index, delimiter, fields = candidates[0]
    return Schema(
        encoding=encoding,
        delimiter=delimiter,
        header_row_zero_based=header_index,
        cycle=_unique_field(fields, "cyccount"),
        time=_unique_field(fields, "timeh"),
        current=_unique_field(fields, "ia"),
        voltage=_unique_field(fields, "uv"),
        temperature=None,
        discharge_capacity=_unique_field(fields, "ahah"),
    )


def read_prediction_fields(raw: bytes, schema: Schema) -> pd.DataFrame:
    frame = pd.read_csv(
        io.BytesIO(raw),
        sep=schema.delimiter,
        skiprows=schema.header_row_zero_based,
        header=0,
        usecols=[schema.cycle, schema.time, schema.current, schema.voltage],
        encoding=schema.encoding,
        dtype=str,
        engine="python",
        on_bad_lines="skip",
    )
    return frame.rename(
        columns={
            schema.cycle: "cycle_raw",
            schema.time: "time_raw",
            schema.current: "current_raw",
            schema.voltage: "voltage_raw",
        }
    )


def read_scoring_fields(raw: bytes, schema: Schema) -> pd.DataFrame:
    frame = pd.read_csv(
        io.BytesIO(raw),
        sep=schema.delimiter,
        skiprows=schema.header_row_zero_based,
        header=0,
        usecols=[schema.cycle, schema.current, schema.discharge_capacity],
        encoding=schema.encoding,
        dtype=str,
        engine="python",
        on_bad_lines="skip",
    )
    return frame.rename(
        columns={
            schema.cycle: "cycle_raw",
            schema.current: "current_raw",
            schema.discharge_capacity: "cumulative_ah_raw",
        }
    )


def capacity_map(zip_path: Path) -> tuple[dict[str, float], dict[str, object]]:
    raw, member = load_aging_bytes(zip_path)
    schema = detect_schema(raw)
    frame = read_scoring_fields(raw, schema)
    keys = frame["cycle_raw"].map(stable_cycle_key)
    current = numeric_series(frame["current_raw"])
    cumulative_ah = numeric_series(frame["cumulative_ah_raw"])
    mapping: dict[str, float] = {}
    missing = 0
    key_values = keys.to_numpy(str)
    for key in dict.fromkeys(keys.tolist()):
        positions = np.flatnonzero(key_values == key)
        value = recover_cycle_discharge_capacity(
            current[positions], cumulative_ah[positions]
        )
        if value is None:
            missing += 1
        else:
            mapping[key] = value
    return mapping, {
        "cell_id": zip_path.stem,
        "zip_name": zip_path.name,
        "aging_member": member,
        "cycles_with_capacity": len(mapping),
        "cycles_without_valid_capacity": missing,
        "capacity_source_field": schema.discharge_capacity,
        "delimiter_family": "comma" if schema.delimiter == "," else "whitespace",
        "capacity_rule": "sum positive consecutive decreases in cumulative Ah at negative-current rows within cycle",
    }
