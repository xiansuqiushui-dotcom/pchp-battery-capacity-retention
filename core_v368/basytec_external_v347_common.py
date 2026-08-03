"""Schema-amended BaSyTec utilities developed only on excluded cell F0001.

The V343 parser expected a per-cycle discharge-capacity column that the raw
BaSyTec export does not contain.  This module is a new protocol implementation,
not a patch to V343.  It treats ``Ah[Ah]`` as cumulative net ampere-hours and
recovers each cycle's discharge capacity as its total negative variation.
Numeric ``~Time[h]`` values are converted from hours to seconds.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

import basytec_external_v343_common as v343


ANCHOR_RECORDS = v343.ANCHOR_RECORDS
CURRENT_ACTIVITY_THRESHOLD_A = v343.CURRENT_ACTIVITY_THRESHOLD_A
EXPECTED_ZIP_COUNT = 47
MINIMUM_POST_ANCHOR_RECORDS = v343.MINIMUM_POST_ANCHOR_RECORDS
NOMINAL_CAPACITY_AH = v343.NOMINAL_CAPACITY_AH
WINDOW_SECONDS = v343.WINDOW_SECONDS
Schema = v343.Schema
contiguous_blocks = v343.contiguous_blocks
exact_two_sided_sign_p = v343.exact_two_sided_sign_p
load_aging_bytes = v343.load_aging_bytes
numeric_series = v343.numeric_series
ordered_cycle_keys = v343.ordered_cycle_keys
sha256_file = v343.sha256_file
stable_cycle_key = v343.stable_cycle_key


def detect_schema(raw: bytes) -> Schema:
    """Resolve only the schema frozen after structural development on F0001."""

    text, encoding = v343._decode_prefix(raw[:1_000_000])
    lines = text.splitlines()
    best: tuple[int, int, str, list[str]] | None = None
    for index, line in enumerate(lines[:512]):
        for delimiter in ("\t", ";", ","):
            fields = next(v343.csv.reader([line], delimiter=delimiter))
            normalized = [v343.normalized_header(field) for field in fields]
            markers = (
                any(value in {"cyccount", "cyclecount"} for value in normalized),
                any(value in {"ia", "currenta"} for value in normalized),
                any(value in {"uv", "voltagev"} for value in normalized),
                any(value in {"timeh", "times"} for value in normalized),
                any(value == "ahah" for value in normalized),
            )
            score = int(sum(markers))
            candidate = (score, -index, delimiter, fields)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None or best[0] != 5:
        raise RuntimeError("raw export does not match the frozen V347 schema")
    _, negative_index, delimiter, fields = best
    header_index = -negative_index

    cycle = v343._select_column(fields, exact=("cyccount", "cyclecount"))
    time = v343._select_column(fields, exact=("timeh", "times"))
    current = v343._select_column(fields, exact=("ia", "currenta"))
    voltage = v343._select_column(fields, exact=("uv", "voltagev"))
    temperature = v343._select_column(
        fields,
        exact=("tc", "t1c", "temperaturec", "temperature"),
        contains_all=(("temp",),),
        excludes=("set", "target", "chamber"),
    )
    cumulative_ah = v343._select_column(fields, exact=("ahah",))
    required = (cycle, time, current, voltage, cumulative_ah)
    if any(value is None for value in required):
        raise RuntimeError("V347 schema resolver did not find a unique frozen field")
    return Schema(
        encoding=encoding,
        delimiter=delimiter,
        header_row_zero_based=header_index,
        cycle=str(cycle),
        time=str(time),
        current=str(current),
        voltage=str(voltage),
        temperature=temperature,
        discharge_capacity=str(cumulative_ah),
    )


def read_prediction_fields(raw: bytes, schema: Schema) -> pd.DataFrame:
    """Select no outcome column; the cumulative-Ah field is schema-only here."""

    usecols = [schema.cycle, schema.time, schema.current, schema.voltage]
    if schema.temperature is not None:
        usecols.append(schema.temperature)
    frame = pd.read_csv(
        io.BytesIO(raw),
        sep=schema.delimiter,
        skiprows=schema.header_row_zero_based,
        header=0,
        usecols=usecols,
        encoding=schema.encoding,
        dtype=str,
        engine="python",
        on_bad_lines="skip",
    )
    renamed = {
        schema.cycle: "cycle_raw",
        schema.time: "time_raw",
        schema.current: "current_raw",
        schema.voltage: "voltage_raw",
    }
    if schema.temperature is not None:
        renamed[schema.temperature] = "temperature_raw"
    return frame.rename(columns=renamed)


def time_seconds(values: pd.Series) -> np.ndarray:
    """Convert the frozen numeric BaSyTec ``~Time[h]`` field to seconds."""

    raw = values.astype(str).str.strip()
    numeric = pd.to_numeric(raw.str.replace(",", ".", regex=False), errors="coerce")
    if float(numeric.notna().mean()) < 0.95:
        raise RuntimeError("V347 requires numeric BaSyTec time in hours")
    return numeric.to_numpy(float) * 3600.0


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


def recover_cycle_discharge_capacity(
    current: np.ndarray,
    cumulative_ah: np.ndarray,
) -> float | None:
    """Return total negative variation of cumulative Ah within one cycle.

    For consecutive finite rows \(j-1,j\), the contribution is
    ``max(A[j-1] - A[j], 0)`` only when the current at row ``j`` is below the
    frozen negative-current threshold.  No smoothing, block selection, or
    result-dependent filtering is performed.
    """

    current = np.asarray(current, dtype=float)
    cumulative_ah = np.asarray(cumulative_ah, dtype=float)
    if len(current) < 2 or len(cumulative_ah) != len(current):
        return None
    finite_pair = (
        np.isfinite(cumulative_ah[:-1])
        & np.isfinite(cumulative_ah[1:])
        & np.isfinite(current[1:])
    )
    negative = current[1:] < -CURRENT_ACTIVITY_THRESHOLD_A
    drops = cumulative_ah[:-1] - cumulative_ah[1:]
    contributions = drops[finite_pair & negative]
    contributions = contributions[contributions > 0.0]
    if len(contributions) == 0:
        return None
    value = float(contributions.sum())
    return value if np.isfinite(value) and value > 0.0 else None


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
        "capacity_rule": "sum positive consecutive decreases in cumulative Ah at negative-current rows within cycle",
    }
