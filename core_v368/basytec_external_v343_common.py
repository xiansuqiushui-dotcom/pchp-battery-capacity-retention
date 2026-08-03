"""Shared, outcome-isolated utilities for the V343 BaSyTec confirmation.

The prediction role can access only cycle identity plus charge-side time,
current, voltage, and temperature.  The scoring role can access only cycle
identity, current sign, and the frozen discharge-capacity field.  Keeping the
roles separate makes the data-access boundary executable and auditable.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RECORD_ID = 15_755_725
RECORD_API = f"https://zenodo.org/api/records/{RECORD_ID}"
RECORD_DOI = "10.5281/zenodo.15755725"
EXPECTED_TITLE = "Degradation path indicators for lithium-ion batteries"
EXPECTED_VERSION = "1"
EXPECTED_LICENSE = "cc-by-4.0"
EXPECTED_ZIP_COUNT = 48
EXPECTED_FILE_COUNT = 49
NOMINAL_CAPACITY_AH = 0.124
WINDOW_SECONDS = 600.0
ANCHOR_RECORDS = 5
MINIMUM_POST_ANCHOR_RECORDS = 10
CURRENT_ACTIVITY_THRESHOLD_A = 1e-5


@dataclass(frozen=True)
class Schema:
    encoding: str
    delimiter: str
    header_row_zero_based: int
    cycle: str
    time: str
    current: str
    voltage: str
    temperature: str | None
    discharge_capacity: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - required to verify repository checksum
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def canonical_json_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def normalized_header(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("µ", "u").replace("μ", "u").replace("°", "")
    return re.sub(r"[^a-z0-9]+", "", text)


def _decode_prefix(raw: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise RuntimeError("cannot decode BaSyTec text export")


def _header_score(fields: Iterable[str]) -> int:
    normalized = [normalized_header(value) for value in fields]
    markers = (
        any("cycle" in value or "cyc" in value for value in normalized),
        any(value in {"i", "ia", "current", "currenta"} or "current" in value for value in normalized),
        any(value in {"u", "uv", "voltage", "voltagev"} or "voltage" in value for value in normalized),
        any("time" in value or "dauer" in value for value in normalized),
        any("ahcycdischarge" in value or "ahcycledischarge" in value for value in normalized),
    )
    return int(sum(markers))


def _select_column(
    fields: list[str],
    *,
    exact: tuple[str, ...] = (),
    contains_all: tuple[tuple[str, ...], ...] = (),
    excludes: tuple[str, ...] = (),
) -> str | None:
    normalized = [(field, normalized_header(field)) for field in fields]
    for candidate in exact:
        matches = [field for field, norm in normalized if norm == candidate]
        if len(matches) == 1:
            return matches[0]
    for tokens in contains_all:
        matches = [
            field
            for field, norm in normalized
            if all(token in norm for token in tokens)
            and not any(token in norm for token in excludes)
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def detect_schema(raw: bytes) -> Schema:
    text, encoding = _decode_prefix(raw[:1_000_000])
    lines = text.splitlines()
    best: tuple[int, int, str, list[str]] | None = None
    for index, line in enumerate(lines[:512]):
        for delimiter in ("\t", ";", ","):
            fields = next(csv.reader([line], delimiter=delimiter))
            if len(fields) < 5:
                continue
            score = _header_score(fields)
            candidate = (score, -index, delimiter, fields)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None or best[0] < 4:
        raise RuntimeError("no admissible BaSyTec data header found")
    _, negative_index, delimiter, fields = best
    header_index = -negative_index

    cycle = _select_column(
        fields,
        exact=("cyclecount", "cyccount", "cycle", "cyclenumber"),
        contains_all=(("cycle", "count"), ("cyc", "count")),
    )
    time = _select_column(
        fields,
        exact=("times", "time", "testtimes", "programtimes", "progtime"),
        contains_all=(("time",), ("dauer",)),
        excludes=("date", "timestamp", "start"),
    )
    current = _select_column(
        fields,
        exact=("ia", "i", "currenta", "current", "ccur"),
        contains_all=(("current",),),
        excludes=("limit", "set", "target", "charge", "discharge", "capacity"),
    )
    voltage = _select_column(
        fields,
        exact=("uv", "u", "voltagev", "voltage", "cvol"),
        contains_all=(("voltage",),),
        excludes=("limit", "set", "target", "min", "max"),
    )
    temperature = _select_column(
        fields,
        exact=("tc", "t1c", "temperaturec", "temperature", "csurftemp"),
        contains_all=(("temp",),),
        excludes=("set", "target", "chamber"),
    )
    capacity = _select_column(
        fields,
        exact=(
            "ahcycdischarge0",
            "ahcycledischarge0",
            "ahcycdischarge",
            "ahcycledischarge",
        ),
        contains_all=(("ah", "cyc", "discharge"), ("ah", "cycle", "discharge")),
    )
    required = {
        "cycle": cycle,
        "time": time,
        "current": current,
        "voltage": voltage,
        "discharge_capacity": capacity,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(f"BaSyTec schema lacks frozen fields: {missing}; fields={fields}")
    return Schema(
        encoding=encoding,
        delimiter=delimiter,
        header_row_zero_based=header_index,
        cycle=str(cycle),
        time=str(time),
        current=str(current),
        voltage=str(voltage),
        temperature=temperature,
        discharge_capacity=str(capacity),
    )


def aging_member(archive: zipfile.ZipFile) -> str:
    matches = [
        name
        for name in archive.namelist()
        if name.replace("\\", "/").lower().endswith("/m0002_aging.txt")
        or name.replace("\\", "/").lower() == "m0002_aging.txt"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one M0002_aging.txt, found {matches}")
    return matches[0]


def load_aging_bytes(zip_path: Path) -> tuple[bytes, str]:
    with zipfile.ZipFile(zip_path) as archive:
        member = aging_member(archive)
        return archive.read(member), member


def numeric_series(values: pd.Series) -> np.ndarray:
    text = values.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce").to_numpy(float)


def time_seconds(values: pd.Series) -> np.ndarray:
    raw = values.astype(str).str.strip()
    numeric = pd.to_numeric(raw.str.replace(",", ".", regex=False), errors="coerce")
    if float(numeric.notna().mean()) >= 0.95:
        return numeric.to_numpy(float)
    output = np.full(len(raw), np.nan, dtype=float)
    pattern = re.compile(r"^(?P<h>\d+):(?P<m>\d{1,2}):(?P<s>\d{1,2}(?:[\.,]\d+)?)$")
    for index, value in enumerate(raw):
        match = pattern.match(value)
        if not match:
            continue
        output[index] = (
            3600.0 * float(match.group("h"))
            + 60.0 * float(match.group("m"))
            + float(match.group("s").replace(",", "."))
        )
    return output


def read_prediction_fields(raw: bytes, schema: Schema) -> pd.DataFrame:
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
            schema.discharge_capacity: "discharge_capacity_raw",
        }
    )


def stable_cycle_key(value: object) -> str:
    text = str(value).strip()
    numeric = pd.to_numeric(pd.Series([text.replace(",", ".")]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return format(float(numeric), ".17g")
    return text


def ordered_cycle_keys(values: pd.Series) -> list[str]:
    keys = values.map(stable_cycle_key)
    first_position: dict[str, int] = {}
    for position, key in enumerate(keys):
        first_position.setdefault(key, position)
    return sorted(first_position, key=first_position.__getitem__)


def contiguous_blocks(indices: np.ndarray) -> list[np.ndarray]:
    index = np.asarray(indices, dtype=int)
    if len(index) == 0:
        return []
    cuts = np.flatnonzero(np.diff(index) > 1) + 1
    return [part for part in np.split(index, cuts) if len(part)]


def exact_two_sided_sign_p(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    tail = min(wins, losses)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return float(min(1.0, 2.0 * probability))

