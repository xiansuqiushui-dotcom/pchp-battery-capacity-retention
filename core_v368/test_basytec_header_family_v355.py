"""Test both frozen BaSyTec header families without confirmation outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from basytec_external_v352_common import (
    capacity_map,
    detect_schema,
    read_prediction_fields,
    time_seconds,
)


ROOT = Path(__file__).resolve().parent
INVENTORY = ROOT / "external_basytec_v347" / "header_inventory_v351.json"
PILOT = ROOT / "external_basytec_v343" / "downloads" / "F0001.zip"


def synthetic_text(fields: list[str], delimiter: str) -> bytes:
    actual_delimiter = "," if delimiter == "," else " "
    row = []
    for field in fields:
        value = {
            "timeh": "0.1",
            "uv": "3.5",
            "ia": "0.1",
            "ahah": "OUTCOME_SENTINEL",
            "cyccount": "1",
        }.get(field, "0")
        row.append(value)
    header_alias = {
        "timeh": "~Time[h]",
        "uv": "U[V]",
        "ia": "I[A]",
        "ahah": "Ah[Ah]",
        "cyccount": "Cyc-Count",
    }
    header = [header_alias.get(field, field) for field in fields]
    return (
        "~Resultfile from Basytec Battery Test System\n"
        + actual_delimiter.join(header)
        + "\n"
        + actual_delimiter.join(row)
        + "\n"
    ).encode("utf-8")


def test_all_inventory_signatures() -> None:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    examples = {}
    for record in inventory["records"]:
        examples.setdefault(record["structural_signature_sha256"], record)
    assert len(examples) == 4
    for record in examples.values():
        raw = synthetic_text(record["normalized_fields"], record["delimiter"])
        schema = detect_schema(raw)
        frame = read_prediction_fields(raw, schema)
        assert "OUTCOME_SENTINEL" not in frame.to_string()
        assert np.allclose(time_seconds(frame["time_raw"]), [360.0])


def test_excluded_pilot_capacity_reference() -> None:
    mapping, audit = capacity_map(PILOT)
    first_five = np.asarray(list(mapping.values())[:5], dtype=float)
    assert len(mapping) >= 60
    assert 0.08 < float(np.median(first_five)) < 0.16
    assert audit["delimiter_family"] == "comma"


def main() -> int:
    test_all_inventory_signatures()
    test_excluded_pilot_capacity_reference()
    print("5 V352 header-family tests passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
