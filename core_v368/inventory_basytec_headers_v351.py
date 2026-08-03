"""Inventory only header lines across the 47-cell confirmation roster.

The script stops reading each archive member immediately after the header line.
It never reads a data row and therefore never accesses an Ah value.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOWNLOADS = ROOT / "external_basytec_v343" / "downloads"
ROSTER = ROOT / "external_basytec_v347" / "confirmation_roster_v347.json"
OUT = ROOT / "external_basytec_v347" / "header_inventory_v351.json"


def normalized_fields(header: str, delimiter: str) -> list[str]:
    if delimiter == "whitespace":
        fields = re.split(r"\s+", header.strip())
    else:
        fields = header.split(delimiter)
    return [re.sub(r"[^a-z0-9]+", "", item.strip().lower()) for item in fields]


def header_only(zip_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.replace("\\", "/").lower().endswith("/m0002_aging.txt")
        ]
        if len(members) != 1:
            raise RuntimeError(f"{zip_path.name}: expected one aging member")
        member = members[0]
        with archive.open(member) as binary:
            header = None
            lines_read = 0
            for raw_line in binary:
                lines_read += 1
                line = raw_line.decode("latin-1").rstrip("\r\n")
                if line.startswith("~Time") and "Cyc-Count" in line:
                    header = line
                    break
                if lines_read >= 512:
                    break
    if header is None:
        raise RuntimeError(f"{zip_path.name}: no header before line 512")
    if "," in header:
        delimiter = ","
    elif "\t" in header:
        delimiter = "\t"
    elif ";" in header:
        delimiter = ";"
    else:
        delimiter = "whitespace"
    fields = normalized_fields(header, delimiter)
    required = {"timeh", "uv", "ia", "ahah", "cyccount"}
    if not required.issubset(set(fields)):
        raise RuntimeError(f"{zip_path.name}: required frozen semantics absent: {fields}")
    signature = hashlib.sha256(
        json.dumps([delimiter, fields], separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()
    return {
        "zip_name": zip_path.name,
        "aging_member": member,
        "lines_read_including_header": lines_read,
        "data_rows_read": 0,
        "delimiter": delimiter,
        "normalized_fields": fields,
        "structural_signature_sha256": signature,
    }


def main() -> int:
    roster = json.loads(ROSTER.read_text(encoding="utf-8"))["downloaded_cell_zips"]
    records = [header_only(DOWNLOADS / item["key"]) for item in roster]
    counts = Counter(item["delimiter"] for item in records)
    signatures = Counter(item["structural_signature_sha256"] for item in records)
    payload = {
        "status": "47_CELL_HEADER_ONLY_INVENTORY_COMPLETE",
        "capacity_values_accessed": False,
        "data_rows_read": 0,
        "cells": len(records),
        "delimiter_counts": dict(sorted(counts.items())),
        "structural_signature_counts": dict(sorted(signatures.items())),
        "records": records,
    }
    OUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: payload[key] for key in payload if key != "records"}, indent=2))
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
