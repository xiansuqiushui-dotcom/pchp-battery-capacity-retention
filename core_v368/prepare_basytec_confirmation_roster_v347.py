"""Create the 47-cell confirmation roster after permanently excluding F0001."""

from __future__ import annotations

import json
from pathlib import Path

from basytec_external_v343_common import sha256_file


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "external_basytec_v343" / "download_receipt_v343.json"
OUT_DIR = ROOT / "external_basytec_v347"
OUT = OUT_DIR / "confirmation_roster_v347.json"


def main() -> int:
    receipt = json.loads(SOURCE.read_text(encoding="utf-8"))
    roster = [
        item for item in receipt["downloaded_cell_zips"] if item["key"] != "F0001.zip"
    ]
    if len(roster) != 47 or any(item["key"] == "F0001.zip" for item in roster):
        raise RuntimeError("confirmation roster is not the frozen 47-cell complement")
    payload = {
        "status": "BASYTEC_V347_CONFIRMATION_ROSTER_FROZEN",
        "development_cell_excluded": "F0001.zip",
        "downloaded_cell_zips": roster,
        "source_receipt": str(SOURCE),
        "source_receipt_sha256": sha256_file(SOURCE),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"froze {len(roster)} confirmation cells: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
