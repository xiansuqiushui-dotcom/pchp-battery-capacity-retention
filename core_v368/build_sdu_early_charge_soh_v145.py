"""Schema-only SDU adapter around the frozen V109 early-charge builder.

BatteryLife v11 SDU objects contain two preprocessing-audit fields that are
absent from the original eight-domain schema.  They are admitted here but are
never read by the feature or label pipeline.  All scientific transformations
remain in the hash-frozen V109 builder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_batterylife_early_charge_soh_v109 as frozen


ROOT = Path(__file__).resolve().parent
DEFAULT_ARCHIVE_DIR = ROOT / "public_data_batterylife_v109" / "archives"
DEFAULT_PARQUET = ROOT / "batterylife_sdu_early_charge_soh_v145.parquet"
DEFAULT_CELLS = ROOT / "batterylife_sdu_early_charge_soh_v145_cells.csv"
DEFAULT_JSON = ROOT / "batterylife_sdu_early_charge_soh_v145_results.json"
SDU_AUDIT_ONLY_KEYS = {
    "hardcoded_removed_indices",
    "median_removed_indices",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--output-cells", type=Path, default=DEFAULT_CELLS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    original_schema = set(frozen.EXPECTED_OBJECT_KEYS)
    frozen.EXPECTED_OBJECT_KEYS = original_schema | SDU_AUDIT_ONLY_KEYS
    results, cells = frozen.run_build(
        args.archive_dir,
        args.output_parquet,
        domains=("SDU",),
        window_seconds=600.0,
    )
    results["schema_adapter"] = {
        "role": "non-model BatteryLife SDU compatibility only",
        "admitted_audit_only_keys": sorted(SDU_AUDIT_ONLY_KEYS),
        "feature_or_label_use": False,
        "frozen_core_builder": str(
            Path(frozen.__file__).resolve()
        ),
    }
    cells.to_csv(args.output_cells, index=False, float_format="%.15g")
    args.output_json.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
