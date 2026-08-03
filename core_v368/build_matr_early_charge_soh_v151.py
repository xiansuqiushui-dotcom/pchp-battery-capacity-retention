"""Outcome-blind MATR builder with explicit per-cycle causal-order QC.

The V109 scientific transformations remain unchanged.  The wrapper admits
the non-model ``Qdlin`` field and excludes only cycles whose first active
current is negative, because their early charge segment occurs after the
same-cycle discharge-capacity target.  No label value or model output enters
the exclusion rule.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import build_batterylife_early_charge_soh_v109 as frozen


ROOT = Path(__file__).resolve().parent
DEFAULT_ARCHIVE_DIR = ROOT / "public_data_batterylife_v109" / "archives"
DEFAULT_PARQUET = ROOT / "batterylife_matr_early_charge_soh_v151.parquet"
DEFAULT_CELLS = ROOT / "batterylife_matr_early_charge_soh_v151_cells.csv"
DEFAULT_JSON = ROOT / "batterylife_matr_early_charge_soh_v151_results.json"
MATR_AUDIT_ONLY_CYCLE_KEYS = {"Qdlin"}
CURRENT_THRESHOLD_A = 0.01
_ORIGINAL_ALIGNED_ROWS_FOR_CELL = frozen.aligned_rows_for_cell
EXCLUDED_CYCLE_LEDGER: list[dict[str, object]] = []


def first_active_stage(cycle: dict[str, object]) -> str:
    current = np.asarray(cycle.get("current_in_A"), dtype=float).reshape(-1)
    active = np.flatnonzero(
        np.isfinite(current) & (np.abs(current) >= CURRENT_THRESHOLD_A)
    )
    if len(active) == 0:
        return "indeterminate"
    return "charge" if current[int(active[0])] > 0.0 else "discharge"


def causal_aligned_rows_for_cell(
    battery: dict[str, object],
    *,
    domain: str,
    source_member: str,
    member_sha256: str,
    window_seconds: float,
) -> tuple[list[dict[str, object]], Counter[str]]:
    rows, counts = _ORIGINAL_ALIGNED_ROWS_FOR_CELL(
        battery,
        domain=domain,
        source_member=source_member,
        member_sha256=member_sha256,
        window_seconds=window_seconds,
    )
    invalid_cycle_numbers: set[int] = set()
    for cycle in battery["cycle_data"]:
        stage = first_active_stage(cycle)
        cycle_number = int(cycle["cycle_number"])
        if stage == "indeterminate":
            raise RuntimeError(
                f"indeterminate first active current in {source_member} "
                f"cycle {cycle_number}"
            )
        if stage == "discharge":
            invalid_cycle_numbers.add(cycle_number)
            EXCLUDED_CYCLE_LEDGER.append(
                {
                    "source_member": source_member,
                    "cell_id": str(battery["cell_id"]),
                    "cycle_number": cycle_number,
                    "reason": "first_active_current_is_discharge",
                    "label_or_model_output_used": False,
                }
            )
    kept = [
        row
        for row in rows
        if int(row["feature_cycle_number"]) not in invalid_cycle_numbers
    ]
    counts["causal_order_excluded"] += len(rows) - len(kept)
    return kept, counts


def apply_outcome_blind_compatibility() -> None:
    frozen.EXPECTED_CYCLE_KEYS = (
        set(frozen.EXPECTED_CYCLE_KEYS) | MATR_AUDIT_ONLY_CYCLE_KEYS
    )
    frozen.aligned_rows_for_cell = causal_aligned_rows_for_cell


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--output-cells", type=Path, default=DEFAULT_CELLS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    EXCLUDED_CYCLE_LEDGER.clear()
    apply_outcome_blind_compatibility()
    results, cells = frozen.run_build(
        args.archive_dir,
        args.output_parquet,
        domains=("MATR",),
        window_seconds=600.0,
    )
    results["outcome_blind_compatibility"] = {
        "admitted_audit_only_cycle_keys": sorted(
            MATR_AUDIT_ONLY_CYCLE_KEYS
        ),
        "current_threshold_in_A": CURRENT_THRESHOLD_A,
        "causal_exclusion_rule": "first active current is negative",
        "excluded_cycles": len(EXCLUDED_CYCLE_LEDGER),
        "excluded_cycle_ledger": EXCLUDED_CYCLE_LEDGER,
        "label_or_model_output_used": False,
        "feature_or_label_transformation_changed": False,
        "frozen_core_builder": str(Path(frozen.__file__).resolve()),
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
