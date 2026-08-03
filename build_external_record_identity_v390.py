"""Build a stable row-identity ledger for the frozen external predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = (
    ROOT
    / "updates_v389"
    / "external_mechanism_decision_v380"
    / "external_mechanism_predictions_v380.parquet"
)
OUTPUT = SOURCE.with_name("external_record_identity_v390.csv")
KEYS = ["domain", "cell_id", "target_cycle_number"]


def main() -> None:
    frame = pd.read_parquet(SOURCE, columns=KEYS)
    if frame[KEYS].isna().any().any():
        raise RuntimeError("external identity columns contain missing values")
    ledger = frame.copy()
    ledger.insert(0, "record_id", [f"EXT-{index:05d}" for index in range(1, len(frame) + 1)])
    ledger.insert(1, "frozen_row_index", np.arange(len(frame), dtype=int))
    ledger["within_nominal_cycle_ordinal"] = (
        ledger.groupby(KEYS, sort=False, dropna=False).cumcount().astype(int)
    )
    if not ledger["record_id"].is_unique:
        raise RuntimeError("generated external record identifiers are not unique")
    ledger.to_csv(OUTPUT, index=False, lineterminator="\n")
    print(
        f"Wrote {OUTPUT.relative_to(ROOT)}: {len(ledger)} records; "
        f"{int(ledger.duplicated(KEYS, keep=False).sum())} rows share a nominal-cycle key"
    )


if __name__ == "__main__":
    main()

